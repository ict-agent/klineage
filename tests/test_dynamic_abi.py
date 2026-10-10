import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from kernel_fixtures import make_kernel, write_kernel

from klineage.artifact import Kernel, Problem
from klineage.backend import backends
from klineage.backend.runtime import BundleCallable, BundleLoader, OutputStyle


def workload(rows):
    return {
        "uuid": f"rows-{rows}",
        "axes": {"N": rows, "M": rows + 1},
        "inputs": {name: {"type": "random"} for name in ("x", "y")},
    }


def dynamic_problem(workloads=(), output_axis="N"):
    return Problem(
        "add",
        {
            "name": "add",
            "op_type": "add",
            "axes": {
                "N": {"type": "var"},
                "M": {"type": "var"},
                "C": {"type": "const", "value": 2},
            },
            "inputs": {
                name: {"dtype": "float32", "shape": ["N", "C"]} for name in ("x", "y")
            },
            "outputs": {"out": {"dtype": "float32", "shape": [output_axis, "C"]}},
            "reference": "def run(x, y): return x + y\n",
        },
        workloads,
        language="cuda",
        platform="cuda",
    )


class DynamicABITests(unittest.TestCase):
    def setUp(self):
        self.problem = dynamic_problem()
        self.backend = SimpleNamespace(kind="cuda", device_type="cpu")
        self.enterContext(
            patch("klineage.backend.runtime.get_backend", return_value=self.backend)
        )
        self.enterContext(
            patch("klineage.backend.runtime.load_torch", return_value=torch)
        )
        self.enterContext(
            patch(
                "klineage.backend.runtime.device_runtime",
                return_value=SimpleNamespace(device=lambda device: nullcontext()),
            )
        )

    def test_dynamic_destinations(self):
        def add(x, y, out):
            out.copy_(x + y)

        fn = BundleCallable(self.problem, add, OutputStyle.DESTINATION)
        for rows in (3, 7, 0):
            x, y = torch.ones(rows, 2), torch.ones(rows, 2)
            result = fn(x, y)
            torch.testing.assert_close(result, x + y)

    def test_dynamic_return(self):
        fn = BundleCallable(self.problem, lambda x, y: x + y, OutputStyle.RETURN)
        for rows in (3, 7):
            torch.testing.assert_close(
                fn(torch.ones(rows, 2), torch.ones(rows, 2)), torch.full((rows, 2), 2.0)
            )

    def test_repeated_axis_conflict(self):
        function = Mock()
        fn = BundleCallable(self.problem, function, OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "axis.*N|N.*axis"):
            fn(torch.ones(3, 2), torch.ones(4, 2))
        function.assert_not_called()

    def test_constant_axis_conflict(self):
        fn = BundleCallable(self.problem, Mock(), OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "shape"):
            fn(torch.ones(3, 4), torch.ones(3, 4))

    def test_dynamic_input_contract(self):
        invalid = (
            ([1, 2], TypeError, "Tensor"),
            (torch.ones(3, 2, dtype=torch.int64), ValueError, "dtype"),
            (torch.ones(3, 2, 1), ValueError, "shape"),
            (torch.empty(3, 2, device="meta"), ValueError, "cpu"),
        )
        for value, error, message in invalid:
            with self.subTest(message=message):
                function = Mock()
                fn = BundleCallable(self.problem, function, OutputStyle.RETURN)
                with self.assertRaisesRegex(error, message):
                    fn(value, torch.ones(3, 2))
                function.assert_not_called()

    def test_dynamic_output_shape(self):
        fn = BundleCallable(self.problem, lambda x, y: x[:1], OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "output.*shape"):
            fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_output_only_axis(self):
        function = Mock()
        fn = BundleCallable(
            dynamic_problem(output_axis="M"), function, OutputStyle.DESTINATION
        )
        with self.assertRaisesRegex(ValueError, "cannot infer.*M|M.*cannot infer"):
            fn(torch.ones(3, 2), torch.ones(3, 2))
        function.assert_not_called()

    def test_explicit_case_binding(self):
        cases = (workload(3), workload(7))
        problem = dynamic_problem(cases)
        fn = BundleCallable(
            problem, lambda x, y: x + y, OutputStyle.RETURN, workload=cases[1]
        )
        torch.testing.assert_close(
            fn(torch.ones(7, 2), torch.ones(7, 2)), torch.full((7, 2), 2.0)
        )
        with self.assertRaisesRegex(ValueError, "input.*shape"):
            fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_explicit_output_axis(self):
        case = workload(3)
        fn = BundleCallable(
            dynamic_problem(output_axis="M"),
            lambda x, y, out: out.fill_(1),
            OutputStyle.DESTINATION,
            workload=case,
        )
        torch.testing.assert_close(
            fn(torch.ones(3, 2), torch.ones(3, 2)), torch.ones(4, 2)
        )

    def test_single_case_stays_bound(self):
        fn = BundleCallable(dynamic_problem((workload(3),)), Mock(), OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "input.*shape"):
            fn(torch.ones(7, 2), torch.ones(7, 2))

    def test_case_specific_strides(self):
        cases = (workload(3), workload(7))
        for case in cases:
            rows = case["axes"]["N"]
            with self.subTest(rows=rows):
                strides = {("output", "out"): (1, rows)}
                fn = BundleCallable(
                    dynamic_problem(cases),
                    lambda x, y, out: out.copy_(x + y),
                    OutputStyle.DESTINATION,
                    strides,
                    workload=case,
                )
                result = fn(torch.ones(rows, 2), torch.ones(rows, 2))
                self.assertEqual(result.stride(), (1, rows))
                torch.testing.assert_close(result, torch.full((rows, 2), 2.0))

    def load_python(self, problem, source, root):
        sources = {
            "config.toml": '[solution]\nname="add"\ndefinition="add"\nauthor="test"\n[build]\nlanguage="python"\nentry_point="kernel.py::run"\n',
            "solution/kernel.py": source,
        }
        directory = write_kernel(make_kernel("add", problem, sources), root / "kernel")
        return Kernel.load_kernel(directory)

    def test_tensor_python_checks_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fn = self.load_python(
                self.problem, "def run(x, y): return x[:1]\n", Path(temporary)
            )
            with self.assertRaisesRegex(ValueError, "output.*shape"):
                fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_load_zero_and_many_cases(self):
        for cases in ((), (workload(3), workload(7))):
            with (
                self.subTest(cases=len(cases)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fn = self.load_python(
                    dynamic_problem(cases),
                    "def run(x, y): return x + y\n",
                    Path(temporary),
                )
                for rows in (3, 5, 7):
                    result = fn(torch.ones(rows, 2), torch.ones(rows, 2))
                    torch.testing.assert_close(result, torch.full((rows, 2), 2.0))


class BackendWorkloadTests(unittest.TestCase):
    def test_platform_forwards_workload(self):
        case = {"axes": {"N": 3}}
        for backend in backends():
            with (
                self.subTest(backend=backend.kind),
                patch.object(BundleLoader, "build", return_value=object()) as build,
            ):
                kernel = object()
                backend.kernel_type.build_kernel(kernel, Path("build"), workload=case)
                self.assertEqual(build.call_args.kwargs["workload"], case)


if __name__ == "__main__":
    unittest.main()
