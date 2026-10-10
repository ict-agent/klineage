from __future__ import annotations

import gc
import importlib.util
import io
import json
import os
import tempfile
import unittest
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from problem_fixtures import io_problem

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from klineage.backend import get_backend
    from klineage.backend import runtime as tensor
    from klineage.backend.runtime import OutputStyle
    from klineage.harness import eval as worker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class EvalWorkerTests(unittest.TestCase):
    def setUp(self):
        # Exercise tensor contracts on CPU; backend tests cover device selection.
        runtime = SimpleNamespace(device=lambda device: nullcontext())
        self.enterContext(
            patch("klineage.backend.runtime.device_runtime", return_value=runtime)
        )
        self.enterContext(patch("klineage.backend.runtime.current_device", return_value=torch.device("cpu")))

    def test_numeric_error_details(self):
        expected = torch.tensor([1.0, 2.0, 3.0])
        for value in (2.1, float("nan")):
            with self.subTest(value=value):
                actual = torch.tensor([1.0, value, 3.0])
                with self.assertRaises(AssertionError) as caught:
                    worker.check_close(actual, expected)
                message = str(caught.exception)
                self.assertIn("Mismatched elements:", message)
                self.assertIn("Greatest absolute difference:", message)
                self.assertIn("index (1,)", message)

    def test_numeric_gate_unchanged(self):
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).T
        for value in (1.005, 1.1, float("nan"), float("inf")):
            actual = expected.contiguous()
            actual[0, 0] = value
            with self.subTest(value=value):
                if torch.allclose(actual, expected, rtol=worker.NUMERICAL_TOLERANCE, atol=worker.NUMERICAL_TOLERANCE):
                    worker.check_close(actual, expected)
                else:
                    with self.assertRaises(AssertionError):
                        worker.check_close(actual, expected)

    def test_bundle_destination_outputs(self):
        problem = io_problem(
            inputs={"x": {"dtype": "float32", "shape": [2]}},
            outputs={
                "sum": {"dtype": "float32", "shape": [2]},
                "product": {"dtype": "float32", "shape": [2]},
            },
        )
        calls = []

        def run(x, summed, product):
            calls.append((x, summed, product))
            summed.copy_(x + 1)
            product.copy_(x * 2)

        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = tensor.BundleCallable(problem, run, OutputStyle.DESTINATION)
            first = fn(torch.tensor([1.0, 2.0]))
            second = fn(torch.tensor([3.0, 4.0]))

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(first[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(second[1], torch.tensor([6.0, 8.0]))

    def test_bundle_return_contract(self):
        problem = io_problem(
            inputs={"x": {"dtype": "float32", "shape": [2]}},
            outputs={"output": {"dtype": "float32", "shape": [2]}},
        )
        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = tensor.BundleCallable(problem, lambda x: x + 1, OutputStyle.RETURN)
            torch.testing.assert_close(
                fn(torch.tensor([2.0, 3.0])), torch.tensor([3.0, 4.0])
            )
            with self.assertRaisesRegex(ValueError, "input count"):
                fn()
            with self.assertRaisesRegex(ValueError, "dtype"):
                fn(torch.tensor([2, 3]))
            bad = tensor.BundleCallable(problem, lambda x: x[:1], OutputStyle.RETURN)
            with self.assertRaisesRegex(ValueError, "shape"):
                bad(torch.tensor([2.0, 3.0]))

    def test_bundle_tuple_outputs(self):
        value = {"dtype": "float32", "shape": [2]}
        problem = io_problem(
            inputs={"x": value}, outputs={"x": value, "other": value}
        )
        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            for container in (tuple, list):
                with self.subTest(container=container):
                    fn = tensor.BundleCallable(
                        problem,
                        lambda x, container=container: container((x + 1, x + 2)),
                        OutputStyle.RETURN,
                    )
                    outputs = fn(torch.tensor([2.0, 3.0]))
                    self.assertIsInstance(outputs, tuple)
                    torch.testing.assert_close(outputs[0], torch.tensor([3.0, 4.0]))
                    torch.testing.assert_close(outputs[1], torch.tensor([4.0, 5.0]))

    def test_bundle_without_inputs(self):
        class DeviceTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda:0")

        output = torch.ones(2).as_subclass(DeviceTensor)
        problem = io_problem(outputs={"output": {"dtype": "float32", "shape": [2]}})
        with (
            patch("klineage.backend.runtime.current_device", return_value=output.device),
            patch.object(tensor, "allocate", return_value=output),
        ):
            for style in OutputStyle:
                with self.subTest(style=style):
                    fn = tensor.BundleCallable(problem, lambda *args: output, style)
                    self.assertIs(fn(), output)

    def test_inspects_operator(self):
        module = SimpleNamespace(PROBLEM_NAME="new_gemm", OPERATOR="gemm")
        with (
            patch.object(
                worker,
                "load_problem_module",
                return_value=(module, Path("gemm/reference.py")),
            ),
            patch.object(worker, "load_inputs", return_value={}),
            patch.object(worker, "compute_reference", return_value=torch.ones(1)),
            patch.object(worker, "detect_backend", return_value=get_backend("cuda")),
            patch.object(type(get_backend("cuda")), "info", return_value={"platform": "sm80"}),
            patch.object(
                worker, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(Path, "read_text", return_value="def torch_ref(): return ()"),
            patch.object(worker, "trace_workload", return_value={}),
        ):
            result = worker.inspect_reference(
                "reference.py", workload_path=Path("inputs.safetensors")
            )
            self.assertEqual(result["definition"]["op_type"], "gemm")
            del module.OPERATOR
            result = worker.inspect_reference(
                "reference.py", workload_path=Path("inputs.safetensors")
            )
            self.assertEqual(result["definition"]["op_type"], "new_gemm")
            for value in ("", " ", None, 5):
                module.OPERATOR = value
                with self.assertRaises(ValueError):
                    worker.inspect_reference(
                        "reference.py", workload_path=Path("inputs.safetensors")
                    )

    def test_trace_preserves_io_order(self):
        from safetensors.torch import load_file

        source = """import torch
from collections import namedtuple
OPERATOR = "elementwise"
def make_inputs():
    return {"z": torch.arange(6, dtype=torch.float32).reshape(2, 3).T,
            "a": torch.ones(3, 2)}
def torch_ref(z, a):
    return namedtuple("Outputs", "z_out a_out")(z - a, z + a)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "reference.py"
            path.write_text(source)
            request = {
                "operation": "inspect",
                "problem_path": str(path),
                "workload_path": str(root / "inputs.safetensors"),
            }
            stdout = io.StringIO()
            with (
                patch.object(worker.sys, "stdin", io.StringIO(json.dumps(request))),
                patch.object(worker.sys, "stdout", stdout),
                patch.object(
                    worker,
                    "require_tensor",
                    side_effect=lambda value, *args: value,
                ),
                patch.object(
                    worker, "detect_backend", return_value=get_backend("cuda")
                ),
                patch.object(type(get_backend("cuda")), "info", return_value={"platform": "sm90"}),
            ):
                self.assertEqual(worker.main(), 0)
            result = json.loads(stdout.getvalue())
            definition, workload = result["definition"], result["workloads"][0]
            self.assertEqual(list(definition["inputs"]), ["z", "a"])
            self.assertEqual(list(definition["outputs"]), ["z_out", "a_out"])
            self.assertEqual(workload["axes"], {})
            descriptor = workload["inputs"]["z"]
            data = load_file(descriptor["path"])
            torch.testing.assert_close(
                data[descriptor["tensor_key"]],
                torch.arange(6, dtype=torch.float32).reshape(2, 3).T,
            )
            namespace = {}
            exec(definition["reference"], namespace)  # noqa: S102 - Execute the captured test reference.
            actual = namespace["run"](data["z"], data["a"])
            torch.testing.assert_close(actual.z_out, data["z"] - data["a"])

    def test_runtime_keeps_strides(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        expected = x + 1
        problem = io_problem(
            inputs={"x": {"dtype": "float32", "shape": list(x.shape)}},
            outputs={"output": {"dtype": "float32", "shape": list(expected.shape)}},
        )
        layouts = worker.tensor_strides({"x": x}, expected, problem)
        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = tensor.BundleCallable(
                problem,
                lambda value, output: output.copy_(value + 1),
                OutputStyle.DESTINATION,
                layouts,
            )
            actual = fn(x)
            torch.testing.assert_close(actual, expected, check_stride=True)
            with self.assertRaisesRegex(ValueError, "stride"):
                fn(x.contiguous())

    def test_operation_is_required(self):
        with self.assertRaisesRegex(ValueError, "operation"):
            worker.request_operation({"action": "evaluate"})

    def test_topk_allows_output_order(self):
        values = torch.tensor([[5.0, 1.0, 4.0, 3.0]])
        module = SimpleNamespace(
            OPERATOR="topk", torch_ref=lambda x: torch.topk(x, 2, sorted=False)
        )
        runtime = worker.problem_runtime(module, 0)
        with patch.object(worker, "load_inputs", return_value={"values": values}):
            inputs = runtime.make_inputs()
        expected = runtime.reference(*inputs.args)
        runtime.check_outputs(
            (expected.values.flip(-1), expected.indices.flip(-1)), expected
        )

    def test_topk_checks_indices(self):
        values = torch.tensor([[5.0, 5.0, 5.0, 0.0]])
        module = SimpleNamespace(
            OPERATOR="topk", torch_ref=lambda x: torch.topk(x, 2, sorted=False)
        )
        runtime = worker.problem_runtime(module, 0)
        with patch.object(worker, "load_inputs", return_value={"values": values}):
            inputs = runtime.make_inputs()
        expected = runtime.reference(*inputs.args)
        runtime.check_outputs(
            (torch.tensor([[5.0, 5.0]]), torch.tensor([[0, 1]])), expected
        )
        for indices in (
            torch.tensor([[0, 0]]),
            torch.tensor([[0, 3]]),
            torch.tensor([[-1, 0]]),
            torch.tensor([[0, 4]]),
            torch.tensor([[0, 1]], dtype=torch.int32),
        ):
            with (
                self.subTest(indices=indices),
                self.assertRaises((ValueError, AssertionError)),
            ):
                runtime.check_outputs((torch.tensor([[5.0, 5.0]]), indices), expected)

    def test_build_root_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = root / "reference.py"
            problem.touch()
            with self.assertRaisesRegex(ValueError, "build_root"):
                worker.parse_config({"problem_path": str(problem), "build": str(root)})


    def test_reply_survives_kernel_output(self):
        # A compiled kernel's printf reaches descriptor 1, above the worker's
        # redirect, so its output can land ahead of the reply.
        stdout = '[diag] cuda_dev=0 sms=92\n{"compile_passed": true}\n'
        self.assertEqual(worker.result_payload(stdout), {"compile_passed": True})

    def test_reply_must_be_the_final_line(self):
        with self.assertRaises(ValueError):
            worker.result_payload("[diag] only noise\n")
        with self.assertRaises(ValueError):
            worker.result_payload("")

    def test_descriptor_stdout_diverts_native_writes(self):
        # os.write reaches descriptor 1 directly, unlike a print statement.
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always", ResourceWarning)
            read_fd, write_fd = os.pipe()
            real_stdout = os.dup(1)
            try:
                with os.fdopen(write_fd, "w", buffering=1) as stream, open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), 1)
                    with worker._descriptor_stdout(stream):
                        os.write(1, b"native kernel output\n")
                    captured = os.read(read_fd, 1024).decode()
            finally:
                os.dup2(real_stdout, 1)
                os.close(real_stdout)
                os.close(read_fd)
            gc.collect()
        self.assertEqual(captured, "native kernel output\n")
        self.assertFalse([item for item in notices if issubclass(item.category, ResourceWarning)])


if __name__ == "__main__":
    unittest.main()
