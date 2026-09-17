from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from problem_fixtures import io_problem

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from klineage.artifact import tensor
    from klineage.backend import Backend, get_backend
    from klineage.contract import ABIValue, OutputStyle
    from klineage.harness import eval as worker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class EvalWorkerTests(unittest.TestCase):
    def test_bundle_destination_outputs(self):
        problem = io_problem(
            inputs=(ABIValue("x", "float32", (2,)),),
            outputs=(
                ABIValue("sum", "float32", (2,)),
                ABIValue("product", "float32", (2,)),
            ),
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
            inputs=(ABIValue("x", "float32", (2,)),),
            outputs=(ABIValue("output", "float32", (2,)),),
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
        value = ABIValue("x", "float32", (2,))
        problem = io_problem(
            inputs=(value,), outputs=(value, ABIValue("other", "float32", (2,)))
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
        problem = io_problem(outputs=(ABIValue("output", "float32", (2,)),))
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
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
            patch.object(Backend, "platform", return_value="sm80"),
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
                patch.object(Backend, "platform", return_value="sm90"),
            ):
                self.assertEqual(worker.main(), 0)
            result = json.loads(stdout.getvalue())
            definition, workload = result["definition"], result["workload"]
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
            inputs=(ABIValue("x", "float32", tuple(x.shape)),),
            outputs=(ABIValue("output", "float32", tuple(expected.shape)),),
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
        read_fd, write_fd = os.pipe()
        real_stdout = os.dup(1)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        try:
            with worker._descriptor_stdout(os.fdopen(write_fd, "w", buffering=1)):
                os.write(1, b"native kernel output\n")
            captured = os.read(read_fd, 1024).decode()
        finally:
            os.dup2(real_stdout, 1)
            os.close(real_stdout)
            os.close(read_fd)
        self.assertEqual(captured, "native kernel output\n")


if __name__ == "__main__":
    unittest.main()
