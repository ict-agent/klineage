from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from klineage.contract import ABIValue, KernelABI, OutputStyle
    from klineage.harness import _cuda_worker as worker
    from klineage.kernel import Kernel, TargetContext


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is provided by the CUDA image")
class CudaWorkerTests(unittest.TestCase):
    def test_bundle_destination_outputs(self) -> None:
        abi = KernelABI(
            inputs=(ABIValue("x", "float32", (2,)),),
            outputs=(ABIValue("sum", "float32", (2,)), ABIValue("product", "float32", (2,))),
        )
        calls = []

        def launch(x, summed, product):
            calls.append((x, summed, product))
            summed.copy_(x + 1)
            product.copy_(x * 2)

        with (
            patch.object(worker, "_require_cuda_tensor", side_effect=lambda value, label: value),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = worker._BundleCallable(abi, launch, OutputStyle.DESTINATION)
            first = fn(torch.tensor([1., 2.]))
            second = fn(torch.tensor([3., 4.]))

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(first[0], torch.tensor([2., 3.]))
        torch.testing.assert_close(second[1], torch.tensor([6., 8.]))

    def test_bundle_return_contract(self) -> None:
        abi = KernelABI(
            inputs=(ABIValue("x", "float32", (2,)),),
            outputs=(ABIValue("output", "float32", (2,)),),
        )
        with (
            patch.object(worker, "_require_cuda_tensor", side_effect=lambda value, label: value),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = worker._BundleCallable(abi, lambda x: x + 1, OutputStyle.RETURN)
            torch.testing.assert_close(fn(torch.tensor([2., 3.])), torch.tensor([3., 4.]))
            with self.assertRaisesRegex(ValueError, "input count"):
                fn()
            with self.assertRaisesRegex(ValueError, "dtype"):
                fn(torch.tensor([2, 3]))
            bad = worker._BundleCallable(abi, lambda x: x[:1], OutputStyle.RETURN)
            with self.assertRaisesRegex(ValueError, "shape"):
                bad(torch.tensor([2., 3.]))

    def test_bundle_ffi_tuple_outputs(self) -> None:
        import tvm_ffi

        value = ABIValue("x", "float32", (2,))
        abi = KernelABI(inputs=(value,), outputs=(value, ABIValue("other", "float32", (2,))))
        with (
            patch.object(worker, "_require_cuda_tensor", side_effect=lambda value, label: value),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = worker._BundleCallable(
                abi, lambda x: tvm_ffi.convert([x + 1, x + 2]), OutputStyle.RETURN,
            )
            outputs = fn(torch.tensor([2., 3.]))

        self.assertIsInstance(outputs, tuple)
        torch.testing.assert_close(outputs[0], torch.tensor([3., 4.]))
        torch.testing.assert_close(outputs[1], torch.tensor([4., 5.]))

    def test_bundle_without_inputs(self) -> None:
        class DeviceTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda:0")

        output = torch.ones(2).as_subclass(DeviceTensor)
        abi = KernelABI(outputs=(ABIValue("output", "float32", (2,)),))
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(worker, "_allocate", return_value=output),
        ):
            for style in OutputStyle:
                with self.subTest(style=style):
                    fn = worker._BundleCallable(abi, lambda *args: output, style)
                    self.assertIs(fn(), output)

    def test_load_uses_owned_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = root / "reference.py"
            source = root / "kernel.cu"
            include = root / "include"
            build = root / "build"
            problem.touch()
            source.write_text("__global__ void kernel() {}", encoding="utf-8")
            include.mkdir()
            build.mkdir()
            config = worker._config(
                {
                    "problem_path": str(problem),
                    "build_root": str(build),
                    "include_paths": [str(include)],
                    "seed": 17,
                }
            )
            kernel = Kernel(
                "kernel",
                source.read_text(),
                TargetContext("example", "cuda", "sm120"),
                artifact_path=source,
                abi=KernelABI(),
            )

            with patch.object(worker, "load_torch_extension") as compile:
                compile.return_value = SimpleNamespace(launch=lambda *_: None)
                loader = worker._RawLoader(config)
                self.assertTrue(callable(loader.load(kernel)))

            options = compile.call_args.kwargs
            self.assertEqual(
                options["sources"], (str(worker._BINDING_SOURCE), str(source))
            )
            self.assertEqual(options["extra_include_paths"], (str(include),))
            self.assertEqual(Path(options["build_directory"]).parent, build)
            self.assertFalse(options["verbose"])
            self.assertEqual(config.seed, 17)
            self.assertEqual(loader._builds[kernel.fingerprint]["cuda_flags"], list(options["extra_cuda_cflags"]))

    def test_inspects_operator(self) -> None:
        module = SimpleNamespace(PROBLEM_NAME="new_gemm", OPERATOR="gemm")
        with (
            patch.object(worker, "_load_problem", return_value=(module, Path("gemm/reference.py"))),
            patch.object(worker, "_make_inputs", return_value={}),
            patch.object(worker, "_reference"),
            patch.object(worker, "_infer_abi", return_value=KernelABI()),
            patch.object(worker, "_platform", return_value="sm80"),
        ):
            self.assertEqual(worker.inspect_problem("reference.py")["operator"], "gemm")
            del module.OPERATOR
            self.assertEqual(worker.inspect_problem("reference.py")["operator"], "new_gemm")
            for value in ("", " ", None, 5):
                module.OPERATOR = value
                with self.assertRaises(ValueError):
                    worker.inspect_problem("reference.py")

    def test_operation_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "operation"):
            worker._operation({"action": "evaluate"})

    def test_build_root_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem = root / "reference.py"
            problem.touch()
            with self.assertRaisesRegex(ValueError, "build_root"):
                worker._config({"problem_path": str(problem), "build": str(root)})


if __name__ == "__main__":
    unittest.main()
