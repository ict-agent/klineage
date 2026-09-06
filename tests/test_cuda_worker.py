from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from klineage.contract import KernelABI
    from klineage.harness import _cuda_worker as worker
    from klineage.kernel import Kernel, TargetContext


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is provided by the CUDA image")
class CudaWorkerTests(unittest.TestCase):
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
