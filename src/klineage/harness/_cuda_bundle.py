"""Load CUDA bundles described by config.toml."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from klineage.contract import CUDA_SOLUTION, CudaBuild
from klineage.harness.artifacts import _read_source_tree
from klineage.harness.callable_eval import _IMPORT_LOCK, _import_source_module
from klineage.kernel import Kernel

_NATIVE_SUFFIXES = (".cu", ".c", ".cc", ".cpp", ".cxx")
_ARCH_ENV = "TVM_FFI_CUDA_ARCH_LIST"
_HASH_LENGTH = 20
_TORCH_LIBRARIES = ("c10", "c10_cuda", "torch_cpu", "torch_cuda", "torch")


class _BundleLoader:
    def __init__(self, build_root: Path) -> None:
        self._build_root = Path(build_root).resolve()
        self._modules: dict[str, Any] = {}

    def load(self, kernel: Kernel) -> tuple[Callable[..., Any], CudaBuild]:
        if kernel.artifact_path is None or kernel.source_files is None:
            raise ValueError("CUDA bundle requires an artifact and source files")
        root = kernel.artifact_path.absolute()
        if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError("CUDA bundle must be a regular source directory")
        if _read_source_tree(root, "CUDA bundle") != kernel.source_files:
            raise ValueError("CUDA bundle does not match its recorded sources")
        if self._build_root.is_relative_to(root):
            raise ValueError("build products must remain outside the source bundle")

        build = CudaBuild.from_sources(kernel.source_files)
        module = self._native(root, kernel) if build.language == "cuda" else None
        if Path(build.source_path).suffix == ".py":
            module = _python_module(root, build, kernel)
        function = getattr(module, build.symbol)
        if not callable(function):
            raise TypeError(f"CUDA entry point {build.symbol!r} is not callable")
        return function, build

    def _native(self, root: Path, kernel: Kernel):
        native_files = {path: source for path, source in kernel.source_files.items()
                        if path.startswith(CUDA_SOLUTION + "/") and Path(path).suffix != ".py"}
        sources = [str(root / path) for path in sorted(native_files)
                   if Path(path).suffix in _NATIVE_SUFFIXES]
        if not sources:
            raise ValueError("CUDA build requires native source files")

        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
        from tvm_ffi.cpp import load

        # ATen stream/device helpers need Torch headers and their native libraries.
        abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
        flags = {
            "extra_cflags": ["-O3", abi],
            "extra_cuda_cflags": ["-O3", abi, "--expt-relaxed-constexpr", "--expt-extended-lambda"],
            "extra_include_paths": include_paths(device_type="cuda"),
            "extra_ldflags": [*(f"-L{path}" for path in library_paths(device_type="cuda")),
                              *(f"-l{name}" for name in _TORCH_LIBRARIES)],
        }
        with _IMPORT_LOCK:
            previous = os.environ.get(_ARCH_ENV)
            architecture = previous or ".".join(
                str(value) for value in torch.cuda.get_device_capability()
            )
            # Artifact location and Python bindings must not duplicate native registrations.
            digest = hashlib.sha256(json.dumps(
                [native_files, flags, architecture], sort_keys=True,
            ).encode()).hexdigest()[:_HASH_LENGTH]
            name = f"klineage_ffi_{digest}"
            if name in self._modules:
                return self._modules[name]

            directory = self._build_root / name
            directory.mkdir(parents=True, exist_ok=True)
            flags["extra_include_paths"] = [str(root / CUDA_SOLUTION), *flags["extra_include_paths"]]
            os.environ[_ARCH_ENV] = architecture
            try:
                module = load(name=name, sources=sources, build_directory=str(directory),
                              backend="cuda", **flags)
            finally:
                if previous is None:
                    os.environ.pop(_ARCH_ENV, None)
                else:
                    os.environ[_ARCH_ENV] = previous
            self._modules[name] = module
            return module


def _python_module(root: Path, build: CudaBuild, kernel: Kernel):
    # Preserve source-only artifacts and isolate same-named local Python imports.
    with _IMPORT_LOCK:
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            return _import_source_module(
                root / CUDA_SOLUTION, root / build.source_path, kernel,
            )
        finally:
            sys.dont_write_bytecode = previous
