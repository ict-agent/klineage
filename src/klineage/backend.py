"""Execution targets and their native toolchain/device interfaces."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from klineage._utils import safe_name

CFLAGS = ("-O3", "-std=c++17")
CUDA_FLAGS = (*CFLAGS, "--expt-relaxed-constexpr", "--expt-extended-lambda")
CUDA_ARCH_ENV = "TORCH_CUDA_ARCH_LIST"
HIP_ARCH_ENV = "PYTORCH_ROCM_ARCH"
ASCEND_ARCH_ENV = "ASCEND_ARCH"
BACKEND_ENV = "KLINEAGE_BACKEND"
ASCEND_HOME = Path("/usr/local/Ascend/ascend-toolkit/latest")
HIP_HOME = Path("/opt/dtk")
COMMON_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")


class BackendKind(StrEnum):
    CUDA = "cuda"
    HYGON = "hygon"
    ASCEND = "ascend"


class AscendFormat(IntEnum):
    """ACL base formats whose storage follows the tensor's logical strides."""

    NCHW = 0
    NHWC = 1
    ND = 2
    NCDHW = 30


@dataclass(frozen=True, slots=True)
class Backend:
    kind: BackendKind
    language: str
    device_type: str
    skill_name: str
    timing_backend: str
    raw_source: str
    error_type: str
    stream_type: str
    entry_suffixes: tuple[str, ...]

    @property
    def raw_abi(self) -> str:
        return (
            f'extern "C" {self.error_type} klineage_launch('
            f"const void* const* inputs, void* const* outputs, {self.stream_type} stream)"
        )

    @property
    def native_suffixes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.entry_suffixes, *COMMON_SUFFIXES)))

    def binding(self) -> str:
        headers, stream, success, error = RAW_APIS[self.kind]
        values = {
            "HEADERS": headers,
            "ABI": self.raw_abi,
            "ERROR_TYPE": self.error_type,
            "STREAM_TYPE": self.stream_type,
            "STREAM": stream,
            "SUCCESS": success,
            "ERROR": error,
            "DEVICE_CHECK": "anchor.is_cuda()"
            if self.device_type == "cuda"
            else "anchor.device().type() == c10::DeviceType::PrivateUse1",
        }
        source = RAW_BINDING
        for key, value in values.items():
            source = source.replace(f"@{key}@", value)
        return source

    def torch(self):
        torch = importlib.import_module("torch")
        if self.kind is BackendKind.ASCEND:
            importlib.import_module("torch_npu")
        return torch

    def prepare(self):
        """Register vendor Python operators before loading a Python bundle."""
        if self.kind is BackendKind.ASCEND:
            self.torch()

    def validate_tensor(self, tensor):
        if self.kind is not BackendKind.ASCEND:
            return
        npu = importlib.import_module("torch_npu")
        try:
            AscendFormat(npu.get_npu_format(tensor))
        except ValueError:
            raise ValueError(
                "Ascend tensor requires a base storage format; "
                "cast to ND before defining the workload"
            ) from None

    def runtime(self):
        torch = self.torch()
        hip = getattr(torch.version, "hip", None)
        if self.kind is BackendKind.CUDA and hip:
            raise RuntimeError("CUDA backend cannot execute with HIP-enabled PyTorch")
        if self.kind is BackendKind.HYGON and not hip:
            raise RuntimeError("Hygon backend requires HIP-enabled DTK PyTorch")
        runtime = getattr(torch, self.device_type, None)
        if runtime is None or not runtime.is_available():
            raise RuntimeError(f"{self.kind} is unavailable to this process")
        return runtime

    def device(self):
        runtime = self.runtime()
        return self.torch().device(self.device_type, runtime.current_device())

    def platform(self) -> str:
        runtime, torch = self.runtime(), self.torch()
        if self.kind is BackendKind.CUDA:
            major, minor = runtime.get_device_capability()
            version = getattr(torch.version, "cuda", None)
            if not version:
                raise RuntimeError("PyTorch does not report its CUDA version")
            return f"nvidia-sm{major}{minor}-cuda{version.split('.')[0]}"
        if self.kind is BackendKind.HYGON:
            properties = runtime.get_device_properties(runtime.current_device())
            architecture = getattr(properties, "gcnArchName", properties.name)
            architecture = safe_name(architecture, default="dcu")
            return f"hygon-{architecture}-hip{torch.version.hip.split('.')[0]}"
        model = safe_name(
            runtime.get_device_name(runtime.current_device()), default="npu"
        )
        return f"ascend-{model}"

    def build_options(self) -> dict[str, Any]:
        """Freeze toolchain and architecture inputs before computing a build key."""
        torch = self.torch()
        if self.kind is BackendKind.ASCEND:
            architecture = os.environ.get(ASCEND_ARCH_ENV)
            if not architecture:
                raise RuntimeError(
                    "Set ASCEND_ARCH to the target bisheng NPU architecture"
                )
            home = Path(os.environ.get("ASCEND_HOME_PATH", ASCEND_HOME)).resolve()
            compiler = shutil.which("bisheng")
            if compiler is None:
                candidates = (
                    home / "tools/bisheng_compiler/bin/bisheng",
                    home / "compiler/ccec_compiler/bin/bisheng",
                )
                compiler = next(
                    (str(path) for path in candidates if path.is_file()), None
                )
            if compiler is None:
                raise RuntimeError("AscendC requires CANN bisheng with -x asc support")
            npu = importlib.import_module("torch_npu")
            return {
                "architecture": architecture,
                "compiler": str(Path(compiler).resolve()),
                "toolkit": str(home),
                "torch_npu": str(Path(npu.__file__).resolve().parent),
                "torch_version": torch.__version__,
                "npu_version": getattr(npu, "__version__", "unknown"),
                "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
                "pybind_abi": {
                    key: value
                    for key in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI")
                    if (value := getattr(torch._C, f"_PYBIND11_{key}", None))
                    is not None
                },
                "cflags": CFLAGS,
            }

        if self.kind is BackendKind.HYGON:
            if not getattr(torch.version, "hip", None):
                raise RuntimeError("Hygon builds require HIP-enabled DTK PyTorch")
            # DTK torch uses cpp_extension's ROCM_HOME convention for hipcc.
            if "ROCM_HOME" not in os.environ:
                home = os.environ.get("HIP_HOME") or os.environ.get("DTK_HOME")
                if home or HIP_HOME.is_dir():
                    os.environ["ROCM_HOME"] = str(home or HIP_HOME)
            extension = importlib.import_module("torch.utils.cpp_extension")
            if not extension.IS_HIP_EXTENSION:
                raise RuntimeError(
                    "Set ROCM_HOME to DTK before importing torch.utils.cpp_extension"
                )
            architecture = os.environ.get(HIP_ARCH_ENV)
            if not architecture:
                runtime = self.runtime()
                architecture = runtime.get_device_properties(
                    runtime.current_device()
                ).gcnArchName
                architecture = architecture.split(":", 1)[0]
            return {
                "architecture": architecture,
                "arch_env": HIP_ARCH_ENV,
                "compiler": str(Path(extension.ROCM_HOME) / "bin/hipcc"),
                "torch_version": torch.__version__,
                "runtime_version": torch.version.hip,
                "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
                "extra_cflags": (*CFLAGS, *extension.COMMON_HIP_FLAGS),
                "extra_cuda_cflags": CFLAGS,
            }

        if getattr(torch.version, "hip", None):
            raise RuntimeError("CUDA builds cannot use HIP-enabled PyTorch")
        extension = importlib.import_module("torch.utils.cpp_extension")
        architecture = os.environ.get(CUDA_ARCH_ENV) or ".".join(
            str(value) for value in torch.cuda.get_device_capability()
        )
        return {
            "architecture": architecture,
            "arch_env": CUDA_ARCH_ENV,
            "compiler": str(
                Path(extension.CUDA_HOME or "/usr/local/cuda") / "bin/nvcc"
            ),
            "torch_version": torch.__version__,
            "runtime_version": torch.version.cuda,
            "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
            "extra_cflags": CFLAGS,
            "extra_cuda_cflags": CUDA_FLAGS,
        }

    def compile(
        self,
        name: str,
        sources: list[str],
        directory: Path,
        includes: list[str],
        options: dict[str, Any],
    ):
        if self.kind is BackendKind.ASCEND:
            return compile_ascend(name, sources, directory, includes, options)

        extension = importlib.import_module("torch.utils.cpp_extension")
        variable = options["arch_env"]
        previous = os.environ.get(variable)
        os.environ[variable] = options["architecture"]
        try:
            return extension.load(
                name=name,
                sources=sources,
                build_directory=str(directory),
                with_cuda=True,
                verbose=False,
                extra_cflags=options["extra_cflags"],
                extra_cuda_cflags=options["extra_cuda_cflags"],
                extra_include_paths=includes,
            )
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


BACKENDS = (
    Backend(
        BackendKind.CUDA,
        "cuda",
        "cuda",
        "cuda",
        "cupti",
        "kernel.cu",
        "cudaError_t",
        "cudaStream_t",
        (".cu",),
    ),
    Backend(
        BackendKind.HYGON,
        "hip",
        "cuda",
        "hip",
        "hip-events",
        "kernel.hip",
        "hipError_t",
        "hipStream_t",
        (".hip", ".cpp", ".cu"),
    ),
    Backend(
        BackendKind.ASCEND,
        "ascendc",
        "npu",
        "ascendc",
        "npu-events",
        "kernel.asc",
        "aclError",
        "aclrtStream",
        (".asc", ".cpp"),
    ),
)

RAW_APIS = {
    BackendKind.CUDA: (
        "#include <ATen/cuda/CUDAContext.h>\n#include <cuda_runtime_api.h>",
        "at::cuda::getCurrentCUDAStream(anchor.get_device()).stream()",
        "cudaSuccess",
        "cudaGetErrorString(status)",
    ),
    BackendKind.HYGON: (
        "#include <c10/hip/HIPStream.h>\n#include <hip/hip_runtime_api.h>",
        "c10::hip::getCurrentHIPStream(anchor.get_device()).stream()",
        "hipSuccess",
        "hipGetErrorString(status)",
    ),
    BackendKind.ASCEND: (
        '#include <acl/acl.h>\n#include "torch_npu/csrc/core/npu/NPUStream.h"',
        "c10_npu::getCurrentNPUStream(anchor.get_device()).stream(true)",
        "ACL_SUCCESS",
        '(aclGetRecentErrMsg() ? aclGetRecentErrMsg() : "Ascend launch failed")',
    ),
}
RAW_BINDING = r"""
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <vector>
@HEADERS@

@ABI@;

void launch(const std::vector<torch::Tensor>& inputs,
            const std::vector<torch::Tensor>& outputs) {
  TORCH_CHECK(!inputs.empty() || !outputs.empty(), "the raw ABI requires a tensor");
  const auto& anchor = inputs.empty() ? outputs.front() : inputs.front();
  TORCH_CHECK(@DEVICE_CHECK@, "tensor device does not match the backend");
  c10::DeviceGuard guard(anchor.device());
  std::vector<const void*> input_ptrs;
  std::vector<void*> output_ptrs;
  for (const auto& value : inputs) {
    TORCH_CHECK(value.device() == anchor.device(), "ABI tensors must share a device");
    input_ptrs.push_back(value.data_ptr());
  }
  for (const auto& value : outputs) {
    TORCH_CHECK(value.device() == anchor.device(), "ABI tensors must share a device");
    output_ptrs.push_back(value.data_ptr());
  }
  @STREAM_TYPE@ stream = @STREAM@;
  @ERROR_TYPE@ status = klineage_launch(input_ptrs.data(), output_ptrs.data(), stream);
  TORCH_CHECK(status == @SUCCESS@, "klineage_launch failed: ", @ERROR@);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("launch", &launch);
}
"""


def platform_backend(platform: str) -> Backend | None:
    name = platform.lower()
    prefixes = (
        ("cuda", "nvidia", "sm"),
        ("hygon", "hip", "dcu", "dtk"),
        ("ascend", "npu"),
    )
    for backend, names in zip(BACKENDS, prefixes, strict=True):
        if name.startswith(names):
            return backend
    return None


def get_backend(language: str, platform: str = "") -> Backend:
    target = platform_backend(platform)
    if language == "python":
        if target is None:
            raise ValueError("Python bundles require an explicit supported platform")
        return target
    for backend in BACKENDS:
        if language != backend.language:
            continue
        if target is not None and target is not backend:
            raise ValueError(
                f"language {language!r} conflicts with platform {platform!r}"
            )
        return backend
    raise ValueError(f"unsupported kernel language {language!r}")


def detect_backend() -> Backend:
    requested = os.environ.get(BACKEND_ENV)
    if requested:
        backend = get_backend("python", requested)
        backend.runtime()
        return backend
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return get_backend("hip" if getattr(torch.version, "hip", None) else "cuda")
    try:
        backend = get_backend("ascendc")
        backend.runtime()
        return backend
    except (ImportError, RuntimeError):
        raise RuntimeError(
            "No supported CUDA, Hygon HIP, or Ascend device is available"
        ) from None


def compile_ascend(
    name: str,
    sources: list[str],
    directory: Path,
    includes: list[str],
    options: dict[str, Any],
):
    """Build a pybind extension using CANN's mixed host/device compiler."""
    from torch.utils.cpp_extension import include_paths, library_paths

    toolkit = Path(options["toolkit"])
    npu = Path(options["torch_npu"])
    include_dirs = [
        *include_paths(),
        sysconfig.get_path("include"),
        *includes,
        str(npu / "include"),
        str(npu / "include/third_party/hccl/inc"),
        str(toolkit / "include"),
    ]
    libraries = [*library_paths(), str(npu / "lib"), str(toolkit / "lib64")]
    output = directory / f"{name}{sysconfig.get_config_var('EXT_SUFFIX')}"
    with TemporaryDirectory(prefix=".compile-", dir=directory) as temporary:
        pending = Path(temporary) / output.name
        command = [
            options["compiler"],
            "-x",
            "asc",
            f"--npu-arch={options['architecture']}",
            "-shared",
            "-fPIC",
            *options["cflags"],
            f"-D_GLIBCXX_USE_CXX11_ABI={options['abi']}",
            f"-DTORCH_EXTENSION_NAME={name}",
            *(
                f'-DPYBIND11_{key}="{value}"'
                for key, value in options["pybind_abi"].items()
            ),
            *sources,
            "-o",
            str(pending),
            *(f"-I{path}" for path in include_dirs),
            *(f"-L{path}" for path in libraries),
            *(f"-Wl,-rpath,{path}" for path in libraries),
            "-ltorch_python",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-ltorch_npu",
            "-lascendcl",
        ]
        log = directory / "build.log"
        with log.open("w") as stream:
            stream.write(repr(command) + "\n")
            stream.flush()
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode:
            raise RuntimeError(f"AscendC compilation failed; see {log}")
        # Publishing a new inode preserves modules already mapped by other workers.
        pending.replace(output)
    spec = importlib.util.spec_from_file_location(name, output)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AscendC extension {output}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["BACKENDS", "Backend", "BackendKind", "detect_backend", "get_backend"]
