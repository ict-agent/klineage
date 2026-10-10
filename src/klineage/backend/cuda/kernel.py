"""CUDA kernel format, compiler and device metadata."""

import importlib
import os
from pathlib import Path

from klineage.artifact.kernel import KERNEL, Kernel
from klineage.artifact.tree import Node
from klineage.backend.kernel import Backend, BackendKind
from klineage.constants import BUILD_DIRECTORY, KERNEL_METADATA

from . import timing

# torch.utils.cpp_extension chooses the standard required by its headers.
CFLAGS = ("-O3",)

CUDA_FLAGS = (*CFLAGS, "--expt-relaxed-constexpr", "--expt-extended-lambda")
CUDA_ARCH_ENV = "TORCH_CUDA_ARCH_LIST"
DEVICE_LIMITS = (
    "multi_processor_count",
    "warp_size",
    "max_threads_per_block",
    "max_threads_per_multi_processor",
    "regs_per_multiprocessor",
    "shared_memory_per_block",
    "shared_memory_per_block_optin",
    "shared_memory_per_multiprocessor",
    "L2_cache_size",
    "total_memory",
)


class CudaKernel(Kernel):
    """CUDA artifact supporting native .cu sources and Python host entries.

    ``submission/config.toml`` selects a host function whose source owns its
    binding, launch parameters, and stream selection. Native entries build
    through PyTorch's extension loader and nvcc; the shared loader adapts the
    problem's tensor ABI. Load a directory with ``Kernel.load_kernel``.
    """

    __slots__ = ()
    artifact_kind = "kernel.cuda"
    entry_suffixes = (".cu",)

    @classmethod
    def kernel_spec(cls):
        from klineage.artifact.tree import file_ref

        submission = Node.directory(
            "submission",
            description="CUDA host entry and device implementation. Use nvcc; native entry sources use .cu.",
            children=(
                Node.file(
                    "config.toml",
                    description="[solution] declares name, definition and author. "
                    "[build] declares language, entry_point and destination_passing_style. "
                    "entry_point names a host callable that owns all device launch parameters. "
                    "The entry source supplies its Python/pybind11 binding and obtains the current stream. "
                    'Use language="cuda" for native sources or language="python" for a Python host entry.',
                    one_shot=file_ref(
                        Path(__file__).parent / "skills/assets/add_one/config.toml"
                    ),
                ),
                Node.directory(
                    "solution",
                    (Node.file("kernel.cu", description="CUDA host entry and device code."),),
                    description="Complete CUDA source. Define other file layouts in kernel_spec().",
                ),
            ),
        )
        return Node.directory(
            ".",
            description="CUDA kernel artifact.\n" + KERNEL.description,
            children=(KERNEL.child(KERNEL_METADATA), submission),
        )

    def build_kernel(
        self, build_root=BUILD_DIRECTORY, *, include_paths=(), strides=None, workload=None
    ):
        from klineage.backend.runtime import BundleLoader

        return BundleLoader(
            build_root, include_paths=include_paths, compiler=compile_native
        ).build(self, backend=CudaBackend(), strides=strides, workload=workload)


class CudaBackend(Backend):
    """CUDA discovery and CUPTI timing with the CudaKernel artifact format.

    ``info()`` distinguishes the actual device architecture from the compilation
    target overridden by TORCH_CUDA_ARCH_LIST. Timing requires FlashInfer and
    CUPTI 13 or newer; it does not fall back to CUDA event timing.
    """

    kind = BackendKind.CUDA
    language = "cuda"
    device_type = "cuda"
    platform_prefixes = ("cuda", "nvidia", "sm")
    skill_root = Path(__file__).with_name("skills")
    kernel_type = CudaKernel
    timing = timing
    device_patterns = (
        "nvidia[0-9]*", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools",
        "nvidia-caps/nvidia-cap[0-9]*",
    )
    bench = staticmethod(timing.bench)

    def info(self):
        from klineage.backend.runtime import load_torch, runtime_info

        torch = load_torch(self)
        info = runtime_info(self, torch)
        extension = importlib.import_module("torch.utils.cpp_extension")
        properties = (
            torch.cuda.get_device_properties(torch.cuda.current_device())
            if info["available"]
            else None
        )
        architecture = (
            ".".join(str(value) for value in torch.cuda.get_device_capability())
            if info["available"]
            else None
        )
        version = torch.version.cuda
        return info | {
            "device_limits": {
                name: getattr(properties, name)
                for name in DEVICE_LIMITS
                if hasattr(properties, name)
            },
            "architecture": os.environ.get(CUDA_ARCH_ENV) or architecture,
            "device_architecture": f"sm{architecture.replace('.', '')}"
            if architecture
            else None,
            "platform": f"nvidia-sm{architecture.replace('.', '')}-cuda{version.split('.')[0]}"
            if architecture and version
            else "nvidia",
            "arch_env": CUDA_ARCH_ENV,
            "compiler": str(
                Path(extension.CUDA_HOME or "/usr/local/cuda") / "bin/nvcc"
            ),
            "runtime_version": version,
        }


def compile_native(sources, build_root, *, include_paths=(), compile_flags=()):
    from klineage.backend.runtime import COMPILE_LOCK, compile_sources

    # Architecture discovery must not see another build's temporary environment.
    with COMPILE_LOCK:
        options = CudaBackend().info() | {
            "extra_cflags": CFLAGS,
            "extra_cuda_cflags": (*CUDA_FLAGS, *compile_flags),
        }
        if not options["architecture"]:
            raise RuntimeError(
                f"Set {CUDA_ARCH_ENV} when building without a CUDA device"
            )
        return compile_sources(
            sources,
            build_root,
            include_paths,
            options,
            CudaKernel.entry_suffixes,
            compile_extension,
        )


def compile_extension(name, sources, directory, includes, options):
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
