"""Hygon HIP kernel format, compiler and device metadata."""

import importlib
import os
from pathlib import Path

from klineage.artifact.kernel import KERNEL, Kernel
from klineage.artifact.tree import Node
from klineage.backend.kernel import Backend, BackendKind
from klineage.constants import BUILD_DIRECTORY, KERNEL_METADATA
from klineage.utils import safe_name

from . import timing

CFLAGS = ("-O3", "-std=c++17")

HIP_ARCH_ENV = "PYTORCH_ROCM_ARCH"
HIP_HOME = Path("/opt/dtk")


class HipKernel(Kernel):
    """Hygon HIP artifact supporting native sources and Python host entries.

    Native .hip, .cpp, or .cu entries build through DTK hipcc and HIP-enabled
    PyTorch. The host entry declared in config.toml owns bindings and launches;
    the shared loader adapts the problem's tensor ABI.
    """

    __slots__ = ()
    artifact_kind = "kernel.hip"
    entry_suffixes = (".hip", ".cpp", ".cu")

    @classmethod
    def kernel_spec(cls):
        submission = Node.directory(
            "submission",
            description="HIP host entry and device implementation. Use DTK hipcc with HIP-enabled PyTorch; native entry sources use .hip, .cpp or .cu.",
            children=(
                Node.file(
                    "config.toml",
                    description="[solution] declares name, definition and author. "
                    "[build] declares language, entry_point and destination_passing_style. "
                    "entry_point names a host callable that owns all device launch parameters. "
                    "The entry source supplies its Python/pybind11 binding and obtains the current stream. "
                    'Use language="hip" for native sources or language="python" for a Python host entry.',
                ),
                Node.directory(
                    "solution",
                    (Node.file("kernel.hip", description="HIP host entry and device code."),),
                    description="Complete sources and local headers. An implementation requires config.toml and this directory; "
                    "only a source-free input may omit them.",
                ),
            ),
        )
        return Node.directory(
            ".",
            description="HIP kernel artifact.\n" + KERNEL.description,
            children=(KERNEL.child(KERNEL_METADATA), submission),
        )

    def build_kernel(
        self, build_root=BUILD_DIRECTORY, *, include_paths=(), strides=None, workload=None
    ):
        from klineage.backend.runtime import BundleLoader

        return BundleLoader(
            build_root, include_paths=include_paths, compiler=compile_native
        ).build(self, backend=HipBackend(), strides=strides, workload=workload)


class HipBackend(Backend):
    """Hygon HIP discovery and current-stream event timing.

    PyTorch exposes HIP tensors through its ``cuda`` device API. Device metadata
    reports the detected architecture separately from PYTORCH_ROCM_ARCH, and
    timing rejects cold-L2 requests.
    """

    kind = BackendKind.HYGON
    language = "hip"
    device_type = "cuda"
    platform_prefixes = ("hygon", "hip", "dcu", "dtk")
    skill_root = Path(__file__).with_name("skills")
    kernel_type = HipKernel
    timing = timing
    device_patterns = ("kfd", "dri/renderD[0-9]*")
    bench = staticmethod(timing.bench)

    def info(self):
        from klineage.backend.runtime import load_torch, runtime_info

        torch = load_torch(self)
        info = runtime_info(self, torch)
        # DTK cpp_extension reads ROCM_HOME when first imported.
        if "ROCM_HOME" not in os.environ:
            home = os.environ.get("HIP_HOME") or os.environ.get("DTK_HOME")
            if home or HIP_HOME.is_dir():
                os.environ["ROCM_HOME"] = str(home or HIP_HOME)
        extension = importlib.import_module("torch.utils.cpp_extension")
        if not extension.IS_HIP_EXTENSION:
            raise RuntimeError(
                "Set ROCM_HOME to DTK before importing torch.utils.cpp_extension"
            )
        architecture = None
        if info["available"]:
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            architecture = getattr(properties, "gcnArchName", properties.name).split(
                ":", 1
            )[0]
        return info | {
            "architecture": os.environ.get(HIP_ARCH_ENV) or architecture,
            "device_architecture": architecture,
            "platform": f"hygon-{safe_name(architecture, default='dcu')}-hip{torch.version.hip.split('.')[0]}"
            if architecture
            else "hygon",
            "arch_env": HIP_ARCH_ENV,
            "compiler": str(Path(extension.ROCM_HOME) / "bin/hipcc"),
            "runtime_version": torch.version.hip,
        }


def compile_native(sources, build_root, *, include_paths=(), compile_flags=()):
    from klineage.backend.runtime import COMPILE_LOCK, compile_sources

    # Architecture discovery must not see another build's temporary environment.
    with COMPILE_LOCK:
        info = HipBackend().info()
        extension = importlib.import_module("torch.utils.cpp_extension")
        options = info | {
            "extra_cflags": (*CFLAGS, *extension.COMMON_HIP_FLAGS),
            "extra_cuda_cflags": (*CFLAGS, *compile_flags),
        }
        if not options["architecture"]:
            raise RuntimeError(f"Set {HIP_ARCH_ENV} when building without a HIP device")
        return compile_sources(
            sources,
            build_root,
            include_paths,
            options,
            HipKernel.entry_suffixes,
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
