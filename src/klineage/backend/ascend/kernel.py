"""AscendC kernel format, compiler and device metadata."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sysconfig
from enum import IntEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from klineage.artifact.kernel import KERNEL, Kernel
from klineage.artifact.tree import Node
from klineage.backend.kernel import Backend, BackendKind
from klineage.constants import BUILD_DIRECTORY, KERNEL_METADATA
from klineage.utils import safe_name

from . import timing

CFLAGS = ("-O3", "-std=c++17")
BUILD_ERROR_BYTES = 16 * 1024

ASCEND_ARCH_ENV = "ASCEND_ARCH"
ASCEND_HOME = Path("/usr/local/Ascend/ascend-toolkit/latest")


class AscendFormat(IntEnum):
    """ACL storage formats matching a tensor's logical strides."""

    NCHW = 0
    NHWC = 1
    ND = 2
    NCDHW = 30


def check_storage(tensor):
    npu = importlib.import_module("torch_npu")
    try:
        AscendFormat(npu.get_npu_format(tensor))
    except ValueError:
        raise ValueError(
            "Ascend tensor requires a base storage format; cast to ND before defining the workload"
        ) from None


class AscendKernel(Kernel):
    """AscendC artifact supporting native sources and Python host entries.

    Native .asc or .cpp entries build with CANN bisheng and an explicit
    ASCEND_ARCH target. The host entry declared in config.toml owns bindings and
    launches; the shared loader adapts the problem's tensor ABI.
    """

    __slots__ = ()
    artifact_kind = "kernel.ascendc"
    entry_suffixes = (".asc", ".cpp")

    @classmethod
    def kernel_spec(cls):
        submission = Node.directory(
            "submission",
            description="AscendC host entry and device implementation. Use CANN bisheng -x asc; native entry sources use .asc or .cpp.",
            children=(
                Node.file(
                    "config.toml",
                    description="[solution] declares name, definition and author. "
                    "[build] declares language, entry_point and destination_passing_style. "
                    "entry_point names a host callable that owns all device launch parameters. "
                    "The entry source supplies its Python/pybind11 binding and obtains the current stream. "
                    'Use language="ascendc" for native sources or language="python" for a Python host entry.',
                ),
                Node.directory(
                    "solution",
                    (Node.file("kernel.cpp", description="AscendC host entry and device code."),),
                    description="Complete sources and local headers. An implementation requires config.toml and this directory; "
                    "only a source-free input may omit them.",
                ),
            ),
        )
        return Node.directory(
            ".",
            description="AscendC kernel artifact.\n" + KERNEL.description,
            children=(KERNEL.find(KERNEL_METADATA), submission),
        )

    def build_kernel(
        self, build_root=BUILD_DIRECTORY, *, include_paths=(), strides=None, workload=None
    ):
        from klineage.backend.runtime import BundleLoader

        return BundleLoader(
            build_root, include_paths=include_paths, compiler=compile_native
        ).build(self, backend=AscendBackend(), strides=strides, workload=workload)


class AscendBackend(Backend):
    """Ascend discovery and current-stream NPU event timing through torch_npu.

    Metadata describes the device, CANN toolchain, and compilation target from
    ASCEND_ARCH. Event timing rejects cold-L2 requests.
    """

    kind = BackendKind.ASCEND
    language = "ascendc"
    device_type = "npu"
    platform_prefixes = ("ascend", "npu")
    skill_root = Path(__file__).with_name("skills")
    kernel_type = AscendKernel
    timing = timing
    device_patterns = ("davinci[0-9]*", "davinci_manager", "devmm_svm", "hisi_hdc")
    bench = staticmethod(timing.bench)

    def info(self):
        from klineage.backend.runtime import load_torch, runtime_info

        torch = load_torch(self)
        info = runtime_info(self, torch)
        home = Path(os.environ.get("ASCEND_HOME_PATH", ASCEND_HOME)).resolve()
        compiler = shutil.which("bisheng")
        if compiler is None:
            candidates = (
                home / "tools/bisheng_compiler/bin/bisheng",
                home / "compiler/ccec_compiler/bin/bisheng",
            )
            compiler = next((str(path) for path in candidates if path.is_file()), None)
        npu = importlib.import_module("torch_npu")
        model = (
            torch.npu.get_device_name(torch.npu.current_device())
            if info["available"]
            else None
        )
        return info | {
            "architecture": os.environ.get(ASCEND_ARCH_ENV),
            "device_architecture": model,
            "platform": f"ascend-{safe_name(model, default='npu')}"
            if model
            else "ascend",
            "compiler": str(Path(compiler).resolve()) if compiler else None,
            "toolkit": str(home),
            "torch_npu": str(Path(npu.__file__).resolve().parent),
            "runtime_version": getattr(npu, "__version__", "unknown"),
            "pybind_abi": {
                key: value
                for key in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI")
                if (value := getattr(torch._C, f"_PYBIND11_{key}", None)) is not None
            },
        }


def compile_native(sources, build_root, *, include_paths=(), compile_flags=()):
    from klineage.backend.runtime import compile_sources

    options = AscendBackend().info() | {"cflags": (*CFLAGS, *compile_flags)}
    if not options["architecture"]:
        raise RuntimeError(
            f"Set {ASCEND_ARCH_ENV} to the target bisheng NPU architecture"
        )
    if not options["compiler"]:
        raise RuntimeError("AscendC requires CANN bisheng with -x asc support")
    return compile_sources(
        sources,
        build_root,
        include_paths,
        options,
        AscendKernel.entry_suffixes,
        compile_ascend,
    )


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
            # Evaluators return the exception; their build directory is private.
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - BUILD_ERROR_BYTES))
                detail = stream.read().decode(errors="replace")
            raise RuntimeError(f"AscendC compilation failed; see {log}\n{detail}")
        # Publishing a new inode preserves modules already mapped by other workers.
        pending.replace(output)
    spec = importlib.util.spec_from_file_location(name, output)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AscendC extension {output}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
