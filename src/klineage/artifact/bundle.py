"""Build native bundles and isolate Python bundle imports."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard

from klineage.artifact.source import snapshot
from klineage.artifact.tensor import BundleCallable
from klineage.backend import Backend, get_backend, platform_backend
from klineage.constants import BUNDLE_CONFIG, BUNDLE_SOLUTION, MODULE_HASH_LENGTH
from klineage.contract import OutputStyle, ValueRole

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel

#: Serialize temporary module/environment changes within this process.
IMPORT_LOCK = threading.RLock()
#: Mutable source staging directory used by compilers.
BUILD_SOURCES = "sources"
#: File lock covering source staging and compilation across processes.
BUILD_LOCK = ".klineage-build.lock"


@contextmanager
def source_import_scope(root: Path, modules: dict[str, ModuleType]):
    """Prevent ordinary local imports from leaking between two artifacts."""

    with IMPORT_LOCK:
        local_names = local_top_level_names(root)
        displaced = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name in modules or name.split(".", 1)[0] in local_names
        }
        for name in displaced:
            sys.modules.pop(name, None)
        before = set(sys.modules)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.modules.update(modules)
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            sys.dont_write_bytecode = previous
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
            for name in set(sys.modules) - before:
                module = sys.modules.get(name)
                if module_is_under(module, root):
                    modules[name] = module
                    sys.modules.pop(name, None)
            sys.modules.update(displaced)


def local_top_level_names(root: Path) -> set[str]:
    names = {path.stem for path in root.glob("*.py") if path.name != "__init__.py"}
    # Directories without __init__.py are importable namespace packages too.
    names.update(path.name for path in root.iterdir() if path.is_dir())
    return names


def module_is_under(module: ModuleType | None, root: Path) -> TypeGuard[ModuleType]:
    if module is None:
        return False
    filename = getattr(module, "__file__", None)
    locations = (filename,) if filename else getattr(module, "__path__", ())
    try:
        return any(Path(path).resolve().is_relative_to(root) for path in locations)
    except (OSError, RuntimeError, ValueError):
        return False


class BundleLoader:
    def __init__(self, build_root: Path, *, include_paths: Sequence[Path] = ()):
        self.build_root = Path(build_root).resolve()
        self.include_paths = tuple(Path(path).resolve() for path in include_paths)
        self.modules: dict[str, Any] = {}

    def build(
        self,
        kernel: Kernel,
        *,
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
    ) -> Callable[..., Any]:
        function = self.load(kernel)
        if (
            kernel.language == "python"
            and kernel.output_style is OutputStyle.RETURN
            and strides is None
        ):
            return function
        return BundleCallable(kernel.problem, function, kernel.output_style, strides)

    def load(self, kernel: Kernel) -> Callable[..., Any]:
        if BUNDLE_CONFIG in kernel.source_files and kernel.language == "python":
            backend = platform_backend(kernel.problem.platform)
            if backend is not None:
                backend.prepare()
            root = Path(snapshot(self.build_root, kernel.source_files))
            return self.load_python(root, kernel)

        backend = get_backend(kernel.problem.language, kernel.problem.platform)
        if BUNDLE_CONFIG not in kernel.source_files:
            if set(kernel.source_files) != {backend.raw_source}:
                raise ValueError(
                    f"raw {backend.kind} adapters require a single {backend.raw_source} source"
                )
            sources = {
                backend.raw_source: kernel.source_files[backend.raw_source],
                "binding.cpp": backend.binding(),
            }
            native_root = Path(snapshot(self.build_root, sources))
            module = self.load_native(
                native_root, sources, backend, kernel.compile_flags
            )
            launch = getattr(module, "launch", None)
            if not callable(launch):
                raise TypeError("native extension did not expose launch")
            input_count = len(kernel.problem.values(ValueRole.INPUTS))

            def run(*tensors):
                launch(list(tensors[:input_count]), list(tensors[input_count:]))

            return run
        root = Path(snapshot(self.build_root, kernel.source_files))
        module = self.load_native(
            root, kernel.source_files, backend, kernel.compile_flags
        )
        return self.entry(module, kernel.symbol)

    def load_python(self, root: Path, kernel: Kernel) -> Callable[..., Any]:
        name = f"_klineage_kernel_{kernel.fingerprint[:MODULE_HASH_LENGTH]}"
        spec = importlib.util.spec_from_file_location(name, root / kernel.source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Python entry {kernel.source_path!r}")
        module = importlib.util.module_from_spec(spec)
        modules = {name: module}
        directory = root / BUNDLE_SOLUTION
        with source_import_scope(directory, modules):
            spec.loader.exec_module(module)
            function = self.entry(module, kernel.symbol)

        # Delayed imports need the same private modules as initial imports.
        def run(*args):
            with source_import_scope(directory, modules):
                return function(*args)

        return run

    @staticmethod
    def entry(module: Any, symbol: str) -> Callable[..., Any]:
        function = getattr(module, symbol)
        if not callable(function):
            raise TypeError(f"bundle entry point {symbol!r} is not callable")
        return function

    def load_native(
        self,
        root: Path,
        source_files: Mapping[str, str],
        backend: Backend,
        compile_flags: Sequence[str] = (),
    ):
        native_files = {
            path: source
            for path, source in source_files.items()
            if path != BUNDLE_CONFIG and Path(path).suffix != ".py"
        }
        sources = [
            path
            for path in sorted(native_files)
            if Path(path).suffix in backend.native_suffixes
        ]
        if not sources:
            raise ValueError(f"{backend.kind} build requires native source files")
        include_paths = [str(root / BUNDLE_SOLUTION), *map(str, self.include_paths)]

        with IMPORT_LOCK:
            options = backend.build_options(compile_flags=compile_flags)
            # Cache complete sources, flags, includes, and architecture together.
            digest = hashlib.sha256(
                json.dumps(
                    [backend.kind, native_files, options, include_paths],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:MODULE_HASH_LENGTH]
            name = f"klineage_{backend.kind}_{digest}"
            if name in self.modules:
                return self.modules[name]
            directory = self.build_root / name
            directory.mkdir(parents=True, exist_ok=True)
            # Compilers may write beside inputs; frozen snapshots stay read-only.
            with native_sources(directory, native_files) as compile_root:
                sources = [str(compile_root / path) for path in sources]
                include_paths[0] = str(compile_root / BUNDLE_SOLUTION)
                module = backend.compile(
                    name, sources, directory, include_paths, options
                )
            self.modules[name] = module
            return module


@contextmanager
def native_sources(directory: Path, sources: Mapping[str, str]):
    # Lock staging as well as compilation across worker processes.
    with (directory / BUILD_LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = directory / BUILD_SOURCES
        for name, source in sources.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.encode("utf-8"))
        yield root
