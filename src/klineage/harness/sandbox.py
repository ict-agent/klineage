"""Linux process isolation with explicit filesystem and IPC access."""

from __future__ import annotations

import os
import re
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from klineage.backend import Backend

SYSTEM_RUNTIME = tuple(Path(value) for value in (
    "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/etc/alternatives", "/etc/ld.so.cache",
))
PRIVATE_USER = "agent"
PRIVATE_HOME = f"/home/{PRIVATE_USER}"
PRIVATE_TEMP = "/tmp"
PRIVATE_MOUNTS = (Path("/proc"), Path("/dev"))
BASE_ENVIRONMENT = {
    "HOME": PRIVATE_HOME,
    # Runtime caches need a username without exposing the host passwd database.
    "USER": PRIVATE_USER,
    "TMPDIR": PRIVATE_TEMP,
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
CODEX_HELPER = "codex-code-mode-host"
LOADER_ENVIRONMENT = frozenset(("LD_PRELOAD", "LD_AUDIT", "PYTHONHOME", "PYTHONSTARTUP"))
PRIVATE_ENVIRONMENT = frozenset((
    *(BASE_ENVIRONMENT.keys() - {"PATH", "LANG"}),
    "CODEX_HOME", "CLAUDE_CONFIG_DIR", "KLINEAGE_SERVER", "KLINEAGE_PROBLEM",
    "KLINEAGE_PROCESS_SCOPE", "KLINEAGE_LOG_DIR",
    *LOADER_ENVIRONMENT,
))
SECRET_ENVIRONMENT = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.I)
RUNTIME_PATHS = (
    "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "ROCM_HOME", "ROCM_PATH",
    "HIP_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH", "DTK_ROOT", "CXX", "CC", "CUDAHOSTCXX",
)
RUNTIME_ENVIRONMENT = (
    *RUNTIME_PATHS, "KLINEAGE_BACKEND", "MAX_JOBS", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "TORCH_CUDA_ARCH_LIST", "TORCH_ROCM_ARCH_LIST",
    "PYTORCH_ROCM_ARCH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_ARCH",
    "HCCL_WHITELIST_DISABLE",
)


def runtime_environment(
    explicit: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy runtime variables, rejecting explicit credentials and loader hooks."""
    environment = {
        name: os.environ[name] for name in RUNTIME_ENVIRONMENT if name in os.environ
    }
    for name, value in (explicit or {}).items():
        if SECRET_ENVIRONMENT.search(name) or name in LOADER_ENVIRONMENT:
            raise ValueError(
                f"runtime environment cannot contain credentials or loader hooks: {name}"
            )
        environment[name] = value
    return environment


def runtime_mounts(
    info: Mapping[str, Any], environment: Mapping[str, str]
) -> tuple[Path, ...]:
    """Find external runtime paths without exposing home or workspace ancestors."""
    candidates = [info[name] for name in ("toolkit", "torch_npu") if info.get(name)]
    if info.get("compiler"):
        compiler = Path(info["compiler"])
        candidates.append(
            compiler.parent.parent if compiler.parent.name == "bin" else compiler
        )
    for name in RUNTIME_PATHS:
        candidates.extend(environment.get(name, "").split(os.pathsep))

    paths = set()
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.exists():
            continue
        source = path.resolve()
        if Path.home().is_relative_to(source) or Path.cwd().is_relative_to(source):
            continue
        if any(source.is_relative_to(root) for root in (*PRIVATE_MOUNTS, Path("/sys"))):
            continue
        # Preserve aliases and targets when a toolkit crosses the system mounts.
        for candidate in (path, source):
            if candidate.parent == Path("/"):
                continue
            if any(candidate.is_relative_to(root) for root in SYSTEM_RUNTIME):
                continue
            paths.add(candidate)
    selected = []
    for path in sorted(paths, key=lambda value: (len(value.parts), str(value))):
        if not any(path.is_relative_to(root) for root in selected):
            selected.append(path)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class AgentRuntime:
    """Configure the agent's interpreter, dependencies and device access.

    Supply device nodes explicitly or discover a local backend with ``local``;
    visibility variables alone do not mount devices. Model-provider credentials
    belong to the host runner, not this environment.

    Attributes:
        python: Executable interpreter; defaults to the current Python.
        device_nodes: Device nodes mounted into the sandbox; empty by default.
        read_only: Additional files or directories exposed at their absolute paths.
        environment: Explicit runtime variables. Credential names, loader hooks
            and reserved process controls are rejected.
    """

    python: Path = field(default_factory=lambda: Path(sys.executable))
    device_nodes: tuple[Path, ...] = ()
    read_only: tuple[Path, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def local(cls, backend: Backend) -> AgentRuntime:
        """Use the current interpreter and accessible devices of one backend.

        Backends select device nodes and sandbox visibility. Ascend exposes its
        current physical NPU as logical device 0. Host credentials are omitted.
        Toolchains outside system mounts are exposed read-only; the backend's
        compiler directory is added to PATH. The default constructor remains device-free.

        Args:
            backend: Backend selected in the current Python environment.

        Returns:
            Runtime settings usable by both the agent and its local worker.

        Raises:
            RuntimeError: The backend or its device nodes are unavailable.
        """
        from klineage.backend.registry import BACKEND_ENV
        from klineage.backend.runtime import device_nodes

        info = backend.info()
        if not info["available"]:
            raise RuntimeError(f"{backend.kind} is unavailable to this process")
        environment = runtime_environment({BACKEND_ENV: str(backend.kind)})
        devices, environment = backend.device_access(device_nodes(backend), environment)
        if not devices:
            raise RuntimeError(
                f"no accessible {backend.kind} device nodes; configure AgentRuntime explicitly"
            )
        mounts = runtime_mounts(info, environment)
        compiler = info.get("compiler")
        if compiler and Path(compiler).is_absolute():
            paths = (
                str(Path(compiler).parent),
                *environment.get("PATH", BASE_ENVIRONMENT["PATH"]).split(os.pathsep),
            )
            environment["PATH"] = os.pathsep.join(dict.fromkeys(paths))
        return cls(
            device_nodes=devices,
            read_only=mounts,
            environment=environment,
        )

    def __post_init__(self):
        python = Path(self.python).expanduser().absolute()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"agent Python must be executable: {python}")
        object.__setattr__(self, "python", python)
        for name in ("device_nodes", "read_only"):
            values = getattr(self, name)
            if isinstance(values, (str, Path)):
                raise TypeError(f"{name} must be a sequence of paths")
            paths = tuple(Path(path).expanduser().absolute() for path in values)
            object.__setattr__(self, name, paths)

        environment = dict(self.environment)
        for name, value in environment.items():
            if not isinstance(name, str) or not name.isidentifier() or not isinstance(value, str) or "\0" in value:
                raise ValueError("agent environment requires valid names and text values")
            if name in PRIVATE_ENVIRONMENT or SECRET_ENVIRONMENT.search(name):
                raise ValueError(f"agent environment cannot override credentials or process controls: {name}")
        object.__setattr__(self, "environment", MappingProxyType(environment))


def absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(path))


def runtime_paths(
    python: str | Path, *executables: str | Path,
) -> tuple[Path, ...]:
    """Expose a venv, its base interpreter, and selected executable files."""
    interpreter = absolute_path(python)
    paths = [interpreter.parent.parent, interpreter.resolve(strict=True).parent.parent]
    target = interpreter
    while target.is_symlink():
        target = absolute_path(target.parent / os.readlink(target))
        paths.append(target.parent.parent)
    for value in executables:
        path = absolute_path(value)
        resolved = path.resolve(strict=True)
        paths.extend((path, resolved))
        # Standalone Codex locates its code-mode host beside the binary.
        helper = resolved.with_name(CODEX_HELPER)
        if resolved.name == "codex" and helper.is_file():
            paths.append(helper)
    return tuple(dict.fromkeys(paths))


class Sandbox:
    """Build a Linux bubblewrap command with explicit filesystem and IPC access.

    Pass both `command(argv)` and `environment` to `subprocess.Popen`. The
    workspace is writable; home, temporary storage and network are private.
    Missing bubblewrap or namespace support is an error, never a fallback.

    Args:
        workspace: Existing dedicated writable directory.
        read_only: Additional mounts retaining their absolute paths.
        writable: Explicit writable state directories or output files.
        sockets: Existing Unix sockets exposed as individual endpoints.
        devices: Explicit device nodes; no accelerator is mounted by default.
        environment: Variables added to a minimal environment. Host environment
            variables are not inherited; callers must filter explicit values.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        read_only: Sequence[str | Path] = (),
        writable: Sequence[str | Path] = (),
        sockets: Sequence[str | Path] = (),
        devices: Sequence[str | Path] = (),
        environment: Mapping[str, str] | None = None,
    ):
        executable = shutil.which("bwrap")
        if sys.platform != "linux" or executable is None:
            raise RuntimeError("Linux with bubblewrap is required for sandbox isolation")
        self.executable = executable
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("sandbox workspace must be a directory")
        if self.workspace in (Path("/"), Path.home(), Path(PRIVATE_HOME)):
            raise ValueError("sandbox workspace must be a dedicated directory")

        runtime = [path for path in SYSTEM_RUNTIME if path.exists()]
        self.read_only = tuple(dict.fromkeys([
            *runtime, *(absolute_path(path) for path in read_only),
        ]))
        self.writable = tuple(absolute_path(path) for path in writable)
        self.sockets = tuple(absolute_path(path) for path in sockets)
        self.devices = tuple(absolute_path(path) for path in devices)
        for path in (*self.read_only, *self.writable):
            self.check_mount(path)
        for path in self.sockets:
            if not stat.S_ISSOCK(path.stat().st_mode):
                raise ValueError(f"sandbox socket is not a Unix socket: {path}")
        for path in self.devices:
            mode = path.stat().st_mode
            if not (stat.S_ISCHR(mode) or stat.S_ISBLK(mode)):
                raise ValueError(f"sandbox device is not a device node: {path}")

        # Only inherit the explicitly supplied process environment.
        self.environment = {**BASE_ENVIRONMENT, **(environment or {})}
        for key in ("HOME", "TMPDIR"):
            if self.environment[key] != BASE_ENVIRONMENT[key]:
                raise ValueError(f"sandbox {key} must use private storage")

    def check_mount(self, path: Path):
        source = path.resolve(strict=True)
        if self.workspace.is_relative_to(source) or Path.home().is_relative_to(source):
            raise ValueError(f"mount reveals host workspaces or home: {path}")
        if any(source.is_relative_to(root) for root in PRIVATE_MOUNTS):
            raise ValueError(f"mount replaces private process or device state: {path}")

    def command(self, argv: Sequence[str]) -> list[str]:
        if not argv:
            raise ValueError("sandbox command cannot be empty")
        command = [
            self.executable,
            "--unshare-all", "--unshare-user", "--die-with-parent",
            "--new-session", "--cap-drop", "ALL",
            "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", PRIVATE_TEMP, "--tmpfs", PRIVATE_HOME,
            "--symlink", PRIVATE_TEMP, "/var/tmp",
            "--bind", str(self.workspace), str(self.workspace),
        ]
        for path in self.writable:
            command.extend(("--bind", str(path.resolve(strict=True)), str(path)))
        for path in self.read_only:
            command.extend(("--ro-bind", str(path.resolve(strict=True)), str(path)))
        for path in self.sockets:
            command.extend(("--ro-bind", str(path.resolve(strict=True)), str(path)))
        for path in self.devices:
            command.extend(("--dev-bind", str(path.resolve(strict=True)), str(path)))

        # NVIDIA initialization needs writable private procfs; agent procfs is RO.
        if not self.devices:
            command.extend(("--remount-ro", "/proc"))
        command.extend((
            "--chdir", str(self.workspace), "--remount-ro", "/", "--", *argv,
        ))
        return command
