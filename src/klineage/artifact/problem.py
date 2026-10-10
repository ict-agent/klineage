"""Problem definitions, input data and their artifact format."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import uuid
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from klineage.artifact.tree import (
    Artifact,
    ArtifactKind,
    FileRef,
    Node,
    check_ref,
    file_ref,
    json_node,
)
from klineage.constants import KERNEL_METADATA, MODULE_HASH_LENGTH
from klineage.serialization import from_json, raw_text, restore
from klineage.tools import agent_function
from klineage.utils import json_mapping, mapping, nonempty

if TYPE_CHECKING:
    import torch

    from klineage.backend import Backend


DEFINITION_FILE = "definition.json"
WORKLOADS_FILE = "workloads.jsonl"
TARGET_FILE = "target.json"
STRIDES_METADATA = "klineage.strides"


class InputSource(StrEnum):
    RANDOM = "random"
    SAFETENSORS = "safetensors"


def axis_size(
    axis: str, axes: Mapping[str, Any], bindings: Mapping[str, Any] | None
) -> int | str:
    """Resolve a constant or bound axis, retaining unbound variable names."""
    axis = nonempty(axis, "tensor axis")
    if axis not in axes:
        raise ValueError(f"undefined tensor axis {axis!r}")
    spec = mapping(axes[axis], f"axis {axis}")
    kind = spec["type"]
    if kind == "const":
        size = spec["value"]
    elif kind == "var":
        if bindings is None:
            return axis
        if axis not in bindings:
            raise ValueError(f"workload axis {axis!r} is missing")
        size = bindings[axis]
    else:
        raise ValueError(f"axis {axis!r} must be const or var")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"axis {axis!r} must resolve to a nonnegative integer")
    if (
        kind == "const" and bindings is not None
        and axis in bindings and bindings[axis] != size
    ):
        raise ValueError(f"workload overrides constant axis {axis!r}")
    return size


def checked_workloads(workloads) -> tuple[dict[str, Any], ...]:
    if isinstance(workloads, (str, bytes)) or not isinstance(workloads, Sequence):
        raise TypeError("problem workloads must be a sequence")
    cases = []
    identifiers = set()
    for index, value in enumerate(workloads):
        case = json_mapping(value, f"workloads[{index}]")
        identifier = nonempty(case.get("uuid"), f"workloads[{index}].uuid")
        if identifier in identifiers:
            raise ValueError(f"duplicate workload uuid: {identifier}")
        case["uuid"] = identifier
        identifiers.add(identifier)
        cases.append(case)
    return tuple(cases)


def parse_workloads(text: str, definition: str) -> tuple[dict[str, Any], ...]:
    cases = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = mapping(json.loads(line), f"{WORKLOADS_FILE} line {index}")
        if record.get("definition") != definition:
            raise ValueError(f"{WORKLOADS_FILE} line {index} references a different definition")
        cases.append(record["workload"])
    return checked_workloads(cases)


def load_trace(path: Path) -> ModuleType:
    """Load a problem directory and execute its Trace reference.

    Args:
        path: Directory containing ``definition.json`` and ``workloads.jsonl``.
            The JSONL file is required but may be empty. Fixed input paths are
            relative to this directory unless absolute.

    Returns:
        A module exposing ``definition``, ``workloads``, ``run`` and ``torch_ref``.
        Fixed input paths are absolute; tensor data is not loaded here.
    """
    definition, workloads = read_problem(path)
    return trace_module(definition, workloads)


def read_problem(path: Path) -> tuple[dict, tuple[dict, ...]]:
    """Read a problem directory without importing or executing its reference."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"expected a problem directory: {root}")
    definition = json_mapping(json.loads((root / DEFINITION_FILE).read_text()), DEFINITION_FILE)
    text = (root / WORKLOADS_FILE).read_text()
    workloads = parse_workloads(text, definition["name"])
    for workload in workloads:
        if set(workload["inputs"]) != set(definition["inputs"]):
            raise ValueError("Workload inputs must match definition inputs")

        # Artifact consumers run in other directories; retain absolute input paths.
        for descriptor in workload["inputs"].values():
            if descriptor["type"] == InputSource.SAFETENSORS:
                descriptor["path"] = str((root / descriptor["path"]).resolve(strict=True))

    return definition, workloads


def trace_module(
    definition: Mapping[str, Any], workloads: Sequence[Mapping[str, Any]] = ()
) -> ModuleType:
    """Execute a definition's reference and expose its ordered workload tuple."""
    workloads = checked_workloads(workloads)
    module = ModuleType(definition["name"])
    exec(  # noqa: S102 - Trace references are executable Python.
        compile(definition["reference"], f"<{definition['name']}>", "exec"),
        module.__dict__,
    )
    if not callable(getattr(module, "run", None)):
        raise TypeError("Trace reference must define callable run()")
    module.__dict__.update(
        PROBLEM_NAME=definition["name"],
        OPERATOR=definition["op_type"],
        definition=definition,
        workloads=workloads,
        torch_ref=module.run,
    )
    return module


def trace_inputs(
    definition: Mapping[str, Any],
    workload: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    from klineage.backend import get_backend

    # Stage NPU inputs on CPU; generator and safetensors device support vary by release.
    target = str(device)
    on_npu = target.split(":", 1)[0] == "npu"
    if on_npu:
        from klineage.backend.runtime import load_torch

        load_torch(get_backend("ascendc"))
    source_device = "cpu" if on_npu else target
    generator = torch.Generator(device=source_device).manual_seed(seed)
    inputs = {}
    for name, spec in definition["inputs"].items():
        if spec["shape"] is None:
            raise ValueError("The evaluator requires tensor inputs")
        shape = tuple(
            axis_size(axis, definition["axes"], workload["axes"])
            for axis in spec["shape"]
        )
        dtype = getattr(torch, spec["dtype"])
        descriptor = workload["inputs"][name]
        source = InputSource(descriptor["type"])
        if source == InputSource.RANDOM:
            if not dtype.is_floating_point:
                raise ValueError(
                    "Random integer inputs require an explicit safetensors source"
                )
            value = torch.randn(
                shape, dtype=dtype, device=source_device, generator=generator
            )
            if on_npu:
                value = value.to(device)
        else:
            with safe_open(
                descriptor["path"], framework="pt", device=source_device
            ) as tensors:
                value = tensors.get_tensor(descriptor["tensor_key"])
                if on_npu:
                    value = value.to(device)
                layouts = json.loads(
                    (tensors.metadata() or {}).get(STRIDES_METADATA, "{}")
                )
                stride = layouts.get(descriptor["tensor_key"])
                if stride is not None and tuple(stride) != value.stride():
                    # Safetensors stores contiguous values; restore captured layout.
                    restored = torch.empty_strided(
                        value.shape, stride, dtype=value.dtype, device=value.device
                    )
                    restored.copy_(value)
                    value = restored
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Workload input {name!r} does not match its shape/dtype")
        inputs[name] = value
    return inputs


def trace_definition(
    name: str,
    operator: str,
    source: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    axes, tensors = {}, {}
    for role, values in (("inputs", inputs), ("outputs", outputs)):
        tensors[role] = {}
        for tensor_name, value in values.items():
            shape = []
            for index, size in enumerate(value.shape):
                axis = f"{role}_{tensor_name}_{index}"
                axes[axis] = {"type": "const", "value": size}
                shape.append(axis)
            tensors[role][tensor_name] = {
                "shape": shape,
                "dtype": str(value.dtype).removeprefix("torch."),
                "description": f"Tensor with element strides {list(value.stride())}.",
            }

    # Trace references expose run; existing problem modules expose torch_ref.
    args = ", ".join(inputs)
    reference = f"{source.rstrip()}\n\ndef run({args}):\n    return torch_ref({args})\n"
    compile(reference, "<trace reference>", "exec")
    return {
        "name": name,
        "op_type": operator,
        "axes": axes,
        **tensors,
        "reference": reference,
    }


def trace_workload(inputs: Mapping[str, Any], path: Path) -> dict[str, Any]:
    from safetensors.torch import save_file

    # Preserve actual inputs; make_inputs may encode a nonrandom distribution.
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in inputs.items()
        },
        str(path),
        metadata={
            STRIDES_METADATA: json.dumps(
                {name: list(value.stride()) for name, value in inputs.items()}
            )
        },
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_OID, digest)),
        "axes": {},
        "inputs": {
            name: {"type": "safetensors", "path": str(path), "tensor_key": name}
            for name in inputs
        },
    }


@agent_function
def load_problem(
    value: str | Path | FileRef | Artifact | Mapping,
    *,
    backend: Backend | None = None,
) -> Problem:
    """Resolve a problem directory, artifact or serialized artifact JSON.

    Args:
        value: A ``Problem``; an artifact whose ``metadata.json`` references
            one; a directory reference with ``path/name/language/platform``;
            a ``FileRef`` or path to serialized problem JSON; or a directory
            containing ``definition.json`` and ``workloads.jsonl``. Bare Trace
            definition files are not complete problems.
        backend: Target for a directory without ``target.json``. Defaults to
            detected hardware; stored target metadata takes precedence.

    Returns:
        The problem specification. A problem ``FileRef`` is checked before its
        JSON is read. Referenced tensor files are checked by ``resolve_inputs``
        when consumed, not when serialized problem JSON is restored.

    Loading a problem does not execute its reference code. Kernel metadata stores
    its directory and execution target; that directory must be locally accessible.
    """
    if isinstance(value, Problem):
        return value
    if isinstance(value, Artifact):
        if value.kind == Problem.artifact_kind:
            return Problem.from_tree(value.kind, value.tree)
        try:
            metadata = mapping(
                json.loads(raw_text(value.node(KERNEL_METADATA))), KERNEL_METADATA
            )
            return load_problem(mapping(metadata["problem"], "problem reference"))
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"{KERNEL_METADATA}/problem: {error}") from error
    if isinstance(value, FileRef):
        return from_json(raw_text(value), Problem)
    if isinstance(value, Mapping):
        if set(value) != {"path", "name", "language", "platform"}:
            raise ValueError("problem reference requires path, name, language and platform")
        path = Path(nonempty(value["path"], "problem path"))
        if not path.is_absolute():
            raise ValueError("problem path must be absolute")
        definition, workloads = read_problem(path)
        return Problem(
            value["name"], definition, workloads,
            language=value["language"], platform=value["platform"],
        )

    path = Path(value).expanduser().resolve(strict=True)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("artifact_type") == ArtifactKind.PROBLEM
        ):
            return restore(payload, Problem)
        raise ValueError("problem input must be a directory or serialized Problem JSON")

    definition, workloads = read_problem(path)
    target = path / TARGET_FILE
    if target.exists():
        metadata = mapping(json.loads(target.read_text()), TARGET_FILE)
        return Problem(
            name=metadata["name"], definition=definition,
            workloads=workloads, language=metadata["language"],
            platform=metadata["platform"],
        )
    from klineage.backend import detect_backend

    backend = backend or detect_backend()
    return Problem(
        name=definition["name"],
        definition=definition,
        workloads=workloads,
        language=backend.language,
        platform=backend.info()["platform"],
    )


def problem_reference(path: Path, problem: Problem) -> dict[str, str]:
    """Reference an existing directory with its registered execution target."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"expected a problem directory: {root}")
    return {"path": str(root), **problem.target()}


def load_problem_module(value: str | Path) -> tuple[ModuleType, Path]:
    """Load a problem directory or Python reference as an executable module."""

    path = Path(value).expanduser().absolute().resolve(strict=True)
    if path.is_dir():
        return load_trace(path), path / DEFINITION_FILE
    if not path.is_file() or path.suffix != ".py":
        raise ValueError(
            "problem_path must identify a problem directory or Python file"
        )
    name = f"_klineage_problem_{hashlib.sha256(str(path).encode()).hexdigest()[:MODULE_HASH_LENGTH]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("make_inputs", "torch_ref"):
        if not callable(getattr(module, symbol, None)):
            raise TypeError(f"problem must define callable {symbol}()")
    return module, path



class Problem(Artifact):
    """One operator contract, optional workloads and an execution target.

    The tree holds ``target.json``, ``definition.json`` and required
    ``workloads.jsonl`` (empty for zero cases). Safetensors inputs become checked ``FileRef``
    nodes under ``inputs/``; their bytes remain in external files. Properties
    read tree content rather than maintaining a second copy of the contract.

    Args:
        name (str): Nonempty problem name.
        definition (Mapping[str, Any]): Trace definition, including ordered
            inputs/outputs and axes.
        workloads (Sequence[Mapping[str, Any]]): Ordered Trace cases with unique
            nonempty UUIDs, axis bindings and input descriptors; defaults to none.
            Safetensors paths must exist when constructing these cases.
        language (str): Backend language, such as ``cuda`` or ``ascendc``.
        platform (str): Backend target identifier, such as ``nvidia``.

    Construction pins fixed inputs. JSON restoration preserves the pins without
    reading tensor files; ``resolve_inputs`` checks them before execution.
    """

    __slots__ = ()
    artifact_kind = ArtifactKind.PROBLEM

    @classmethod
    @agent_function
    def problem_spec(cls) -> Node:
        """Return the base problem layout; workloads determine fixed-input files.

        Construction and loading check this layout with ``Node.check``.
        Subclasses can override it to define their own layout.
        """
        return Node.directory(
            ".",
            description="\n".join(
                (
                    "One operator contract, zero or more workloads and an execution target.",
                    "Keep referenced input files available while the problem is in use.",
                )
            ),
            children=(
                Node.file(TARGET_FILE, description="Problem name, language and platform."),
                Node.file(
                    DEFINITION_FILE,
                    description="\n".join((
                        "FlashInfer Trace definition.",
                        "Preserve ordered inputs/outputs, dtypes, shapes, axes and reference.run.",
                        "An optional check_outputs cannot bypass the numerical comparison.",
                    )),
                ),
                Node.file(
                    WORKLOADS_FILE,
                    description="\n".join((
                        "Required JSONL file containing zero or more ordered Trace records with unique workload UUIDs.",
                        "Each record contains definition and workload; an empty file means no evaluation cases.",
                        "Preserve axis bindings, input descriptors and tensor strides.",
                        "File inputs use paths relative to this problem tree.",
                    )),
                ),
            ),
        )

    def __init__(self, name, definition, workloads=(), *, language, platform):
        target = {
            key: nonempty(value, f"problem {key}")
            for key, value in (
                ("name", name),
                ("language", language),
                ("platform", platform),
            )
        }
        definition = json_mapping(definition, "problem definition")
        workloads = checked_workloads(workloads)
        inputs = {}
        references = {}
        for workload in workloads:
            for descriptor in workload.get("inputs", {}).values():
                if descriptor.get("type") != "safetensors":
                    continue
                path = Path(descriptor["path"]).expanduser().absolute()
                if path not in references:
                    references[path] = file_ref(path)
                reference = references[path]
                filename = reference.sha256 + ".safetensors"
                descriptor["path"] = "inputs/" + filename
                inputs.setdefault(
                    filename,
                    Node.file(
                        filename, reference=reference, description="Fixed workload input."
                    ),
                )
        records = "".join(
            json.dumps(
                {"definition": definition.get("name", name), "workload": workload},
                ensure_ascii=False,
            ) + "\n"
            for workload in workloads
        )
        children = [
            json_node(TARGET_FILE, target),
            json_node(DEFINITION_FILE, definition),
            Node.file(WORKLOADS_FILE, records),
        ]
        if inputs:
            children.append(Node.directory("inputs", tuple(inputs.values())))
        tree = Node.directory(".", children, description=self.problem_spec().description)
        Artifact.__init__(self, self.artifact_kind, tree)

    def validate(self):
        """Check target fields, workload UUIDs and definition links without loading tensors."""
        layout = self.problem_spec()
        layout.check(Node.directory(self.tree.name, (
            child for child in self.tree.children if child.name != "inputs"
        )))
        target = self.target()
        for field in ("name", "language", "platform"):
            nonempty(target[field], f"problem {field}")
        json_mapping(self.definition, "problem definition")
        inputs = {}
        for workload in self.workload_records():
            for descriptor in workload.get("inputs", {}).values():
                if descriptor.get("type") != InputSource.SAFETENSORS:
                    continue
                path = Path(descriptor["path"])
                if len(path.parts) != 2 or path.parts[0] != "inputs":
                    raise ValueError("fixed input path must be inputs/<filename>")
                inputs[path.name] = Node.file(path.name)
        if inputs:
            layout = layout.with_child(Node.directory("inputs", inputs.values()))
        layout.check(self.tree)

    def target(self):
        """Return a fresh mapping containing name, language and platform."""
        return json.loads(raw_text(self.node(TARGET_FILE)))

    @property
    def name(self):
        """Problem name stored in the target file."""
        return self.target()["name"]

    @property
    def language(self):
        """Backend language stored in the target file."""
        return self.target()["language"]

    @property
    def platform(self):
        """Backend platform stored in the target file."""
        return self.target()["platform"]

    @property
    def definition(self):
        """Return a fresh Trace definition mapping, preserving tensor order."""
        return json.loads(raw_text(self.node(DEFINITION_FILE)))

    def workload_records(self) -> tuple[dict[str, Any], ...]:
        """Read workload records with tree-relative fixed-input paths."""
        return parse_workloads(
            raw_text(self.node(WORKLOADS_FILE)), self.definition.get("name", self.name)
        )

    @property
    def workloads(self) -> tuple[dict[str, Any], ...]:
        """Return ordered fresh workload mappings with absolute fixed-input paths.

        This view does not access or check referenced tensor files. Call
        ``resolve_inputs`` before using the paths for execution.
        """
        workloads = self.workload_records()
        for workload in workloads:
            for descriptor in workload.get("inputs", {}).values():
                if descriptor.get("type") != "safetensors":
                    continue
                node = self.node(descriptor["path"])
                if node.reference is None:
                    raise ValueError("binary inputs require a checked file reference")
                descriptor["path"] = node.reference.path
        return workloads

    def resolve_inputs(
        self, workload_id: str | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Check fixed inputs and return all workloads, or the selected UUID as a tuple.

        Args:
            workload_id: One workload UUID, or ``None`` to check every case.

        Raises:
            ValueError: The UUID is unknown or a selected fixed input is missing,
                changed or lacks a file reference.
        """
        workloads = self.workload_records()
        if workload_id is not None:
            workloads = tuple(case for case in workloads if case["uuid"] == workload_id)
            if not workloads:
                raise ValueError(f"unknown workload uuid: {workload_id}")
        checked = set()
        for workload in workloads:
            for descriptor in workload.get("inputs", {}).values():
                if descriptor.get("type") != "safetensors":
                    continue
                reference = self.node(descriptor["path"]).reference
                if reference is None:
                    raise ValueError("binary inputs require a checked file reference")
                if reference not in checked:
                    check_ref(reference)
                    checked.add(reference)
                descriptor["path"] = reference.path
        return workloads

    def replace(self, **changes):
        """Construct a new problem from field overrides.

        Retained fixed inputs are checked before rebuilding the artifact. A new
        ``workloads`` sequence replaces them without requiring old files to exist.
        """
        fields = {
            name: getattr(self, name)
            for name in ("name", "definition", "language", "platform")
        }
        fields["workloads"] = (
            changes["workloads"] if "workloads" in changes else self.resolve_inputs()
        )
        return type(self)(**(fields | changes))

    def values(
        self, role: str, workload: Mapping[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Read ordered definition fields, binding shapes when one case is selected.

        Args:
            role: ``"inputs"`` or ``"outputs"``.
            workload: Explicit case to bind. If omitted, a sole stored case is
                used; zero or multiple cases retain variable axes as strings.

        Returns:
            A detached mapping from names to definition fields, in their original
            order. Tensor shapes are lists; scalar shapes remain None. Constant
            axes always resolve to integers. Tensor data is never read.
        """

        if role not in ("inputs", "outputs"):
            raise ValueError("role must be 'inputs' or 'outputs'")
        definition = self.definition
        specs = dict(mapping(definition[role], f"definition {role}"))
        axes = mapping(definition["axes"], "definition axes")
        if workload is None:
            cases = self.workloads
            workload = cases[0] if len(cases) == 1 else None
        bindings = (
            mapping(workload["axes"], "workload axes") if workload is not None else None
        )

        for name, raw in specs.items():
            nonempty(name, f"definition {role} name")
            spec = mapping(raw, f"definition {role}.{name}")
            if spec["dtype"] is not None:
                nonempty(spec["dtype"], f"definition {role}.{name}.dtype")
            if spec.get("description") is not None and not isinstance(
                spec["description"], str
            ):
                raise TypeError(f"definition {role}.{name}.description must be text")
            shape = spec["shape"]
            if shape is not None and not isinstance(shape, list):
                raise TypeError(
                    f"definition {role}.{name}.shape must be an array or null"
                )
            if shape is not None:
                spec["shape"] = [axis_size(axis, axes, bindings) for axis in shape]
        return specs
