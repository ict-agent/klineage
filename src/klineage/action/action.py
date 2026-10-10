"""Action execution and collection of output artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from klineage.artifact import (
    ACTION_TYPES,
    Artifact,
    ArtifactKind,
    Kernel,
    load_artifact,
    load_problem,
    source_files,
)
from klineage.artifact.kernel import capture_kernel, latest_kernel
from klineage.artifact.tree import (
    ACTION_PREFIX,
    Node,
    capture,
    nested,
)
from klineage.constants import (
    AGENT_FILES,
    BUILD_DIRECTORY,
    KERNELS_DIRECTORY,
    MAX_RETRIES,
    SKILL_DIRECTORY,
    SKILL_FILE,
    STATE_DIRECTORY,
    TIMEOUT,
    RunKind,
)
from klineage.errors import StructuredOutputError
from klineage.harness import AgentRuntime, CodexRunner, session_id
from klineage.memory.storage import load_skill
from klineage.prompts import action_spec
from klineage.serialization import raw_text
from klineage.server import Client
from klineage.utils import operation_id


class Action:
    """Run an agent and collect its files as an artifact tree.

    Built-in kernel actions require a service evaluation matching the returned
    Kernel. Collection checks the artifact format and handoff consistency;
    semantic optimization goals remain instructions to the agent.

    Args:
        prompt: Instructions supplied to the agent.
        workdir: Workspace for generated sources and evaluation records.
        timeout: Seconds allowed for each agent attempt.
        max_retries: Additional attempts after runner or collection failures.
        output_type: Artifact class for custom outputs. Its Node schema is
            appended to the prompt and checked during construction.
        evaluation: Evaluation service client. Built-in kernel actions create
            a client from the environment when omitted.
        agent: Optional sandbox runtime, mounts, and diagnostic devices.
            The default runtime exposes no device nodes.
    """

    artifact_type = ArtifactKind.ACTION

    def __init__(
        self,
        prompt: str,
        workdir: Path,
        timeout: int = TIMEOUT,
        *,
        max_retries: int = MAX_RETRIES,
        output_type: type[Artifact] | None = None,
        evaluation: Client | None = None,
        agent: AgentRuntime | None = None,
    ):
        if type(max_retries) is not int:
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")

        self.output_type = output_type or Artifact
        if output_type is not None:
            if not issubclass(output_type, Artifact) or output_type.artifact_kind is None:
                raise TypeError("output_type must be a concrete Artifact class")
            if not isinstance(output_type.schema, Node):
                raise TypeError("custom action output requires a Node schema")
            self.artifact_type = output_type.artifact_kind
            prompt += "\n\nArtifact output contract:\n" + raw_text(output_type.schema)
        self.prompt = prompt
        self.workdir = Path(workdir).expanduser().resolve()
        self.max_retries = max_retries
        self.evaluation = evaluation
        self.agent = agent or AgentRuntime()
        self.read_only = ()
        self.runner = CodexRunner(
            self.workdir, timeout=timeout, evaluation=evaluation, agent=self.agent,
        )

    def run(self) -> Artifact:
        """Execute, check the handoff, and return the collected artifact.

        Each failed attempt may resume its own recorded agent session. Retries
        retain the configured runner and receive the failure as feedback.
        After the final attempt, the original exception is propagated.

        Returns:
            An Artifact of the selected kind. Built-in kernel actions include
            a ``kernel`` part, available evaluation and agent records, and a
            ``skillcard`` part when Decompose changed the implementation.
        """
        if self.artifact_type in ACTION_TYPES and self.artifact_type != ArtifactKind.ACTION:
            self.evaluation = self.evaluation or Client()
            self.runner.evaluation = self.evaluation
            before = getattr(self, "input_kernel", None)
            if before is not None:
                self.read_only = tuple(dict.fromkeys((*self.read_only, *input_paths(before))))
            self.runner.read_only = self.read_only
        prompt = self.prompt
        for retry in range(self.max_retries + 1):
            result = None
            try:
                result = self.runner(prompt, run_id=operation_id(type(self).__name__))
                return collect_output(
                    self.workdir,
                    self.artifact_type,
                    getattr(self, "input_kernel", None),
                    evaluation=self.evaluation,
                    output_type=self.output_type,
                )
            except Exception as error:
                if retry == self.max_retries:
                    raise
                # Continue the failed attempt's own session when it left a trace,
                # so the retry keeps the context it built instead of restarting.
                trace = getattr(error, "trace_path", None) or getattr(result, "trace_path", None)
                self.resume_from(trace)
                prompt = (
                    f"Previous attempt failed: {type(error).__name__}: {error}\n"
                    "Read the prior generation records in this workdir. "
                    "Correct the failure. For kernel actions, fix trial sources "
                    "under kernel/ and evaluate again, or select "
                    "an accepted job with client.result(job_id, work=Path.cwd()). "
                    "Leave saved versions under kernels/ unchanged.\n\n"
                    f"{self.prompt}"
                )

    def resume_from(self, trace: Path | None) -> None:
        """Continue the generation session after runner or artifact-check failures."""

        if trace is None:
            return
        resumed = session_id(trace)
        if resumed is None:
            return
        self.runner.resume_id = resumed


def task_input(value: Artifact | Path):
    return (
        value
        if isinstance(value, Artifact)
        else str(Path(value).expanduser().resolve())
    )


def input_paths(value) -> tuple[Path, ...]:
    """Mount explicit source files and the referenced problem directory."""
    paths = []
    if not isinstance(value, Artifact):
        path = Path(value).expanduser().resolve(strict=True)
        if path.is_dir() and (path / KERNELS_DIRECTORY).exists():
            path = latest_kernel(path)
        paths.append(path)
        value = (
            Artifact(ArtifactKind.KERNEL, capture_kernel(path))
            if path.is_dir() else load_artifact(path)
        )

    def visit(node):
        if node.reference is not None:
            paths.append(Path(node.reference.path))
        for child in node.children:
            visit(child)

    if value.kind == ArtifactKind.PROBLEM:
        visit(value.tree)
    if value.kind in (ArtifactKind.INIT, ArtifactKind.APPLY, ArtifactKind.DECOMPOSE):
        value = value.part("kernel", Kernel)
    if value.kind.startswith("kernel"):
        visit(value.node("submission"))
        metadata = json.loads(raw_text(value.node("metadata.json")))
        problem = Path(metadata["problem"]["path"]).expanduser().resolve(strict=True)
        if not problem.is_dir():
            raise NotADirectoryError(f"metadata.json/problem: expected a directory: {problem}")
        paths.append(problem)
    return tuple(dict.fromkeys(paths))


def same_implementation(before: Kernel, after: Kernel) -> bool:
    before_meta = json.loads(raw_text(before.node("metadata.json")))
    after_meta = json.loads(raw_text(after.node("metadata.json")))
    return (
        load_problem(before) == load_problem(after)
        and source_files(before.node("submission"))
        == source_files(after.node("submission"))
        and before_meta.get("compile_flags", []) == after_meta.get("compile_flags", [])
    )


def check_handoff(before: Kernel, after: Kernel, stage: RunKind, card):
    before_problem, after_problem = load_problem(before), load_problem(after)
    if before_problem != after_problem:
        raise StructuredOutputError(f"{stage} changed the problem")
    if same_implementation(before, after):
        if card is not None:
            raise StructuredOutputError(
                f"{stage} emitted a card for an unchanged kernel"
            )
        before_meta = json.loads(raw_text(before.node("metadata.json")))
        after_meta = json.loads(raw_text(after.node("metadata.json")))
        if before_meta["name"] != after_meta["name"]:
            raise StructuredOutputError(f"{stage} changed an unchanged kernel's metadata")
        return
    if stage is RunKind.DECOMPOSE and card is None:
        raise StructuredOutputError("decompose changed a kernel without SKILL.md")


def capture_output(work: Path, layout: Node) -> Node:
    """Capture user outputs, preserving reserved names declared by the layout."""
    names = {child.name for child in layout.children}
    reserved = {
        STATE_DIRECTORY, SKILL_DIRECTORY.parts[0], BUILD_DIRECTORY.name,
        "problems", *AGENT_FILES,
    }
    return Node.directory(".", (
        capture(path) for path in sorted(work.iterdir())
        if path.name in names or path.name not in reserved
    ))


def collect_output(
    work: Path, kind: str, before=None, *, evaluation=None,
    output_type: type[Artifact] = Artifact,
) -> Artifact:
    if output_type is not Artifact:
        return output_type.from_tree(kind, capture_output(work, output_type.schema))
    children = []
    if kind != ArtifactKind.ACTION:
        if evaluation is None:
            raise StructuredOutputError("kernel actions require an evaluation service")
        stage = RunKind(kind.removeprefix(ACTION_PREFIX))
        try:
            directory = latest_kernel(work)
        except FileNotFoundError as error:
            raise StructuredOutputError("no evaluated kernel snapshot") from error
        kernel_kind = evaluation.info()["kernel_kind"]
        current = Kernel.from_tree(kernel_kind, capture_kernel(directory))
        measured = evaluation.verify(current)
        if not measured.accepted:
            raise StructuredOutputError("handoff did not pass server evaluation")
        current = measured.kernel
        skill_path = work / SKILL_FILE
        if skill_path.is_symlink():
            raise StructuredOutputError("SKILL.md cannot be a symlink")
        card = load_skill(skill_path) if skill_path.exists() else None
        if stage is not RunKind.DECOMPOSE and card is not None:
            raise StructuredOutputError(f"{stage} must not emit SKILL.md")
        if before is not None:
            if not isinstance(before, Artifact):
                path = Path(before).expanduser()
                if path.is_dir() and (path / KERNELS_DIRECTORY).exists():
                    path = latest_kernel(path)
                before = load_artifact(path, Kernel if path.is_dir() else Artifact)
            if not isinstance(before, Kernel):
                before = (
                    Kernel.from_tree(before.kind, before.tree)
                    if before.kind.startswith("kernel.") else before.part("kernel", Kernel)
                )
            check_handoff(before, current, stage, card)
        children.append(nested("kernel", current))
        if card is not None:
            children.append(nested("skillcard", card))

    records = work / ".klineage" / "codex-runs"
    if records.exists():
        children.append(replace(capture(records), name="records"))
    schema = action_spec(kind.removeprefix(ACTION_PREFIX))
    tree = Node.directory(".", children, description=schema.description)
    return Artifact.from_tree(kind, tree)


__all__ = ["Action"]
