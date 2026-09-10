"""Shared workspace paths, operation names, and execution defaults."""

from enum import StrEnum
from pathlib import Path


class RunKind(StrEnum):
    """Operation names used in prompts and workspace directories."""

    #: Initial kernel generation.
    INIT = "init"
    #: Removal of one optimization.
    DECOMPOSE = "decompose"
    #: One full kernel optimization round, with optional skill memory.
    APPLY = "apply"
    #: Review of an action's result.
    VERIFY = "verify"
    #: Skill memory generation from an expert kernel.
    INIT_MEMORY = "init-memory"
    #: Combined memory generation and optimization.
    WORKFLOW = "workflow"
    #: Iterative optimization with optional skill memory.
    OPTIMIZE = "optimize"
    #: Agent process execution.
    CODEX = "codex"
    #: Kernel correctness and timing evaluation.
    EVALUATE = "evaluate"


#: Parent directory for automatically allocated workspaces.
WORKSPACE_DIR = "agent-workspace"
#: Serialized kernel artifact filename.
KERNEL_FILE = "kernel.json"
#: Source bundle configuration filename.
BUNDLE_CONFIG = "config.toml"
#: Source directory within a configured bundle.
BUNDLE_SOLUTION = "solution"
#: Generated source bundle within an action workspace.
SUBMISSION_DIRECTORY = Path("submission")
#: Skill card filename within a skill directory.
SKILL_FILE = "SKILL.md"
#: Internal workspace state directory.
STATE_DIRECTORY = ".klineage"
#: Directory name for generated or mounted skill memory.
MEMORY_DIR = "memory"
#: Workspace directory discovered by the agent for skills.
SKILL_DIRECTORY = Path(".agents") / "skills"
#: Skill memory mount relative to the agent workspace.
MEMORY_DIRECTORY = SKILL_DIRECTORY / MEMORY_DIR
#: Message schemas relative to the agent workspace.
MESSAGE_DIRECTORY = Path(STATE_DIRECTORY) / "message"
#: Build artifacts relative to the working directory.
BUILD_DIRECTORY = Path("build")
#: Evaluation logs relative to the working directory.
EVALUATIONS_DIRECTORY = Path("evaluations")
#: Workspace instruction files; a nonempty override takes precedence.
AGENT_FILES = ("AGENTS.md", "AGENTS.override.md")
#: Modules defining built-in agent functions beyond tools.profile.
AGENT_MODULES = (
    "klineage.backend",
    "klineage.artifact.kernel",
    "klineage.artifact.repository",
    "klineage.harness.artifacts",
    "klineage.harness.eval",
)
#: Start marker for generated agent function documentation.
FUNCTION_START = "<!-- klineage:agent-functions:start -->"
#: End marker for generated agent function documentation.
FUNCTION_END = "<!-- klineage:agent-functions:end -->"
#: Default retry limit after an action's first attempt.
MAX_RETRIES = 3
#: Default action timeout in seconds.
TIMEOUT = 3600
#: Default maximum number of Decompose invocations.
MAX_DECOMPOSE_STEPS = 15
#: Default maximum number of Apply optimization rounds.
MAX_APPLY_STEPS = 15
