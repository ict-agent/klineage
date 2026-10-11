"""Shared workspace paths, operation names, and execution defaults."""

from enum import StrEnum
from pathlib import Path


class RunKind(StrEnum):
    """Operation names used in prompts and workspace directories."""

    #: Kernel generation from a problem on the local backend.
    GENERATION = "generation"
    #: Initial kernel generation.
    INIT = "init"
    #: Removal of one optimization.
    DECOMPOSE = "decompose"
    #: One full kernel optimization round, with optional skill memory.
    APPLY = "apply"
    #: Skill memory generation from an expert kernel.
    INIT_MEMORY = "init-memory"
    #: Agent process execution.
    CODEX = "codex"


#: Parent directory for automatically allocated workspaces.
WORKSPACE_DIR = "experiment"
KERNEL_METADATA = "metadata.json"
#: Agent-owned trial sources within the workspace.
CURRENT_KERNEL_DIRECTORY = Path("current_kernel")
#: Framework-owned kernel snapshots, numbered from zero.
VERSIONS_DIRECTORY = Path("versions")
#: Source bundle configuration filename.
BUNDLE_CONFIG = "config.toml"
#: Digest prefix length used in generated module names.
MODULE_HASH_LENGTH = 20
#: Source directory within a configured bundle.
BUNDLE_SOLUTION = "solution"
#: Generated source bundle within an action workspace.
SUBMISSION_DIRECTORY = Path("submission")
#: Skill card filename within a skill directory.
SKILL_FILE = "SKILL.md"
#: Supporting files kept outside artifact directories.
WORK_DIRECTORY = Path("workspace")
#: Internal workspace state directory.
STATE_DIRECTORY = "workspace/.klineage"
#: Directory name for generated or mounted skill memory.
MEMORY_DIR = "memory"
#: Skill files available to the agent on demand.
SKILL_DIRECTORY = WORK_DIRECTORY / ".agents" / "skills"
#: Generated native memory skills relative to the agent workspace.
MEMORY_DIRECTORY = SKILL_DIRECTORY / MEMORY_DIR
#: Maximum name length in Codex SKILL.md frontmatter.
CODEX_SKILL_NAME_LIMIT = 64
#: Maximum description length in native skill frontmatter.
CODEX_SKILL_DESCRIPTION_LIMIT = 1024
#: Digest suffix distinguishing normalized or shortened skill IDs.
CODEX_SKILL_HASH_LENGTH = 12
#: Build artifacts relative to the working directory.
BUILD_DIRECTORY = Path("build")
#: Workspace instruction files; a nonempty override takes precedence.
AGENT_FILES = ("workspace/AGENTS.md", "workspace/AGENTS.override.md")
#: Stable delimiters used to refresh instructions in existing workspaces.
AGENT_DOCS_START = "<!-- klineage:agent-functions:start -->"
AGENT_DOCS_END = "<!-- klineage:agent-functions:end -->"
#: Default retry limit after an action's first attempt.
MAX_RETRIES = 3
#: Default timeout for each agent attempt, in seconds.
TIMEOUT = 3600
#: Absolute and relative tolerances for numerical output comparisons.
NUMERICAL_TOLERANCE = 1e-2
#: Default maximum number of Decompose invocations.
MAX_DECOMPOSE_STEPS = 15
#: Default maximum number of Apply optimization rounds.
MAX_APPLY_STEPS = 15
