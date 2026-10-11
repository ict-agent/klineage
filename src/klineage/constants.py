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
#: Modules defining functions for the generated API catalog.
AGENT_MODULES = (
    "klineage.backend",
    "klineage.artifact",
    "klineage.harness",
    "klineage.tools",
    "klineage.server",
)
#: Workspace guidance placed above the generated agent-function catalog.
AGENTS_TEMPLATE = """Write optimized device kernels in this isolated workspace.
Use only the devices, runtimes and tools explicitly supplied to this sandbox.
Official correctness and acceptance timings come from the configured evaluation
server through `klineage.server.Client`.

Query artifact layouts through Python when needed; print Node layouts with
`klineage.serialization.raw_text`. Use `Problem.problem_spec()` from
`klineage.artifact`, the selected backend's `kernel_type.kernel_spec()` from
`klineage.backend.get_backend(language, platform)`, `SkillCard.schema` from
`klineage.memory`, and `action_spec(stage)` from `klineage.prompts`.
Read artifact data from the input paths supplied in the task.

Requirements:
- Read `Client().info()` for the device and registered problems. Select the task's ID with `Client(problem_id=...)`.
- Write trial metadata.json and submission/ under current_kernel/. The entry source owns launch parameters.
- Submit with the selected client's `evaluate(Path("current_kernel"), Path.cwd())`; read reply.validation/reply.diagnostics or the saved Kernel metadata.json.
- Client.evaluate/result with work automatically append complete Kernel directories under versions/0, versions/1, ... including failures; metadata.json contains validation.workloads and evaluation={job_id,error,diagnostics}. Do not save or edit these versions.
- Keep all files outside the artifact directories under workspace/: temporary sources, backups, scripts, diagnostics, logs, reports and compiled binaries. Keep the workdir root free of scratch files.
- The final result is the largest numbered version; a failed latest version does not fall back. Retrieve an earlier accepted job with client.result(job_id, work=Path.cwd()) to append it as the latest version.
- With supplied device/toolchain access, write diagnostic scripts and invoke vendor tools directly, such as CUDA ncu or HIP/CANN profilers.
- Diagnostic measurements do not populate validation or replace service results; keep the oracle and evaluation policy fixed.
- Do not run diagnostics and service benchmarks concurrently on the same device. Do not change host driver permissions.
- Keep supplied inputs, mounted repositories and skills unchanged.
"""


#: Start marker for generated agent function documentation.
FUNCTION_START = "<!-- klineage:agent-functions:start -->"
#: End marker for generated agent function documentation.
FUNCTION_END = "<!-- klineage:agent-functions:end -->"
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
