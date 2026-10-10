# Skill memory API

Decompose records reusable optimizations as SkillCards; Apply retrieves them for
later kernels. Each card contains a Markdown recipe, scope and prerequisites.
See [Reuse skill memory](../concepts/actions.md#reuse-skill-memory) for the workflow.

## Describe a skill

::: klineage.memory.Scope

::: klineage.memory.SkillCard

## Read, write and retrieve

Use `load_skill` and `save_skill` for `SKILL.md` files. Retrieval filters cards by
scope and excluded IDs, preserving their order for the agent to consider.

::: klineage.memory.load_skill

::: klineage.memory.save_skill

::: klineage.memory.parse_skill

::: klineage.memory.skill_text

::: klineage.memory.retrieve
