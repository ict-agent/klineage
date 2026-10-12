# Skill memory API

Decompose records reusable optimizations as SkillCards; Apply exposes supplied
cards to the agent for later kernels. Each card contains a Markdown recipe, scope and prerequisites.
See [Reuse skill memory](../concepts/actions.md#reuse-skill-memory) for the workflow.

## Describe a skill

::: klineage.memory.Scope

::: klineage.memory.SkillCard

## Read, write and retrieve

Use `load_skill` and `save_skill` for `SKILL.md` files. Retrieval filters cards by
scope and excluded IDs, preserving their order. `retrieve(skills, target)` accepts
SkillCards, paths to stored cards, or a directory searched recursively in sorted
path order. Apply does not invoke this filter automatically; its agent checks
the mounted cards' scope and prerequisites.

::: klineage.memory.load_skill

::: klineage.memory.save_skill

::: klineage.memory.parse_skill

::: klineage.memory.skill_text

::: klineage.memory.retrieve
