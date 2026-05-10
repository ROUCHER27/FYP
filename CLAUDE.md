# FYP Project — Claude Instructions

## Required Project Context

Before writing, auditing, or reorganizing the final report, read:

1. `SCHEMA.md`
2. `doc/agent_handoff.md`
3. `doc/thesis/*`
4. The evidence files referenced by `SCHEMA.md`

Treat `SCHEMA.md` as the highest-level evidence and claim-boundary contract.

For the writing phase, use Kiro as the primary drafting and figure-generation workflow. The local skills below should be read before drafting, revising, planning chapter structure, or generating figures. Kiro is not the final reviewer; after Kiro produces a draft, Codex should audit it against `SCHEMA.md`, `doc/agent_handoff.md`, and the evidence files. If `.kiro` is missing, restore/install the Kiro skill bundle before substantial writing work; use `SCHEMA.md` and `doc/agent_handoff.md` only as the continuity fallback.

## Skills

The following Kiro skill files are the preferred writing-stage tools. Read them when the relevant task arises.

### Scientific Writing / Manuscript Editing
@.kiro/skills/sciwrite/SKILL.md

### Plot From Data (paper-quality matplotlib figures)
@.kiro/skills/paper-plot-skills/plot-from-data/SKILL.md

### Plot From Image (reproduce paper figures)
@.kiro/skills/paper-plot-skills/plot-from-image/SKILL.md

### Research & Experiment Skills (EvoSkills)
@.kiro/skills/EvoSkills/skills/paper-writing/SKILL.md
@.kiro/skills/EvoSkills/skills/paper-navigator/SKILL.md
@.kiro/skills/EvoSkills/skills/paper-planning/SKILL.md
@.kiro/skills/EvoSkills/skills/paper-rebuttal/SKILL.md
@.kiro/skills/EvoSkills/skills/academic-slides/SKILL.md
@.kiro/skills/EvoSkills/skills/experiment-craft/SKILL.md
@.kiro/skills/EvoSkills/skills/experiment-pipeline/SKILL.md
@.kiro/skills/EvoSkills/skills/experiment-iterative-coder/SKILL.md
@.kiro/skills/EvoSkills/skills/research-ideation/SKILL.md
@.kiro/skills/EvoSkills/skills/research-survey/SKILL.md
@.kiro/skills/EvoSkills/skills/evo-memory/SKILL.md
@.kiro/skills/EvoSkills/skills/nano-banana/SKILL.md
