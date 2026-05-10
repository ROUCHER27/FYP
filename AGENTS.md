# Repository Guidelines

## Project Structure & Module Organization
- Root CSV files (e.g., `04-09.csv`) are the main input data for model training and evaluation.
- `G:MADL/` contains GMADL research notebooks, figures, and notes (primarily Phase 1 analysis and visualization).
- `Model_Train/` is reserved for Python training/evaluation scripts and experiment configuration files.
- `loss_function_design_empirical___to_do.pdf` describes the assignment phases; keep new code and reports aligned with this document.

## Build, Test, and Development Commands
- Use Python 3.10+ in a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies as needed (typical: `pip install numpy pandas matplotlib seaborn jupyter`).
- Run notebooks in `G:MADL/` via `jupyter lab` or `jupyter notebook`, executing cells top-down.
- Run training scripts from the repo root, e.g., `python Model_Train/train_gmadl.py --config configs/base.yaml` (keep CLI flags short and explicit).

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and UTF-8 encoding; comments may be in Chinese or English.
- Use `snake_case` for variables/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Name notebooks descriptively by task (e.g., `GMADL_Improvement_Test.ipynb`) and avoid spaces in filenames.
- When available, format Python code with `black` and lint with `ruff` or `flake8` before committing.

## Testing Guidelines
- Place tests under a `tests/` directory (create it if missing), mirroring the structure of `Model_Train/`.
- Name test files `test_*.py` and test functions `test_*`.
- Use `pytest` as the default test runner: `pytest -q`.
- For new loss functions, add numeric sanity tests (direction correctness, symmetry, extreme values, and gradient behaviour).

## Skills

For final-report work, the hard project contract is `SCHEMA.md`; read it before `doc/thesis/*`, experiment reports, or old phase notes. `doc/agent_handoff.md` gives the current source-of-truth evidence map and cross-branch lookup commands.

The following local skill files are optional helpers. Use them when present, but do not block if they are missing; `SCHEMA.md` and `doc/agent_handoff.md` are sufficient for FYP continuity.

| Task | Skill file |
|------|-----------|
| Scientific writing review / manuscript editing | `.kiro/skills/sciwrite/SKILL.md` |
| Generate paper-quality matplotlib figure from data | `.kiro/skills/paper-plot-skills/plot-from-data/SKILL.md` |
| Reproduce a paper figure from an image | `.kiro/skills/paper-plot-skills/plot-from-image/SKILL.md` |
| Paper writing | `.kiro/skills/EvoSkills/skills/paper-writing/SKILL.md` |
| Paper navigation / reading | `.kiro/skills/EvoSkills/skills/paper-navigator/SKILL.md` |
| Paper review | `.kiro/skills/EvoSkills/skills/paper-review/SKILL.md` |
| Paper planning | `.kiro/skills/EvoSkills/skills/paper-planning/SKILL.md` |
| Rebuttal writing | `.kiro/skills/EvoSkills/skills/paper-rebuttal/SKILL.md` |
| Academic slides | `.kiro/skills/EvoSkills/skills/academic-slides/SKILL.md` |
| Experiment design | `.kiro/skills/EvoSkills/skills/experiment-craft/SKILL.md` |
| Experiment pipeline | `.kiro/skills/EvoSkills/skills/experiment-pipeline/SKILL.md` |
| Iterative coding for experiments | `.kiro/skills/EvoSkills/skills/experiment-iterative-coder/SKILL.md` |
| Research ideation | `.kiro/skills/EvoSkills/skills/research-ideation/SKILL.md` |
| Literature survey | `.kiro/skills/EvoSkills/skills/research-survey/SKILL.md` |

## Commit & Pull Request Guidelines
- Write concise commit messages in the form `area: summary`, e.g., `loss: add gmadl heatmap notebook`.
- Keep each commit focused on a single logical change.
- Pull requests should describe purpose, key changes, how to run code/tests, and reference related assignment phases or issues.
- Attach or link key figures (heatmaps, comparison plots) when they are central to the change.
