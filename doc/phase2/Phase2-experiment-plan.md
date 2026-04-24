# Phase 2 Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, paper-ready Phase 2 experiment line that completes the missing 24-month baselines, reruns or consolidates a fair 7-loss comparison under one fixed configuration, and adds robustness checks for the shortlisted losses.

**Architecture:** Phase 2 is split into three workstreams: first freeze the true experimental configuration and repair documentation drift, then generate one authoritative 24-month comparison set for all seven losses in an isolated output root, and finally run a small robustness matrix on the finalists plus fixed baselines. All outputs must be separated from the existing local `sanity_outputs/` because that directory currently contains mixed 2-month local artifacts and is not a reliable source of final paper tables.

**Tech Stack:** Python 3.10+, PyTorch, pandas, matplotlib, existing `run_sanity_check_*.py` runners, `run_all_experiments.py`, Google Colab, Google Drive

---

## Context Locked Before Phase 2

- Phase 1 produced a valid 24-month Colab run for `gmadl`, `imadl`, `dirhuber`, `hybrid_add`, and `hybrid_mul`, recorded in `/Users/roucher/Documents/FYP/phase1.md`.
- Phase 1 did **not** produce successful 24-month baselines for `mse` and `medse` because the first batch failed on a missing `best_hyperparameters.txt` path.
- The real locked model configuration is the one in `/Users/roucher/Documents/FYP/best_hyperparameters.txt`:

```text
Config: {'input_dim': 15, 'hidden_dims': [64, 32, 16], 'activation': 'tanh', 'dropout': 0.0}
```

- `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md` still says `relu` and `0.2`, so Phase 2 must treat that as a documentation bug, not as the source of truth.
- The local directory `/Users/roucher/Documents/FYP/sanity_outputs/` currently contains 2-month local outputs for all seven losses, so it must **not** be reused as the authoritative Phase 2 result root.

## File Structure

- Create: `/Users/roucher/Documents/FYP/doc/Phase2-experiment-plan.md`
- Create: `/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md`
- Modify: `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md`
- Use: `/Users/roucher/Documents/FYP/phase1.md`
- Use: `/Users/roucher/Documents/FYP/best_hyperparameters.txt`
- Use: `/Users/roucher/Documents/FYP/run_all_experiments.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- Use: `/Users/roucher/Documents/FYP/plot_cumulative_returns.py`
- Use: `/Users/roucher/Documents/FYP/plot_loss_strategy_cumreturn.py`
- Use: `/Users/roucher/Documents/FYP/plot_signalweighted_sharpe.py`
- Use: `/Users/roucher/Documents/FYP/plot_strategy_long_short.py`
- Use: `/Users/roucher/Documents/FYP/tests/test_losses.py`
- Use: `/Users/roucher/Documents/FYP/tests/test_run_all_experiments.py`
- Use: `/Users/roucher/Documents/FYP/tests/test_sanity_check_signal_tilted.py`

### Output Roots To Reserve

- Main 24-month comparison:
  - `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs`
  - `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints`
- Robustness runs:
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed52_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed62_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap`

### Selection Rule For Phase 2 Finalists

- Fixed baselines that must stay in every comparison:
  - `mse`
  - `medse`
- Directional-loss candidates to evaluate in the main 24-month run:
  - `gmadl`
  - `imadl`
  - `dirhuber`
  - `hybrid_add`
  - `hybrid_mul`
- Finalists promoted to robustness stage:
  - Top 2 losses by `long_short_sharpe` among the five directional candidates
  - Plus fixed baselines `mse` and `medse`

### Phase 2 Success Criteria

- One clean 24-month `all_losses_comparison.csv` built from seven losses under one identical configuration
- `mse` and `medse` 24-month baselines are present and directly comparable with the five directional losses
- One robustness table comparing finalist losses across seeds and weight-cap settings
- Runbook and summary docs reflect the actual locked configuration and actual result paths
- Figures and tables are sufficient to write the experiment section without reading raw Colab logs

### Task 1: Freeze Ground Truth And Repair Documentation Drift

**Files:**
- Create: `/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md`
- Modify: `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md`
- Use: `/Users/roucher/Documents/FYP/phase1.md`
- Use: `/Users/roucher/Documents/FYP/best_hyperparameters.txt`

- [ ] **Step 1: Draft a one-page Phase 1 factual summary**

Create `/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md` with these sections:

```markdown
# Phase 1 Factual Summary

## Confirmed successful 24-month runs
- gmadl
- imadl
- dirhuber
- hybrid_add
- hybrid_mul

## Confirmed failed baseline runs
- mse
- medse

## Failure cause
- The first Colab batch pointed `--best-config-path` to `/content/drive/MyDrive/FYP/code/best_hyperparameters.txt`, which did not exist at runtime.

## Locked model config
- input_dim = 15
- hidden_dims = [64, 32, 16]
- activation = tanh
- dropout = 0.0

## Phase 1 directional-loss ranking by logged 24-month Sharpe
1. imadl
2. gmadl
3. hybrid_mul
4. dirhuber
5. hybrid_add
```

- [ ] **Step 2: Update the runbook to match the true locked config**

Edit the network section in `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md` so the table reads:

```markdown
| `input_dim` | `15` |
| `hidden_dims` | `[64, 32, 16]` |
| `activation` | `tanh` |
| `dropout` | `0.0` |
```

- [ ] **Step 3: Add a warning note that local `sanity_outputs/` is not the final Phase 2 source**

Insert this note near the batch-experiment or deliverables section:

```markdown
> [!warning]
> 当前本地仓库中的 `sanity_outputs/` 混有 2 个月本地短测产物，不应直接作为最终论文表格来源。Phase 2 必须使用独立输出目录重新生成或汇总 24 个月主实验结果。
```

- [ ] **Step 4: Verify the documentation edits are the only intended changes**

Run:

```bash
git diff -- /Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md /Users/roucher/Documents/FYP/doc/Phase2-result-summary.md
```

Expected: only the config correction, warning note, and phase1 factual summary appear.

- [ ] **Step 5: Commit the documentation-only change**

Run:

```bash
git add /Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md /Users/roucher/Documents/FYP/doc/Phase2-result-summary.md
git commit -m "doc: lock phase2 experiment ground truth"
```

### Task 2: Build One Authoritative 7-Loss 24-Month Main Comparison

**Files:**
- Use: `/Users/roucher/Documents/FYP/run_all_experiments.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- Create: `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs`
- Create: `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints`

- [ ] **Step 1: Run the local test suite before any long Colab job**

Run locally:

```bash
pytest -q
```

Expected: all tests in `/Users/roucher/Documents/FYP/tests/` pass before the 24-month rerun starts.

- [ ] **Step 2: Create clean Phase 2 output roots in Colab Drive**

Run in Colab:

```python
from pathlib import Path

for path in [
    "/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs",
    "/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints",
]:
    Path(path).mkdir(parents=True, exist_ok=True)
    print("READY", path)
```

Expected: both directories print `READY` and are empty or only contain Phase 2 files.

- [ ] **Step 3: Confirm the real config file exists before running**

Run in Colab:

```bash
cat /content/drive/MyDrive/FYP/code/best_hyperparameters.txt
```

Expected:

```text
Config: {'input_dim': 15, 'hidden_dims': [64, 32, 16], 'activation': 'tanh', 'dropout': 0.0}
```

- [ ] **Step 4: Run the full 7-loss batch in a fresh output root**

Run in Colab:

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto
```

Expected: seven completed losses, no missing summary files, and one new `all_losses_comparison.csv`.

- [ ] **Step 5: Verify every loss produced the four required artifacts**

Run in Colab:

```python
from pathlib import Path

root = Path("/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs")
losses = ["mse", "medse", "gmadl", "imadl", "dirhuber", "hybrid_add", "hybrid_mul"]
for loss in losses:
    required = [
        root / f"sanity_metrics_{loss}.csv",
        root / f"sanity_summary_{loss}.json",
        root / f"{loss}_loss_curve.png",
        root / f"{loss}_returns_curve.png",
    ]
    print(loss, all(path.exists() for path in required))
```

Expected: every loss prints `True`.

- [ ] **Step 6: Snapshot the authoritative 7-loss table locally for writing**

Run locally after downloading or syncing the file:

```bash
cp /content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs/all_losses_comparison.csv /Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv
```

Expected: `/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv` exists and becomes the writing copy for the thesis/report.

- [ ] **Step 7: Commit the copied authoritative comparison table if it is intentionally versioned**

Run:

```bash
git add /Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv
git commit -m "results: add phase2 main 24m comparison table"
```

### Task 3: Shortlist Finalists And Define The Robustness Matrix

**Files:**
- Use: `/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv`
- Create: `/Users/roucher/Documents/FYP/doc/phase2_finalists.md`

- [ ] **Step 1: Rank the five directional losses by Phase 2 main-table Sharpe**

Run locally:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv")
rows = list(csv.DictReader(path.open()))
directional = [r for r in rows if r["loss"] not in {"mse", "medse"}]
ranked = sorted(directional, key=lambda r: float(r["long_short_sharpe"]), reverse=True)
for row in ranked:
    print(row["loss"], row["long_short_sharpe"], row["long_short_cumulative_return"])
PY
```

Expected: a descending Sharpe ranking for `gmadl`, `imadl`, `dirhuber`, `hybrid_add`, and `hybrid_mul`.

- [ ] **Step 2: Write the finalist memo**

Create `/Users/roucher/Documents/FYP/doc/phase2_finalists.md` with this structure:

```markdown
# Phase 2 Finalists

## Fixed baselines
- mse
- medse

## Promoted directional finalists
- <top directional loss by Sharpe>
- <second directional loss by Sharpe>

## Rejected directional variants
- <remaining three losses with one-line reason each>

## Promotion rule
- Ranked by Phase 2 main 24-month `long_short_sharpe`
- Checked sign of `long_short_cumulative_return`
- Rejected any candidate with clearly unstable return profile despite acceptable error metrics
```

- [ ] **Step 3: Lock the robustness matrix before rerunning anything**

Write this exact experiment matrix into the same file:

```markdown
## Robustness Matrix

Seeds:
- 42
- 52
- 62

Weight-cap settings:
- max_weight = 0.05
- max_weight = None

Models to run:
- mse
- medse
- finalist_1
- finalist_2
```

- [ ] **Step 4: Commit the finalist-selection memo**

Run:

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_finalists.md
git commit -m "doc: define phase2 finalists and robustness matrix"
```

### Task 4: Run Robustness Checks On Baselines Plus Finalists

**Files:**
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- Use: `/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- Create: `/content/drive/MyDrive/FYP/outputs/phase2_robustness/...`
- Create: `/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv`

- [ ] **Step 1: Run the seed robustness batch with `max_weight=0.05`**

For each selected loss and each seed in `42, 52, 62`, run the matching single-loss runner in Colab using isolated directories. Template:

```python
LOSS = "imadl"
SEED = 52

!python run_sanity_check_{LOSS}.py \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed{SEED}_cap005/{LOSS} \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed{SEED}_cap005/checkpoints/{LOSS} \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --seed {SEED} \
    --max-weight 0.05 \
    --resume-mode auto
```

Expected: one summary JSON per `(loss, seed)` pair.

- [ ] **Step 2: Run the no-cap comparison at seed 42**

For each selected loss, run:

```python
LOSS = "imadl"

!python run_sanity_check_{LOSS}.py \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap/{LOSS} \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap/checkpoints/{LOSS} \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --seed 42 \
    --max-weight None \
    --resume-mode auto
```

Expected: one summary JSON per selected loss under the no-cap portfolio setting.

- [ ] **Step 3: Aggregate the robustness summaries into one CSV**

Run locally after syncing summary JSON files:

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

roots = [
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed42_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed52_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed62_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed42_nocap"),
]
rows = []
for root in roots:
    for path in root.rglob("sanity_summary_*.json"):
        payload = json.loads(path.read_text())
        rows.append({
            "scenario": root.name,
            "loss": payload["loss"],
            "avg_mse": payload["avg_mse"],
            "avg_medse": payload["avg_medse"],
            "avg_r2": payload["avg_r2"],
            "avg_directional_accuracy": payload["avg_directional_accuracy"],
            "long_short_cumulative_return": payload["long_short_cumulative_return"],
            "long_short_sharpe": payload["long_short_sharpe"],
        })

out = Path("/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

Expected: `/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv` exists with one row per `(scenario, loss)` pair.

- [ ] **Step 4: Commit the robustness table**

Run:

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv
git commit -m "results: add phase2 robustness comparison"
```

### Task 5: Produce Paper-Ready Tables And Figures

**Files:**
- Use: `/Users/roucher/Documents/FYP/plot_cumulative_returns.py`
- Use: `/Users/roucher/Documents/FYP/plot_loss_strategy_cumreturn.py`
- Use: `/Users/roucher/Documents/FYP/plot_signalweighted_sharpe.py`
- Use: `/Users/roucher/Documents/FYP/plot_strategy_long_short.py`
- Use: `/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv`
- Use: `/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv`
- Create: `/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md`

- [ ] **Step 1: Export the four core paper tables**

Create `/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md` with these headings:

```markdown
# Phase 2 Results For Writing

## Table 1: 7-loss main 24-month comparison

## Table 2: Directional-loss ranking

## Table 3: Finalist robustness across seeds

## Table 4: Weight-cap sensitivity at seed 42
```

- [ ] **Step 2: Generate the cumulative-return and long-short figures from the authoritative Phase 2 outputs**

Run the existing plotting scripts against the authoritative Phase 2 output root. If a script needs path edits, patch it first and keep its input root fixed to:

```text
/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs
```

Expected: one figure each for cumulative return, long-short return, Sharpe comparison, and strategy comparison.

- [ ] **Step 3: Write the conclusion bullets directly under the tables**

Append this structure to `/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md`:

```markdown
## Conclusion Bullets

- Best overall loss by main-table Sharpe:
- Best directional-accuracy loss:
- Most stable finalist across seeds:
- Effect of removing weight cap:
- Final thesis recommendation:
```

- [ ] **Step 4: Commit the paper-writing bundle**

Run:

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md
git commit -m "doc: add phase2 writing bundle"
```

## Self-Review

- Spec coverage: the plan covers documentation repair, baseline completion, authoritative 7-loss rerun, finalist selection, robustness runs, and final paper artifacts.
- Placeholder scan: every task has exact file paths and exact commands. The only dynamic slots are `finalist_1` and `finalist_2`, which are explicitly defined by Task 3 rather than left ambiguous.
- Type consistency: the same seven canonical loss names are used throughout, the real config source is consistently `best_hyperparameters.txt`, and all final writing artifacts live under `/Users/roucher/Documents/FYP/doc/`.
