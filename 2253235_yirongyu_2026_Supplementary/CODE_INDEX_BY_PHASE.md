# Code Index by Phase

本文件把 PPT 中的 Phase 1–Phase 4 与论文 Chapter 5 §5.2–§5.6 的实证结果，逐项映射到仓库内对应的代码、运行入口、CLI 参数和证据 CSV 路径。所有 Phase ↔ 章节 ↔ 表格的对应关系以下表为准。

## 一、分支总览

| 分支 | HEAD | 角色 | 覆盖的 Phase |
|---|---|---|---|
| **`main`** | `0ed0cc0` | 最终交付分支：代码 + 论文 LaTeX + deck + Phase 3/4 证据镜像 + Phase 3/4 Colab 复现路径 | Phase 1 / Phase 2 代码<br>Phase 3a / Phase 3b / Phase 4 证据 + Colab 复现路径 |
| **`phase3-4`** | `2fd9ae8` 起 | Phase 3/4 代码原产地：多 seed runner、归一化探针、Phase 4 诊断脚本 | Phase 3a / Phase 3b / Phase 4 代码 + 原始证据 |
| `phase2-fixes` | `a921a95` | `phase3-4` 的前驱（Phase 3 完成、Phase 4 未起） | Phase 3a / Phase 3b 代码 |
| `phase2/loss-combinations` | `84ecabe` | Phase 3b α/β/λ runner 起点（codegen 自 `generate_phase2_runners.py`） | Phase 3b 早期 |
| `codex/phase15-colab-drive` | `eb26783` | Phase 2 Colab 早期探索 | Phase 2 探索 |
| `codex/hybrid-lambda-sweep` | `3d921cd` | Phase 2 hybrid λ sweep runner 雏形 | Phase 2 前奏 |
| `codex/colab-repo-cleanup` | `b872bab` | Phase 1 Colab 流水线基础设施 | Phase 1 工具链 |
| `script-revision-v2` | `164fc0c` | 已合入 main（5 behind / 0 ahead） | — |

> **分支命名说明**：`phase3-4` 由原分支 `phase2.2-fix` 重命名而来。仓库内若干目录路径仍保留旧名（例如 `doc/phase2-fix/phase2.2-fix/` 与 main 上的镜像目录 `doc/phase2-fix/phase2.2-fix1/`），为历史路径，未一并重命名以保持论文 Appendix B 中既有引用的一致性。

`main` 与 `phase3-4` 是阅读本仓库实证证据所需的两条分支；其余分支为开发过程快照。

---

## 二、各 Phase 代码与证据明细

### 通用脚本包头模板

所有 `run_sanity_check_<loss_id>.py` 都是 thin wrapper，区别仅在 `build_arg_parser` 的描述字符串与 `run_sanity_check` 调用的 loss id。共享 pipeline 实现位于 `sanity_check_signal_tilted.py`。

**Phase 1 / Phase 2 入口脚本**（在 `main` 上）：

```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Sanity check (<LOSS NAME>)")
    args = parser.parse_args()
    run_sanity_check("<loss_id>", args)


if __name__ == "__main__":
    main()
```

**Phase 3a / Phase 3b / Phase 4 入口脚本**（在 `phase3-4` 上，描述字符串换成 `Phase 2 sanity check`，并把 `parser.parse_args()` 直接内联）：

```python
from sanity_check_signal_tilted import build_arg_parser, run_sanity_check


def main() -> None:
    parser = build_arg_parser("Phase 2 sanity check (<loss_id>)")
    run_sanity_check("<loss_id>", parser.parse_args())


if __name__ == "__main__":
    main()
```

**编排器（多 loss × 多 seed 批跑）模块包头**：

```python
# run_phase2_robustness.py
"""
Phase 2 robustness batch runner.

Runs sanity-check scripts with stable output/checkpoint/log paths so local and
Colab executions can be resumed safely.
"""

# run_phase2_gamma_refinement.py
"""Focused Phase 2.2 gamma refinement runner around robust M2 winners."""
# 内部通过 from run_phase2_robustness import run_batch 调用

# run_loss_scale_diagnostics.py
"""Run representative P0.2 sanity checks and analyze available scale outputs."""
# 跑代表性 loss 后调用 analyze_loss_scales.main 聚合

# analyze_loss_scales.py
"""
Analyze loss-scale diagnostics from available sanity outputs.

Per-batch component CSVs are preferred. If they are absent, the script falls
back to a metrics proxy from sanity_metrics_*.csv so P0 diagnostics still
produce a CSV without modifying the training loop.
"""
```

### 2.1 Phase 1 · 7 条 baseline（单 seed = 42，24 个月静态测试窗口）

> 论文 Ch.5 §5.2 *Phase 1: Baseline loss comparison* — Table 5.1 + Fig 5.1。Deck Slide 11。

| Loss ID | 入口脚本 | 说明 |
|---|---|---|
| `mse` | `run_sanity_check_mse.py` | 标准均方误差，回归默认基线（论文 §3.3.1, §A.1） |
| `medse` | `run_sanity_check_medse.py` | 中位平方误差，鲁棒回归基线（§3.3.1, §A.1） |
| `madl` | `run_sanity_check_madl.py` | Mean Absolute Directional Loss，Michańków et al. 2024（§3.3.2, §A.2） |
| `gmadl` | `run_sanity_check_gmadl.py` | Generalised MADL，引入幂权重（§3.3.2, §A.2） |
| `imadl` | `run_sanity_check_imadl.py` | IMADL（rebalanced），加方向再加权（§3.3.2, §A.2） |
| `hybrid_mul_m1` | `run_sanity_check_hybrid_mul_m1.py` | 乘性 hybrid，λ_dir = 2.0（§3.3.3, §A.3） |
| `hybrid_mul_m2` | `run_sanity_check_hybrid_mul_m2.py` | 乘性 hybrid，λ_dir = 5.0；Phase 3 γ refinement 的起点（§3.3.3, §A.3） |

**所在分支**：`main`（仓库根目录与 `2253235_yirongyu_2026_Supplementary/code/` 双份同步）
**CLI 模板**：`--data-dir <repo-root> --pattern '*.csv' --train-start 1990-01 --train-end 1994-12 --test-start 1995-01 --test-months 24 --max-epochs 20 --batch-size 1024 --seed 42`
**证据路径**：
- `doc/final_report_24m_baselines/results/{mse,medse}/sanity_metrics_*.csv`
- `doc/final_report_all_24m_evidence/results/baseline/<loss>/sanity_metrics_*.csv`
- 汇总：`doc/final_report_all_24m_evidence/reports/baseline_24m_summary.csv`

### 2.2 Phase 2 · 9 条 hybrid A/M variant（单 seed = 42）

> 论文 Ch.5 §5.3 *Phase 2: Hybrid A/M variant sweep* — Table 5.2 + Fig 5.2。Deck Slide 12。

| Variant | Loss ID | 入口脚本 | 超参 | 说明 |
|---|---|---|---|---|
| A1 | `hybrid_add_a1` | `run_sanity_check_hybrid_add_a1.py` | λ_dir = 5.0, λ_hub = 1.0 | 加性 hybrid 起点 |
| A2 | `hybrid_add_a2` | `run_sanity_check_hybrid_add_a2.py` | λ_dir = 10.0, λ_hub = 1.0 | 加性 hybrid，更强方向权重 |
| A3 | `hybrid_add_a3` | `run_sanity_check_hybrid_add_a3.py` | λ_dir = 1.0, λ_hub = 0.1 | 加性 hybrid，单 seed 表现最好的一行 |
| A4 | `hybrid_add_a4` | `run_sanity_check_hybrid_add_a4.py` | λ_dir = 5.0, λ_hub = 0.1 | 加性 hybrid，磁力较弱的 Huber backbone |
| A5 | `hybrid_add_a5` | `run_sanity_check_hybrid_add_a5.py` | λ_dir = 10.0, λ_hub = 0.1 | 加性 hybrid，Huber 磁力极弱 |
| M1 | `hybrid_mul_m1` | `run_sanity_check_hybrid_mul_m1.py` | λ_dir = 2.0 | 乘性 hybrid 在 Phase 1/2 中的最佳行 |
| M2 | `hybrid_mul_m2` | `run_sanity_check_hybrid_mul_m2.py` | λ_dir = 5.0 | 乘性 hybrid，γ refinement 在 Phase 3a 的扩展起点 |
| M3 | `hybrid_mul_m3` | `run_sanity_check_hybrid_mul_m3.py` | λ_dir = 0.5 | 乘性 hybrid，方向权重过弱 |
| M4 | `hybrid_mul_m4` | `run_sanity_check_hybrid_mul_m4.py` | λ_dir = 0.1 | 乘性 hybrid，方向几乎不起作用 |

**所在分支**：`main`（与 Phase 1 同目录、同 CLI 模板）
**证据路径**：
- 单行：`doc/final_report_all_24m_evidence/results/phase15/<loss>/sanity_metrics_*.csv`
- 汇总：`doc/final_report_all_24m_evidence/reports/phase15_24m_summary.csv`

### 2.3 Phase 3a · γ refinement（每行 3 seed = {42, 52, 62}）

> 论文 Ch.5 §5.4 *Phase 3a: Multi-seed γ refinement* — Table 5.3 + Fig 5.3, 5.4。Deck Slide 13。
> Loss 形式：`L = L_M2 + γ · Var(ŷ)`，扫描 5 个 γ 值。

| Loss ID | 入口脚本 | γ 值 | 说明 |
|---|---|---|---|
| `m2_robust_gamma03` | `run_sanity_check_m2_robust_gamma03.py` | 0.3 | 方差正则不足，CV 1.057（最不稳定） |
| `m2_robust_gamma05` | `run_sanity_check_m2_robust_gamma05.py` | 0.5 | 接近稳定区下沿 |
| `m2_robust_gamma07` | `run_sanity_check_m2_robust_gamma07.py` | 0.7 | Phase 3a 推荐：mean Sharpe 0.9156，CV 0.1808（同时最大化 Sharpe 与最小化 CV） |
| `m2_robust_gamma10` | `run_sanity_check_m2_robust_gamma10.py` | 1.0 | mean Sharpe 1.0043（最高），但 CV 0.5613（seed 散布大） |
| `m2_robust_gamma15` | `run_sanity_check_m2_robust_gamma15.py` | 1.5 | 过度压缩信号，性能与稳定性双降 |

**批量编排**：`run_phase2_gamma_refinement.py`（封装 5 × 3 = 15 次实验，调用 `run_phase2_robustness.run_batch`）。
**所在分支**：`phase3-4`（代码唯一原产地，main 上不含这些 runner；如需运行需切到 `phase3-4` 分支）。
**CLI 模板**：`--data-dir <repo-root> --pattern '*.csv' --seeds 42,52,62 --caps 0.05 --train-start 1990-01 --train-end 1994-12 --test-start 1995-01 --test-months 24 --max-epochs 20 --batch-size 1024`
**证据路径**：
- 原产地：`phase3-4:doc/phase2-fix/reports/phase2_grouped_summary.csv` + `phase2_raw_runs.csv`（与 Phase 3b 整合在一份 CSV 内）
- main 镜像（已拆分）：`main:doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv` + `phase2_raw_runs.csv`

### 2.4 Phase 3b · 整合 α / β / λ + 细 γ（每行 3 seed）

> 论文 Ch.5 §5.5 *Phase 3b: Integrated α, β, λ sweeps* — Table 5.4 + Fig 5.5, 5.6。Deck Slide 14。

#### 2.4.1 IMADL-M2 α 扫描

| Loss ID | 入口脚本 | α 值 | 说明 |
|---|---|---|---|
| `imadl_m2_alpha02` | `run_sanity_check_imadl_m2_alpha02.py` | 0.2 | IMADL 与 M2 线性组合的最低 α |
| `imadl_m2_alpha03` | `run_sanity_check_imadl_m2_alpha03.py` | 0.3 | |
| `imadl_m2_alpha04` | `run_sanity_check_imadl_m2_alpha04.py` | 0.4 | CV 0.1853（最稳定但 mean Sharpe 不高） |
| `imadl_m2_alpha05` | `run_sanity_check_imadl_m2_alpha05.py` | 0.5 | |
| `imadl_m2_alpha06` | `run_sanity_check_imadl_m2_alpha06.py` | 0.6 | Phase 3b α 峰值：mean Sharpe 0.6895，cum +30.42%（fallback 推荐） |
| `imadl_m2_alpha07` | `run_sanity_check_imadl_m2_alpha07.py` | 0.7 | 越过峰值开始衰减 |
| `imadl_m2_alpha08` | `run_sanity_check_imadl_m2_alpha08.py` | 0.8 | |

#### 2.4.2 IMADL-GMADL β 扫描

| Loss ID | 入口脚本 | β 值 | 说明 |
|---|---|---|---|
| `imadl_gmadl_beta03` | `run_sanity_check_imadl_gmadl_beta03.py` | 0.3 | β 家族最佳行（仍 < gamma07） |
| `imadl_gmadl_beta05` | `run_sanity_check_imadl_gmadl_beta05.py` | 0.5 | CV 10.13（坍塌） |
| `imadl_gmadl_beta07` | `run_sanity_check_imadl_gmadl_beta07.py` | 0.7 | CV 139.51（彻底坍塌，论文标记为 weak evidence） |

#### 2.4.3 Adaptive λ 扫描

| Loss ID | 入口脚本 | λ 值 | 说明 |
|---|---|---|---|
| `adaptive_lambda10` | `run_sanity_check_adaptive_lambda10.py` | 1.0 | 自适应方向权重，三档中表现最好 |
| `adaptive_lambda50` | `run_sanity_check_adaptive_lambda50.py` | 5.0 | |
| `adaptive_lambda100` | `run_sanity_check_adaptive_lambda100.py` | 10.0 | |

#### 2.4.4 细 γ 补充（论文 Table 5.4 末两行）

| Loss ID | 入口脚本 | γ 值 | 说明 |
|---|---|---|---|
| `m2_robust_gamma001` | `run_sanity_check_m2_robust_gamma001.py` | 0.01 | γ → 0 极限 |
| `m2_robust_gamma01` | `run_sanity_check_m2_robust_gamma01.py` | 0.1 | 比 0.3 更弱的方差正则 |

**批量编排**：`run_phase2_robustness.py`（默认 16 个 loss × 3 seed = 48 次实验）
**所在分支**：`phase3-4`（α/β/λ runner 早期版本见 `phase2/loss-combinations:generate_phase2_runners.py`）
**证据路径**：与 Phase 3a 共享 `phase2_grouped_summary.csv` / `phase2_raw_runs.csv`

### 2.5 Phase 4 · loss-component normalisation probe（诊断性，3 seed）

> 论文 Ch.5 §5.6 *Phase 4: Loss-component normalisation probe* — Table 5.5 + Fig 5.7。Deck Slide 15。
> 把 directional / magnitude 两个组件按 batch 内尺度比例归一化后重跑，检查领头候选是否依赖 scale artefact。

| Loss ID | 入口脚本 | 对应原 loss | 探针结果 |
|---|---|---|---|
| `m2_robust_gamma07_normalized` | `run_sanity_check_m2_robust_gamma07_normalized.py` | `m2_robust_gamma07` | 0.9156 → 0.9112，通过（不是 scale artefact） |
| `m2_robust_gamma10_normalized` | `run_sanity_check_m2_robust_gamma10_normalized.py` | `m2_robust_gamma10` | 1.0043 → 0.4072，显著退化（scale-sensitive） |
| `imadl_m2_alpha06_normalized` | `run_sanity_check_imadl_m2_alpha06_normalized.py` | `imadl_m2_alpha06` | 0.6895 → −0.0161，坍塌（fallback 不通过探针） |

**配套诊断**：
- `run_loss_scale_diagnostics.py` — 跑代表性 loss 并统计 batch 内组件尺度比
- `analyze_loss_scales.py` — 把上一步输出聚合为 `loss_scale_*.csv`
- `notebooks/phase2_loss_component_analysis.ipynb` — 交互式分析与作图

**所在分支**：`phase3-4`（代码唯一原产地）
**证据路径**：
- 原产地：`phase3-4:doc/phase2-fix/phase2.2-fix/{phase1_summary.json, phase2_summary.json, LOSS_COMPONENT_ANALYSIS_RESULTS.md}`
- main 镜像：`main:doc/phase2-fix/phase2.2-fix1/{phase1_summary.json, phase2_summary.json}`
- 注：目录名 `phase2.2-fix` 与 `phase2.2-fix1` 是历史路径（沿用原分支名 `phase2.2-fix` 的命名），未随分支重命名一并迁移；论文 Appendix B §B.2 依此引用 main 上的 `phase2.2-fix1` 路径。

---

## 三、Colab 复现路径目录

`2253235_yirongyu_2026_Supplementary/colab_runs/` 收录 Phase 1–Phase 4 在 Google Colab 上的实际跑批 notebook 与操作手册（原样自仓库内拷入，未做格式转换）。该目录的目的为留档复现路径与操作历史；批改人无需在 Colab 重跑。Notebook 内部出现的 `/content/drive/MyDrive/...` 等 Drive 挂载路径为开发过程中的个人挂载点，重跑时替换为本地或自有 Drive 路径即可。

| 文件 | 对应 Phase | 跑了什么 | 来源（branch:path） |
|---|---|---|---|
| `final_report_24m_baselines_colab.ipynb` | Phase 1（一部分） | mse + medse 两条 baseline 的 Colab 跑批 | `main:doc/final_report_24m_baselines_colab.ipynb` |
| `final_report_24m_baselines_colab.md` | Phase 1（操作手册） | 上面 notebook 的逐步说明 | `main:doc/final_report_24m_baselines_colab.md` |
| `final_report_all_24m_evidence_colab.ipynb` | Phase 1 + Phase 2 | 7 baseline + 9 hybrid = 16 条 single-seed 跑批 | `main:doc/final_report_all_24m_evidence_colab.ipynb` |
| `final_report_all_24m_evidence_colab.md` | Phase 1 + Phase 2（操作手册） | 上面 notebook 的逐步说明 | `main:doc/final_report_all_24m_evidence_colab.md` |
| `Phase2_2_Experiment_Runner.ipynb` | Phase 3a | 5 γ 值 × 3 seed = 15 次 γ refinement 实验的 Colab 编排 | `phase3-4:Phase2_2_Experiment_Runner.ipynb` |
| `Phase2_Fixes_Colab_Runner.ipynb` | Phase 3b + Phase 4 | α/β/λ + 细 γ + 3 normalised 在 Colab 上的统一跑批 | `phase3-4:Phase2_Fixes_Colab_Runner.ipynb` |
| `phase2_loss_component_analysis.ipynb` | Phase 4 | normalisation probe 的交互式分析与作图 | `phase3-4:notebooks/phase2_loss_component_analysis.ipynb` |
| `COLAB_USAGE.md` | 通用操作 | Colab 端挂载 Drive、resume、batch 跑批的通用文档 | `phase3-4:COLAB_USAGE.md` |
| `COLAB_PHASE2_P0_GUIDE.md` | Phase 3/4 操作 | Phase 3/4 跑批在 Colab 上的逐步操作指南 | `phase3-4:COLAB_PHASE2_P0_GUIDE.md` |

shell 跑批入口位于 `2253235_yirongyu_2026_Supplementary/code/scripts/`：`run_final_report_24m_baselines_colab.sh` 与 `run_final_report_all_24m_evidence_colab.sh` 覆盖 Phase 1 + Phase 2；Phase 3/4 跑批由 notebook 内嵌调用 `run_phase2_robustness.py` / `run_phase2_gamma_refinement.py` 完成。

---

## 四、命名与路径备注

- **分支**：`phase3-4` 由 `phase2.2-fix` 重命名而来；后者旧名仅为历史引用，仓库内涉及该旧名的几个目录（`doc/phase2-fix/phase2.2-fix/` 与 `doc/phase2-fix/phase2.2-fix1/`）保留为历史路径。
- **目录名差异**：`phase2.2-fix1` 是 main 上对原 phase2.2-fix 分支 Phase 4 产物的镜像目录名；`phase2.2-fix`（无 1）是 phase3-4 分支上的同源原始目录名。论文 Appendix B 与 §3.7 中 `phase2.2-fix` / `phase2.2-fix1` 字样均指上述目录或原分支名，与目前的 `phase3-4` 分支等价。

---

*最后更新：2026-05-22*
