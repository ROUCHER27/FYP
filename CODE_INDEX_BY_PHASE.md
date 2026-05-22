# Code Index by Phase

> 把 PPT 上的 Phase 1–4 与论文 Chapter 5 §5.2–§5.6 的实证结果，逐项对应到仓库里的代码与运行入口。给老师交付时只看本文件即可定位每条 loss 的 runner 与说明。

## 一、分支总览

| 分支 | HEAD | 角色 | 覆盖的 Phase |
|---|---|---|---|
| **`main`** | `0ed0cc0` | **最终交付分支**：单一来源，包含全部代码 + 论文 LaTeX + deck + Phase 3/4 证据镜像 + Phase 3/4 Colab 复现路径文档 | Phase 1 / Phase 2 代码<br>Phase 3a / Phase 3b / Phase 4 证据 + Colab 复现路径 |
| **`phase2.2-fix`** | `2fd9ae8` | **Phase 3/4 代码原产地**：多 seed runner、归一化探针、Phase 4 诊断 | Phase 3a / Phase 3b / Phase 4 代码 + 原始证据 |
| `phase2-fixes` | `a921a95` | `phase2.2-fix` 的前驱（Phase 3 完成、Phase 4 未起） | Phase 3a / Phase 3b 代码 |
| `phase2/loss-combinations` | `84ecabe` | Phase 3b α/β/λ runner 起点（codegen 自 `generate_phase2_runners.py`） | Phase 3b 早期 |
| `codex/phase15-colab-drive` | `eb26783` | Phase 2 Colab 早期探索（旧名 "Phase 1.5"） | Phase 2 探索 |
| `codex/hybrid-lambda-sweep` | `3d921cd` | Phase 2 hybrid λ sweep runner 雏形 | Phase 2 前奏 |
| `codex/colab-repo-cleanup` | `b872bab` | Phase 1 Colab 流水线基础设施 | Phase 1 工具链 |
| `script-revision-v2` | `164fc0c` | 已合入 main（5 behind / 0 ahead），可删 | — |

> 给老师的口径：只看 `main` 即可；`phase2.2-fix` 作为多 seed 实验的"原产地存证"保留。其余 6 条都是开发过程快照，建议交付前归档（命名为 `archive/<原名>`）或删除。

---

## 二、各 Phase 代码与证据明细

### 通用脚本包头模板

所有 `run_sanity_check_<loss_id>.py` 都是同一个 thin wrapper，只换两个参数：`build_arg_parser` 的描述字符串与 `run_sanity_check` 的 loss id。共享 pipeline 在 `sanity_check_signal_tilted.py` 中实现。

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

**Phase 3a / Phase 3b / Phase 4 入口脚本**（在 `phase2.2-fix` 上，描述字符串换成 `Phase 2 sanity check`，并把 `parser.parse_args()` 直接内联）：

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
**所在分支**：`phase2.2-fix`（**code 唯一原产地，main 上没有这些 runner**）。
**CLI 模板**：`--data-dir <repo-root> --pattern '*.csv' --seeds 42,52,62 --caps 0.05 --train-start 1990-01 --train-end 1994-12 --test-start 1995-01 --test-months 24 --max-epochs 20 --batch-size 1024`
**证据路径**：
- 原产地：`phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv` + `phase2_raw_runs.csv`（与 Phase 3b 整合在一份 CSV 内）
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
**所在分支**：`phase2.2-fix`（α/β/λ runner 早期版本见 `phase2/loss-combinations:generate_phase2_runners.py`）
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

**所在分支**：`phase2.2-fix`（**唯一原产地**）
**证据路径**：
- 原产地：`phase2.2-fix:doc/phase2-fix/phase2.2-fix/{phase1_summary.json, phase2_summary.json, LOSS_COMPONENT_ANALYSIS_RESULTS.md}`
- main 镜像：`main:doc/phase2-fix/phase2.2-fix1/{phase1_summary.json, phase2_summary.json}`
- ⚠️ 目录名差一个 "1"：main 镜像是 `phase2.2-fix1`，phase2.2-fix 分支上是 `phase2.2-fix`。论文 Appendix B §B.2 写的 `phase2.2-fix1` 路径只在 `main` 上存在。

---

## 三、Colab 复现路径文档（统一收到 main）

按你的决定：**Phase 3 / Phase 4 在 `phase2.2-fix` 上的 Colab 复现产物，连同 Phase 1 / Phase 2 的 Colab 资产，一并保存在 `main` 的交付包里**，让老师 clone main 就能拿到完整的复现路径。

### 3.1 计划布局（所有路径都在 `main` 上）

```
2253235_yirongyu_2026_Supplementary/
├── code/                            # 已存在：Phase 1/2 单 loss runner + sanity_check_signal_tilted.py
├── colab_runs/                      # 新增目录：Colab 复现路径文档
│   ├── README.md                    # 索引：每个 notebook 跑了哪个 Phase / 多少 loss / 多少 seed
│   ├── phase1_2_baselines.html      # 由 doc/final_report_24m_baselines_colab.ipynb nbconvert
│   ├── phase1_2_all_evidence.html   # 由 doc/final_report_all_24m_evidence_colab.ipynb nbconvert
│   ├── phase3_2_2_runner.html       # 由 phase2.2-fix:Phase2_2_Experiment_Runner.ipynb nbconvert
│   ├── phase3_4_fixes_runner.html   # 由 phase2.2-fix:Phase2_Fixes_Colab_Runner.ipynb nbconvert
│   ├── phase4_loss_component.html   # 由 phase2.2-fix:notebooks/phase2_loss_component_analysis.ipynb nbconvert
│   ├── COLAB_USAGE.md               # 从 phase2.2-fix 拷贝并清理 Drive 私人路径
│   └── COLAB_PHASE2_P0_GUIDE.md     # 从 phase2.2-fix 拷贝并清理 Drive 私人路径
└── ...
```

### 3.2 收编步骤（执行时附加到 PR）

1. **从 `phase2.2-fix` 拉 5 个 notebook + 2 个 md**：
   - `Phase2_2_Experiment_Runner.ipynb`
   - `Phase2_Fixes_Colab_Runner.ipynb`
   - `notebooks/phase2_loss_component_analysis.ipynb`
   - `COLAB_USAGE.md`
   - `COLAB_PHASE2_P0_GUIDE.md`
2. **对 5 个 notebook 跑 `jupyter nbconvert --to html --no-input`**，得到带输出 cell（保留）但不暴露源码 cell（避免 Drive token 类敏感信息泄露）的 HTML 静态页。
3. **对 2 个 md 做 sed 清理**：
   - `s|/content/drive/MyDrive/[^[:space:]]*|<DRIVE_PATH>|g` 替换 Drive 私人路径
   - 删除任何形如 `https://drive.google.com/file/d/...` 的私人共享链接
4. **新增 `2253235_yirongyu_2026_Supplementary/colab_runs/README.md`**，按下表填写 5 个 HTML 的对应关系：

| 文件 | 对应 Phase | 跑了什么 | 来源 notebook（branch:path） |
|---|---|---|---|
| `phase1_2_baselines.html` | Phase 1 | mse + medse 两条 baseline 的 Colab 跑批 | `main:doc/final_report_24m_baselines_colab.ipynb` |
| `phase1_2_all_evidence.html` | Phase 1 + Phase 2 | 7 baseline + 9 hybrid = 16 条 single-seed 跑批 | `main:doc/final_report_all_24m_evidence_colab.ipynb` |
| `phase3_2_2_runner.html` | Phase 3a | 5 γ 值 × 3 seed = 15 次实验的 Colab 编排 | `phase2.2-fix:Phase2_2_Experiment_Runner.ipynb` |
| `phase3_4_fixes_runner.html` | Phase 3b + Phase 4 | α/β/λ + 细 γ + 3 normalised 的统一 Colab 编排 | `phase2.2-fix:Phase2_Fixes_Colab_Runner.ipynb` |
| `phase4_loss_component.html` | Phase 4 | normalisation probe 的交互式分析 + 作图 | `phase2.2-fix:notebooks/phase2_loss_component_analysis.ipynb` |

5. **`scripts/colab_backup.sh`** 不进交付包（个人 Drive 备份脚本）。
6. **探索性 notebook**（`Feature_Pipeline_Check.ipynb` / `G:MADL/*.ipynb` / `test/Feature_Pipeline_CheckX*.ipynb`）不进交付包，已被论文 §3.3 / §3.4 / §4 文字归纳。

---

## 四、需要顺手做的清理：删除注释里的 "中文：" 前缀（共 13 行）

**只删字面量 "中文："**（含全角冒号），保留后面的中文句子。涉及 6 份文件，全部是 `Model_Train/data_preprocess.py` 与 `Model_Train/train_rolling.py` 在不同位置的镜像。

| 分支 | 文件 | 命中行号 |
|---|---|---|
| main | `Model_Train/data_preprocess.py` | 19, 46, 60, 80, 103, 125, 146（7 行） |
| main | `Model_Train/train_rolling.py` | 31, 42, 95, 117, 142, 166（6 行） |
| main | `2253235_yirongyu_2026_Supplementary/code/Model_Train/data_preprocess.py` | 同上 7 行（与顶层 md5 一致） |
| main | `2253235_yirongyu_2026_Supplementary/code/Model_Train/train_rolling.py` | 同上 6 行（与顶层 md5 一致） |
| phase2.2-fix | `Model_Train/data_preprocess.py` | 19, 46, 59, 77, 100, 122, 143（7 行） |
| phase2.2-fix | `Model_Train/train_rolling.py` | 同 main（md5 一致） |

**改法示例**（`data_preprocess.py:19`）：

```diff
     """
     Load and vertically concatenate raw CSV files.
-    中文：批量读取目录中的原始 CSV 并纵向合并。
+    批量读取目录中的原始 CSV 并纵向合并。

     Parameters
     ----------
```

`losses.py` 等文件里的中文注释（如"对逐元素损失按照 mean/sum/none 指令做标准化聚合"）没有"中文："前缀，**不在本次清理范围内**。

---

*最后更新：2026-05-22*
