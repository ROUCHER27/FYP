# Colab Runs · 复现路径文档

本目录收录了 Phase 1 / Phase 2 / Phase 3 / Phase 4 全部实证证据在 Google Colab 上的实际跑批 notebook 与操作手册，原样自仓库（含跨分支）拷入。给老师的目的是：**不用切分支、不用重跑，就能看到当时是怎么把每一行结果跑出来的**。

> 本目录里的脚本不是为了让批改人重跑（一次完整 sweep 在 Colab T4 上要 2~7 小时），而是作为复现路径与操作历史的留档。每条 loss 的跑批入口在 `../code/`，shell 跑批脚本在 `../code/scripts/`。

## 文件总览

| 文件 | 对应 Phase | 跑了什么 | 来源 (branch:path) |
|---|---|---|---|
| `final_report_24m_baselines_colab.ipynb` | Phase 1（一部分） | mse + medse 两条 baseline 的 Colab 跑批 | `main:doc/final_report_24m_baselines_colab.ipynb` |
| `final_report_24m_baselines_colab.md` | Phase 1（操作手册） | 上面 notebook 的逐步说明 | `main:doc/final_report_24m_baselines_colab.md` |
| `final_report_all_24m_evidence_colab.ipynb` | Phase 1 + Phase 2 | 7 baseline + 9 hybrid = 16 条 single-seed 跑批 | `main:doc/final_report_all_24m_evidence_colab.ipynb` |
| `final_report_all_24m_evidence_colab.md` | Phase 1 + Phase 2（操作手册） | 上面 notebook 的逐步说明 | `main:doc/final_report_all_24m_evidence_colab.md` |
| `Phase2_2_Experiment_Runner.ipynb` | Phase 3a | 5 个 γ 值 × 3 seed = 15 次 γ refinement 实验的 Colab 编排 | `phase2.2-fix:Phase2_2_Experiment_Runner.ipynb` |
| `Phase2_Fixes_Colab_Runner.ipynb` | Phase 3b + Phase 4 | α/β/λ + 细 γ + 3 条 normalised 在 Colab 上的统一跑批 | `phase2.2-fix:Phase2_Fixes_Colab_Runner.ipynb` |
| `phase2_loss_component_analysis.ipynb` | Phase 4 | normalisation probe 的交互式分析 + 作图 | `phase2.2-fix:notebooks/phase2_loss_component_analysis.ipynb` |
| `COLAB_USAGE.md` | 通用操作 | Colab 端挂载 Drive、resume、batch 跑批的通用文档 | `phase2.2-fix:COLAB_USAGE.md` |
| `COLAB_PHASE2_P0_GUIDE.md` | Phase 3/4 操作 | Phase 3/4 跑批在 Colab 上的逐步操作指南 | `phase2.2-fix:COLAB_PHASE2_P0_GUIDE.md` |

## 注意事项

- 这些 notebook 中的 Drive 路径（如 `/content/drive/MyDrive/FYP/...`）与 Drive 共享链接是个人挂载点，批改人不需要、也无法访问它们。如果要在新的 Colab 环境里重跑，**只需替换为自己的 Drive 路径或本地路径**即可，主体代码不变。
- 跑批脚本的 shell 入口在 `../code/scripts/`：`run_final_report_24m_baselines_colab.sh` 与 `run_final_report_all_24m_evidence_colab.sh` 已经随交付包提供，覆盖 Phase 1 + Phase 2。Phase 3/4 的 shell 入口由对应 notebook 内嵌调用 `run_phase2_robustness.py` / `run_phase2_gamma_refinement.py` 完成。
- 完整的 Phase ↔ 代码 ↔ 证据映射见仓库根目录 `CODE_INDEX_BY_PHASE.md`（PR-1 #5）。
