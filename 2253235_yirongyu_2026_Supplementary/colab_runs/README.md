# Colab Runs · 复现路径文档

本目录收录 Phase 1–Phase 4 在 Google Colab 上的实际跑批 notebook 与操作手册，原样自仓库内（跨分支）拷入，目的为留档复现路径与操作历史。本目录的脚本不必重跑（完整 sweep 在 Colab T4 GPU 上耗时约 2–7 小时）；每条 loss 的本地跑批入口在 `../code/`，shell 跑批脚本在 `../code/scripts/`。

## 文件总览

| 文件 | 对应 Phase | 跑了什么 | 来源（branch:path） |
|---|---|---|---|
| `final_report_24m_baselines_colab.ipynb` | Phase 1（一部分） | mse + medse 两条 baseline 的 Colab 跑批 | `main:doc/final_report_24m_baselines_colab.ipynb` |
| `final_report_24m_baselines_colab.md` | Phase 1（操作手册） | 上面 notebook 的逐步说明 | `main:doc/final_report_24m_baselines_colab.md` |
| `final_report_all_24m_evidence_colab.ipynb` | Phase 1 + Phase 2 | 7 baseline + 9 hybrid = 16 条 single-seed 跑批 | `main:doc/final_report_all_24m_evidence_colab.ipynb` |
| `final_report_all_24m_evidence_colab.md` | Phase 1 + Phase 2(操作手册) | 上面 notebook 的逐步说明 | `main:doc/final_report_all_24m_evidence_colab.md` |
| `Phase2_2_Experiment_Runner.ipynb` | Phase 3a | 5 个 γ 值 × 3 seed = 15 次 γ refinement 实验的 Colab 编排 | `phase3-4:Phase2_2_Experiment_Runner.ipynb` |
| `Phase2_Fixes_Colab_Runner.ipynb` | Phase 3b + Phase 4 | α/β/λ + 细 γ + 3 条 normalised 在 Colab 上的统一跑批 | `phase3-4:Phase2_Fixes_Colab_Runner.ipynb` |
| `phase2_loss_component_analysis.ipynb` | Phase 4 | normalisation probe 的交互式分析与作图 | `phase3-4:notebooks/phase2_loss_component_analysis.ipynb` |
| `COLAB_USAGE.md` | 通用操作 | Colab 端挂载 Drive、resume、batch 跑批的通用文档 | `phase3-4:COLAB_USAGE.md` |
| `COLAB_PHASE2_P0_GUIDE.md` | Phase 3/4 操作 | Phase 3/4 跑批在 Colab 上的逐步操作指南 | `phase3-4:COLAB_PHASE2_P0_GUIDE.md` |

## 注意事项

- Notebook 中出现的 Drive 路径（如 `/content/drive/MyDrive/FYP/...`）与 Drive 共享链接为开发过程中的个人挂载点。如需在新的 Colab 环境中重跑，替换为本地或自有 Drive 路径即可，主体代码不变。
- shell 跑批入口：`../code/scripts/run_final_report_24m_baselines_colab.sh` 与 `../code/scripts/run_final_report_all_24m_evidence_colab.sh` 覆盖 Phase 1 + Phase 2；Phase 3/4 跑批由对应 notebook 内嵌调用 `run_phase2_robustness.py` / `run_phase2_gamma_refinement.py` 完成。
- 完整的 Phase ↔ 代码 ↔ 证据映射见 `../CODE_INDEX_BY_PHASE.md`。
