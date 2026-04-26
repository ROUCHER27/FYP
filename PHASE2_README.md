# Phase 2 实验指南

## 概述

Phase 2 实验旨在通过组合 Phase 1.5 中表现最好的损失函数（IMADL 和 M2）来设计新的混合损失函数，以平衡稳定性和高收益。

## 实验设计

### Phase 2.1: 初始筛选
- **目标**: 快速评估 13 个损失函数的基本性能
- **配置**: 13 losses × 3 seeds × 1 cap = **39 runs**
- **测试期**: 24 个月（1995-01 到 1996-12）

### Phase 2.2: 扩展验证
- **目标**: 对 Phase 2.1 中表现最好的 2-3 个损失函数进行深度验证
- **配置**: 3 losses × 6 seeds × 2 caps = **36 runs**
- **测试期**: 48 个月（1995-01 到 1998-12）

## 四种损失函数变体

### Variant 1: IMADL + M2 线性组合（7 个）
```
L = α * IMADL + (1-α) * M2
α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
```

### Variant 2: IMADL + GMADL 加权组合（3 个）
```
L = β * IMADL + (1-β) * GMADL
β ∈ {0.3, 0.5, 0.7}
```

### Variant 3: M2 鲁棒性增强（3 个）
```
L = M2 - γ * std(y_pred)²
γ ∈ {0.01, 0.1, 1.0}
```

### Variant 4: 自适应混合（3 个）
```
L = IMADL * exp(-λ|y|) + M2 * (1 - exp(-λ|y|))
λ ∈ {1.0, 5.0, 10.0}
```

## 文件结构

```
phase2/loss-combinations/
├── Model_Train/
│   └── losses.py                          # 13 个新损失函数
├── run_phase2_robustness.py               # 主运行器
├── run_sanity_check_*.py                  # 13 个单独运行器
├── aggregate_phase2_results.py            # 结果聚合脚本
├── generate_phase2_runners.py             # 运行器生成脚本
├── Phase2_Colab_Runner.ipynb              # Colab notebook
└── PHASE2_README.md                       # 本文档
```

## 本地测试

### 1. 测试单个损失函数

```bash
# 测试 IMADL + M2 (alpha=0.5)
python run_sanity_check_imadl_m2_alpha05.py \
  --data-dir . \
  --train-start 1990-01 \
  --train-end 1994-12 \
  --test-start 1995-01 \
  --test-months 6 \
  --max-epochs 5 \
  --batch-size 512 \
  --seed 42 \
  --output-dir ./test_output \
  --checkpoint-dir ./test_checkpoints
```

### 2. 测试主运行器（小规模）

```bash
# 仅测试 3 个损失函数
python run_phase2_robustness.py \
  --data-dir . \
  --losses imadl_m2_alpha05,imadl_gmadl_beta05,adaptive_lambda50 \
  --seeds 42 \
  --test-months 6 \
  --max-epochs 5 \
  --matrix-mode light \
  --drive-root ./test_drive
```

## Colab 部署

### 1. 上传代码到 GitHub

```bash
# 推送到远程仓库
git push origin phase2/loss-combinations
```

### 2. 准备 Google Drive

在 Google Drive 中创建以下目录结构：

```
/MyDrive/FYP/
├── data/                    # 上传所有 CSV 数据文件
│   ├── 89.12-94.csv
│   ├── 94-99.csv
│   └── ...
└── phase2/                  # 将自动创建
    ├── results/
    ├── checkpoints/
    └── logs/
```

### 3. 在 Colab 中运行

1. 打开 `Phase2_Colab_Runner.ipynb`
2. 上传到 Google Colab
3. 按顺序执行所有 cells
4. 监控执行进度

### 4. 中断恢复

如果 Colab 会话中断：

1. 重新运行 notebook
2. `--skip-existing` 会自动跳过已完成的运行
3. `--resume-mode auto` 会从 checkpoint 恢复未完成的运行

## Phase 2.1 执行

```bash
python run_phase2_robustness.py \
  --drive-root /content/drive/MyDrive/FYP \
  --data-dir /content/drive/MyDrive/FYP/data \
  --train-start 1990-01 \
  --train-end 1994-12 \
  --test-start 1995-01 \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --matrix-mode light \
  --resume-mode auto \
  --skip-existing
```

**预计时间**: 约 6-8 小时（取决于 Colab GPU）

## 结果聚合

```bash
python aggregate_phase2_results.py \
  --drive-root /content/drive/MyDrive/FYP \
  --seeds 42,52,62 \
  --caps 0.05
```

**输出文件**:
- `phase2_raw_runs.csv` - 每次运行的详细指标
- `phase2_grouped_summary.csv` - 按损失函数分组的统计
- `phase2_summary_report.txt` - 文本格式的总结报告

## Phase 2.2 执行

在 Phase 2.1 完成后，选择 Top 3 损失函数：

```bash
# 假设 Top 3 是: imadl_m2_alpha05, imadl_gmadl_beta05, adaptive_lambda50
python run_phase2_robustness.py \
  --drive-root /content/drive/MyDrive/FYP \
  --data-dir /content/drive/MyDrive/FYP/data \
  --losses imadl_m2_alpha05,imadl_gmadl_beta05,adaptive_lambda50 \
  --seeds 42,52,62,72,82,92 \
  --test-months 48 \
  --matrix-mode full \
  --resume-mode auto \
  --skip-existing
```

**预计时间**: 约 8-10 小时

## 成功标准

### Phase 2.1
- 至少 1 个损失函数的平均 Sharpe > 0.6
- 至少 2 个损失函数的 CV < 1.0
- 至少 3 个损失函数优于 IMADL 基线（Sharpe=0.464）

### Phase 2.2
- Top 1 损失函数在 48 个月测试期的 Sharpe > 0.7
- Top 1 损失函数的失败率 < 10%
- Top 1 损失函数在无权重上限时仍保持稳定

## 故障排查

### 问题 1: 找不到运行器脚本

**错误**: `FileNotFoundError: Runner script not found: run_sanity_check_*.py`

**解决**: 确保所有 13 个运行器脚本都已创建：
```bash
python generate_phase2_runners.py
```

### 问题 2: Google Drive 未挂载

**错误**: `Drive not mounted!`

**解决**: 在 Colab 中运行：
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 问题 3: 内存不足

**错误**: `CUDA out of memory`

**解决**: 减小 batch size：
```bash
--batch-size 512  # 或更小
```

### 问题 4: 会话超时

**解决**: 
1. 使用 `--skip-existing` 跳过已完成的运行
2. 使用 `--resume-mode auto` 从 checkpoint 恢复
3. 考虑分批运行（使用 `--losses` 参数）

## 参考基线（Phase 1.5）

| 损失函数 | 平均 Sharpe | 标准差 | CV | 失败率 |
|---------|-----------|--------|-----|--------|
| IMADL   | 0.464     | 0.414  | 0.892 | 0%   |
| M2      | 0.914     | 1.042  | 1.396 | 33%  |
| GMADL   | 0.307     | 0.358  | 1.168 | 0%   |

## 下一步

1. 完成 Phase 2.1 并分析结果
2. 选择 Top 3 损失函数
3. 运行 Phase 2.2 扩展验证
4. 撰写论文结果章节

## 联系方式

如有问题，请查看：
- 计划文档: `.claude/plans/doc-phase2-2-colab-drive-python-replicated-sutherland.md`
- Phase 2 文档: `doc/phase2/` 目录

---

**创建日期**: 2026-04-26  
**分支**: phase2/loss-combinations  
**状态**: 准备部署
