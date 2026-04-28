# Phase 2 修复实验文档

## 实验背景

Phase 2.1 初步实验结果显示严重问题：
- **最佳 Sharpe 仅 0.0399** (imadl_gmadl_beta03)，远低于 Phase 1.5 基线
- **IMADL 基线**: Sharpe 0.464, CV 0.892
- **M2 基线**: Sharpe 0.914, CV 1.396
- **GMADL 基线**: Sharpe 0.307, CV 1.168

## 已完成的修复 (P0 阶段)

### ✅ 修复 1: M2 实现对齐 (P0.3)

**问题**: Phase 2 使用简化版 M2，与 Phase 1.5 不一致

**Phase 2.1 错误实现**:
```python
def m2_loss(y_true, y_pred):
    loss = -torch.sign(y_true) * (y_pred - y_true) ** 2
```

**Phase 2-fixes 正确实现**:
```python
def m2_loss(y_true, y_pred, reduction="mean"):
    """Phase 2 M2 baseline, aligned with Phase 1.5 hybrid_mul M2."""
    return hybrid_dir_huber_mul_loss(
        y_true, y_pred, lambda_dir=2.0, reduction=reduction
    )
```

**影响范围**: 10 个 loss 函数
- Variant 1: 7 个 `imadl_m2_alpha*` 
- Variant 4: 3 个 `adaptive_lambda*`

### ✅ 修复 2: Robustness Penalty 符号修正 (P0.4)

**问题**: 符号错误导致鼓励方差而非惩罚方差

**Phase 2.1 错误实现**:
```python
loss = m2_loss(y_true, y_pred) - gamma * torch.var(y_pred)
# 最小化时鼓励高方差
```

**Phase 2-fixes 正确实现**:
```python
loss = m2_loss(y_true, y_pred) + gamma * torch.var(y_pred)
# 最小化时惩罚高方差
```

**影响范围**: 3 个 loss 函数
- `m2_robust_gamma001`
- `m2_robust_gamma01`
- `m2_robust_gamma10`

### ✅ 修复 3: 辅助函数添加

添加 Phase 1.5 的核心辅助函数：
- `_normalized_direction_term()` - 归一化方向惩罚
- `_huber_term()` - Huber 损失项

## 本次实验目标

### 主要观察目标

1. **M2 修复效果**
   - Variant 1 (IMADL+M2) 是否出现正 Sharpe
   - 是否能继承 Phase 1.5 M2 的高收益潜力

2. **Robustness Penalty 效果**
   - Variant 3 (M2 robust) 是否降低 CV
   - 是否在稳定性和收益间取得平衡

3. **整体性能提升**
   - 至少 1 个 loss 的 Sharpe > 0.3 (超过 GMADL)
   - 至少 1 个 loss 的 CV < 1.0 (超过 M2)

### 期待的改进

#### 最低期待 (必须达到)
- ✅ Variant 1 至少 1 个 alpha 值的 Sharpe > 0.2
- ✅ Variant 3 至少 1 个 gamma 值的 CV < 1.2
- ✅ 无 NaN/Inf 错误

#### 目标期待 (期望达到)
- 🎯 Variant 1 至少 2 个 alpha 值的 Sharpe > 0.4
- 🎯 Variant 3 至少 1 个 gamma 值的 Sharpe > 0.3 且 CV < 1.0
- 🎯 Variant 4 至少 1 个 lambda 值的 Sharpe > 0.5

#### 理想期待 (最佳情况)
- 🌟 至少 1 个 loss 的 Sharpe > 0.6 且 CV < 0.9
- 🌟 明确识别出最优 alpha/beta/gamma/lambda 值
- 🌟 理解为什么某些组合有效

## 实验配置

### Phase 2.1 重跑配置

```bash
python run_phase2_robustness.py \
  --losses imadl_m2_alpha02,imadl_m2_alpha03,imadl_m2_alpha04,imadl_m2_alpha05,\
imadl_m2_alpha06,imadl_m2_alpha07,imadl_m2_alpha08,\
imadl_gmadl_beta03,imadl_gmadl_beta05,imadl_gmadl_beta07,\
m2_robust_gamma001,m2_robust_gamma01,m2_robust_gamma10,\
adaptive_lambda10,adaptive_lambda50,adaptive_lambda100 \
  --seeds 42,52,62 \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --max-weight 0.05 \
  --matrix-mode light \
  --output-dir /content/drive/MyDrive/FYP/phase2-fixes/results \
  --checkpoint-dir /content/drive/MyDrive/FYP/phase2-fixes/checkpoints \
  --skip-existing \
  --resume-mode auto
```

**实验规模**:
- 16 losses × 3 seeds × 1 cap = **48 runs**
- 预计时间: 8-10 小时 (Colab T4)

## 剩余待办事项 (Phase 2.1b - 2.3)

### P1: 短期任务 (基于 P0 结果)

#### P1.1: Phase 2.1b 对齐实验
**目标**: 验证 Phase 2 runner 能否复现 Phase 1.5 结果

**配置**:
- Losses: `imadl`, `gmadl`, `hybrid_mul`
- Seeds: 42, 52, 62
- Test period: 24 months
- Total: 9 runs, 2 hours

**成功标准**:
- IMADL: Sharpe 0.464 ± 10%
- M2 (hybrid_mul): Sharpe 0.914 ± 10%
- GMADL: Sharpe 0.307 ± 10%

**状态**: ⏳ 待执行

#### P1.2: Loss Scale 诊断
**目标**: 测量 loss 组件量级差异

**配置**:
- Losses: `imadl_m2_alpha05`, `imadl_gmadl_beta05`, `m2_robust_gamma01`, `adaptive_lambda50`
- Seed: 42
- Test period: 6 months
- Total: 4 runs, 1 hour

**诊断指标**:
- 每个 batch 的 IMADL/GMADL/M2 分量大小
- 梯度范数
- 量级比率 (max_component / min_component)

**决策点**:
- 比率 < 5x: 无需归一化
- 比率 5-10x: 建议归一化
- 比率 > 10x: 必须归一化

**状态**: ⏳ 待执行

#### P1.3: Normalized Combination (条件执行)
**触发条件**: P1.2 发现 scale 比率 > 10x

**目标**: 通过归一化解决 loss scale 失衡

**新 Losses**:
- `imadl_m2_alpha*_normalized` (7 个)

**配置**:
- Seeds: 42, 52, 62
- Test period: 24 months
- Total: 21 runs, 4 hours

**状态**: ⏸️ 待 P1.2 结果决定

#### P1.4: 新型 Robustness Penalty (条件执行)
**触发条件**: P0 修正后 Variant 3 仍然 Sharpe < 0.3

**目标**: 测试更合理的 robustness penalty

**新 Penalty 类型**:
1. Weight Concentration (Herfindahl Index)
2. Turnover Proxy

**新 Losses**:
- `m2_weight_concentration_gamma01/10`
- `m2_turnover_gamma01/10`

**配置**:
- Seeds: 42, 52, 62
- Test period: 24 months
- Total: 12 runs, 3 hours

**状态**: ⏸️ 待 P0 结果决定

### P2: 中期任务 (系统性改进)

#### P2.1: Seed Ensemble
**目标**: 通过多 seed 平均降低方差

**方法**:
- 训练 3 个不同 seed 的模型
- 预测时取平均
- 基于 ensemble 预测排序

**配置**:
- Losses: Top 2-3 from P1
- Ensemble seeds: {42, 52, 62}
- Test period: 24 months
- Total: 9 runs, 3 hours

**成功标准**: Ensemble CV < 单 seed CV × 0.8

**状态**: ⏳ 待 P1 完成

#### P2.2: Validation Sharpe Early Stopping
**目标**: 基于交易表现而非预测误差选择模型

**实现**:
- Validation period: 1994-01 to 1994-12
- Validation Sharpe 频率: 每 5 epochs
- Early stopping patience: 10 epochs

**配置**:
- Losses: Top 3 from P1
- Seeds: 42, 52, 62
- Test period: 24 months
- Total: 18 runs, 5 hours

**成功标准**: Validation Sharpe ES 的测试 Sharpe > 固定 epoch + 0.1

**状态**: ⏳ 待 P1 完成

### P3: Phase 2.1 完整重跑

**前置条件**: P0-P2 完成，所有修复已验证有效

**目标**: 用修复后的实现重新运行完整 Phase 2.1

**配置**:
- Losses: 全部 13-24 个 (取决于 P1 结果)
- Seeds: 42, 52, 62
- Test period: 24 months
- Total: 39-72 runs, 8-15 hours

**成功标准**:
- 最低: 至少 1 个 loss 的 Sharpe > 0.3
- 目标: 至少 2 个 loss 的 Sharpe > 0.5
- 理想: 至少 1 个 loss 的 Sharpe > 0.6 且 CV < 0.9

**状态**: ⏳ 待 P0-P2 完成

## 关键决策点

### 决策点 1: P0 完成后 (当前)
**时间**: 实验运行后 8-10 小时

**评估指标**:
- Variant 1 最佳 Sharpe
- Variant 3 最佳 CV
- Failure rate

**决策树**:
```
IF Variant 1 最佳 Sharpe < 0.2:
    → M2 修复可能无效，需要深入诊断
    → 执行 P1.2 (Loss Scale 诊断)
ELSE IF Variant 1 最佳 Sharpe 0.2-0.4:
    → M2 修复部分有效
    → 执行 P1.1 (对齐实验) + P1.2 (诊断)
ELSE IF Variant 1 最佳 Sharpe > 0.4:
    → M2 修复成功
    → 跳过 P1.2，直接进入 P2

IF Variant 3 最佳 CV > 1.2:
    → Robustness penalty 修复无效
    → 执行 P1.4 (新型 penalty)
ELSE:
    → Robustness penalty 修复有效
    → 继续 P2
```

### 决策点 2: P1 完成后
**时间**: P0 后 2-3 天

**评估指标**:
- 对齐误差
- Loss scale 比率
- 最佳 loss 的 Sharpe 和 CV

**决策树**:
```
IF 对齐误差 > 15%:
    → Runner 有问题，修复后重跑
ELSE IF Loss scale 比率 > 10x:
    → 执行 P1.3 (Normalized Combination)
ELSE IF 最佳 Sharpe > 0.5:
    → 直接进入 P2 (Ensemble + Validation ES)
ELSE:
    → 执行 P1.4 (新型 penalty)
```

### 决策点 3: P2 完成后
**时间**: P1 后 2-3 天

**评估指标**:
- Ensemble 是否降低 CV
- Validation ES 是否提升 Sharpe

**决策树**:
```
IF Ensemble 有效 AND Validation ES 有效:
    → P3 使用两种方法
ELSE IF 只有一种有效:
    → P3 只使用有效方法
ELSE:
    → P3 使用原始设置
```

### 决策点 4: P3 完成后 (最终)
**时间**: P2 后 2-3 天

**评估指标**:
- Phase 2.1 最佳 loss 的综合表现

**决策树**:
```
IF 最佳 Sharpe > 0.6 AND CV < 0.9:
    → Phase 2.1 大成功
    → 选择 top 2-3 losses 进入 Phase 2.2
ELSE IF 最佳 Sharpe > 0.5 OR CV < 1.0:
    → Phase 2.1 成功
    → 选择 top 2-3 losses 进入 Phase 2.2
ELSE:
    → Phase 2.1 失败
    → 不进入 Phase 2.2
    → 重点转向 Seed Ensemble
```

## 时间线

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| P0 | M2 修复 + Robustness 修复 | 已完成 | ✅ |
| P0 实验 | Phase 2.1 重跑 (48 runs) | 8-10 小时 | ⏳ |
| P1.1 | 对齐实验 (9 runs) | 2 小时 | ⏳ |
| P1.2 | Loss scale 诊断 (4 runs) | 1 小时 | ⏳ |
| P1.3 | Normalized (21 runs) | 4 小时 | ⏸️ |
| P1.4 | 新 penalty (12 runs) | 3 小时 | ⏸️ |
| P2.1 | Seed Ensemble (9 runs) | 3 小时 | ⏳ |
| P2.2 | Validation ES (18 runs) | 5 小时 | ⏳ |
| P3 | 完整重跑 (39-72 runs) | 8-15 小时 | ⏳ |
| **总计** | | **34-53 小时** | |

## 文件结构

```
phase2-fixes/
├── Model_Train/
│   └── losses.py                    # ✅ 已修复
├── doc/
│   └── phase2-fix/
│       └── README.md                # 本文档
├── run_phase2_robustness.py         # 待创建
├── run_phase21b_alignment.py        # 待创建
├── diagnose_loss_scales.py          # 待创建
└── Phase2_Fixes_Colab_Runner.ipynb  # 待创建
```

## 下一步行动

1. ✅ 完成 P0 代码修复
2. ⏳ 创建 Colab notebook
3. ⏳ 在 Colab 运行 Phase 2.1 重跑实验
4. ⏳ 分析结果并做出决策点 1 的决策
5. ⏳ 根据结果执行 P1 任务

---

**文档创建时间**: 2026-04-28  
**分支**: phase2-fixes  
**状态**: P0 完成，等待 Colab 实验
