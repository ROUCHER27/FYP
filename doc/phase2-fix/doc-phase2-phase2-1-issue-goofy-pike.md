# Phase 2.1 问题修复与下一步实验计划

## 背景

Phase 2.1 实验已在 worktree 中完成，但暴露了严重问题：

1. **结果极差**: 最佳 Sharpe 仅 0.0399 (imadl_gmadl_beta03)，远低于 Phase 1.5 基线
2. **M2 实现不一致**: 当前简化的 `m2_loss` 与 Phase 1.5 的 `hybrid_mul_m2` 不同
3. **Robustness Penalty 符号错误**: 公式 `M2 - γ·std(y_pred)²` 鼓励方差而非惩罚方差
4. **Loss Scale 失衡**: 线性组合可能因量级差异而失败
5. **缺少 Validation Sharpe**: Early stopping 仅基于训练 loss，未考虑交易表现
6. **Seed 敏感性**: Phase 1.5 显示高 seed 方差，但 Phase 2.1 未解决此问题

Issue 文档建议在进入 Phase 2.2 前完成 6 项优先任务。

**关键调整**: 基于 issue 分析，本计划采用"先诊断、再修复、后扩展"的策略，并支持并行执行。

---

## 关键文件

### 主代码库（需要更新）
- `Model_Train/losses.py` - 添加 Phase 2 losses，修复 M2/robustness 实现
- `Model_Train/train.py` - 添加 validation Sharpe early stopping
- `sanity_check_signal_tilted.py` - 添加 loss scale 诊断
- `run_phase2_robustness.py` - 创建 Phase 2.1b 对齐实验脚本

### Worktree（参考实现）
- `.claude/worktrees/priceless-poitras-3c4921/Model_Train/losses.py` - 当前 Phase 2 实现
- `.claude/worktrees/priceless-poitras-3c4921/run_phase2_robustness.py` - 编排脚本

### 文档
- `doc/phase2/Phase2-experiment-plan.md` - 原始计划
- `phase2.1 issue.md` - 问题分析与建议

---

## 实施计划（按优先级分组）

### P0: 立即执行（诊断 + 基础修复）

**目标**: 在 24 小时内完成诊断和基础修复，确定问题根源

#### 任务 P0.1: Phase 2.1b 对齐实验（并行任务 A）
**目标**: 验证 Phase 2 runner 能否复现 Phase 1.5 结果

**实验设计**:
- **Losses**: `imadl`, `gmadl`, `hybrid_mul` (Phase 1.5 的 M2)
- **Seeds**: 42, 52, 62
- **Weight Cap**: 0.05
- **Test Period**: 24 个月
- **总运行数**: 3 losses × 3 seeds = 9 runs
- **预计时间**: 2 小时 (Colab T4)

**成功标准**:
- IMADL: Sharpe 0.464 ± 10% (0.418-0.510)
- M2: Sharpe 0.914 ± 10% (0.823-1.005)
- GMADL: Sharpe 0.307 ± 10% (0.276-0.338)

**操作步骤**:
1. 创建 `run_phase2_1b_alignment.py`
2. 在 Colab 运行: `python run_phase2_1b_alignment.py --seeds 42,52,62`
3. 生成对比报告: `python compare_phase15_phase21b.py`

**决策点**: 如果对齐失败（偏差 >15%），立即停止修复 runner 差异，不继续后续实验

#### 任务 P0.2: Loss Scale 诊断（并行任务 B）
**目标**: 测量各 loss 组件的量级差异，判断是否需要归一化

**实验设计**:
- **Losses**: 选择 4 个代表性组合
  - `imadl_m2_alpha05` (Variant 1 中点)
  - `imadl_gmadl_beta05` (Variant 2 中点)
  - `m2_robust_gamma01` (Variant 3 中点)
  - `adaptive_lambda50` (Variant 4 中点)
- **Seeds**: 42 (单 seed 足够诊断)
- **Test Period**: 6 个月 (快速诊断)
- **总运行数**: 4 runs
- **预计时间**: 1 小时 (Colab T4)

**诊断指标**:
- 每个 batch 记录:
  - IMADL 分量大小
  - GMADL/M2 分量大小
  - 组合 loss 总值
  - 各分量梯度范数
- 计算量级比率: `max_component / min_component`

**判断标准**:
- **比率 < 5x**: Scale 平衡，无需归一化
- **比率 5-10x**: 中度失衡，建议归一化
- **比率 > 10x**: 严重失衡，必须归一化

**操作步骤**:
1. 修改 `sanity_check_signal_tilted.py` 添加 scale logging
2. 运行诊断实验
3. 生成可视化: `python analyze_loss_scales.py`

**决策点**: 如果发现比率 > 10x，Phase 2.1e (normalized combination) 升级为 P1 优先级

#### 任务 P0.3: 修正 M2 实现
**目标**: 确保 Phase 2 使用与 Phase 1.5 相同的 M2

**当前问题**:
```python
# 当前简化版本（错误）
m2_loss = -sign(y_true) * (y_pred - y_true) ** 2

# Phase 1.5 版本（正确）
m2_loss = hybrid_dir_huber_mul_loss(y_true, y_pred, lambda_dir=2.0)
```

**修复步骤**:
1. 读取 Phase 1.5 的 `hybrid_dir_huber_mul_loss` 实现
2. 更新 `Model_Train/losses.py`:
   - 修改 `m2_loss()` 函数
   - 更新所有 Variant 1 losses (7 个 `imadl_m2_alpha*`)
   - 更新所有 Variant 4 losses (3 个 `adaptive_lambda*`)
3. 添加 docstring 说明使用的 M2 版本

**影响范围**: 10 个 loss 函数

#### 任务 P0.4: 修正 Robustness Penalty 符号
**目标**: 让 robustness penalty 真正惩罚方差

**当前问题**:
```python
# 错误：最小化时鼓励高方差
loss = m2_loss(y_true, y_pred) - gamma * torch.var(y_pred)

# 正确：最小化时惩罚高方差
loss = m2_loss(y_true, y_pred) + gamma * torch.var(y_pred)
```

**修复步骤**:
1. 更新 `m2_robustness_enhanced_loss()` 函数
2. 更新所有 Variant 3 losses (3 个 `m2_robust_gamma*`)
3. 添加单元测试验证惩罚方向:
   ```python
   def test_robustness_penalty_direction():
       # 高方差预测应该有更高的 loss
       y_true = torch.tensor([0.1, 0.2])
       y_pred_low_var = torch.tensor([0.1, 0.2])
       y_pred_high_var = torch.tensor([0.0, 0.4])
       
       loss_low = m2_robust_gamma01_loss(y_true, y_pred_low_var)
       loss_high = m2_robust_gamma01_loss(y_true, y_pred_high_var)
       
       assert loss_high > loss_low, "High variance should have higher loss"
   ```

**影响范围**: 3 个 loss 函数 + 1 个测试

**P0 阶段总时间**: 
- 代码修改: 4 小时
- 实验运行: 3 小时 (并行)
- 分析决策: 2 小时
- **总计**: 约 1 个工作日

---

### P1: 短期执行（基于 P0 诊断结果）

**前置条件**: P0 完成，已知 scale 比率和对齐状态

#### 任务 P1.1: Phase 2.1d - 新型 Robustness Penalty（条件执行）
**触发条件**: P0.4 修正后，Variant 3 仍然表现不佳（Sharpe < 0.3）

**目标**: 测试 3 种更合理的 robustness penalty

**新 Penalty 类型**:

1. **Prediction Variance Penalty** (已在 P0.4 修正)
   ```python
   penalty = gamma * torch.var(y_pred)
   ```

2. **Weight Concentration Penalty** (Herfindahl Index)
   ```python
   # 惩罚权重过度集中在少数股票
   weights_normalized = torch.softmax(y_pred, dim=0)
   penalty = gamma * torch.sum(weights_normalized ** 2)
   ```

3. **Turnover Proxy Penalty**
   ```python
   # 惩罚预测值剧烈变化（代理换手率）
   # 需要在训练时记录上一 batch 的预测
   penalty = gamma * torch.mean(torch.abs(y_pred - y_pred_prev))
   ```

**实验设计**:
- **新 Losses**: 
  - `m2_weight_concentration_gamma01`
  - `m2_weight_concentration_gamma10`
  - `m2_turnover_gamma01`
  - `m2_turnover_gamma10`
- **Seeds**: 42, 52, 62
- **Test Period**: 24 个月
- **总运行数**: 4 losses × 3 seeds = 12 runs
- **预计时间**: 3 小时 (Colab T4)

**成功标准**: 至少 1 个新 penalty 的 Sharpe > 0.3 且 CV < 1.2

#### 任务 P1.2: Phase 2.1e - Normalized Combination（条件执行）
**触发条件**: P0.2 诊断发现 scale 比率 > 10x

**目标**: 通过归一化解决 loss scale 失衡问题

**归一化方法**:

**方法 A: Min-Max 归一化**
```python
def normalize_loss_component(loss_val, running_min, running_max):
    return (loss_val - running_min) / (running_max - running_min + 1e-8)

# 在训练时维护每个组件的 running statistics
imadl_normalized = normalize_loss_component(imadl_val, imadl_min, imadl_max)
m2_normalized = normalize_loss_component(m2_val, m2_min, m2_max)
combined = alpha * imadl_normalized + (1 - alpha) * m2_normalized
```

**方法 B: Z-score 归一化**
```python
def standardize_loss_component(loss_val, running_mean, running_std):
    return (loss_val - running_mean) / (running_std + 1e-8)

imadl_std = standardize_loss_component(imadl_val, imadl_mean, imadl_std)
m2_std = standardize_loss_component(m2_val, m2_mean, m2_std)
combined = alpha * imadl_std + (1 - alpha) * m2_std
```

**方法 C: 可学习权重**
```python
# 让模型自动学习最优权重
log_alpha = nn.Parameter(torch.tensor(0.0))
log_beta = nn.Parameter(torch.tensor(0.0))

alpha = torch.sigmoid(log_alpha)
beta = torch.sigmoid(log_beta)

combined = alpha * imadl_val + beta * m2_val
```

**实验设计**:
- **测试方法**: 先测试方法 A (最简单)
- **Losses**: 重新运行 Variant 1 的 7 个 alpha 值
  - `imadl_m2_alpha02_normalized`
  - `imadl_m2_alpha03_normalized`
  - ... (共 7 个)
- **Seeds**: 42, 52, 62
- **Test Period**: 24 个月
- **总运行数**: 7 losses × 3 seeds = 21 runs
- **预计时间**: 4 小时 (Colab T4)

**成功标准**: 至少 1 个归一化 loss 的 Sharpe > 原始版本 + 0.2

**P1 阶段总时间**: 
- 代码实现: 6 小时
- 实验运行: 7 小时 (可能只运行其中一个任务)
- 分析: 3 小时
- **总计**: 约 2 个工作日

---

### P2: 中期执行（系统性改进）

**前置条件**: P0 和 P1 完成，已有至少 1 个 Sharpe > 0.3 的 loss

#### 任务 P2.1: Seed Ensemble
**目标**: 通过多 seed 平均降低方差

**方法**:
```python
# 训练阶段：用 3 个 seed 训练同一 loss
models = []
for seed in [42, 52, 62]:
    model = train_model(loss_name, seed=seed)
    models.append(model)

# 预测阶段：平均 3 个模型的预测
predictions = []
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

y_pred_ensemble = torch.mean(torch.stack(predictions), dim=0)

# 组合构建：基于 ensemble 预测排序
portfolio = construct_portfolio(y_pred_ensemble)
```

**实验设计**:
- **Losses**: 选择 P1 中表现最好的 2-3 个 loss
- **Seeds**: 42, 52, 62 (ensemble)
- **对比**: Ensemble vs 单 seed
- **Test Period**: 24 个月
- **总运行数**: 3 losses × 3 seeds = 9 runs (训练) + 3 runs (ensemble 评估)
- **预计时间**: 3 小时 (Colab T4)

**成功标准**: Ensemble CV < 单 seed CV × 0.8 (降低 20%)

#### 任务 P2.2: Validation Sharpe Early Stopping
**目标**: 基于交易表现而非预测误差选择模型

**实现**:

1. **Validation Set 划分**:
   - 训练期: 1990-01 到 1993-12 (4 年)
   - 验证期: 1994-01 到 1994-12 (1 年)
   - 测试期: 1995-01 到 1996-12 (2 年)

2. **Validation Sharpe 计算**:
   ```python
   def compute_validation_sharpe(model, val_data, val_months):
       monthly_returns = []
       for month in val_months:
           X_month, y_month = get_month_data(val_data, month)
           y_pred = model.predict(X_month)
           
           # 构建 long-short 组合
           portfolio_return = construct_and_evaluate_portfolio(
               y_pred, y_month
           )
           monthly_returns.append(portfolio_return)
       
       # 计算 Sharpe
       mean_return = np.mean(monthly_returns)
       std_return = np.std(monthly_returns)
       sharpe = mean_return / (std_return + 1e-8) * np.sqrt(12)
       
       return sharpe
   ```

3. **Early Stopping 逻辑**:
   ```python
   best_val_sharpe = -np.inf
   patience_counter = 0
   patience = 10  # 10 epochs without improvement
   
   for epoch in range(max_epochs):
       train_loss = train_one_epoch(model, train_data)
       
       if epoch % 5 == 0:  # 每 5 epochs 评估一次
           val_sharpe = compute_validation_sharpe(model, val_data)
           
           if val_sharpe > best_val_sharpe:
               best_val_sharpe = val_sharpe
               save_checkpoint(model, "best_val_sharpe.pt")
               patience_counter = 0
           else:
               patience_counter += 1
           
           if patience_counter >= patience:
               print(f"Early stopping at epoch {epoch}")
               break
   
   # 加载最佳 checkpoint 用于测试
   model.load_state_dict(torch.load("best_val_sharpe.pt"))
   ```

**实验设计**:
- **Losses**: P1 中表现最好的 3 个 loss
- **Seeds**: 42, 52, 62
- **对比**: Validation Sharpe ES vs 固定 epoch
- **Test Period**: 24 个月
- **总运行数**: 3 losses × 3 seeds × 2 modes = 18 runs
- **预计时间**: 5 小时 (Colab T4)

**成功标准**: Validation Sharpe ES 的测试 Sharpe > 固定 epoch + 0.1

**P2 阶段总时间**: 
- 代码实现: 8 小时
- 实验运行: 8 小时
- 分析: 4 小时
- **总计**: 约 3 个工作日
---

### P3: Phase 2.1 完整重跑（整合所有修复）

**前置条件**: P0-P2 完成，所有修复已验证有效

**目标**: 用修复后的实现重新运行完整 Phase 2.1，获得可靠结果

**实验设计**:
- **Losses**: 全部 13 个 Phase 2 losses (修复后版本)
  - Variant 1: 7 个 `imadl_m2_alpha*` (使用正确的 M2)
  - Variant 2: 3 个 `imadl_gmadl_beta*`
  - Variant 3: 3 个 `m2_robust_gamma*` (修正符号)
  - 如果 P1.1 成功，添加 4 个新 penalty losses
  - 如果 P1.2 成功，添加 7 个 normalized losses
- **Seeds**: 42, 52, 62
- **Weight Cap**: 0.05
- **Test Period**: 24 个月
- **总运行数**: 13-24 losses × 3 seeds = 39-72 runs
- **预计时间**: 8-15 小时 (Colab T4)

**配置**:
- Validation Sharpe early stopping: 启用
- Loss scale diagnostics: 启用
- Checkpoint: 每 5 epochs 保存

**输出**:
- 每个 loss 的 `sanity_metrics_{loss}.csv`
- 每个 loss 的 `sanity_summary_{loss}.json`
- Loss scale 诊断报告
- 总对比表 `phase2_1_final_comparison.csv`

**成功标准**:
- **最低**: 至少 1 个 loss 的 Sharpe > 0.3 (超过 GMADL)
- **目标**: 至少 2 个 loss 的 Sharpe > 0.5 (超过 IMADL)
- **理想**: 至少 1 个 loss 的 Sharpe > 0.6 且 CV < 0.9 (同时超过两个基线)

---

## 关键决策点

### 决策点 1: P0 完成后
**时间**: 第 1 天结束

**评估指标**:
- Phase 2.1b 对齐误差
- Loss scale 比率

**决策树**:
```
IF 对齐误差 > 15%:
    → 停止，修复 runner 差异
    → 重新运行 P0.1
ELSE IF scale 比率 > 10x:
    → P1.2 (Normalized Combination) 升级为必做
    → 继续 P1
ELSE IF scale 比率 5-10x:
    → P1.2 标记为建议
    → 继续 P1
ELSE:
    → 跳过 P1.2
    → 继续 P1.1
```

### 决策点 2: P1 完成后
**时间**: 第 3-4 天结束

**评估指标**:
- 最佳 loss 的 Sharpe
- 最佳 loss 的 CV
- Failure rate

**决策树**:
```
IF 最佳 Sharpe < 0.3:
    → Phase 2.1 失败
    → Pivot 到 Seed Ensemble (P2.1)
    → 论文定位: "Naive combination fails, ensemble works"
ELSE IF 最佳 Sharpe 0.3-0.5:
    → Phase 2.1 部分成功
    → 继续 P2 (Ensemble + Validation Sharpe)
    → 论文定位: "Combination + ensemble improves robustness"
ELSE IF 最佳 Sharpe > 0.5:
    → Phase 2.1 成功
    → 继续 P2，准备 Phase 2.2
    → 论文定位: "Novel loss combination achieves better trade-off"
```

### 决策点 3: P2 完成后
**时间**: 第 6-7 天结束

**评估指标**:
- Ensemble 是否降低 CV
- Validation Sharpe ES 是否提升测试 Sharpe

**决策树**:
```
IF Ensemble CV 降低 > 20% AND Validation ES 提升 > 0.1:
    → 两种方法都有效
    → Phase 2.2 使用 Ensemble + Validation ES
    → 扩展到 6 seeds × 2 caps × 48 months
ELSE IF 只有一种方法有效:
    → Phase 2.2 只使用有效方法
    → 另一种方法作为 ablation study
ELSE:
    → Phase 2.2 使用原始设置
    → Ensemble 和 Validation ES 作为 future work
```

### 决策点 4: P3 完成后（最终决策）
**时间**: 第 10-12 天结束

**评估指标**:
- Phase 2.1 最佳 loss 的综合表现
- 与 Phase 1.5 基线的对比

**决策树**:
```
IF 最佳 Sharpe > 0.6 AND CV < 0.9:
    → Phase 2.1 大成功
    → 选择 top 2-3 losses 进入 Phase 2.2
    → 论文贡献: "Novel loss design achieves superior trade-off"
ELSE IF 最佳 Sharpe > 0.5 OR CV < 1.0:
    → Phase 2.1 成功
    → 选择 top 2-3 losses 进入 Phase 2.2
    → 论文贡献: "Loss combination improves one dimension"
ELSE:
    → Phase 2.1 失败
    → 不进入 Phase 2.2
    → 论文贡献: "Why naive combination fails + ensemble solution"
    → 重点转向 Seed Ensemble 作为主要贡献
```

---

## 并行执行策略

### 第 1 天（P0）
```
并行任务 A: Phase 2.1b 对齐实验 (2h Colab)
并行任务 B: Loss scale 诊断 (1h Colab)
串行任务: 代码修复 (M2 + Penalty) (4h 本地)

总时间: 约 6-7 小时
```

### 第 2-3 天（P1）
```
IF scale 比率 > 10x:
    并行任务 A: 新 Robustness Penalty (3h Colab)
    并行任务 B: Normalized Combination (4h Colab)
    串行任务: 代码实现 (6h 本地)
ELSE:
    串行任务: 新 Robustness Penalty (3h Colab + 6h 本地)

总时间: 约 1-2 天
```

### 第 4-6 天（P2）
```
串行任务 1: Seed Ensemble 实现 + 实验 (1 天)
串行任务 2: Validation Sharpe ES 实现 + 实验 (1.5 天)

总时间: 约 2.5 天
```

### 第 7-10 天（P3）
```
串行任务: Phase 2.1 完整重跑 (8-15h Colab)
并行任务: 结果分析 + 可视化 (1 天)

总时间: 约 2-3 天
```

**总时间估计**: 10-14 天（约 2-3 周）

---

## 验证步骤

### 每个阶段后的验证清单

#### P0 验证
- [ ] `pytest tests/test_losses.py` 全部通过
- [ ] Phase 2.1b 对齐误差 < 15%
- [ ] Loss scale 诊断报告生成
- [ ] M2 实现与 Phase 1.5 一致
- [ ] Robustness penalty 符号正确（单元测试通过）

#### P1 验证
- [ ] 新 penalty losses 训练不报错
- [ ] Normalized losses 的 scale 比率 < 2x
- [ ] 至少 1 个新 loss 的 Sharpe > 0.2
- [ ] 输出文件格式正确

#### P2 验证
- [ ] Ensemble 预测生成正确
- [ ] Validation Sharpe 计算逻辑正确
- [ ] Early stopping 触发正常
- [ ] Checkpoint 保存/加载正常

#### P3 验证
- [ ] 所有 13+ losses 完整运行
- [ ] 无 NaN/Inf 错误
- [ ] 总对比表生成
- [ ] 结果可复现（相同 seed 相同结果）

---

## 风险缓解

### 风险 1: P0 对齐失败（对齐误差 > 15%）
**概率**: 中等  
**影响**: 高（阻塞所有后续工作）

**缓解措施**:
1. 详细对比 Phase 1.5 和 Phase 2 runner 的差异
2. 检查数据预处理、特征构建、训练循环
3. 如果差异无法修复，使用 Phase 1.5 runner 运行 Phase 2 losses

**Fallback**: 使用 Phase 1.5 codebase，手动添加 Phase 2 losses

### 风险 2: 修复后结果仍然很差（最佳 Sharpe < 0.3）
**概率**: 中等  
**影响**: 中等（需要 pivot 研究方向）

**缓解措施**:
1. 深入分析为什么 naive combination 失败
2. Pivot 到 Seed Ensemble 作为主要贡献
3. 论文定位调整为 "negative result + ensemble solution"

**Fallback**: 
- 主贡献: Seed Ensemble 降低方差
- 次要贡献: Loss combination 的失败分析
- Future work: 更复杂的 combination 策略

### 风险 3: Loss scale 归一化无效
**概率**: 低  
**影响**: 低（只影响 P1.2）

**缓解措施**:
1. 尝试 3 种归一化方法（Min-Max, Z-score, Learnable）
2. 如果都无效，说明问题不在 scale
3. 转向分析 loss 组件的梯度方向冲突

**Fallback**: 跳过 normalized combination，专注于其他改进

### 风险 4: Colab 资源不足或频繁断线
**概率**: 中等  
**影响**: 中等（延长实验时间）

**缓解措施**:
1. 使用 Colab Pro 获得更长运行时间
2. 实现完善的 checkpoint/resume 机制
3. 将长实验拆分为多个短实验
4. 考虑使用 AWS/GCP 替代 Colab

**Fallback**: 
- 减少 seeds 数量（3 → 2）
- 减少测试月数（24 → 12）
- 优先运行最有希望的 losses

### 风险 5: Validation Sharpe 过于 noisy
**概率**: 中等  
**影响**: 低（只影响 P2.2）

**缓解措施**:
1. 使用 rolling average 平滑 validation Sharpe
2. 增加 validation period（12 → 18 months）
3. 使用 validation loss + validation Sharpe 的加权组合

**Fallback**: 
- 继续使用固定 epoch 训练
- Validation Sharpe 仅用于分析，不用于 early stopping

---

## 成功标准总结

### 最低标准（必须达到）
- [ ] Phase 2.1b 对齐成功（误差 < 15%）
- [ ] 至少 1 个 Phase 2 loss 的 Sharpe > 0.3
- [ ] Robustness penalty 确实降低方差
- [ ] 无严重 loss scale 失衡（比率 < 5x 或已归一化）

### 目标标准（期望达到）
- [ ] 至少 2 个 Phase 2 loss 的 Sharpe > 0.5
- [ ] 至少 1 个 Phase 2 loss 的 CV < 1.0
- [ ] Validation Sharpe ES 提升测试 Sharpe > 10%
- [ ] Seed Ensemble 降低 CV > 20%

### 理想标准（论文贡献）
- [ ] 至少 1 个 Phase 2 loss 的 Sharpe > 0.6 且 CV < 0.9
- [ ] 清晰理解哪些 combination 有效、为什么有效
- [ ] 可推广的 loss function design insights
- [ ] 完整的 ablation study 支持结论

---

## 下一步行动（立即开始）

### 今天（第 1 天上午）
1. **创建 P0 分支**: `git checkout -b phase2-p0-fixes`
2. **任务 P0.3**: 修正 M2 实现（2 小时）
   - 读取 Phase 1.5 的 `hybrid_dir_huber_mul_loss`
   - 更新 `Model_Train/losses.py`
   - 提交: `git commit -m "fix: align M2 with Phase 1.5 implementation"`
3. **任务 P0.4**: 修正 Robustness Penalty（1 小时）
   - 修改符号
   - 添加单元测试
   - 提交: `git commit -m "fix: correct robustness penalty sign"`

### 今天（第 1 天下午）
4. **任务 P0.2**: 实现 Loss Scale 诊断（2 小时）
   - 修改 `sanity_check_signal_tilted.py`
   - 创建 `analyze_loss_scales.py`
   - 提交: `git commit -m "feat: add loss scale diagnostics"`
5. **任务 P0.1**: 创建 Phase 2.1b 脚本（1 小时）
   - 创建 `run_phase2_1b_alignment.py`
   - 创建 `compare_phase15_phase21b.py`
   - 提交: `git commit -m "feat: add Phase 2.1b alignment experiment"`

### 今晚（第 1 天晚上）
6. **启动 Colab 实验**:
   - 上传代码到 Google Drive
   - 运行 Phase 2.1b 对齐实验（2h）
   - 运行 Loss scale 诊断（1h）
   - 设置为后台运行，明早查看结果

### 明天（第 2 天上午）
7. **分析 P0 结果**:
   - 检查对齐误差
   - 检查 scale 比率
   - 做出决策点 1 的决策
8. **规划 P1 任务**:
   - 根据诊断结果决定执行 P1.1 还是 P1.2
   - 开始实现相应代码

---

## 附录：实验配置速查表

### Phase 2.1b 对齐实验
```bash
python run_phase2_1b_alignment.py \
  --losses imadl,gmadl,hybrid_mul \
  --seeds 42,52,62 \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --max-weight 0.05 \
  --output-dir /content/drive/MyDrive/FYP/phase2_1b
```

### Loss Scale 诊断
```bash
python run_loss_scale_diagnostics.py \
  --losses imadl_m2_alpha05,imadl_gmadl_beta05,m2_robust_gamma01,adaptive_lambda50 \
  --seed 42 \
  --test-months 6 \
  --max-epochs 20 \
  --output-dir /content/drive/MyDrive/FYP/diagnostics
```

### Phase 2.1 完整重跑
```bash
python run_phase2_robustness.py \
  --matrix-mode light \
  --seeds 42,52,62 \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --max-weight 0.05 \
  --enable-validation-sharpe \
  --enable-loss-diagnostics \
  --output-dir /content/drive/MyDrive/FYP/phase2_1_final \
  --checkpoint-dir /content/drive/MyDrive/FYP/checkpoints \
  --skip-existing \
  --resume-mode auto
```
