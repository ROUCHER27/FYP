# 下一步工作大纲：Loss 实验 + Colab 迁移

## 1. 研究问题

### 核心研究问题
- 能否设计出同时兼顾**方向性**、**预测精度**、**鲁棒性**的损失函数，用于股票收益预测与多空组合构建？

### 子问题
- 新设计的方向型损失，是否比 `MSE`、`MedSE`、`GMADL` 更适合你的数据？
- 改进后的 loss 是否能改善组合表现，而不只是改变点预测误差？
- 在 24 个月测试期下，结果是否仍然稳定，而不是只在 6 个月窗口里偶然有效？

## 2. 本轮目标

### 工程目标
- 在现有仓库中接入 3 个新 loss：
  - `imadl`
  - `dirhuber`
  - `hybrid`
- 保持现有训练/评估主链路不重写，只扩展接口。
- 建立一套可以在本地和 Colab 共用的批量实验入口。

### 实验目标
- 在同一设置下完成 6 个 loss 的统一对比：
  - `mse`
  - `medse`
  - `gmadl`
  - `imadl`
  - `dirhuber`
  - `hybrid`
- 测试期从 6 个月扩展到 24 个月。
- 输出统一格式的指标表、汇总 JSON、曲线图和总对比表。

### 研究产出目标
- 形成一条清楚的论文叙事链：
  - `MSE`：传统基线
  - `MedSE`：鲁棒基线
  - `GMADL`：方向性基线
  - `Improved MADL / Directional Huber / Hybrid`：你的新方案

## 3. 执行顺序总览

### 原则
- **先本地改代码并做短测试，再迁移 Colab。**
- **先打通单实验，再做批量自动化。**
- **先做 static sanity check 主实验，再考虑 rolling。**

### 原因
- 如果一边改训练逻辑、一边迁 Colab，报错时很难判断是代码问题还是环境问题。
- 当前真实实验入口是 `sanity_check_signal_tilted.py`，不是计划文档里反复提到的 `sanity_check_core.py`，必须先把本地链路认准。

## 4. 详细步骤

### Step 1：锁定本轮实验范围
- 固定特征集为 `X1`。
- 固定实验形态为：
  - 训练窗口：`1990-01` 到 `1994-12`
  - 测试窗口：从 `1995-01` 开始的 `24` 个月
- 固定网络结构沿用当前最优配置：
  - hidden dims = `[64, 32, 16]`
  - activation = `relu`
  - dropout = `0.2`

**注意事项**
- 本轮不要同时引入 `X2/X3`。
- 本轮不要先做 ranking loss。
- 本轮不要先把重点放在 rolling 实验；rolling 只做接口同步，不作为主结果来源。

### Step 2：先改 loss 定义层
- 修改 `Model_Train/losses.py`：
  - 新增 `imadl_rebalanced_loss`
  - 新增 `directional_huber_loss`
  - 新增 `hybrid_dir_huber_loss`
- 把这些 loss 的默认超参数按当前计划固定下来，先不要开放过多 CLI 参数。

**注意事项**
- 第一版先追求“数值稳定、能训练、能对比”，不要第一轮就做复杂超参数搜索。
- 要特别检查：
  - 是否出现 `nan` / `inf`
  - 方向惩罚项是否量级过弱
  - 新 loss 的输出是否全部接近某个常数，导致训练无效

### Step 3：接入真实训练入口
- 修改 `sanity_check_signal_tilted.py` 中的 `train_model()`。
- 让 `loss_name` 支持：
  - `mse`
  - `medse`
  - `gmadl`
  - `imadl`
  - `dirhuber`
  - `hybrid`
- 在月度评估结果中补充方向指标：
  - `directional_accuracy`
  - `sign_mismatch_large_y`

**注意事项**
- 不要只改 `sanity_check_core.py`；当前 runner 并没有走那条链路。
- 输出字段一旦确定，后面本地和 Colab 都必须沿用同一格式，否则汇总脚本会很乱。

### Step 4：补运行脚本
- 新增：
  - `run_sanity_check_imadl.py`
  - `run_sanity_check_dirhuber.py`
  - `run_sanity_check_hybrid.py`
- 风格和现有：
  - `run_sanity_check_mse.py`
  - `run_sanity_check_medse.py`
  保持一致。

**注意事项**
- runner 只做参数透传，不要复制训练逻辑。
- 各 runner 的 CLI 接口保持统一，后面批量脚本才容易调用。

### Step 5：先在本地做短烟雾测试
- 本地先跑小规模验证：
  - `max_epochs=2~3`
  - `test_months=2`
- 至少先测：
  - `mse`
  - `medse`
  - `imadl`

**注意事项**
- 这一步目标不是拿研究结论，而是验证：
  - CLI 能跑通
  - 输出文件能生成
  - loss 不爆炸
  - JSON / CSV / PNG 格式正确
- 本地短测不过，不要上 Colab。

### Step 6：补最小单元测试
- 在 `tests/` 下增加 loss 相关测试。
- 最少覆盖：
  - 方向正确 vs 方向错误时 loss 相对大小
  - 极端小收益下数值稳定性
  - 大误差下 Huber 分段行为
  - `backward()` 是否能正常产生梯度

**注意事项**
- 这类 loss 很容易“代码能跑但公式意义错了”。
- 单元测试比长时间训练更适合先抓公式接反、量级错位的问题。

### Step 7：做批量实验脚本
- 新增 `run_all_experiments.py`，顺序调用 6 个 runner。
- 自动收集：
  - `sanity_summary_{loss}.json`
  - `sanity_metrics_{loss}.csv`
- 自动生成总表：
  - `all_losses_comparison.csv`

**注意事项**
- 第一版先做“按 loss 级别跳过已完成实验”。
- 先不要优先做复杂的 epoch 级 resume；对你当前任务，loss 级别断点续跑更实用。

### Step 8：准备 Colab 目录结构
- 在 Google Drive 建：
  - `FYP/Model_Train/`
  - `FYP/sanity_outputs/`
  - `FYP/checkpoints/`
  - `FYP/notebooks/`
- 上传：
  - `Model_Train/`
  - 所有 `run_*.py`
  - `sanity_check_signal_tilted.py`
  - CSV 数据
  - `best_hyperparameters.txt`

**注意事项**
- 代码和数据都放 Drive，但训练时复制到 `/content/FYP` 再运行，避免直接从 Drive 读写拖慢速度。
- Colab 环境是临时的，Drive 才是持久存储。

### Step 9：先做 Colab 单实验验证
- 在 Colab notebook 里完成：
  - 挂载 Drive
  - 安装依赖
  - 复制代码到 `/content/FYP`
  - 切换目录
  - 跑单个 `mse`
- 确认：
  - GPU 可用
  - 输出能写回 Drive
  - 单实验流程完整

**注意事项**
- 第一轮优先用 `T4 GPU`。
- 现在不要优先选 `TPU`，因为你当前代码是标准 PyTorch，不是 `torch_xla` 路线。
- 只有单实验通过，才允许上批量运行。

### Step 10：Colab 批量跑完整实验
- 调用 `run_all_experiments.py` 一次性跑：
  - `mse`
  - `medse`
  - `gmadl`
  - `imadl`
  - `dirhuber`
  - `hybrid`
- 统一输出到 Drive 的 `sanity_outputs/`。

**注意事项**
- 先确认批量脚本支持跳过已完成 loss。
- 全量运行前先确认：
  - 输出目录正确
  - 日志可追踪
  - 失败时不会覆盖已有结果

### Step 11：整理实验结果
- 汇总每个 loss 的：
  - 平均误差
  - 平均 R²
  - 平均方向准确率
  - long-short cumulative return
  - Sharpe ratio
- 生成总图表：
  - Sharpe 对比
  - 累计收益对比
  - 方向准确率对比
  - `R² vs Sharpe` 散点图

**注意事项**
- 先保证文件命名和字段稳定，再谈分析。
- 不要手工拼结果，必须由脚本统一读取和导出。

## 5. Colab 迁移步骤（单独列出）

### 5.1 环境准备
- 在 Colab 选择 `T4 GPU`
- 挂载 Google Drive
- 安装依赖：
  - `torch`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

### 5.2 代码与数据同步
- 从 Drive 复制代码到 `/content/FYP`
- 从 Drive 复制 CSV 到 `/content/FYP`
- 切换工作目录到 `/content/FYP`

### 5.3 先跑单实验
- 运行一个最简单的 `mse` 任务
- 确认 summary / csv / png 正常落盘

### 5.4 再跑批量实验
- 运行 `run_all_experiments.py`
- 让脚本自动顺序执行所有 loss

### 5.5 结果回收
- 所有实验结果保存回 Drive
- Colab session 结束后，本地只需要同步结果文件，不需要重新训练

## 6. 芯片/硬件时间预估

> 这是当前阶段的经验性估计，用于排计划，不是严格 benchmark。

### 本地
- `Apple M1 Pro`：
  - 单个实验约 `5–7` 小时
  - 6 个 loss 全跑约 `30–42` 小时

### Colab
- `CPU`：
  - 不推荐，通常会比你本地更慢
- `T4 GPU`：
  - 单个实验约 `1–1.5` 小时
  - 6 个 loss 约 `6–9` 小时
- `L4 / A100`：
  - 单个实验约 `20–50` 分钟
  - 6 个 loss 约 `2–5` 小时
- `TPU v5e`：
  - 理论可能更快，但当前代码不能直接拿来用，迁移成本高，**不建议第一轮采用**

## 7. 实验设计要点

### 固定设置
- 特征集：`X1`
- 训练窗口：`1990-01` 到 `1994-12`
- 测试窗口：`1995-01` 起 `24` 个月
- 网络结构：`[64, 32, 16] + ReLU + Dropout 0.2`
- 组合方式：沿用当前 signal-weighted long-short 逻辑

### 对比对象
- `mse`
- `medse`
- `gmadl`
- `imadl`
- `dirhuber`
- `hybrid`

### 重点指标
- `MSE`
- `MedSE`
- `R²`
- `Directional Accuracy`
- `Sign Mismatch on Large |y|`
- `Cumulative Return`
- `Sharpe Ratio`

## 8. 关键注意事项

### 代码层面
- 当前真实主入口是 `sanity_check_signal_tilted.py`，不要改错文件。
- 新增 loss 后，所有 runner、批量脚本、汇总脚本都要用同一套 loss 名称。
- 输出字段必须前后一致，否则后面批量汇总会失败。

### 实验层面
- 第一轮目标是**能稳定比较 6 个 loss**，不是做所有扩展。
- 不要同时改 feature set、网络结构、训练窗口，否则结论会混乱。
- 先看相对比较，再看绝对数值。

### Colab 层面
- Colab 会断开，结果必须落到 Drive。
- 代码运行时尽量放在 `/content`，不要长期直接从 Drive 训练。
- 批量脚本必须支持失败后重跑，而不是每次全量重头开始。

### 论文层面
- 先保证实验链路完整，再写理论包装。
- 先形成对比结果，再决定哪一个新 loss 作为“主贡献”。
- 如果新 loss 没明显优于 `MedSE`，也仍然可以写成负结果分析，但前提是实验设计必须扎实。

## 9. 这一版完成后的最低验收标准
- 本地可以成功跑通至少一个新 loss。
- Colab 可以成功跑通单个 `mse` 实验并保存输出。
- 批量脚本可以顺序调用全部 6 个 loss。
- 输出目录中存在：
  - `sanity_metrics_{loss}.csv`
  - `sanity_summary_{loss}.json`
  - `{loss}_loss_curve.png`
  - `{loss}_returns_curve.png`
  - `all_losses_comparison.csv`
