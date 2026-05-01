# 学期2研究计划：股票收益预测的鲁棒方向损失函数

## 研究背景与动机

### 学期1已完成的工作

1. **强基线结果**：MedSE显著优于MSE（Sharpe 3.23 vs -1.46）
2. **GMADL缺陷识别**：
   - 对称性问题：奖励和惩罚过于对称，不符合"避免损失>追逐利润"
   - 弱梯度问题：当 ŷ→0 时梯度很弱
   - 忽略精度：只看方向和|y|，不惩罚|y-ŷ|
3. **完整基础设施**：
   - 特征工程：X1（15维动量+换手率）
   - 模型架构：MLP [64,32,16] + ReLU + Dropout 0.2
   - 实验框架：静态训练窗口 + 逐月调仓回测

### 导师文档核心指导（plan_semester2.pdf）

**设计目标**：创建可微分损失函数，重罚符号错误，对同符号的幅度误差更宽容

**五个改进方向**：
1. **3.2.1 Improved MADL**：翻转符号 + 添加幅度惩罚
2. **3.2.2 Directional Huber**：基于Huber的鲁棒方向损失
3. **3.2.3 Quantile Loss**：结合分位数损失的风险厌恶优化
4. **3.2.4 Bounded GMADL**：添加正则化防止损失趋向-∞
5. **3.2.5 Ranking Loss**：混合排序损失（复杂度高）

**理论分析要求**：
- 收敛性：凸性、Lipschitz连续性
- 一阶导数：平滑性、梯度行为
- 鲁棒性：对异常值的敏感度

### 研究范围收缩（基于Pre-plan建议）

**核心研究问题**：能否设计出平衡**方向**、**精度**、**鲁棒性**三者的损失函数？

**聚焦策略**：
- ✅ 实现3个新损失函数（不是全部5个）
- ✅ 迁移到Google Colab GPU加速实验
- ✅ 扩展测试期从6个月到24个月
- ❌ 暂不做ranking loss（太重，留作future work）
- ❌ 暂不测试X2/X3特征集（先验证损失函数有效性）

## 阶段1：损失函数设计与实现

### 1.1 改进型MADL（Rebalanced Improved MADL）

**基于导师文档3.2.1**

**原始公式（导师版本）**：
```
L = (1/N)Σ[(1 - Sign(r_i·r̂_i))·|r_i|^b + c·|r_i - r̂_i|^d]
```

**问题**：当月收益率很小时（如0.01），方向项 `|r|^b` 会非常弱，被幅度项淹没

**改进版本（Rebalanced）**：
```python
L_i = λ_dir · (1 - σ(a·y_i·ŷ_i)) · w̃_i + λ_mag · |y_i - ŷ_i|^d

其中：
  w̃_i = |y_i|^b / (mean(|y|^b) + ε)  # 归一化权重，防止方向项过弱
  σ(x) = 1/(1+e^(-x))                # sigmoid平滑符号函数

超参数：
  a = 100      # sigmoid陡峭度
  b = 2        # 收益率幂次
  d = 2        # 幅度误差范数（L2）
  λ_dir = 1.0  # 方向项权重
  λ_mag = 1.0  # 幅度项权重
  ε = 1e-8     # 数值稳定项
```

**设计理由**：
- 修复GMADL的奖励反转问题（负损失变正损失）
- 显式添加幅度精度惩罚 `|y-ŷ|^d`
- 通过归一化防止方向项被小收益率淹没
- 保持"投资模拟"的金融直觉

**预期效果**：
- 方向错误 → 高损失
- 方向正确但幅度差 → 中等损失
- 方向和幅度都准确 → 低损失

---

### 1.2 方向Huber损失（Directional Huber）

**基于导师文档3.2.2**

**公式**：
```python
L_i = λ_dir · pen(y_i, ŷ_i) + λ_hub · H_δ(y_i - ŷ_i)

其中：
  pen(y_i, ŷ_i) = 0.5 · (1 - tanh(a·y_i·ŷ_i))  # 方向惩罚项

  H_δ(e) = { 0.5·e²              if |e| ≤ δ     # Huber损失
           { δ·(|e| - 0.5·δ)     if |e| > δ

  e = y_i - ŷ_i  # 预测误差

超参数：
  a = 10       # tanh陡峭度（导师建议用10而非100，更平滑）
  δ = 0.01     # Huber阈值（需根据收益率尺度调整）
  λ_dir = 1.0  # 方向项权重
  λ_hub = 1.0  # Huber项权重
```

**设计理由**：
- 自然延伸MedSE的鲁棒性优势
- Huber对大误差（异常值）更鲁棒，小误差仍用L2
- 加法结构允许独立调节方向和幅度的重要性
- 导师文档建议用tanh而非sigmoid（输出范围[-1,1]更对称）

**与导师原版的差异**：
- 导师版本：`pen × H_δ(e)` （乘法结构）
- 我们版本：`pen + H_δ(e)` （加法结构）
- 理由：加法更容易解释和调参，方向和幅度效应可拆解

**预期效果**：
- 对极端收益率（outliers）更稳健
- 小误差仍保持二次惩罚（平滑梯度）
- 方向错误时额外惩罚

---

### 1.3 混合方向-Huber损失（Hybrid Directional-Huber）

**结合3.2.1和3.2.2的优势**

**公式（加法型）**：
```python
L_i = λ_1 · D_i + λ_2 · H_δ(e_i)

其中：
  D_i = (1 - σ(a·y_i·ŷ_i)) · w̃_i  # 归一化方向项
  w̃_i = |y_i|^b / (mean(|y|^b) + ε)
  H_δ(e_i) = Huber损失
  e_i = y_i - ŷ_i

超参数：
  a = 100
  b = 2
  δ = 0.01
  λ_1 = 1.0  # 方向项权重
  λ_2 = 1.0  # Huber项权重
  ε = 1e-8
```

**可选：乘法型（用于消融实验）**：
```python
L_i = (1 + λ·D_i) · H_δ(e_i)

含义：方向错误时，整个误差被额外放大
金融解释："错方向的误差在经济上代价更高"
```

**设计理由**：
- 统一三个属性：方向 + 精度 + 鲁棒性
- 最可能成为论文主要贡献
- 符合导师"把两者结合起来"的建议

**论文叙事**：
- GMADL → 有方向，但奖励结构有问题
- Improved MADL → 修复方向+精度
- Directional Huber → 加入鲁棒性
- **Hybrid → 三者统一的最优方案**

---

### 1.4 实现计划

**需要修改的文件**：

1. **`Model_Train/losses.py`**（新增3个损失函数）：
```python
def imadl_rebalanced_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    lambda_dir: float = 1.0,
    lambda_mag: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Rebalanced Improved MADL: 修复GMADL的精度问题
    基于导师文档3.2.1，添加归一化权重防止方向项过弱
    """
    # 方向项
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    dir_penalty = 1.0 - sigmoid

    # 归一化权重
    abs_y = torch.abs(y_true)
    weight = abs_y ** b
    mean_weight = weight.mean() + 1e-8
    normalized_weight = weight / mean_weight

    dir_term = dir_penalty * normalized_weight

    # 幅度项
    mag_term = (y_true - y_pred) ** 2

    # 组合
    loss = lambda_dir * dir_term + lambda_mag * mag_term
    return _reduce(loss, reduction)


def directional_huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 10.0,
    delta: float = 0.01,
    lambda_dir: float = 1.0,
    lambda_hub: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Directional Huber: 结合方向感知和鲁棒误差处理
    基于导师文档3.2.2，使用加法结构
    """
    # 方向惩罚项（使用tanh）
    product = a * y_true * y_pred
    tanh_val = torch.tanh(product)
    pen = 0.5 * (1.0 - tanh_val)

    # Huber损失
    error = y_true - y_pred
    abs_error = torch.abs(error)
    huber = torch.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )

    # 组合（加法）
    loss = lambda_dir * pen + lambda_hub * huber
    return _reduce(loss, reduction)


def hybrid_dir_huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    a: float = 100.0,
    b: float = 2.0,
    delta: float = 0.01,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    Hybrid Directional-Huber: 统一方向+精度+鲁棒性
    论文主要贡献
    """
    # 归一化方向项
    product = a * y_true * y_pred
    sigmoid = torch.sigmoid(product)
    dir_penalty = 1.0 - sigmoid

    abs_y = torch.abs(y_true)
    weight = abs_y ** b
    mean_weight = weight.mean() + 1e-8
    normalized_weight = weight / mean_weight

    dir_term = dir_penalty * normalized_weight

    # Huber项
    error = y_true - y_pred
    abs_error = torch.abs(error)
    huber = torch.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )

    # 组合
    loss = lambda_1 * dir_term + lambda_2 * huber
    return _reduce(loss, reduction)
```

2. **`sanity_check_core.py`**（扩展损失函数支持）：
   - 修改 `train_model()` 函数的 lines 208-213
   - 添加新的损失函数分支：
```python
if loss_name == "mse":
    criterion = lambda a, b: mse_loss(a, b, reduction="mean")
elif loss_name == "medse":
    criterion = lambda a, b: medse_loss(a, b, reduction="median")
elif loss_name == "gmadl":
    criterion = lambda a, b: gmadl_loss(a, b, reduction="mean")
elif loss_name == "imadl":
    criterion = lambda a, b: imadl_rebalanced_loss(a, b, reduction="mean")
elif loss_name == "dirhuber":
    criterion = lambda a, b: directional_huber_loss(a, b, reduction="mean")
elif loss_name == "hybrid":
    criterion = lambda a, b: hybrid_dir_huber_loss(a, b, reduction="mean")
else:
    raise ValueError(f"Unsupported loss: {loss_name}")
```

3. **创建新的实验运行脚本**：
   - `run_sanity_check_imadl.py`
   - `run_sanity_check_dirhuber.py`
   - `run_sanity_check_hybrid.py`

   （复制 `run_sanity_check_mse.py`，修改 `loss_name` 参数）

## 阶段2：Google Colab迁移

### 2.1 环境配置

**Colab硬件选项**（从你的截图）：
- CPU（基线，最慢）
- T4 GPU（免费，比CPU快~4倍）
- **v5e-1 TPU**（推荐用于本项目）
- A100/H100/L4 GPU（付费，快10-20倍）

**训练时间估算（单个实验）**：
- Apple M1 Pro（当前）：~15-20分钟/epoch × 20 epochs = **5-7小时**
- Colab T4 GPU：~3-4分钟/epoch × 20 epochs = **1-1.5小时**
- Colab TPU v5e：~2-3分钟/epoch × 20 epochs = **40-60分钟**
- Colab A100 GPU：~1-2分钟/epoch × 20 epochs = **20-40分钟**

**数据规模**：~82MB总计（5个CSV文件 × 15-20MB）

**推荐配置**：
- 第一轮测试：T4 GPU（免费，足够快）
- 正式实验：如果需要多次迭代，考虑升级到A100
- 6个实验 × 1.5小时 = 9小时（可在一个session内完成）

### 2.2 Colab工作流设计

**关键需求**（基于Colab_GAN撞南墙实录.md）：
1. 挂载Google Drive实现数据持久化
2. 每个epoch保存checkpoint防止数据丢失
3. 支持24小时超时后断点续训
4. 批量实验自动化运行

**Notebook结构**：

```python
# ============================================
# 第1部分：环境设置
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# 检测GPU
!nvidia-smi

# 安装依赖
!pip install torch pandas numpy matplotlib seaborn

# ============================================
# 第2部分：代码和数据同步
# ============================================
import os
import shutil

# 从Drive复制代码到Colab本地（加速读取）
drive_base = '/content/drive/MyDrive/FYP'
local_base = '/content/FYP'

# 复制代码
!cp -r {drive_base}/Model_Train {local_base}/
!cp {drive_base}/*.py {local_base}/

# 复制数据（一次性，约1分钟）
!cp {drive_base}/*.csv {local_base}/

# 切换工作目录
os.chdir(local_base)

# ============================================
# 第3部分：批量实验自动化
# ============================================
import subprocess
import json
import time
from pathlib import Path

# 实验配置
losses = ['mse', 'medse', 'gmadl', 'imadl', 'dirhuber', 'hybrid']
results =

# 输出目录（保存到Drive）
output_dir = f'{drive_base}/sanity_outputs'
model_dir = f'{drive_base}/models'

for i, loss_name in enumerate(losses):
    print(f"\n{'='*60}")
    print(f"实验 {i+1}/6: {loss_name.upper()}")
    print(f"预计耗时: 1-1.5小时（T4 GPU）")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 运行实验
    cmd = [
        'python', f'run_sanity_check_{loss_name}.py',
        '--test-months', '24',  # 扩展到24个月
        '--output-dir', output_dir,
        '--max-epochs', '20',
        '--batch-size', '1024'
    ]

    try:
        subprocess.run(cmd, check=True)

        # 保存模型checkpoint到Drive
        model_path = f'{model_dir}/{loss_name}_model.pt'
        # 假设模型在训练后保存为 'trained_model.pt'
        if os.path.exists('trained_model.pt'):
            shutil.copy('trained_model.pt', model_path)
            print(f"✓ 模型已保存到: {model_path}")

        # 读取结果
        summary_path = f'{output_dir}/sanity_summary_{loss_name}.json'
        with open(summary_path) as f:
            results[loss_name] = json.load(f)

        elapsed = time.time() - start_time
        print(f"\n✓ {loss_name.upper()} 完成！耗时: {elapsed/60:.1f}分钟")

    except subprocess.CalledProcessError as e:
        print(f"✗ {loss_name.upper()} 失败: {e}")
        results[loss_name] = {"error": str(e)}

# ============================================
# 第4部分：结果汇总
# ============================================
import pandas as pd

# 创建对比表
comparison_data = []
for loss, res in results.items():
    if 'error' not in res:
        comparison_data.append({
            'Loss': loss.upper(),
            'Sharpe': res.get('long_short_sharpe', float('nan')),
            'Cumulative Return': res.get('long_short_cumulative_return', float('nan')),
            'Avg R²': res.get('avg_r2', float('nan')),
            'Avg Long-Short': res.get('avg_long_short', float('nan'))
        })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('Sharpe', ascending=False)

# 保存对比表
comparison_path = f'{output_dir}/all_losses_comparison.csv'
df_comparison.to_csv(comparison_path, index=False)

print("\n" + "="*60)
print("所有实验完成！结果对比：")
print("="*60)
print(df_comparison.to_string(index=False))
print(f"\n完整结果已保存到: {output_dir}")

# ============================================
# 第5部分：可视化对比
# ============================================
import matplotlib.pyplot as plt

# Sharpe比率对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_comparison['Loss'], df_comparison['Sharpe'])
ax.set_xlabel('Loss Function')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio Comparison Across Loss Functions')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/sharpe_comparison.png', dpi=200)
plt.show()

# 累计收益对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_comparison['Loss'], df_comparison['Cumulative Return'])
ax.set_xlabel('Loss Function')
ax.set_ylabel('Cumulative Return')
ax.set_title('Cumulative Return Comparison')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/return_comparison.png', dpi=200)
plt.show()

print("\n✓ 可视化完成！")
```

### 2.3 Checkpoint保存机制

**修改 `sanity_check_core.py` 的 `train_model()` 函数**：

```python
def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MLPConfig,
    loss_name: str,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    checkpoint_dir: str = None,  # 新增参数
) -> MLP:
    """
    单次训练流程，支持checkpoint保存
    """
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # 选择损失函数
    if loss_name == "mse":
        criterion = lambda a, b: mse_loss(a, b, reduction="mean")
    elif loss_name == "medse":
        criterion = lambda a, b: medse_loss(a, b, reduction="median")
    elif loss_name == "gmadl":
        criterion = lambda a, b: gmadl_loss(a, b, reduction="mean")
    elif loss_name == "imadl":
        criterion = lambda a, b: imadl_rebalanced_loss(a, b, reduction="mean")
    elif loss_name == "dirhuber":
        criterion = lambda a, b: directional_huber_loss(a, b, reduction="mean")
    elif loss_name == "hybrid":
        criterion = lambda a, b: hybrid_dir_huber_loss(a, b, reduction="mean")
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    model.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(batch_y, preds)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == max_epochs:
            print(f"Epoch {epoch + 1}/{max_epochs} | Loss: {avg_loss:.6f}")

        # 保存checkpoint（每5个epoch或最后一个epoch）
        if checkpoint_dir and ((epoch + 1) % 5 == 0 or epoch + 1 == max_epochs):
            checkpoint_path = Path(checkpoint_dir) / f"{loss_name}_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    return model
```

### 2.4 迁移检查清单

- [ ] 在Google Drive创建文件夹结构：
  - `FYP/Model_Train/`
  - `FYP/sanity_outputs/`
  - `FYP/models/`
- [ ] 上传CSV数据文件到Drive（一次性，约5分钟）
- [ ] 上传代码文件到Drive
- [ ] 创建Colab notebook并测试环境
- [ ] 运行单个实验（MSE）验证流程
- [ ] 确认checkpoint保存到Drive成功
- [ ] 运行完整批量实验（6个损失函数）

## 阶段3：系统实验设计

### 3.1 实验配置

**固定设置**：
- 训练窗口：1990-01 至 1994-12（5年）
- 测试窗口：**1995-01 至 1996-12（24个月）** ← 从6个月扩展
- 特征集：X1（15维：动量+换手率）
- 网络架构：MLP [64, 32, 16] + ReLU + Dropout 0.2
- 组合策略：P2 Signal-Weighted（主要）+ P1 Equal（次要）

**对比的损失函数**（6个）：
1. **MSE**（基线）
2. **MedSE**（鲁棒基线）
3. **GMADL**（方向基线）
4. **Rebalanced Improved MADL**（新，修复精度）
5. **Directional Huber**（新，鲁棒+方向）
6. **Hybrid Directional-Huber**（新，主要贡献）

**评估指标**：

**预测质量**：
- MSE：均方误差
- MedSE：中位数平方误差
- R²：决定系数（解释度）
- **Directional Accuracy**（新）：`sign(y) == sign(ŷ)` 的比例
- **Sign Mismatch on Large |y|**（新）：当 `|y| > 75th percentile` 时的符号错误率

**组合表现**：
- Cumulative Return：累计收益
- Sharpe Ratio：夏普比率（年化）
- Max Drawdown：最大回撤
- Monthly Long/Short Returns：逐月多空收益

### 3.2 新增评估指标实现

**在 `sanity_check_core.py` 中添加**：

```python
def compute_directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算方向准确性指标
    """
    if y_true.size == 0:
        return {
            "directional_accuracy": float("nan"),
            "sign_mismatch_large_y": float("nan")
        }

    # 整体方向准确率
    sign_match = (np.sign(y_true) == np.sign(y_pred))
    dir_acc = float(np.mean(sign_match))

    # 大波动时的符号错误率
    threshold = np.percentile(np.abs(y_true), 75)
    large_y_mask = np.abs(y_true) > threshold

    if large_y_mask.sum() > 0:
        sign_mismatch_large = float(np.mean(~sign_match[large_y_mask]))
    else:
        sign_mismatch_large = float("nan")

    return {
        "directional_accuracy": dir_acc,
        "sign_mismatch_large_y": sign_mismatch_large
    }
```

**修改 `run_sanity_check()` 函数**：

```python
# 在月度循环中，计算指标后添加：
dir_metrics = compute_directional_metrics(y_month, preds)

month_records.append({
    "month": format_period(period),
    "sample_size": int(mask.sum()),
    label.lower(): loss_value,
    "r2": metrics["r2"],
    "directional_accuracy": dir_metrics["directional_accuracy"],  # 新增
    "sign_mismatch_large_y": dir_metrics["sign_mismatch_large_y"],  # 新增
    "long_return": port["long"],
    "short_return": port["short"],
    "long_short_return": port["long_short"],
})
```

### 3.3 批量执行策略

**Colab session管理**：
- 单个实验：~1-1.5小时（T4 GPU）
- 6个实验 × 1.5小时 = **9小时总计**
- 建议：晚上运行，避免中断
- 所有结果自动保存到Google Drive

**自动化脚本**（`run_all_experiments.py`）：

```python
#!/usr/bin/env python3
"""
批量运行所有损失函数实验
用法：python run_all_experiments.py --output-dir /path/to/output
"""

import argparse
import subprocess
import json
import time
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--test-months', type=int, default=24)
    parser.add_argument('--max-epochs', type=int, default=20)
    args = parser.parse_args()

    losses = ['mse', 'medse', 'gmadl', 'imadl', 'dirhuber', 'hybrid']
    results = {}

    print("="*70)
    print("批量实验开始")
    print(f"输出目录: {args.output_dir}")
    print(f"测试期长度: {args.test_months}个月")
    print(f"训练轮数: {args.max_epochs}")
    print("="*70)

    total_start = time.time()

    for i, loss in enumerate(losses):
        print(f"\n{'='*70}")
        print(f"实验 {i+1}/6: {loss.upper()}")
        print(f"{'='*70}\n")

        start_time = time.time()

        cmd = [
            'python', f'run_sanity_check_{loss}.py',
            '--test-months', str(args.test_months),
            '--output-dir', args.output_dir,
            '--max-epochs', str(args.max_epochs),
            '--batch-size', '1024'
        ]

        try:
            subprocess.run(cmd, check=True)

            # 读取结果
            summary_path = Path(args.output_dir) / f'sanity_summary_{loss}.json'
            with open(summary_path) as f:
                results[loss] = json.load(f)

            elapsed = time.time() - start_time
            print(f"\n✓ {loss.upper()} 完成！耗时: {elapsed/60:.1f}分钟")

        except subprocess.CalledProcessError as e:
            print(f"✗ {loss.upper()} 失败: {e}")
            results[loss] = {"error": str(e)}

    # 生成对比表
    comparison_data = []
    for loss, res in results.items():
        if 'error' not in res:
            comparison_data.append({
                'Loss': loss.upper(),
                'Sharpe': res.get('long_short_sharpe', float('nan')),
                'Cum. Return': res.get('long_short_cumulative_return', float('nan')),
                'Avg R²': res.get('avg_r2', float('nan')),
                'Avg Dir. Acc.': res.get('avg_directional_accuracy', float('nan')),
                'Avg Long-Short': res.get('avg_long_short', float('nan'))
            })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Sharpe', ascending=False)

    # 保存
    output_path = Path(args.output_dir) / 'all_losses_comparison.csv'
    df.to_csv(output_path, index=False)

    total_elapsed = time.time() - total_start

    print("\n" + "="*70)
    print("所有实验完成！")
    print(f"总耗时: {total_elapsed/3600:.2f}小时")
    print("="*70)
    print("\n结果对比：")
    print(df.to_string(index=False))
    print(f"\n完整结果保存在: {output_path}")

if __name__ == '__main__':
    main()
```

### 3.4 结果收集

**每个实验的输出**：
- `sanity_metrics_{loss}.csv`：逐月指标
- `sanity_summary_{loss}.json`：汇总统计
- `{loss}_loss_curve.png`：损失曲线
- `{loss}_returns_curve.png`：多空收益曲线
- `models/{loss}_model.pt`：训练好的模型checkpoint

**对比可视化**（`plot_comparison_all_losses.py`）：

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_all_comparisons(output_dir: str):
    """
    生成所有损失函数的对比图
    """
    output_dir = Path(output_dir)

    # 读取对比表
    df = pd.read_csv(output_dir / 'all_losses_comparison.csv')

    # 1. Sharpe比率对比
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in df['Sharpe']]
    ax.bar(df['Loss'], df['Sharpe'], color=colors, alpha=0.7)
    ax.set_xlabel('Loss Function', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Sharpe Ratio Comparison (24-Month Test Period)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpe_comparison.png', dpi=200)
    plt.close()

    # 2. 累计收益对比
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in df['Cum. Return']]
    ax.bar(df['Loss'], df['Cum. Return'] * 100, color=colors, alpha=0.7)
    ax.set_xlabel('Loss Function', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Cumulative Return Comparison', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'return_comparison.png', dpi=200)
    plt.close()

    # 3. R² vs Sharpe散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Avg R²'], df['Sharpe'], s=100, alpha=0.7)
    for i, loss in enumerate(df['Loss']):
        ax.annotate(loss, (df['Avg R²'].iloc[i], df['Sharpe'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Average R²', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Prediction Quality vs Portfolio Performance', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_vs_sharpe.png', dpi=200)
    plt.close()

    # 4. 方向准确率对比
    if 'Avg Dir. Acc.' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df['Loss'], df['Avg Dir. Acc.'] * 100, alpha=0.7)
        ax.set_xlabel('Loss Function', fontsize=12)
        ax.set_ylabel('Directional Accuracy (%)', fontsize=12)
        ax.set_title('Directional Accuracy Comparison', fontsize=14)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'directional_accuracy.png', dpi=200)
        plt.close()

    print(f"✓ 所有对比图已保存到: {output_dir}")

if __name__ == '__main__':
    import sys
    plot_all_comparisons(sys.argv[1] if len(sys.argv) > 1 else 'sanity_outputs')
```

## 阶段4：分析与论文撰写

### 4.1 研究问题

**RQ1：显式方向惩罚是否能改善纯误差型损失的交易表现？**
- 对比：MSE/MedSE vs GMADL/Improved MADL/Directional Huber
- 预期：方向感知损失在Sharpe和方向准确率上更优
- 关键指标：Sharpe ratio, Directional accuracy

**RQ2：鲁棒误差处理（Huber）是否比L2更适合股票收益预测？**
- 对比：MSE vs MedSE vs Directional Huber
- 预期：Huber在有异常值时更稳定
- 关键指标：MedSE, 大|y|时的表现

**RQ3：结合方向和鲁棒性的最优结构是什么？**
- 对比：Additive (Hybrid) vs 单独的方向/鲁棒损失
- 预期：Hybrid统一三个属性，表现最佳
- 关键指标：综合排名（Sharpe + Dir. Acc. + 稳定性）

### 4.2 预期叙事结构

1. **问题识别**：GMADL有已知缺陷（对称性、弱梯度、无精度惩罚）
2. **基线验证**：MedSE > MSE 证明鲁棒性在你的数据上有价值
3. **方案1**：Rebalanced Improved MADL 修复GMADL的精度问题
4. **方案2**：Directional Huber 扩展MedSE加入方向感知
5. **主要贡献**：Hybrid损失统一方向+精度+鲁棒性
6. **实证验证**：24个月测试期的系统对比

### 4.3 论文结构

**第1章：引言**
- 研究动机：截面股票预测的挑战
- 问题陈述：标准损失函数忽略方向信息
- 研究目标：设计平衡方向、精度、鲁棒性的损失函数
- 贡献：3个新损失函数 + 系统实证对比

**第2章：文献综述**
- 2.1 股票收益预测的机器学习方法
- 2.2 损失函数设计
  - MSE/MAE及其局限
  - GMADL/MADL家族
  - Huber损失与鲁棒回归
- 2.3 方向感知损失函数
- 2.4 研究空白：缺乏统一框架

**第3章：方法论**
- 3.1 问题设定
  - 截面收益预测任务
  - 多空组合构建
- 3.2 GMADL的缺陷分析
  - 对称性问题
  - 弱梯度问题
  - 忽略精度问题
- 3.3 提出的损失函数
  - 3.3.1 Rebalanced Improved MADL（公式+推导）
  - 3.3.2 Directional Huber（公式+推导）
  - 3.3.3 Hybrid Directional-Huber（公式+推导）
- 3.4 理论性质分析（基于导师文档第4节）
  - 收敛性：凸性、Lipschitz连续性
  - 一阶导数：平滑性、梯度行为
  - 鲁棒性：对异常值的敏感度

**第4章：实验设计**
- 4.1 数据
  - 数据源、时间范围、样本量
  - 特征工程（X1）
- 4.2 模型架构
  - MLP [64,32,16]
  - 超参数选择
- 4.3 实验设置
  - 训练/测试窗口
  - 组合策略（P2 Signal-Weighted）
  - 评估指标
- 4.4 实现细节
  - PyTorch实现
  - Google Colab GPU加速

**第5章：结果**
- 5.1 预测质量对比
  - MSE, MedSE, R²
  - 方向准确率
  - 大|y|时的符号错误率
- 5.2 组合表现对比
  - Sharpe ratio（主要指标）
  - 累计收益
  - 逐月收益分布
- 5.3 消融实验
  - 超参数敏感性（λ_dir, λ_mag, δ）
  - 加法 vs 乘法结构
- 5.4 案例分析
  - 选择典型月份（如1995-03）
  - 分析不同损失函数的预测差异

**第6章：讨论**
- 6.1 为什么Hybrid有效（或无效）
  - 方向项的作用
  - Huber的鲁棒性贡献
  - 权重平衡的重要性
- 6.2 与GMADL的对比
  - 修复了哪些问题
  - 仍存在的局限
- 6.3 实践意义
  - 对量化投资的启示
  - 超参数调优建议
- 6.4 局限性
  - 单一特征集（X1）
  - 静态训练窗口
  - 未考虑交易成本

**第7章：结论与未来工作**
- 7.1 主要发现
- 7.2 贡献总结
- 7.3 未来方向
  - X2/X3特征集对比
  - 滚动窗口回测（Step 4）
  - Ranking损失（3.2.5）
  - 集成方法

### 4.4 关键图表清单

**必须包含的图表**：
1. GMADL缺陷示意图（3个问题的可视化）
2. 6个损失函数的Sharpe对比（柱状图）
3. 累计收益曲线（6条线叠加）
4. 方向准确率对比（柱状图）
5. R² vs Sharpe散点图（揭示预测质量≠组合表现）
6. 逐月收益热力图（6×24矩阵）
7. 超参数敏感性曲线（λ_dir vs Sharpe）
8. 案例月份的预测分布对比（箱线图）

**可选的补充图表**：
9. 损失函数曲面图（3D可视化）
10. 梯度行为对比（不同误差范围）
11. 最大回撤对比
12. 多空收益的分布直方图

### 4.5 写作时间安排

**第6-7周：初稿**
- 第3章（方法论）：3天
- 第4章（实验设计）：2天
- 第5章（结果）：4天
- 第1-2章（引言+文献）：3天
- 第6章（讨论）：2天

**第8周：修订**
- 导师反馈整合
- 图表优化
- 语言润色

**第9-10周：定稿**
- 格式调整
- 参考文献整理
- 准备答辩PPT

## 阶段5：验证与扩展（可选）

### 5.1 鲁棒性检查

**时间泛化**：
- 测试不同时间段（1996-1997, 1998-1999）
- 验证损失函数在不同市场环境下的稳定性

**超参数敏感性**：
- λ_dir ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
- λ_mag ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
- δ ∈ [0.005, 0.01, 0.02, 0.05]
- 绘制热力图展示最优区域

**特征集对比**（如果时间允许）：
- X1 vs X2 vs X3
- 验证损失函数的优势是否依赖特征选择

### 5.2 未来工作（仅讨论，不实现）

**滚动窗口回测**（Step 4）：
- 每月重新训练模型
- 更接近真实交易场景
- 计算量大（需要更多GPU时间）

**Ranking损失**（导师文档3.2.5）：
- 混合GMADL + Ranking term
- 需要pairwise采样
- 复杂度O(N²)，训练不稳定

**集成方法**：
- 结合多个损失函数训练的模型
- Stacking或投票机制
- 可能进一步提升表现

## 时间线总览

**第1-2周**（2026-04-21 至 2026-05-04）：
- 实现3个新损失函数（`losses.py`）
- 修改训练脚本支持新损失（`sanity_check_core.py`）
- 设置Google Colab环境
- 上传数据到Drive
- 测试单个实验（MSE）验证流程

**第3周**（2026-05-05 至 2026-05-11）：
- 运行批量实验（6个损失函数，9小时）
- 收集结果并生成对比图表
- 初步分析结果，识别最佳损失函数

**第4-5周**（2026-05-12 至 2026-05-25）：
- 扩展分析：超参数敏感性
- 消融实验：加法 vs 乘法结构
- 案例分析：典型月份的预测差异
- 如需要，重新运行部分实验

**第6-8周**（2026-05-26 至 2026-06-15）：
- 撰写论文初稿
- 制作所有图表
- 整理参考文献

**第9-10周**（2026-06-16 至 2026-06-29）：
- 根据导师反馈修订
- 论文定稿
- 准备答辩材料

**缓冲时间**：预留1-2周应对意外情况

## 关键文件清单

### 核心实现

**损失函数**：
- `Model_Train/losses.py`（新增3个函数）

**训练脚本**：
- `sanity_check_core.py`（扩展损失支持 + 新增方向指标）
- `run_sanity_check_imadl.py`
- `run_sanity_check_dirhuber.py`
- `run_sanity_check_hybrid.py`

**批量自动化**：
- `run_all_experiments.py`（批量运行6个实验）

### Colab环境

**主Notebook**：
- `FYP_Semester2_Experiments.ipynb`（完整实验流程）

**数据位置**：
- 本地：`/Users/roucher/Documents/FYP/*.csv`
- Colab Drive：`/content/drive/MyDrive/FYP/*.csv`
- 结果：`/content/drive/MyDrive/FYP/sanity_outputs/`

### 分析脚本

**可视化**：
- `plot_comparison_all_losses.py`（跨损失对比图）
- `analyze_directional_accuracy.py`（方向指标分析）
- `plot_hyperparameter_sensitivity.py`（超参数热力图）

**结果汇总**：
- `sanity_outputs/all_losses_comparison.csv`（主对比表）
- `sanity_outputs/sharpe_comparison.png`
- `sanity_outputs/return_comparison.png`
- `sanity_outputs/r2_vs_sharpe.png`
- `sanity_outputs/directional_accuracy.png`

## 风险缓解

### Colab限制

**24小时超时**：
- 对策：每5个epoch保存checkpoint
- 实际影响：单个实验1.5小时，远低于24小时

**实例回收**：
- 对策：所有数据保存到Google Drive
- 验证：每次实验后检查Drive中的文件

**GPU配额限制**：
- 对策：优先使用T4（免费）
- 备选：如果配额不足，升级到Colab Pro（$9.99/月）

### 实验风险

**新损失可能不收敛**：
- 对策：从保守的超参数开始（λ=1.0, a=100）
- 监控：每5个epoch打印损失值，及时发现异常

**结果可能不如MedSE**：
- 对策：这仍是有价值的负面结果
- 论文角度：分析为什么不work，提供洞察

**24个月测试可能显示过拟合**：
- 对策：诚实报告，在讨论中分析原因
- 补充：增加不同时间段的测试

### 时间风险

**实验耗时超预期**：
- 对策：优先完成6个基础实验
- 可选：超参数敏感性和消融实验可后置

**论文写作时间不足**：
- 对策：边实验边写方法论章节
- 优先级：第3-5章（方法+实验+结果）> 第1-2章（引言+文献）

## 成功标准

### 最低可行论文（Minimum Viable Thesis）

✅ 3个新损失函数实现并测试
✅ 6个损失函数在24个月测试期的系统对比
✅ 清晰分析方向 vs 鲁棒性的权衡
✅ 完整的方法论和实验章节

### 优秀论文（Strong Thesis）

✅ Hybrid损失优于所有基线
✅ 清晰解释为什么有效（理论+实证）
✅ 超参数敏感性分析
✅ 鲁棒性检查（不同时间段）

### 卓越论文（Excellent Thesis）

✅ 关于金融损失函数设计的新洞察
✅ 可复现的Colab notebook供未来研究
✅ 扩展到X2/X3特征或滚动窗口
✅ 发表潜力（会议/期刊）

## 导师沟通计划

**第2周末**：展示损失函数实现和Colab环境
**第3周末**：汇报初步实验结果
**第5周末**：讨论深入分析方向
**第7周末**：提交论文初稿
**第9周末**：提交修订稿

## 参考文献（初步）

**GMADL/MADL**：
- Michańków et al. (2024) - 原始GMADL论文

**Huber损失**：
- arXiv:2108.12426 - Point forecasting with generalized Huber loss
- arXiv:1911.02088 - Alternative probabilistic interpretation of Huber loss

**股票预测**：
- Gu, Kelly, Xiu (2020) - Empirical asset pricing via machine learning
- 相关的因子投资文献

**损失函数理论**：
- Vapnik (1999) - Statistical learning theory
- 优化理论相关文献

---

**计划制定完成时间**：2026-04-21
**预计论文完成时间**：2026-06-29
**总工作量估算**：10周（约70天）
