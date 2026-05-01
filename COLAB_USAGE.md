# Google Colab 使用指南：断点续训实战

本文档展示如何在 Google Colab 中使用断点续训功能，包括推荐的 Drive 路径、启动命令，以及如何验证恢复机制。

---

## 📁 推荐的 Google Drive 目录结构

```
/content/drive/MyDrive/FYP/
├── data/                          # 原始 CSV 数据
│   ├── stock_001.csv
│   ├── stock_002.csv
│   └── ...
├── code/                          # 代码仓库（从 GitHub clone）
│   ├── sanity_check_signal_tilted.py
│   ├── run_all_experiments.py
│   ├── Model_Train/
│   └── ...
├── outputs/                       # 实验输出（断点续训的关键）
│   ├── sanity_outputs/
│   │   ├── checkpoints/          # 断点状态文件
│   │   │   ├── mse/
│   │   │   │   ├── run_spec.json
│   │   │   │   ├── train_state.json
│   │   │   │   ├── progress.json
│   │   │   │   └── train_checkpoint.pt
│   │   │   ├── medse/
│   │   │   └── ...
│   │   ├── sanity_metrics_mse.csv
│   │   ├── sanity_summary_mse.json
│   │   └── ...
│   └── best_hyperparameters.txt  # Step 3 最佳超参数
└── logs/                          # 运行日志（可选）
```

---

## 🚀 Colab Notebook 启动模板

### 1. 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 安装依赖

```python
!pip install torch pandas numpy matplotlib scikit-learn
```

### 3. 切换到代码目录

```python
import os
os.chdir('/content/drive/MyDrive/FYP/code')
```

### 4. 单个 Loss 实验（带断点续训）

```python
# 首次运行
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto

# 断线后重新运行（自动从断点恢复）
# 命令完全相同，--resume-mode auto 会自动检测并恢复
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto
```

### 5. 批量运行所有 Loss（带断点续训）

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto
```

---

## 🧪 验证断点续训：故意中断测试

### 测试场景 1：训练阶段中断

**目标**：验证训练到第 5 个 epoch 时中断，重启后从第 6 个 epoch 继续。

#### Step 1: 启动训练（设置较少 epoch 便于观察）

```python
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --resume-mode auto
```

#### Step 2: 在训练过程中手动中断

- 方法 1：点击 Colab 的 "停止" 按钮
- 方法 2：在代码单元格中按 `Ctrl+M I`（中断执行）
- 方法 3：等待 Colab 自动断线（模拟真实场景）

#### Step 3: 检查断点文件

```python
import json
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints/mse')

# 查看训练状态
train_state = json.loads((checkpoint_dir / 'train_state.json').read_text())
print("训练状态:", train_state)
# 预期输出: {'completed_epochs': 5, 'max_epochs': 10, 'checkpoint_file': 'train_checkpoint.pt'}

# 查看进度状态
progress = json.loads((checkpoint_dir / 'progress.json').read_text())
print("进度状态:", progress)
# 预期输出: {'stage': 'training', 'completed_months': [], 'completed_epochs': 5}

# 检查模型权重文件是否存在
checkpoint_file = checkpoint_dir / 'train_checkpoint.pt'
print(f"模型权重文件存在: {checkpoint_file.exists()}")
print(f"文件大小: {checkpoint_file.stat().st_size / 1024:.2f} KB")
```

#### Step 4: 重新运行（验证恢复）

```python
# 完全相同的命令
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --resume-mode auto
```

**预期输出**：
```
Resuming MSE training from epoch 6.
Epoch 6/10 finished for MSE.
Epoch 10/10 finished for MSE.
```

---

### 测试场景 2：评估阶段中断

**目标**：验证评估到第 3 个月时中断，重启后跳过前 3 个月，从第 4 个月继续。

#### Step 1: 让训练完成，在评估阶段中断

```python
# 运行到评估阶段
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --resume-mode auto
```

#### Step 2: 在看到类似输出时中断

```
Epoch 10/10 finished for MSE.
Rebalancing monthly: Top 10% predictions -> Long, Bottom 10% -> Short.
Month 1995-01 | MSE | MSE 0.123456 | ...
Month 1995-02 | MSE | MSE 0.234567 | ...
Month 1995-03 | MSE | MSE 0.345678 | ...
[此时手动中断]
```

#### Step 3: 检查已完成的月份

```python
import pandas as pd
from pathlib import Path

# 查看 CSV 中已记录的月份
csv_path = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs/sanity_metrics_mse.csv')
df = pd.read_csv(csv_path)
print("已完成的月份:")
print(df['month'].tolist())
# 预期输出: ['1995-01', '1995-02', '1995-03']

# 查看进度文件
checkpoint_dir = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints/mse')
progress = json.loads((checkpoint_dir / 'progress.json').read_text())
print("\n进度状态:", progress)
# 预期输出: {'stage': 'evaluating', 'completed_months': ['1995-01', '1995-02', '1995-03'], 'completed_epochs': 10}
```

#### Step 4: 重新运行（验证跳过已完成月份）

```python
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/data \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/outputs/best_hyperparameters.txt \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --resume-mode auto
```

**预期输出**：
```
Resuming MSE training from epoch 10.
Epoch 10/10 finished for MSE.
Rebalancing monthly: Top 10% predictions -> Long, Bottom 10% -> Short.
Month 1995-04 | MSE | MSE 0.456789 | ...  # 直接从第 4 个月开始
Month 1995-05 | MSE | MSE 0.567890 | ...
Month 1995-06 | MSE | MSE 0.678901 | ...
```

---

### 测试场景 3：批量实验中断

**目标**：验证运行 7 个 loss 时，完成 3 个后中断，重启后跳过已完成的 3 个。

#### Step 1: 启动批量实验

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto
```

#### Step 2: 在完成部分 loss 后中断

假设完成了 `mse`, `medse`, `gmadl` 后中断。

#### Step 3: 检查已完成的 loss

```python
from pathlib import Path

output_dir = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs')

completed_losses = []
for loss in ['mse', 'medse', 'gmadl', 'imadl', 'dirhuber', 'hybrid_add', 'hybrid_mul']:
    summary_file = output_dir / f'sanity_summary_{loss}.json'
    if summary_file.exists():
        completed_losses.append(loss)

print("已完成的 loss:", completed_losses)
# 预期输出: ['mse', 'medse', 'gmadl']
```

#### Step 4: 重新运行（验证跳过已完成的 loss）

```python
# 完全相同的命令
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --test-months 6 \
    --max-epochs 10 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto
```

**预期输出**：
```
# 跳过 mse, medse, gmadl（因为 --skip-existing）
# 直接从 imadl 开始
Running experiment: imadl
...
Running experiment: dirhuber
...
Running experiment: hybrid_add
...
Running experiment: hybrid_mul
...
Completed losses: mse, medse, gmadl, imadl, dirhuber, hybrid_add, hybrid_mul
```

---

## 🔧 Resume Mode 参数说明

| 参数值 | 行为 | 使用场景 |
|--------|------|----------|
| `auto` | 如果存在断点文件且参数匹配，自动恢复；否则从头开始 | **推荐**：Colab 日常使用 |
| `never` | 忽略所有断点文件，强制从头开始 | 调试或重新实验 |
| `require` | 必须存在断点文件才能运行，否则报错 | 确保不会意外重新开始 |

---

## 📊 监控训练进度

### 实时查看训练状态

```python
import json
from pathlib import Path

def check_progress(loss_name='mse'):
    checkpoint_dir = Path(f'/content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints/{loss_name}')
    
    if not checkpoint_dir.exists():
        print(f"{loss_name} 尚未开始")
        return
    
    # 训练状态
    train_state_file = checkpoint_dir / 'train_state.json'
    if train_state_file.exists():
        train_state = json.loads(train_state_file.read_text())
        print(f"训练进度: {train_state['completed_epochs']}/{train_state['max_epochs']} epochs")
    
    # 评估状态
    progress_file = checkpoint_dir / 'progress.json'
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        print(f"阶段: {progress['stage']}")
        print(f"已完成月份: {len(progress['completed_months'])} 个")
        if progress['completed_months']:
            print(f"最新月份: {progress['completed_months'][-1]}")

# 使用示例
check_progress('mse')
```

### 查看所有 loss 的完成情况

```python
import pandas as pd
from pathlib import Path

output_dir = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs')
comparison_file = output_dir / 'all_losses_comparison.csv'

if comparison_file.exists():
    df = pd.read_csv(comparison_file)
    print("所有 loss 的 Sharpe 排名:")
    print(df[['loss', 'long_short_sharpe', 'long_short_cumulative_return']].to_string(index=False))
else:
    print("批量实验尚未完成")
```

---

## ⚠️ 常见问题

### Q1: 修改了超参数后能否继续恢复？

**A**: 不能。`run_spec.json` 会检测参数变化，如果不匹配会报错：
```
ValueError: Run spec mismatch for mse: /path/to/run_spec.json
```

**解决方案**：
- 使用 `--resume-mode never` 强制重新开始
- 或删除对应 loss 的 checkpoint 目录

### Q2: 如何清空某个 loss 的断点重新开始？

```python
import shutil
from pathlib import Path

loss_name = 'mse'
checkpoint_dir = Path(f'/content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints/{loss_name}')
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)
    print(f"已清空 {loss_name} 的断点文件")
```

### Q3: 如何清空所有断点重新开始？

```python
import shutil
from pathlib import Path

checkpoint_root = Path('/content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints')
if checkpoint_root.exists():
    shutil.rmtree(checkpoint_root)
    print("已清空所有断点文件")
```

### Q4: Colab 断线后 Drive 文件是否安全？

**A**: 安全。所有写入操作使用原子写入（atomic write）：
1. 先写入临时文件
2. `fsync` 确保落盘
3. `os.replace` 原子替换

即使在写入过程中断线，也不会损坏已有文件。

---

## 🎯 最佳实践

1. **始终使用 Drive 路径**：确保断点文件持久化
2. **使用 `--skip-existing`**：批量实验时避免重复计算
3. **定期检查进度**：使用上述监控脚本
4. **保留日志**：重定向输出到文件
   ```python
   !python run_all_experiments.py ... > /content/drive/MyDrive/FYP/logs/run.log 2>&1
   ```
5. **测试小规模**：先用 `--test-months 6 --max-epochs 5` 验证流程

---

## 📝 完整示例：从零开始

```python
# 1. 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆代码（首次运行）
!git clone https://github.com/your-repo/FYP.git /content/drive/MyDrive/FYP/code

# 3. 安装依赖
!pip install torch pandas numpy matplotlib scikit-learn

# 4. 切换目录
import os
os.chdir('/content/drive/MyDrive/FYP/code')

# 5. 运行批量实验（支持断点续训）
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs/checkpoints \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto

# 6. 查看结果
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/FYP/outputs/sanity_outputs/all_losses_comparison.csv')
print(df.sort_values('long_short_sharpe', ascending=False))
```

---

## 🚀 高级技巧：并行运行多个 Notebook

如果有多个 Colab 账号或 Pro 订阅，可以并行运行不同的 loss：

**Notebook 1**:
```python
!python run_sanity_check_mse.py --resume-mode auto ...
!python run_sanity_check_medse.py --resume-mode auto ...
```

**Notebook 2**:
```python
!python run_sanity_check_gmadl.py --resume-mode auto ...
!python run_sanity_check_imadl.py --resume-mode auto ...
```

**Notebook 3**:
```python
!python run_sanity_check_dirhuber.py --resume-mode auto ...
!python run_sanity_check_hybrid_add.py --resume-mode auto ...
!python run_sanity_check_hybrid_mul.py --resume-mode auto ...
```

最后在任意一个 Notebook 中运行：
```python
!python run_all_experiments.py --skip-existing --resume-mode auto ...
```

会自动汇总所有已完成的结果。

---

**恢复机制总结**：
- ✅ 训练按 epoch 恢复，保存模型和优化器状态
- ✅ 评估按 month 恢复，CSV 作为已完成月份的真源
- ✅ 批量实验按 loss 恢复，跳过已完成的 loss
- ✅ 原子写入保证文件安全，断线不会损坏数据
- ✅ 参数变化检测，避免不一致的恢复

现在可以放心在 Colab 上跑长时间实验了！🎉
