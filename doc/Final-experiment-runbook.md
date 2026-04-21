---
title: Final Experiment Runbook
tags:
  - fyp
  - experiment
  - colab
  - runbook
  - loss-function
---

# 最终试验文档

> [!summary]
> 这份文档把当前仓库的最终实验线统一成一份可执行 runbook，覆盖：
> 1. 当前主实验步骤
> 2. 各脚本参数与固定配置
> 3. Google Colab 的标准运行方式
> 4. 断点续训与恢复机制
> 5. 最终需要交付的产物清单

## 1. 当前实验主线

> [!important]
> 当前仓库里真正应该作为最终主结果线的，不是 rolling 实验，而是 `static sanity check + batch comparison + Colab full run`。

### 1.1 当前建议采用的主线顺序

1. `run_step2_check.py`
   验证数据读取、预处理和特征矩阵是否正常。
2. `run_step3_grid_search.py`
   确认并锁定 MLP 最佳结构，结果保存到 `best_hyperparameters.txt`。
3. 单个 loss 的静态 sanity check
   入口脚本为 `run_sanity_check_*.py`，核心逻辑在 `sanity_check_signal_tilted.py`。
4. 批量对比
   使用 `run_all_experiments.py` 顺序跑完全部 loss。
5. Colab 完整实验
   在 Google Drive 持久化输出、checkpoint 和汇总结果。
6. 最终结果整理
   汇总 CSV、JSON、PNG，并写入论文 / 报告分析部分。

### 1.2 为什么 rolling 不是当前主线

- `run_step4_rolling.py` 已经实现了 rolling-window 接口，但它更适合后续扩展或补充实验。
- 导师当前要求的主线更偏向先证明：
  - 代码正确
  - loss 可训练
  - 月度指标可解释
  - 多空组合收益可比较
- 因此当前最稳妥的最终实验口径应为：
  固定训练窗口训练一次，然后在未来若干个月逐月预测和调仓。

## 2. 当前固定实验设定

### 2.1 数据与时间窗口

| 项目 | 当前设定 |
|---|---|
| 数据读取目录 | 仓库根目录 `*.csv` |
| 训练期 | `1990-01` 到 `1994-12` |
| 测试起点 | `1995-01` |
| 默认测试长度 | 单脚本默认 `6` 个月；批量主实验常用 `24` 个月 |
| 特征集 | `X1` |
| 目标变量 | `target_ret = r_{t+1}` |

### 2.2 特征工程设定

当前主实验只使用 `X1`，由 `Model_Train/features.py` 构建。

`X1` 包含以下 10 个原始滚动特征：

- `cr_1m`, `co_1m`
- `cr_3m`, `co_3m`
- `cr_6m`, `co_6m`
- `cr_9m`, `co_9m`
- `cr_12m`, `co_12m`

再加上原表中未被排除的基础列，当前 `best_hyperparameters.txt` 对应的输入维度为 `15`。

### 2.3 网络结构

当前锁定配置来自 `best_hyperparameters.txt`：

| 项目 | 设定 |
|---|---|
| `input_dim` | `15` |
| `hidden_dims` | `[64, 32, 16]` |
| `activation` | `relu` |
| `dropout` | `0.2` |

### 2.4 当前支持的 loss

主实验支持 7 个 loss：

- `mse`
- `medse`
- `gmadl`
- `imadl`
- `dirhuber`
- `hybrid_add`
- `hybrid_mul`

> [!note]
> 旧文档里有时会把最后一类简写成 `hybrid`，但当前代码实际区分为 `hybrid_add` 和 `hybrid_mul` 两个版本。最终文档和实验记录里应统一使用代码中的真实名字。

### 2.5 组合构建规则

每个测试月都执行一次调仓：

- 将当月股票按预测值排序
- 取 Top 10% 做 `Long`
- 取 Bottom 10% 做 `Short`
- 在 bucket 内基于预测值 z-score 构造权重
- 默认启用单票权重上限 `max_weight = 0.05`

对应输出指标包括：

- `mse`
- `medse`
- `r2`
- `directional_accuracy`
- `sign_mismatch_large_y`
- `long_return`
- `short_return`
- `long_short_return`
- `cumulative_long_short_return`

## 3. 脚本职责与推荐执行顺序

### 3.1 Step 2：数据与特征检查

命令：

```bash
python run_step2_check.py
```

作用：

- 检查 CSV 能否成功加载
- 检查 `prepare_panel_data()` 是否正常
- 检查 `X1/X2/X3` 是否都能构建
- 确认至少 `X1` 非空

推荐用途：

- 新机器首次运行
- Colab 环境第一次部署后做快速校验

### 3.2 Step 3：超参数锁定

命令：

```bash
python run_step3_grid_search.py
```

当前网格内容：

| 参数 | 候选值 |
|---|---|
| `hidden_dims` | `[32,16]`, `[64,32]`, `[128,64]`, `[64,32,16]` |
| `activation` | `relu`, `tanh` |
| `dropout` | `0.0`, `0.2` |
| `lr` | `1e-3`, `5e-4` |
| `batch_size` | `512`, `1024` |
| `max_epochs` | `20` |

输出：

- `best_hyperparameters.txt`

> [!important]
> 最终主实验应固定使用这份配置，不建议在最终对比阶段再次混入结构调参，否则 loss 对比就不再公平。

### 3.3 单个 loss 的静态 sanity check

示例命令：

```bash
python run_sanity_check_mse.py --test-months 24 --max-epochs 20 --batch-size 1024
```

所有 `run_sanity_check_*.py` 都只做一件事：

- 解析 CLI
- 将 loss 名称传入 `sanity_check_signal_tilted.py`

真正核心逻辑包括：

- 读取原始 CSV
- 构建 `X1`
- 按固定 5 年窗口训练一次
- 在未来若干个月逐月预测
- 月度调仓并记录多空收益
- 保存 CSV / JSON / PNG

### 3.4 批量实验

推荐命令：

```bash
python run_all_experiments.py \
  --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
  --best-config-path best_hyperparameters.txt \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --skip-existing \
  --resume-mode auto
```

作用：

- 顺序调用所有单 loss runner
- 自动跳过已完整产出的 loss
- 汇总每个 loss 的 summary
- 生成总对比表 `all_losses_comparison.csv`

## 4. 关键 CLI 参数说明

### 4.1 单个 loss runner 关键参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data-dir` | `.` | CSV 数据目录 |
| `--pattern` | `*.csv` | 读取哪些 CSV |
| `--lookback-months` | `12` | X1 lookback 月数 |
| `--train-start` | `1990-01` | 训练起始月 |
| `--train-end` | `1994-12` | 训练结束月 |
| `--test-start` | `1995-01` | 测试起始月 |
| `--test-months` | `6` | 测试连续月数 |
| `--best-config-path` | `best_hyperparameters.txt` | Step 3 最优配置 |
| `--max-epochs` | `20` | 训练 epoch 数 |
| `--batch-size` | `1024` | mini-batch 大小 |
| `--output-dir` | `sanity_outputs` | 输出目录 |
| `--seed` | `42` | 随机种子 |
| `--max-weight` | `0.05` | 单票最大权重，`None` 可关闭 |
| `--resume-mode` | `auto` | `auto / never / require` |
| `--checkpoint-dir` | `None` | checkpoint 根目录，默认在 `<output-dir>/checkpoints` |

### 4.2 批量脚本参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--losses` | 全部 7 个 loss | 逗号分隔的 loss 名单 |
| `--output-dir` | `sanity_outputs` | 所有实验共享输出目录 |
| `--test-months` | `24` | 传给单实验 runner |
| `--max-epochs` | `20` | 传给单实验 runner |
| `--batch-size` | `1024` | 传给单实验 runner |
| `--skip-existing` | 关闭 | 若完整结果已存在则跳过 |
| `--stop-on-error` | 关闭 | 一旦某个 loss 失败就停止 |
| `--resume-mode` | `None` | 原样转发给单实验 runner |
| `--checkpoint-dir` | `None` | 原样转发给单实验 runner |

## 5. 本地推荐执行流程

### 5.1 烟雾测试

```bash
python run_sanity_check_mse.py --test-months 2 --max-epochs 2
python run_sanity_check_medse.py --test-months 2 --max-epochs 2
python run_sanity_check_imadl.py --test-months 2 --max-epochs 2
pytest -q
```

目标：

- CLI 正常
- loss 不报错
- 输出文件齐全
- 测试通过

### 5.2 最终本地全量跑法

```bash
python run_all_experiments.py \
  --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --skip-existing \
  --resume-mode auto
```

## 6. Google Colab 标准执行方案

> [!important]
> Colab 是临时环境，代码可以丢，Google Drive 上的输出和 checkpoint 不能丢。所以最终实验必须把 `output-dir` 和 `checkpoint-dir` 指向 Drive。

### 6.1 推荐的 Drive 目录

```text
/content/drive/MyDrive/FYP/
├── data/
├── code/
├── outputs/
│   ├── sanity_outputs/
│   └── checkpoints/
└── logs/
```

### 6.2 Colab 初始化

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install -r /content/drive/MyDrive/FYP/code/requirements.txt
```

```python
import os
os.chdir('/content/drive/MyDrive/FYP/code')
```

### 6.3 单个 loss 最终运行模板

```python
!python run_sanity_check_mse.py \
    --data-dir /content/drive/MyDrive/FYP/code \
    --pattern "*.csv" \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto
```

### 6.4 批量最终运行模板

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto
```

> [!note]
> 当前仓库默认直接从 `--data-dir` 读取 `*.csv`。如果你把数据和代码分开存放，就需要把 `--data-dir` 指向真正放 CSV 的目录。

## 7. Colab 断点续训说明

### 7.1 当前断点机制保存了什么

每个 loss 都会在 checkpoint 目录下生成一个独立子目录，例如：

```text
checkpoints/
└── mse/
    ├── run_spec.json
    ├── train_state.json
    ├── progress.json
    └── train_checkpoint.pt
```

各文件作用如下：

| 文件 | 作用 |
|---|---|
| `run_spec.json` | 记录本次运行配置，防止换参数后误续训 |
| `train_state.json` | 记录已完成 epoch 数 |
| `progress.json` | 记录当前阶段和已完成月份 |
| `train_checkpoint.pt` | 模型权重和优化器状态 |

### 7.2 `resume-mode` 的含义

| 模式 | 行为 |
|---|---|
| `auto` | 有 checkpoint 就恢复，没有就从头开始 |
| `never` | 忽略已有状态，强制重跑 |
| `require` | 必须找到 checkpoint，否则报错 |

### 7.3 训练阶段中断后如何恢复

如果在第 `k` 个 epoch 后中断，重新执行完全相同的命令即可：

```python
!python run_sanity_check_mse.py \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/checkpoints \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto
```

预期行为：

- 自动读取 `train_checkpoint.pt`
- 读取 `train_state.json`
- 从下一个未完成 epoch 继续

### 7.4 评估阶段中断后如何恢复

如果训练已经完成，评估进行到某个月中断，则重新运行同一命令后：

- 已写入 `sanity_metrics_{loss}.csv` 的月份不会重复计算
- `progress.json` 中已经完成的月份会被跳过
- 程序直接从下一个未完成月份继续

### 7.5 断点续训必须满足的条件

以下内容必须保持一致，否则 `run_spec.json` 校验会失败：

- `loss_name`
- `data_dir`
- `pattern`
- `lookback_months`
- `train_start`
- `train_end`
- `test_start`
- `test_months`
- `best_config_path`
- `batch_size`
- `max_epochs`
- `seed`
- `max_weight`
- `model_config`

> [!warning]
> 也就是说，断点续训不是“随便改了参数继续跑”，而是“同一个实验实例恢复执行”。如果你想换测试月数、换 batch size、换模型结构，应使用新的输出目录或新的 checkpoint 目录。

## 8. 最终输出文件说明

### 8.1 每个 loss 的标准产物

每个 loss 跑完后，`output-dir` 中至少应出现 4 个核心文件：

- `sanity_metrics_{loss}.csv`
- `sanity_summary_{loss}.json`
- `{loss}_loss_curve.png`
- `{loss}_returns_curve.png`

### 8.2 批量实验额外产物

- `all_losses_comparison.csv`

### 8.3 文件内容解释

`sanity_metrics_{loss}.csv`：

- 月度粒度结果
- 适合画时序图和写逐月分析

`sanity_summary_{loss}.json`：

- 单个 loss 的总览指标
- 适合被批量脚本读取后汇总比较

`all_losses_comparison.csv`：

- 所有 loss 的最终对比总表
- 最适合直接做论文里的主结果表

## 9. 最终建议交付产物

> [!success]
> 如果你的目标是“给导师提交一套完整可复现实验结果”，建议最终至少交付以下内容。

### 9.1 代码与配置

- 当前仓库完整代码
- `requirements.txt`
- `best_hyperparameters.txt`
- 本文档 `doc/Final-experiment-runbook.md`

### 9.2 原始实验结果

- 7 个 loss 各自的 `sanity_metrics_{loss}.csv`
- 7 个 loss 各自的 `sanity_summary_{loss}.json`
- 7 个 loss 各自的两张曲线图
- `all_losses_comparison.csv`

### 9.3 报告中建议引用的核心图表

- 各 loss 的月度 `MSE` 曲线
- 各 loss 的月度 `Long-Short Return` 曲线
- 各 loss 的累计收益对比图
- 各 loss 的 Sharpe 排序表
- `R^2 / Directional Accuracy / Sharpe` 综合对比表

### 9.4 文字交付物

- 方法说明：每个 loss 的公式和动机
- 实验设定：时间窗、特征、模型、参数、调仓规则
- Colab 执行说明：如何运行、如何恢复
- 结果分析：谁在误差、方向性、收益上更优
- 结论：最终推荐的 loss 以及原因

## 10. 一套可直接执行的最终命令

### 10.1 本地

```bash
pytest -q
python run_all_experiments.py \
  --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
  --best-config-path best_hyperparameters.txt \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --skip-existing \
  --resume-mode auto
```

### 10.2 Colab

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --skip-existing \
    --resume-mode auto
```

> [!tip]
> 如果只想先验证 Colab 环境，先把 `--test-months` 改成 `2`，`--max-epochs` 改成 `2` 做 smoke test，通过后再切回最终参数。
