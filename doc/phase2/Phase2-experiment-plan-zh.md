# Phase 2 实验实施计划

> **给 Agent 工作者的提示：** 必需的子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务实施此计划。步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 构建一个干净的、可用于论文的 Phase 2 实验线，完成缺失的 24 个月基线，在统一固定配置下重新运行或整合 7 个损失函数的公平对比，并为入围的损失函数添加鲁棒性检查。

**架构：** Phase 2 分为三个工作流：首先冻结真实的实验配置并修复文档漂移，然后在独立的输出根目录中为所有七个损失函数生成一个权威的 24 个月对比集，最后对入围者加上固定基线运行一个小型鲁棒性矩阵。所有输出必须与现有的本地 `sanity_outputs/` 分离，因为该目录当前包含混合的 2 个月本地产物，不是最终论文表格的可靠来源。

**技术栈：** Python 3.10+, PyTorch, pandas, matplotlib, 现有的 `run_sanity_check_*.py` 运行器, `run_all_experiments.py`, Google Colab, Google Drive

---

## Phase 2 之前锁定的上下文

- Phase 1 为 `gmadl`、`imadl`、`dirhuber`、`hybrid_add` 和 `hybrid_mul` 生成了有效的 24 个月 Colab 运行，记录在 `/Users/roucher/Documents/FYP/phase1.md` 中。
- Phase 1 **没有**为 `mse` 和 `medse` 生成成功的 24 个月基线，因为第一批运行在缺失的 `best_hyperparameters.txt` 路径上失败了。
- 真实锁定的模型配置是 `/Users/roucher/Documents/FYP/best_hyperparameters.txt` 中的配置：

```text
Config: {'input_dim': 15, 'hidden_dims': [64, 32, 16], 'activation': 'tanh', 'dropout': 0.0}
```

- `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md` 仍然写着 `relu` 和 `0.2`，所以 Phase 2 必须将其视为文档错误，而不是真实来源。
- 本地目录 `/Users/roucher/Documents/FYP/sanity_outputs/` 当前包含所有七个损失函数的 2 个月本地输出，因此**不能**作为权威的 Phase 2 结果根目录重用。

## 文件结构

- 创建：`/Users/roucher/Documents/FYP/doc/Phase2-experiment-plan.md`
- 创建：`/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md`
- 修改：`/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md`
- 使用：`/Users/roucher/Documents/FYP/phase1.md`
- 使用：`/Users/roucher/Documents/FYP/best_hyperparameters.txt`
- 使用：`/Users/roucher/Documents/FYP/run_all_experiments.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- 使用：`/Users/roucher/Documents/FYP/plot_cumulative_returns.py`
- 使用：`/Users/roucher/Documents/FYP/plot_loss_strategy_cumreturn.py`
- 使用：`/Users/roucher/Documents/FYP/plot_signalweighted_sharpe.py`
- 使用：`/Users/roucher/Documents/FYP/plot_strategy_long_short.py`
- 使用：`/Users/roucher/Documents/FYP/tests/test_losses.py`
- 使用：`/Users/roucher/Documents/FYP/tests/test_run_all_experiments.py`
- 使用：`/Users/roucher/Documents/FYP/tests/test_sanity_check_signal_tilted.py`

### 需要预留的输出根目录

- 主要 24 个月对比：
  - `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs`
  - `/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints`
- 鲁棒性运行：
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed52_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed62_cap005`
  - `/content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap`

### Phase 2 入围者的选择规则

- 必须保留在每次对比中的固定基线：
  - `mse`
  - `medse`
- 在主要 24 个月运行中评估的方向型损失候选：
  - `gmadl`
  - `imadl`
  - `dirhuber`
  - `hybrid_add`
  - `hybrid_mul`
- 晋级到鲁棒性阶段的入围者：
  - 五个方向型候选中按 `long_short_sharpe` 排名前 2 的损失函数
  - 加上固定基线 `mse` 和 `medse`

### Phase 2 成功标准

- 一个干净的 24 个月 `all_losses_comparison.csv`，由七个损失函数在一个相同配置下构建
- `mse` 和 `medse` 的 24 个月基线存在且可以直接与五个方向型损失函数对比
- 一个鲁棒性表格，对比入围损失函数在不同种子和权重上限设置下的表现
- 运行手册和摘要文档反映实际锁定的配置和实际结果路径
- 图表和表格足以编写实验部分，无需阅读原始 Colab 日志

### 任务 1：冻结真实情况并修复文档漂移

**文件：**
- 创建：`/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md`
- 修改：`/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md`
- 使用：`/Users/roucher/Documents/FYP/phase1.md`
- 使用：`/Users/roucher/Documents/FYP/best_hyperparameters.txt`

- [ ] **步骤 1：起草一页 Phase 1 事实摘要**

创建 `/Users/roucher/Documents/FYP/doc/Phase2-result-summary.md`，包含以下部分：

```markdown
# Phase 1 事实摘要

## 确认成功的 24 个月运行
- gmadl
- imadl
- dirhuber
- hybrid_add
- hybrid_mul

## 确认失败的基线运行
- mse
- medse

## 失败原因
- 第一批 Colab 运行将 `--best-config-path` 指向 `/content/drive/MyDrive/FYP/code/best_hyperparameters.txt`，该文件在运行时不存在。

## 锁定的模型配置
- input_dim = 15
- hidden_dims = [64, 32, 16]
- activation = tanh
- dropout = 0.0

## Phase 1 方向型损失函数按记录的 24 个月 Sharpe 排名
1. imadl
2. gmadl
3. hybrid_mul
4. dirhuber
5. hybrid_add
```

- [ ] **步骤 2：更新运行手册以匹配真实锁定配置**

编辑 `/Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md` 中的网络部分，使表格显示为：

```markdown
| `input_dim` | `15` |
| `hidden_dims` | `[64, 32, 16]` |
| `activation` | `tanh` |
| `dropout` | `0.0` |
```

- [ ] **步骤 3：添加警告说明本地 `sanity_outputs/` 不是最终 Phase 2 来源**

在批量实验或交付物部分附近插入此说明：

```markdown
> [!warning]
> 当前本地仓库中的 `sanity_outputs/` 混有 2 个月本地短测产物，不应直接作为最终论文表格来源。Phase 2 必须使用独立输出目录重新生成或汇总 24 个月主实验结果。
```

- [ ] **步骤 4：验证文档编辑是唯一预期的更改**

运行：

```bash
git diff -- /Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md /Users/roucher/Documents/FYP/doc/Phase2-result-summary.md
```

预期：只出现配置更正、警告说明和 phase1 事实摘要。

- [ ] **步骤 5：提交仅文档的更改**

运行：

```bash
git add /Users/roucher/Documents/FYP/doc/Final-experiment-runbook.md /Users/roucher/Documents/FYP/doc/Phase2-result-summary.md
git commit -m "doc: lock phase2 experiment ground truth"
```

### 任务 2：构建一个权威的 7 个损失函数 24 个月主对比

**文件：**
- 使用：`/Users/roucher/Documents/FYP/run_all_experiments.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- 创建：`/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs`
- 创建：`/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints`

- [ ] **步骤 1：在任何长时间 Colab 作业之前运行本地测试套件**

本地运行：

```bash
pytest -q
```

预期：`/Users/roucher/Documents/FYP/tests/` 中的所有测试在 24 个月重新运行开始之前通过。

- [ ] **步骤 2：在 Colab Drive 中创建干净的 Phase 2 输出根目录**

在 Colab 中运行：

```python
from pathlib import Path

for path in [
    "/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs",
    "/content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints",
]:
    Path(path).mkdir(parents=True, exist_ok=True)
    print("READY", path)
```

预期：两个目录都打印 `READY` 并且为空或仅包含 Phase 2 文件。

- [ ] **步骤 3：在运行之前确认真实配置文件存在**

在 Colab 中运行：

```bash
cat /content/drive/MyDrive/FYP/code/best_hyperparameters.txt
```

预期：

```text
Config: {'input_dim': 15, 'hidden_dims': [64, 32, 16], 'activation': 'tanh', 'dropout': 0.0}
```

- [ ] **步骤 4：在全新的输出根目录中运行完整的 7 个损失函数批处理**

在 Colab 中运行：

```python
!python run_all_experiments.py \
    --losses mse,medse,gmadl,imadl,dirhuber,hybrid_add,hybrid_mul \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_main_24m/checkpoints \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --resume-mode auto
```

预期：七个完成的损失函数，没有缺失的摘要文件，以及一个新的 `all_losses_comparison.csv`。

- [ ] **步骤 5：验证每个损失函数都生成了四个必需的产物**

在 Colab 中运行：

```python
from pathlib import Path

root = Path("/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs")
losses = ["mse", "medse", "gmadl", "imadl", "dirhuber", "hybrid_add", "hybrid_mul"]
for loss in losses:
    required = [
        root / f"sanity_metrics_{loss}.csv",
        root / f"sanity_summary_{loss}.json",
        root / f"{loss}_loss_curve.png",
        root / f"{loss}_returns_curve.png",
    ]
    print(loss, all(path.exists() for path in required))
```

预期：每个损失函数都打印 `True`。

- [ ] **步骤 6：在本地快照权威的 7 个损失函数表格以供写作**

下载或同步文件后在本地运行：

```bash
cp /content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs/all_losses_comparison.csv /Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv
```

预期：`/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv` 存在并成为论文/报告的写作副本。

- [ ] **步骤 7：如果有意进行版本控制，则提交复制的权威对比表**

运行：

```bash
git add /Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv
git commit -m "results: add phase2 main 24m comparison table"
```

### 任务 3：入围入围者并定义鲁棒性矩阵

**文件：**
- 使用：`/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv`
- 创建：`/Users/roucher/Documents/FYP/doc/phase2_finalists.md`

- [ ] **步骤 1：按 Phase 2 主表 Sharpe 对五个方向型损失函数进行排名**

本地运行：

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv")
rows = list(csv.DictReader(path.open()))
directional = [r for r in rows if r["loss"] not in {"mse", "medse"}]
ranked = sorted(directional, key=lambda r: float(r["long_short_sharpe"]), reverse=True)
for row in ranked:
    print(row["loss"], row["long_short_sharpe"], row["long_short_cumulative_return"])
PY
```

预期：`gmadl`、`imadl`、`dirhuber`、`hybrid_add` 和 `hybrid_mul` 的 Sharpe 降序排名。

- [ ] **步骤 2：编写入围者备忘录**

创建 `/Users/roucher/Documents/FYP/doc/phase2_finalists.md`，结构如下：

```markdown
# Phase 2 入围者

## 固定基线
- mse
- medse

## 晋级的方向型入围者
- <按 Sharpe 排名第一的方向型损失函数>
- <按 Sharpe 排名第二的方向型损失函数>

## 被拒绝的方向型变体
- <剩余三个损失函数，每个附一行原因>

## 晋级规则
- 按 Phase 2 主要 24 个月 `long_short_sharpe` 排名
- 检查 `long_short_cumulative_return` 的符号
- 拒绝任何尽管误差指标可接受但收益曲线明显不稳定的候选
```

- [ ] **步骤 3：在重新运行任何内容之前锁定鲁棒性矩阵**

将此确切的实验矩阵写入同一文件：

```markdown
## 鲁棒性矩阵

种子：
- 42
- 52
- 62

权重上限设置：
- max_weight = 0.05
- max_weight = None

要运行的模型：
- mse
- medse
- finalist_1
- finalist_2
```

- [ ] **步骤 4：提交入围者选择备忘录**

运行：

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_finalists.md
git commit -m "doc: define phase2 finalists and robustness matrix"
```

### 任务 4：对基线加入围者运行鲁棒性检查

**文件：**
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_mse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_medse.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_gmadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_imadl.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_dirhuber.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_add.py`
- 使用：`/Users/roucher/Documents/FYP/run_sanity_check_hybrid_mul.py`
- 创建：`/content/drive/MyDrive/FYP/outputs/phase2_robustness/...`
- 创建：`/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv`

- [ ] **步骤 1：使用 `max_weight=0.05` 运行种子鲁棒性批处理**

对于每个选定的损失函数和 `42, 52, 62` 中的每个种子，在 Colab 中使用独立目录运行匹配的单损失运行器。模板：

```python
LOSS = "imadl"
SEED = 52

!python run_sanity_check_{LOSS}.py \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed{SEED}_cap005/{LOSS} \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed{SEED}_cap005/checkpoints/{LOSS} \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --seed {SEED} \
    --max-weight 0.05 \
    --resume-mode auto
```

预期：每个 `(loss, seed)` 对一个摘要 JSON。

- [ ] **步骤 2：在种子 42 下运行无上限对比**

对于每个选定的损失函数，运行：

```python
LOSS = "imadl"

!python run_sanity_check_{LOSS}.py \
    --output-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap/{LOSS} \
    --checkpoint-dir /content/drive/MyDrive/FYP/outputs/phase2_robustness/seed42_nocap/checkpoints/{LOSS} \
    --best-config-path /content/drive/MyDrive/FYP/code/best_hyperparameters.txt \
    --test-months 24 \
    --max-epochs 20 \
    --batch-size 1024 \
    --seed 42 \
    --max-weight None \
    --resume-mode auto
```

预期：在无上限组合设置下，每个选定损失函数一个摘要 JSON。

- [ ] **步骤 3：将鲁棒性摘要聚合到一个 CSV 中**

同步摘要 JSON 文件后在本地运行：

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

roots = [
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed42_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed52_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed62_cap005"),
    Path("/Users/roucher/Documents/FYP/phase2_sync/seed42_nocap"),
]
rows = []
for root in roots:
    for path in root.rglob("sanity_summary_*.json"):
        payload = json.loads(path.read_text())
        rows.append({
            "scenario": root.name,
            "loss": payload["loss"],
            "avg_mse": payload["avg_mse"],
            "avg_medse": payload["avg_medse"],
            "avg_r2": payload["avg_r2"],
            "avg_directional_accuracy": payload["avg_directional_accuracy"],
            "long_short_cumulative_return": payload["long_short_cumulative_return"],
            "long_short_sharpe": payload["long_short_sharpe"],
        })

out = Path("/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

预期：`/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv` 存在，每个 `(scenario, loss)` 对一行。

- [ ] **步骤 4：提交鲁棒性表格**

运行：

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv
git commit -m "results: add phase2 robustness comparison"
```

### 任务 5：生成可用于论文的表格和图表

**文件：**
- 使用：`/Users/roucher/Documents/FYP/plot_cumulative_returns.py`
- 使用：`/Users/roucher/Documents/FYP/plot_loss_strategy_cumreturn.py`
- 使用：`/Users/roucher/Documents/FYP/plot_signalweighted_sharpe.py`
- 使用：`/Users/roucher/Documents/FYP/plot_strategy_long_short.py`
- 使用：`/Users/roucher/Documents/FYP/doc/all_losses_comparison_phase2_main_24m.csv`
- 使用：`/Users/roucher/Documents/FYP/doc/phase2_robustness_results.csv`
- 创建：`/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md`

- [ ] **步骤 1：导出四个核心论文表格**

创建 `/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md`，包含以下标题：

```markdown
# Phase 2 写作结果

## 表 1：7 个损失函数主要 24 个月对比

## 表 2：方向型损失函数排名

## 表 3：入围者跨种子鲁棒性

## 表 4：种子 42 下的权重上限敏感性
```

- [ ] **步骤 2：从权威的 Phase 2 输出生成累计收益和多空图表**

针对权威的 Phase 2 输出根目录运行现有的绘图脚本。如果脚本需要路径编辑，请先修补它并将其输入根目录固定为：

```text
/content/drive/MyDrive/FYP/outputs/phase2_main_24m/sanity_outputs
```

预期：累计收益、多空收益、Sharpe 对比和策略对比各一个图表。

- [ ] **步骤 3：直接在表格下方编写结论要点**

将此结构附加到 `/Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md`：

```markdown
## 结论要点

- 按主表 Sharpe 的最佳整体损失函数：
- 最佳方向准确率损失函数：
- 跨种子最稳定的入围者：
- 移除权重上限的影响：
- 最终论文建议：
```

- [ ] **步骤 4：提交论文写作包**

运行：

```bash
git add /Users/roucher/Documents/FYP/doc/phase2_results_for_writing.md
git commit -m "doc: add phase2 writing bundle"
```

## 自我审查

- 规范覆盖：该计划涵盖文档修复、基线完成、权威 7 个损失函数重新运行、入围者选择、鲁棒性运行和最终论文产物。
- 占位符扫描：每个任务都有确切的文件路径和确切的命令。唯一的动态槽是 `finalist_1` 和 `finalist_2`，它们由任务 3 明确定义，而不是留下模糊。
- 类型一致性：整个过程中使用相同的七个规范损失函数名称，真实配置源始终是 `best_hyperparameters.txt`，所有最终写作产物都位于 `/Users/roucher/Documents/FYP/doc/` 下。
