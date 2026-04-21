---
title: Experiment Diagrams
tags:
  - fyp
  - loss-function
  - mermaid
  - experiment-design
---

# Experiment Diagrams

> [!summary]
> 这份笔记把当前 FYP 的实验主线收拢成三张图：实验设计流程、MLP 网络结构、时间线。
> 主线顺序固定为：`sanity check -> batch run -> colab smoke -> full run`。

## 0. Loss 主线

| Slot | Loss | 角色 |
|---|---|---|
| 1 | `mse` | 传统基线 |
| 2 | `medse` | 鲁棒基线 |
| 3 | `gmadl` | 方向基线 |
| 4 | `imadl` | 改进 MADL |
| 5 | `dirhuber` | 鲁棒 + 方向 |
| 6 | `hybrid` | 方向 + 精度 + 鲁棒性 |
| 7 | `ranking` / stretch slot | 若时间允许的扩展位 |

> [!note]
> 这里保留 7 个实验槽位，便于和 FYP 的“7-loss 主线”对齐；当前主结果仍以 6 个核心 loss 为主。

## 1. 实验设计流程图

```mermaid
flowchart TB
    A["FYP experiment scope"] --> B["Lock inputs<br/>X1 features + fixed MLP + fixed portfolio rule"]
    B --> C["Define loss portfolio<br/>mse / medse / gmadl / imadl / dirhuber / hybrid / ranking slot"]
    C --> D["Sanity check<br/>1 loss at a time<br/>small sample + 2-3 epochs"]
    D --> E{"Sanity check passes?"}
    E -- "no" --> F["Fix formula / runner / metrics"]
    F --> D
    E -- "yes" --> G["Batch run<br/>all core losses locally"]
    G --> H["Colab smoke<br/>single loss + checkpoint save"]
    H --> I["Full run<br/>24-month test window"]
    I --> J["Consolidate outputs<br/>metrics tables / curves / notes"]
```

## 2. MLP 网络结构图

```mermaid
flowchart LR
    X["X1 features<br/>~15 dims"] --> L1["Dense 64"]
    L1 --> A1["ReLU"]
    A1 --> D1["Dropout 0.2"]
    D1 --> L2["Dense 32"]
    L2 --> A2["ReLU"]
    A2 --> D2["Dropout 0.2"]
    D2 --> L3["Dense 16"]
    L3 --> A3["ReLU"]
    A3 --> O["Output<br/>1 return forecast"]
```

> [!tip]
> 这就是当前固定的 baseline MLP：`[64, 32, 16] + ReLU + Dropout 0.2`。

## 3. 时间线可视化

```mermaid
flowchart LR
    T1["Setup<br/>2026-04-21 → 2026-04-28"] --> T2["Sanity check<br/>2-3 epochs / one loss at a time<br/>2026-04-29 → 2026-05-05"]
    T2 --> T3["Batch run (local)<br/>all core losses<br/>2026-05-06 → 2026-05-12"]
    T3 --> T4["Colab smoke<br/>single loss + checkpoint save<br/>2026-05-13 → 2026-05-19"]
    T4 --> T5["Full run<br/>24-month test window<br/>2026-05-20 → 2026-06-02"]
    T5 --> T6["Wrap-up<br/>tables / plots / write-up<br/>2026-06-03 → 2026-06-10"]
```

## 4. Source Files

- `doc/diagrams/experiment-flow.mmd`
- `doc/diagrams/mlp-structure.mmd`
- `doc/diagrams/timeline.mmd`

> [!info]
> 如果后续要在别处复用图，只需要直接引用 `doc/diagrams/*.mmd`。
