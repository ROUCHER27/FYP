- 仓库路径：`~/Documents/FYP`
- 中期报告：`~/Documents/FYP/2253235_YirongYu_2025.pdf`
- 导师文档：`~/Documents/FYP/loss_function_design___to_do_v0.2.pdf`
- 模型代码：～/model-Train


---

**1. 你现在的研究进度，适合怎么收缩题目**

从你的中期报告看，你 Semester 1 已经完成了两块很重要的基础：

**A. 你已经把 GMADL 的问题讲清楚了**
报告里你已经明确指出了 3 个核心问题：

- **Symmetry issue**：奖励和惩罚过于对称，不符合 “avoid loss > chase profit”
- **Weak signal near critical points**：当 `ŷ -> 0` 时梯度很弱
- **Ignoring precision**：只看方向和 `|y|`，没有真正惩罚 `|y-ŷ|`

这和导师文档 3.2.1 / 3.2.2 的建议是**完全对得上的**。
所以你现在最合理的路线不是再发散，而是：

> **把 GMADL 的已知缺陷，收敛到 2~3 个可实现、可对比、可解释的新 loss 上。**

**B. 你已经有一个可靠 baseline：MedSE 明显强于 MSE**
你报告中的静态 sanity check 已经有很强的信号：

- P1 Equal: MedSE Sharpe **2.6773** vs MSE **0.3730**
- P2 Signal Weighted: MedSE Sharpe **3.2286** vs MSE **-1.4559**
- P3 Capped: MedSE Sharpe **3.2286** vs MSE **-1.5668**

这说明：

> **robustness to outliers** 在你的数据和任务里是真有价值的。

这正好支持你导师建议里 3.2.2 的 **Directional Huber** 路线。因为 Huber 本质上就是把 “MSE 的平方惩罚” 换成“中小误差二次、大误差线性”，是鲁棒性的自然延伸。

---

**2. 我看了你当前代码后，对可行性的判断**

你当前代码里：

- `Model_Train/losses.py` 已有
  `mse_loss / medse_loss / madl_loss / gmadl_loss`
- `sanity_check_core.py` 训练入口目前只支持
  `mse` 和 `medse`
- best config 已经固定为：
  - hidden dims = `[64, 32, 16]`
  - activation = `relu`
  - dropout = `0.2`

所以从工程角度看，**最适合你的下一步不是重新搭框架，而是：**

1. 在 `losses.py` 里新增 2~3 个 loss
2. 在 `sanity_check_core.py` 中扩展 `loss_name`
3. 沿用你现有 static sanity check 流程跑第一轮对比

也就是说，**你现在完全具备做 loss function ablation 的条件**。

---

**3. 导师建议的两个方向，分别值不值得做？**

---

**方向 1：3.2.1 Improved MADL**
导师建议公式核心是：

\[
L_i = (1-\sigma(a y_i \hat y_i)) |y_i|^b + c |y_i-\hat y_i|^d
\]

本质上它做了两件事：

- 第一项：保留 GMADL 的**方向惩罚**
- 第二项：加入真实的**幅度误差惩罚**

**这个方向为什么值得做**
因为它正面解决了你报告里提到的第 3 个问题：

- GMADL 只奖励方向，不重视精度
- Improved MADL 加了 `|y-ŷ|^d` 以后，至少不会再把 `ŷ=1.0` 看得比 `ŷ=0.1` 更好

**但它有一个很重要的潜在问题**
我按你导师文档里的形式做了一个简单数值检查。
例如：

- `y = 0.01`
- `ŷ = -0.1`

这时 3.2.1 里的两部分量级大概是：

- directional term ≈ `5.25e-05`
- magnitude term (若 d=2, c=1) ≈ `0.0121`

也就是说：

> **方向项只占幅度项的 0.4% 左右**

这很关键。说明如果直接用 3.2.1 原公式、且 `c=1,d=2`，那么模型优化时很可能**几乎退化成 MSE/L2 loss**，方向信息反而被淹没。

**所以我对 3.2.1 的判断是**
**值得做，但不能直接照抄。**
你真正可以写成论文贡献的是：

> “We refine the improved MADL by rebalancing the directional and magnitude components so that the directional penalty remains active under small-return regimes.”

**推荐你把它做成这个版本**
**版本 A：加权加法型**
\[
L_i = \lambda_{dir} \cdot D_i + \lambda_{mag}\cdot |y_i-\hat y_i|^d
\]
其中
\[
D_i = (1-\sigma(a y_i\hat y_i)) \cdot \tilde w_i
\]

这里 `\tilde w_i` 不建议直接用 `|y|^b`，因为你月收益率往往很小，方向项会特别弱。
更合理的是：

- 用 batch normalization 形式：
\[
\tilde w_i = \frac{|y_i|^b}{\text{mean}(|y|^b)+\epsilon}
\]
或者
- 直接不要 `|y|^b`，先做一个更干净的 directional penalty baseline

**结论**
**3.2.1 很适合做你的第一个新 loss。**
但论文里最好强调：你不是机械复现，而是做了**scale rebalancing**。

---

**方向 2：3.2.2 Directional Huber**
导师建议：

\[
L_i = pen(y_i,\hat y_i)\cdot H_\delta(y_i-\hat y_i)
\]
其中
\[
pen = \frac12(1-\tanh(a y_i\hat y_i))
\]

**这个方向为什么很适合你**
因为它和你当前结果天然呼应：

- 你已经证明 **MedSE > MSE**
- Huber 正是鲁棒误差族里的经典方案
- 所以你可以把这条线写成：

> “Since MedSE already showed that robustness is beneficial in our setting, a directional Huber loss is a natural next step that combines directional awareness with outlier robustness.”

**这个方向的优势**
1. **理论上比 GMADL 更像一个正常 loss**
   - 非负
   - 有明确 minimum
   - 优化器更容易处理

2. **比 MSE 更稳**
   - 大误差不会像平方项那样爆炸
   - 对极端 return / outlier 更鲁棒

3. **和你现有 MedSE 结果很容易形成 narrative**
   - MSE → MedSE 说明 robustness 重要
   - GMADL → Directional Huber 说明 direction 也重要
   - 于是你得到一个“robust + directional”的自然组合

**它的风险**
如果你直接用乘法形式：

\[
pen \times H_\delta(e)
\]

可能会出现一个问题：

> Huber 主体仍然主要由误差 `e=y-\hat y` 决定，而方向项只是乘了一个 `0~1` 的权重，方向惩罚未必够强。

所以我更推荐你不要只做“原版 directional huber”，而是把它作为**中间 baseline**。

---

**4. 真正最有潜力的，是把 3.2.1 和 3.2.2 合起来**

你导师也提到：
> “考虑有没有方法把这两个加到一起，或者修改一下设计。”

我觉得这正是你最值得做的主线。

---

**5. 我建议你优先做的 3 个候选 loss**

---

**候选 1：Rebalanced Improved MADL**
最容易落地，最适合作为 first new baseline。

\[
L_i = \lambda_{dir}(1-\sigma(a y_i\hat y_i))\tilde w_i + \lambda_{mag}|y_i-\hat y_i|^2
\]

**你可以卖点写成**
- 修复 GMADL 的 sign/reward inversion
- 显式加入 magnitude accuracy
- 通过 reweighting 防止方向项被小收益率淹没

**风险**
- 如果调参不好，会退化成 MSE
- 论文 novelty 中等，但容易解释

---

**候选 2：Directional Huber**
作为 robust-direction baseline，非常适合和 MedSE 接起来。

\[
L_i = \lambda_{dir} \cdot pen(y_i,\hat y_i) + \lambda_{hub}\cdot H_\delta(y_i-\hat y_i)
\]

我建议你先用**加法型**，而不是乘法型。
因为加法型更容易解释，也更容易调。

其中
\[
pen(y_i,\hat y_i)=1-\sigma(a y_i\hat y_i)
\]
或
\[
pen(y_i,\hat y_i)=\frac12(1-\tanh(a y_i\hat y_i))
\]

**为什么我更推荐加法**
- 方向错时，penalty 可以直接增大
- same sign 但 magnitude 很差时，Huber 仍然有效
- 两个效应更清晰可拆解

---

**候选 3：Hybrid Directional Huber-MADL**
这是我认为**最可能成为你最终主模型**的方向。

**方案 A：Additive hybrid**
\[
L_i = \lambda_1 D_i + \lambda_2 H_\delta(e_i)
\]
其中
- \(D_i = (1-\sigma(a y_i\hat y_i))\tilde w_i\)
- \(e_i = y_i-\hat y_i\)

这是最稳的主推方案。

**方案 B：Multiplicative hybrid**
\[
L_i = (1+\lambda D_i)\cdot H_\delta(e_i)
\]

含义是：

- magnitude error 仍由 Huber 控制
- 若方向错，则整个误差被额外放大

这个形式很有金融解释：

> “wrong-direction errors are not just errors; they are economically more costly errors.”

**我对两者的建议**
- **论文主模型**：先推 additive hybrid
- **附加实验**：再试 multiplicative hybrid

因为 additive 更容易写理论和实验解释。

---

**6. 我不建议你现在优先做的方向**

**不建议优先做：Ranking loss**
虽然导师文档 3.2.5 提到 ranking，而且从投资上很合理，但对你当前阶段不划算：

- 你现在还没有把 3.2.1 / 3.2.2 跑通
- ranking 会引入 pairwise sampling、复杂度、训练不稳定
- 写起来也更重，容易把本科论文搞得过大

**更好的做法**：
把 ranking loss 放成 discussion / future work。

你中期报告原计划里有 “Hybrid Ranking-Aware Loss”，但现在导师已经明显在帮你缩题。
我建议顺着这个缩题思路走。

---

**7. 一个很关键的研究判断：你论文最好的主线是什么？**

我觉得你现在最好的主线不是：

- “我发明了一个超级新的 loss”

而是：

> **从 GMADL 的已知缺陷出发，逐步构建一个兼顾 direction、precision、robustness 的 loss family，并在固定网络与固定 portfolio pipeline 下进行系统比较。**

这个 narrative 对本科论文非常强，因为它：

1. 有清楚的问题定义
2. 有导师文档支持
3. 有你中期报告积累
4. 有现成代码能做实验
5. 有学术上可解释的对比链条

你可以形成这样一条线：

- **GMADL**：有 direction，但 reward structure 有问题
- **Improved MADL**：修复方向+精度
- **Directional Huber**：加入 robustness
- **Hybrid loss**：direction + precision + robustness 三者统一

这比直接跳去 ranking 更完整。

---

**8. 我建议你的实验设计，尽量别一开始就太大**

**第一轮实验**
固定现有设置不变：

- 数据：你已经在用的 static sanity check
  - train: 1990-1994
  - test: 1995.01-1995.06
- feature：先只用 X1
- network：`[64,32,16]`, relu, dropout 0.2
- portfolio：先保留 P1 Equal，必要时再加 P2/P3

**对比对象**
建议第一轮只跑 6 个：

1. MSE
2. MedSE
3. GMADL
4. Rebalanced Improved MADL
5. Directional Huber
6. Hybrid Directional-Huber

**指标**
除了你已有的：

- MSE
- MedSE
- \(R^2\)
- cumulative return
- Sharpe

再加两个非常关键的：
- **Directional Accuracy**
  \[
  \mathbb{1}[\text{sign}(y)=\text{sign}(\hat y)]
  \]
- **Sign Mismatch Rate on large-|y| observations**

这个特别重要，因为你的 loss 设计核心就是：
> “错方向在大波动月份更贵”

---

**9. 你论文里最值得写出来的“研究问题”**

我帮你整理成更论文化的形式：

**RQ1**
在固定模型与特征下，加入显式 directional penalty 是否能改善纯误差型 loss 的交易表现？

**RQ2**
在方向敏感 loss 中，引入 robust error term（Huber）是否比 L2 error term 更适合股票收益预测？

**RQ3**
在你的数据集上，最优 loss 是否应采用：
- 方向项 + 幅度项 的加法结构，
还是
- 方向项 × Huber 的乘法结构？

这 3 个问题就足够写一篇本科 thesis 的核心实验部分了。

---

**10. 初步文献/理论支撑**

我顺手做了一个初步检索，能给你一些理论支撑方向：

1. **GMADL / MADL line**
   - 你已经在报告里复现并分析了这条线，足够作为起点

2. **Huber / generalized asymmetric Huber**
   - arXiv: **2108.12426**
     *Point forecasting and forecast evaluation with generalized Huber loss*
     说明 Huber 及其 asymmetric variants 在 forecast evaluation 上是有理论基础的
   - arXiv: **1911.02088**
     *An Alternative Probabilistic Interpretation of the Huber Loss*
     可支持 Huber 的鲁棒性与参数解释

3. **ranking for stock selection**
   - arXiv 检索里有
     *On Evaluating Loss Functions for Stock Ranking: An Empirical Analysis With Transformer Model*
   这能说明 ranking 方向是合理 future work，但你现在不必主攻

---

**11. 我给你的结论：最值得做的方向排序**

**第一优先级**
**Rebalanced Improved MADL**

理由：
- 最贴合导师 3.2.1
- 最直接修复 GMADL 缺陷
- 代码实现最容易

**第二优先级**
**Directional Huber**

理由：
- 和你现有 MedSE 结果最一致
- 容易形成 “robustness matters” 的论证链

**第三优先级**
**Hybrid: Directional term + Huber term**

理由：
- 最可能成为最终主模型
- 最符合导师“把两者结合起来”的建议
- 最适合做 thesis 的 main contribution

**暂缓**
**Ranking-aware loss**

理由：
- 太重
- 不适合当前阶段先做
- 更适合写成 extension/future work

---

**12. 如果你要我给一个最实用的下一步方案**

我会建议你把题目收缩成下面这个版本：

> **Design and empirical comparison of robust directional loss functions for neural-network-based equity strategies**

并且只做这 3 个新 loss：

1. `imadl_rebalanced`
2. `directional_huber`
3. `hybrid_dir_huber`

---

如果你愿意，我下一步可以直接继续帮你做这两件事之一：

1. **帮你把这 3 个 loss 的论文级公式和变量定义写出来**
2. **直接基于你这个仓库的代码结构，给你设计一个最小实现方案（losses.py + sanity_check_core.py 怎么改）**


# 训练环境迁移
当前代码～/Model_Train 完全跑在本地，我用的apple m1pro 芯片
现在我需要迁移到google colab上用gpu跑，文章参考：https://zhuanlan.zhihu.com/p/479544866
使用agent-reach阅读
![[截屏2026-04-21 13.48.45.png]]
- 在你列计划大纲时把迁移过程也加入进去，帮我估计一下使用各种芯片的训练耗时
- 安排合理计划，柔和两部分工作做完整计划。因为用到云端计算，是不是可以安排每轮实验自动进行，例如第一轮五个function，一次性跑完并收集结果，保存数据
