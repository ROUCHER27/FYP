# 1. baseline improvement
### 建议：让 Baseline 反复出现作为锚点

|在哪里加|加什么|效果|
|---|---|---|
|**Slide 2（Background）**|"Under MSE, the best you get in this protocol is a negative Sharpe — the portfolio actually loses money."|一开始就建立 "MSE 不行" 的印象|
|**Slide 7（Loss Design）结尾**|"Remember, MSE gives Sharpe minus 0.46. Everything I design next is trying to beat that."|过渡到结果时重新植入锚点|
|**Slide 12（Phase 2）**|"A3 reaches 0.57 — already more than double what any pure regression or directional baseline achieves."|用倍数说明|
|**Slide 13（Gamma07）**|"Gamma07 mean Sharpe is 0.92 — that is roughly **20 times** the best regression baseline MedSE, and **twice** the best single-seed hybrid."|**这是最关键的一句**，让考官直观感受到进步幅度|
|**Slide 16（Cumulative Paths）**|"While MSE loses 11% over two years, gamma07 gains 28%."|用正负对比收尾|
|**Slide 17（回答 RQ）**|"RQ1: loss choice takes you from minus 0.46 to plus 0.92 — a swing of 1.38 Sharpe units."|用一个数字总结全部改进| 
# 2.dropout different
"Activation 和 dropout 的差异会影响 Sharpe 的绝对数值，但不太可能改变 gamma 之间的相对排序。原因是 variance penalty 的最优强度遵循一个非单调规律——太小不稳定、太大丢信号——这个规律是 penalty 本身的数学性质，不依赖于中间层用什么 activation。Phase 3 内部的排序结论——gamma07 是最佳平衡点——在这个扰动量级下是安全的。" 
我决定用这个说辞，告诉我应该

**Slide 6（原文）：**

> "Phase 1 and Phase 2 use ReLU with dropout 0.2. Phase 3 keeps the same layer widths but uses tanh with dropout 0.0. Therefore, within-phase comparisons are the strongest evidence. Cross-phase comparisons are useful for design interpretation, but I do not treat them as exact single-factor improvement claims."


# 3. 不合理 / 不准确之处

|位置|问题|建议修改|
|---|---|---|
|**Slide 7**|"D = [1 - sigmoid(a y y_hat)] \|y\|^b / (batch mean of \|y\|^b + epsilon)" — 描述中漏了关键信息：这里的 `a=100, b=2` 具体在哪个variant中？在脚本上下文不够清晰|在公式后补一句 "with a equals 100 and b equals 2 in the implementation"|
|**Slide 7**|M2-robust extension 的公式写为 "L_M2-robust = L_M2 + gamma Var(y_hat)"，但根据 Chapter 3，L_M2 的 λ_dir = **5.0**（M2 variant），然而实际 Phase 3 代码中 M2-robust 的 base 是 λ_dir = **2.0**（与M1一致）。论文原文说得很清楚是 λ_dir=2.0。|脚本中应明确说 "L_M2 here uses lambda_dir equal to 2" 或者直接避免说"M2"以免混淆（因为 Phase 2 的 M2 variant 用的是 λ_dir=5）|
|**Slide 11**|"hybrid_mul_m1, with Sharpe 0.4435 and cumulative return plus 5.09 percent" — 这是 Phase 1 baseline table 中的值，但 hybrid_mul_m1 严格来说不是 "baseline"，它已经是 hybrid 了|措辞上应注意区分："the best baseline row" 这个说法有误导性。建议说 "the best-performing row in Phase 1"|
|**Slide 13**|"Gamma03 is unstable, with mean Sharpe 0.3234 and CV 1.0570" — CV > 1 意味着 standard deviation 大于 mean，但脚本没有提示这意味着某些 seed 可能是负 Sharpe|补充一句 "meaning some seeds produce negative Sharpe" 让考官理解 CV 的含义|
|**Slide 10**|"449,018 observations and 10,987 unique securities" 这是 source file 包含 61 个月（含1989-12）的数据，实际 training 用的是 60 个月。直接说 "in the training period" 略有不精确|改为 "the training-era source file contains 449,018 rows covering 10,987 unique securities across 60 training months"|

# 4. 口语化提升
|          |                                                                                                                                                            |                                                                                                                                                          |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Slide 4  | "conditional on a fixed prediction pipeline, can loss-function design improve the portfolio signal?"                                                       | 口语中 "conditional on" 很生硬。改为 "given a fixed pipeline, can loss design alone improve the portfolio signal?"                                                |
| Slide 5  | "I give a bounded recommendation"                                                                                                                          | "bounded" 不是自然口语。改为 "I give a recommendation with clear limits" 或 "a qualified recommendation"                                                           |
| Slide 6  | "The 15 inputs come from feature set X1. There are 10 engineered columns: cumulative return and cumulative turnover at 1, 3, 6, 9, and 12-month horizons." | 列举 5 个数字节奏太快。改为 "There are 10 engineered features — cumulative return and cumulative turnover, each at five lookback windows from 1 month to 12 months." |
| Slide 7  | "D(y, y_hat) is the directional gate" 然后立刻给公式                                                                                                              | 在presentation中念公式是大忌。建议改为指向 slide 上的公式："The directional gate D, shown here on the slide, does three things..." 然后口头只描述含义                                 |
|          |                                                                                                                                                            |                                                                                                                                                          |
| Slide 13 | "So gamma07 is not selected because it maximises one metric. It is selected because it gives the best joint Sharpe-stability profile."                     | 这句很好但口语中 "joint Sharpe-stability profile" 偏术语。改为 "It is selected because it balances high Sharpe and low seed sensitivity — the best of both."           |
| Slide 15 | "The normalisation probe equalises component scales and re-runs three leading candidates across three seeds."                                              | "equalises component scales" 口语中不直观。改为 "forces the two loss components to have equal scale, then re-runs the experiments."                               |
| Slide 17 | "So the answers are not universal claims, but they are clear within this evidence boundary."                                                               | 改为 "These are not universal claims — they hold within this specific protocol."                                                                           |
| Slide 19 | "The main takeaway is that loss-function design should be treated as a first-class design variable for portfolio-oriented prediction."                     | 口语中 "first-class design variable" 比较拗口。改为 "The takeaway is: how you define the loss function matters as much as how you build the model."                |

# 5. slide7 太长
**Slide 7 — Loss Function Families**

> This is the key methodology slide. There are four loss families, shown here from simple to most complex.
> 
> **Family one: regression.** MSE is the standard baseline — it minimises squared error. MedSE replaces the mean with the median, so it is more robust to outliers. But neither of them cares about prediction direction.
> 
> **Family two: directional.** MADL and GMADL reward the model when predicted and realised returns share the same sign. This is closer to what a portfolio actually needs. But pure directional losses can lose control over prediction scale — the model may predict the right direction at a completely wrong magnitude.
> 
> **Family three: additive hybrid.** As shown in the formula here — we simply add a directional penalty and a Huber magnitude term together. It works, but the two components can fight each other at different scales.
> 
> **Family four: multiplicative hybrid — my main design focus.** The formula is here. The intuition is simple: the Huber term is the backbone that controls magnitude. The directional gate — D — acts as a multiplier. When direction is correct, D is near zero, and the loss is just Huber. When direction is wrong, D amplifies the Huber loss. So wrong-direction predictions on large-return stocks get penalised the most.
> 
> One more extension — and I will show the results on this later in Slide 13 — we add a prediction-variance penalty controlled by a parameter gamma. Gamma controls how much the model is allowed to spread its predictions apart. Too little gamma means instability across seeds; too much gamma compresses the signal. The experiment scans gamma from 0.3 to 1.5 to find the sweet spot.
> 
> To summarise: we move from MSE, which ignores direction, through directional losses that ignore magnitude, to a multiplicative hybrid that handles both — and then add variance control on top.


# 6. 实证分析
#### Slide 11（Phase 1 Baseline）— 加一句 insight

当前结尾：

> "This suggests that combining a directional component with robust magnitude control gives a better ranking signal than MSE or pure directional loss alone."

**建议加：**

> "Notice the key decoupling: GMADL has the worst R-squared in the table — minus 7 billion — but its portfolio Sharpe is positive. **This tells us the portfolio trades ranks, not calibrated values.** Once you accept this, the question becomes: how do we design a loss that produces better ranks? That is exactly what the next slides answer."

这句话的作用：**把 Phase 1 的观察变成后续所有设计的逻辑起点。**

---

#### Slide 12（Phase 2）— 解释为什么有些崩了

当前只说 "A5 collapses to -0.41, M3 collapses to -0.97"。

**建议加：**

> "Why do A5 and M3 collapse? In both cases, the directional weight is too large relative to the magnitude backbone. The model starts chasing sign-correctness so aggressively that it distorts the ranking signal. This is why the later gamma family adds explicit variance control — to prevent this kind of over-correction."

这句话的作用：**把 Phase 2 的失败案例变成 Phase 3 设计的动机。**

---

#### Slide 13（Gamma Refinement）— 加 baseline 锚 + design insight

当前：

> "Gamma07 achieves mean Sharpe 0.9156..."

**建议改为：**

> "Remember MSE gives Sharpe minus 0.46. Gamma07 reaches plus 0.92 — a swing of nearly 1.4 Sharpe units, with the lowest CV in the table.
> 
> Why does gamma07 work? The intuition is: gamma 0.7 allows enough prediction spread to preserve ranking differences between stocks, but not so much that the model's output becomes unstable across seeds. Below 0.7, the model is under-regularised and seed-sensitive. Above 0.7, the signal gets compressed and you start losing return."

---

#### Slide 14（Alpha/Beta/Lambda）— 加结构性功能

当前太弱。**建议重写为：**

> "The gamma sweep gives us one winner. But is gamma07 just a lucky point? Or is the whole hybrid-multiplicative region productive?
> 
> The answer is: the region is productive. Alpha06 — a related parameterisation from the IMADL-m2 family — reaches mean Sharpe 0.69 with CV 0.24. It is below gamma07, but it is stably positive. This corroborates the broader design direction.
> 
> On the other hand, the beta family and the adaptive-lambda family do not reach the stability zone. Their CV values are large and their mean Sharpe is inconsistent.
> 
> So the conclusion is not just 'gamma07 is good.' It is: 'multiplicative hybrid plus variance control is the productive design space. Gamma07 is the best point within it.'"