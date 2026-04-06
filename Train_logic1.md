# **代码逻辑改进措施清单 (基于教授反馈)**

## **1\. 实验范围：暂停滚动，改为“静态”验证**

教授认为在确认代码完全正确前，跑大规模滚动实验是不科学的。

* **动作：** 暂时搁置 run\_step4\_rolling.py 的开发。  
* **新目标：** 创建一个简单的\*\*“静态切分”\*\*脚本（Sanity Check）。  
* **数据划分：**  
  * **Training：** 只取一段固定的 5 年数据（例如 1990-1994）。  
  * **Testing：** 紧接着的 6 个月数据（例如 1995.01 \- 1995.06）。  
* **核心逻辑：** 在这 6 个月的测试期内，不进行 Retraining，直接使用那一个训练好的模型进行预测。

## **2\. 模型架构：解耦模型，独立实验**

教授强调 MSE 和 Median (Median Squared Error) 是完全不同的模型逻辑，不应混在复杂的循环里。

* **独立运行：** 不要试图写一个大循环一次性跑完所有 Loss。  
* **实验 A (MSE)：** 建立一个 NN，Loss \= Minimize Mean Squared Error (MSE)。  
* **实验 B (Median)：** 建立另一个 NN，Loss \= Minimize Median Squared Error。  
* **代码调整：** 确保两个实验除了 **Loss Function 不同**外，网络结构 (MLPConfig) 保持一致（沿用 Grid Search 找到的配置），以便公平对比。

## **3\. 交易逻辑：明确“调仓 (Rebalance)”定义**

教授澄清了 Rebalance 和 Retraining 的区别。

* **频率对齐：** 预测频率 \= 调仓频率 \= 每月一次。  
* **操作流程（6个月测试期内）：**  
  1. 用 5 年数据训练好模型（只练一次）。  
  2. 对未来第 1 个月进行预测 $\\rightarrow$ 构建 Portfolio $\\rightarrow$ 记录收益。  
  3. 对未来第 2 个月进行预测（**用同一个模型**） $\\rightarrow$ 构建 Portfolio $\\rightarrow$ 记录收益。  
  4. ...重复直到第 6 个月。  
* **组合构建规则 (Rebalance)：**  
  * **Long：** forecast return 最高的 10% 股票。  
  * **Short：** forecast return 最低的 10% 股票。

## **4\. 评估指标：细化到“月度”颗粒度**

需要具体的数值来证明模型在工作，而不仅仅是最终的一条曲线。

* **误差分析：** 计算并画出这 6 个月中，**每个月**的 MSE 和 Median Squared Error 变化。  
  * **注意：** 教授特别提到了看看 $R^2$ 是多少，这需要新增计算逻辑。  
* **收益分析：** 计算并画出这 6 个月**每月** Long-Short Portfolio 的实际 Return（正还是负）。

## **建议的执行方案：新的独立脚本**

基于以上反馈，建议我们编写一个新的、独立的脚本 **run\_sanity\_check.py**，逻辑如下：

1. **加载数据：** 仅加载 1990-1995 年中的相关数据。  
2. **构建特征：** 仅计算 Feature Set X1（先聚焦一个）。  
3. **模型训练 (Train Loop)：**  
   * 定义 MLP。  
   * Case 1: 使用 mse\_loss 训练，保存模型 model\_mse。  
   * Case 2: (可选) 使用 median\_loss 训练，保存模型 model\_median。  
4. **月度预测与回测 (Inference Loop)：**  
   * 循环 6 个测试月。  
   * 载入当月数据，使用 model\_mse (或 model\_median) 预测。  
   * **计算 Metrics：** 当月的 MSE, $R^2$。  
   * **构建 Portfolio：** Top 10% / Bottom 10%。  
   * **计算 Return：** 基于 target\_ret 计算该月组合收益。  
5. **输出结果：**  
   * 打印或绘制这 6 个月的 Metrics 表格。  
   * 两个模型分别在六个月的 MSE、Median Squared Error 的折线图。  
   * 两个模型六个月收益率的折线图。  
   * 两个模型最终收益率是正或负。