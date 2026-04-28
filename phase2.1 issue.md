结合 Phase1.5 结论：

1. **Phase1.5 说 M2 有高收益潜力但 seed 敏感。**  
    Phase2.1 证明：简单把 M2 和 IMADL/GMADL/adaptive 混合，并没有自然继承 M2 的收益潜力。最好的 imadl_gmadl_beta03 Sharpe 也只有 0.0399，累计收益均值 -0.095%。
    
2. **m2_robust_gamma10 的意义是“稳定化有效，但收益被压没了”。**  
    它 0% failure、CV 0.373，但累计收益均值 -0.402%，平均月收益只有 0.018%。说明 robustness penalty 方向上有价值，但当前形式太弱或目标不对。
    
3. **Phase2.1 初步证伪了“naive loss-level combination”。**  
    V1 IMADL+M2 linear 全部负 Sharpe；V4 adaptive 也没有起色；V2 只有 beta03 勉强正。这个结果可以写成：组合损失不是简单线性加权就能改善交易稳健性。
    
4. **更关键：当前 Phase2.1 可能没有严格测试原文档假设。**  
    doc/phase2 里的 M2 是 hybrid_mul_m2 / hybrid_dir_huber_mul_loss，但分支实现用了简化 m2_loss = -sign(y_true) * squared_error。另外 robustness penalty 实现是 M2 - gamma * std(y_pred)^2，从最小化角度看这不像“惩罚”，反而可能鼓励更大预测方差。这个要先修正或明确为 ablation。
    

除了 Phase2.2，我建议优先做这几件：

**1. 做 Phase2.1b 对齐实验**  
把 Phase1.5 的原始 baseline 放进同一 runner 重跑：imadl, gmadl, hybrid_mul_m2, hybrid_add_a4，同样 seeds 42,52,62、cap 0.05。目的不是扩大实验，而是确认 Phase2 runner 和 Phase1.5 结果可比。

**2. 修正/重命名 M2**  
决定到底用：

- Phase1.5 的真实 hybrid_mul_m2
- 还是当前简化 signed squared error

如果用当前实现，论文里不能直接说它是 Phase1.5 的 M2 延续。

**3. 做 loss scale 诊断**  
线性组合前要看 IMADL、GMADL、M2 每个 batch 的 loss magnitude 和 gradient norm。现在很多组合结果很差，可能不是理论失败，而是某个 loss 项尺度支配了训练。

**4. 改 robustness penalty**  
建议测试：

- M2 + gamma * pred_variance
- M2 + gamma * weight_concentration_proxy
- M2 + gamma * turnover_proxy

至少先把 “减 penalty” 改成可解释的 “加 penalty”，或者把正负号作为 ablation 明确测试。

**5. 不只看 final epoch，加入 validation Sharpe early stopping**  
Phase1.5 已经显示 MSE/R² 和交易结果脱节。继续只按训练 loss 结束，容易选到预测误差相似但交易表现差异巨大的模型。

**6. 做 seed ensemble**  
Phase1.5 最大问题是 seed sensitivity。一个直接调整是每个 loss 训练 3 个 seed，预测取平均，再做 portfolio ranking。这比继续发明 loss 更贴近已发现的问题。

我会把当前 Phase2.1 定位为：**naive combination screening failed, but it clarified that future work should move from loss interpolation to implementation-aligned M2, normalized components, explicit robustness regularization, and seed ensembling.**