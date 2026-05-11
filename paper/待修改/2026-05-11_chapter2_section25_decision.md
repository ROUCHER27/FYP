# Chapter 2 Section 2.5 Decision - 2026-05-11

Source reviewed: `paper/已修改/chapter2_literature_review.md`.

Question: whether `2.5 Hybrid loss design and the M2-robust family` is necessary in the Literature Review, and whether it can be deleted.

## Decision

Delete `2.5 Hybrid loss design and the M2-robust family` as a standalone section.

Reason: the current `2.5` is mostly project-method description, not literature review. It explains the additive family, multiplicative family, M2-robust γ family, IMADL-m2 α family, β sweep, and adaptive-λ sweep. These are the report's own experimental designs, so they belong in:

- Chapter 3 §3.3: loss definitions and methodology;
- Chapter 5 §5.3-§5.5: empirical evaluation of A/M, γ, α, β, and adaptive-λ variants;
- Appendix A: formulas.

Keeping `2.5` in Chapter 2 creates two problems:

1. It makes the literature review read like a methods chapter.
2. It conflicts with the current `2.7 Literature synthesis`, which correctly says Chapter 2's contribution is to identify a literature gap, not to define the new method in detail.

## What Chapter 2 should keep

Chapter 2 should keep only the literature-side logic:

- cross-sectional return prediction often varies architecture and features while leaving the loss at MSE/MAE;
- robust regression handles heavy-tailed residuals;
- directional/trading-aware losses target sign/ranking information;
- the unfilled gap is a controlled comparison, plus a motivation for hybridising robust and directional ideas.

Chapter 2 does not need to explain the M2-robust γ family. It only needs one bridging sentence saying that the limitations of existing losses motivate the hybrid designs defined later.

## Exact edit plan

### 1. Change the Chapter 2 opening paragraph

Current opening over-commits by saying the chapter culminates in the M2-robust family. Replace it with:

```markdown
The design space explored in this report sits at the intersection of three strands of literature: (i) machine learning for cross-sectional stock-return prediction, (ii) robust regression under heavy-tailed outcomes, and (iii) trading-aware loss functions that target portfolio rather than point-prediction objectives. This chapter reviews each strand in turn and uses the review to motivate a controlled comparison of regression, robust, directional, and hybrid losses. The project-specific hybrid formulations are defined in Chapter 3 and evaluated in Chapter 5.
```

### 2. Keep only a short bridge after `2.4`

Current `2.4` can still end by motivating hybrid losses, but it should not introduce the detailed families. Replace the final paragraph of `2.4` with:

```markdown
These limitations motivate the hybrid designs evaluated later in the report. Rather than treating regression accuracy, robustness, and directional alignment as separate objectives, the project tests whether they can be combined in one training loss. Chapter 3 gives the formal definitions of the additive, multiplicative, and M2-robust variants; Chapter 5 evaluates them empirically.
```

### 3. Delete current `2.5` entirely

Delete this section:

```text
## 2.5 Hybrid loss design and the M2-robust family
```

Delete all three paragraphs currently under it.

Do not move these paragraphs wholesale elsewhere. The useful information already exists in Chapter 3 and Chapter 5. Repeating it would recreate the same redundancy problem.

### 4. Renumber the remaining sections

After deleting `2.5`, renumber:

```text
2.6 Validation, overfitting, and multiple testing -> 2.5 Validation, overfitting, and multiple testing
2.7 Literature synthesis -> 2.6 Literature synthesis
```

Update any cross-references to `§2.5`, `§2.6`, or `§2.7`.

### 5. Adjust references inside earlier sections

Specific phrases to change:

- In `2.2`, replace "hybrid designs introduced in §2.5" with "hybrid designs defined in Chapter 3".
- In `2.3`, replace "motivates the hybrid designs in §2.5" with "motivates the hybrid designs defined in Chapter 3".
- In `2.4`, remove "GMADL is the parent of the adaptive and hybrid families developed later in this report" if it reads too method-specific; safer replacement:

```markdown
GMADL provides one of the directional components later reused in the project's hybrid losses.
```

## Replacement `2.6 Literature synthesis`

Use this after renumbering:

```markdown
## 2.6 Literature synthesis

The literature reviewed above points to a focused gap. Return-prediction papers routinely vary architecture and features while leaving the loss function at MSE or MAE. Robust-regression losses address heavy-tailed residuals, and directional losses address sign or ranking alignment, but these families are rarely compared under the same data, feature set, architecture, training protocol, and portfolio rule.

This report addresses that gap through a controlled empirical comparison of regression, robust, directional, and hybrid losses. Within each comparison table the protocol isolates the loss choice; cross-phase comparisons are treated as design motivation rather than direct causal claims. The literature review therefore motivates the need for hybrid loss design, while the actual hybrid formulas and empirical rankings are left to Chapter 3 and Chapter 5.
```

## Expected result

This change makes Chapter 2 cleaner and more defensible:

- It stays a literature review instead of becoming a partial methods chapter.
- It removes premature explanation of the report's own new method.
- It aligns the chapter ending with the report structure: Chapter 2 motivates the gap, Chapter 3 defines the method, Chapter 5 tests it.
- It reduces repetition with Chapter 3 §3.3 and Chapter 5 §5.3-§5.5.

Final judgement: `2.5` is not necessary as a standalone literature-review section. Delete it, keep only a short transition from directional-loss limitations to the hybrid designs defined later.
