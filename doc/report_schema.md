# FYP Report Schema and Formatting Guide

> Status: legacy formatting/reference guide. For final-report structure, evidence scope, and claim boundaries, `SCHEMA.md` is authoritative. If this file conflicts with `SCHEMA.md`, follow `SCHEMA.md`.

This document provides comprehensive guidelines for writing the FYP final report based on the official template and interim report analysis.

---

## 1. Overall Document Structure

### Required Sections (in order)

1. **Title Page**
   - Project title
   - Student name and ID
   - Supervisor name
   - School name: "School of Mathematics & Physics"
   - Submission date

2. **Abstract** (1 page max)
   - Research problem and motivation
   - Methodology overview
   - Key findings
   - Implications

3. **Acknowledgments** (optional, 1 page max)

4. **Table of Contents** (auto-generated)

5. **List of Figures** (if applicable)

6. **List of Tables** (if applicable)

7. **Main Chapters** (see Chapter Guidelines below)

8. **Bibliography**

### Page Numbering
- Front matter (Abstract, Acknowledgments, TOC): Roman numerals (i, ii, iii)
- Main content (Chapter 1 onwards): Arabic numerals (1, 2, 3)
- Each chapter starts on a new page

---

## 2. Chapter Guidelines

### Chapter 1: Introduction

**Purpose**: Set context, identify research gap, motivate the study

**Typical Length**: 3-5 pages

**Required Subsections**:

1. **Background and Motivation** (2-3 paragraphs)
   - Introduce the domain (e.g., financial machine learning, portfolio optimization)
   - Explain why this topic matters
   - Connect to real-world applications

2. **Research Gap** (1-2 paragraphs)
   - What's missing in current literature
   - Why existing approaches are insufficient
   - Example from interim: "these studies rarely discuss the systematic design and comparison of different loss functions"

3. **Research Objectives** (numbered list)
   - Clear, specific, measurable objectives
   - Example: "systematically evaluate the impact of different loss functions on model predictive ability"

4. **Scope and Limitations**
   - What's included/excluded
   - Boundary conditions
   - Assumptions made

5. **Thesis Structure** (1 paragraph)
   - Brief overview of remaining chapters
   - Roadmap for the reader

**Example Opening** (from interim report):
"This study aims to use Deep Neural Networks (DNNs) for cross-sectional stock return prediction and systematically evaluate the impact of different loss functions (MSE, MedSE, MADL, etc.) on model predictive ability and portfolio performance."

---

### Chapter 2: Literature Review

**Purpose**: Survey existing research, identify gaps, position your work

**Typical Length**: 4-8 pages

**Structure**:

1. **Thematic Organization** (not chronological)
   - Group papers by topic/approach
   - Example themes: "Machine Learning for Asset Pricing", "Loss Function Design", "Portfolio Construction"

2. **Critical Analysis** (not just summary)
   - Identify strengths and limitations of each approach
   - Compare and contrast different methods
   - Build narrative toward your research gap

3. **Citation Integration**
   - Cite as you discuss: "Michańków et al. proposed the Mean Absolute Directional Loss (MADL) to bridge the gap [7]"
   - Use citations to support claims, not just list papers

4. **Transition to Your Work**
   - Final paragraph connects literature to your methodology
   - Example: "This project builds upon MADL/GMADL to further investigate..."

**Writing Style**:
- Third person, past tense for describing prior work
- Present tense for general truths: "Machine learning models mine effective cross-sectional signals"
- Critical but respectful tone

---

### Chapter 3: Methodology

**Purpose**: Describe your approach in sufficient detail for replication

**Typical Length**: 5-10 pages

**Required Subsections**:

1. **Data Description and Preprocessing**
   - Data source: "Raw data is sourced from the CRSP monthly stock database via WRDS"
   - Sample period with dates
   - Variable selection (features and target)
   - Missing value handling
   - Any filters or exclusions

2. **Feature Engineering**
   - Mathematical definitions of all features
   - Rationale for each feature set
   - Example: "Feature Set 1 (X¹): Cumulative Momentum and Turnover"
   - Include formulas with proper notation

3. **Model Architecture**
   - Model type and justification
   - Hyperparameters (layers, activation, regularization)
   - How hyperparameters were chosen (e.g., Grid Search)
   - Example: "Configuration: 3 Hidden Layers [64, 32, 16]"

4. **Training Protocol and Evaluation**
   - Train/test split strategy
   - Validation approach (static vs rolling window)
   - Evaluation metrics (statistical and economic)
   - Example: "Static Sanity Check: Training Window Jan 1990 - Dec 1994"

5. **Portfolio Construction and Metrics**
   - Portfolio strategy (long/short, equal-weighted, etc.)
   - Rebalancing frequency
   - Performance metrics: R², Sharpe Ratio, Cumulative Return

**Key Principle**: "Provide enough detail to allow other researchers to replicate your work"

---

### Chapter 4: Results / Experimental Results

**Purpose**: Present findings objectively without interpretation

**Typical Length**: 8-15 pages

**Structure**:

1. **Organize by Research Question/Experiment**
   - Each major experiment gets its own section
   - Example: "4.1 Replication and Analysis of MADL & GMADL"
   - Example: "4.2 MSE vs. MedSE Portfolio Backtest Comparison"

2. **Present Results Systematically**
   - Start with experiment design/setup
   - Show visualizations (figures)
   - Present quantitative results (tables)
   - State key findings without interpretation

3. **Figures and Tables**
   - Every figure/table must be referenced in text
   - Captions are descriptive and self-contained
   - Example: "Figure 4.1: GMADL Loss Function Behavior in Different Scenarios"

4. **Key Findings Subsection** (optional)
   - Bullet points summarizing main observations
   - Example: "Symmetry Issue: Reward magnitude equals penalty magnitude"
   - No interpretation yet (save for Discussion)

**Example Section Structure** (from interim):
```
4.2 MSE vs. MedSE Portfolio Backtest Comparison

I conducted backtests using three portfolio construction methods...

Portfolio Definitions:
• P1 (Equal): Long top 10%, Short bottom 10%, equal weights
• P2 (Signal Weighted): Z-score normalization within Long/Short buckets
• P3 (Capped): Adds a single stock weight cap to P2

Results: The results indicate significant differences. Table 4.1 summarizes...
```

---

### Chapter 5: Discussion / Analysis

**Purpose**: Interpret results, explain implications, connect to literature

**Typical Length**: 4-8 pages

**Structure**:

1. **Interpretation of Results**
   - Explain WHY you observed these results
   - Connect findings to theory
   - Example: "MedSE generates consistently positive and higher Sharpe ratios"

2. **Comparison with Literature**
   - How do your results compare to prior work?
   - Do they confirm or contradict existing findings?
   - What's new or surprising?

3. **Practical Implications**
   - What do these findings mean for practitioners?
   - How could this be applied in real trading?
   - Risk considerations

4. **Limitations**
   - Acknowledge weaknesses in your approach
   - Data limitations
   - Model assumptions
   - Generalizability concerns

**Writing Style**:
- More interpretive and analytical than Results chapter
- Use phrases like "This suggests that...", "One possible explanation is..."
- Connect back to research objectives

---

### Chapter 6: Conclusion

**Purpose**: Summarize contributions, suggest future work

**Typical Length**: 2-3 pages

**Required Elements**:

1. **Summary of Work** (1-2 paragraphs)
   - Restate research objectives
   - Briefly summarize methodology
   - Highlight key findings

2. **Main Contributions** (bullet points or numbered)
   - What did you accomplish?
   - What's new or improved?
   - Example: "Demonstrated that MedSE outperforms MSE across all portfolio constructions"

3. **Future Work** (bullet points)
   - What questions remain unanswered?
   - How could this research be extended?
   - Example from interim: "Implement Method 1 (Directional Penalty) and Method 2 (Ranking Loss)"

4. **Closing Statement** (1 paragraph)
   - Broader impact or significance
   - Final thought on the research

**Avoid**:
- Introducing new results or analysis
- Lengthy repetition of methodology
- Overstating contributions

---

### Bibliography

**Citation Style**: SIAM (Society for Industrial and Applied Mathematics)

**Format**:
- Numbered references [1], [2], [3]
- Alphabetical by first author's last name
- In-text citations: [1], [1, 2], [1-3]

**Example Entries** (from interim report):

```
[1] Lopez de Prado, M. (2013). What to look for in a backtest. Available at SSRN.

[7] Michańków, J., Sakowski, P., & Ślepaczuk, R. (2024b). Mean absolute directional 
    loss as a new loss function for machine learning problems in algorithmic investment 
    strategies. Journal of Computational Science, 81, 102375.

[9] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. 
    Review of Financial Studies, 33(5), 2223-2273.
```

**In-Text Citation Examples**:
- Single source: "Michańków et al. proposed MADL [7]"
- Multiple sources: "Several studies have explored this [1, 5, 9]"
- Range: "Prior work [1-3] has shown..."

---

## 3. Formatting Rules

### Text Formatting

**Font and Spacing**:
- Font: Times New Roman or similar serif font
- Size: 12pt for body text, larger for headings
- Line spacing: 1.5 or double-spaced
- Paragraph spacing: 6pt after paragraphs
- Margins: 1 inch (2.54 cm) on all sides

**Headings**:
- Chapter titles: Large, bold, centered or left-aligned
- Section headings (e.g., 3.1): Bold, numbered
- Subsection headings (e.g., 3.1.1): Bold or italic, numbered

**Emphasis**:
- Use **bold** for key terms on first introduction
- Use *italic* for emphasis (sparingly)
- Avoid underlining

---

### Mathematical Notation

**Inline Math**:
- Use for simple expressions within sentences
- Markdown: Use single backticks or $...$ if supported
- Examples: `π`, `x_i`, `1/n`, `r_{i,t}`

**Display Equations**:
- Use for important formulas
- Number equations for reference
- Markdown: Use code blocks with ```math

**Example** (from interim report):
```math
L_i = -[\sigma(a \cdot y \cdot \hat{y}) - 0.5] \times |y|^b
```

**Equation Numbering**:
- Format: (Chapter.Number) e.g., (3.1), (3.2)
- Reference in text: "as shown in Equation (3.1)"

**Common Notation**:
- Subscripts: `r_{i,t}` (return of stock i at time t)
- Superscripts: `X^1` (feature set 1)
- Greek letters: α, β, σ, π
- Vectors: **x** or x (bold)
- Matrices: **X** or X (bold capital)

**Special Symbols**:
- Summation: ∑
- Product: ∏
- Approximately: ≈
- Less/greater than or equal: ≤, ≥

---

### Figures

**Numbering Convention**:
- Format: Figure X.Y (Chapter.Number)
- Example: Figure 4.1, Figure 4.2
- Sequential within each chapter

**Caption Format**:
```
Figure 4.1: GMADL Loss Function Behavior in Different Scenarios. The sigmoid 
transition provides a smooth gradient, identifying correct directions (negative 
loss/reward) versus wrong directions (positive loss/penalty).
```

**Caption Structure**:
- **Bold figure number and title**
- Followed by descriptive text explaining what the figure shows
- Self-contained (reader should understand without reading main text)

**Placement**:
- Place figures after first reference in text
- Center-align figures
- Leave space above and below

**Referencing in Text**:
- "Figure 4.1 shows..."
- "As illustrated in Figure 4.2..."
- "The results (Figure 4.3) indicate..."

**Figure Quality**:
- High resolution (300 DPI minimum)
- Clear labels and legends
- Readable font sizes
- Color-blind friendly palettes when possible

---

### Tables

**Numbering Convention**:
- Format: Table X.Y (Chapter.Number)
- Example: Table 4.1, Table 4.2
- Sequential within each chapter

**Caption Format**:
```
Table 4.1: Strategy Performance Comparison (Jan-Jun 1995)
```

**Caption Placement**:
- **Above the table** (unlike figures, which have captions below)

**Table Structure** (from interim report):
```
| Strategy            | Loss  | Std    | Sharpe  | CumReturn |
|---------------------|-------|--------|---------|-----------|
| P1_Equal            | MSE   | 0.0147 | 0.3730  | 0.9%      |
| P1_Equal            | MedSE | 0.0116 | 2.6773  | 5.48%     |
| P2_SignalWeighted   | MSE   | 0.0288 | -1.4559 | -7.23%    |
| P2_SignalWeighted   | MedSE | 0.0282 | 3.2286  | 16.64%    |
```

**Formatting Guidelines**:
- Align numbers by decimal point
- Use consistent decimal places
- Include units in column headers
- Bold or highlight key results
- Keep tables simple and readable

**Referencing in Text**:
- "Table 4.1 summarizes..."
- "As shown in Table 4.2..."
- "The performance metrics (Table 4.1) reveal..."

---

## 4. Academic Writing Style

### Tone and Voice

**Formal and Objective**:
- Third person preferred: "This study investigates..." (not "I investigate...")
- Exception: Methodology can use first person: "I conducted backtests..."
- Avoid colloquialisms and contractions
- Maintain professional distance

**Active vs Passive Voice**:
- Prefer active voice for clarity: "The model predicts returns" (not "Returns are predicted by the model")
- Passive voice acceptable for emphasis: "The data was collected from CRSP"

**Tense Usage**:
- **Introduction**: Present tense for general truths, past tense for prior work
- **Literature Review**: Past tense for what others did
- **Methodology**: Past tense for what you did
- **Results**: Past tense for what you found
- **Discussion**: Present tense for interpretation
- **Conclusion**: Present tense for contributions

### Sentence Structure

**Clarity**:
- One main idea per sentence
- Average sentence length: 15-25 words
- Vary sentence length for readability
- Avoid run-on sentences

**Paragraph Organization**:
- Topic sentence first
- Supporting sentences with evidence
- Concluding/transition sentence
- Typical length: 4-8 sentences

**Transitions**:
- Between paragraphs: "Furthermore,", "However,", "In contrast,"
- Between sections: "Having established X, we now turn to Y"
- Between chapters: "Building on the methodology described in Chapter 3..."

### Terminology

**Consistency**:
- Use the same term throughout (don't alternate between "loss function" and "objective function")
- Define abbreviations on first use: "Mean Squared Error (MSE)"
- Create a notation table if using many symbols

**Precision**:
- "significant" only for statistical significance
- "optimal" only when proven mathematically optimal
- "better" requires quantitative comparison

---

## 5. Specific Requirements for This Project

### Presenting Loss Functions

**Definition Format**:
```
The GMADL loss is defined as:

L_i = -[σ(a · y · ŷ) - 0.5] × |y|^b

where a = 100, b = 2.
```

**Components to Explain**:
- Mathematical formula
- Parameter values and their meaning
- Intuition behind the design
- How it differs from alternatives

### Presenting Experimental Results

**Comparative Analysis Structure**:
1. State the comparison (MSE vs MedSE)
2. Define the experimental setup
3. Present quantitative results (table)
4. Show visual results (figure)
5. Summarize key differences

**Performance Metrics to Report**:
- Statistical: R², MSE, Median SE
- Economic: Cumulative Return, Annualized Return, Sharpe Ratio
- Risk: Standard Deviation, Maximum Drawdown

### Algorithm Pseudocode

**Format** (if needed):
```
Algorithm 1: Rolling Window Backtest

Input: Features X, Returns y, window_size, test_months
Output: Predictions, Portfolio returns

1: for each test month t do
2:    Train model on [t - window_size, t - 1]
3:    Predict returns for month t
4:    Construct portfolio based on predictions
5:    Calculate portfolio return
6: end for
7: return cumulative returns
```

### Portfolio Metrics Presentation

**Standard Format**:
- Always include: Strategy name, Loss function, Std, Sharpe, Cumulative Return
- Optional: Max Drawdown, Win Rate, Turnover
- Group by portfolio construction method
- Highlight best performers

---

## 6. Markdown Adaptation for Obsidian

### Equations in Obsidian

**Inline Math**:
```
The return $r_{i,t}$ is calculated as...
```

**Display Math**:
````
```math
cr_{i,m} = \prod_{j=t-m}^{t-2} (1 + r_{ij}) - 1
```
````

### Tables in Markdown

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

### Figures in Markdown

```markdown
\![Figure 4.1: Caption text](path/to/image.png)

**Figure 4.1**: Detailed caption explaining the figure.
```

### Cross-References

**Internal Links** (Obsidian):
```markdown
As discussed in [[Chapter 3 - Methodology]], we use...
See [[#Section 4.2]] for details.
```

**Citation References**:
```markdown
Michańków et al. proposed MADL [7].
```

### Callouts for Important Notes

```markdown
> [\!note] Key Assumption
> We assume markets are frictionless with no transaction costs.

> [\!warning] Limitation
> Results may not generalize to non-US markets.
```

---

## 7. Quality Checklist

### Chapter 1: Introduction
- [ ] Clear research question/objective stated
- [ ] Research gap identified and justified
- [ ] Scope and limitations defined
- [ ] Thesis structure outlined
- [ ] Proper citations for background claims
- [ ] Engaging opening that motivates the work

### Chapter 2: Literature Review
- [ ] Thematic organization (not chronological)
- [ ] Critical analysis (not just summary)
- [ ] Proper citations throughout
- [ ] Clear connection to your research gap
- [ ] Transition to your methodology

### Chapter 3: Methodology
- [ ] Sufficient detail for replication
- [ ] All formulas properly formatted and numbered
- [ ] Hyperparameters and design choices justified
- [ ] Data sources clearly documented
- [ ] Evaluation metrics defined

### Chapter 4: Results
- [ ] All figures properly numbered and captioned
- [ ] All tables properly numbered and captioned
- [ ] Every figure/table referenced in text
- [ ] Results presented objectively (no interpretation)
- [ ] Consistent notation throughout

### Chapter 5: Discussion
- [ ] Results interpreted and explained
- [ ] Comparison with literature
- [ ] Limitations acknowledged
- [ ] Practical implications discussed

### Chapter 6: Conclusion
- [ ] Main contributions summarized
- [ ] Future work suggested
- [ ] No new results introduced
- [ ] Concise and impactful

### Overall Document
- [ ] Consistent terminology throughout
- [ ] Proper grammar and spelling
- [ ] Logical flow between sections
- [ ] All citations in bibliography
- [ ] Page numbers correct
- [ ] Table of contents updated

---

## 8. Common Pitfalls to Avoid

### Citation Errors
- ❌ Missing citations for claims
- ❌ Inconsistent citation format
- ❌ Citing without reading (citation padding)
- ✅ Cite as you make claims
- ✅ Use consistent SIAM format
- ✅ Only cite papers you've actually read

### Figure/Table Errors
- ❌ Figures without captions
- ❌ Tables not referenced in text
- ❌ Poor quality or unreadable figures
- ❌ Inconsistent numbering
- ✅ Every figure/table has descriptive caption
- ✅ Reference before showing
- ✅ High-quality, readable visuals

### Equation Formatting
- ❌ Inline equations that should be displayed
- ❌ Unnumbered important equations
- ❌ Inconsistent notation
- ❌ Undefined variables
- ✅ Display important formulas
- ✅ Number equations for reference
- ✅ Define all variables

### Writing Style
- ❌ Informal language: "pretty good results"
- ❌ Vague claims: "much better performance"
- ❌ Unsupported assertions
- ✅ Formal academic tone
- ✅ Quantitative comparisons: "16.64% vs -7.23%"
- ✅ Evidence-based claims

### Structure Issues
- ❌ Results mixed with interpretation
- ❌ Methodology in results section
- ❌ New results in conclusion
- ✅ Clear separation of sections
- ✅ Logical flow
- ✅ Appropriate content in each chapter

---

## 9. Chapter-Specific Templates

### Template: Chapter 1 (Introduction)

```markdown
# Chapter 1: Introduction

## 1.1 Background and Motivation

[2-3 paragraphs introducing the domain and its importance]

The field of [domain] has seen significant advances...

Recent developments in [specific area] have enabled...

However, challenges remain in [specific problem]...

## 1.2 Research Gap

[1-2 paragraphs identifying what's missing]

While existing studies have explored [X], they rarely address [Y]...

This gap is particularly important because...

## 1.3 Research Objectives

This study aims to:

1. [Specific objective 1]
2. [Specific objective 2]
3. [Specific objective 3]

## 1.4 Scope and Limitations

This research focuses on [scope]...

The following limitations apply: [list]...

## 1.5 Thesis Structure

The remainder of this thesis is organized as follows. Chapter 2 reviews...
Chapter 3 describes... Chapter 4 presents... Chapter 5 discusses...
Chapter 6 concludes...
```

### Template: Chapter 3 (Methodology)

```markdown
# Chapter 3: Methodology

## 3.1 Data Description and Preprocessing

Raw data is sourced from [source] covering [period]...

### 3.1.1 Variable Selection

The following variables are used:
- **Return (RET)**: [definition]
- **Volume (VOL)**: [definition]

### 3.1.2 Preprocessing Steps

1. [Step 1]
2. [Step 2]

## 3.2 Feature Engineering

Based on [literature], three feature sets are constructed:

**Feature Set 1 (X¹)**: [Name]

[Mathematical definition]

```math
cr_{i,m} = \prod_{j=t-m}^{t-2} (1 + r_{ij}) - 1
```

## 3.3 Model Architecture

A [model type] is used with the following configuration:
- Layers: [specification]
- Activation: [function]
- Regularization: [method]

## 3.4 Training Protocol

[Describe train/test split, validation strategy]

## 3.5 Evaluation Metrics

**Statistical Metrics**:
- R²: [definition]
- MSE: [definition]

**Economic Metrics**:
- Sharpe Ratio: [definition]
- Cumulative Return: [definition]
```

### Template: Chapter 4 (Results)

```markdown
# Chapter 4: Experimental Results

## 4.1 [Experiment Name]

### 4.1.1 Experiment Design

[Describe setup, parameters, conditions]

### 4.1.2 Results

[Present quantitative results]

Table 4.1 summarizes the performance metrics...

| Strategy | Loss | Metric1 | Metric2 |
|----------|------|---------|---------|
| ...      | ...  | ...     | ...     |

Figure 4.1 illustrates the cumulative returns...

\![Figure 4.1](path/to/figure.png)

**Figure 4.1**: [Detailed caption]

### 4.1.3 Key Findings

- Finding 1: [Observation]
- Finding 2: [Observation]

## 4.2 [Next Experiment]

[Repeat structure]
```

---

## 10. Example Sections from Interim Report

### Well-Formatted Equation

```math
L_i = -[\sigma(a \cdot y \cdot \hat{y}) - 0.5] \times |y|^b
```

where `a = 100`, `b = 2`. The sigmoid function `σ(·)` provides smooth gradients.

### Well-Formatted Table

**Table 4.1**: Strategy Performance Comparison (Jan-Jun 1995)

| Strategy            | Loss  | Std    | Sharpe  | CumReturn |
|---------------------|-------|--------|---------|-----------|
| P1_Equal            | MSE   | 0.0147 | 0.3730  | 0.9%      |
| P1_Equal            | MedSE | 0.0116 | 2.6773  | 5.48%     |
| P2_SignalWeighted   | MSE   | 0.0288 | -1.4559 | -7.23%    |
| P2_SignalWeighted   | MedSE | 0.0282 | 3.2286  | 16.64%    |

### Well-Formatted Figure Caption

**Figure 4.1**: GMADL Loss Function Behavior in Different Scenarios. The sigmoid transition provides a smooth gradient, identifying correct directions (negative loss/reward) versus wrong directions (positive loss/penalty).

### Well-Formatted Citation

Michańków et al. proposed the Mean Absolute Directional Loss (MADL) to bridge the gap between traditional error functions and real investment goals [7]. MADL combines "directional correctness" with "magnitude of true return," mapping each prediction to a potential "gain or loss if traded in that direction."

### Well-Structured Paragraph

The results indicate significant differences between MSE and MedSE loss functions. Table 4.1 summarizes the performance metrics across three portfolio construction methods. MedSE consistently generates higher Sharpe ratios (2.6773 vs 0.3730 for P1, 3.2286 vs -1.4559 for P2), suggesting superior risk-adjusted returns. The cumulative return advantage is substantial: MedSE achieves 5.48% vs 0.9% for P1_Equal and 16.64% vs -7.23% for P2_SignalWeighted. This performance boost appears to stem from effective signal capture rather than excessive leverage, as volatilities remain comparable.

---

## 11. Final Checklist Before Submission

### Content Completeness
- [ ] All required chapters present
- [ ] Abstract summarizes entire work
- [ ] Introduction clearly states objectives
- [ ] Literature review comprehensive
- [ ] Methodology detailed and replicable
- [ ] Results presented systematically
- [ ] Discussion interprets findings
- [ ] Conclusion summarizes contributions
- [ ] Bibliography complete

### Formatting Consistency
- [ ] Consistent font and spacing throughout
- [ ] All equations properly formatted
- [ ] All figures numbered and captioned
- [ ] All tables numbered and captioned
- [ ] Page numbers correct
- [ ] Headers/footers consistent
- [ ] Table of contents updated

### Quality Assurance
- [ ] Spell-checked
- [ ] Grammar-checked
- [ ] Citations verified
- [ ] Figures high-quality
- [ ] Tables readable
- [ ] No placeholder text (e.g., "TODO", "XXX")
- [ ] Consistent terminology
- [ ] Logical flow

### Technical Accuracy
- [ ] All equations correct
- [ ] All numbers verified
- [ ] All claims supported
- [ ] No contradictions
- [ ] Notation consistent
- [ ] Units specified

### Academic Integrity
- [ ] All sources cited
- [ ] No plagiarism
- [ ] Original work clearly distinguished
- [ ] Proper attribution
- [ ] Ethical considerations addressed

---

## 12. Resources and References

### Writing Guides
- SIAM Style Manual (for citation format)
- Strunk & White's "Elements of Style"
- University writing center resources

### LaTeX/Markdown
- Obsidian documentation for markdown syntax
- MathJax for equation rendering
- Pandoc for format conversion

### Domain-Specific
- Financial econometrics textbooks for notation
- Machine learning papers for methodology structure
- Prior FYP reports for examples

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-06  
**Based On**: FYP Final Report Template + Interim Report 2253235_YirongYu_2025
