Implementation Plan — Academic Tables & Equations Densification

Problem Statement

当前

ppt_package/fyp_oral_presentation/index.html

（19 页 Swiss Academic）虽然版式骨架对了（S22/S24/S26/S27），但论文里 5 个表格（5.1–5.5）和 appendix A 的全套 loss/gradient 公式基本没体现到 PPT 上：P11/P12/P14/P15 只有右侧 4 段文字 bullets、P07 5 个 component 卡里嵌的 0.72vw 行内 KaTeX 在投影上看不清、P08 评估指标完全没写公式。这导致 deck 在学术答辩场景下的"数学严谨性 + 表格证据"维度严重失分。

Requirements (来自用户对齐)

- R1：保持 19 页总数，

就地改造

（不增页、不减页）→ 决策 1=a - R2：P07 改造成单页

S25 Equation Hero

，hybrid_mul 主公式居中 + 4 个 component（λ / D / H_δ / a, b），M2-robust-γ 作为底部 accent 横条 → 决策 2=a - R3：

不动全局 CSS 基线

（

body-sm

、

t-meta

、

h-xl

等不改），仅对过空页填论文实证内容（表 + 公式）；表格 / 公式 KPI 数字允许局部覆盖更大字号 → 决策 3=c - R4：每张图都来源于

2253235_yirongyu_2026_Supplementary/latex/figures/

；每条数字都直接引自 supplementary 论文表 - R5：所有公式必须用 KaTeX

$$ ... $$

渲染，不得用 unicode（layouts-swiss-academic.md 陷阱 1+4） - R6：表格采用 S23 学术版式（顶/底 1.5px hairline、行间无线、mono+tabular-nums、best/second 行高亮） - R7：执行完成后用浏览器截图逐页审核效果

Background

-

当前 deck 已用 CSS

：已加载 KaTeX 0.16.11；

acad-table

/

acad-row-best

/

acad-row-second

已定义并在 P13 使用；S22/S24/S26/S27 layout 已使用 -

构建方式

：

build_deck.mjs

是 Node 脚本，从

assets/template-swiss-academic.html

注入 SLIDE 内容，从 supplementary

figures/

复制图片到本地

images/

。改

index.html

的方式有两种：(a) 直接编辑

index.html

，(b) 改

build_deck.mjs

后重跑。我推荐

(a) 直接编辑 index.html

——build script 似乎是初次脚手架用的，当前 19 页内容都在

slides

模板字串里，改 index 更直接、不丢手工细节 -

数据源 — 完整数字 vault

： - Table 5.1 (P11)：MSE −0.4643/−0.1125, MedSE 0.0932/0.0060, MADL −0.3058/−0.0756, GMADL 0.2025/0.0279, IMADL −0.3732/−0.0944,

hybrid_mul_m1

0.4435/0.0509 (best)

, hybrid_mul_m2 −0.0017/−0.0032 - Table 5.2 (P12)：A1 0.1241, A2 0.2173,

A3 0.5738 (best)

, A4 0.2311, A5 −0.4110,

M1 0.4435 (second)

, M2 −0.0017, M3 −0.9691, M4 −0.3440 - Table 5.3 (P13) 已有但缺 std 列：γ03 std=0.3418, γ05 std=0.1488, γ07 std=0.1655, γ10 std=0.5638, γ15 std=0.3724 - Table 5.4 selected (P14)：alpha04 0.354/0.185, alpha05 0.582/0.548,

alpha06 0.690/0.244 (best)

, alpha07 0.402/0.613, beta05 0.041/10.13, beta07 −0.002/139.5, lambda10 0.494/1.543, lambda50 0.276/0.578,

gamma10 1.004/0.561

- Table 5.5 (P15)：γ07 113× / 0.9156→0.9112 (per-seed 0.5956, 1.4064, 0.7317), γ10 113× / 1.0043→0.4072 (per-seed 0.6254, 0.1181, 0.4780), α06 34× / 0.6895→−0.0161 (per-seed 0.5628, −0.8335, 0.2224) -

公式 vault

：P07 主公式 $L_{\mathrm{mul},\lambda}(y,\hat y) = (1+\lambda,D(y,\hat y))\cdot H_\delta(y-\hat y)$；where $D = [1-\sigma(ay\hat y)]\cdot\frac{|y|^b}{\mathbb{E}

{\mathcal B}[|y|^b]+\epsilon}$；M2-robust-γ $L

{M2\text{-}robust,\gamma} = L_{M2} + \gamma,\mathrm{Var}(\hat y)$；P08 Sharpe $\sqrt{12},\bar r/\sigma_r$、CV $\sigma_S/|\mu_S|$、Huber $H_\delta(e) = \tfrac12 e^2$ if $|e|\le\delta$ else $\delta(|e|-\tfrac\delta2)$

Proposed Solution (页面级方案)

P07 · S25 Equation Hero（核心改造）

graph TD A[Title: 一页四族 + 我们的扩展] --> B[公式区 hairline 上下夹] B --> B1[主: L_mul,λ = 1+λD · H_δ] B --> B2[where: D = 1-σ ay-ŷ · y^b / E|y|^b+ε] C[4 component grid-12 span-3] --> C1[λ Directional Weight λ∈0.1,0.5,2,5; M1 λ=2 M2 λ=5] C --> C2[D Directional Gate D≥0, D→0 sign-correct] C --> C3[H_δ Huber Backbone 分段公式 δ=0.01] C --> C4[a,b Shape σay-ŷ a=100 b=2] D[Accent Strip OUR EXTENSION] --> D1[L_M2-robust,γ = L_M2 + γ·Var ŷ; γ∈0.3,0.5,0.7,1.0,1.5] 中央公式区每行 KaTeX font ~1.35vw（投影清晰）；4 component card 等高 stretch；accent strip 在 component 之下，IKB 实色横条，公式白字。M2-robust-γ 的特殊视觉权重通过这条 strip 体现（不挤入 4 个 card）。

P08 · 加 Evaluation 公式条带

：右下增加紧凑公式块（accent border-left）："Sharpe = $\sqrt{12},\bar r/\sigma_r$ · CV = $\sigma_S/|\mu_S|$ · Cap = 5% per-name"，replacement of 当前 metrics span-4 块的纯文字。

P11 · S24 split + 7-row baseline table

：左 7 列保留 fig5_1 原图（缩到 16:10），右 5 列嵌 S23 7 行 × 4 列表（Loss / Sharpe / Cum / Avg R²，R² 用科学计数 −7.0×10⁹），best=hybrid_mul_m1 高亮 IKB 短色块，second=GMADL hairline。表下方一行 t-meta caption "FIG 5.1 + TABLE 5.1 · SEED 42"。

P12 · S24 split + 9-row A/M table

：左 7 列保留 fig5_2 图，右 5 列 9 行 × 3 列表（Variant / Loss ID / Sharpe），best=A3、second=M1。底部加一行 "M3 collapse −0.97 → motivates multi-seed"。

P13 · 扩到 5 行 × 6 列

：当前缺 Sharpe std 列；改为 Loss / Runs / Sharpe mean / Sharpe std / Cum return / CV，单元字号从 1.25vh → 1.6vh（局部覆盖），best=γ07 三列同时高亮（0.9156/0.1655/0.1808），second=γ10。图保留在 span-7 不变。

P14 · 替换 frontier 图为 S23 selected-rows table

：左 6 列保留 fig5_5 imadl_alpha_sweep 图，右 6 列改 S23 9 行 × 3 列 selected integrated table（family group：α04/α05/

α06 best

/α07 + β05/β07 collapse + λ10/λ50 + γ10 ref）。frontier 图作为 backup 移到 P14 下方 thumb-strip 或直接舍弃（已有 alpha sweep + table 信息更丰富）；倾向

舍弃 frontier 图 + 整页给 alpha + table

。

P15 · S24 split + Normalisation probe table

：左 7 列保留 fig5_7 图，右 5 列改 S23 3 行 × 5 列表（Loss / Scale ratio / Original / Normalised / Per-seed Sharpes），best=γ07 高亮（0.9156→0.9112 同行）；保留三个箭头数字但用表化呈现（mono tabular-nums）。

P02 / P04 / P09（过空页填补，不动 CSS）

： - P02 底部 4-strip 之下加一条 t-meta citation row："Gu Kelly Xiu 2020 · Daniel Moskowitz 2016 · Huber 1964 · Michańków 2024"，把"literature mismatch"具象到引用 - P04 timeline 下增加 chapter anchor 一行："→ §3.1 Research design · §3.5 Portfolio · §3.6 Metrics" - P09 timeline 下加 evidence-strength tag row："P1/P2 single-seed · P3a/P3b 3-seed · P4 diagnostic probe"

P10 / P16 / P17 / P18 / P19

：保持不变，已 OK。

Task Breakdown

Task 1: P07 改造 S25 Equation Hero（hybrid_mul + 4 component + M2-robust-γ accent strip）

Objective

：把 P07 从 5-card 小公式排版重构为标准 S25——中央 hybrid_mul 公式 + 4 component + 底部 IKB 横条展示 M2-robust-γ。

Implementation guidance

： - 删除当前 P07 中 5 张卡 layout，参考 layouts-swiss-academic.md S25 骨架；公式区两行 KaTeX（主 1.35vw + where 1.05vw 灰色），上下用

border-top/bottom:

1px solid var(--border-subtle)

hairline 包夹 - 4 个 component card：λ / D / H_δ / a, b；每张内含大字符号（accent / ink）+ t-meta + KaTeX 块 + 1 行说明 + mono 数值底部块（M-series 取值、a=100 b=2 - 4 component 之下加 accent strip（

background:var(--accent);color:#fff;padding:1.4vh 1.4vw

），左侧

t-meta "OUR EXTENSION · M2-ROBUST-γ"

，右侧 KaTeX

$$L_{M2\text{-}robust,\gamma}=L_{M2}+\gamma\,\mathrm{Var}(\hat y)$$

+ γ 取值 mono 行 - 严格 grep

appendix_A_loss_definitions.tex

确认每条公式与论文 LaTeX 一致；保留 hybrid loss surfaces 图作为右下 thumb（如版面允许）或移除（layouts S25 不要求图）

Test requirements

：在 Chrome 打开 deck → 进入 P07 → 验证（i）所有公式渲染为数学符号而非 raw

$$

；（ii）4 个 card 等高、底部 mono 数值贴底；（iii）accent strip 不挤进 nav 安全区。

Demo

：打开

index.html

，→ 第 7 页，能看到一条主公式横贯页面、4 个等高 component card、底部 IKB 横条标 OUR EXTENSION + M2-robust-γ 公式；投影读取距离 5m 公式仍清晰。

Task 2: P08 加 Evaluation 公式条带（Sharpe / CV / Cap）

Objective

：在 P08（Portfolio Construction & Evaluation）右下当前 METRICS span-4 块替换为含 KaTeX 的紧凑公式条。

Implementation guidance

：

- 替换当前 "MSE, MedSE, R² as scale diagnostic; monthly LS return..." 文字为：3 个紧凑公式 row，每 row 一个 t-meta 标签 + KaTeX inline display - SHARPE:

$$\mathrm{Sharpe}=\sqrt{12}\,\bar r / \sigma_r$$

- CV:

$$\mathrm{CV}=\sigma_S / |\mu_S|$$

- CAP:

$$w_i \le 0.05$$ per-name

- 公式 font ~0.92vw；左侧 1px hairline border-left 与上方 SIGNAL/WEIGHTS 块视觉对齐 - 保留"R² as scale diagnostic"作为下方 t-meta 注释一行

Test requirements

：刷新页面 → P08 → 三条公式按学术风格渲染（√、上下标、绝对值符号正确）。

Demo

：第 8 页右下角紧凑展示三条评估公式，在投影上一眼能读出 √12 退化系数。

Task 3: P11 改造为 S24 + S23 7-row baseline table（fig5_1 + Table 5.1）

Objective

：把 P11 当前的"图 7 列 + 4 段文字 5 列"改为"图 7 列 + 7 行 baseline table 5 列"，所有 7 个 loss 的 Sharpe/Cum/R² 全展示。

Implementation guidance

： - 左 span-7 保留现

images/11-baseline-comparison.png

16:10 contain 不变，下方 caption 改为 "FIG 5.1 + TABLE 5.1 · SEED 42 · 24-MONTH OOS" - 右 span-5 用

<table class="acad-table">

：cols = Loss / Sharpe / Cum / Avg R²；7 rows -

best row

=

<tr class="acad-row-best"><td>hybrid_mul_m1</td><td>0.4435</td><td>+5.09%</td><td>−4.79</td></tr>

-

second row

=

<tr class="acad-row-second"><td>GMADL</td><td>0.2025</td><td>+2.79%</td><td>−7.0×10⁹</td></tr>

- 单元字号若投影偏小可局部覆盖

font-size:1.55vh

（不改全局 CSS） - 表下加 1 行 t-meta：FIG / TABLE 数据归属 - 底部如版面有余加一句

body-sm

："R² and Sharpe decouple under directional objectives → motivates hybrid design."

Test requirements

：浏览器看 P11 → 表 7 行整齐对齐，hybrid_mul_m1 行有 IKB 短色块 + 文字加粗；R² 列用 1e9 科学计数不超出列宽。

Demo

：第 11 页同时展示 fig5_1 + 完整 7 行 baseline 数据，观众一眼能读出哪个 loss best、哪个 R² 爆负。

Task 4: P12 改造为 S24 + S23 9-row A/M table（fig5_2 + Table 5.2）

Objective

：P12 当前 4 段文字改为 9 行 A/M 完整表，让 A1–A5 + M1–M4 全部数据可见。

Implementation guidance

： - 左 span-7 保留

images/12-phase15-variants.png

不变 - 右 span-5 嵌

<table class="acad-table">

：cols = Variant / Loss ID / Sharpe / Cum return；9 rows - best=A3 (

acad-row-best

, hybrid_add_a3, 0.5738, +8.13%)；second=M1 (

acad-row-second

, hybrid_mul_m1, 0.4435, +5.09%) - A5 / M3 / M4 collapse 行允许保留默认样式（不专门标 collapse 色，让 best 视觉聚焦） - 表下方一行 body-sm："M3 collapse (−0.97) shows the parameterisation is sensitive → motivates multi-seed γ refinement." - 数据全部源自 chapter5 §5.3 Table 5.2

Test requirements

：浏览器看 P12 → 9 行表整齐；A3 行加粗 IKB 强调，M1 行 hairline 强调。

Demo

：第 12 页展示 fig5_2 + 9 行完整 A/M 数据；负责评分的老师可以核对每一行数字与论文 Table 5.2 一致。

Task 5: P13 扩展为 6 列、字号上调

Objective

：P13 当前

acad-table

是 4 列（Loss/Sharpe/CV/Cum），缺 std 列；扩到 6 列（Loss / Runs / Sharpe mean / Sharpe std / Cum return mean / CV），并把字号从 1.25vh 提到 1.55-1.6vh（局部覆盖）。

Implementation guidance

： - 在现有

<table class="acad-table">

thead 前加列：

<th>Runs</th>

居中 3 /

<th>Sharpe std</th>

；总 6 列 - tbody 5 行加数据：γ03 (3, 0.3234, 0.3418, +8.18%, 1.0570) / γ05 (3, 0.7054, 0.1488, +23.92%, 0.2109) /

γ07 best

(3,

0.9156

, 0.1655,

+27.99%

,

0.1808

) / γ10 (3, 1.0043, 0.5638, +23.68%, 0.5613) / γ15 (3, 0.8163, 0.3724, +22.77%, 0.4562) - best=γ07 行用

acad-row-best

；second=γ10 用

acad-row-second

- table inline style 加

font-size:1.55vh

（局部不改全局 token） - 删除现"gamma10 has higher mean Sharpe..."body-sm 一段（信息已在表中），改成更短的一句："γ=0.7 simultaneously minimises CV and maximises cumulative return → joint criterion winner."

Test requirements

：浏览器看 P13 → 6 列整齐，γ07 行 IKB 强调；与 fig5_4 三联调参曲线一一映射。

Demo

：第 13 页表 6 列完整呈现 mean+std+CV+Cum，比当前 4 列信息密度提高 50%。

Task 6: P14 替换 frontier 图为 selected-rows S23 table（fig5_5 + Table 5.4 selected）

Objective

：P14 当前 2 张图并排（imadl + frontier），把右图替换为 9 行 selected integrated summary table，让 β/λ collapse 和 α06 peak 数据并列可见。

Implementation guidance

： - 左 span-6 保留

images/14-imadl-alpha-sweep.png

+ caption "FIG 5.5 · IMADL-M2 ALPHA SWEEP" - 右 span-6 移除

images/14-sharpe-cv-frontier.png

改为

<table class="acad-table">

9 行 × 4 列：Loss / Sharpe mean / Cum return / CV - 选 9 行（按论文 Table 5.4 selected）：alpha04 / alpha05 /

alpha06 best

/ alpha07 / alpha08 /

gamma10 second

/ lambda10 / beta05 / beta07 - alpha06 =

acad-row-best

(0.6895, +30.42%, 0.2443)；gamma10 =

acad-row-second

(1.0043, +23.68%, 0.5613) - β05/β07/λ10 行的 CV 是 10/140/1.5 这种"CV explosion"列，让观众一眼看到 collapse - 底部一行 body-sm："α06 corroborates direction from another family; β/λ families collapse → confirms productive region is hybrid_mul + robust."

Test requirements

：浏览器看 P14 → 左图 + 右表对齐；表 9 行 CV 跨度 0.18 → 140 让 collapse 视觉冲击力够强。

Demo

：第 14 页用一张图 + 一个表覆盖整个 integrated sweep 的核心证据，包括 α06 peak、β family collapse、λ family 不及格——比当前两图无表的版本论证力强。

Task 7: P15 改造为 S24 + 完整 Normalisation probe table（fig5_7 + Table 5.5）

Objective

：P15 当前右 5 列三个箭头文字卡改为 3 行 × 5 列完整 Table 5.5，加上论文里关键的 Scale ratio 列和 Per-seed Sharpes 列。

Implementation guidance

： - 左 span-7 保留

images/15-normalisation-probe.png

+ caption "FIG 5.7 + TABLE 5.5 · DIAGNOSTIC PROBE" - 右 span-5 改

<table class="acad-table">

3 行 × 5 列：Loss / Scale ratio (dir vs MSE) / Original / Normalised / Per-seed Sharpes -

γ07 best

：113× / 0.9156 / 0.9112 / 0.60, 1.41, 0.73 - γ10：113× / 1.0043 / 0.4072 / 0.63, 0.12, 0.48 - α06：34× / 0.6895 /

−0.0161

(高亮 collapse) / 0.56,

−0.83

, 0.22 - 表下一行 t-meta："SCALE RATIOS DIAGNOSTICS-ESTIMATED · PER-COMPONENT LOGGER NOT YET IMPLEMENTED"（这是论文明确的 limitation，必须保留） - 底部 body-sm："Only γ07 is approximately stable under the diagnostic probe (within per-seed dispersion). γ10 and α06 degrade materially → γ07 is not a scale artefact."

Test requirements

：浏览器看 P15 → 表 5 列对齐；γ07 行 IKB 强调；α06 的 −0.0161 用红/警示色（或保持默认 hairline）让观众看到 collapse；scale ratio diagnostic 注脚清晰。

Demo

：第 15 页用一张图 + 完整 5 列表呈现 normalisation probe 的全部诊断数据（包括论文核心的 113× 和 34× scale ratio）。

Task 8: P02 / P04 / P09 过空页内容补强（不改 CSS）

Objective

：在不动全局字号的前提下，对当前过空的三页加少量论文锚点信息填补底部空白。

Implementation guidance

： -

P02

：当前底部 4-strip "ML FOR RETURNS / MSE DEFAULT / RANKING OBJECTIVE / HEAVY-TAILED RETURNS" 之下，加一行 t-meta citation："Gu Kelly Xiu (2020) · Daniel & Moskowitz (2016) · Huber (1964) · Michańków et al. (2024)" -

P04

：当前 timeline 下方 t-meta 之上加一行"chapter anchor"：

§3.1 Research Design · §3.2 Architecture · §3.5 Portfolio · §3.6 Metrics

-

P09

：当前 timeline 下方 t-meta "All headline rows..." 之上加 evidence-strength tag 行：5 个 mono tag 横排 —

P1: SEED 42

/

P2: SEED 42

/

P3a: 3 SEEDS

/

P3b: 3 SEEDS

/

P4: 3 SEEDS + DIAGNOSTIC

，每个 tag 用

border:1px solid var(--ink); padding:.4vh .6vw; font-family:var(--mono); font-size:1vh

-

不

改任何全局 CSS token；只局部 inline style

Test requirements

：浏览器看 P02/P04/P09 → 各加了一行论文锚点信息；整体留白比例从 ~50% 降到 ~35%。

Demo

：三个原本最空的方法论 / 引言页现在底部各有一行论文章节 / 引用 / 阶段 evidence 标签，让观众能即时知道每个论点对应论文哪一段。

Task 9: 浏览器逐页截图审核 + 修订

Objective

：用 headless Chrome（或本地 Chrome）打开

index.html

，逐页截图审核效果，确认所有公式渲染、表格对齐、accent 高亮、字号在 1080p 投影下可读，并修订发现的问题。

Implementation guidance

： - 用 Playwright 或 Puppeteer headless 模式，viewport 1920×1080，启动后 wait 2s（KaTeX defer 渲染）；逐页

page.keyboard.press('ArrowRight')

+

page.screenshot()

输出 19 张 PNG 到

ppt_package/fyp_oral_presentation/screenshots/

- 逐张审核 4 个维度：(i) 公式是否渲染成数学符号（不出现 raw

$$

）；(ii) 表格是否 7/9/5/9/3 行完整、best/second 行高亮正确；(iii) accent strip / 横条不挤进底部 nav 安全区；(iv) 投影距离 5-8m 假设（看 1/4 缩略图）数字仍清晰 - 发现问题就回 Task 1–8 对应任务里 fix；最多两轮迭代

Test requirements

：19 张 PNG 全部生成；每页对应论文章节内容齐全；layouts-swiss-academic.md 自检清单（公式 / 表格 / 图表 / 引用 / 推荐 / 整体）逐项 ✓。

Demo

：交付 19 张 1920×1080 PNG 截图（保存到 screenshots/）+ 一份 markdown 审核报告（

SCREENSHOT_REVIEW.md

）逐页列出"OK / 待 fix"。

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Files Modified（预期）

-

ppt_package/fyp_oral_presentation/index.html

（核心，所有页面改动） - 新增

ppt_package/fyp_oral_presentation/screenshots/*.png

（19 张审核截图） - 新增

ppt_package/fyp_oral_presentation/SCREENSHOT_REVIEW.md

（审核报告）

Files NOT Modified

-

build_deck.mjs

（不重跑 build script，直接编辑 index.html） -

images/*.png

（图片不变） - 全局 CSS 基线 token（不改

body-sm

、

t-meta

、

.acad-table

等定义）