import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, '..', '..');
const skillRoot = join(repoRoot, '.kiro', 'skills', 'guizang-ppt-skill');
const templatePath = join(skillRoot, 'assets', 'template-swiss-academic.html');
const figuresSrc = join(repoRoot, '2253235_yirongyu_2026_Supplementary', 'latex', 'figures');
const imagesDir = join(__dirname, 'images');
const localAssetsDir = join(__dirname, 'assets');

mkdirSync(imagesDir, { recursive: true });
mkdirSync(localAssetsDir, { recursive: true });

const assets = [
  ['fig2_1_loss_shapes.png', '07-loss-shapes.png'],
  ['fig3_2_hybrid_loss_surfaces.png', '06-hybrid-loss-surfaces.png'],
  ['fig3_3_triple_property.png', '07-triple-property.png'],
  ['fig3_4_portfolio_flow.png', '08-portfolio-flow.png'],
  ['fig4_1_data_coverage.png', '10-data-coverage.png'],
  ['fig5_1_baseline_comparison.png', '11-baseline-comparison.png'],
  ['fig5_2_phase15_variants.png', '12-phase15-variants.png'],
  ['fig5_4_gamma_tuning_curve.png', '13-gamma-tuning-curve.png'],
  ['fig5_5_imadl_alpha_sweep.png', '14-imadl-alpha-sweep.png'],
  ['fig5_6_sharpe_cv_frontier.png', '14-sharpe-cv-frontier.png'],
  ['fig5_7_normalisation_probe.png', '15-normalisation-probe.png'],
  ['fig5_8_cumulative_return_paths.png', '17-cumulative-return-paths.png'],
];

for (const [src, dest] of assets) {
  const from = join(figuresSrc, src);
  if (!existsSync(from)) throw new Error(`Missing figure: ${from}`);
  copyFileSync(from, join(imagesDir, dest));
}

copyFileSync(join(skillRoot, 'assets', 'motion.min.js'), join(localAssetsDir, 'motion.min.js'));

const slides = String.raw`
<section class="slide accent" data-layout="SWISS-COVER-ASCII" data-animate="hero">
  <div class="canvas-card">
    <canvas class="ascii-bg" aria-hidden="true"></canvas>
    <div class="chrome-min">
      <div class="l">FYP ORAL PRESENTATION · 2026.05.16</div>
      <div class="r">01 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.4vh">
      <div data-anim="kicker" class="t-meta" style="color:rgba(255,255,255,.78);letter-spacing:.22em">FINAL YEAR PROJECT · LOSS FUNCTION DESIGN</div>
      <div data-anim="title" style="align-self:center;display:grid;grid-template-columns:1fr auto;gap:4vw;align-items:start">
        <div style="display:flex;flex-direction:column;gap:2.4vh;align-self:center">
          <h1 style="font-family:var(--sans);font-weight:200;font-size:min(5.4vw,9vh);line-height:1.0;letter-spacing:-.022em;color:#fff;margin:0">Multiplicative <span style="font-style:italic;font-weight:300">Directional-Robust</span> Loss<br/><span style="font-weight:200;font-size:.62em;color:rgba(255,255,255,.86);line-height:1.18;display:inline-block;margin-top:.4em">for Cross-Sectional Stock-Return Prediction</span></h1>
          <div data-anim="lead" class="lead" style="max-width:60ch;color:rgba(255,255,255,.9);font-weight:300;font-size:min(1.55vw,2.6vh);line-height:1.3">Under a fixed portfolio protocol — same data, same MLP, same evaluation. Only the loss changes.</div>
        </div>
        <aside style="display:flex;flex-direction:column;gap:1.6vh;border-left:1px solid rgba(255,255,255,.32);padding:.5vh 0 .5vh 1.8vw;min-width:18ch;align-self:center">
          <div style="display:flex;flex-direction:column;gap:.3vh">
            <div class="t-meta" style="color:rgba(255,255,255,.6);letter-spacing:.22em">SPEAKER</div>
            <div style="font-family:var(--sans);font-weight:300;font-size:min(1.9vw,3.2vh);color:#fff;line-height:1.15">Yirong Yu</div>
            <div style="font-family:var(--mono);font-weight:400;font-size:min(1.1vw,1.9vh);color:rgba(255,255,255,.78);letter-spacing:.04em">2253235</div>
          </div>
          <div style="display:flex;flex-direction:column;gap:.3vh">
            <div style="font-family:var(--sans);font-weight:300;font-size:min(1.25vw,2.1vh);color:rgba(255,255,255,.92);line-height:1.3">BSc Financial Mathematics</div>
          </div>
          <div style="display:flex;flex-direction:column;gap:.3vh">
            <div class="t-meta" style="color:rgba(255,255,255,.6);letter-spacing:.22em">SUPERVISOR</div>
            <div style="font-family:var(--sans);font-weight:300;font-size:min(1.45vw,2.4vh);color:#fff;line-height:1.2">Dr. Yi Cao</div>
          </div>
          <div style="display:flex;flex-direction:column;gap:.3vh">
            <div class="t-meta" style="color:rgba(255,255,255,.6);letter-spacing:.22em">INSTITUTION</div>
            <div style="font-family:var(--sans);font-weight:300;font-size:min(1.05vw,1.85vh);color:rgba(255,255,255,.92);line-height:1.3">XJTLU<br/>School of Mathematics &amp; Physics</div>
          </div>
        </aside>
      </div>
      <div data-anim="bottom" style="border-top:1px solid rgba(255,255,255,.22);padding-top:1.4vh">
        <div class="t-meta" style="color:rgba(255,255,255,.55);letter-spacing:.22em">FEBRUARY 2026 · XJTLU SCHOOL OF MATHEMATICS &amp; PHYSICS</div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S18" data-animate="why-now">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">BACKGROUND · RESEARCH QUESTIONS</div>
      <div class="r">02 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto auto 1fr auto;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1.1vh">
        <div class="t-meta">BACKGROUND &amp; THREE QUESTIONS</div>
        <h2 class="h-xl" style="font-size:min(4.7vw,8.2vh);font-weight:200;line-height:.98;color:var(--ink);margin:0">Return models optimise error.<br/>Portfolios trade ranks.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.1vw;align-items:stretch">
        <div class="span-3" style="border-top:1.5px solid var(--ink);padding-top:1.2vh;display:flex;flex-direction:column;gap:.8vh">
          <div class="t-meta">01 · CONTEXT</div>
          <div class="body-sm">ML is widely used for cross-sectional return prediction and long-short portfolios.</div>
        </div>
        <div class="span-3" style="border-top:1.5px solid var(--ink);padding-top:1.2vh;display:flex;flex-direction:column;gap:.8vh">
          <div class="t-meta">02 · DEFAULT</div>
          <div class="body-sm">Most studies tune architectures and features, while MSE remains the default loss.</div>
        </div>
        <div class="span-3" style="border-top:1.5px solid var(--accent);padding-top:1.2vh;display:flex;flex-direction:column;gap:.8vh">
          <div class="t-meta" style="color:var(--accent)">03 · MISMATCH</div>
          <div class="body-sm">Portfolio construction depends on ranking, not calibrated values, and monthly returns are heavy-tailed.</div>
        </div>
        <div class="span-3" style="border-top:1.5px solid var(--ink);padding-top:1.2vh;display:flex;flex-direction:column;gap:.8vh">
          <div class="t-meta">04 · SETUP</div>
          <div class="body-sm">The talk asks three questions now and answers them at the end.</div>
        </div>
      </div>
      <div class="grid-12" style="gap:1.8vw;align-items:stretch">
        <div class="span-4" style="border-top:1.5px solid var(--ink);padding-top:1.6vh;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">RQ1</div>
          <div style="font-size:2.2vh;font-weight:300;line-height:1.25;color:var(--ink)">How does loss choice affect prediction-level and portfolio-level performance?</div>
          <div style="font-family:var(--mono);font-size:1.15vh;color:var(--text-secondary);margin-top:auto">MSE / MEDSE / MADL / GMADL / IMADL</div>
        </div>
        <div class="span-4" style="border-top:1.5px solid var(--accent);padding-top:1.6vh;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta" style="color:var(--accent)">RQ2</div>
          <div style="font-size:2.2vh;font-weight:300;line-height:1.25;color:var(--ink)">Which hybrid design gives the best Sharpe-stability trade-off?</div>
          <div style="font-family:var(--mono);font-size:1.15vh;color:var(--text-secondary);margin-top:auto">A-SERIES / M-SERIES / GAMMA SWEEP</div>
        </div>
        <div class="span-4" style="border-top:1.5px solid var(--ink);padding-top:1.6vh;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">RQ3</div>
          <div style="font-size:2.2vh;font-weight:300;line-height:1.25;color:var(--ink)">Do the leading candidates remain stable under component normalisation diagnostics?</div>
          <div style="font-family:var(--mono);font-size:1.15vh;color:var(--text-secondary);margin-top:auto">DIAGNOSTIC PROBE · NOT A FINAL PROOF</div>
        </div>
      </div>
      <div class="grid-12" style="gap:0;border-top:1px solid var(--border-subtle);padding-top:1.4vh">
        <div class="span-3" style="font-family:var(--mono);font-size:1.3vh">ML FOR RETURNS</div>
        <div class="span-3" style="font-family:var(--mono);font-size:1.3vh">MSE DEFAULT</div>
        <div class="span-3" style="font-family:var(--mono);font-size:1.3vh">RANKING OBJECTIVE</div>
        <div class="span-3" style="font-family:var(--mono);font-size:1.3vh;color:var(--accent)">HEAVY-TAILED RETURNS</div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S08" data-animate="duo-mirror">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">LITERATURE SNAPSHOT</div>
      <div class="r">03 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.8vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">FOUR QUADRANTS · SAME PROTOCOL GAP</div>
        <h2 class="h-xl" style="font-size:min(4.5vw,8vh);font-weight:200;line-height:1;color:var(--ink);margin:0">What prior work solved,<br/>and what it left open.</h2>
      </div>
      <div class="duo-compare" data-anim="up" style="display:grid;grid-template-columns:1.08fr .92fr;gap:2vw;align-items:stretch">
        <div class="col" style="display:flex;flex-direction:column;gap:1.1vh;border-top:1.5px solid var(--ink);padding-top:1.5vh">
          <div class="t-meta">THREE LITERATURE STREAMS</div>
          <div style="display:grid;grid-template-columns:1fr;gap:.9vh">
            <div style="border-left:2px solid var(--ink);padding-left:1vw"><div class="t-meta">ML FOR RETURN PREDICTION</div><div class="body-sm">Gu, Kelly, and Xiu; Daniel-Moskowitz. Nonlinear ML improves return prediction, but loss choice usually stays at MSE.</div></div>
            <div style="border-left:2px solid var(--border-strong);padding-left:1vw"><div class="t-meta">ROBUST LOSSES</div><div class="body-sm">Huber and MedSE limit heavy-tail influence, but they do not encode directional correctness.</div></div>
            <div style="border-left:2px solid var(--accent);padding-left:1vw"><div class="t-meta" style="color:var(--accent)">DIRECTIONAL LOSSES</div><div class="body-sm">MADL / GMADL reward sign alignment, but need magnitude calibration for portfolio evaluation.</div></div>
          </div>
        </div>
        <div class="col" style="display:flex;flex-direction:column;gap:1vh;border-top:1.5px solid var(--accent);padding-top:1.5vh">
          <div class="t-meta" style="color:var(--accent)">2×2 MAP OF THE GAP</div>
          <div style="display:grid;grid-template-columns:auto 1fr 1fr;grid-template-rows:auto 1fr 1fr;gap:.5vw;flex:1;min-height:0">
            <div></div><div class="t-meta" style="text-align:center">STANDARD</div><div class="t-meta" style="text-align:center;color:var(--accent)">ROBUST</div>
            <div class="t-meta" style="writing-mode:vertical-rl;transform:rotate(180deg);align-self:center">REGRESSION</div>
            <div class="card-fill" style="padding:1.2vh 1vw"><div style="font-family:var(--mono);font-size:2vh">MSE</div><div class="body-sm">default baseline</div></div>
            <div class="card-fill" style="padding:1.2vh 1vw"><div style="font-family:var(--mono);font-size:2vh">MedSE</div><div class="body-sm">heavy-tail robust</div></div>
            <div class="t-meta" style="writing-mode:vertical-rl;transform:rotate(180deg);align-self:center;color:var(--accent)">DIRECTIONAL</div>
            <div class="card-fill" style="padding:1.2vh 1vw"><div style="font-family:var(--mono);font-size:2vh">MADL / GMADL</div><div class="body-sm">sign signal</div></div>
            <div style="background:var(--accent);color:#fff;padding:1.2vh 1vw"><div class="t-meta" style="color:rgba(255,255,255,.78)">THIS PROJECT</div><div style="font-family:var(--mono);font-size:2vh">Hybrid robust-directional</div></div>
          </div>
        </div>
      </div>
      <div class="t-meta" style="border-top:1px solid var(--border-subtle);padding-top:1.2vh;color:var(--text-secondary)">No prior work compares all four quadrants under the same protocol — that is the gap.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">WHAT WE DID</div>
      <div class="r">04 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:3vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">CONTROLLED PIPELINE</div>
        <h2 class="h-xl" style="font-size:min(4.8vw,8.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Hold the pipeline fixed.<br/>Move only the loss.</h2>
      </div>
      <div class="timeline-h" data-anim="up">
        <div class="tl-row" style="display:block;height:30vh">
          <div class="th-node up" style="position:absolute;left:8%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">01</div><div class="name">Data</div><div class="desc">CRSP-style monthly panel</div></div></div>
          <div class="th-node down" style="position:absolute;left:26%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">02</div><div class="name">MLP</div><div class="desc">15 inputs, frozen widths</div></div></div>
          <div class="th-node up accent" style="position:absolute;left:44%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">03</div><div class="name">Loss</div><div class="desc">Regression, directional, hybrid</div></div></div>
          <div class="th-node down" style="position:absolute;left:62%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">04</div><div class="name">Portfolio</div><div class="desc">Top/bottom 10%, cap05</div></div></div>
          <div class="th-node up" style="position:absolute;left:80%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">05</div><div class="name">Evaluation</div><div class="desc">Sharpe, return, CV</div></div></div>
        </div>
      </div>
      <div class="t-meta" style="color:var(--text-secondary)">The main recommendation is scoped to one static 24-month test window and three-seed robustness rows.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S19" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">CONTRIBUTIONS</div>
      <div class="r">05 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:3vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div style="height:2px;background:var(--accent);width:12vw"></div>
        <div class="t-meta">FOUR CONTRIBUTIONS</div>
        <h2 class="h-xl" style="font-size:min(4.4vw,7.8vh);font-weight:200;line-height:1;color:var(--ink);margin:0">A bounded recommendation,<br/>not an absolute claim.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.5vw;align-items:stretch">
        <div class="span-3 card-fill" style="padding:2.3vh 1.4vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">01 · DESIGN</div><div style="font-size:2vh;font-weight:300;line-height:1.25">Additive and multiplicative hybrid losses.</div><div class="body-sm" style="margin-top:auto">Directional alignment plus robust magnitude control.</div>
        </div>
        <div class="span-3 card-accent" style="background:var(--accent);color:#fff;padding:2.3vh 1.4vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta" style="color:rgba(255,255,255,.78)">02 · VARIANCE</div><div style="font-size:2vh;font-weight:300;line-height:1.25">M2-robust-&gamma; family.</div><div class="body-sm" style="margin-top:auto;color:rgba(255,255,255,.82)">A prediction-variance penalty searches for stability.</div>
        </div>
        <div class="span-3 card-fill" style="padding:2.3vh 1.4vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">03 · EVIDENCE</div><div style="font-size:2vh;font-weight:300;line-height:1.25">Single seed, multi-seed, diagnostic probe.</div><div class="body-sm" style="margin-top:auto">Claims are labelled by evidence strength.</div>
        </div>
        <div class="span-3 card-fill" style="padding:2.3vh 1.4vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">04 · OUTPUT</div><div style="font-size:2vh;font-weight:300;line-height:1.25">Primary, high-return alternative, fallback.</div><div class="body-sm" style="margin-top:auto">Each option carries its caveat.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S17" data-animate="system-diagram">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESEARCH DESIGN · ARCHITECTURE</div>
      <div class="r">06 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.6vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">SINGLE-FACTOR COMPARISON</div>
        <h2 class="h-xl" style="font-size:min(4.4vw,7.8vh);font-weight:200;line-height:1;color:var(--ink);margin:0">The architecture is frozen.<br/>The loss is the treatment.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.8vw;align-items:stretch">
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.2vh;border-top:1.5px solid var(--ink);padding-top:1.5vh">
          <div class="t-meta">CONTROLLED FACTORS</div>
          <div class="grid-12" style="gap:.8vw">
            <div class="span-6 card-fill" style="padding:1.3vh 1vw"><div class="t-meta">DATA</div><div class="body-sm">Same X1 panel</div></div>
            <div class="span-6 card-fill" style="padding:1.3vh 1vw"><div class="t-meta">FEATURES</div><div class="body-sm">15 inputs</div></div>
            <div class="span-6 card-fill" style="padding:1.3vh 1vw"><div class="t-meta">WINDOW</div><div class="body-sm">1995-1996 OOS</div></div>
            <div class="span-6 card-fill" style="padding:1.3vh 1vw"><div class="t-meta">PORTFOLIO</div><div class="body-sm">Top/bottom 10%</div></div>
          </div>
        </div>
        <div class="span-7" style="display:flex;flex-direction:column;gap:1.2vh;border-top:1.5px solid var(--accent);padding-top:1.5vh">
          <div class="t-meta" style="color:var(--accent)">MLP CONFIGURATION</div>
          <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:.8vw;align-items:center">
            <div class="card-fill" style="padding:2vh 1vw"><div class="t-meta">INPUT</div><div style="font-size:2.8vh;font-weight:300">15</div></div>
            <div class="t-meta">TO</div>
            <div class="card-fill" style="padding:2vh 1vw"><div class="t-meta">HIDDEN</div><div style="font-size:2.8vh;font-weight:300">64</div></div>
            <div class="t-meta">TO</div>
            <div class="card-fill" style="padding:2vh 1vw"><div class="t-meta">HIDDEN</div><div style="font-size:2.8vh;font-weight:300">32</div></div>
            <div class="t-meta">TO</div>
            <div class="card-fill" style="padding:2vh 1vw"><div class="t-meta">HIDDEN</div><div style="font-size:2.8vh;font-weight:300">16</div></div>
          </div>
          <div class="body-sm">Phase 1/2 use ReLU with dropout 0.2. Phase 3 keeps the widths but uses tanh and dropout 0.0, so cross-phase comparisons are scoped rather than claimed as exact ablations.</div>
        </div>
      </div>
      <div class="t-meta" style="color:var(--text-secondary);border-top:1px solid var(--border-subtle);padding-top:1vh">Interpretation rule: within-phase comparisons are stronger than direct cross-phase improvement claims.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S25" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">LOSS FUNCTION FAMILIES</div>
      <div class="r">07 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto auto 1fr auto;gap:1.5vh">
      <div data-anim="kicker" class="t-meta">METHODOLOGY · LOSS HIERARCHY</div>
      <h2 data-anim="title" class="h-xl" style="font-size:min(3.55vw,6.6vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Four loss families,<br/>one highlighted design.</h2>
      <div data-anim="components" class="grid-12" style="gap:1.4vw;align-items:stretch">
        <div class="span-5" style="display:grid;grid-template-rows:repeat(5,1fr);gap:.7vh">
          <div class="card-fill" style="padding:1vh 1vw"><div class="t-meta">01 · REGRESSION</div><div class="body-sm">MSE, MedSE: magnitude-oriented baselines.</div></div>
          <div class="card-fill" style="padding:1vh 1vw"><div class="t-meta">02 · DIRECTIONAL</div><div class="body-sm">MADL, GMADL, IMADL: reward sign alignment.</div></div>
          <div class="card-fill" style="padding:1vh 1vw"><div class="t-meta">03 · HYBRID ADDITIVE</div><div class="katex-display" style="font-size:.72vw;margin:0;text-align:left">$$L=\lambda_{dir}D+\lambda_{hub}H_\delta$$</div></div>
          <div class="card-fill" style="padding:1vh 1vw"><div class="t-meta">04 · HYBRID MULTIPLICATIVE</div><div class="katex-display" style="font-size:.72vw;margin:0;text-align:left">$$L=(1+\lambda_{dir}D)H_\delta$$</div></div>
          <div style="background:var(--accent);color:#fff;padding:1vh 1vw"><div class="t-meta" style="color:rgba(255,255,255,.78)">05 · M2-ROBUST-γ</div><div class="katex-display" style="font-size:.72vw;margin:0;text-align:left;color:#fff">$$L_{M2\text{-}robust}=L_{M2}+\gamma\operatorname{Var}(\hat{y})$$</div></div>
        </div>
        <div class="span-7" style="display:grid;grid-template-rows:1fr auto;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/06-hybrid-loss-surfaces.png" data-image-slot="s24-half-figure-contain" alt="Hybrid loss surfaces">
          </div>
          <div class="grid-12" style="gap:0;border-top:1px solid var(--border-subtle);padding-top:1vh">
            <div class="span-4" style="padding-right:1vw;border-right:1px solid var(--border-subtle)"><div class="t-meta">DIRECTION</div><div class="body-sm">Wrong sign receives a larger penalty.</div></div>
            <div class="span-4" style="padding:0 1vw;border-right:1px solid var(--border-subtle)"><div class="t-meta">ROBUST MAGNITUDE</div><div class="body-sm">Huber controls heavy-tail residuals.</div></div>
            <div class="span-4" style="padding-left:1vw"><div class="t-meta" style="color:var(--accent)">STABILITY</div><div class="body-sm">γ penalises prediction variance.</div></div>
          </div>
        </div>
      </div>
      <div class="t-meta" style="color:var(--text-secondary);border-top:1px solid var(--border-subtle);padding-top:1vh">Highlighted object for refinement: M2-robust-γ, evaluated at γ ∈ {0.3, 0.5, 0.7, 1.0, 1.5}.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">PORTFOLIO CONSTRUCTION · EVALUATION</div>
      <div class="r">08 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto auto 1fr auto;gap:2vh">
      <div data-anim="kicker" class="t-meta">SAME PORTFOLIO RULE FOR EVERY LOSS</div>
      <h2 data-anim="title" class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Predictions become a capped<br/>long-short portfolio.</h2>
      <div data-anim="image" class="frame-img r-21x9 fit-contain" style="background:var(--paper);border-top:1px solid var(--border-subtle);border-bottom:1px solid var(--border-subtle)">
        <img src="images/08-portfolio-flow.png" data-image-slot="s22-hero-21x9" alt="Portfolio construction flow">
      </div>
      <div data-anim="kpi" class="grid-12" style="gap:0;border-top:1.5px solid var(--ink);padding-top:1.4vh">
        <div class="span-4" style="border-right:1px solid var(--border-subtle);padding-right:1.2vw"><div class="t-meta">SIGNAL</div><div class="body-sm">Rank predictions into top and bottom deciles.</div></div>
        <div class="span-4" style="border-right:1px solid var(--border-subtle);padding:0 1.2vw"><div class="t-meta">WEIGHTS</div><div class="body-sm">Within-bucket z-score, clipped to [-3, 3].</div></div>
        <div class="span-4" style="padding-left:1.2vw"><div class="t-meta">METRICS</div><div class="body-sm">MSE, MedSE, R<sup>2</sup> as scale diagnostic; monthly LS return, annualised Sharpe, cumulative return, cross-seed CV.</div></div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S11" data-animate="timeline-walk">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">EXPERIMENTAL PHASES</div>
      <div class="r">09 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:3vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">FROM MOTIVATION TO ROBUSTNESS</div>
        <h2 class="h-xl" style="font-size:min(4.4vw,7.8vh);font-weight:200;line-height:1;color:var(--ink);margin:0">The empirical design is staged,<br/>not one pooled leaderboard.</h2>
      </div>
      <div class="timeline-h" data-anim="up">
        <div class="tl-row" style="display:block;height:30vh">
          <div class="th-node up" style="position:absolute;left:8%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">P1</div><div class="name">Baselines</div><div class="desc">7 losses · seed 42</div></div></div>
          <div class="th-node down" style="position:absolute;left:26%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">P2</div><div class="name">A/M sweep</div><div class="desc">9 hybrids · seed 42 motivation</div></div></div>
          <div class="th-node up accent" style="position:absolute;left:44%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">P3a</div><div class="name">Gamma</div><div class="desc">5 values · 3 seeds per row</div></div></div>
          <div class="th-node down" style="position:absolute;left:62%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">P3b</div><div class="name">Integrated</div><div class="desc">alpha / beta / lambda sweeps</div></div></div>
          <div class="th-node up" style="position:absolute;left:80%;top:50%;transform:translateY(-50%)"><div class="dot"></div><div class="label"><div class="yr">P4</div><div class="name">Probe</div><div class="desc">Diagnostic normalisation</div></div></div>
        </div>
      </div>
      <div class="t-meta" style="color:var(--text-secondary)">All headline rows use test 1995-01..1996-12. Phase 1/2 tables are single-seed; Phase 3/4 tables are 3-seed grouped rows.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S22" data-animate="image-hero">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">DATA · FEATURES</div>
      <div class="r">10 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto auto 1fr auto;gap:2vh">
      <div data-anim="kicker" class="t-meta">CRSP-STYLE MONTHLY US EQUITY PANEL</div>
      <h2 data-anim="title" class="h-xl" style="font-size:min(4.2vw,7.5vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Heavy-tailed returns are<br/>part of the problem.</h2>
      <div data-anim="image" class="frame-img r-21x9 fit-contain" style="background:var(--paper);border-top:1px solid var(--border-subtle);border-bottom:1px solid var(--border-subtle)">
        <img src="images/10-data-coverage.png" data-image-slot="s22-hero-21x9" alt="Data coverage and training-era return distribution">
      </div>
      <div data-anim="kpi" class="grid-12" style="gap:0;border-top:1.5px solid var(--ink);padding-top:1.4vh">
        <div class="span-3" style="border-right:1px solid var(--border-subtle);padding-right:1vw"><div class="t-meta">MONTHLY PANEL</div><div style="font-family:var(--mono);font-size:2.25vh">~2,000 stocks/mo.</div></div>
        <div class="span-3" style="border-right:1px solid var(--border-subtle);padding:0 1vw"><div class="t-meta">X1 FEATURES</div><div class="body-sm">5 base columns + 10 cumulative return / turnover features.</div></div>
        <div class="span-3" style="border-right:1px solid var(--border-subtle);padding:0 1vw"><div class="t-meta">OOS WINDOW</div><div style="font-family:var(--mono);font-size:1.85vh">1995-01..1996-12</div><div class="body-sm">24 months.</div></div>
        <div class="span-3" style="padding-left:1vw"><div class="t-meta">DATA HYGIENE</div><div class="body-sm">No look-ahead; delisting observations retained where available.</div></div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S24" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESULTS · PHASE 1</div>
      <div class="r">11 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">BASELINE LOSS COMPARISON · SEED 42</div>
        <h2 class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Calibration metrics and portfolio metrics<br/>separate sharply.</h2>
      </div>
      <div class="grid-12" data-anim="up" style="gap:2vw;align-items:start">
        <div class="span-7" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/11-baseline-comparison.png" data-image-slot="s24-half-figure-contain" alt="Phase 1 baseline comparison">
          </div>
          <div class="t-meta" style="font-size:.95vh;color:var(--text-secondary)">FIG 5.1 · SEED 42 · 24-MONTH OOS · CAP05</div>
        </div>
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.4vh">
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">MSE</div><div class="body-sm">Sharpe -0.4643, cumulative return -11.25%.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">GMADL</div><div class="body-sm">Sharpe +0.2025 despite average R<sup>2</sup> around -7.02e9. Rank signal and scale calibration diverge.</div></div>
          <div style="border-top:1px solid var(--accent);padding-top:1.2vh"><div class="t-meta" style="color:var(--accent)">HYBRID_MUL_M1</div><div class="body-sm">Seed-42 baseline peak: Sharpe 0.4435, cumulative return +5.09%, lowest monthly LS standard deviation.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">BOUNDARY</div><div class="body-sm">This motivates hybrid design. It is not yet multi-seed robustness evidence.</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S24" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESULTS · PHASE 2</div>
      <div class="r">12 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">HYBRID A/M SWEEP · SEED 42</div>
        <h2 class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Some hybrids help.<br/>Some collapse.</h2>
      </div>
      <div class="grid-12" data-anim="up" style="gap:2vw;align-items:start">
        <div class="span-7" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/12-phase15-variants.png" data-image-slot="s24-half-figure-contain" alt="Phase 2 hybrid variants">
          </div>
          <div class="t-meta" style="font-size:.95vh;color:var(--text-secondary)">FIG 5.2 · A1-A5 AND M1-M4 · SEED 42</div>
        </div>
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.4vh">
          <div style="border-top:1px solid var(--accent);padding-top:1.2vh"><div class="t-meta" style="color:var(--accent)">A3 PEAK</div><div class="body-sm">hybrid_add_a3 reaches Sharpe 0.5738 and cumulative return +8.13% at seed 42.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">M1</div><div class="body-sm">M-series peak Sharpe 0.4435 and comparatively low volatility.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">M3 COLLAPSE</div><div class="body-sm">Sharpe -0.9691 shows the parameterisation is sensitive.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">USE OF THIS SLIDE</div><div class="body-sm">Read as seed-42 design motivation for the multi-seed gamma study, not a final ranking.</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S24" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESULTS · PHASE 3a</div>
      <div class="r">13 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">GAMMA REFINEMENT · THREE SEEDS PER ROW</div>
        <h2 class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">&gamma;=0.7 gives the best<br/>Sharpe-stability balance.</h2>
      </div>
      <div class="grid-12" data-anim="up" style="gap:2vw;align-items:start">
        <div class="span-7" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/13-gamma-tuning-curve.png" data-image-slot="s24-half-figure-contain" alt="Gamma tuning curve">
          </div>
          <div class="t-meta" style="font-size:.95vh;color:var(--text-secondary)">FIG 5.4 · SHARPE / CV / PORTFOLIO VOLATILITY</div>
        </div>
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.2vh">
          <table class="acad-table" style="font-size:1.25vh">
            <thead><tr><th>Loss</th><th>Sharpe</th><th>CV</th><th>Cum.</th></tr></thead>
            <tbody>
              <tr><td>gamma03</td><td>0.3234</td><td>1.0570</td><td>+8.18%</td></tr>
              <tr><td>gamma05</td><td>0.7054</td><td>0.2109</td><td>+23.92%</td></tr>
              <tr class="acad-row-best"><td>gamma07</td><td>0.9156</td><td>0.1808</td><td>+27.99%</td></tr>
              <tr class="acad-row-second"><td>gamma10</td><td>1.0043</td><td>0.5613</td><td>+23.68%</td></tr>
              <tr><td>gamma15</td><td>0.8163</td><td>0.4562</td><td>+22.77%</td></tr>
            </tbody>
          </table>
          <div class="body-sm">gamma10 has higher mean Sharpe, but roughly three times the CV of gamma07. The recommendation is therefore based on the joint criterion, not Sharpe alone.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S16" data-animate="field-notes">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESULTS · PHASE 3b</div>
      <div class="r">14 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.2vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">INTEGRATED ALPHA / BETA / LAMBDA SWEEPS</div>
        <h2 class="h-xl" style="font-size:min(4vw,7.2vh);font-weight:200;line-height:1;color:var(--ink);margin:0">The productive region is<br/>hybrid-multiplicative plus robust control.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.4vw;align-items:stretch">
        <div class="span-6" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="border:1px solid var(--border-subtle);background:var(--paper)">
            <img src="images/14-imadl-alpha-sweep.png" data-image-slot="s16-brief-contain" alt="IMADL alpha sweep">
          </div>
          <div class="t-meta">IMADL-M2 ALPHA SWEEP · alpha06 Sharpe 0.6895 · CV 0.2443 · Cum +30.42%</div>
        </div>
        <div class="span-6" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="border:1px solid var(--border-subtle);background:var(--paper)">
            <img src="images/14-sharpe-cv-frontier.png" data-image-slot="s16-brief-contain" alt="Sharpe CV frontier">
          </div>
          <div class="t-meta">FRONTIER · gamma07 highest Sharpe among low-CV points</div>
        </div>
      </div>
      <div class="body-sm" style="border-top:1px solid var(--border-subtle);padding-top:1vh">alpha06 corroborates the direction from another family, but it is a fallback, not the Sharpe leader.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S24" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RESULTS · PHASE 4</div>
      <div class="r">15 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">LOSS-COMPONENT NORMALISATION PROBE</div>
        <h2 class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Normalisation is diagnostic.<br/>It is not a universal fix.</h2>
      </div>
      <div class="grid-12" data-anim="up" style="gap:2vw;align-items:start">
        <div class="span-7" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/15-normalisation-probe.png" data-image-slot="s24-half-figure-contain" alt="Normalisation probe">
          </div>
          <div class="t-meta" style="font-size:.95vh;color:var(--text-secondary)">FIG 5.7 · SCALE RATIOS ARE DIAGNOSTICS-ESTIMATED</div>
        </div>
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.2vh">
          <div style="border-top:1px solid var(--accent);padding-top:1.2vh"><div class="t-meta" style="color:var(--accent)">gamma07</div><div style="font-family:var(--mono);font-size:2.4vh">0.9156 &rarr; 0.9112</div><div class="body-sm">Approximately stable under the diagnostic probe.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">gamma10</div><div style="font-family:var(--mono);font-size:2.4vh">1.0043 &rarr; 0.4072</div><div class="body-sm">High-Sharpe alternative is scale-sensitive.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">alpha06</div><div style="font-family:var(--mono);font-size:2.4vh">0.6895 &rarr; -0.0161</div><div class="body-sm">Fallback loses performance after normalisation.</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S24" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">HEADLINE FINDINGS · CUMULATIVE PATHS</div>
      <div class="r">16 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:2.4vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">RESULTS SYNTHESIS · FIG 5.8</div>
        <h2 class="h-xl" style="font-size:min(4.1vw,7.4vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Only gamma07 is positive,<br/>stable, and probe-resistant.</h2>
      </div>
      <div class="grid-12" data-anim="up" style="gap:2vw;align-items:start">
        <div class="span-7" style="display:flex;flex-direction:column;gap:1vh">
          <div class="frame-img r-16x10 fit-contain" style="background:var(--paper);border:1px solid var(--border-subtle)">
            <img src="images/17-cumulative-return-paths.png" data-image-slot="s24-half-figure-contain" alt="Cumulative return paths across baseline and gamma sweep">
          </div>
          <div class="t-meta" style="font-size:.95vh;color:var(--text-secondary)">FIG 5.8 · BASELINE PATHS AND GAMMA MULTI-SEED ENVELOPE</div>
        </div>
        <div class="span-5" style="display:flex;flex-direction:column;gap:1.3vh">
          <div style="border-top:1px solid var(--accent);padding-top:1.2vh"><div class="t-meta" style="color:var(--accent)">gamma07</div><div class="body-sm">All three seed paths move upward and the envelope remains comparatively narrow.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">MSE</div><div class="body-sm">The same protocol sends the default regression loss downward in cumulative return.</div></div>
          <div style="border-top:1px solid var(--border-subtle);padding-top:1.2vh"><div class="t-meta">gamma10</div><div class="body-sm">The best seed is high, but the envelope is much wider, matching its CV penalty.</div></div>
          <div style="border-top:1px solid var(--ink);padding-top:1.2vh"><div class="t-meta">TAKEAWAY</div><div class="body-sm">Within the same protocol, only gamma07 delivers consistent positive returns across all seeds and survives the normalisation probe.</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S13" data-animate="three-forces">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">ANSWERING THE THREE QUESTIONS</div>
      <div class="r">17 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr;gap:3vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div class="t-meta">SYNTHESIS</div>
        <h2 class="h-xl" style="font-size:min(4.5vw,8vh);font-weight:200;line-height:1;color:var(--ink);margin:0">The answers are conditional,<br/>but actionable.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.6vw;align-items:stretch">
        <div class="span-4 card-fill" style="padding:2.4vh 1.5vw;display:flex;flex-direction:column;gap:1.2vh">
          <div style="font-family:var(--sans);font-weight:200;font-size:min(4vw,7vh);line-height:1;color:var(--ink)">A1</div>
          <div class="t-meta">LOSS CHOICE MATTERS</div>
          <div class="body-sm">R<sup>2</sup> and Sharpe decouple under directional objectives. Portfolio evaluation cannot be inferred from calibration alone.</div>
          <div class="t-meta" style="margin-top:auto;color:var(--text-secondary)">PHASE 1</div>
        </div>
        <div class="span-4 card-accent" style="background:var(--accent);color:#fff;padding:2.4vh 1.5vw;display:flex;flex-direction:column;gap:1.2vh">
          <div style="font-family:var(--sans);font-weight:200;font-size:min(4vw,7vh);line-height:1;color:#fff">A2</div>
          <div class="t-meta" style="color:rgba(255,255,255,.78)">gamma07 IS PRIMARY</div>
          <div class="body-sm" style="color:rgba(255,255,255,.86)">Mean Sharpe 0.9156, CV 0.1808, cumulative return +27.99% across three seeds.</div>
          <div class="t-meta" style="margin-top:auto;color:rgba(255,255,255,.7)">PHASE 3a</div>
        </div>
        <div class="span-4 card-fill" style="padding:2.4vh 1.5vw;display:flex;flex-direction:column;gap:1.2vh">
          <div style="font-family:var(--sans);font-weight:200;font-size:min(4vw,7vh);line-height:1;color:var(--ink)">A3</div>
          <div class="t-meta">PROBE SURVIVES ONLY ONE</div>
          <div class="body-sm">gamma07 is approximately stable under component normalisation; gamma10 and alpha06 degrade materially.</div>
          <div class="t-meta" style="margin-top:auto;color:var(--text-secondary)">PHASE 4</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S26" data-animate="four-cards">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">RECOMMENDATION · LIMITATIONS · FUTURE WORK</div>
      <div class="r">18 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto 1fr auto;gap:2.2vh">
      <div data-anim="line" style="display:flex;flex-direction:column;gap:1vh">
        <div style="height:2px;background:var(--accent);width:12vw"></div>
        <div class="t-meta">THREE-TIER RECOMMENDATION</div>
        <h2 class="h-xl" style="font-size:min(4.2vw,7.5vh);font-weight:200;line-height:1;color:var(--ink);margin:0">Use gamma07 first.<br/>Keep the caveats visible.</h2>
      </div>
      <div data-anim="up" class="grid-12" style="gap:1.4vw;align-items:stretch">
        <div class="span-4" style="background:var(--accent);color:#fff;padding:2.4vh 1.6vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta" style="color:rgba(255,255,255,.78)">PRIMARY · BEST SUPPORTED</div>
          <div style="font-family:var(--mono);font-size:2.1vh;font-weight:500;color:#fff">m2_robust_gamma07</div>
          <div style="font-family:var(--mono);font-size:1.45vh;margin-top:auto;border-top:1px solid rgba(255,255,255,.3);padding-top:1vh">SHARPE 0.9156<br/>CV 0.1808<br/>CUM +27.99%</div>
          <div class="body-sm" style="color:rgba(255,255,255,.84)">Best joint Sharpe-stability profile and approximately stable under the diagnostic probe.</div>
        </div>
        <div class="span-4" style="background:var(--ink);color:#fff;padding:2.4vh 1.6vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta" style="color:rgba(255,255,255,.78)">HIGH-RETURN · SEED-SENSITIVE</div>
          <div style="font-family:var(--mono);font-size:2.1vh;font-weight:500;color:#fff">m2_robust_gamma10</div>
          <div style="font-family:var(--mono);font-size:1.45vh;margin-top:auto;border-top:1px solid rgba(255,255,255,.3);padding-top:1vh">SHARPE 1.0043<br/>CV 0.5613<br/>NORMALISED 0.4072</div>
          <div class="body-sm" style="color:rgba(255,255,255,.84)">Highest mean Sharpe but much wider seed dispersion.</div>
        </div>
        <div class="span-4" style="background:var(--paper);color:var(--ink);border:1px solid var(--ink);padding:2.4vh 1.6vw;display:flex;flex-direction:column;gap:1.2vh">
          <div class="t-meta">STABLE FALLBACK</div>
          <div style="font-family:var(--mono);font-size:2.1vh;font-weight:500">imadl_m2_alpha06</div>
          <div style="font-family:var(--mono);font-size:1.45vh;margin-top:auto;border-top:1px solid var(--ink);padding-top:1vh">SHARPE 0.6895<br/>CV 0.2443<br/>CUM +30.42%</div>
          <div class="body-sm">Independent corroboration from another family; not stable under normalisation.</div>
        </div>
      </div>
      <div class="t-meta" style="color:var(--text-secondary);border-top:1px solid var(--border-subtle);padding-top:1vh">Limitations: single static 24-month window · 3 seeds · one feature set · no transaction costs · per-component logger not implemented. Future work: 10+ seeds, rolling windows, component logging.</div>
    </div>
  </div>
</section>

<section class="slide light" data-layout="S27" data-animate="grid-reveal">
  <div class="canvas-card">
    <div class="chrome-min">
      <div class="l">REFERENCES</div>
      <div class="r">19 / 19</div>
    </div>
    <div style="flex:1;padding:0;display:grid;grid-template-rows:auto auto 1fr;gap:2.4vh">
      <div data-anim="kicker" class="t-meta">CITED IN THIS PRESENTATION</div>
      <h2 data-anim="title" class="h-xl" style="font-size:min(4.6vw,8.5vh);font-weight:200;line-height:1;color:var(--ink);margin:0">References</h2>
      <div data-anim="bib" class="bib-list" style="overflow-y:auto;max-height:60vh;padding-right:1vw">
        <div class="bib-line"><div class="bib-num">[1]</div><div class="bib-body">Gu, S., Kelly, B., &amp; Xiu, D. (2020). <em>Empirical asset pricing via machine learning</em>. Review of Financial Studies, 33(5), 2223-2273.</div></div>
        <div class="bib-line"><div class="bib-num">[2]</div><div class="bib-body">Daniel, K., &amp; Moskowitz, T. J. (2016). <em>Momentum crashes</em>. Journal of Financial Economics, 122(2), 221-247.</div></div>
        <div class="bib-line"><div class="bib-num">[3]</div><div class="bib-body">Medhat, M., &amp; Schmeling, M. (2021). <em>Short-term momentum</em>. Review of Financial Studies, 35(3), 1480-1526.</div></div>
        <div class="bib-line"><div class="bib-num">[4]</div><div class="bib-body">Huber, P. J. (1964). <em>Robust estimation of a location parameter</em>. Annals of Mathematical Statistics, 35(1), 73-101.</div></div>
        <div class="bib-line"><div class="bib-num">[5]</div><div class="bib-body">Michańków, J., Ślepaczuk, R., &amp; Bielak, P. (2024). <em>Mean Absolute Directional Loss as a new loss function for ML-based trading strategies</em>. Working paper.</div></div>
        <div class="bib-line"><div class="bib-num">[6]</div><div class="bib-body">Bailey, D. H., &amp; López de Prado, M. (2014). <em>The deflated Sharpe ratio</em>. Journal of Portfolio Management, 40(5), 94-107.</div></div>
        <div class="bib-line"><div class="bib-num">[7]</div><div class="bib-body">CRSP, University of Chicago Booth School of Business. <em>CRSP Monthly Stock File</em>. Data snapshot used in this project.</div></div>
      </div>
    </div>
  </div>
</section>
`;

let html = readFileSync(templatePath, 'utf8');
html = html
  .replace('<html lang="zh-CN">', '<html lang="en">')
  .replace(/<title>[\s\S]*?<\/title>/, '<title>Multiplicative Directional-Robust Loss · FYP Oral Presentation</title>')
  .replace(/if\(hint\) hint\.textContent = `[^`]+`;/, "if(hint) hint.textContent = `← → Slides · B ${window.__lowPowerMode ? 'Motion' : 'Static'} · ESC Index`;")
  .replace(/<div id="hint">[\s\S]*?<\/div>/, '<div id="hint">← → Slides · B Static · ESC Index</div>');

html = html.replace(/<div id="deck">[\s\S]*?<\/div>\s*<div id="nav"><\/div>/, `<div id="deck">\n${slides}\n</div>\n\n<div id="nav"></div>`);

const requiredPlaceholder = '\u005b\u5fc5\u586b\u005d';
if (html.includes(requiredPlaceholder)) {
  throw new Error('Generated deck still contains a required placeholder.');
}

writeFileSync(join(__dirname, 'index.html'), html);
console.log(`Built ${join(__dirname, 'index.html')}`);
