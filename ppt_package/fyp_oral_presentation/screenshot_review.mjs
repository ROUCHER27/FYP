/* Screenshot 19 deck slides at 1920×1080.
   Uses playwright shipped at /Users/roucher/agent-my/gstack/node_modules/playwright. */
import { chromium } from '/Users/roucher/agent-my/gstack/node_modules/playwright/index.mjs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outDir = path.join(__dirname, 'screenshots');
mkdirSync(outDir, { recursive: true });

const deckUrl = pathToFileURL(path.join(__dirname, 'index.html')).href;
const TOTAL = 19;
const VIEWPORT = { width: 1920, height: 1080 };
// Disable WebGL background and Motion to keep screenshots deterministic.
const HASH_PREFIX = '';

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: 1 });
const page = await context.newPage();

console.log(`[deck] opening ${deckUrl}`);

for (let i = 1; i <= TOTAL; i += 1) {
  const url = `${deckUrl}?slide=${i}${HASH_PREFIX}`;
  await page.goto(url, { waitUntil: 'networkidle', timeout: 45_000 });
  // Wait for KaTeX to finish auto-rendering.
  await page.waitForTimeout(1500);
  // Force-skip in-flight Motion animations by setting all data-anim opacities to 1.
  await page.evaluate(() => {
    document.querySelectorAll('[data-anim]').forEach((el) => {
      el.style.opacity = '1';
      el.style.transform = 'none';
    });
    document.querySelectorAll('[data-animate]').forEach((el) => {
      el.querySelectorAll?.('*').forEach((c) => {
        if (c.style && c.style.opacity === '0') c.style.opacity = '1';
      });
    });
  });
  await page.waitForTimeout(250);
  const file = path.join(outDir, `slide-${String(i).padStart(2, '0')}.png`);
  await page.screenshot({ path: file, fullPage: false, omitBackground: false });
  console.log(`[deck] slide ${i}/19 -> ${file}`);
}

await browser.close();
console.log(`[deck] done. screenshots saved to ${outDir}`);
