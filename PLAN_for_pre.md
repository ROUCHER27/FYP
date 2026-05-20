# English-Only FYP Oral Presentation PPT Plan

## Summary
- Create the deck in `ppt_package/fyp_oral_presentation/`.
- Main file: `ppt_package/fyp_oral_presentation/index.html`.
- Image folder: `ppt_package/fyp_oral_presentation/images/`.
- All visible slide text will be English: titles, subtitles, bullets, captions, figure callouts, citations, references, navigation labels, and generated infographics.
- Reuse the Swiss Academic demo cover structure, but replace all visible wording and rebuild the rest of the deck from the corrected plan.

## Key Changes
- Use `template-swiss-academic.html` with Swiss Academic layouts.
- Build 17 main slides plus 1 References slide:
  1. Cover
  2. Background & Research Questions
  3. Literature Snapshot
  4. What We Did
  5. Four Contributions
  6. Research Design & Architecture
  7. Loss Function Families
  8. Portfolio Construction & Evaluation
  9. Experimental Phases
  10. Data & Features
  11. Phase 1 Baseline Comparison
  12. Phase 2 Hybrid A/M Sweep
  13. Phase 3a Gamma Refinement
  14. Phase 3b Integrated Sweeps
  15. Normalisation Probe
  16. Answering the Three Questions
  17. Recommendation, Limitations & Future Work
  18. References
- Apply corrected evidence wording:
  - Phase 1/2 and Phase 3/4 architecture caveat included.
  - Data slide avoids unsupported survivorship claims.
  - Phase 2 framed as seed-42 motivation, not robustness proof.
  - M2 notation caveat handled carefully.
- Generate only conceptual English infographics with Image 2; keep thesis result figures unchanged.

## Execution Plan
- SubAgents will be used during execution:
  - Content audit agent: verifies every English claim and number against LaTeX/evidence.
  - Layout agent: builds the Swiss Academic HTML deck.
  - Asset agent: prepares images and English Image 2 prompts.
  - QA agent: checks English-only visible text, references, validation, and visual spacing.
- After implementation, run validator, grep for placeholders/old wording/non-English visible text, and inspect the deck in browser.

## Assumptions
- Figure images from the thesis are acceptable as-is if their embedded labels are already English.
- References remain in English APA/IEEE-style compact format.
- The deck title remains: “Multiplicative Directional-Robust Loss”.
