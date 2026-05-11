# Review: Abstract

Source reviewed: `paper/abstract.md`.

## Summary

The abstract is strong as a standalone summary, but it currently compresses too much methodology into long paragraphs and contains two evidence-boundary problems. The main risks are an internal mismatch in the count of experimental phases and use of "robust" language for single-seed baseline evidence.

## Content and evidence issues

1. `abstract.md:5` says the model consumes a "15-dimensional cumulative-return and cumulative-turnover feature vector." This is under-specified and likely misleading: five horizons x two variables explains only 10 engineered features. Align this with the exact 15 input columns after Chapter 4 is fixed.

2. `abstract.md:7` says "Four experimental phases are run" but then lists baseline, Phase 1.5, Phase 2 γ, integrated Phase 2, and Phase 2.2-fix1 normalisation, which reads as five items. Either treat the normalisation probe as a diagnostic outside the four phases or change the count.

3. `abstract.md:7` says the phases run under "identical data, features, architecture, and portfolio settings." This is defensible within phase tables, but the integrated Phase 2 and normalisation probe come from different phase/branch contexts and should be phrased more carefully: "within each comparison, the core data/model/portfolio protocol is fixed."

4. `abstract.md:9` says traditional regression and pure directional losses "do not produce robust positive Sharpes at the same single seed." Robustness cannot be inferred at a single seed. Suggested direction: "do not produce economically meaningful positive Sharpes in the seed-42 same-window baseline."

5. `abstract.md:9` says the stable fallback has the "largest mean cumulative return of the three candidates." True numerically, but this can distract from the primary recommendation because α06 is not the Sharpe winner and degrades under normalisation. Add one qualifier that the fallback is chosen by stability/return trade-off, not by Sharpe dominance.

## SciWrite issues

1. `abstract.md:3`, `abstract.md:7`, and `abstract.md:9` are very long paragraphs with multiple claims per sentence. Split the abstract into shorter, topic-specific sentences: problem, protocol, phases, headline results, boundaries.

2. Replace "almost always optimises" with a less absolute phrase unless backed by a literature survey count. Suggested: "commonly optimises."

3. "operators knowingly accept higher seed-sensitivity" is clear but operationally loaded for an academic abstract. Consider "when higher seed sensitivity is acceptable."

## Top priority revisions

1. Correct the four/five phase wording.
2. Fix X1 feature-vector description.
3. Remove "robust" from single-seed baseline claims.
4. Shorten the result paragraph for readability.
