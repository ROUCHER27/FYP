# Review: Appendix B Per-Seed Raw Results and Reproducibility

Source reviewed: `paper/appendix_B_per_seed_raw.md`.

## Summary

Appendix B is useful for auditability and fits the evidence-gate contract. It mainly needs small reproducibility clarifications and one consistency check around seed identifiers.

## Content and evidence issues

1. `appendix_B_per_seed_raw.md:7` states γ refinement seeds are `{42,52,62}`. Good. Ensure Chapter 3 and Chapter 5 do not imply one global seed set across every multi-seed phase.

2. `appendix_B_per_seed_raw.md:33` says the normalisation probe reused the same three seeds but the mapping is not preserved. Keep this caveat, and make sure Chapter 5 does not present the normalised per-seed values as directly paired with γ refinement seed IDs.

3. `appendix_B_per_seed_raw.md:47` says every row reproduces on `main` at commit `6c0fbde`, but Tables 5.3 and 5.4 depend on Phase 2 branch artifacts. Rephrase: baseline and Phase 1.5 reproduce on `main @ 6c0fbde`; Phase 2 summaries are reproduced/read from the listed `phase2.2-fix` artifacts.

4. `appendix_B_per_seed_raw.md:89-94` says the γ refinement was run from `phase2.2-fix` but the grouped summary is "locally at" `doc/phase2-fix/phase2_2/...` on main. Clarify whether these files have been copied into main, or whether they are branch artifacts.

5. `appendix_B_per_seed_raw.md:137` says grouped-summary values are stable to the fourth decimal across supervisor re-runs. As in Chapter 3, keep only if a rerun artifact exists; otherwise soften or remove.

## SciWrite issues

1. The appendix is clear. Consider naming "seed 1/2/3" in Table B.2 as "normalisation-run slot 1/2/3" to avoid implying unknown seed values.

2. The reproduction command section is practical, but the final report might not want `<output-root>` placeholders. A later technical appendix can keep them; the final PDF may use a shorter reproducibility note.

## Top priority revisions

1. Split main-branch reproduction from phase-branch artifact reading.
2. Preserve the normalisation seed-ID caveat.
3. Remove unsupported "fourth decimal rerun" claim unless evidence is added.
