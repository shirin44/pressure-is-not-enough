# 03 — Demonstration seeding: multi-domain (fan + valve → lamp held out)

## What we tried
Retrained from scratch across two training domains simultaneously (fan,
valve; declared mapping), holding a third domain (lamp) out entirely for
zero-shot generalization testing.

## Why we tried it
Single-domain seeding (`02`) taught domain-specific surface form, not the
abstract strategy — hypothesis: ≥2 training domains are needed to force
the abstraction to generalize.

## Key steps / iterations
| Step | Pooled accuracy | Notable finding |
|---|---|---|
| 20 | 46.3% | — |
| 40 | 70.7% | — |
| 60 | 82.3% | — |
| 80 | 88.3% | fan lagging (78% tracking vs. valve 95%); trajectory rule flagged fan for a wording fix |
| 100 | 88.7% | fan wording fix applied: tracking 82% (+4pts, **below** the +8pt bar that would confirm wording was the bottleneck); lamp drifted down 4-5pts |
| 130 | 89.3% | continued from 100 **without a recorded review in between** — a process gap, noted not repeated; fan tracking 85%, both fan/valve (13pt) and fan/lamp (4pt) gaps now within the 15pt threshold, full 95% gate still not met |

Then the real test: **Stage 3.5 undeclared** (no mapping supplied,
self-consistency scoring, checkpoint 130, held-out lamp): verified
nonliteral encoding rate **10%**, final-answer accuracy **49%**, failed.
The matched *declared*-mapping lamp baseline at the same checkpoint scored
86% (close to pooled 89.3% and fan 85%) — ruling out domain difficulty as
a confound. Clean negative result: invention failure, not a harder domain.

> **Correction (2026-08-24): the 10% figure above is not an unqualified
> signal — see `05_self_bootstrapping/README.md`'s "Lamp Dark/Lit
> asymmetry" finding.** All 10 of those verified successes are
> "Dark"-truth examples; the matched 50 "Lit"-truth prompts in the same
> 100-example set scored **0%**, not 10%. This asymmetry was confirmed
> structural (persists under both greedy and temperature-sampled
> decoding, 4,050 total attempts) — not noticed until a much later
> stage.
>
> **Resolved (2026-08-24): the 86% declared-mapping control does NOT
> carry the same asymmetry.** Recovered per-example ground truth by
> regenerating the deterministic lamp eval set (same seed, `example_id`
> join, sanity-checked to reproduce 86/100 exactly) against the saved
> `step130_eval_progress.json` per-example results. Dark-truth 84.0%,
> Lit-truth 88.0% — essentially balanced, if anything slightly favoring
> Lit. The "domain difficulty ruled out" conclusion below stands as
> originally stated. **The bias is specific to the undeclared/
> self-invention condition, not a general lamp-domain answer-generation
> bias** — when the mapping is supplied, direction doesn't matter; it
> only breaks down when the model must invent its own code and commit
> to a final answer without an externally-anchored mapping.

## Final result
Checkpoint 130 reliably **executes** a supplied code (86-89% across
domains) but does not reliably **invent** its own under undeclared
conditions (10% verified rate, n=100 — see the correction above: this
10% was ~20% on half the eval set and 0% on the other half, not a flat
10% across the task).

## What this means for the next step
Two threads opened from here: (1) checkpoint 500 — the separate,
pre-seeding Coin Flip RL checkpoint that any eventual Stage 4 merge would
need to build on — turned out to be unstable to resume at all, diagnosed
in `04_stability_investigation`; (2) rather than only diagnosing why the
10% rate is low, directly amplify the model's own rare successes via
SFT — `05_self_bootstrapping`.

---
**Checkpoint-130 naming collision**: this experiment's final checkpoint is
`multidomain-stage3-v1/trainer-output/checkpoint-130` on Drive.
`02_demonstration_seeding_single_domain` *also* has a checkpoint numbered
130 (`latch-seed-stage3-decode-back-sft-v3/trainer-output/checkpoint-130`)
— same number, **completely different, unrelated training lineage**. Do
not confuse the two.

---
**Note on `stage35_third_domain_zero_shot.ipynb`'s output cells**: this
notebook's real execution results were accidentally overwritten by a
rebuild-script verification pass during the 2026-08-22 repository
reorganization, and have been conservatively reconstructed from the
conversation record (see the notice at the top of that notebook). The
headline numbers above are cross-checked against this file and
`multidomain_stage3_plan.json` in this same folder, both of which were
never affected by the incident.
