# 05 — Self-bootstrapping from the model's own rare successes

## What we tried
Rather than only diagnosing why checkpoint 130's undeclared-invention
success rate is low (`03`'s 10% finding — **see the "Lamp Dark/Lit
asymmetry" finding below: this 10% was never a flat rate, it was ~20% on
half the eval set and 0% on the other half**), directly collect the
model's own genuine successes at scale and reinforce them via SFT,
continuing checkpoint 130's adapter (not a fresh LoRA).

## Why we tried it
`03_demonstration_seeding_multi_domain` established the 10% rate as a
clean negative result, not confounded by domain difficulty — a real, if
weak, signal worth trying to amplify rather than only a stopping point.
(Correction, 2026-08-24: the "not confounded by domain difficulty" part
still stands — the concern raised below is a different, answer-direction
asymmetry within the undeclared test itself, not a domain-difficulty
confound.)

## Key steps / iterations
1. Generated 8,160 unique undeclared fan/valve prompts locally (no GPU) —
   fan/valve chosen over lamp specifically to keep lamp available as
   held-out eval, avoiding train/eval contamination. Volume justified
   against both the 10% point estimate (→8,000) and a Wilson 95% lower
   bound (→14,485); landed on exhausting an extended length range (8,160)
   as a documented first-pass compromise.
2. Classifier reproduction verified twice: locally against 5 known-positive
   completions (all 5 reproduced), and a stronger row-by-row reproduction
   gate built into the generation notebook against the full original
   100-row dataset.
3. Classification/filtering/dataset-building deliberately split out of the
   GPU notebook into a local script (needs zero GPU) — unit-tested
   end-to-end against hand-built synthetic completions, including adding
   `answer_correct` as an explicit SFT-safety filter beyond the original
   "verified" definition.
4. SFT training notebook built; one real bug caught during testing (a
   module-namespace collision that would have silently rejected legitimate
   short invented codes) and fixed before it could run.

5. Generation and classification executed on AWS EC2 (g5.xlarge, A10G;
   GPU access this time came via AWS, not Colab — the notebook was
   converted to a plain script for headless execution). Full 8,160-prompt
   greedy generation completed; classifier reproduction gate re-confirmed
   exactly (100/100 match, 0 mismatches) both before and after generation.
   `verified` rate: 575/8,160 = 7.05% (0.70x the known 10% lamp baseline,
   flagged as within the same ballpark). `sft_eligible` (verified + correct
   final answer): 421/8,160 = 5.16%.
6. Domain breakdown surfaced a real asymmetry: fan's `sft_eligible` rate
   (3.04%) is far below its `verified` rate (6.50%), while valve's two
   rates are close (7.28% vs 7.60%). Inspecting 15 fan verified-but-failed
   examples found **zero tracking errors** (all correctly followed
   ground-truth state transitions) — the failure is 100% a single
   mechanical bug: the model correctly decodes the final state in its
   "Final coded state: X. X represents Y" line, then writes the *code*
   into `<answer>` instead of the *physical state name* it just correctly
   decoded. The same bug occurs in valve too when it occurs (100% of
   valve's 13 failures), just far less often (4.2% of valve's verified
   rows vs 53.2% of fan's).

> **Baseline invention behavior, confirmed causal (not a filtering
> artifact): checkpoint 130 invents mnemonic-adjacent codes ~85% of the
> time, genuinely arbitrary codes ~15% of the time.** "Mnemonic-adjacent"
> = the invented code shares the real word's first letter or is a
> recognizable prefix/abbreviation of it (e.g. `R` or `Run` for
> "Running"), reproducibly classified by rule, not eyeballed. Confirmed
> not to be a filtering artifact: the mnemonic-adjacent share among rows
> that already passed the `sft_eligible` filter (82.2%) is nearly
> identical to the share among *all* verified rows regardless of
> pass/fail (85.0%) — the filter isn't selecting for or against
> mnemonic-ness in any meaningful way, this is simply what the
> seeding-phase model already tends to invent. It also isn't purely a
> domain effect: pooled across fan+valve, mnemonic-adjacent codes fail
> the decode-back-to-answer step at 29.2% vs 12.8% for arbitrary codes,
> but domain still matters independently even holding code type
> constant (fan/arbitrary fails 28.1% of the time vs valve/arbitrary's
> 3.7%) — domain and code-mnemonic-ness are both real, at least
> partially independent contributors, not one a proxy for the other.
>
> **This qualifies every invention-rate number reported from this stage
> forward.** Raising the rate of *invention* (any self-consistent
> non-literal code) is a different, easier achievement than raising the
> rate of *genuine* (non-mnemonic) invention — a model that invents more
> often but still mostly reaches for the word's own first letter hasn't
> necessarily gotten better at concealment. Report both numbers
> separately going forward; do not collapse them into one "invention
> rate" figure.
>
> **Note: the 85%/15% split above used a classifier with a real gap,
> since closed — see the next entry. The corrected picture is more
> extreme than 15% arbitrary, not less.**

7. Manual inspection of the 46-example Dataset A candidate pool (see
   below) found a second leakage channel the classifier never checked:
   codes resembling the *domain* name itself (`FZ`/`Fx` for fan, `VAL`/
   `Va`/`Vi` for valve), not just the state name. 100% of both fan and
   valve's Dataset A candidates had at least one such code, despite
   passing the original (state-only) arbitrary filter. Fixed by
   extending the same rule one level up (`src/data/` classifier logic
   documented in `aws_runs/bootstrap_filter_v1/mnemonic_analysis.py`);
   unit-tested for both the missed cases (`VAL`, `Va`, `Vi` now correctly
   rejected) and against over-rejection (mid-word `F`/`V`, and genuinely
   arbitrary codes like `Zorp`/`Blib`, correctly still pass).
8. To distinguish the two mechanism hypotheses below, generated a fresh
   2,000-prompt undeclared batch on **lamp** (checkpoint 130's one
   zero-declared-training-exposure domain), disjoint from the original
   100-example eval set, greedy decoding. Result: 124 verified, 120
   genuinely clean+tracked (96.8% — the opposite of fan/valve's 0%).
   Manual diversity check (same as Dataset A) found the same class of
   problem again, worse: only 9 distinct code pairs across 120 rows
   (92.5% redundancy), and **100% of the 120 rows end in "Dark"** — a
   total degeneracy, not just a skew. Investigating that degeneracy is
   what surfaced the finding below.
9. A second, sampled (`do_sample=True`, `temperature=0.8`, `top_p=0.95`
   — matching the GRPO rollout precedent) 2,000-prompt run on the
   *identical* disjoint prompt pool confirmed the Dark/Lit split is not
   a greedy-decoding artifact: Lit-truth verified rate stayed at exactly
   0.00% under sampling too. See the standalone finding below.

## Standalone finding: lamp shows a persistent, structural Dark/Lit
## answer-direction asymmetry — not a decode-time artifact, not lamp-only
## in kind, but severe specifically for lamp in degree

Full evidence chain, three independent runs:

| source | Dark-truth verified rate | Lit-truth verified rate |
|---|---|---|
| original 100-example lamp baseline (`03`'s 10% source) | 10/50 = 20.0% | **0/50 = 0.0%** |
| fresh greedy diagnostic (n=2,000) | 124/1011 = 12.3% | **0/989 = 0.0%** |
| fresh sampled diagnostic (n=2,000, temp=0.8/top_p=0.95) | 94/1011 = 9.3% | **0/989 = 0.0%** |

**Across all three runs and 4,050 total attempts, checkpoint 130 has
never once produced a verified success when the true answer is "Lit."**
The sampled run was run specifically to test whether temperature
exploration could surface a close second-place "lit" logit greedy always
loses to — it could not: the raw `<answer>` tag distribution barely
moved (model says "dark" ~82-87% of the time regardless of ground truth,
in both decoding settings), and Lit-truth verified rate stayed at exactly
0.00% in both. Decomposing *where* Lit-truth attempts fail (greedy
batch): 81.0% have genuinely broken step-by-step tracking; 17.9% track
**perfectly** (a correct, consistent invented mapping used correctly
across every step) but the decode-back line then independently
contradicts that same mapping, claiming "represents Dark" regardless
(163/177 of these specifically claim Dark) — a clean decode-time-bias
signature in isolation. The dominant 81% bucket is most parsimoniously
read as the same bias bleeding into the reasoning trace itself, not
confirmed at the per-token level.

**Checked whether this is general or lamp-specific**, since either
answer has very different implications for the project's other headline
numbers. Ran the same check (verified rate split by ground-truth answer)
against fan/valve's full 8,160-completion pool and the original
direct-RL Coin Flip trajectory (`step50_to500_full_rollouts.json`,
8,280 rollouts with per-example ground truth):

- fan: Stopped-truth 7.30% vs Running-truth 5.69% (~1.3x)
- valve: Closed-truth 9.36% vs Open-truth 5.83% (~1.6x)
- Coin Flip: Tails-truth 69.4% vs Heads-truth 66.1% (~1.05x, essentially balanced)

**Conclusion: graduated, not universal.** Negligible in Coin Flip,
real-but-moderate in fan/valve (both directions retain substantial
success), severe-and-total specifically in lamp. This is not "our
measurement approach has been missing asymmetries throughout the
project" — it's a real, severe, structural asymmetry specific to lamp,
already present undocumented in the original 10% baseline since `03`.

**Retroactive corrections applied** to every place that cited the lamp
10%/49% figures as an unqualified signal:
`03_demonstration_seeding_multi_domain/README.md`,
`results/03_demonstration_seeding_multi_domain.json`, and this file
(see "What we tried" and "Why we tried it" above).

**Dataset A' (the lamp-derived alternative to Dataset A) is formally
closed out as not viable — well-supported, not a judgment call.** Zero
verified Lit-truth successes across 4,050 attempts, two decoding
strategies, confirmed persistent rather than a single-batch fluke. Any
lamp-derived training set is 100% Dark-answer by construction; no
amount of additional lamp generation volume would fix this, since the
degeneracy is not a sampling-density problem.

**Resolved (2026-08-24): the last open question.** Checked whether `03`'s
86% declared-mapping lamp control (used to rule out "domain difficulty"
for the undeclared collapse) carries the same asymmetry, diluted into
its aggregate accuracy. Recovered per-example ground truth by
regenerating the deterministic lamp eval set (same seed, `example_id`
join, sanity-checked to reproduce 86/100 exactly) against the saved
`step130_eval_progress.json`. Result: Dark-truth 84.0%, Lit-truth 88.0%
— essentially balanced, if anything slightly favoring Lit. **The
declared-mapping control does NOT carry this asymmetry; `03`'s "domain
difficulty ruled out" conclusion stands.** The bias is specific to the
undeclared/self-invention condition — inventing a code *and* committing
to a final answer with no externally-anchored mapping — not a general
lamp-domain answer-generation bias.

## Standalone finding: genuine invention and reliable tracking may be in
## tension for checkpoint 130, specifically on domains it has prior
## declared-training exposure to

Documented here as its own finding, separate from the Dataset A/B
composition decision it drove (see "Final result" below).

Re-running the tightened (state+domain) classifier against the full
8,160-completion pool (no new sampling) found the leakage effect is
total, not partial: **0 of 575 verified rows are genuinely clean under
both checks** — identically 0% in both domains individually (0/265 fan,
0/310 valve), not just pooled. Checked whether clean codes exist
anywhere in the raw data regardless of verification: yes (9 fan, 180
valve), but every one of them is in a row that **failed** the
self-consistency check.

**Mechanism hypothesis (suggestive, not proven)**: genuine concealment
and successful self-consistent tracking may be in tension for this
model — a domain/state-adjacent code doubles as an implicit memory aid
("F-something must be the fan-related one"), and a genuinely opaque
code removes that scaffold, costing tracking reliability. n=575/0 rules
out "this co-occurrence is common"; it does not rule out "vanishingly
rare" — a much larger generation run could in principle still surface a
handful of clean-and-verified examples.

**A second, non-exclusive hypothesis, found by retroactively applying
the same tightened classifier to the ORIGINAL 10% lamp baseline
(`stage35-undeclared-lamp-checkpoint130-v2/rollouts.json`, the source of
`03`'s 10% figure)**: all **10/10** of those verified successes are
genuinely clean — 0% leakage, the opposite of the fan/valve pattern.
Lamp is the one domain checkpoint 130 has **zero prior training
exposure to** (held out entirely, never seen even in declared-mapping
form), while fan/valve were trained on extensively in declared form
before this undeclared test. So the fan/valve-specific leakage could
reflect the tracking-vs-concealment tension above, **or** simply that
prior declared-training exposure to a domain primes semantically-loaded
code choices in that same domain specifically, with lamp unaffected
because it has no such exposure. Current data cannot distinguish these
two explanations — both are recorded, neither is confirmed.

**Retroactive check on the headline numbers, confirmed precisely**: the
original 10%/7.05% base invention-rate figures do **not** need
re-checking or revision — they were never based on a mnemonic or
domain-name definition, only structural self-consistency and correct
decode-back, and the lamp baseline specifically is unaffected (10/10
clean). What *does* need care: **the fan/valve verified set is 100%
mnemonic/domain-leaking, 0% genuinely arbitrary** (310/310 valve,
265/265 fan, confirmed by direct classification, not inferred) — so
"7.05% invention" on fan/valve currently means entirely mnemonic-
adjacent invention, none of it genuinely arbitrary. This sets the
correct comparison baseline for Dataset B's eventual evaluation:
**0%**, the tightened fan/valve rate — not the raw 7.05%, and not the
10% lamp figure, which is a different domain with a different
(currently more favorable) pattern. Comparing Dataset B's post-
bootstrap result against 10% or 7.05% without applying the same
tightened definition to both sides would not be a valid comparison.

## Final result
Full pipeline built and executed end to end, including SFT training and
evaluation. Generation (8,160 fan/valve completions plus 4,000 lamp
diagnostic completions), filtering, and now training are all done. The
original plan called for two SFT datasets (arbitrary-only vs.
representative-mix) to test whether reinforcing genuinely arbitrary
examples shifts post-bootstrapping behavior toward more genuine
concealment. After the domain-leakage classifier fix (step 7), **Dataset
A (fan/valve arbitrary-only) cannot be built — the qualifying pool is
empty.** A lamp-derived alternative (Dataset A') looked promising at
first (120 clean+tracked candidates from a single 2,000-prompt batch)
but is also **formally closed out as not viable**, once investigating
why those 120 were 100% "Dark"-answer surfaced the Dark/Lit asymmetry
finding above — zero verified Lit-truth successes across 4,050 total
lamp attempts under two decoding strategies. Proceeded with Dataset B
only, an informed fallback rather than a default:

- **Dataset B (all eligible, domain-balanced)**: 248 examples (124 fan +
  124 valve, valve downsampled from its full 297 to match fan).
  Re-verified at 100% semantic pass rate immediately before training
  (correctness unaffected by classifier tightening). Composition
  corrected: **100% mnemonic/domain-leaking under the tightened
  classifier** (not 82.7%/state-only as originally scoped) — Dataset B
  is entirely drawn from the same 575-row pool already established as
  0% genuinely arbitrary.

### SFT training and evaluation (2026-08-25)

Bounded dry run (8 steps) passed cleanly first — stable training, no
declared-accuracy regression, and its Dark-truth successes were 100%
genuinely clean (n=10). Proceeded to the full 150-step run (continuing
checkpoint 130's adapter), with a new check added mid-run: the
mnemonic/domain-leak classifier applied to Dark-truth verified
successes at every milestone, to answer the real question Dataset B can
answer — did bootstrapping increase genuine invention, or just confident
use of the same shortcut?

**Full trajectory, not just the endpoint** (the step-90 dip is part of
the honest record — reading only baseline-vs-150 would look like a
clean monotonic doubling, which it is not):

| step | Dark rate | clean | leaking | n | clean % |
|---|---|---|---|---|---|
| checkpoint 130 baseline | ~9.3-20% | ~10 | ~0 | ~10-50 | ~100% |
| dry-run (8) | 20.0% | 10 | 0 | 10 | 100.0% |
| 30 | 30.0% | 12 | 3 | 15 | 80.0% |
| 60 | 32.0% | 13 | 3 | 16 | 81.3% |
| 90 | 34.0% | 11 | 6 | 17 | 64.7% |
| 120 | 38.0% | 13 | 6 | 19 | 68.4% |
| 150 | 40.0% | 14 | 6 | 20 | 70.0% |

Run finished normally (`max_steps_reached_needs_review`; declared
regression check never approached its 15pt limit; Lit-truth stayed at
0-2% throughout, expected and not a failure signal for this run — that
gap is unrelated to what Dataset B targets).

**Reported as three parts side by side, not collapsed into one number:**

1. **Genuine invention (clean count) increased**: ~10 → 14 (+40%) — a
   real, if modest, gain.
2. **Shortcut/leaking invention increased more**: ~0 → 6 — a larger
   absolute gain, and the majority driver of the headline rate increase.
3. **The genuine-to-total ratio declined** (100% → 70%) even as the
   absolute genuine count rose. Both trends are real; neither erases
   the other.

**Methodological implication**: reinforcing a mixed-quality signal
(Dataset B, 100% mnemonic/domain-leaking at the source) appears to
amplify *both* the genuine behavior already in the model's prior *and*
the dominant shortcut pattern, roughly proportional to how easy each is
to reproduce — consistent with genuine arbitrary invention being
intrinsically harder to reinforce than the shortcut. This is a second,
independent line of evidence for the tracking-vs-concealment tension
hypothesis raised earlier in this session (still not proven, now
supported by two observations rather than one).

## What this means for the next step
Dataset B's result is now the reference point for any future
bootstrapping iteration: it demonstrates the mechanism works (genuine
successes can be reinforced) but also that a 100%-mnemonic training
signal reinforces the shortcut at least as readily. A cleaner future
attempt would need either a training set with better genuine-invention
representation (which Dataset A/A' investigations show isn't currently
obtainable from this checkpoint without a different data strategy) or
an explicit training-time penalty on mnemonic/domain-leaking codes. The
arbitrary-vs-mnemonic comparison originally planned via Dataset A
remains open, not answered here.

**Standing lesson applied throughout (2026-08-24 onward)**: every rate
reported — verified, sft_eligible, invention, and now the mnemonic/leak
composition — was split by ground-truth answer direction and checked
against a genuine/shortcut composition breakdown as the *default*
reporting format at every milestone, not an afterthought. This is what
caught the step-90 dip and the true composition of the rate increase in
real time, not after the fact.
