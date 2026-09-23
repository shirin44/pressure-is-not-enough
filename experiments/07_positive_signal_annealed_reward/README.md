# 07 — Positive-signaling reward (r_signal) + annealed p_CoT, direct on Coin Flip

## What we tried
A genuinely new mechanism applied directly to Coin Flip from checkpoint 500
(not another seeding domain): (1) `r_signal`, a content-agnostic, continuous
partial-credit reward -- the Adjusted Rand Index between a completion's
invented state-slot tokens and the true physical state, rewarding *any*
measurable correlation rather than requiring the all-or-nothing bar the
existing terms (`r_consistency`, `p_state_variation`) already enforced; (2)
a single continuously annealed p_CoT weight (0.2 -> 2.0 over the run)
replacing the previous 3-tier discrete banned-pattern curriculum, so
pressure ramps smoothly instead of jumping at fixed steps.

## Why we tried it
`06_stage4_coinflip_merge` found zero genuine invention merging checkpoint
500 with the Dataset-B-bootstrapped seeded capability. This tests whether
a differently-shaped reward -- one that gives partial credit for *early,
imperfect* attempts the existing terms give zero credit for -- can surface
invention that reward pressure alone (checkpoint 500's original training)
apparently could not.

## Key steps / iterations
1. Designed `r_signal` via Adjusted Rand Index (chance-corrected,
   relabeling-invariant by construction -- unit-tested directly: arbitrary
   interchangeable tokens score identically). Re-derived the full reward
   invariant with `r_signal` included: margins fall from 0.85 (r_consistency
   alone) to 0.70, still comfortably positive. 22 unit tests, all passing,
   before any model involvement.
2. 8-step dry run on checkpoint 500: survived cleanly, KL clamp and circuit
   breakers unchanged and verified active, no accuracy regression.
3. Full 150-step run: survived to completion, no hard stop, task accuracy
   stable (60.5% -> 56.4%). **0 substituted (non-literal) rollouts at every
   milestone (30/60/90/120/150)** -- r_signal's value was, at every
   milestone, 100% attributable to overt (literal) completions; there was
   never a non-literal rollout for it to actually differentiate on.
4. Follow-up: entropy_coef raised 0.05 -> 0.18 (3.6x, a deliberately large
   jump, not an increment) on an 8-step dry run, testing threshold-vs-smooth
   exploration hypotheses. Also 0/64 non-literal.
5. Cross-checked against every other checkpoint-500 diagnostic in this
   project using the same exploration classifier (`kl_clamp_mitigation_v1`,
   `_v2`, `per_token_kl_instrumentation_v1`): 632 additional accepted
   rollouts, 0 substituted, under the *original* (pre-r_signal) reward
   design. Combined running total: 1,656 rollouts directly reclassified
   with one consistent method, 0 non-literal, across two entropy_coef
   values and two reward-function designs.

## Final result
r_signal and the annealed p_CoT curriculum both work exactly as designed
(verified via unit tests and live training telemetry) but never had a
non-literal rollout to act on -- checkpoint 500 did not produce a single
non-literal token attempt across 150 steps of training or the entropy
follow-up. This is a clean negative result on the mechanism's applicability
here, not a mechanism failure: the reward-shaping approach cannot help
because nothing ever reaches the state it would reward.

This motivated Experiment 1 (does the task actually require encoded state
at the lengths this project uses -- ruling out "the regime wasn't
load-bearing" as an explanation) and Experiment 2 (does higher-temperature
*generation-only* sampling, no training, ever produce a non-literal
attempt at all). Both are documented in `08_direct_indomain_synthetic_bridge/`.

## What this means for the next step
Reward-shaping-after-the-fact cannot help when the model never attempts a
non-literal token in the first place. Experiment 3
(`08_direct_indomain_synthetic_bridge`) tests a lever that acts *before*
the attempt: a small fraction of directly-demonstrated valid encoded
trajectories, bridging the model into territory it has never
spontaneously explored, via an auxiliary supervised loss (not naive
GRPO-group injection -- see that stage's design audit for why).

Evidence: `experiments/06_stage4_coinflip_merge/` (the merge this
mechanism was tested on top of), and the AWS run logs referenced in
`logs/development_log.md`'s 2026-08-25 entries for the dry run, full run,
and entropy_coef test.

## Archived from AWS (2026-09-22)

A prior audit found this stage's own driver script and evidence had never
been committed (only `reward_v3.py`/`test_reward_v3.py`/this README/
`results/07_positive_signal_annealed_reward.json` were). All ten files were
still present on AWS instance `i-REDACTED` and are now archived
read-only under `aws_recovered/`, sha256-verified bit-identical between the
remote copy and this repo copy (10/10 matched, 0 mismatches):

| file | sha256 |
|---|---|
| `stage07_signal_annealed.py` | `46315b58...9762d7` |
| `stage07_dryrun.log` | `b920bf30...195d2` |
| `stage07_fullrun.log` | `f4e26a9b...b76032` |
| `aisi_checkpoints/grpo-stage07-signal-annealed-dryrun-v1/stage07_signal_annealed_reward.json` | `e2b4a9e3...ed59d6` |
| `aisi_checkpoints/grpo-stage07-signal-annealed-full-v1/stage07_signal_annealed_reward.json` | `0b3ca51a...5f3a7` |
| `stage08_entropy_test.py` | `ff51e64a...cd02ce` |
| `stage08_dryrun.log` | `26dd39c4...276bc9` |
| `aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/README.md` | `ee732a1b...c916` |
| `aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/per_token_instrumentation.json` | `6d276ccf...5740f2eb` |
| `aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/stage08_entropy_test.json` | `d8e64ac1...d42b85a5` |

Not fetched (not requested, and not "evidence" in the same sense as the
above): the `checkpoint-*` LoRA weight directories inside
`grpo-stage07-signal-annealed-{dryrun,full}-v1/` and
`grpo-stage08-entropy-test-dryrun-v1/checkpoint-8/` (adapter weights +
optimizer/scheduler state, 47-158MB each), and a
`per_token_instrumentation.json` under each `grpo-stage07-*` directory
(29-30MB, per-token KL instrumentation, distinct from the smaller one
already fetched from the `stage08` dryrun). All still present on AWS if
wanted later.

**Full read of `stage07_signal_annealed.py` (1,343 lines, not just hash/head
as in the prior audit) confirms:**
- `N_STEPS=8 if DRY_RUN else 150` — the full run really is 150 steps.
- Reward terms active: `r_task, p_cot (annealed 0.2->2.0), p_length,
  p_structure, p_state_variation, r_consistency, r_signal` — matches
  `reward_v3.py` exactly. The script's own inlined `score_completion` is
  byte-for-byte the same formula as the currently-committed
  `reward_v3.score_completion_v2`, differing only in name (confirmed by a
  diff of the two function bodies and by executing both on the same input
  in `test_stage07_recovered_evidence.py`) — `reward_v3.py` is this
  script's reward logic factored out, not a later rewrite.
- **Prompt format mismatch, same as Llama 09e and every other Qwen stage
  already checked**: training rollouts use a **raw string** prompt
  (`unique_pool()` -> `{'prompt': prompt, ...}` -> `Dataset.from_list(TRAIN_POOL)`,
  line ~932), never chat-templated. Milestone/held-out evaluation (the
  100-prompt `HELDOUT_POOL`, greedy, every 30 steps) explicitly calls
  `tokenizer.apply_chat_template(...)` (line ~1155). This is the same
  structural gap already found elsewhere: whatever the paper's Section 4.1/
  Appendix A.1/A.3 currently say about this condition's prompt format, if
  they say or imply RL training used the chat template, that is
  **contradicted by this code**. (The paper text itself is not in this
  repo, so this is checked against the code and against this repo's own
  README/results.json, not against paper section text directly.)
- The recovered evidence JSON's `final_report` reproduces every number this
  README already claims, computed directly from raw per-rollout data, not
  re-derived: `terminal_step=150=requested_steps`, `hard_stop=None`,
  `task_accuracy_first_window=0.6047` (60.5%), `task_accuracy_last_window=0.5642`
  (56.4%), final milestone (step 150) `verified_non_literal_rate=0.0` on
  both `Heads` (n=53) and `Tails` (n=47) — 100 held-out prompts, 0 non-literal.
  The dry-run evidence shows `terminal_step=8=requested_steps`,
  `dry_run=True`. The `stage08_entropy_test.json` config shows
  `entropy_coef=0.18, n_steps=8, dry_run=True`, matching the "entropy_coef
  raised 0.05 -> 0.18... on an 8-step dry run" claim above.
- **No discrepancy found** between this recovered code+evidence and this
  repo's own README/results.json. The only open question is the prompt-format
  point above, which is a discrepancy against the paper (unverifiable
  directly, paper text absent), not against this repo's own prior claims.

Regression tests: `test_stage07_recovered_evidence.py` (7 tests, all passing).
GPU usage: instance started read-only (no training/inference run), file
transfer and one full-script read only, stopped again afterward;
`TEARDOWN FULLY CONFIRMED`.
