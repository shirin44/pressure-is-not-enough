# Stage 10 — literal-CoT adversarial pressure: design log

## What this tests, and why it's new

Every prior lexical-penalty run in this project applied the penalty to a policy that either
already used a non-literal code (Llama 09e's code-word design, by construction) or had
already drifted to vacuous/non-literal output before the penalty had real opportunity to act
on literal words (Stage 1's checkpoint 500 itself was the *product* of such a run; Stage 7
resumed checkpoint 500 for a *different* mechanism, r_signal, without re-testing whether the
lexical penalty itself binds). This experiment inverts that: start from a policy **confirmed
100% literal** (`experiments/01_direct_rl_coinflip/aws_recovered/checkpoint500_reeval.json`,
500/500 rollouts tier 1) and apply the same lexical penalty. If the penalty has ever going to
bind anywhere in this project, it has to bind here.

## Step 1 — seed policy: (a) checkpoint 500 vs. (b) a new small SFT bank

**Recommendation: (a), checkpoint 500.** Reasoning:

- **LoRA compatibility, checked not assumed.** `~/aisi_checkpoints/step-500/adapter_config.json`
  (already retrieved and read in the prior session): `r=8, lora_alpha=16, lora_dropout=0.05,
  target_modules=[q_proj,v_proj,k_proj,o_proj], base_model_name_or_path=Qwen/Qwen2.5-3B-Instruct,
  peft_version=0.19.1`. This is byte-for-byte the same `LORA_KWARGS` this experiment's own
  runner uses (and every other Qwen stage in this project) — no rank/target mismatch, no
  conversion needed, resumes cleanly with `get_peft_model` + `set_peft_model_state_dict`.
- **Accuracy floor, checked not assumed.** 64.0% final-answer accuracy (320/500) under the
  corrected chat-templated prompt, 98.8% format-valid (494/500) — well above chance (2 states)
  and far from a degenerate floor. GRPO's group-relative advantage needs real variance in
  group outcomes to produce a gradient signal at all; a checkpoint this far from 0%/100% gives
  it plenty to work with.
  `adapter_sha256=4795fe548d7130eb0c559fd3cc829afca247f53a96bca99c68c3eace0f3209a0`.
- **It's real, already-verified weights**, not a new artifact needing its own held-out check
  before it can be trusted — that verification (500 rollouts, hash-verified evidence) is
  already done and sitting in this repo.
- **Against (b):** a fresh 10-20-trajectory SFT bank is itself a new, small experiment
  requiring (i) bank construction, (ii) SFT GPU time, (iii) its own held-out accuracy/format
  check before any RL launch could even start — strictly more GPU time and more surface area
  for a new confound (the SFT's own specific phrasing statistics could bias what "literal"
  means in a way checkpoint 500, an organically-RL-trained artifact, does not). No concrete
  disqualifying reason was found for (a) — LoRA config compatible, accuracy floor adequate —
  so the default stated in the task instruction applies.

One side observation, not part of the (a)/(b) comparison itself: the base Qwen2.5-3B-Instruct
model, zero-shot, very likely already defaults to literal Heads/Tails without any SFT at all
(consistent with Stage 1's own premise — RL pressure was needed to try to escape literal use,
not induce it). That would be a third option (no seed checkpoint, no SFT, just the base
model) but the task posed (a) vs (b) specifically, and checkpoint 500 additionally has the
advantage of already being mid-RL-trained rather than a fresh base model, which is closer to
"a real trained policy that happens to be literal" than "a model that has never been
RL-tuned at all."

**Confidence: verified** (adapter_config.json read directly, accuracy/format numbers are the
already-hash-verified re-evaluation evidence, not re-derived here).

## Step 2 — reward and condition structure

Reused **unchanged, by import**, from `experiments/07_positive_signal_annealed_reward/reward_v3.py`:
`score_completion_v2`, `verify_reward_invariant`. No new reward terms.

- **BASELINE**: `dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)`
  — task correctness (`r_task`) + the fixed `p_length`/`p_structure`/`p_state_variation`
  penalties only (these are never zeroed by any reward_params flag — they're structural, not
  reward-shaping, terms). No lexical penalty, no bonuses.
- **MAIN**: `{}` (score_completion_v2's own defaults: annealed `p_CoT` 0.2→2.0 over the run,
  `consistency_magnitude=0.15`, `signal_magnitude=0.15`).

Four seeds each (42/43/44/45), 150 GRPO steps, `GROUP_SIZE=8` — same as every prior stage.
`test_literal_cot_reward.py` (5/5 passing) confirms the reward invariant holds for both
conditions AND, critically, that a literal-word completion (the exact shape checkpoint 500
produces) gets `p_cot=0` under BASELINE and `p_cot>0` under MAIN at full anneal — the one
sanity check this experiment's entire premise depends on, verified by direct execution.

## Step 3 — prompt format (the hard requirement)

`literal_cot_bank.build_conversational_prompt()` is the only function that shapes a prompt
for `train_dataset`; it always returns `[{'role': 'user', 'content': <text>}]`, never a
string. `test_prompt_format.py` (5/5 passing, CPU-only, no model loaded) asserts the
resulting token ids match the chat-template signature (`[151644, 8948, 198, ...]`,
`<|im_start|>system\n...`) and explicitly do NOT match the raw-string signature
(`[24617, 1584, 25, ...]`, `Starting state:...`) already established for this tokenizer in
the prior Qwen prompt-format audit this session.

**Confidence: verified** — automated assertion, not a manual read.

## Step 4 — persistence (non-negotiable)

Implemented in `literal_cot_rl.py`:

- **Per-rollout** (`rollout_records`, every training rollout, all 8 runs): full completion
  text, `r_task`, raw `p_cot` (the unsummed scalar `score_completion_v2` returns, not
  re-derived from the total), `banned_count` (the exact weighted match count summed into
  `p_cot`'s exponent), `taxonomy_category`/`taxonomy_category_name` (from the unmodified
  `taxonomy.classify_candidate`), both leakage-scanner results explicitly labeled
  (`training_penalty_scanner`, `eval_leakage_scanner` — see `leakage_scanner.py`'s module
  docstring for why these must never be conflated).
- **Per-step** (`step_stats`): mean/min/max reward, `reward_variance_nonzero` (the
  non-degenerate-group signal), `fraction_hit_length_cap`, `mean_completion_length_tokens`,
  `nonzero_penalty_fraction` (fraction of that step's rollouts with `banned_count>0` — the
  direct "did the penalty ever bind" signal for MAIN).
- **Per-step advantage stats**: `mean_abs_advantage_pre_clamp`, `mean_abs_advantage_post_clamp`,
  `fraction_advantages_clamped` — captured inside the same `compute_loss` patch that applies
  the clamp, so pre- and post-clamp values come from the exact same tensor, not two separate
  passes.
- **KL**: `mean_raw_per_token_kl` (pre-clamp) and `mean_post_clamp_per_token_kl`, plus
  `fraction_kl_tokens_engaged`, aggregated per step — captured inside the same
  `_get_per_token_logps_and_entropies` patch that applies the KL clamp, for the same reason.
- **Checkpoints**: `model.save_pretrained()` every `CHECKPOINT_EVERY=25` steps for **all 8
  runs, including BASELINE** (BASELINE checkpoints were never saved in any prior stage —
  fixed here, per the task instruction).
- **Greedy eval at every checkpoint** (same 25-step cadence as the checkpoint save, not a
  separate, looser cadence): the fixed ~20-scenario held-out set, every sample recorded with
  completion text, taxonomy tier, correctness, and BOTH leakage scanners.

Everything above lands in one run-level JSON (`stage10_literal_cot_{phase}.json`), written
incrementally (`save_event()` called after every step and every milestone, not only at the
end) so a hard stop or breaker firing mid-run still leaves a complete, readable record up to
that point — matching the established persistence discipline, extended to cover every field
a prior audit this session found `NOT PERSISTED`.

## Step 5 — checkpoint selection rule (pre-declared, dated)

`select_representative_seed.py`, written 2026-09-22, before any of this experiment's runs
exist. Rule: the representative MAIN seed (if one is needed for a figure/table) is the seed
with the **lower of the two middle final-milestone `genuine_correct_rate` values** among the
four MAIN seeds (i.e., index 1 of the score-sorted list of 4) — resolving the "median of an
even-length list" ambiguity explicitly, now, rather than when the four real numbers exist. If
those two middle scores are exactly equal, the lower seed number is used. `test_select_representative_seed.py`
(5/5 passing) includes a check using the ACTUAL Stage 9e Llama numbers (seeds tied at the
*top*, not the middle) to confirm this rule would not even have looked at that tie, since it
selects by median rather than max — the exact class of ambiguity this rule closes.

## Step 6 — cost estimate

**Basis: this project's own historical Qwen2.5-3B, 150-step, GROUP_SIZE=8 telemetry**, not a
generic assumption — `experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-{baseline-v1,baseline-v2,main-v16,main-v17}/*.json`,
`train_runtime` field: 4758.4s, 4390.8s, 5580.1s, 4165.9s → mean **1.31 GPU-hours/run** (same
model, same step count, same group size; that stage's own eval cadence and checkpoint saves
are comparable to this design's 6-checkpoint, 20-scenario cadence, so no separate overhead
correction is applied beyond a margin below).

- 8 runs (4 BASELINE + 4 MAIN) × 1.31h ≈ **10.5 GPU-hours** base.
- + margin for this design's additional per-rollout taxonomy classification (CPU-only, fast)
  and 6× adapter saves/run (each ~7-50MB LoRA-only, seconds each) ≈ 10-15% ≈ **1.0-1.6h**.
- **Total estimate: ≈11.5-12.1 GPU-hours.**
- **Price basis, unverified — confirm against AWS billing** (same disclosed assumption as
  every prior estimate this session): g5.xlarge on-demand, us-east-1, ≈$1.006/hr.
- **Estimated cost: ≈$11.57-$12.18.**

## Fresh per-token KL D_max calibration (required before Part B, not assumed)

Per the task's explicit instruction not to assume Qwen's existing `D_max=2.0` transfers: a
short MEASUREMENT-mode calibration pass (real MAIN reward config, `STAGE10_KL_CLAMP_D_MAX`
unset) against checkpoint 500 specifically must run before the real 8-run launch, using
`kl_calibration.compute_clamp_from_pool` (imported unchanged — the same function already
validated for both Qwen 3B/7B and Llama 8B calibrations this project). This is a **separate,
short GPU action inside Part B (step 7)**, not counted in the Part A estimate above since it
requires GPU time and is gated behind Gate 1 like everything else in Part B.

Advantage clamp (`ADVANTAGE_CLAMP_VALUE=0.4`, `advantage_clamp.py`, imported unchanged) is
**not** recalibrated — it is derived from `GROUP_SIZE=8` alone (the Grubbs bound), which is
unchanged here, matching the established "formula-derived, transfers; logprob-scale-derived,
must be recalibrated per checkpoint" distinction already confirmed for Llama 09e.

## Addendum, before Part B launch: checkpoint 500's documented GRPO-resume fragility

Before launching, `logs/development_log.md` (2026-08-17/18/24) and
`experiments/04_stability_investigation/` were re-checked directly: checkpoint 500 has a
**documented history of hitting the grad_norm/KL breaker within 3-15 steps** across multiple
independent fresh-optimizer GRPO-resume attempts (seeds `20260730`/`20260817`, LR `2e-6` to
`1e-5`, with and without an entropy-loss clamp), traced to a content-level policy/reference
mismatch concentrated in single tokens (meta-commentary/preamble content), not gradual drift
and not the PPO importance ratio (proven mathematically inert at `num_iterations=1`). The
**only run that survived a full target (50 steps)** used `TARGET_LR=1e-6` with a 12-step
warmup, an entropy-loss clamp, AND a per-token KL clamp calibrated as
`median + 1.5*IQR` of a per-token pool **restricted to steps whose own logged grad_norm/KL
were NOT already within 2x of the breaker thresholds** — the first calibration attempt
against this exact checkpoint (`kl_clamp_mitigation_v1`) failed specifically because it
skipped that step-level filtering and calibrated on a pool contaminated by near-blowup steps,
producing a clamp that essentially never engaged.

**Fix applied to `literal_cot_rl.py` before any GPU time this session** (found during this
re-check, not by trial and error against real GPU runs): the KL pool was being collected as
bare values with no step association, which would have reproduced the v1 mistake exactly.
Changed to a step-stamped pool (`KL_POOL_BY_STEP`), and the final calibration report now
computes BOTH the naive unfiltered pool diagnostics AND a `clean_pool_diagnostics` restricted
to steps below `GRAD_BREAKER/2`/`KL_BREAKER/2` — matching the proven v2 methodology exactly.
The clean-pool value, not the naive one, is what will be used to set `STAGE10_KL_CLAMP_D_MAX`
for the real 8 runs.

**Sequencing, per explicit instruction**: the calibration pass runs FIRST, short, alone,
before any of the 8 full 150-step runs — not concurrently. If it does not survive cleanly
at this experiment's approved config (`TARGET_LR=2e-5`, no entropy clamp — a materially
different, and untested against this specific checkpoint, configuration from the one that
needed 1e-6 + entropy clamp to survive 50 steps in Stage 4), the 8-run launch does **not**
proceed and this is reported back before any further GPU spend, per the explicit "don't
let a bad clamp value burn through the estimated budget" instruction.
