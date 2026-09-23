# 10 — Literal-CoT adversarial pressure

## What we tried
Applied the project's standard lexical-penalty condition (`reward_v3.score_completion_v2`'s
annealed `p_CoT`, 0.2→2.0 over the run) directly to checkpoint 500 — a policy **confirmed
100% literal** (`experiments/01_direct_rl_coinflip/aws_recovered/checkpoint500_reeval.json`,
500/500 rollouts tier 1) at initialization. BASELINE (correctness + fixed structural penalties
only) vs MAIN (BASELINE + annealed lexical penalty + consistency/signal bonuses), 4 seeds each
(42/43/44/45), 150 GRPO steps, `GROUP_SIZE=8` — 8 runs total.

## What this experiment tests that prior experiments did not
**This is the only run in the project where the lexical penalty had substantial, sustained
opportunity to bind at initialization.** Every prior lexical-penalty run either started from a
policy that already used non-literal code by construction (Llama 09e), or resumed a policy that
had already drifted away from literal output before the penalty could act on literal words
(Stage 1 itself produced checkpoint 500 as the *outcome* of such drift; Stage 7 resumed
checkpoint 500 for a different mechanism, r_signal, without re-testing lexical-penalty binding).
Stage 7's own final result was 0/500 non-literal rollouts at every milestone — the penalty had
nothing to penalize. Stage 10 inverts that setup on purpose: start 100% literal, apply the same
penalty, and see whether adversarial pressure can induce or sustain non-literal encoding when
the penalty actually has literal words to act on from step 1.

## Why we tried it
If the lexical penalty is ever going to demonstrably bind and shape behavior anywhere in this
project, it has to bind here — nowhere else gives it literal-word rollouts to penalize in the
first place. A null result here is a much stronger claim than a null result anywhere else in
the project: it isn't "the penalty had nothing to act on," it's "the penalty acted on every
single rollout and still didn't move the model off literal encoding."

## Config
- Seed policy: checkpoint 500 (LoRA `r=8, lora_alpha=16, target_modules=[q_proj,v_proj,k_proj,o_proj]`,
  resumed via fresh `get_peft_model()` + `set_peft_model_state_dict()` per run — no `base_model`
  reuse across calls). See `design.md` Step 1 for the full (a) vs (b) seed-policy comparison.
- Reward: `reward_v3.score_completion_v2` imported unchanged. BASELINE zeros
  `signal_magnitude`/`consistency_magnitude`/`cot_min_scale`/`cot_max_scale`; MAIN uses the
  function's own defaults (annealed `p_CoT`, `consistency_magnitude=0.15`,
  `signal_magnitude=0.15`). Fixed structural penalties (`p_length`, `p_structure`,
  `p_state_variation`) apply to both conditions unconditionally.
- Prompts: `literal_cot_bank.build_conversational_prompt()` — conversational
  (list-of-dict, `[{'role': 'user', 'content': ...}]`) only, verified against the actual
  Qwen2.5-3B-Instruct chat-template token signature by `test_prompt_format.py` (5/5 passing,
  CPU-only, no manual read).
- Advantage clamp: `ADVANTAGE_CLAMP_VALUE=0.4` (`advantage_clamp.py`, unchanged — formula-derived
  from `GROUP_SIZE=8`, does not need recalibration per checkpoint).
- Per-token KL clamp: **freshly calibrated for this checkpoint**, not assumed —
  `D_max = 0.12955999374389648`, from a 12-step MEASUREMENT-mode pass
  (`stage10-literal-cot-main-v1`) whose telemetry was fully clean (max grad_norm 2.75 vs 200
  breaker, max KL 0.078 vs 5.0 breaker — no step within 2x of either breaker), so the clean-pool
  and naive-pool diagnostics were identical (n=3048 nonzero tokens).
- Checkpoints + greedy eval every 25 steps, for **all 8 runs including BASELINE** (fixing that
  BASELINE checkpoints were never saved in any prior stage), on a fixed ~20-scenario held-out
  set, scored by both `training_penalty_scanner` (case-sensitive, pre-`<answer>`, exactly what
  the reward function sees) and `eval_leakage_scanner` (case-insensitive, whole-completion) —
  see `leakage_scanner.py`.
- Checkpoint-selection rule, pre-declared 2026-09-22 before any run existed:
  `select_representative_seed.py` — lower of the two middle-ranked final-milestone scores among
  the 4 MAIN seeds, tie-broken to the lower seed number.

## Results

**Final-milestone (step 150) genuine-correct rate, per condition (n=4 seeds each):**

| condition | mean | sd | min | max |
|---|---|---|---|---|
| BASELINE | 0.700 | 0.0913 | 0.60 | 0.80 |
| MAIN | 0.700 | 0.0707 | 0.60 | 0.75 |

Welch's t-test (MAIN vs BASELINE): **t = 0.0, df = 5.647, p = 1.0** — the two conditions'
means are numerically identical; no evidence of any difference.

**Full 10-tier taxonomy distribution, final checkpoint (step 150), all 8 runs (n=20
held-out scenarios each):**

| run | tier 1 | tiers 2–10 |
|---|---|---|
| BASELINE seed 42 | 20/20 | 0 |
| BASELINE seed 43 | 20/20 | 0 |
| BASELINE seed 44 | 20/20 | 0 |
| BASELINE seed 45 | 20/20 | 0 |
| MAIN seed 42 | 20/20 | 0 |
| MAIN seed 43 | 20/20 | 0 |
| MAIN seed 44 | 20/20 | 0 |
| MAIN seed 45 | 20/20 | 0 |

Every one of 160 final-checkpoint held-out completions, across all 8 runs, is tier 1 (literal).
**No completion at any final checkpoint, in any run, ever left tier 1** — not tier 2 (vacuous),
not any of tiers 3–8, and not tier 9/10.

**No tier 9 or tier 10 result anywhere** — not at any milestone checkpoint (every 25 steps,
all 8 runs) and not in any of the ~9,600 individual training rollouts across all 150 steps ×
4 MAIN seeds × GROUP_SIZE=8. Because no tier 9/10 result appeared, the bidirectional
counterfactual intervention protocol (required only if one did) was not run.

**Per-rollout penalty engagement rate, MAIN specifically** — the fraction of training rollouts
with nonzero `banned_count` (i.e. the penalty actually fired):

| seed | nonzero fraction | mean `p_cot_raw` (when computed) |
|---|---|---|
| 42 | 1.000 (1200/1200) | 0.3874 |
| 43 | 1.000 (1200/1200) | 0.3935 |
| 44 | 1.000 (1200/1200) | 0.3957 |
| 45 | 1.000 (1200/1200) | 0.3839 |

**100% of every single training rollout, in every 10-step bin, across all 150 steps, for all
four MAIN seeds, had the lexical penalty fire** — a categorically different result from every
other lexical-penalty run in this project (0% or near-0% engagement everywhere else). The
penalty had every opportunity this experiment was designed to give it.

**Pre-declared representative seed** (per the rule above): **seed 42** (score 0.70), no tie —
sorted MAIN pairs `[(44,0.60), (42,0.70), (43,0.75), (45,0.75)]`, index 1 = seed 42.

## Final result
The lexical penalty bound on literally every training rollout of every MAIN run (100%
engagement, mean magnitude ≈0.38–0.40, a substantial fraction of the 0.70 reward-invariant
margin between correct and wrong) and still produced **zero measurable effect**: MAIN's
final-milestone accuracy is statistically indistinguishable from BASELINE's (t=0.0, p=1.0), and
not a single held-out or training rollout across all 8 runs ever left tier 1 (literal). This is
the strongest negative result on lexical-penalty efficacy available in this project, precisely
because it is the one condition engineered to give the penalty maximum opportunity to act. The
mechanism itself works as designed (verified via unit tests before any GPU time, and via
100%-nonzero engagement telemetry during training) — it simply does not move this policy off
literal encoding, even under continuous, saturating pressure.

## Scope
This experiment establishes a null result for **lexical-penalty pressure alone, applied to
checkpoint 500, at this reward magnitude, over 150 steps**. It does not test: higher penalty
magnitude beyond `score_completion_v2`'s existing anneal ceiling, longer training horizons,
combination with positive-signal bonuses at higher weight, or a different seed policy (a fresh
SFT bank, option (b) from `design.md` Step 1, was not run). No claim is made about lexical
pressure's effect on any other checkpoint, model family, or task design in this project.

## Cost — actual vs. estimated
- **Estimated** (Gate 1, `design.md` Step 6, based on `09d` decode-back historical
  `train_runtime` telemetry, price basis explicitly unverified ≈$1.006/hr g5.xlarge on-demand):
  **≈11.5–12.1 GPU-hours / ≈$11.57–$12.18**, for the 8-run training batch alone (calibration
  budgeted separately, per-run overhead not separately itemized).
- **Actual, full Stage 10 billed instance time** (all sessions: Part A dependency-file
  prechecks + the boot-failure incident + the retry/calibration/8-run-batch/teardown session),
  reconstructed from `stage10_timing.txt` markers cross-checked against
  `aws ec2 describe-instances` `LaunchTime`/`StateReason` and the final teardown's own
  independently-reverified `stopped` state:

  | segment | window (UTC) | duration |
  |---|---|---|
  | Part A dependency precheck | 04:57:01–05:05:54 | 8m53s |
  | Part B precheck | 05:06:21–05:11:49 | 5m28s |
  | Part B step-7 precheck | 05:16:14–05:34:19 | 18m5s |
  | Boot-failure attempt (UEFI loop, no useful work) | 05:53:48–06:21:07 | 27m19s |
  | Retry boot → calibration → 8-run batch → teardown | 08:04:22–~02:20:00 (+1d) | ≈18h16m |
  | **Total billed** | | **≈19h16m ≈ 19.3 GPU-hours** |

  **Actual cost: ≈$19.35** at the same unverified $1.006/hr basis — about 60% over the Gate-1
  estimate. The overrun is not attributable to the 8-run training itself (150-step GRPO time at
  this scale is consistent with the `09d` telemetry the estimate was based on); it is
  attributable to (a) the boot-failure incident (27m19s, wholly unproductive, an infrastructure
  fault unrelated to this experiment's design — see the mid-session incident report), and
  (b) an anomalously long ≈4-hour gap between the retry's `start-instances` call (`LaunchTime`
  08:04:22Z) and confirmed SSH/status-check reachability (`RETRY_BOOT_OK` 12:05:58Z), which this
  session's coarse timing markers do not further decompose — flagged honestly as unexplained
  rather than assumed benign, since the instance was billed as "running" throughout regardless
  of reachability. The 8-run batch's own per-run training+eval+checkpoint overhead (not
  separately itemized in the Gate-1 estimate, which counted step time only) also plausibly
  accounts for part of the gap between the ≈18h16m continuous session and the ≈11.5–12.1h
  training-only estimate — 48 greedy held-out evaluation passes (6 checkpoints × 8 runs) and 8
  separate fresh-checkpoint LoRA loads were not itemized in the original estimate.

Final teardown: **TEARDOWN FULLY CONFIRMED** (`i-REDACTED`), independently re-verified
via a fresh `describe-instances` call (`State.Name=stopped`,
`StateReason=Client.UserInitiatedShutdown`) rather than accepting the teardown script's own
report at face value.

Evidence: `aws_runs/` (all 9 run directories — 1 calibration + 4 BASELINE + 4 MAIN — each with
hash-verified `stage10_literal_cot_{phase}.json` and a filtered stdout log with the raw log's
sha256 recorded for provenance); `design.md` (full Part A write-up, pre-launch checkpoint-500
GRPO-resume-fragility addendum, and the KL-pool step-stamping fix made before any GPU time);
`select_representative_seed.py`/`test_select_representative_seed.py` (pre-declared rule);
`test_stage10_results.py` (this README's key numbers, pinned against the actual evidence JSONs).
