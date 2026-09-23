# Development log

## 2026-08-11 — Stage 3 checkpoint accepted with threshold exception

Checkpoint 130 is the final accepted Stage 3 seeding result. It did not clear
the registered 95% threshold across all seven metrics: final-answer accuracy
92.5%, structure 100%, transition tracking 90.5%, global consistency 89.0%,
mapping adherence 89.5%, decode-back 92.0%, and nonliteral use 100%.

The exception is explicit and intentional. The mapping anchor eliminated the
original consistent-swap defect (8 of 11 mapping failures at step 100, versus
0 of 2 at step 130), while structure and nonliteral code use reached 100%.
Remaining failures were diffuse transition errors. A checkpoint-130-to-140
explicit-transition diagnostic reduced pure transition failures from 19 to 14
but reintroduced 25 consistent mapping swaps and reduced answer accuracy to
79%; checkpoint 140 is therefore rejected and must not be used downstream.

Stage 3.5 is authorized as a read-only, zero-shot third-domain evaluation of
checkpoint 130. Stage 4 adapter composition remains blocked pending review of
that report.

## 2026-08-11 — Decode-back promoted to an independent transfer metric

Both Stage 3.5 controls showed cross-domain final answers copying code words
instead of returning the physical-domain state. The multi-domain Stage 3 plan
therefore treats decode-back generalization independently from tracking and
mapping consistency. Every evaluation checkpoint and the next Stage 3.5 rerun
must report the overall `answer_is_code_word_rate`, its fraction among answer
failures, and the mutually exclusive failure category
`code_word_instead_of_physical_state`. A run that fixes mapping consistency but
not decode-back is a partial success, not an aggregate pass. The complete
pre-registration is in `experiments/03_demonstration_seeding_multi_domain/multidomain_stage3_plan.json`.

Before multi-domain training is authorized, the exact declared-mapping control
must also be evaluated against the unadapted Qwen2.5-3B-Instruct base. A base
adherence result at or above 30% versus checkpoint 130's 5% would be evidence
that latch SFT actively damaged general capability; a result in the 5–15% band
would instead support intrinsic task difficulty at this model scale. No
multi-domain training is authorized until this matched control is reviewed.

The matched base control subsequently completed on 100 prompts
(`experiments/03_demonstration_seeding_multi_domain/notebooks/stage35_base_declared_mapping_control.ipynb`,
verified against its actual executed output, not just this summary). The
unadapted base achieved: declared-mapping adherence 3%, final-answer
accuracy 3%, structure 83%, global consistency 7%, nonliteral encoding
83%, and an 85% code-word-in-answer rate (87.6% of its wrong answers were
the declared code word instead of the physical state).

The checkpoint-130 (latch) comparison is recorded in
`experiments/03_demonstration_seeding_multi_domain/multidomain_stage3_plan.json`'s
`untrained_base_control.checkpoint130_comparison` block: declared-mapping
adherence 5%, final-answer accuracy 1%, structure 88%, **global consistency
11%**, and nonliteral encoding 88%. This block has no code-word-in-answer
field for checkpoint 130 -- the notebook that would have computed one
(`stage35_declared_mapping_control.ipynb`) was built but never executed,
so that specific metric was never measured on the checkpoint-130 side.
(An earlier version of this entry listed only 4 checkpoint-130 numbers
without naming which metric each was, which could be misread as one
value having gone missing; global consistency was always the 4th number,
11%, and no value was ever lost -- the code-word-rate metric was simply
never computed for checkpoint 130 in the first place. Corrected 2026-08-23
after this ambiguity was flagged and traced back to source.)

Comparing the four metrics reported on both sides (adherence, accuracy,
structure, global consistency), the base and latch-adapted checkpoint 130
are similarly poor across all of them. Therefore the latch adapter did not
erase a strong pre-existing transfer capability; the task is intrinsically
difficult for this 3B model zero-shot, and the multi-domain training
design is supported. Stage 4 remains blocked.

## 2026-08-17 — Multi-domain Stage 3.5 undeclared test: invention failure, not a domain-difficulty confound

The read-only Stage 3.5 undeclared/self-invented-mapping test
(`experiments/03_demonstration_seeding_multi_domain/notebooks/stage35_third_domain_zero_shot.ipynb`) ran on the multi-domain
checkpoint 130 (`multidomain-stage3-v1/trainer-output/checkpoint-130`),
zero-shot on the held-out lamp domain, no mapping supplied, self-consistency
scoring only: verified nonliteral encoding rate 10%, final-answer accuracy
49%, `passed: false`.

The matched declared-mapping lamp baseline at the same checkpoint (identical
seed `20260812`, identical 100 held-out examples, mapping supplied) scored
86% accuracy — close to pooled (89.3%) and fan (85%) at that checkpoint. Lamp
is therefore not an inherently harder domain when given a mapping, so the
49%/10% undeclared collapse is attributable to invention failure, not domain
difficulty. Training also continued unreviewed from checkpoint 100 to 130
(`experiments/03_demonstration_seeding_multi_domain/notebooks/multidomain_stage3_step100_to130.ipynb`) without an explicit
authorization recorded between those steps; both results and gaps are on
record in `experiments/03_demonstration_seeding_multi_domain/multidomain_stage3_plan.json`. Stage 4 remains blocked
pending a decision on how to treat the invention-failure result.

## 2026-08-17 — Checkpoint-500 GRPO resume is unstable within 3-4 steps, independent of seed

Two separate checkpoint-500 diagnostic dry-runs, under different seeds and
different LR schedules, both hit the `grad_norm>=50 or kl>=5` safety breaker
within the first few optimizer steps:

- Seed `20260730`, 5-step linear warmup to a 2e-6 target LR
  (`experiments/01_direct_rl_coinflip/notebooks/checkpoint500_corrected_full_reward_warmup8_dryrun.ipynb`):
  stopped at step 3, `grad_norm=1736`, `kl=102`.
- Seed `20260817`, flat `1e-5` LR with no warmup
  (`experiments/04_stability_investigation/notebooks/checkpoint500_seed_diversity_diagnostic.ipynb`, built to test
  seed-driven exploration diversity): stopped at step 4, `grad_norm=151`,
  `kl=17.2`.

Both runs resume checkpoint-500 adapter weights only with a fresh
optimizer/scheduler (`weights_only_fresh_schedule`), same reward stack, same
`entropy_coef=0.05`. Because both seeds blow up almost immediately regardless
of how aggressive the LR schedule is, the original seed-diversity question
(does a different seed show different early exploration diversity in the
first 50-100 steps) is inconclusive — neither run got far enough to observe
it. The more load-bearing finding is that checkpoint 500 itself appears
fragile to a fresh-optimizer GRPO resume under the current reward/entropy
configuration, independent of seed. n=2, so this is suggestive rather than
conclusive.

Cross-reference: Skaf et al. (arXiv 2506.01926) Appendix A.1.1 report an
entropy loss coefficient of 0.02 (vs. our 0.05) paired with an explicit max
entropy-loss clamp (0.6 for their Coin Flip task) specifically "to avoid the
entropy loss exploding." We have no equivalent clamp. This is suggestive that
an entropy-loss clamp, gradient clipping, or a longer/gentler warmup should be
tried before any further checkpoint-500 continuation or seed-diversity
sweep — not a confirmed fix, but the most plausible next experiment given
where both runs failed.

## 2026-08-17 — Entropy clamp + gentler warmup delayed but did not prevent the checkpoint-500 blowup

`experiments/04_stability_investigation/notebooks/checkpoint500_entropy_clamp_diagnostic.ipynb` added a hard clamp
on the entropy-loss contribution (`entropy_coef * entropy` bounded at
`0.05 * 1.70`, by clamping `entropy_from_logits()`'s output at the source)
plus a gentler 12-step warmup to a 1e-6 target LR, same seed (`20260817`),
same reward stack, run against checkpoint 500. Goal was survival to 50+
steps; result: `survived: false`, hard stop at **step 7** (`grad_norm=128`,
`kl=6.96`).

This is a partial improvement over both prior attempts (stopped at step 3-4
with far larger blowups: `grad_norm` 1736 and 151, `kl` 102 and 17.2) but
nowhere near the 50-step survival bar. Two points against entropy or LR
magnitude being the actual driver: (1) the clamp did engage during the run
(10.3% of micro-batches, raw pre-clamp entropy up to 6.9 nats observed) yet
the blowup still happened; (2) entropy at the blowup step itself was
unremarkable (0.68, mid-range) and the learning rate at that point
(5.9e-7) was smaller than the LR at either prior run's blowup point. KL did
creep upward across steps 1-7 (0 -> 0.35 -> 1.49 -> 0.06 -> 0.85 -> 1.97 ->
6.96) before the final jump, consistent with gradual policy drift compounding
into a threshold-crossing step rather than a single random catastrophic
outlier, though this is a single run and not conclusive.

Per the pre-registered decision rule for this diagnostic: entropy/LR
hyperparameter tuning has now been tried twice and both times failed to
reach survival, with direct evidence against entropy magnitude as the
mechanism. Next step is reviving the per-token importance-ratio
investigation (interrupted earlier in this project by a lost kernel state)
rather than further hyperparameter guesses. Not yet started.

## 2026-08-18 — Per-token KL instrumentation: importance ratio proven inert, blowup traced to a content-level policy/reference mismatch

`experiments/04_stability_investigation/notebooks/checkpoint500_per_token_kl_instrumentation.ipynb` revived the
never-executed `checkpoint500_warmup_per_token_kl_diagnostic.ipynb` design
(same seed `20260817`, same entropy clamp + 12-step warmup config) with
full per-token/per-rollout capture. Recovered per-token KL matched TRL's own
logged `kl` metric to ~7 significant figures at every step through
termination, validating the capture pipeline before trusting any
interpretation of it.

Two findings:

1. **PPO-style importance ratio is exactly 1.0 at every token, every step,
   by construction** — not empirically observed, provable from
   `num_iterations=1` and `steps_per_generation == gradient_accumulation_steps
   == 8`, which triggers TRL's own `old_per_token_logps = None ->
   per_token_logps.detach()` fallback. The clipped-surrogate mechanism is
   mathematically inert in this configuration; it cannot be the blowup
   driver regardless of what any run shows.
2. This run survived to **step 15** (further than either prior attempt)
   before a single catastrophic jump: `grad_norm=2864`, `kl=197.7`, ~600x
   the surrounding steps (1-14 were noisy but flat, 0.02-3.0, no creeping
   trend — retracting the "gradual drift" read from the entropy-clamp run
   above). The blowup was concentrated in one token of one rollout out of
   eight (`blowup_concentrated_in_single_rollout: true`; position 1 alone
   contributed KL=400,346). The completion text at that token, and at the
   two next-most-elevated steps (5 and 8), all opened with meta-commentary/
   instructional-preamble content before genuine step-by-step reasoning
   (e.g. "complete example (but without the final answer tag...)", "Replace
   the question completely.", "Note: The reasoning isn't a step..."). No
   verbatim rollout repeated across steps (corrected recurrence check, see
   below, confirmed `genuine_cross_step_recurrence_found: false`) — this is
   the same failure *category* recurring with different literal wording
   each time, not the same completion.

A bug was caught and fixed before trusting this: the instrumentation
notebook's own `recurring_rollout_hashes_across_steps` field counted
elevated tokens per rollout, not distinct steps per rollout hash, so it
could not actually show cross-step recurrence.
`experiments/04_stability_investigation/notebooks/checkpoint500_kl_evidence_reader.ipynb` (CPU-only, reads the saved evidence
JSON, no GPU) recomputed it correctly.

## 2026-08-19 — Greedy decoding on checkpoint 500 does not show the preamble pattern: points to sampling, not the checkpoint itself

`experiments/04_stability_investigation/notebooks/checkpoint500_greedy_preamble_check.ipynb` ran plain greedy
(`do_sample=False`) decoding on 200 diverse fresh-seed Coin Flip prompts
against checkpoint 500, pure inference, no training. Detector (verified
against the three real positive completions above, all three flagged
correctly, plus a clean negative unflagged) found **0/200 flagged** for the
meta-commentary/preamble pattern. Rule-of-three bound: rate is roughly
<=1.5% with ~95% confidence, not proven zero. Greedy task accuracy at
checkpoint 500: 68.5% (informational).

Per the pre-registered decision rule: this is the "not evident under greedy
decoding" case, which points toward high-temperature GRPO sampling
occasionally landing in a region the reference model strongly disfavors,
rather than a pre-existing latent behavior baked into checkpoint 500 itself.
Candidate fix is a per-token KL clip or a lower rollout-sampling
temperature, not an upstream reward-function/base-training change. No
mitigation has been built yet pending this decision being reviewed.

## 2026-08-19 — KL clamp mitigation built; two operational bugs caught before it could run

`experiments/04_stability_investigation/notebooks/checkpoint500_kl_clamp_mitigation.ipynb` adds a per-token KL
clamp (`diff = ref_per_token_logps - per_token_logps` clamped to
`[-D_MAX, D_MAX]` via a mutated `inputs['ref_per_token_logps']`, so TRL's
own unmodified k3 formula produces the bounded result -- no reimplementation
of `compute_loss`). `CLAMP_VALUE` is derived empirically from the
instrumentation run's own data (8x the healthy-steps' max per-token KL,
where "healthy" is a data-driven 3x-median rule -- step 10 turned out to be
elevated too, not just 5/8/15 as originally assumed). Mechanism verified in
four cases against a stand-in for TRL's real formula: pathological token
bounded exactly, healthy tokens pass through with zero distortion, gradient
through the clamp matches the analytical straight-through value, and the
unclamped-region gradient is bit-for-bit identical to a no-clamp baseline.

Two real bugs surfaced before this could run, both now fixed:

1. The user's first run got killed -- diagnosed as Jupyter/Colab's IOPub
   message-rate limiter, not memory: this run targets 50 steps (3-6x any
   prior attempt), so per-call library warnings that were a minor nuisance
   over a handful of steps compounded into real volume. Fixed with explicit
   warning/logging suppression in the setup cell.
2. The second run failed with `ValueError: Your setup doesn't support
   bf16/gpu` deep inside trainer construction. Root cause: this notebook's
   setup cell had been rewritten from scratch (to fold in Drive-based
   clamp-value derivation) and had dropped the GPU/version guard every
   other script in this project carries (`torch.cuda.is_available()`, the
   L4-specific check, exact package versions) -- so a non-L4 GPU assignment
   (bf16 requires Ampere+) wasn't caught until deep in a confusing stack
   trace instead of immediately in cell 2. Restored the guard, and added
   the same L4 check to `checkpoint500_greedy_preamble_check.ipynb`, which
   had the identical latent gap (bf16 model load, no L4 check) even though
   it hadn't been triggered yet.

Not yet re-run. This is the next result to check once GPU access returns.

## 2026-08-24 — KL clamp mitigation run 1 (AWS EC2, g5.xlarge): did not survive, clamp barely engaged

GPU access returned via AWS (EC2 `g5.xlarge`, A10G, us-east-1), not Colab.
The notebook was converted to a plain Python script for headless execution
(`experiments/04_stability_investigation/aws_runs/kl_clamp_mitigation_v1/run_script.py`,
mechanically extracted from the notebook's own cell source with only
Colab-specific bits removed -- Drive mount, `%pip install`, and the
hardware gate widened from L4-only to any compute-capability-8.0+ GPU,
since bf16 is an Ampere+ requirement, not an L4-specific one). Checkpoint
500 and the instrumentation run's saved evidence were pulled from Google
Drive onto the instance via `rclone`; the checkpoint's `manifest.json`
`roundtrip_verified: true` gate was checked before use.

**Result: did not survive. Hard stop at step 14** (`grad_norm=109.5`,
`kl=13.08`), one step short of even matching the earlier *unclamped*
instrumentation run's own step-15 stopping point. The clamp mechanism
itself worked correctly (TRL's own logged `kl` matched the offline
reconstruction exactly, confirming the clamp genuinely bounded the trained
metric) but was calibrated far too loose to matter: `CLAMP_VALUE` was
derived as 8x the raw max of per-token KL values across steps classified
"healthy" by a 3x-median rule (`10334.09`). Across the entire run's 23,712
scored tokens, only **1 token ever exceeded that value** (a spike to
21,399.7, clamped to 10,334.09) -- everything else passed through
untouched. That single clamp wasn't enough to pull the step's *aggregate*
KL/grad_norm back under the breaker thresholds, since the breaker fires on
the batch mean, not one token's value.

Full evidence: `checkpoint500_kl_clamp_mitigation.json`,
`per_token_instrumentation.json`, and the raw `stdout.log`, all in
`experiments/04_stability_investigation/aws_runs/kl_clamp_mitigation_v1/`.

## 2026-08-24 — KL clamp recalibration: why 8x-of-max was the wrong baseline, and what replaced it

Before a second attempt, two questions were checked directly against the
data rather than guessing at a new multiplier:

1. **Was the v1 baseline itself contaminated by borderline data?** Re-read
   the prior instrumentation run's own per-step logged `grad_norm`/`kl`
   telemetry and excluded any step within 2x of that run's own breaker
   thresholds (`grad_norm>=25` or `kl>=2.5`), not just the single worst
   step. Steps 5, 8, 10, and 15 (the hard stop) were excluded on this basis
   -- steps 5, 8, and 10 had previously been included in the "healthy" pool
   under the old 3x-median rule. From the resulting 15,640-value clean
   pool, `median=0.0`, `Q3=0.0675`, so `median + 1.5x IQR` (Tukey's
   standard mild-outlier fence) gives a baseline of `0.1012` -- roughly four
   orders of magnitude below the old method's raw max of `1291.76`. The old
   "max of healthy steps" approach was a single extreme order statistic,
   inheriting noise from steps that had only narrowly avoided tripping the
   breaker themselves.
2. **Is this one catastrophic token or many moderately-bad ones?** Recomputed
   the full per-token KL distribution at v1's failing step (14) across all
   8 rollouts (1,232 tokens total, not just the single worst rollout
   printed in the original report). The worst single token was 78.7% of
   the step's total KL sum -- but even excluding it entirely, the remaining
   sum still averaged 4.69, right at the breaker's edge. Not a single clean
   outlier: a second token independently spiked to 5,364, plus a real tail
   of tokens in the 10-100 range. A per-token clamp is a plausible tool for
   this (removing 14% of tokens' excess should address most of the
   aggregate), but the margin is thin.

New `CLAMP_VALUE = 1.5x` the recalibrated baseline = `0.152` (vs. v1's
`10334.09`). An offline replay against v1's actual step-14 data (clamp the
same recorded per-token values, no live rerun) predicted a post-clamp mean
KL of `0.027` against the `5.0` breaker, with 14% of tokens engaged --
evidence the mechanism is now in a plausible range, though not proof the
live run survives, since clamping changes the actual gradients applied
during training, not just an offline metric.

Run 2, with this recalibration built into the script's own derivation
logic (not a bolted-on external value), is in
`experiments/04_stability_investigation/aws_runs/kl_clamp_mitigation_v2/`.
See that entry for the outcome.

## 2026-08-24 — KL clamp mitigation run 2: survived the full 50-step target

Before the run, a real operational bug was caught and fixed: the loading
config carried `llm_int8_enable_fp32_cpu_offload=True`, left over from the
original notebook. On a 24GB A10G loading a ~3B int8 model (v1's *entire
training loop*, weights + LoRA + activations, peaked at 6.5GB), this flag
had no VRAM-pressure reason to be set and was forcing slow, unnecessary
per-layer CPU-side work during model load. Removed it (confirmed safe
given the wide VRAM headroom, not merely assumed -- this exact class of
assumption caused the earlier L4 OOM issue on the original Coin Flip runs).
A second, unrelated false alarm during this run: the process appeared
stuck for ~20 minutes with zero new log output, which turned out to be
stdout buffering (Python fully buffers stdout when redirected to a file
rather than a terminal) hiding real progress, not an actual hang --
confirmed via `py-spy` thread dump (live in a real forward pass, not
blocked) and by reading the JSON evidence file directly, which writes to
disk independently of the buffered stdout log and showed the run was
already many steps into training.

**Result: survived to the full 50-step target.** `terminal_step=50`,
`hard_stop=null`. `kl` stayed in a healthy 0.005-0.07 range for the entire
run, never approaching the 5.0 breaker; `grad_norm` stayed well under 50
throughout. Unlike v1, the recalibrated clamp (`CLAMP_VALUE=0.152`, vs.
v1's `10334.09`) engaged meaningfully: 27.5% of tokens were bounded across
the run (vs. 0.02% in v1), including one raw spike of 2,371,240 clamped
down to 0.152. `kl_reconstruction_verified_against_trl_logged_metric:
true` -- TRL's own logged metric matches the offline reconstruction
exactly, so this is a verified result, not a self-consistent-only
artifact.

Full evidence: `checkpoint500_kl_clamp_mitigation.json`,
`per_token_instrumentation.json`, and `stdout.log`, all in
`experiments/04_stability_investigation/aws_runs/kl_clamp_mitigation_v2/`.

**Conclusion**: checkpoint 500 can survive a GRPO resume under this
reward/entropy configuration when per-token KL is clamped at a properly
calibrated value. The naive 8x-of-raw-max calibration (run 1) failed not
because per-token clamping is the wrong tool, but because "max of a
per-step-mean-filtered healthy pool" is a single extreme order statistic
that inherits noise from borderline steps -- a tighter, more principled
baseline (median + 1.5x IQR of a per-token pool from steps excluded if
within 2x of the run's own breaker thresholds) fixed it. Per the
pre-registered plan, this was the last hyperparameter-side iteration on
this mechanism regardless of outcome; since it succeeded, checkpoint 500
is now a viable Stage 4 merge target, and work proceeds to
`05_self_bootstrapping`.

## 2026-08-19 — Option 1 (self-bootstrapping) setup built, not run; repo cleanup

Built the full pipeline for reinforcing checkpoint 130's own rare
undeclared-invention successes via SFT, everything possible without GPU
access done and verified:

- `experiments/05_self_bootstrapping/data_generation/generate_bootstrap_undeclared_prompts.py` -- run locally
  (no GPU): 8,160 unique undeclared fan/valve prompts, length range
  extended from the original 3-8 to 3-10 (combinatorial ceiling at 3-8 is
  only 2,016 total, far short of the volume needed). Fan/valve chosen over
  lamp specifically to keep lamp available as held-out eval, avoiding
  train/eval contamination. Volume justified against both the point
  estimate (10% -> 8,000) and a Wilson 95% lower bound (5.52% -> 14,485);
  landed on exhausting the 3-10 combinatorial space (8,160) as a
  documented first-pass compromise, not a hard ceiling.
- Classifier reproduction verified twice: locally against the 5 known-
  positive completions already in this repo's saved notebook outputs (all
  5 correctly reproduced), and a stronger row-by-row (not just aggregate)
  reproduction gate built into
  `experiments/05_self_bootstrapping/notebooks/option1_bootstrap_generation.ipynb` against
  the full original 100-row dataset, to run when Drive access returns.
- `experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py` -- classification/filtering/
  dataset-building deliberately split out of the GPU notebook (needs zero
  GPU); unit-tested end-to-end against hand-built synthetic completions
  covering genuine success, literal-state failure, and a self-consistent-
  but-wrong-answer case (added `answer_correct` as an explicit SFT-safety
  filter beyond the original "verified" definition, reported side by side
  for comparability).
- `experiments/05_self_bootstrapping/notebooks/option1_sft_training.ipynb` -- built, not run. One real bug
  caught during testing: its declared-condition scoring cell redefined the
  module-level name `STEP_RE`, silently clobbering the undeclared module's
  own `STEP_RE` for the rest of the notebook's execution (Python closures
  resolve globals at call time) -- and the declared-flavored regex's
  `{2,9}` bound (correct for its own multi-character nonce tokens) would
  have silently rejected legitimate short 2-character invented codes like
  "BL" from real Stage 3.5 data. Fixed by renaming to `DECLARED_STEP_RE`
  etc.; re-verified against real known-positive short-code examples in the
  notebook's actual shared-namespace execution order afterward.

Also this session: repo cleanup pass (stale `__pycache__` removed, this
log and README brought current, archive-move proposal presented for
review before any files were moved).

## 2026-08-24 — Bootstrap generation executed on AWS; domain-leakage gap found and fixed in the mnemonic classifier

GPU access this round came via AWS EC2 (`g5.xlarge`, A10G), not Colab.
`option1_bootstrap_generation.ipynb` was converted to a plain script for
headless execution (`experiments/05_self_bootstrapping/aws_runs/bootstrap_generation_v1/run_script.py`).
The classifier reproduction gate re-confirmed exactly (100/100, 0
mismatches) immediately before generating. All 8,160 completions
generated (batch size raised from an initial 16 to 64 after an OOM smoke
test confirmed wide VRAM headroom -- peak 4.24GB of 23.69GB -- cutting
the run from an estimated ~2.6h to ~40min).

`verified`: 575/8,160 = 7.05% (0.70x the known 10% lamp baseline, within
the same ballpark). `sft_eligible` (verified + correct final answer):
421/8,160 = 5.16%. Domain breakdown surfaced an asymmetry: fan's
`sft_eligible` rate (3.04%) is far below its `verified` rate (6.50%),
while valve's are close (7.28% vs 7.60%). Inspecting 15 fan
verified-but-failed completions found zero tracking errors -- 100% is a
single mechanical bug: the model correctly decodes the final state in
its own "Final coded state: X. X represents Y" line, then writes the
code X into `<answer>` instead of the physical state name Y it just
correctly decoded. Same bug in valve (100% of its 13 failures), far less
often (4.2% of valve's verified rows vs 53.2% of fan's).

A mnemonic-vs-arbitrary code classifier was built to check whether
checkpoint 130's invented codes actually conceal the tracked state or
just restate it in shorthand (e.g. `R` for "Running"). Initial result:
~85% mnemonic-adjacent, ~15% genuinely arbitrary, confirmed not a
filtering artifact (82.2% among `sft_eligible` vs 85.0% among all
verified is nearly identical). Two SFT datasets were planned to test
this directly: Dataset A (arbitrary-only, 46 examples: 23 fan + 23
valve) vs Dataset B (representative mix, 248 examples).

Before training, manual inspection of Dataset A's 46 candidates (full
read, not a sample -- feasible at this size) found the classifier had a
real gap: it only checked each code against its own STATE name, never
against the DOMAIN name. 100% of both fan and valve's 46 candidates had
at least one code starting with the domain's own first letter (`FZ`,
`Fx` for fan; `VAL`, `Va`, `Vi` for valve) -- technically non-mnemonic
by the original rule, but still leaking "what task this is" even when
correctly avoiding "which state this is." Also found: only 12 distinct
code pairs across the 46 rows (74% redundancy, one pair alone covering
10/23 fan rows), and valve's `final_answer` was "Closed" in all 23/23
rows with zero exceptions -- a spurious-shortcut risk independent of the
leakage issue.

Fixed the classifier by extending the same rule one level up (does the
code resemble the domain name, not just the state name) -- unit-tested
for both the missed cases (`VAL`, `Va`, `Vi` now correctly rejected) and
against over-rejection (mid-word `F`/`V`, and genuinely arbitrary codes
like `Zorp`/`Blib`, still correctly pass). Re-ran the tightened
classifier against the full 8,160-completion pool (no new sampling):
the effect is total, not partial. **0 of 575 verified rows are
genuinely clean under both checks.** Checked whether clean codes exist
anywhere in the raw data regardless of verification: yes (9 fan, 180
valve) -- but every single one is in a row that *failed* the
self-consistency check. Genuine concealment and successful
self-consistent tracking look mutually exclusive in this data, not
merely rare together, plausibly because a domain/state-adjacent code
doubles as an implicit memory aid the model leans on to stay
consistent, and removing that scaffold costs tracking reliability.
Suggestive at n=575/0, not proof, but not a pattern more generation
volume at the same rate would obviously fix.

The 10%/7.05% base invention-rate numbers do **not** need re-checking
against this fix -- they were never based on a mnemonic or domain-name
definition to begin with, only structural self-consistency and correct
decode-back. The gap only affects the separate, deeper mnemonic-vs-
arbitrary sub-analysis introduced to look at *what kind* of codes get
invented, not the base rate itself.

**Decision**: Dataset A (arbitrary-only) is infeasible from current
data -- not thin, empty. Proceeding with Dataset B only (248 examples,
124 fan + 124 valve, representative of the model's actual invention
mix) for the SFT training step. The arbitrary-vs-mnemonic comparison
remains open, not answered here; revisiting it would need either a much
larger generation run or a redesigned experiment accepting
imperfectly-tracked arbitrary-coded traces as training targets.

## 2026-08-24 — Follow-up: precise 575-row breakdown, and the lamp baseline is unaffected

Two follow-up checks against the finding above, before Dataset B's
eventual evaluation locks in a comparison baseline.

**Precise breakdown of the 575 verified rows** (the exact set the
7.05%/0.70x-baseline figure is based on), not inferred from the earlier
0/575-clean count but classified directly: **575/575 (100%) are
mnemonic-or-domain-leaking, 0/575 (0%) genuinely arbitrary** --
identically 0% in each domain individually (0/265 fan, 0/310 valve).
"7.05% invention" on fan/valve currently means entirely mnemonic-
adjacent invention, none of it genuinely arbitrary.

**Retroactively applied the same tightened classifier to the ORIGINAL
10% lamp baseline** (`stage35-undeclared-lamp-checkpoint130-v2/rollouts.json`,
the source of `03`'s finding), to check whether that number would also
collapse under a consistent definition: it does not. **10/10** of the
lamp baseline's verified successes are genuinely clean -- 0% leakage,
the opposite pattern from fan/valve. This sharpens rather than confirms
the tension hypothesis: lamp is the one domain checkpoint 130 has zero
prior training exposure to (held out entirely, never seen even in
declared-mapping form), while fan/valve were trained on extensively in
declared form before this undeclared test. So fan/valve's leakage could
reflect the tracking-vs-concealment tension recorded above, or simply
that prior declared-training exposure to a domain primes semantically-
loaded code choices in that same domain specifically, with lamp
unaffected because it has no such exposure. Both hypotheses are now on
record; current data cannot distinguish them.

**Practical consequence**: the 10%/7.05% headline figures are both
confirmed unchanged (neither was ever mnemonic/domain-defined, and the
lamp figure specifically holds up under the tightened rule too), but
they are not the same baseline. Dataset B's eventual post-bootstrap
fan/valve result should be compared against **0%** -- the tightened
fan/valve rate confirmed here -- not the raw 7.05%, and not the 10%
lamp figure, which is a different domain currently showing a
different, more favorable pattern. Comparing against either without
applying the same tightened definition on both sides would not be a
valid comparison.

## 2026-08-24 — Standalone finding: lamp shows a persistent, structural Dark/Lit asymmetry, not a decode-time artifact

Investigating why Dataset A' candidates (see prior entry) were 100%
"Dark"-answer surfaced a much larger finding than a dataset-composition
problem. Full evidence chain, three independent runs, all agreeing:

| source | n (Dark-truth / Lit-truth) | Dark-truth verified rate | Lit-truth verified rate |
|---|---|---|---|
| original 100-example lamp baseline (source of `03`'s 10% figure) | 50 / 50 | 10/50 = 20.0% | **0/50 = 0.0%** |
| fresh greedy diagnostic (n=2000, checkpoint 130, `do_sample=False`) | 1011 / 989 | 124/1011 = 12.3% | **0/989 = 0.0%** |
| fresh sampled diagnostic (n=2000, checkpoint 130, `do_sample=True`, `temperature=0.8`, `top_p=0.95` -- matching the GRPO rollout precedent, the only "genuine sampling" reference point in this project) | 1011 / 989 | 94/1011 = 9.3% | **0/989 = 0.0%** |

**Across all three runs and 4,050 total attempts, checkpoint 130 has
never once produced a verified undeclared-invention success when the
ground-truth final answer is "Lit."** Sampling was run specifically to
test whether this is a greedy-decoding artifact (a close second-place
logit that temperature exploration could surface) -- it is not. The
raw `<answer>` tag distribution barely moved between greedy and sampled
(regardless of ground truth, the model says "dark" ~82-87% of the time
in both settings) and the Lit-truth verified rate stayed at exactly
0.00% in both. This rules out "sampling just needs to get lucky" as an
explanation.

Decomposing *where* Lit-truth attempts fail (greedy batch, n=989):
81.0% have genuinely broken step-by-step tracking (`global_consistent=
False`); 17.9% track *perfectly* (a consistent, correct invented-code
mapping used correctly across every step) but then the decode-back line
independently contradicts that same mapping, claiming "represents Dark"
when the model's own trace said otherwise (163/177 of these specifically
claim Dark); 1.1% are malformed. The clean 17.9% sub-bucket is a pure
decode-time bias signature in isolation. The dominant 81% bucket is most
parsimoniously explained as the same underlying bias bleeding into the
step-by-step reasoning itself rather than staying confined to the final
line, though this is not proven at the per-token level -- flagged as the
best-supported interpretation, not a certainty.

**This is not lamp-only in kind, but is severe specifically for lamp in
degree.** The same check (verified rate split by ground-truth answer)
was run against fan and valve's full 8,160-completion pool:

- fan: Stopped-truth 7.30% vs Running-truth 5.69% (~1.3x)
- valve: Closed-truth 9.36% vs Open-truth 5.83% (~1.6x)

Both real, moderate asymmetries -- worth noting, not disqualifying,
since both directions retain substantial nonzero success in both
domains. Nothing close to lamp's complete 0%-vs-20% wall. Also checked
against the original direct-RL Coin Flip trajectory (`step50_to500_
full_rollouts.json`, 8,280 rollouts with per-example ground truth and
task-correctness): Tails-truth 69.4% vs Heads-truth 66.1% (~1.05x) --
essentially balanced, not a meaningful finding on its own.

**Conclusion: this is not a case of "our measurement approach has been
missing asymmetries throughout this project."** It is a graduated
pattern -- negligible in Coin Flip, real-but-moderate in fan/valve,
severe-and-total in lamp specifically. Lamp's asymmetry is real,
confirmed structural (survives both decoding strategies), and was
already present, undocumented, in the original 10% baseline this entire
project has cited since `03`.

**Retroactive corrections applied**: every place in `experiments/03_
demonstration_seeding_multi_domain/README.md`, `results/03_
demonstration_seeding_multi_domain.json`, `experiments/05_
self_bootstrapping/README.md`, and `results/05_self_bootstrapping.json`
that cited the lamp 10%/49% figures as an unqualified signal now
carries an explicit caveat pointing to this entry, not a silent
footnote.

**Dataset A' formally closed out as not viable.** This is now
well-supported, not a judgment call: zero verified Lit-truth successes
across 4,050 attempts (original baseline + 2 fresh diagnostics, 2
decoding strategies), confirmed persistent rather than a single-batch
fluke. Any lamp-derived training set would necessarily be 100%
Dark-answer by construction, teaching the same shortcut regardless of
how much additional lamp generation volume is spent chasing it.
Proceeding with Dataset B (05_self_bootstrapping) as the sole SFT
training path.

**Open, not resolved here**: whether the Dark/Lit asymmetry was already
diluted into `03`'s reported 86% *declared*-mapping lamp control
accuracy (used at the time to rule out "lamp is an inherently harder
domain" as an explanation for the 49%/10% undeclared collapse) is not
checked in this entry -- that control's aggregate accuracy was never
broken out by ground-truth answer direction, so it cannot yet be ruled
out that a milder version of the same asymmetry was present there too,
just absorbed into the aggregate figure. Flagged for a future check, not
answered here.

**Resolved (2026-08-24)**: checked directly, not left open. Recovered
per-example ground truth for `03`'s 86% declared-mapping lamp control by
regenerating the deterministic lamp eval set (same seed 20260812,
`example_id` join) against the saved `step130_eval_progress.json`
(pulled from Drive: `multidomain-stage3-v1/step130_eval_progress.json`),
sanity-checked to reproduce 86/100 exactly before trusting the split.
Result: Dark-truth 84.0%, Lit-truth 88.0% -- essentially balanced,
slightly favoring Lit. The declared-mapping control does **not** carry
the undeclared test's severe asymmetry; `03`'s "domain difficulty ruled
out" conclusion stands as originally written. The bias is specific to
the undeclared/self-invention condition -- inventing a code *and*
committing to a final answer with no externally-anchored mapping -- not
a general lamp-domain answer-generation bias. (See
`experiments/03_demonstration_seeding_multi_domain/aws_runs/declared_lamp_answer_direction_check/`
for the script and pulled evidence.)

## 2026-08-25 — Dataset B SFT training complete: rate increase real, composition mixed, not a clean win

Dry run (8 steps, `DRY_RUN=1`) passed cleanly: stable training
(`grad_norm` 0.36-0.49, loss 0.07-0.10), no declared-accuracy regression,
and its one Dark-truth data point (10 verified successes) was **100%
genuinely clean** -- no leakage at all. On that basis, proceeded to the
full 150-step run (same script, `DRY_RUN=0`), continuing checkpoint
130's adapter on Dataset B (248 examples, 124 fan + 124 valve, 100%
mnemonic/domain-leaking at the source -- see the standalone finding
above). Milestone evaluation every 30 steps re-ran the full declared
regression check (300 examples, fan/valve/lamp) plus the undeclared
lamp primary metric (100 examples), both split by ground-truth answer
direction by default, plus a new mnemonic/domain-leak composition check
specifically on Dark-truth verified successes at every milestone --
added mid-run, at the point where a promising-looking rate increase
prompted the question of whether it was genuine.

**Full trajectory (not just the endpoint) -- the step-90 dip is part of
the honest record, not erased by the later recovery:**

| step | Dark rate | clean | leaking | n | clean % |
|---|---|---|---|---|---|
| dry-run (8) | 20.0% | 10 | 0 | 10 | 100.0% |
| 30 | 30.0% | 12 | 3 | 15 | 80.0% |
| 60 | 32.0% | 13 | 3 | 16 | 81.3% |
| 90 | 34.0% | 11 | 6 | 17 | 64.7% |
| 120 | 38.0% | 13 | 6 | 19 | 68.4% |
| 150 | 40.0% | 14 | 6 | 20 | 70.0% |

A reader looking only at checkpoint 130's baseline (20%) versus step 150
(40%) could mistake this for a clean, monotonic doubling. It is not: the
clean count *dropped* at step 90 (13 to 11) before recovering to a
run-high of 14 by step 150, and the clean-fraction never returned to the
dry run's 100% or even to steps 30/60's ~80% -- it settled at 64.7-70.0%
for the back half of the run. The run finished normally
(`max_steps_reached_needs_review`, no early-success or early-failure
trigger; declared regression check never approached its 15pt drop
limit throughout). Lit-truth stayed at 0-2% the entire run, as expected
-- a known, pre-existing, unrelated gap this training has no mechanism
to touch, correctly not treated as a failure signal for this run.

**The finding, reported as three parts side by side, not collapsed into
one number:**

1. **Genuine invention (clean count) increased**: ~10 to 14 in absolute
   terms (+40%) -- a real, if modest, gain.
2. **Shortcut/leaking invention increased more**: ~0 to 6 in absolute
   terms -- a larger absolute gain, and the majority driver of the
   headline rate increase.
3. **The ratio of genuine-to-total success declined** (100% to 70%)
   even though the *absolute* count of genuine successes rose. Both
   trends are real; neither erases the other.

**Methodological implication, connecting back to earlier in this
session**: reinforcing a mixed-quality signal (Dataset B, 100%
mnemonic/domain-leaking at the source) appears to amplify *both* the
genuine behavior already buried in the model's prior *and* the dominant
shortcut pattern, roughly proportionally to how easy each is for the
model to reproduce. This is consistent with genuine arbitrary invention
being intrinsically harder to reinforce than the shortcut -- a second,
independent line of evidence for the tracking-vs-concealment tension
hypothesis raised earlier in this session (mnemonic/domain-adjacent
codes may function as an implicit tracking aid; genuinely opaque codes
may not benefit from that aid, making them harder to learn from even
when explicitly targeted by SFT). Still not proven, but now supported
by two independent observations rather than one.

An operational note from mid-run: the EC2 instance's own AWS SSO session
(separate from the local session used to launch it) expired partway
through, silently breaking the background S3 checkpoint-backup loop for
several hours with no visible error. Caught while confirming checkpoint
90 specifically was durably preserved (per an explicit request to check,
given this project's history of losing intermediate state) -- all five
milestone checkpoints (30/60/90/120/150) are confirmed present in S3
with real adapter files, not empty placeholders, after a manual
catch-up sync. Worth a standing habit: verify backup freshness
directly, not just that the loop process is still alive, on any run
that crosses a long real-time gap.

Full evidence: `experiments/05_self_bootstrapping/aws_runs/dataset_b_sft_v1/`
(dry run and full run scripts, logs, and milestone evaluation JSON).

## 2026-08-25 — Stage 4 merge onto Coin Flip: zero genuine invention on the real target task, both directions

Assembled the merge every prior stage had been pointing toward but never
built: checkpoint 500 (direct-RL Coin Flip, stability-fixed in `04`) as
the base, with checkpoint 150 (the Dataset-B-bootstrapped descendant of
the multi-domain seeded checkpoint 130) attached on top as a LoRA
adapter. Base choice was not an either/or -- checkpoint 130 has never
seen a Coin Flip prompt; checkpoint 500 is the only artifact actually
RL-adapted to the real target task, and `01`/`04` both already named it
as the thing any Stage 4 merge builds on.

Ran the same undeclared-invention diagnostic used throughout `03`/`05`
on Coin Flip itself for the first time: 6,000 sampled completions
(temperature 0.8/top_p 0.95, matching checkpoint 500's own GRPO rollout
distribution), classified with the fixed domain-leak-aware mnemonic
classifier extended for Coin Flip's own domain words ("coin", "flip"),
split by ground-truth answer direction (Heads-truth vs. Tails-truth)
from the first pass, per the standing lesson from the lamp
investigation.

**Result: 0/6,000 verified non-literal encodings, 0.0% in both the
Heads-truth (n=2,987) and Tails-truth (n=3,013) splits.** No
asymmetry to report -- both directions are flatly zero, which is itself
the finding: there's no invention present at either direction for an
asymmetry to appear in. Sanity checks (77.4% final-answer accuracy,
99.8% well-formed output, and confirmation that the handful of
non-literal token attempts that did occur -- 5 distinct tokens, 37
occurrences -- were all in rows that never reached a consistent global
code, not a hidden invented code the classifier missed) rule out a
broken merge or a classifier miss as confounds. This is a clean
negative result, not an inconclusive one, and it directly extends `01`'s
original zero-invention finding under pure RL: merging in the
seeded-invention capability did not transfer to the actual target task.

This also closes the second half of the question that motivated this
run as moot, not unanswered: whether bootstrapping on Coin Flip's own
genuine successes would reproduce Dataset B's "amplifies both good and
bad" pattern. There is no genuine-success pool on Coin Flip to
bootstrap from at this scale -- the mechanism from `05` has no
precondition to act on here.

One process note: the first generation attempt required 6,000 *unique*
prompt texts, but Coin Flip's prompt text depends only on (starting
state, operation sequence) -- for `n_flips` in [2,6] there are only 248
distinct possible prompts, so the uniqueness constraint made the
generation loop spin forever with zero visible progress. Caught via a
`py-spy` stack dump after ~17 minutes (the stuck frame was
`random.choice` inside the prompt-building loop, not model loading as
first suspected) -- fixed by dropping the uniqueness constraint and
switching to sampled decoding, where repeats of the same underlying
scenario are independent, informative draws rather than wasted greedy
repeats.

Full evidence: `experiments/06_stage4_coinflip_merge/aws_runs/stage4_coinflip_undeclared_diagnostic_v1/`
(merge + generation script, classifier, raw completions, classified
results, summary).

## 2026-08-27 — GPU teardown silently failed after the Step 14 full run: a
compound shell command's trailing `echo` masked a mid-sequence SSO token
expiry, instance kept running undetected. Fixed with a verified script, not
just more careful commands.

Every GPU run in this project up to and including Step 14's full 150-step
run was stopped with ad hoc shell commands issued directly in the moment,
never a committed, reusable script. After that run completed, the stop
sequence was a single background-backgrounded command of the shape:

```
aws ec2 stop-instances ...
aws ec2 wait instance-stopped ...
echo "STOPPED"
```

The AWS SSO session token expired partway through this multi-hour session.
Both the `stop-instances` and `wait` calls failed with an SSO token error,
but in a `;`-separated shell sequence the overall exit code is the LAST
command's -- and `echo "STOPPED"` always exits 0. The background-task
runner reported the command "completed successfully." The instance (a
g5.2xlarge fallback, since g5.xlarge had no capacity at launch) kept
running, undetected, until an unrelated later state check happened to show
`State.Name: running` with a `LaunchTime` matching the original launch, not
a restart -- proving it had never actually stopped, not that it had
stopped and been restarted.

No data or run integrity was affected -- this is purely a cost/safety gap
in the teardown tooling, unrelated to Step 14's own experimental result
(logged separately in `experiments/09_direct_indomain_synthetic_bridge/`).
It was caught and the instance was stopped properly (each command's exit
code checked explicitly this time) before any further work proceeded.

**Fix, not a one-off patch**: `scripts/gpu_teardown.py`, a single reusable,
tested procedure that replaces every future ad hoc stop/wait/echo
invocation. Structural choices directly answering how this failure mode
happened:
- Every AWS call's exit code is checked in Python control flow -- no shell
  `;`/`&&` chaining, so no later command's exit code can ever paper over an
  earlier one's failure.
- Credentials are checked before the stop call AND implicitly re-checked at
  every later call (a token can expire mid-run, not just at the start);
  any credential failure aborts immediately with "CREDENTIALS EXPIRED OR
  INVALID -- TEARDOWN NOT CONFIRMED -- MANUAL VERIFICATION REQUIRED" and a
  non-zero exit, never silently falling through.
- Teardown is only ever declared complete after a FRESH `describe-instances`
  call issued as the final action, whose `State.Name` is read and checked
  to literally equal `"stopped"` -- this is mandatory, not conditional on
  whether the stop/wait calls "looked like" they succeeded.
- If the instance isn't stopped within the timeout, the stop call is
  retried once explicitly, re-verified, and only then does the script fail
  loudly (distinct exit codes: 1 = credentials, 2 = stop call rejected,
  3 = never reached stopped, 4 = stopped but a cleanup step -- instance-type
  restore or SSH-rule revoke -- failed).

Tested against real failure paths, not just the happy path
(`scripts/test_gpu_teardown.py`): an invalid/nonexistent AWS profile (exit
1, confirmed no false success banner), a nonexistent instance id with valid
credentials (exit 2), and a cleanup step against a rule that doesn't exist
(exit 4, confirming the instance-stopped state is still reported correctly
even when cleanup fails). Writing these tests caught a real bug in the
script itself before it was ever used for a live teardown: the
`describe-instances` JMESPath query's projection flattens to a list of
plain dicts, not a list of per-reservation lists, and the first draft
indexed it as if it were still nested (`rows[0][0]`), crashing with
`KeyError: 0` on every real call. A fourth test runs the real script,
read-only in effect, against this project's actual (already-stopped)
instance to confirm the happy path also genuinely exits 0 via a real fresh
verification, not just against synthetic failures.
