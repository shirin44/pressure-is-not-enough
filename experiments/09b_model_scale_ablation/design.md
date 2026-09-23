# Stage 9b — Model-Scale Ablation

Stage 9 (Steps 0-14b, `experiments/09_direct_indomain_synthetic_bridge/`) tested three
structurally different mechanisms for getting Qwen2.5-3B-Instruct to adopt a demonstrated
non-literal code on the Coin Flip task: a soft auxiliary CE loss (Steps 10-13b), a hard
reward gate zeroing literal/vacuous reward on bank scenarios (Step 14), and a guaranteed
1-of-8 rollout-group injection giving GRPO a real comparison point on nearly every bank
round (Step 14b). All three produced zero non-literal completions. A completion-level
qualitative pass on Step 14b's data (2026-08-27 addendum, `step14b_design.md`) found no
detectable partial trace either -- no confidence-margin dip, no phrasing drift, no
code-leak language -- after tracing an apparent outlier to an unrelated malformation
confound and ruling it out.

One variable was never varied across any of Steps 0-14b: model size. Stage 9b tests it,
using its own top-level directory (not folded into Stage 9's step numbering) because it
changes the model itself, not just the mechanism applied to a fixed model.

**This is an exploratory pass, not a confirmatory one.** It changes two things
simultaneously: model size (3B -> 7B) and injection strength (1-of-8 -> 2-of-8, or 3-of-8
as a same-pass fallback). If something shifts, this run alone cannot say whether size,
injection strength, or their interaction is responsible -- a follow-up isolating each
variable would be needed before drawing that conclusion. This is stated here explicitly
per the design brief and will be restated in the results write-up regardless of outcome.

## 1. Model

`Qwen/Qwen2.5-7B-Instruct` -- same tokenizer and instruction-tuning family as the 3B work,
so architecture/training lineage stays constant and size is isolated as cleanly as this
project's tooling allows.

## 2. Foundation decision: zero-shot check before committing to a Step-0 equivalent

Before running any 7B-specific training, we run a quick zero-shot accuracy check of the
untouched 7B-Instruct base against the *same* clean-21 length-5 eval set used throughout
Stage 9 (`build_clean_length5_train_eval_split(seed=20260831, n_eval=21)`, SHA256
`947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e` -- re-verified against
the identical hardcoded seed the 3B scripts use, not a fresh draw with the same method,
since this must be the same 21 scenarios, not merely an equivalent one).

Reference bar: the 3B foundation used throughout Stage 9 is Step-0 milestone 12, selected
at **68.69% macro accuracy** on this same eval set
(`step0_stopping_criterion.md`). Decision rule, per the design brief:
- 7B zero-shot in the **68-90%** range (comparable to or better than the 3B foundation):
  skip a dedicated Step-0-equivalent run, inject directly onto the untouched 7B base.
- 7B zero-shot **notably worse** than 68.69%: STOP and report before proceeding. Building
  a foundation first is a bigger scope decision to make deliberately, not by default.

Result of this check, and the decision it produced, is reported below once run.

## 3. Injection mechanism

Reuses Step 14b's design unchanged in structure: on the 16 gated bank scenarios, some of
the GROUP_SIZE=8 rollouts in the comparison group are replaced with the verified-correct
Nib/Nomo trajectory before reward/logp computation. Only the injected *count* changes:
**2-of-8** first. If 2-of-8 still shows zero signal after the 50-step run and there's
appetite to push further within this same exploratory pass, **3-of-8** is the documented
fallback -- not run by default, only on an explicit decision after seeing the 2-of-8
result.

`step14b_injection.py`'s `INJECTIONS_PER_GROUP` constant controls this; the 7B scripts
override it to 2 (or 3) rather than editing the shared module, so Stage 9's own Step 14b
scripts and evidence are untouched.

## 4. Recalibration from the 7B model's own measurements

Two constants in Step 14b's design were calibrated from measured data on the 3B model and
must **not** be reused as-is, since gradient magnitude and per-token KL scale with model
size:

- **KL-clamp bound (`D_MAX` / `CLAMP_VALUE`)**: the 3B run derived this from a *separate,
  already-completed* 500-step per-token-KL instrumentation run on the 3B model (median +
  1.5*IQR mild-outlier fence over per-token KL pooled across steps whose own logged
  grad_norm/KL stayed below half the breaker thresholds, times a 1.5x clamp multiplier --
  full derivation and methodology history in `step14b_bridge_reward_gate_injection_dryrun.py`
  lines ~78-168 and `step14_design.md`). No equivalent multi-hundred-step 7B instrumentation
  run exists, and running one purely for calibration is out of scope for an exploratory
  pass. Instead we measure grad_norm and per-token KL from the 7B model's own **first real
  GRPO group(s)**, run with the clamp effectively disabled (D_MAX = +inf, pass-through),
  before the real dry run starts. This is the same statistical methodology (exclude any
  step whose own grad_norm/KL already comes within 2x of the breaker thresholds, mild-
  outlier fence over what's left, apply the clamp multiplier) applied to a much smaller
  measurement sample than the original's few hundred clean steps. **This sample-size
  reduction is a disclosed limitation of doing this within a short exploratory run, not a
  silent substitution** -- flagged here and in the calibration script's own output.
- **Breaker thresholds (`grad_norm>=50`, `kl>=5`)**: kept unchanged from the 3B run per the
  design brief's instruction to confirm rather than blindly reuse. These are *absolute*
  circuit breakers (values, not model-size-relative multipliers) meant to catch numerical
  blowup; nothing about them is 3B-specific in derivation (they were not fit to the 3B
  model's typical gradient scale, they're stability tripwires) so there is no a priori
  reason 7B needs different absolute values. The measured-first-group grad_norm/KL values
  (same measurement pass as the clamp calibration above) are checked against these
  thresholds before trusting them for this run; if the 7B model's healthy first-group
  values already sit close to these thresholds, that is flagged explicitly rather than
  proceeding.

## 5. LoRA config: unchanged from the 3B setup

`r=8, lora_alpha=16, target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_dropout=.05,
bias='none'` -- identical to Step 14b, not scaled up for the larger base model, so adapter
capacity is not a third uncontrolled variable alongside model size and injection strength.

## 6. Reused unchanged

Reward function (`reward_v3.py`), taxonomy (`taxonomy.py`), the reward-gate module
(`step14_reward_gate.py`), the dynamic-sampling safety net, quantization (8-bit
`BitsAndBytesConfig(load_in_8bit=True)`), GROUP_SIZE=8, MAX_NEW_TOKENS=256,
`per_device_train_batch_size=1` / `gradient_accumulation_steps=GROUP_SIZE` /
`num_iterations=1` (so the GRPO ratio-is-unconditionally-one invariant from Step 14b's
design still holds unchanged -- re-verified against the 7B trainer's live args before
launch, same as Step 14b). Bank-16 scenario set reused as-is (`CODED_TRAJECTORIES` /
`ACTIVE_TRAJECTORIES`, scenario-matching logic in `step14_reward_gate.py`) since the task
and scenario definitions are model-independent; only the model generating completions
against them changes.

## 7. Infrastructure sizing (analysis before launch, no GPU spent to "discover" an OOM)

Existing Stage 9 GPU work uses `g5.xlarge` (1x A10G, 24GB VRAM, 4 vCPU, 16GiB RAM),
falling back to `g5.2xlarge` (same A10G/24GB, 8 vCPU, 32GiB RAM) only on
`InsufficientInstanceCapacity`.

**GPU VRAM**: at `load_in_8bit=True`, Qwen2.5-7B-Instruct weights are ~7GB (vs. ~3GB for
the 3B). LoRA adapter (rank 8, four projection types) adds a few tens of MB regardless of
base size. `gradient_checkpointing=True` and `per_device_train_batch_size=1` keep
activation memory bounded to a single sequence's forward/backward at a time. KV-cache for
GROUP_SIZE=8 generation at MAX_NEW_TOKENS=256 with Qwen2.5-7B's GQA (few KV heads) is on
the order of hundreds of MB, not GB. Estimated peak: roughly 12-16GB against a 24GB budget
-- comfortable, not tight. **No GPU-family change needed**; a real VRAM upgrade would mean
leaving the A10G family entirely (e.g. an A100/H100 instance), which this analysis does not
support needing.

**CPU RAM**: this is the more relevant sizing question for a 7B model. `g5.xlarge`'s 16GiB
system RAM is comfortable for the 3B's ~6GB bf16-equivalent checkpoint but is tighter
headroom for the 7B's ~14GB bf16 checkpoint during sharded loading (before/while
bitsandbytes quantizes each shard to 8-bit). **Recommendation: launch this ablation
directly on `g5.2xlarge` (32GiB RAM) rather than starting with `g5.xlarge` and falling
back on capacity errors** -- same GPU either way, so this is a RAM headroom choice for a
bigger model, not a capacity-availability fallback, and is called out as a deliberate
deviation from Stage 9's launch convention for that reason.

This is an analytical estimate, not an empirical one. The launch scripts print
`torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` immediately after
model load and again after the first real GRPO group, so if this estimate is wrong it
shows up immediately and cheaply rather than as a mid-run OOM.

## 8. Step budget

8-step GPU dry run (mandatory) first: confirm no breaker, finite losses/gradients, and
injection firing at the new 2-of-8 rate, before anything longer. Then a **50-step** run
(not 150) -- exploratory, not confirmatory. Any sign of non-literal completions, or a
pattern meaningfully different from the 3B results, is sufficient justification to extend
further as a *follow-up decision*, not something pre-committed here.

## 9. Teardown

`scripts/gpu_teardown.py` for every phase, requiring the "TEARDOWN FULLY CONFIRMED" banner
before considering any phase closed -- same discipline as every GPU task since the
teardown-reliability fix.

## 10. Naming convention

This document and directory (`experiments/09b_model_scale_ablation/`) constitute
**"Stage 9b: Model-Scale Ablation"** -- deliberately a sibling of, not a subdirectory or
continuation of, `09_direct_indomain_synthetic_bridge/`'s own step numbering, so the
project record clearly distinguishes the 3B-based Steps 1-14b from this model-scale
ablation. Scripts in this directory follow their own local naming
(`zeroshot_check_7b.py`, `calibration_7b.py`, `injection_dryrun_7b.py`,
`injection_full_7b.py`) rather than Stage 9's `stepN_...` convention, since there is no
step sequence being extended here.

---

## Progress log

**2026-08-28 -- Infrastructure**: `i-REDACTED` resized from `g5.xlarge` to
`g5.4xlarge` and started. `g5.2xlarge` (the RAM-headroom instance type recommended in
section 7) hit `InsufficientInstanceCapacity` in us-east-1 at launch time;
`g5.4xlarge` (same A10G/24GB GPU, 64GiB RAM -- even more headroom than the `g5.2xlarge`
recommendation, not less) had capacity and was used instead. Documented here as a
capacity-driven substitution, same reasoning as Stage 9's own `g5.xlarge` ->
`g5.2xlarge` fallback precedent, just one step further along the same GPU family.
Confirmed via SSH: NVIDIA A10G 23028MiB, 62GiB RAM, 153GB free disk -- matches
expectation. SSH connectivity to this instance was intermittently flaky (some
connection attempts timed out, immediate retries succeeded); AWS's own
instance-status and system-status checks reported "ok" throughout, so this looks like
transient network flakiness rather than an instance-health problem, and did not block
progress once retried.

**2026-08-28 -- Gate 1 (zero-shot check)**: `zeroshot_check_7b.py` run against the
untouched `Qwen/Qwen2.5-7B-Instruct` base, no LoRA, on the verified-identical clean-21
set (SHA256 matched). Result: **normal_cot_accuracy = 0.7619** (16/21), answer_only
accuracy = 0.524, vacuous_rate = 0.0, literal_rate = 1.0 (all 21 samples classified
`literal` -- clean, expected zero-shot behavior, no degenerate output). This clears the
0.68 decision floor and sits above the 3B's own milestone-12 foundation (0.6869) --
squarely in the brief's stated 68-90% "comparable to or better than" range.
**Decision: SKIP_STEP0_INJECT_DIRECTLY** -- no Step-0-equivalent foundation run for the
7B model; injection proceeds directly onto this untouched base. GPU memory during this
check: 8.71GB after model load, 9.88GB peak during generation, against a 23GB budget --
consistent with (and better than) section 7's analytical estimate; no memory concern
observed.

**2026-08-28 -- Gates 2+3 (calibration + 8-step dry run)**: `injection_dryrun_7b.py`
launched. Structure: Phase 0 runs `CALIBRATION_STEPS=6` real GRPO steps with the KL
clamp disabled (pass-through) and a fresh LoRA init, purely to measure grad_norm/KL on
the 7B model's own first groups; that adapter is then discarded (base 8-bit weights
reused, never re-downloaded) and Phase 1 builds a second, independently fresh LoRA
adapter for the real, reported 8-step dry run with the calibrated clamp enabled and
`INJECTIONS_PER_GROUP=2`. `injection_scaling.py` (new, this directory) generalizes Step
14b's hardcoded 1-of-8 injection selector to N-of-8; verified byte-identical to
`step14b_injection.py`'s own functions at n=1 in `test_injection_scaling.py` (9/9
passed locally, no GPU needed) before being trusted for n=2 on GPU.

**Incident, disclosed rather than silently fixed**: the first launch of
`injection_dryrun_7b.py` (a) hit a real bug caught by its own train/eval
disjointness assertion -- an earlier version called a second, independently-seeded
`build_train_eval_split` for the train pool instead of reusing
`build_clean_length5_train_eval_split`'s own matched `train_rows`, so train/eval
overlap was possible by construction (same class of bug this project's own
`build_train_eval_split` docstring warns about from the length-4 precedent). Fixed
by reusing the coordinated split directly; 0 GPU time lost (caught before model
load). (b) After that fix and relaunch, the run crashed within its first minute on
`ValueError: Seed must be between 0 and 2**32 - 1` -- `CALIBRATION_SEED =
20260828111` (11 digits) exceeds numpy's valid seed range. Fixed to `20260829`.
No GPU training time was lost to this bug either (crashed before Phase 0's first
real step). (c) Separately and coincidentally, EC2 instance `i-REDACTED`
stopped itself at 12:57:59 GMT while these two bugs were being diagnosed --
confirmed via the guest's own journal as a genuine ACPI power-off signal
(`Power key pressed short` -> `system will power off now`), the standard signature
of an EC2-API-level stop request delivered to a running instance, not an in-guest
shutdown command. Investigated before restarting: ruled out unattended-upgrades
auto-reboot (explicitly disabled in this AMI's `/etc/apt/apt.conf.d/50unattended-
upgrades`) and apt's scheduled timers (both scheduled for later that day/the next
day, neither matches the timestamp). Attempted to identify the caller via
CloudTrail; got an explicit IAM deny on `cloudtrail:LookupEvents` for this account,
so the source could not be confirmed. Restarted (capacity forced a `g5.4xlarge` ->
`g5.2xlarge` instance-type change, same A10G GPU); flagged to the user as an open
question rather than assumed benign, since it has real implications for the
50-step run's required uptime.

Results pending after this second relaunch.

**2026-08-28/29 -- a fourth bug, then a genuine methodology finding (not a bug)**: the
relaunch above hit one more trivial bug (`del calib_trainer, calib_model, calib_calls,
trainer_ref[0]` deletes the LIST ELEMENT `trainer_ref[0]`, shrinking the list to empty,
so the next line `trainer_ref[0] = None` raised `IndexError`; fixed to `del
calib_trainer, calib_model, calib_calls` followed by a plain `trainer_ref[0] = None`).
Cheap to redo (calibration is ~3-4 minutes).

After that fix, Phase 0 completed cleanly and calibration itself surfaced a real,
disclosed finding, not a code bug: the derived clamp came out **degenerate, D_MAX=0.0**
-- next to **D_MAX=0.020** from an otherwise-identical calibration run on a different
random seed a few launches earlier. A 100x swing from an ostensibly deterministic
procedure. Root cause: with only 3-4 "clean" calibration steps surviving the
borderline-step filter, and a FRESH, barely-diverged LoRA adapter at TARGET_LR=1e-6 (the
policy has had almost no opportunity to diverge from the reference), the large majority
of per-token KL values in the pool are EXACTLY zero (policy == reference before real
weight drift). Pooling zeros alongside the genuinely-informative nonzero tokens can push
the median (and even Q3) to exactly 0, collapsing `HEALTHY_BASELINE`/`CLAMP_VALUE` to 0.
This is a failure mode the 3B calibration's own methodology never had to handle, because
it pooled from a mature, already-completed 500-step run where enough real divergence had
accumulated for the pool's median/IQR to never be near-degenerate.

**Fix, decided and verified before trusting it (not silently patched)**: pool only
strictly-positive per-token KL values (`nonzero_only=True` in the new shared
`kl_calibration.py` module, used by both `injection_dryrun_7b.py`'s live Phase 0 and a
new standalone `calibration_seed_stability_check.py`). This addresses the actual
mechanism -- exact-zero tokens are uninformative about the SPREAD among tokens that did
diverge at all, not a representative "typical value" to center a clamp on -- rather than
patching around the symptom.

**Seed-stability verification (`calibration_seed_stability_check.py`, 4 seeds, same 7B
model, same 6-step calibration procedure)**, run BEFORE trusting the fix, per instruction:

| seed | nonzero D_MAX | all-values D_MAX |
|---|---|---|
| 20260829 | 0.050781 | 0.000206 |
| 30000001 | 0.053391 | 0.000080 |
| 30000002 | 0.055413 | 0.000109 |
| 30000003 | 0.058358 | 0.000097 |

Nonzero-only: min=0.050781, max=0.058358, **max/min ratio = 1.15**, same order of
magnitude across all 4 seeds, **0/4 degenerate**. All-values (old method, same trials,
for direct comparison): every single one lands in the 8e-05 to 2e-04 range --
consistently near-degenerate even on the 2 trials that didn't hit exactly 0. **Verdict:
stable.** The fix addresses the actual failure mode, not just the one degenerate
instance already observed; adopted as the calibration method going forward for this
model-scale context.

**Floor backstop (`kl_calibration.floor_d_max`), layered underneath, not a substitute for
real calibration**: if a future run's nonzero-pool calibration still comes out degenerate
(not observed in 0/4 seed trials, but not assumed impossible), `D_MAX` is floored at
**0.050781** -- the smallest nonzero D_MAX actually observed across the 4-seed check
above, a measured, model-specific reference point rather than an arbitrary round number.
The 3B model's own calibrated D_MAX (~0.5047, `step14b_design.md`) is a secondary,
same-order-of-magnitude sanity-check reference, not the floor itself (7B's measured
values are meaningfully tighter than 3B's, consistent with 7B's much lower observed
grad_norm/KL during calibration -- see the breaker sanity check above). The floor prints
a loud, explicit `DEGENERATE CALIBRATION RESULT` message when it engages (never silent) --
verified in `test_kl_calibration.py`'s `test_end_to_end_*` regression tests, which
confirm the pre-fix path correctly REJECTS a degenerate result (`solve_kl_clamp_bound`
raises rather than returning a bogus D_MAX) and the post-fix path (nonzero-only + floor)
produces a stable, sane value; 13/13 tests pass locally, no GPU needed.

`injection_dryrun_7b.py`'s Phase 0 updated to use `kl_calibration.py`'s
`compute_clamp_from_pool(..., nonzero_only=True)` / `solve_kl_clamp_bound` /
`floor_d_max` instead of its own inline median/IQR logic, so the live run and the
verification script share one implementation.

**2026-08-29 -- 8-step dry run: clean, with one further disclosed caveat found while
checking it.** With the fixed calibration deployed, `injection_dryrun_7b.py` completed
end to end: `terminal_step=8`, `hard_stop=None`, `soft_stop=None`,
`survived_to_full_target=True`. Phase 0 calibration produced **D_MAX=0.736196**
(floor backstop not engaged -- `floor_backstop_applied: False`, this is a genuine
measured value). GPU memory stayed comfortable throughout (13.9GB peak of 23GB
available). One accepted group out of 8 drew a bank scenario (in line with the ~37%
per-draw chance and n=8); injection fired exactly once, at exactly 2-of-8 as required
(`injections_per_group_verified: 2`). Milestones 4 and 8: clean accuracy 0.762, bank
accuracy 0.688, all 37 samples classified `literal`, no taxonomy flags -- unsurprising
for 8 steps at a tiny, still-warming-up learning rate; no adoption signal expected this
early, consistent with every prior stage of this project.

**Caveat found and flagged, not glossed over**: this run's calibration used
`CALIBRATION_SEED=20260829` -- the SAME seed value as the first trial in the 4-seed
stability check above, which produced D_MAX=0.050781 for that seed. This live run
produced **D_MAX=0.736196 for the "same" seed -- roughly 14.5x higher**, and notably
closer to the 3B's own calibrated D_MAX (~0.5047) than to any of the 4-seed check's
values (0.0508-0.0584). `torch.manual_seed`/`random.seed` are reset immediately before
each calibration pass in both scripts, so this is not an obvious seeding-order bug;
the most likely explanation is ordinary GPU/CUDA run-to-run nondeterminism (unseeded
kernel-level parallelism, e.g. non-deterministic reduction order in backward passes)
that `manual_seed` does not fully eliminate -- meaning the 4-seed stability check,
run entirely within one process, may have understated true cross-process variance in
the calibrated MAGNITUDE even though it correctly validated that nonzero-only pooling
avoids the degenerate-collapse-to-zero failure mode (which it did here too: 0.736196
is a perfectly sane, comfortably-positive value, nowhere near degenerate). Net
assessment: the core fix (avoiding zero/near-zero collapse) is holding up across a 5th,
independent data point; the NARROWER claim ("D_MAX lands in a tight, predictable
numeric range") is weaker than the seed-stability check by itself suggested, and is
flagged here rather than silently treated as fully settled.

**2026-08-29 -- D_MAX sensitivity check: does the 14.5x swing actually matter?**
(`dmax_sensitivity_check.py`) Two 8-step trials, identical seed/model/injection rate,
differing ONLY in D_MAX -- fixed to the two already-observed endpoints (0.050781 "low",
0.736196 "high") rather than re-calibrated. Scenario draws verified identical (8/8
matching prompt hashes -- same seed reliably controls dataset ordering even though it
does not reliably control the calibrated KL value itself).

| metric | low (D_MAX=0.051) | high (D_MAX=0.736) |
|---|---|---|
| clamp engagement rate | 21.0% | 21.8% |
| grad_norm max / mean | 40.5 / 20.8 | 34.3 / 25.9 |
| **KL max / mean** | **0.00063 / 0.00028** | **0.119 / 0.061** |

Engagement rate and grad_norm are comparable between the two. The logged KL metric is
**~200x different**. Mechanistic explanation, not just a correlation: the clamp bounds
each token's contribution at `CLAMP_VALUE = exp(D_MAX) - D_MAX - 1`, which is ~0.0013 at
D_MAX=0.051 vs. ~0.352 at D_MAX=0.736 -- a ~270x difference in the ceiling itself. Since
~21% of tokens hit that ceiling under EITHER setting, the aggregate KL metric (and
therefore the `beta * KL` term in GRPO's loss, `beta=0.04`) scales directly with wherever
that ceiling happens to land. **Verdict: NOT measured-but-inconsequential.** This is a
real, non-trivial effect on the loss function's regularization strength.

**Decision (user-approved, 2026-08-29): fix D_MAX to a single deliberately-chosen
constant for the entire 7B ablation, replacing per-run calibration.** Per-run
calibration is abandoned not because the nonzero-only-pooling fix was wrong (it
correctly eliminated the zero-collapse failure mode, confirmed across 5 independent
data points now) but because the underlying MAGNITUDE is too unreliable across process
boundaries to trust a fresh per-launch draw, and that magnitude has now been shown to
matter mechanistically. Averaging multiple in-process calibration draws (the alternative
option) was rejected: the 4-seed in-process check understated the true swing by itself
(0.0508-0.0584 in-process vs. 0.736196 from one independent-process run with a "nominally
matching" seed), so averaging a practical number of in-process draws would not
reliably converge on a trustworthy value without expensive, repeated independent-process
sampling.

**FIXED_D_MAX = 0.505**, hardcoded in `injection_dryrun_7b.py` (Phase 0 / calibration
removed entirely from the automatic launch path -- `kl_calibration.py` and
`calibration_seed_stability_check.py` remain in this directory for reference/future use,
per instruction, just no longer exercised automatically). This is the 3B model's OWN
calibrated D_MAX (0.504718, `step14b_bridge_reward_gate_injection_dryrun.py` /
`step14_design.md`), rounded to 0.505.

**Precise framing, stated explicitly rather than glossed over as "reusing a validated
number"**: **this value was NEVER MEASURED on the 7B model.** It is borrowed from a
DIFFERENT model (the 3B) specifically because per-run measurement on the 7B was found
unreliable across process boundaries (the sensitivity-check finding above) -- reliably
measuring a 7B-specific value would require repeated, expensive independent-*process*
sampling, not just multiple seeds within one process. **This is a deliberate, disclosed
exception to this project's "always measure, never guess" practice, not a continuation
of it.** Every other constant used in this ablation -- the zero-shot foundation check
result (0.7619, with its own caveat below), the breaker-threshold sanity check (now
grounded in real Phase-1-equivalent 7B telemetry rather than calibration telemetry:
grad_norm topping out ~34-40 vs. the 50.0 breaker, kl topping out ~0.12 vs. the 5.0
breaker, across the completed dry run and both sensitivity-check trials), and
`ENTROPY_CLAMP_VALUE=1.2189` -- IS measured on the 7B model itself. D_MAX alone is not,
for the specific, disclosed reason above. `ENTROPY_CLAMP_VALUE` carries its own narrower
caveat: it WAS measured on 7B (during the now-removed-from-automatic-launch Phase 0 of
the 2026-08-29 dry run) but is now frozen and reused rather than re-measured per launch,
and its own cross-process stability was never checked the way D_MAX's was (no
entropy-clamp analogue of `dmax_sensitivity_check.py` was run) -- flagged here rather
than silently assumed stable by analogy to D_MAX's fix.

**Related caveat, same category of concern, flagged per instruction**: the zero-shot
foundation check (`zeroshot_check_7b.py`, 0.7619 for 7B vs. 0.6869 for the 3B's own
milestone-12 foundation -- the result that justified skipping a dedicated Step-0 run for
7B) was ALSO a single-process measurement, taken BEFORE the cross-process
nondeterminism issue was discovered via the D_MAX calibration swing. It has not been
independently reproduced across process boundaries. Not rerun here -- there is no
specific reason to think the 0.7619/0.6869 gap (a 0.095 absolute margin over the 0.68
floor) is close enough to the decision boundary to be at risk from GPU-nondeterminism-
scale noise, unlike D_MAX's near-100x swings -- but a future reader should know this
number carries the same class of unverified-reproducibility caveat as the pre-fix D_MAX
measurements did, not a stronger guarantee.

**Regression tests updated** (`test_fixed_dmax_config.py`, new): confirms
`injection_dryrun_7b.py` itself no longer contains calibration-era code
(`CALIBRATION_STEPS`, `calib_trainer`, `CLEAN_CALIB_STEPS`, the Phase 0 header, the
`kl_calibration` import), that `FIXED_D_MAX = 0.505` and `ENTROPY_CLAMP_VALUE = 1.2189`
are present and correctly wired into `CLAMP_VALUE`, and that the KL-clamp mechanism
self-test (which verifies clamp CORRECTNESS, independent of calibration) still runs
against the fixed value. `kl_calibration.py`'s own function-level tests
(`test_kl_calibration.py`) are unchanged and still pass -- the module remains correct
and available, simply not exercised automatically. 30/30 tests pass across Stage 9b
(8 new + 22 previously existing).

**2026-08-29 -- Final dry run under the fixed configuration: clean.**
`injection_dryrun_7b.py` re-run one more time (`stage9b-7b-injection-dryrun-v2`) with
calibration fully removed from the launch path. Confirmed: `FIXED_D_MAX=0.505` correctly
applied (`CLAMP_VALUE=0.152`, matching `exp(0.505)-0.505-1` exactly), no Phase 0 ran
(noticeably faster than the two prior dry runs), `terminal_step=8`, `hard_stop=None`,
`soft_stop=None`, `survived_to_full_target=True`. Injection fired exactly once (one
bank-scenario group drawn out of 8, as expected by chance) at exactly the required
2-of-8 rate. Clamp engagement rate 28.0% -- same order of magnitude as
`dmax_sensitivity_check.py`'s 21-22%, not a new concern. GPU memory comfortable
throughout. Milestones 4 and 8: clean accuracy 0.762, bank accuracy 0.750 -> 0.688, all
37 samples classified `literal`, no taxonomy flags -- no adoption signal, as expected
for 8 steps at a tiny, still-warming-up learning rate, consistent with every prior
stage of this project and with the earlier (calibration-based) dry run.

**This is the final go/no-go checkpoint before the 50-step real run.** Dry run is
clean under the final, fixed configuration (7B base, no Step-0 foundation, 2-of-8
injection, D_MAX=0.505 fixed and disclosed as borrowed-not-measured, breakers
unchanged and sanity-checked against real telemetry). The 50-step run is NOT launched
in this task -- awaiting explicit go-ahead.

**2026-08-29 -- read-only mechanism-behavior sanity check on this final dry run's
evidence** (no new training; a sanity check on WHETHER the mechanism behaves plausibly,
not a code-adoption check -- none expected at this scale).

*Milestone accuracy vs. zero-shot baseline*: clean-21 (greedy) held exactly at 0.7619
at both milestones 4 and 8 -- identical to the pre-training zero-shot baseline, exactly
as expected from a tiny (still-warming-up-LR) 8-step nudge. Bank-16 (greedy) went
0.750 -> 0.688, both milestones fully `literal` (0 vacuous), consistent with normal
task-difficulty variation on a smaller n=16 sample, not a collapse. No implausible
spike or collapse in either direction.

*Injection reward*: exactly 1 injection event (step 1, the only bank scenario drawn in
this 8-step window), 2-of-8 rows injected as required. Both injected rows scored
**reward=4.15 exactly** (`taxonomy_category=9, correct_globally_consistent_code`) --
bit-identical to the reward the SAME verified trajectory scores in the 3B's own runs,
as expected: reward is a pure function of completion text/taxonomy, not model size.

*Comparative sanity check vs. 3B's Step 12 and Step 14b dry runs*: loss (mean -0.019
vs -0.018 vs -0.019) and per-step KL (max 0.068 vs 0.055 vs 0.059) are essentially
indistinguishable across all three runs -- no scale discrepancy. grad_norm mean is
higher for 7B (24.9) than either 3B run (8.8, 13.2) -- same order of magnitude, well
clear of the 50.0 breaker, plausibly explained by LoRA parameter count scaling with
hidden_size (larger for 7B), not flagged as concerning. Step timing is comparable
(7B mean 45.0s vs 3B's 46-56s range) -- no proportionality surprise.

**One real, quantified discrepancy, flagged rather than glossed over**: rollout-level
task correctness (`r_task` across all 64 training rollouts, temperature=0.8 sampling)
was **48.4% for 7B vs. 73.4% (Step 14b) and 54.7% (Step 12) for 3B** -- notably lower,
not just noise-sized. Traced to source before concluding anything: inspected the
step-4 group, where all 8 samples were wrong -- every completion made the SAME
systematic misreading of the scenario's first instruction ("different from previous"
treated as "unchanged"), then correctly applied flip/same logic for every subsequent
step relative to that wrong baseline. This is coherent, well-formatted, on-task
reasoning that is wrong on one specific instruction-application point, reproduced
consistently across independent samples of the SAME scenario -- not degenerate output,
not random hallucination, not a mechanism failure. Given the clean-21 and bank-16
GREEDY eval accuracy (the more reliable metric) show no degradation at all, and this
gap comes from only 8 physical steps / 64 rollouts (highly sensitive to which few
"hard" scenarios happened to get drawn by chance, not a scenario-matched comparison
against the 3B runs), this is not attributable with confidence to a genuine 7B-specific
problem from this sample alone -- but it is a real number, reported precisely, and
flagged as worth watching in the first ~20-30 steps of the 50-step run specifically
(does the gap persist/worsen with more data, or was it sampling noise from a small n).

**Verdict: GO**, with one explicit watch-item (not a blocker). All three checks in
this task are clean by their own criteria (no collapse/spike, correct injection
reward, no scale discrepancy in loss/KL/grad_norm/timing). The rollout-correctness gap
is real, precisely quantified, and traced to a benign-looking cause (specific hard
scenarios, not degeneracy) rather than swept under the rug -- recommended as an early
monitoring point for the 50-step run rather than grounds to hold and investigate
further before launching.

## Full run (2026-08-29)

User go-ahead received. `injection_full_7b.py` created from `injection_dryrun_7b.py`
and diff-verified before launch -- differs ONLY in `N_STEPS` (8->50), the `dry_run`
flag (`True`->`False`, both occurrences), output-path/run-identifier naming
(`stage9b-7b-injection-full-v{n}`, `stage9b_7b_injection_full.json`), and matching
doc-string/print-label wording for the same change (no config or mechanism edits --
confirmed by `diff`, reproduced above). Launch configuration: Qwen2.5-7B-Instruct base
(no Step-0 foundation), fixed `D_MAX=0.505`, `INJECTIONS_PER_GROUP=2`, same reward
gate/taxonomy/breakers/eval framing as Step 14b, `MILESTONE_EVERY=4` (milestones at
4, 8, ..., 48, plus a final one at 50).

**Operational note**: stdout was fully buffered when redirected to the log file
(Python's default for non-TTY output), so the log appeared stuck on the startup banner
for long stretches even though the run was progressing normally underneath. Caught by
checking the evidence JSON directly (which writes immediately on every step via
`save_event()`, independent of stdout buffering) rather than trusting an apparently-
stalled log -- monitoring was switched to polling the JSON's milestone count for the
remainder of the run.

**Result: completed cleanly, all 50 steps, no hard/soft stop.**

| step | clean acc | bank acc | taxonomy | non-literal |
|---|---|---|---|---|
| 4 | 0.762 | 0.688 | all literal | 0 |
| 8 | 0.762 | 0.688 | all literal | 0 |
| 12 | 0.762 | 0.688 | all literal | 0 |
| 16 | 0.762 | 0.688 | all literal | 0 |
| 20 | 0.810 | 0.688 | all literal | 0 |
| 24 | 0.810 | 0.750 | all literal | 0 |
| 28 | 0.762 | 0.688 | all literal | 0 |
| 32 | 0.762 | 0.750 | all literal | 0 |
| 36 | 0.810 | 0.750 | all literal | 0 |
| 40 | 0.762 | 0.688 | all literal | 0 |
| 44 | 0.762 | 0.750 | all literal | 0 |
| 48 | 0.762 | 0.750 | all literal | 0 |
| 50 (final) | 0.762 | 0.688 | all literal | 0 |

Clean accuracy oscillates between 0.762 (zero-shot baseline) and 0.810, never below
baseline. Bank accuracy oscillates between 0.688 and 0.750. No directional trend
either way across 50 steps -- ordinary sampling noise around a stable operating point,
not degradation or a spike. **Zero taxonomy flags at any milestone; `nonliteral_counts`
empty every time.** `taxonomy_flags: []` in the full event log confirms no non-literal
completion was ever recorded, so the "flag immediately" requirement never had cause to
trigger.

**Watch-item resolution (the dry run's 48.4% vs. 73.4%/54.7% training-rollout
correctness gap)**: did not persist or worsen. Bank accuracy across the full run's 13
milestones stayed within the same 0.688-0.750 band throughout, with no decay -- the
dry run's low figure looks, in retrospect, like exactly the small-sample artifact it
was flagged as a candidate for, not an early sign of a real problem.

**Injection and group-collapse (requirement 2)**: 50/50 accepted groups, 20 bank-
scenario groups (20/50 = 40%, matching the ~37% expected draw rate), only **1/20
(5.0%) needed the forced-fallback path** -- lower than the dry run's early 1/9 (11%)
read, and in the same low-single-digit range as the 3B's own historical near-total
resolution (Step 14b's full run: 0/67). Zero bank groups landed at exact zero reward
variance. 27 injection events total, every single one at exactly 2-of-8
(`{2}` as the only value in the injected-count set across all 27) -- the mechanism
fired correctly and consistently for the entire run, not just the dry run's single
sample.

**Telemetry**: grad_norm 0.69-34.75 (mean 18.2), KL 0.0000-0.0691 (mean 0.038) --
comfortably clear of both breakers (50.0, 5.0) for all 50 steps; all rewards/advantages
finite throughout. KL-clamp engagement rate 33.7% over the full run (vs. the dry run's
28.0% and the sensitivity check's 21-22% -- same rough range, no new concern).
`survived_to_full_target: True`. Best milestone by clean accuracy: step 20 (0.810).

**Headline finding, consistent with every prior stage of this project**: across all
400 training rollouts (50 accepted groups x 8 GROUP_SIZE; 160 of those from the 20
bank-scenario groups, per `bank_scenario_rollout_count` totals) and all 481 milestone-
eval completions (13 milestones x 37: 21 clean + 16 bank), the model's own non-injected
output was `literal` every time -- zero non-literal completions anywhere. 27
guaranteed-correct, correctly-scored comparison points were available to GRPO across
the run (up from Step 14b's 1-of-8 to 2-of-8 here) on a 7B model instead of 3B -- both
deliberately varied at once, so this result alone cannot isolate which variable (if
either) would matter in a genuinely controlled follow-up -- and adoption still did not
occur.

**Evidence, retrieved and hash-verified** into
`experiments/09b_model_scale_ablation/aws_runs/stage9b-7b-injection-full-v1/`: full
event log (SHA-256 `d78a5a497e8dcc01283df69ababc8ff95bd15d47455e114352b7a2eb9628fe12`,
13.8MB), per-token instrumentation (SHA-256
`f1339ff9edf1a9b6e6147914f2b1896c78f94d3c9b65c309517e70f18d02146f`, 4.8MB), and the
full stdout log -- local hashes recomputed after transfer and confirmed byte-identical
to the remote-computed hashes before trusting the copy. All 13 milestone checkpoints
(`milestone-4` through `milestone-50`) remain on the instance's EBS volume (not
copied down -- adapter weights only, evidence JSON/instrumentation is the load-bearing
record).

**Teardown**: `scripts/gpu_teardown.py`, restoring `g5.xlarge` from the `g5.2xlarge`
capacity-driven substitution -- printed "TEARDOWN FULLY CONFIRMED", exit code 0.
Independently re-verified via a fresh, separate `describe-instances` call: `stopped`,
`g5.xlarge`. One credential hiccup mid-teardown (AWS SSO token expired between the
launch phase and teardown, same recurring pattern as earlier in this project) --
paused, had the user re-authenticate, verified via `aws sts get-caller-identity`
before retrying rather than proceeding on a stale session. Clean.

**2026-08-29 addendum -- a gap discovered in Stage 9d, disclosed here since it
concerns this stage's own results.** While building Stage 9d (loading a checkpoint
into a second `get_peft_model()` call on a REUSED base-model object across two
phases in one process), an assertion there caught TRL's GRPOTrainer
(`beta != 0`, PEFT) registering a frozen `"ref"` reference-policy adapter directly
onto the shared base model (`trl/trainer/grpo_trainer.py`, confirmed by reading the
installed source) -- residue that persists across `get_peft_model()` calls sharing
the same base-model object, and was never checked for in this stage's own scripts.
`injection_dryrun_7b.py`'s Phase 0 -> Phase 1 transition, `calibration_seed_stability_
check.py`'s 4 trials, and `dmax_sensitivity_check.py`'s 2 trials all reuse a base
model across repeated `get_peft_model()` calls within a single process and never
asserted adapter composition -- if the same leak occurred there, it went undetected.
Whether it actually affected those results (as opposed to being harmlessly inert
under PEFT's active-adapter isolation) was not verified either way here. The reported
50-step full run (`injection_full_7b.py`, calibration already removed from its
launch path by the time it ran) is NOT implicated -- a single `get_peft_model` call
per process, no reuse. Not re-investigated retroactively in this task; recorded here
as a disclosed, unresolved gap rather than left implicit in Stage 9d's own log alone.

**2026-08-30 -- the disclosed gap, investigated: code-pattern confirmed vulnerable in
all three scripts, but the actual reported 50-step result is not affected. Defensively
fixed anyway.** CPU-only, code-inspection task (no GPU run needed to determine the
pattern; a fresh instance was never started for this).

**Per-script inspection**:

- **`injection_dryrun_7b.py`**: **(b) vulnerable in code, plausibly triggered in the
  script's own historical Phase 0 -> Phase 1 run, but the numbers that mattered were
  not affected.** `build_fresh_trainer()` wrapped a module-level `base_model` (loaded
  once) with `get_peft_model()`, `beta=.04`, no adapter-composition assertion --
  confirmed by direct inspection, not inferred. As currently structured, Phase 0 is
  REMOVED from the automatic launch path (a 2026-08-29 decision, unrelated to this
  leak) and `build_fresh_trainer()` is called exactly once per process (Phase 1 only)
  -- so as the script stands today, and as it actually ran to produce the current
  frozen constants, it is NOT vulnerable in practice. The historical exposure is real
  but narrower than it first looks: Phase 0 was the FIRST `get_peft_model`/
  `GRPOTrainer` call in that (now-historical) process, so nothing existed yet to leak
  INTO it -- Phase 0's own measurement (`ENTROPY_CLAMP_VALUE=1.2189`, the one constant
  from this script that IS frozen and reused in the reported config) was produced
  before any "ref" adapter could have been registered. Phase 1, the SECOND call in
  that process, is the one that could plausibly have picked up a stray `['default',
  'ref']` composition -- but Phase 1's own output from that historical dry run
  (`terminal_step=8`, milestone 4/8 accuracy) was used only as a pre-flight/smoke-test
  gate before committing to the real 50-step run, never reported as a Stage 9b finding
  in its own right. **Verdict: vulnerable in code, plausibly triggered in one historical
  phase transition, but the specific value that propagated forward
  (`ENTROPY_CLAMP_VALUE`) was measured before the exposure point, and the one output
  that WAS potentially exposed (Phase 1's dry-run numbers) was never load-bearing.**

- **`calibration_seed_stability_check.py`**: **(b) vulnerable in code, plausibly
  triggered on 3 of its 4 seed trials, but its numeric outputs do not feed the frozen
  config.** `run_one_calibration_trial()` wrapped the same module-level `base_model`
  across all 4 `SEEDS_TO_CHECK` trials, `beta=.04`, no adapter-composition assertion.
  Trial 1 (first `get_peft_model` call in the process) is clean; trials 2-4 could
  plausibly have run under a `['default', 'ref']` composition rather than the intended
  clean `['default']`. This script's actual measured D_MAX values (0.050781-0.058358
  across the 4 seeds, "0/4 degenerate") were used only to reach a DECISION -- per-run
  calibration is unstable across process boundaries, abandon it -- not as the source of
  any constant used in the reported run. The frozen `FIXED_D_MAX=0.505` was
  subsequently borrowed from an entirely different source (the 3B model's own
  calibration, `step14b_design.md` -- a different stage, different process, not
  implicated by this leak at all). **Verdict: vulnerable in code, plausibly triggered
  on trials 2-4, but even if those 3 trials' specific numeric outputs are unreliable,
  the downstream action taken (abandon per-run calibration; use a value borrowed from
  a wholly separate, unaffected source) does not depend on trusting those specific
  numbers.**

- **`dmax_sensitivity_check.py`**: **(b) vulnerable in code, plausibly triggered on its
  second ("high" D_MAX) trial, but the reported mechanistic conclusion does not depend
  solely on that trial's measured value.** `run_one_dmax_trial()` wrapped the same
  module-level `base_model` across both `D_MAX_TRIALS`, `beta=.04`, no
  adapter-composition assertion. Trial 1 ("low", first call) is clean; trial 2 ("high",
  second call) could plausibly have run under a leaked `['default', 'ref']`
  composition. The empirical finding drawn from this script (a ~200x KL-metric
  difference between the two trials) was one leg of a two-part argument -- the OTHER
  leg is a pure closed-form calculation (`CLAMP_VALUE = exp(D_MAX) - D_MAX - 1`, a
  ~270x difference in the clamp ceiling between the two D_MAX endpoints, independent of
  any run's measured output) that stands on its own regardless of whether trial 2's
  specific empirical numbers are trustworthy. **Verdict: vulnerable in code, plausibly
  triggered on the second trial, but the qualitative conclusion it fed into ("D_MAX
  magnitude has a real, non-trivial mechanistic effect") is independently supported by
  a formula-based argument that does not rely on trial 2's exact measured value being
  uncontaminated.**

- **`injection_full_7b.py`** (the actual reported 50-step run): **(a) not vulnerable --
  single-phase load only.** Directly confirmed by inspection: `base_model` is loaded
  once, `build_fresh_trainer()` is defined once and called exactly once in the module
  body (one `get_peft_model` call in the whole process). This is the script whose
  output IS the reported Stage 9b headline finding (0 non-literal completions across
  400 training rollouts + 481 milestone-eval completions) -- confirmed clean,
  independent of the other three scripts' findings above.

**Net assessment**: the reused-base-model pattern is real and was present, unguarded,
in all three flagged scripts. Tracing what each script's output actually FED into
downstream shows the two frozen constants baked into the final reported config
(`FIXED_D_MAX=0.505`, borrowed from a different model/stage entirely;
`ENTROPY_CLAMP_VALUE=1.2189`, measured before the exposure point in its own historical
run) are not plausibly corrupted by this specific leak, and the one script confirmed to
have produced the actual reported result (`injection_full_7b.py`) never exhibited the
pattern at all. **The Stage 9b headline result stands.** The intermediate diagnostic
scripts' own numeric precision (seed-stability range, the exact magnitude of the 200x
KL swing) carries residual uncertainty for the trials past the first call in each
script's process, but the DECISIONS drawn from them do not depend on that precision
being exact -- distinguishing "code pattern fixed defensively" (all three) from
"confirmed to have affected the reported result" (none) as two different conclusions,
per instruction, rather than conflating them.

**Fix applied to all three vulnerable scripts, regardless of whether the leak was
actually triggered in a load-bearing way** -- same pattern as Stage 9d's fix: base
model is now loaded fresh (a genuinely independent `AutoModelForCausalLM.from_pretrained`
call) INSIDE the per-call function (`build_fresh_trainer`, `run_one_calibration_trial`,
`run_one_dmax_trial`) rather than once at module scope, plus `assert
list(model.peft_config) == ['default']` immediately after each `get_peft_model()` call
as a runtime guard. `injection_full_7b.py` deliberately left untouched (correctly not
vulnerable; a regression test guards against it being mistakenly "fixed" into a
different call pattern later without updating the check).

**Regression test added**: `test_checkpoint_isolation_09b.py` (new, 8 tests, same
source-level static-inspection style as Stage 9d's own `test_checkpoint_isolation.py`
-- renamed with a `_09b` suffix to avoid a same-basename collision with that file
under pytest's default collection, since this can't be verified end-to-end without a GPU) -- confirms no module-level
shared `base_model` load remains in any of the three scripts, confirms each per-call
function loads a fresh base model and asserts adapter composition, and confirms
`injection_full_7b.py`'s single-call structure is unchanged. Full existing suite
(38 tests across this directory) re-run and passes, no regressions. No GPU run was
needed or performed for this task.
