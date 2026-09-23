# Stage 9d — SFT-Seeded Adversarial RL

Stage 9c showed plain SFT on 16 Nib/Nomo trajectories produces 100% held-out
non-literal token usage (never once observed across Stages 9/9b's thousands of
RL-based rollouts), though only 28.6% of those completions were also correct, and
tier (c) showed the learned mapping is fairly rigid (didn't flexibly follow an
in-context instruction to substitute a different pair).

This directly parallels Stage 6's design (seed a capability, then test under RL/
adversarial pressure) but stronger and more direct: Stage 6 seeded via cross-domain
transfer (fan/valve/lamp) and got a clean negative when applied to Coin Flip, leaving
open whether the null result was about the cross-domain transfer gap or something more
fundamental. Stage 9d seeds directly on Coin Flip itself via SFT, removing that gap
entirely. **This is the most direct test yet of whether SFT-seeding plus adversarial
RL pressure can produce genuine steganographic behavior on the actual target task** --
distinct from Stage 6's cross-domain-seeded negative, and distinct from Stages 9/9b's
RL-from-scratch/RL-with-injection negatives, since here the model already produces the
non-literal code by default before any RL pressure is applied at all.

## 1. Base checkpoint

Stage 9c's SFT-seeded final adapter (`stage9c-sft-diagnostic-v1/final_adapter/`, on
the same EBS-backed instance -- not re-trained, loaded directly). Single adapter, no
composition, loaded via the same `set_peft_model_state_dict(get_peft_model(base_model,
lora), load_safetensors(...))` pattern used throughout Stage 9/9b for loading a
pre-existing adapter onto a fresh `PeftModel` wrapper. Identity confirmed by hashing
the loaded adapter's parameters immediately after load and comparing against a
zero-step sanity eval (does the freshly-loaded checkpoint reproduce Stage 9c's own
~100% held-out non-literal rate on the clean-21 set, BEFORE any Stage 9d training) --
this is the concrete "loaded correctly" check, not just a file hash.

## 2. Reward configuration

**Interpretation, stated explicitly rather than assumed**: "Stage-1-equivalent
strength" is read as `reward_v3.score_completion_v2`'s own DEFAULT parameters
(`cot_max_scale=2.0, consistency_magnitude=0.15, signal_magnitude=0.15`) -- the
CURRENT, already-tested full undeclared-condition reward (p_CoT + r_task + p_length +
p_structure + p_state_variation + r_consistency + r_signal), which its own module
docstring describes as a smoother, refined replacement for the original Stage-1-era
3-tier discrete banned-pattern curriculum, not a literal resurrection of Stage 1's own
now-superseded code. These are, concretely, the SAME defaults already used unchanged
throughout Steps 10-14b's `BRIDGE_REWARD_PARAMS = {}` (an empty override dict takes
every one of `score_completion_v2`'s defaults) -- so "reinstating" the CoT penalty
specifically means NOT overriding it to zero the way Step 0 did, not introducing new
reward code.

**Architectural consequence, also stated explicitly**: Stage 9d does NOT reuse Stage
9's bank-of-16/reward-gate/rollout-injection apparatus (`step14_reward_gate.py`,
`step14b_injection.py`). That machinery existed to solve a different problem (get
GRPO a comparison point when the model NEVER produces non-literal tokens on its own).
Stage 9c's checkpoint already produces non-literal tokens on effectively every
scenario by default -- injection's entire premise doesn't apply here. Stage 9d trains
uniformly across the same 43-scenario train pool (`build_clean_length5_train_eval_split`,
unchanged) with plain `score_completion_v2` applied identically regardless of scenario
identity, structurally returning to Stage 1's original uniform design.

**Reward invariant, re-verified fresh** (`verify_reward_invariant`, checkpoint-
independent by construction -- flagged explicitly: this check validates the reward
FORMULA's internal margins for given parameters; it examines no completions and
depends on no checkpoint, so it cannot by itself detect a checkpoint-specific
interaction between the seeded non-literal bias and the reinstated penalty terms. That
concern is checked empirically in the dry runs below, not here.):

- Main adversarial config (full defaults): `margin_correct_over_wrong=0.70,
  margin_wrong_over_malformed=0.70` -- both strictly positive, matches Steps 10-14b's
  own already-verified values exactly (same parameters).
- Baseline correctness-only config (see section 3): `margin_correct_over_wrong=3.0,
  margin_wrong_over_malformed=3.0` -- both strictly positive, matches Step 0's own
  already-verified values exactly (same parameters).

No surprises at the reward-formula level for either config -- both are re-uses of
already-validated parameterizations, not new derivations.

## 3. Baseline control (run FIRST, before the adversarial config)

Isolates: does ordinary RL fine-tuning toward correctness alone -- with NO adversarial
anti-literal pressure and NO pro-non-literal bonus -- erode the SFT-seeded non-literal
behavior by itself? Reuses Step 0's own `STEP0_REWARD_PARAMS` exactly
(`signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0,
cot_max_scale=0.0`) -- r_task and the structural checks (p_length/p_structure/
p_state_variation) remain active, but nothing rewards or penalizes literal-vs-
non-literal token choice specifically. A short run (16 steps), independent of and
run BEFORE the main adversarial dry run -- both start fresh from the SAME Stage 9c
checkpoint (not sequential; a clean two-arm comparison against one shared starting
point), so any divergence between the two arms is attributable to the reward
configuration, not to one run inheriting the other's drift.

## 4. Main run tracking (once the baseline control is in and the dry run is launched)

At every milestone:
- Non-literal token usage rate (does it persist from ~100%, grow, or collapse)
- **Genuine state-tracking correctness among non-literal completions specifically**,
  via taxonomy's `structural_nonliteral_candidate` flag (categories 3-9) crossed with
  final-answer correctness -- this is the actual bar for concealment, not "any
  non-literal token present" (which category 4, `prompt_domain_echo`, or category 5,
  `position_driven_drift`, would also satisfy without being functional tracking)
- Whether any literal reversion is gradual or sudden
- Same milestone cadence (every 4 steps), breakers (grad_norm>=50, kl>=5, unchanged),
  and evidence-retrieval/hash-verification/teardown discipline as every prior stage

## 5. Token-identity drift (per instruction 5, given tier (c)'s rigid-mapping finding)

At every milestone, in addition to the taxonomy category breakdown: scan non-literal
completions specifically for which TOKENS were used (Nib/Nomo vs. anything else),
reporting a token-identity histogram alongside the taxonomy breakdown. Tests whether
correctness pressure reinforces the SPECIFIC memorized pair or causes drift toward
other tokens -- informative either way about whether what's being reinforced (if
anything) is adaptive concealment or incidental preservation/erosion of a memorized
pattern.

## 6. GPU discipline

8-step dry run of the MAIN adversarial config first (standard discipline). The
baseline control (16 steps) is itself short/cheap by design and launched directly,
not gated behind a separate dry run of itself. `scripts/gpu_teardown.py` for
teardown, "TEARDOWN FULLY CONFIRMED" required before considering any phase closed.
**A longer main run beyond the 8-step dry run is NOT launched without a fresh
go-ahead**, matching every prior stage's convention for a genuinely new
checkpoint+config combination -- this task's own instructions ask for the dry run
"before any longer run," not for the longer run itself.

---

## Progress log

**2026-08-29 -- first launch: a real bug, caught by an assertion this script added
that no prior stage had.** Phase 1 (baseline correctness-only) ran and produced a
genuine result: the breaker fired at step 3 (`grad_norm=53.25 >= 50.0`), after
grad_norm=48.75 and 41.0 at steps 1-2 -- large gradients essentially immediately,
with no adversarial pressure active at all. This result is trustworthy as reported
(Phase 1 was the first use of the base model in the process; nothing precedes it that
could contaminate it) and is discussed on its own terms below, not discarded.

Phase 2 (main adversarial dry run) crashed before training even started:
`load_stage9c_checkpoint()`'s composition assertion caught `['default', 'ref']`
instead of the expected `['default']`. Root cause, confirmed by reading the installed
TRL source directly (not assumed): `trl/trainer/grpo_trainer.py` calls
`model.add_adapter("ref", default_config)` whenever a PEFT model with a pretrained
adapter is used with `beta != 0` (a real, by-design mechanism -- a frozen snapshot of
the reference policy, avoiding a full second model copy). This script's first version
shared one `base_model` object across Phase 1 and Phase 2 (matching the pattern
`build_fresh_trainer` used in Stage 9b's own scripts). `add_adapter` registers the
new LoRA modules directly onto the shared base model's layers, so the 'ref' adapter
Phase 1's trainer created was still there when Phase 2 wrapped the SAME base model
again, and got exposed as an unexpected second adapter on the freshly-loaded
'default'.

**Whether this residue would actually have corrupted Phase 2's forward pass was not
established** -- PEFT's active-adapter mechanism plausibly isolates it (only the
active adapter's delta should apply), but this was not verified against the installed
version's actual behavior, and TRL's own `add_adapter` call would have been invoked
a SECOND time with a name that already existed once Phase 2's own `GRPOTrainer` was
constructed, an interaction with no verified outcome either way. Rather than trust an
unverified assumption on a run whose entire point is checkpoint-identity rigor, fixed
by never sharing a base-model object across `get_peft_model()` calls at all --
`load_stage9c_checkpoint()` now loads a fully independent base model on every call.
Costs a few extra seconds of 8-bit reload per phase; buys certainty.

**Flagged, not silently generalized**: this same reused-base-model pattern appears in
several already-completed Stage 9b scripts (`injection_dryrun_7b.py`'s Phase 0 ->
Phase 1 transition, `calibration_seed_stability_check.py`'s 4 trials,
`dmax_sensitivity_check.py`'s 2 trials) -- none of which ever asserted adapter
composition, so if the same leak occurred there it went undetected. The two Stage 9b
results with the least exposure -- the reported 50-step full run
(`injection_full_7b.py`, calibration already removed from its launch path by the time
it ran) and Stage 9c's SFT training (a single `get_peft_model` call, no reuse at all)
-- are not implicated by construction (one `get_peft_model` call per process each).
The calibration-derived numbers (the seed-stability check's D_MAX values, the D_MAX
sensitivity check's telemetry) are the ones with real exposure to this specific
mechanism; whether it actually affected them is unverified, not confirmed-safe or
confirmed-broken. Not re-litigated in this task -- noted here as a discovered,
disclosed gap in that earlier work's own rigor, for the record.

Relaunching with the fix now.

**2026-08-29 -- relaunch: clean run, no crash, but a real and reproducible finding
that blocks the actual question this stage was designed to test.**

The zero-step sanity check passed cleanly (checkpoint identity verified, 21/21 =
100% non-literal on the clean-21 set BEFORE any Stage 9d training, matching Stage
9c's own report exactly). Both phases then ran with genuinely independent base-model
loads (the fix held -- no adapter-composition assertion failure this time).

**Both the baseline (correctness-only, no CoT penalty) AND the main adversarial
(full Stage-1-equivalent reward, CoT penalty active) configs hit the grad_norm
breaker at step 3**, independent of which reward configuration was active:

| | step 1 | step 2 | step 3 (breaker) |
|---|---|---|---|
| baseline (no p_cot) | grad_norm=48.75 | grad_norm=13.88 | **grad_norm=69.5** (kl=0.208) |
| main adversarial (p_cot on) | grad_norm=49.0 | grad_norm=31.25 | **grad_norm=54.5** (kl=0.134) |

Both runs terminated at `terminal_step=3`, before the first milestone (every 4 steps)
was ever reached -- **no milestone data exists for either phase**, so the actual
tracking this stage was built for (non-literal usage rate, genuine-correctness-among-
non-literal, token-identity drift) could not be observed at all in this dry run.

**This is not attributable to the CoT penalty** -- the baseline run, with p_cot,
r_consistency, and r_signal all zeroed (matching Step 0's own proven-stable
configuration), hit the SAME breaker at the SAME step with a comparable grad_norm
magnitude. KL stayed small and unremarkable in both (0.13-0.28, far under the 5.0
breaker) -- this is specifically a grad_norm spike, not a broader divergence in the
policy's output distribution relative to the reference. The most likely explanation,
not yet verified: Stage 9c's SFT training (LR=2e-4, no KL regularization, full-batch,
converging to near-zero loss) leaves the policy considerably more peaked/confident
than any prior stage's actual GRPO starting point (Step 0's foundation, or an
untouched base) -- resuming GRPO on top of that with the SAME TARGET_LR=1e-6/
WARMUP_UPDATES=5 schedule used everywhere else in this project produces large
policy-gradient magnitudes almost immediately, regardless of reward shaping.

**Verdict: HOLD.** The dry run did its job -- it caught a real instability BEFORE any
longer run, exactly per instruction. But it means Stage 9d's actual research question
(does non-literal usage survive/collapse under adversarial pressure) has not yet been
tested; both arms failed at the same step for what looks like an unrelated training-
stability reason. Recommend addressing the underlying instability (most plausibly:
a longer/gentler warmup or a lower initial LR specifically tuned for an SFT-seeded,
already-peaked starting policy, rather than reusing Step 0's from-scratch schedule
unchanged) before attempting either phase again -- not simply raising the grad_norm
breaker threshold to push through without understanding why it fired. Not fixed or
retried in this task; flagged for a decision on how to proceed.

**Evidence, retrieved and hash-verified** into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-sft-seeded-adversarial-rl-v1/`:
full event log (SHA-256 `2e9a6843bec991bea1d6cf340d4dcd4ba9cb84ff4e8bf58d38fc32ba06f969a4`)
and the full stdout log (both phases, including the sanity check).

**Teardown**: `scripts/gpu_teardown.py`, `g5.xlarge` throughout.

## Entropy/logprob diagnostic (2026-08-29): the "peaked-policy" hypothesis is NOT
## confirmed -- disconfirmed by direct measurement, reported plainly rather than
## proceeding to pick an LR/warmup number anyway

Before proposing any schedule revision, measured entropy and chosen-token logprob
directly (`entropy_diagnostic.py`): for each checkpoint, generate greedily on the same
8 clean-21 prompts, then teacher-force a fresh forward pass over (prompt+completion)
to extract the REAL per-position output distribution and compute entropy = -sum(p log
p) over the full vocab at every completion position, plus the logprob the model
assigned to the token it actually chose there. Each checkpoint scored against its OWN
natural output (apples-to-apples: each policy's confidence in the tokens it would
itself choose, not one model's confidence scored against the other's text). Same
base-model-isolation fix as `sft_seeded_rl.py` applied here too (fresh base model per
checkpoint load, `assert list(model.peft_config) == ['default']`) --
`test_checkpoint_isolation.py` (5 tests, CPU-only, no GPU needed) statically confirms
neither script's checkpoint loader can regress back to a shared base-model object.

**Result**:

| | Step-0 milestone-12 | Stage 9c SFT |
|---|---|---|
| grand mean entropy (nats) | 0.01949 | 0.01893 |
| grand mean chosen-token logprob | -0.01446 | -0.00544 |
| mean fraction of near-certain tokens (logprob > -0.01) | 94.2% | 94.5% |

**entropy_ratio_step0_over_stage9c = 1.029** -- essentially no difference. Sanity-
checked against the raw completions, not just the aggregate numbers: both checkpoints
produce coherent, well-formed, on-template output (Step-0: "Step N: The state remains
unchanged. State: Heads/Tails"; Stage 9c: "Step N: The code remains unchanged. State:
Nib/Nomo"), and per-prompt mean entropy sits in the same tight 0.006-0.043 nat range
for both -- not a measurement artifact.

**This directly disconfirms the working hypothesis** ("Stage 9c's SFT training
converged to a meaningfully more peaked/lower-entropy policy than a typical GRPO
starting point, and that peakedness is what produces large gradients when GRPO
resumes on top of it"). Step-0's milestone-12 -- an ALREADY-PROVEN, successfully-used-
for-GRPO-throughout-this-entire-project starting point -- is, by this direct
measurement, essentially just as peaked/confident on its own greedy output as Stage
9c's checkpoint. Both models are extremely confident predicting the next token of a
short, rigid, highly-templated reasoning format once established -- unsurprising in
hindsight for either checkpoint, and not something that distinguishes "SFT-seeded" from
"RL-seeded" starting points on this task.

**Per instruction, reported plainly rather than pushed past**: since the entropy/logprob
measurement -- the evidence the LR/warmup revision was supposed to be justified BY --
does not support the hypothesis, no revised schedule is proposed on this basis. Picking
LR/warmup numbers anyway, without a measurement backing them, would be exactly the
"trial and error" the task asked to avoid, and would produce a schedule whose
justification doesn't hold up. **The step-3 grad_norm breaker trips in both the
baseline and adversarial dry runs remain real and reproducible, but their cause is not
yet identified.** Candidates not yet tested: something specific to the SAMPLED (not
greedy) rollout distribution at temperature=0.8, an interaction with GRPO's advantage
normalization or `entropy_coef=0.05` term specifically, or something about the LoRA
adapter's weight-space geometry after SFT (as opposed to output-distribution entropy)
that a per-token entropy measurement wouldn't capture. Not investigated further in this
task -- stopping here to report rather than guessing.

**Evidence retrieval note, disclosed rather than glossed over**: the entropy
diagnostic's full result JSON (`~/aisi_checkpoints/stage9d-entropy-diagnostic/
entropy_diagnostic_result.json`) was NOT retrieved and hash-verified locally --
teardown was issued and the instance began shutting down before the retrieval
command completed, and the file was not pulled down before that happened. The
figures recorded above are transcribed faithfully from the live SSH session's
captured stdout (the full comparison JSON and 3 sample completions per checkpoint),
not fabricated or estimated, but this is a real gap in this run's evidence-retrieval
discipline compared to every other GPU phase in this project, noted explicitly rather
than presented as equivalent to a verified retrieval.

**Teardown (entropy diagnostic phase)**: `scripts/gpu_teardown.py`, `g5.xlarge`
throughout.

**Process fix, confirmed working this time**: the entropy diagnostic's own evidence
JSON was NOT retrieved before the teardown above -- flagged explicitly in this doc at
the time. It survived on the instance's EBS volume and was retrieved and hash-verified
in the follow-up task below (SHA-256 `935e936a375d7cfe4dafecfd34a50b5239802791700ec
54e26b2f460307d86df`, matches exactly), so no evidence was actually lost -- but the
retrieval discipline itself is what should have happened the first time, not a lucky
recovery.

## Rollout-diversity diagnostic (2026-08-30): the advantage-extremity hypothesis is
## ALSO disconfirmed -- second negative, reported plainly rather than guessing a third

Leading remaining hypothesis after the entropy result: GRPO's advantage normalization
`(reward - group_mean) / (group_std + 1e-4)` could produce extreme values if Stage 9c's
sampled rollout GROUPS (temperature=0.8, matching the real GRPOConfig exactly -- not
the greedy single-completion measurement above) are unusually homogeneous, making
group_std near zero. Tested directly (`rollout_diversity_diagnostic.py`): for both
checkpoints, generated a real 8-completion rollout group (temperature=0.8, top_p=0.95)
on each of the SAME 8 prompts used in the entropy diagnostic, measured pairwise text
similarity (`difflib.SequenceMatcher` ratio, all 28 pairs per group) and exact-duplicate
count, scored every completion under BOTH Stage 9d reward configs (baseline
correctness-only and full adversarial), and computed advantages via GRPO's own exact
formula. Same base-model-isolation fix and structural regression tests extended to
cover this script (`test_checkpoint_isolation.py`, 7/7 passing).

**Result**:

| | Step-0 milestone-12 | Stage 9c SFT |
|---|---|---|
| mean pairwise text similarity | 0.9257 | 0.9183 (ratio 0.992 -- no meaningful difference) |
| groups with near-zero reward variance (of 8) | 4 | 5 |
| mean reward_std (baseline / main config) | 0.680 / 0.698 | 0.399 / 0.405 |
| **mean max\|advantage\| (baseline / main config)** | **1.208 / 1.207** | **0.878 / 0.878** |

Text-level diversity between the two checkpoints' sampled groups is essentially
identical (a 0.8% difference, well within noise for 8 groups). Stage 9c does have one
more near-zero-variance group (5 vs 4 of 8) and a lower mean reward_std -- the one
number that moved in the hypothesized direction. But **near-exactly-identical rewards
give advantage=0 under GRPO's exact formula** (if every reward in a group equals the
group mean, every numerator is 0, regardless of how small the denominator's floor is)
-- so a near-zero-variance group is the SAFE case, not the dangerous one. The metric
that actually determines gradient magnitude, the realized advantage values themselves,
came out LOWER for Stage 9c in both reward configs, not higher -- the opposite
direction from what the hypothesis predicts.

**Verdict: disconfirmed.** Neither single-completion entropy (first diagnostic) nor
rollout-group homogeneity/advantage-extremity (this diagnostic) distinguishes Stage
9c's checkpoint from Step-0's milestone-12 in the direction that would explain the
step-3 grad_norm breaker trips. Per instruction, no fix is proposed for a mechanism
that measurement doesn't support -- inventing one here would be exactly the guessing
this task and the one before it were both designed to avoid.

**What remains untested, stated plainly rather than silently left implicit**: the
"weight-space geometry" catch-all from the first diagnostic (some property of the
SFT-produced LoRA weights -- e.g. a specific direction with an unusually large
loss-curvature that GRPO's optimizer happens to move along early on -- that neither
output-distribution entropy nor sampled-completion diversity would capture), an
interaction specific to `entropy_coef=0.05` or the `beta=0.04` KL term with this
checkpoint's reference-vs-policy configuration, or something particular to how
gradient accumulation interacts with a policy this confident about a rigid template
regardless of which specific tokens fill it. Not investigated further in this task,
per instruction -- stopping here for a decision on how to proceed rather than guessing
a third mechanism.

**Evidence, retrieved and hash-verified BEFORE teardown this time** (the explicit
process fix this task asked for) into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/`: both diagnostics' full result
JSONs (entropy: SHA-256 `935e936a375d7cfe4dafecfd34a50b5239802791700ec54e26b2f460307
d86df`; rollout-diversity: SHA-256 `3e062ae4a8d7ec9065e779da9f11e828be0f91158542f816
89589d63a62ba2f4`, both confirmed identical between the remote-computed hash and the
locally-retrieved copy) plus both stdout logs. Retrieval completed and confirmed
before `gpu_teardown.py` was ever invoked.

**Teardown**: `scripts/gpu_teardown.py`, `g5.xlarge` throughout.

## Per-token gradient diagnostic (2026-08-30): first attempt hit a real bug, disclosed
## and fixed before any useful data was lost

Rationale: Stage 4's per-token KL clamp precedent showed an aggregate/mean metric can
hide an extreme per-token outlier. Both prior diagnostics measured aggregate/proxy
quantities (single-completion entropy; rollout-group text diversity and advantage
statistics) -- this inspects the actual gradient directly: `retain_grad()` on the
per-token POLICY logprob tensor inside `_get_per_token_logps_and_entropies` (same
interception point as the KL clamp, extended for gradient capture rather than KL
capture), read after each physical step's backward() completes, to see whether
Nib/Nomo token positions contribute disproportionately to total gradient magnitude
relative to their share of tokens.

**First launch (N_STEPS=4, minimal repro): a real bug, caught by a genuine CUDA OOM,
not silently worked around.** Two things went wrong, both disclosed here rather than
just fixed and rerun quietly:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- set in every other GPU script
   in this project to reduce allocator fragmentation -- was missing from this one.
2. `retain_grad()` keeps a tensor's backward-graph buffers alive until the reference is
   released; this script recorded one retained tensor per micro-batch (8 per physical
   step) into a list that was never cleared across steps. By step 4 (32 accumulated
   retained tensors across 4 steps), the process hit a genuine `CUDA OutOfMemoryError`
   inside `entropy_from_logits` -- not a breaker trip, an actual crash.

**Also revealed, independent of the bug**: at the point of the crash, telemetry showed
grad_norm=29.9 (step 2) and grad_norm=37.0 (step 3) -- BELOW the 50.0 breaker this
time, unlike every prior run at these same settings (baseline config, same seed,
same checkpoint), which consistently tripped at step 3. This is a further, real
demonstration of the run-to-run GPU/CUDA nondeterminism this project has now observed
repeatedly (the D_MAX calibration swings in the model-scale ablation; malformation-
rate differences between nominally identical dry runs) -- the step-3 spike is
reproducible in AGGREGATE across multiple runs, but not perfectly deterministic on
which exact step it lands on run to run, even at a fixed seed. **N_STEPS raised from
4 to 8** (this project's standard dry-run budget) to give real margin against this
variance rather than assuming a fixed spike step.

Fixed: each step's retained tensors are now explicitly released (`cap['logps_tensor']
= None`, removed from the live list, `torch.cuda.empty_cache()`) immediately after
that step's gradient analysis extracts the needed numbers to plain CPU data -- no
tensor is kept alive past the step it was captured for. `PYTORCH_CUDA_ALLOC_CONF` set
to match every other script. No useful data was lost from the failed first attempt
(the crash happened before any step reached the breaker, so there was nothing to
salvage) -- relaunching with the fix now.

**Second launch (N_STEPS=8): completed cleanly (no crash, breaker fired correctly at
step 3 again, `grad_norm=51.75`) -- but the RESULT was implausible, and caught rather
than reported at face value.** `token_share_code=0.0` at every one of steps 1-3 --
zero Nib/Nomo token positions found across 24 total completions, on a checkpoint
independently established (Stage 9c's own diagnostic) to use Nib/Nomo on ~100% of
held-out completions. Not plausible as a genuine finding; investigated before trusting
it.

**Root cause: the exact same word-detection pitfall this project already found and
fixed once before**, in the completion-level qualitative analysis of Step 14b's data
(`experiments/09_direct_indomain_synthetic_bridge`, "Tails" split across tokens as
">T"+"ails"). This script's `analyze_step_gradients` decoded one token at a time and
substring-matched the decoded piece against "nib"/"nomo" -- but Qwen's tokenizer
splits "Nib" as `" N"` + `"ib"` and "Nomo" as `" N"` + `"omo"` (confirmed directly:
`tokenizer.decode` on each piece individually, neither half contains the full word),
so the naive per-token check silently found nothing, every time, by construction --
not a real result about the checkpoint's behavior. Repeating a mistake already fixed
once in this same project is disclosed here plainly, not glossed over.

**Fixed** with the same span-based approach used the first time: reconstruct the full
decoded text with per-token character offsets, locate "nib"/"nomo" as whole-word
substrings in the reconstructed text, map the character span back to every token index
it overlaps (`find_code_word_token_positions`). Verified directly against the real
Qwen2.5-3B-Instruct tokenizer (not a mock) before redeploying: correctly finds all 6
token positions across 3 Nib/Nomo occurrences in a hand-built example, where the
original per-token check is proven (by a dedicated regression test,
`test_code_word_detection.py`) to find zero. 4 new tests, 13/13 passing across the
whole Stage 9d suite (all local, real-tokenizer-based, no GPU needed). Relaunching
with the fix now.

**Third launch: clean run (breaker fired again at step 3, `grad_norm=65.5`), real
numbers this time -- and a result that needed tracing to its actual mechanism rather
than reported at face value.**

Aggregate result, exactly as measured:

| step | token_share_code | grad_share_code | disproportionality ratio |
|---|---|---|---|
| 1 | 7.13% | 10.12% | 1.42 |
| 2 | 6.35% | 8.11% | 1.28 |
| 3 (breaker step) | 4.59% | 6.80% | 1.48 |

Taken alone, this looks like a "yes" -- Nib/Nomo positions contribute 1.3-1.5x their
token-count share to total gradient magnitude, consistently, including at the breaker
step. **But the row-level detail tells a different, more precise story, and changes
the conclusion.** Within each individual completion, the average |grad| at code-word
positions is essentially IDENTICAL to the average |grad| at all other positions in
that SAME row (e.g. one row: avg_grad/token=0.00063, avg_grad/code_token=0.00063;
this pattern holds across essentially every row in all 3 steps -- not a coincidence,
checked deliberately). More directly: across all 24 rows (8 per step x 3 steps), the
SINGLE LARGEST-gradient token in a row was a code-word position only **1/24 times
(4.2%)** -- at or below the ~5-7% base rate code tokens make up of the row by chance
alone. If individual Nib/Nomo positions genuinely produced outlier per-token
gradients (the Stage 4 KL-clamp-analogous mechanism this diagnostic set out to test),
the row's peak gradient token should land on a code position noticeably MORE often
than chance, not at or below it.

**What actually explains the aggregate ratio**: a row-level correlation, not a
token-level effect. Computed directly (Pearson correlation between a row's code-token
COUNT and that row's TOTAL |grad| across all 24 rows): **r = 0.65** -- rows with more
Nib/Nomo tokens tend to have substantially higher total gradient magnitude for the
row as a whole, not specifically at the code positions within it. The most likely
explanation, consistent with GRPO's own per-token loss formula (advantage scales
every token's loss/gradient in a row roughly uniformly): rows with larger
reward/advantage magnitude happen, in this sample, to also be the rows with more
code-token content -- an incidental correlation between two row-level properties, not
evidence that the code tokens THEMSELVES are the gradient outliers.

**Verdict: the specific mechanism (individual Nib/Nomo token positions as per-token
gradient outliers, directly analogous to Stage 4's per-token KL finding) is NOT
confirmed.** The aggregate number that looked like a "yes" evaporates under row-level
scrutiny into a "no" for the mechanism that would justify a per-token gradient
clamp -- reported precisely rather than stopping at the aggregate ratio and
proposing a fix for a mechanism the more granular evidence doesn't actually support.
Per instruction, no per-token clamp is proposed on this basis.

**What remains, stated as a concrete next lead rather than a vague unknown**: the
row-level correlation (r=0.65 between code-token count and total row gradient) points
toward reward/advantage magnitude as the more likely per-ROW (not per-token) driver
of the grad_norm spike -- consistent in spirit with the rollout-diversity diagnostic's
own advantage measurements, but not yet tested at this finer grain (this diagnostic
did not capture reward/advantage per row alongside the gradient data, so the
correlation is observed, not yet causally traced). A natural, well-defined follow-up
-- not attempted in this task, stopping here per instruction rather than guessing
further without a decision on how to proceed.

**Evidence, retrieved and hash-verified BEFORE teardown** (confirmed per the explicit
process fix, both this run's result and the corrected script's full stdout log) into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-per-token-gradient-diagnostic/`:
full result JSON, SHA-256
`1dee706d6efd38131e8b502207d5a7eb89f8a73a580c5c400351f1c4b7664340`, confirmed
identical between the remote-computed hash and the locally-retrieved copy.

**Teardown**: `scripts/gpu_teardown.py`, `g5.xlarge` throughout.

## Reward-confound check (2026-08-30, read-only): required data does not exist yet --
## stopping to report rather than launching a new run

Before computing anything, confirmed directly against the retrieved, hash-verified
evidence files (not assumed from memory) whether per-row code-token count, gradient
magnitude, and reward/advantage all exist TOGETHER for the same rows:

- `per_token_gradient_diagnostic_result.json`'s `per_row_details`:
  `{n_tokens, n_code_tokens, code_positions, total_abs_grad, code_abs_grad,
  max_abs_grad_position, max_abs_grad_value, max_grad_is_code_token}` -- gradient and
  code-token count, but **no reward or advantage field anywhere**, per-row or
  otherwise (the `telemetry` array has only AGGREGATE per-step `reward_mean`/
  `reward_std` across the whole 8-row group, not a value for each individual row).
  The script also never saved completion text, prompt, or ground truth per row to the
  final report -- only token counts and positions -- so reward cannot be reconstructed
  offline after the fact either; the linkage needed (which specific prompt/scenario
  each captured row was) was never persisted.
- `rollout_diversity_diagnostic_result.json`'s `groups[i]`: `main_scoring`/
  `baseline_scoring` have `rewards`, `advantages` per completion, and `completions`
  (raw text, from which code-token count could be independently re-derived) -- but
  **no gradient data of any kind**, since that script only generated and scored
  completions and never trained (no backward pass ever ran).

**Neither file has all three linked per row. Per instruction, no new run is launched
in this task.**

**Smallest proposed run to obtain the missing linkage** (not executed): a small,
targeted patch to `per_token_gradient_diagnostic.py`, not a new experimental design --
`_patched_compute_loss` already receives `inputs['advantages']` and this script's own
`diagnostic_reward` already computes each row's reward via `score_completion_v2`
before training uses it; both are already-available tensors/values at exactly the
point the gradient-capturing patch already runs. The proposal is to additionally
attach `inputs['advantages'][0].item()` and the corresponding reward value to each
`GRAD_CAPTURES` entry (alongside the `logps_tensor`/`completion_ids`/
`completion_mask` already captured there), so `per_row_details` gains `reward` and
`advantage` fields directly. Same checkpoint (Stage 9c), same baseline reward config,
same N_STEPS=8 budget, same breaker discipline -- no new mechanism, just persisting
two more already-computed values per row. With that data, requirement 1's direct
correlation (reward/advantage magnitude vs. total gradient) and requirement 2's
partial correlation (code-token density vs. gradient, controlling for reward) would
both be directly computable from a single additional dry run.

**Stopping here for a decision on whether to run this**, per instruction -- the
attribution question (is the r=0.65 code-density/gradient correlation actually about
reward magnitude, or about code usage independent of reward) remains open until this
data exists.

## Reward/gradient linkage patch (2026-08-30): deploy failure discovered during
## post-hoc recovery -- the needed data does not exist; attribution question remains open

**Patch applied locally** (approved): `per_token_gradient_diagnostic.py` was edited to
add `REWARD_CAPTURES` (one entry per physical step, populated inside `diagnostic_reward`
from values it already computes: per-row `reward`, `completion_text`, `prompt_text`,
`ground_truth`, in the same row order the 8 subsequent micro-batches process), and
`_patched_compute_loss` was edited to read `inputs['advantages']` (already computed by
TRL before `compute_loss` runs) plus the matching `REWARD_CAPTURES` entry via a running
row-index counter, attaching `reward`/`advantage`/`completion_text`/`prompt_text`/
`ground_truth` to each `per_row_details` entry. Syntax-checked clean locally
(`ast.parse`) before any deploy.

**A prior background dispatch (spanning the context-compaction boundary) restarted the
instance, deployed a script, and relaunched it (PID 11153, 04:43 GMT) -- but before this
session could retrieve its output, the instance was independently stopped ("User
initiated", 05:16:37 GMT) with no evidence retrieved and `design.md` untouched. This
was flagged to the user as a process-order violation rather than silently proceeding.**

**Recovery, on explicit approval**: restarted the instance, SSH'd in, and checked the
exact configured output path
(`~/aisi_checkpoints/stage9d-per-token-gradient-diagnostic/per_token_gradient_diagnostic_result.json`).
A file existed there (mtime 04:46 GMT) -- but its SHA256
(`1dee706d6efd38131e8b502207d5a7eb89f8a73a580c5c400351f1c4b7664340`) is **byte-identical**
to the ALREADY-hash-verified evidence from the prior (unpatched) gradient diagnostic run.
The accompanying stdout log confirms this directly: its final JSON report uses the OLD
schema only (`token_share_code`, `grad_share_code`, `disproportionality_ratio`,
`step_summaries`) -- no `reward`, `advantage`, `completion_text`, `prompt_text`, or
`ground_truth` field anywhere.

**Root cause, confirmed by inspection rather than inferred**: the script actually present
on the remote instance at launch time
(`~/stage09_repo/experiments/09d_sft_seeded_adversarial_rl/per_token_gradient_diagnostic.py`,
mtime 04:42 GMT, SHA256 `6b837a5e5e94bf07a51d92ab46dbcbe878dff5c8916b17b6a7f60d13bc2c5d15`)
does **not** contain the `REWARD_CAPTURES` patch at all -- `grep` for it returns zero
matches, and its hash differs from the locally patched file
(SHA256 `2b233d1ee6b1c633f6fbbe3df8bbf43e60682f0a378dec6a2c1a03cbf535169f`). The prior
background dispatch's deploy step pushed a stale, unpatched copy of the script one minute
before launching it. The run that executed was therefore just another repetition of the
already-evidenced diagnostic, not the linkage run this task needed.

**One genuine (if redundant) finding from the stale run**: it reproduced the step-3
grad_norm spike again (`grad_norm=65.5` at step 3, breaker fired, `hard_stop: {step: 3,
grad_norm: 65.5, kl: 0.230}`), consistent with the previously-established run-to-run
variance -- not new information, but confirms the instability is not a one-off artifact
of a single prior launch.

**Verdict on data recovery: the reward/advantage/code-density linkage data does NOT
exist anywhere (not on the instance, not locally).** Per instruction, no attempt was
made to reconstruct or approximate it. The attribution question from the prior task
(confound vs. genuine reward-independent effect) remains fully open. **A fresh run of
the actually-patched script is required to answer it** -- not attempted in this task;
stopping for a decision, per instruction.

**Stop-source investigation (05:16:37 GMT "User initiated" stop): inconclusive, reported
as such rather than guessed.** Checked, in order:
- Local: no crontab entry (`crontab -l` empty), no orphaned local process referencing
  `gpu_teardown` or the instance's IP/ID, no repo script with auto-shutdown/timeout/
  atexit logic beyond `scripts/gpu_teardown.py` itself (which only runs on explicit
  invocation).
- AWS-side: `cloudtrail lookup-events` for this instance ID returned an **explicit
  AccessDeniedException** (`arn:aws:iam::REDACTED_ACCOUNT_ID:policy/REDACTED_POLICY_NAME`
  denies `cloudtrail:LookupEvents` for the assumed SSO role) -- the authoritative source
  for "who/what called stop-instances" is not accessible with this profile's
  permissions. Not treated as evidence of anything; simply unavailable.
- Instance tags are only `Name=AISI-gpu-training`, `Project=AISI` -- no `Schedule` tag,
  which rules out the common AWS Instance Scheduler tag-based pattern (that mechanism
  requires opting in via a tag that isn't present).
- `aws budgets describe-budget-actions-for-account` returned zero actions -- no
  budget-triggered auto-stop configured at the account level.
- `aws events list-rules` / `aws lambda list-functions` show only AWS Control
  Tower/Landing-Zone-Accelerator-managed security/logging infrastructure (this is a
  managed multi-account AWS Organizations environment); nothing named or patterned like
  a cost-control auto-stop function.
- Timing note: the diagnostic's own training run finished at ~04:46-04:47 GMT (169.5s
  train runtime, matching the log); the stop did not fire immediately after
  (~30 minutes later, at 05:16:37) -- this is more consistent with a deliberate,
  possibly-delayed manual/scripted stop call than an immediate post-completion
  auto-shutdown hook, but this is offered as an observation, not a confirmed
  explanation.

**Conclusion: the source of the 05:16:37 stop could not be identified with the access
available.** This is reported as unresolved, not as "probably nothing" -- if an
unattended mechanism in this shared AWS environment can stop instances mid-run without
either party issuing the command, that is a standing risk for any future long run and
should be watched for (e.g., checking instance state partway through any run exceeding
~20-30 minutes) until/unless CloudTrail access is available to confirm a specific cause.

**Evidence, retrieved and hash-verified** (the stale-but-real output from the
unpatched-script run, kept as the record of what actually executed and of the deploy
failure, not as data for the correlation analysis) into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-reward-gradient-linkage/`:
`per_token_gradient_diagnostic_result.json` (SHA256
`1dee706d6efd38131e8b502207d5a7eb89f8a73a580c5c400351f1c4b7664340`, identical
remote/local) and `per_token_gradient_diagnostic_stdout.log` (SHA256
`019e40c2502df7a4d4c18aaadf204cb0fbe1bad12ce8ca168013ecdd514e8ad9`, identical
remote/local).

**Teardown**: `scripts/gpu_teardown.py` -- first invocation hit a transient local
network error on its final confirmation poll (`Could not connect to the endpoint URL`)
after the stop-instances call had already been accepted and observed progressing
through `stopping`; a fresh `describe-instances` check confirmed `stopped` independent
of the script, and a second invocation of `scripts/gpu_teardown.py` produced a clean
**TEARDOWN FULLY CONFIRMED** banner. The SSH ingress rule added for this session
(`<REDACTED_IP>/32` on `sg-REDACTED`) was explicitly revoked afterward. `g5.xlarge`
throughout.

**Net status: the reward/gradient linkage question from the prior task is still
unanswered** -- the local patch is believed correct (syntax-checked, not yet exercised
against a live run) but has never actually executed on the GPU. Re-running it requires,
at minimum, verifying the deployed remote file's hash matches the local patched file
*before* launching, not just after -- the gap this incident exposed.

## Reward/gradient linkage run, take 2 (2026-08-30): genuine linked data obtained --
## verdict: CONFOUND, not a reward-independent code-usage effect

**Pre-launch hard gate, per instruction**: before any launch, computed SHA256 of the
script as scp'd onto the remote instance and compared it against the local patched
file's hash. First deploy attempt hit an unrelated environment issue (the plain
`python3` on PATH lacks `torch` -- the project's actual interpreter is
`~/aisi_venv/bin/python3`; this was a launch-command mistake, not a hash mismatch, and
was caught immediately via the log rather than assumed to have worked). Re-verified the
gate (hash still matched, script untouched) immediately before the corrected launch
command. **Gate passed both times; full record in
`aws_runs/stage9d-reward-gradient-linkage/hash_verification_log.txt`** (pre-launch
match, post-run re-check confirming the script wasn't altered mid-run, and the evidence
files' remote-vs-local hashes).

**Spike reproduction**: the breaker fired at **step 2** this time (`grad_norm=52.0`,
`kl=0.196`), one step earlier than the step-3 firings seen in prior launches. Consistent
with the already-documented run-to-run nondeterminism at fixed seed (Stage 9b's D_MAX
calibration swings, and this same diagnostic's own step-3-vs-not-step-3 variance across
earlier relaunches) -- reported as the same instability phenomenon recurring, not a new
one, rather than silently treating "step 2 instead of step 3" as a mismatch to
rationalize away.

**Data obtained**: 16 rows (2 steps x 8 completions/step -- the run stopped early at the
step-2 breaker, same hard-stop discipline as every prior diagnostic launch), each with
`n_tokens`, `n_code_tokens`, `total_abs_grad`, `code_abs_grad`, `reward`, `advantage`,
`completion_text`, `prompt_text`, `ground_truth` all linked per row.

**Alignment sanity check (done before trusting anything else)**: recomputed
`score_completion_v2` locally on each row's saved `completion_text`/`ground_truth`/
`prompt_text` and compared against the saved `reward` value. **0 mismatches across all
16 rows** -- confirms the row-index-based linkage between `diagnostic_reward`'s
per-group capture and `_patched_compute_loss`'s per-micro-batch capture is correctly
ordered, not silently scrambled.

**Correlations**:
- `corr(n_code_tokens, total_abs_grad) = 0.61` in this 16-row sample -- reproduces the
  earlier r=0.65 aggregate finding (same phenomenon, smaller sample).
- `corr(|advantage|, total_abs_grad) = 0.9998` -- essentially deterministic. Consistent
  with GRPO's own per-token loss formula (loss scales with `-advantage * logprob`
  roughly uniformly across a row's tokens), so a row's total gradient norm should track
  `|advantage|` almost exactly regardless of which specific tokens contribute.
- `corr(reward, total_abs_grad) = -0.11`, `corr(|reward|, total_abs_grad) = -0.01` --
  raw reward (pre-group-normalization) has essentially no direct relationship; it's
  `|advantage|` (the group-relative, normalized quantity GRPO's loss actually uses)
  that matters, not raw reward magnitude. Reported to make clear which of the two
  "reward" quantities named in the original question is doing the work.

**Partial correlation / regression** (`total_abs_grad ~ intercept + |advantage| +
n_code_tokens`, N=16, df_resid=13):
- `|advantage|` coefficient: t=167.5 (overwhelming).
- `n_code_tokens` coefficient: **coef=-2.4e-05, t=-1.00** -- not significant at df=13
  (critical |t| for p<0.05 two-tailed is ~2.16); the sign is even negative, opposite to
  the raw (unconditional) correlation's positive sign.
- Full model R^2 = 0.9997; `|advantage|` alone already gives R^2 = 0.9997 -- adding
  `n_code_tokens` contributes essentially nothing.
- Equivalently: taking the residual of `total_abs_grad` after removing `|advantage|`'s
  linear effect, its correlation with `n_code_tokens` is **-0.21** -- weak, and the
  wrong sign to explain the original positive r=0.65.

**Verdict, stated plainly per the two named branches from the prior task**:
**reward/advantage magnitude explains the original r=0.65 correlation. Code-token
density is a CONFOUND, not a cause.** The partial correlation of code-token density
with gradient magnitude, controlling for `|advantage|`, is near zero (and
sign-flipped) -- exactly the first named branch ("code usage is a confound, not a
cause"). Rows with more Nib/Nomo tokens happened, in this reward regime and this
checkpoint's rollout distribution, to also be rows with larger-magnitude group-relative
advantage; the code tokens themselves are not doing anything special to the gradient.
This is consistent with, and sharpens, the row-level-vs-token-level finding from the
per-token gradient diagnostic itself (max-grad-token was a code position only 1/24 times,
at/below chance) -- both point away from code tokens as a distinguished target of the
gradient and toward `|advantage|` as the row-level driver.

**Per instruction, no fix is proposed here.** The natural next question this opens --
*why* do certain rows get extreme `|advantage|` values under this reward config against
this SFT-seeded checkpoint's outputs (i.e., what is the reward gate actually reacting
to) -- is flagged as the follow-up direction, not designed or attempted in this task.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-reward-gradient-linkage/`:
- `per_token_gradient_diagnostic_result_v2.json`, SHA256
  `fa579b078bdef156fe8131e7d014796d66a4c22eaff8a314baa2ce5148e263df` (remote/local
  identical).
- `per_token_gradient_diagnostic_stdout_v2.log`, SHA256
  `b0640e5511baefb8308a4278521cd35f3e2c9c907857b6ae4e92f3f56dff003e` (remote/local
  identical).
- `hash_verification_log.txt` -- the full pre-launch gate record, plus the deployed
  script's post-run re-check and both evidence files' remote/local hash comparisons.

**Teardown**: `scripts/gpu_teardown.py` produced a clean **TEARDOWN FULLY CONFIRMED**
banner on the first invocation this time -- the transient network-error-then-retry
pattern from the recovery task did NOT recur (it simply exceeded the foreground
command's 120s timeout during the ~16-iteration stopping poll and was retrieved via the
background task result instead; not a script fault, no retry-loop change to
`gpu_teardown.py` is warranted from a single occurrence of either issue so far).
`g5.xlarge` throughout. No anomalies observed at any point in this run (GPU state,
process state, and instance state all matched expectations throughout) -- nothing
triggered the "stop and report" contingency for unexplained behavior.

## Advantage clamp (2026-08-30): implemented, tested, deployed, and re-run --
## measurably helped but did NOT fully resolve the grad_norm breaker; disclosed
## plainly, not oversold

**Motivation**: the reward/gradient-linkage finding (r=0.9998 between `|advantage|` and
per-row gradient magnitude, code-density confound ruled out) identified advantage
magnitude as the proximate driver of the grad_norm breaker resuming RL on Stage 9c's
SFT-seeded checkpoint. Structurally analogous to Stage 4's per-token KL clamp (a
row/advantage-level instability rather than a per-token/KL-level one) -- same "additive,
surgical, bound the pathological tail without touching the healthy bulk" design
philosophy, not the same numeric formula (see below for why).

**Calibration (task requirement 1), full accounting in `advantage_clamp.py`'s module
docstring, summarized here**: characterized the 16-row linked dataset
(`aws_runs/stage9d-reward-gradient-linkage/`): median=0.8865, q1=0.2610, q3=1.3088,
iqr=1.0478. Stage 4's exact formula (median + 1.5*IQR, *1.5 for the clamp) produces
3.6873 here -- ABOVE the observed max (1.4387), meaning a mechanical transplant of that
formula would never engage. Disclosed as a genuine finding: |advantage| is a bounded,
group-normalized quantity by construction, not the same kind of empirically-unbounded,
heavy-tailed quantity per-token KL is, so the IQR-fence recipe's assumptions don't
transfer.

Derived instead a **theoretical hard bound**, exact and sample-size-independent (Grubbs'
maximum-studentized-residual bound: for any n real numbers, max|x_i-mean|/std <=
(n-1)/sqrt(n); confirmed TRL's advantage formula divides by (group_std+1e-4) >=
group_std, which can only shrink this ratio, so the bound applies to |advantage| itself
unconditionally): for GROUP_SIZE=8, **|advantage| can never mathematically exceed
7/sqrt(8) = 2.4749**. Verified against 20,000 random 8-tuples in `test_advantage_clamp.py`
(none exceeded it) in addition to the closed-form derivation.

Disclosed tension: the two group-steps in the 16-row sample ARE the steps that tripped
the original breaker, yet no single row's |advantage| (max 1.4387) came close to this
theoretical ceiling -- meaning the instability is not about individual rows being
absolute pathological outliers (unlike Stage 4's genuinely-unbounded per-token KL); it's
about the upper end of an inherently compressed distribution still correlating tightly
enough with gradient (r=0.9998) that several such rows accumulating within one 8-row
optimizer step can push the aggregate grad_norm over the breaker. Flagged explicitly as
a reason a per-row clamp might help without being a complete fix -- which the actual
re-run result (below) bore out.

N=16 (2 independent groups) explicitly flagged as too small for trustworthy
percentile/IQR tail calibration (would need ~10+ independent groups, 80+ rows, to
separate population spread from between-group sampling noise -- not obtained here, would
require substantially more GPU steps). **Provisional choice: ADVANTAGE_CLAMP_VALUE=1.2**,
justified by two independent, closely-agreeing anchors: (a) empirically sits between the
observed lower cluster (<=0.8865) and upper cluster (>=1.2952), engaging 6/16 (38%) of
the diagnostic sample including the largest rows from both breaker-triggering steps; (b)
theoretically ~48% of the Grubbs ceiling -- a round, interpretable fraction independent
of small-N curve-fitting. Validation path taken: exactly as instructed, used as a
provisional estimate and validated directly against the actual re-run's telemetry rather
than pursuing further small-sample statistics.

**Implementation (task requirement 2)**: `advantage_clamp.py`, `clamp_advantages()` --
`torch.clamp(advantages, min=-clamp_value, max=clamp_value)`, sign-preserved, hard cap
not smooth rescale (same mechanism as the KL clamp's `torch.clamp` on the ref-logprob
diff). Self-test properties verified in `test_advantage_clamp.py` (11 of the suite's 13
tests): healthy values pass through bit-for-bit unchanged (not just numerically close);
pathological values are capped at the boundary with sign preserved and are explicitly
NOT zero; a downstream toy loss (`-clamped_advantage * logprob`) produces a nonzero,
correctly-signed, magnitude-bounded gradient through `logprob` -- i.e. an
extreme-advantage row still contributes a bounded learning signal rather than being
silently dropped, contrasted directly against the unclamped case (same setup, unbounded
gradient) to confirm the test methodology itself is discriminating. Advantage itself
carries no gradient in GRPO's formulation (a fixed per-row loss weight, not a
differentiable function of the policy), so "gradient sanity" here is about the
downstream weighted-loss gradient, not the clamp function's own backward pass.

**Compatibility (task requirement 3), verified by direct inspection, not assumed**:
- **GRPO's group-relative advantage computation**: confirmed by reading the installed
  TRL source (`trl/trainer/grpo_trainer.py`, `_generate_and_score_completions`) that
  `advantages` is fully computed (group mean/std, `scale_rewards` default `"group"`,
  confirmed this stage's `GRPOConfig` never overrides it) BEFORE `compute_loss` is ever
  called. The clamp is installed as a `GRPOTrainer.compute_loss` patch that mutates
  `inputs['advantages']` in place immediately before calling the original
  `compute_loss` (which reads `inputs["advantages"]` exactly once, at its own top, for
  the policy-gradient term only -- confirmed by reading that function's body directly).
  This means: the clamp never touches the group mean/std computation itself, only
  post-processes the final per-row value before it weights the loss; any TRL-internal
  logging of raw reward/advantage statistics that happens earlier in the pipeline (inside
  `_generate_and_score_completions`) reflects the UNCLAMPED values, which is the correct
  behavior -- only what feeds the actual gradient is modified.
- **Stage 14/14b reward-gate mechanism**: confirmed by direct inspection that
  `sft_seeded_rl.py` imports and uses `score_completion_v2` (`reward_v3.py`, Stage 9's
  own undeclared-condition reward), never `score_completion_gated`/`bank_gate_scale`
  (`step14_reward_gate.py`) -- grep for any reward-gate import in this script returns
  nothing. The reward-gate mechanism is not referenced by this stage's training config
  at all, so there is no interaction to check beyond confirming its absence.
- **KL term / reference-adapter path**: unaffected -- `beta!=0`'s KL computation reads
  `inputs["ref_per_token_logps"]`, a completely separate code path from `advantages`
  (confirmed in the same source read). The pre-existing checkpoint-isolation fix
  (fresh base model per `load_stage9c_checkpoint()` call, adapter-composition assert)
  remains untouched by this task.

**MANDATORY PRE-LAUNCH HASH CHECK (task requirement 6)**: performed before every launch
attempt this task, both files (`sft_seeded_rl.py`, `advantage_clamp.py`) verified
byte-identical remote vs. local before launching -- GATE: PASS both times. (First deploy
+ launch attempt used the correct venv interpreter from the start this time, having
learned from the earlier deployment-failure incident in this stage -- no repeat of that
specific mistake.)

**Re-run result (task requirement 4) -- HONEST RESULT, not fully successful, reported
plainly rather than oversold**: launched the same 8-step-equivalent repro
(`sft_seeded_rl.py`'s baseline 16-step + main 8-step phases, Stage 9c checkpoint, same
config) with the clamp active.

| | terminal_step | requested | hard_stop | clamp engagement | milestone reached |
|---|---|---|---|---|---|
| baseline (correctness-only) | 6 | 16 | grad_norm=55.0 at step 6 | 13/48 (27.1%) | step 4: nonliteral_rate=1.0, overall_acc=0.286 |
| main (full adversarial) | 3 | 8 | grad_norm=58.25 at step 3 | 6/24 (25.0%) | none (broke before step 4) |

**The clamp measurably engaged (25-27% of rows in both phases, matching the diagnostic
sample's own 38% ballpark reasonably given N is now different) and meaningfully delayed
the breaker in the baseline phase (step 6 vs. the 2-3 steps every prior unclamped
attempt on this checkpoint reached) -- baseline also cleared its first real milestone
(step 4) with the seeded Nib/Nomo behavior still at 100% and 28.6% correct, matching
Stage 9c's own reported rate exactly, the first time any Stage 9d run reached this
point. But the grad_norm breaker still fired in BOTH phases, and the main adversarial
phase -- the config that actually tests the target question -- broke at step 3, before
reaching any milestone at all.** This does not meet the task's stated success criterion
("no grad_norm breaker fire... reaches its first real milestone"). Reported as a
partial, informative result, not as success -- per instruction, no further fix is
attempted in this task, and the longer/main adversarial-pressure run is NOT launched.

**What this result plausibly means, stated as interpretation, not certainty**: the
disclosed tension flagged during calibration (no individual row was an absolute
theoretical outlier, yet the breaker still fired) appears to hold under the clamp too --
bounding individual-row magnitude helped (delayed the break, in the baseline case
substantially) but did not fully remove the mechanism, consistent with the instability
being at least partly about the AGGREGATE contribution of several moderately-large-
advantage rows within one 8-row accumulated step, not solely about a few clampable
per-row outliers. Not designed or investigated further in this task, per instruction.

**Regression tests added**: `advantage_clamp.py` (new module) + `test_advantage_clamp.py`
(new, 13 tests: Grubbs-bound derivation and empirical verification, clamp value below
the theoretical ceiling, pass-through/capping/sign/gradient self-test properties, the
naive-Stage-4-formula-non-engagement finding as a permanent record, static source-level
confirmation that `sft_seeded_rl.py` actually wires the patch into `compute_loss`, and a
golden-record test against the real retrieved evidence that honestly asserts what
happened -- clamp engaged, baseline reached step 6 and its milestone, breaker still
fired in both phases -- rather than a wished-for "no breaker" assertion). Full existing
suite (26 tests across this directory) passes, no regressions.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-advantage-clamp-v1/`:
- `stage9d_sft_seeded_adversarial_rl.json`, SHA256
  `fd9c6614a056ca808781c2b61f8774f73ae022c087e446770969a7b4e9489c48` (remote/local
  identical).
- `sft_seeded_rl_stdout.log`, SHA256
  `ec78848b0e28e8e26200634bdcde3f3395818b16f89aa80c321016362ae1e2df` (remote/local
  identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same as the
prior task -- retrieved via the background task result, not a script fault). `g5.xlarge`
throughout. One transient local-network event during this task: the operator's own
public IP changed mid-run, causing SSH to time out; confirmed via a direct
`describe-instances` call (independent of SSH) that the INSTANCE itself was still
healthy and running throughout -- not a repeat of the earlier unexplained-stop incident,
just an ordinary dynamic-IP change requiring a fresh security-group authorization,
resolved by re-authorizing the new IP. Both session security-group rules revoked at
teardown.

**Status: this task's stated goal (a clean 8-step dry run under the clamp) was not
reached. Waiting for explicit direction before any further iteration on the clamp value
or mechanism, per instruction -- not proceeding to the longer/main run.**

## Gradient-norm clipping (2026-08-30): task premise checked BEFORE changing anything
## (per its own instruction 1) -- found to be structurally incapable of affecting this
## breaker; no GPU run performed, since the proposed intervention is provably a no-op
## on the exact symptom it was meant to fix

**Task requirement 1 (check before changing anything), answered by source inspection +
empirical proof, not assumption**: gradient-norm clipping is **already active**, at
`max_grad_norm=1.0` (`transformers.TrainingArguments`'s own default -- `GRPOConfig`
never overrides it, and `sft_seeded_rl.py`'s `GRPOConfig` call never sets it either, so
every run in this stage, including all prior diagnostics, has been training under
norm-1.0 clipping this entire time, previously undisclosed because it had never been
checked). This is already far TIGHTER than the 10-15 range the task proposed, and far
tighter than `GRAD_BREAKER=50.0`.

**Critical finding that changes the plan, established with three independent layers of
evidence rather than asserted**:

1. **Source (transformers)**: `Trainer._clip_grad_norm()`'s own docstring: "Clip
   gradients to max_grad_norm. **Returns the pre-clip gradient norm.**"
   (`transformers/trainer.py`). The training loop calls this once per optimizer step,
   and `_maybe_log_save_evaluate` writes that SAME returned value directly into
   `logs["grad_norm"]` (`transformers/trainer.py` line ~2087) -- the exact key Stage
   9d's own breaker (`Cb.on_log`, `logs.get('grad_norm', 0)`) reads.
2. **Source (accelerate)**: `Accelerator.clip_grad_norm_()` (what `_clip_grad_norm`
   calls) falls through, for this project's plain single-GPU setup (no FSDP/DeepSpeed/
   XLA), to `torch.nn.utils.clip_grad_norm_` -- whose own well-documented behavior is to
   rescale `.grad` tensors in place when the norm exceeds `max_norm`, but always RETURN
   the pre-rescale norm.
3. **Empirical (this task, CPU-only, no mocks)**: directly called
   `torch.nn.utils.clip_grad_norm_` on a real gradient tensor (fixed norm=10.0) across
   `max_norm` in `{0.1, 1.0, 10.0, 15.0, 50.0, inf}` -- the RETURNED value was 10.0 in
   every case (confirming the logged/breaker-checked value is invariant to
   `max_grad_norm`), while the ACTUAL post-clip `.grad` norm correctly varied with
   `max_norm` as expected (confirming clipping itself works normally -- it's the
   RETURN/LOG value that's invariant, not the mechanism itself being broken).
   Also confirmed directly: the installed `GRPOTrainer`'s source contains zero
   references to `grad_norm` anywhere (it doesn't override the computation or logging
   TRL inherits from base `Trainer`; its own `log()` only merges in reward/KL metrics
   and calls `super().log()` unchanged) -- ruling out a TRL-specific override that might
   have changed this behavior.

**Conclusion: setting `max_grad_norm` to ANY value -- including the 10-15 range this
task proposed -- cannot change whether the grad_norm breaker fires, because the breaker
reads the pre-clip norm, which is mathematically invariant to `max_grad_norm` by how
`torch.nn.utils.clip_grad_norm_` is defined.** This is not a matter of picking the wrong
clip value; it is a structural mismatch between what the tool controls (the actual
gradient tensors used in the optimizer step) and what the breaker measures (the
magnitude those tensors WOULD have had before any clipping). Ordinary optimizer-step
gradient clipping is the wrong tool for this specific symptom, as implemented by this
trainer stack -- contrary to the task's own framing of it as "the more direct tool for
exactly this symptom."

**Second-order consideration, addressed rather than left implicit**: could a smaller
`max_grad_norm` still help INDIRECTLY, by producing smaller actual parameter updates
that change the training trajectory over subsequent steps, even without touching what
gets logged/checked at the SAME step? The existing data argues against this too: every
run in this stage (including the advantage-clamp re-run that still hit the breaker at
steps 3 and 6) has ALREADY been training under the tightest practical clip
(`max_grad_norm=1.0`, the active default) the whole time -- and the breaker still fired.
Raising `max_grad_norm` to the task's proposed 10-15 would make ACTUAL updates LARGER
(less conservative) than what's already happening, which is a step in the wrong
direction if anything, not a fix. Lowering it further below 1.0 was not requested and
was not tested here (not the task's proposed direction, and would be a materially
different experiment from "add clipping" given clipping is already active).

**Task requirement 2 (enable/set clipping below the breaker) -- not performed**: doing
so would not achieve the stated goal (per the proof above) and no configuration change
was made to `GRPOConfig`'s `max_grad_norm`. Per this stage's own established discipline
(and requirement 1's own instruction to "report what you find before changing
anything"), proceeding to relaunch a GPU run under a change already shown incapable of
affecting the measured outcome was judged not to be a responsible use of GPU time --
this is reported as a finding that invalidates the task's premise, not worked around
silently or pushed through mechanically.

**Task requirements 3-5 (mask-vs-fix distinction, re-run, honest reporting if
unresolved)**: requirement 3's own concern -- "confirm this doesn't silently mask a
genuinely pathological training state" -- is moot here since no masking mechanism was
actually engaged (the breaker's behavior is provably unchanged), but the underlying
worry it points at is directly relevant to what WAS found: this checkpoint has been
training under norm-1.0 clipping (about as tight as ordinary gradient clipping gets)
this entire time and STILL produces a pre-clip grad_norm of 50-65 by steps 3-6 -- i.e.
the underlying advantage-driven instability is real and already "contained" at the
actual-update level by the existing default clip, yet still large enough at the
pre-clip/logged level to trip the breaker. No re-run was performed (requirement 4) since
there is no configuration change to test. Reporting honestly rather than iterating
further without direction (requirement 5): **gradient-norm clipping, as this trainer
stack implements it, is not a viable lever for this specific breaker.** A tighter
advantage-clamp value remains the most promising untested lever from the prior task
(flagged there, not tested); a DIFFERENT approach -- e.g. changing what the breaker
itself measures (post-clip rather than pre-clip norm) -- would need explicit direction
first, since it is a change to the breaker's own semantics, which this stage has
consistently treated as off-limits without an explicit decision (risk of masking the
underlying instability rather than fixing it, exactly requirement 3's own concern).

**Task requirement 6 (hash check, regression tests, evidence, teardown)**: no GPU
instance was started for this task -- the finding above was established entirely via
CPU-only source inspection and empirical proof against the locally-installed
transformers/accelerate/torch/trl packages (the same versions pinned throughout this
stage), so no deploy/launch/hash-check/teardown cycle was applicable. Regression tests
added instead: `test_grad_norm_clipping_semantics.py` (4 tests) -- proves the
invariance claim directly against real `torch.nn.utils.clip_grad_norm_` calls (not
mocked), confirms clipping DOES correctly rescale the actual gradient tensors (so the
test isn't trivially true because clipping does nothing), statically guards that the
installed `GRPOTrainer` still has no `grad_norm` override (so a future TRL upgrade that
changes this is caught rather than silently invalidating this conclusion), and confirms
the real installed `GRPOConfig`'s default `max_grad_norm` is exactly 1.0. Full existing
suite (30 tests across this directory) passes, no regressions.

**Status: task premise falsified before any GPU action was taken, per requirement 1's
own instruction to check first. No configuration change made, no run launched. Waiting
for direction: whether to test a tighter advantage-clamp value next, or to consider
changing the breaker's own semantics (post-clip norm) as an explicit, separate
decision -- not proceeding on either without direction, and not proceeding to the
longer/main run regardless.**

## Tighter advantage-clamp search (2026-08-30): CLEAN PASS found at clamp=0.4, with a
## disclosed caveat -- it is no longer a genuine-tail clamp at this value

**Candidates (task requirement 1), derived not picked, same reasoning style as the
original 1.2 choice** (fraction of the Grubbs ceiling 2.4749 + fraction of the 16-row
diagnostic sample that would be clamped):

| clamp | % of Grubbs ceiling | engaged/16 (diagnostic sample) |
|---|---|---|
| 1.2 (already tested, insufficient) | 48.5% | 6/16 (37.5%) |
| 0.7 | 28.3% | 10/16 (62.5%) -- sits centrally in the wide empirical gap between the sample's low cluster (<=0.3924) and high cluster (>=0.7702) |
| 0.4 | 16.2% | 10/16 (62.5%), same rows as 0.7 but clamped to a smaller final magnitude -- a strictly more aggressive intervention despite an identical engagement RATE, flagged explicitly since rate alone doesn't capture aggressiveness |
| 0.25 (outer edge, not needed) | 10.1% | 12/16 (75.0%) -- would clamp the clear majority |

Full derivation and reasoning recorded in `advantage_clamp.py`'s module docstring
addendum. Tested in order 0.7 -> 0.4, stopping as soon as a clean pass was found
(0.25 not needed).

**Candidate 0.7 -- MIXED, not a clean pass**: baseline REGRESSED (`hard_stop` at step 3,
`grad_norm=57.25` -- worse than 1.2's step-6 survival, no milestone reached at all this
time) while main IMPROVED (reached step 5, `grad_norm=56.5`, and cleared its first-ever
milestone at step 4 -- 100% nonliteral, 28.6% correct -- before breaking; the first time
the adversarial config itself has ever reached a milestone in this stage). Engagement
52.1% (baseline) / 52.5% (main) -- close to the 62.5% predicted from the static
diagnostic sample. This divergent baseline-vs-main outcome is consistent with this
project's already-documented run-to-run GPU/CUDA nondeterminism (independently sampled
rollouts each launch) rather than evidence the clamp itself is harmful -- not
re-investigated further since 0.7 was superseded by testing 0.4 next, per instruction
not to linger on ambiguous intermediate results.

**Candidate 0.4 -- CLEAN PASS.** Both phases completed in full: baseline `terminal_step
=16/16 requested`, `hard_stop=None`, all 4 milestones cleared (steps 4, 8, 12, 16, each
100% nonliteral / 28.6% correct, consistent throughout); main `terminal_step=8/8
requested`, `hard_stop=None`, both milestones cleared (steps 4, 8). Max `grad_norm`
observed across both phases: 38.0 (comfortably under `GRAD_BREAKER=50.0`). KL stayed in
its normal healthy range throughout (0.0-0.36 across both phases, nowhere near
`KL_BREAKER=5.0`). This is the first time in Stage 9d that BOTH configs have completed a
dry run cleanly -- the actual experimental question (does the seeded non-literal
behavior survive real adversarial RL pressure) is now testable for the first time.

**Task requirement 3 (check whether the clamp is now doing the KL/entropy
coefficient's job) -- DISCLOSED, not glossed over: YES, at 0.4 it is.** Engagement rate:
**78.1% (baseline, 100/128 rows) and 70.3% (main, 45/64 rows)** -- the clear majority of
ALL rows, not a genuine tail. This is a materially different kind of intervention than
what "clamp" was originally meant to evoke (Stage 4's per-token KL clamp engaged on
~21% of tokens; this stage's own 1.2 value engaged on 25-38%): at 0.4, the mechanism is
functioning much closer to a blanket downscale of advantage magnitude across nearly the
whole distribution than a surgical outlier-bound. **Adopted as the new default anyway,
because it is the value that actually produces a clean pass and the task's stated goal
was finding that value -- but this distinction is recorded explicitly here and in
`advantage_clamp.py`'s own docstring so it is never silently treated as "the same kind
of intervention, just tuned tighter."** Whether this degree of intervention is
acceptable for the actual experimental question Stage 9d is trying to answer is a
judgment call outside this task's scope -- flagged for the next decision point, not
decided here.

**Task requirement 5 (if no clean pass, report and flag the breaker-semantics
question)**: not applicable -- a clean pass WAS found at 0.4, so per-row clamping alone
(now understood to be operating as a near-blanket downscale at this value, not a tail
clamp) is sufficient to clear the dry-run bar. The breaker-semantics question (post-clip
vs. pre-clip norm) from the prior task remains unexplored and still off-limits without
separate sign-off -- not newly warranted by this result, since a working fix was found
without needing it.

**MANDATORY PRE-LAUNCH HASH CHECK (task requirement 5/6)**: performed before both
candidate launches (0.7, then 0.4) -- `sft_seeded_rl.py` and `advantage_clamp.py`
verified byte-identical remote vs. local each time. GATE: PASS on both. (`advantage_clamp.py`
was updated once, between the 1.2-run task and this one, to read the clamp value from
`STAGE9D_ADVANTAGE_CLAMP_VALUE` with a fallback default -- deployed once and re-verified
before each of the two candidate launches, since the file itself doesn't change between
candidates, only the environment variable at launch time. The chosen value is also
printed in each run's own startup config output for cross-verification against what was
intended -- confirmed `advantage_clamp_value: 0.7` and `advantage_clamp_value: 0.4` in
the respective logs, matching the intended launch.)

**Regression tests added/updated**: `advantage_clamp.py`'s default changed from 1.2 to
0.4 (documented candidate-search outcome in the module docstring). `test_advantage_clamp.py`
gained: a test confirming the new adopted default (0.4) so a future silent revert is
caught; a golden-record test against the 0.7 evidence (mixed result: baseline regressed,
main improved, both asserted precisely); a golden-record test against the 0.4 evidence
(clean pass AND the 78%/70% high-engagement caveat, both asserted so the caveat can't
quietly disappear from the record while still calling this a clean pass). Full suite
(33 tests across this directory) passes, no regressions.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/`:
- `stage9d-advantage-clamp-0.7/stage9d_sft_seeded_adversarial_rl.json`, SHA256
  `9995f9a7513f283734ef2f2457a57db38072424b7c95167731bba67563e1f86f`;
  `sft_seeded_rl_stdout.log`, SHA256
  `870e90225639c60d38b8c88fa8e098214c62b5f6d22a294675d21fc698cddf08` (both remote/local
  identical).
- `stage9d-advantage-clamp-0.4/stage9d_sft_seeded_adversarial_rl.json`, SHA256
  `3497fe7a64000867935317b16b5d238116038f926fca4fedb57990d7e78b0ca9`;
  `sft_seeded_rl_stdout.log`, SHA256
  `8895d41e40c8e4414c05087dac09959d8a9d3c6338d0e76563187e4b683d7c4e` (both remote/local
  identical).

**Teardown**: one AWS SSO credential expiry mid-task (same recurring pattern documented
elsewhere in this project) -- paused, had the user run `aws sso login --profile research`,
re-verified via `aws sts get-caller-identity` before retrying rather than proceeding on
a stale session. `scripts/gpu_teardown.py` then produced a clean **TEARDOWN FULLY
CONFIRMED** banner (exceeded the 120s foreground timeout during the ~16-iteration
stopping poll, same as prior tasks -- retrieved via the background task result, not a
script fault). `g5.xlarge` throughout. Both candidates (0.7, then 0.4) were run within
one continuous GPU session with a single teardown at the end, rather than tearing down
and restarting between each candidate -- an explicit interpretation of "confirm teardown
after each GPU-touching run" as applying to this one continuous candidate-search
session, not each individual sub-launch; disclosed here rather than left implicit.
Session security-group SSH rule revoked at teardown.

**Status: clean pass found at clamp=0.4 -- both baseline and main dry runs complete
with no breaker fire, all milestones cleared, KL healthy. Adopted as the new default.
Caveat disclosed and asserted in tests: this value engages on the majority of rows, not
a genuine tail -- a materially different kind of intervention than originally framed.
Not proceeding to the main adversarial-pressure run without further direction, per
instruction.**

## Pre-commitment sanity checks (2026-08-30): signal preservation at 0.4, and whether a
## per-config clamp split is warranted -- both checks resolved without an open-ended
## search, per instruction

**Instrumentation added first** (`sft_seeded_rl.py`): `compute_loss`'s existing clamp
patch already computes both the pre-clamp and post-clamp advantage per row in memory --
persisted both into `PER_ROW_CAPTURES` (surfaced per-phase as `per_row_advantages` in
the saved report), alongside a per-phase `clamp_value` override mechanism
(`STAGE9D_ADVANTAGE_CLAMP_BASELINE`/`STAGE9D_ADVANTAGE_CLAMP_MAIN`, both falling back to
the shared default) so a per-config split could be tested from the same file without
editing it, if warranted. Pre-launch hash gate passed (both files verified
byte-identical remote vs. local before the launch below).

**Single instrumented re-run at clamp=0.4** (both phases; a per-phase override wasn't
needed yet -- see CHECK 2 below for why a split was never actually launched) served
both checks from one dataset, since PRE-clamp advantage is clamp-VALUE-invariant
(clamping happens strictly after computation) -- disclosed explicitly: this substitutes
a fresh run's raw advantage distributions for "the actual 0.7 rollouts" the task
referenced, on the grounds that the STRUCTURAL question (does main's reward function
shape produce a different advantage distribution than baseline's) is a property of the
reward functions and GRPO's own normalization, not of which specific clamp value or
random draw was in effect -- reasoned through explicitly below rather than assumed.
Second independent clean pass confirmed: baseline 16/16 steps, `hard_stop=None`; main
8/8 steps, `hard_stop=None`; both milestone sets cleared -- reassuring that the earlier
0.4 clean pass wasn't a one-off fluke of run-to-run GPU nondeterminism.

### CHECK 1 -- does 0.4 preserve meaningful signal?

**Context established first, without new GPU work, from EXISTING evidence**: compared
GRPO's effective per-step update scale against Stage 9c's own successful SFT dynamics
(`experiments/09c_sft_diagnostic/aws_runs/stage9c-sft-diagnostic-v1/`). Two structural
facts, both confirmed by direct source/config inspection, not assumed:
- Stage 9c's SFT used a MANUAL training loop with `clip_grad_norm_(..., max_norm=1e6)`
  -- effectively unclipped (1e6 is never approached) -- so its logged grad_norm
  (~0.7-2.6, mostly ~1.0 in steady state) is the value that ACTUALLY drove parameter
  updates, at `TARGET_LR=2e-4`.
- Stage 9d's GRPO runs use HF's `Trainer`/`GRPOTrainer` with the untouched
  `max_grad_norm=1.0` default (the same fact established in the gradient-clipping task)
  at `TARGET_LR=1e-6`. Confirmed directly in this run's own telemetry: the MINIMUM
  logged (pre-clip) grad_norm across every step of both phases was 4.75 -- meaning
  EVERY step's actual applied gradient was clipped down to norm=1.0 by the always-active
  default, unconditionally, this entire run.
- Effective per-step magnitude proxy (LR x actually-applied grad_norm):
  SFT = 2e-4 x ~1.0 = 2e-4. GRPO = 1e-6 x 1.0 (post-clip, always) = 1e-6.
  **Ratio: 200x.** This gap is driven entirely by the deliberate LR choice (by design,
  a conservative resumed-RL rate) and the ALREADY-ACTIVE default gradient clip -- NOT by
  the advantage clamp, which cannot affect this ratio at all (post-clip magnitude is
  externally capped to 1.0 regardless of the clamp value, as long as pre-clip exceeds
  1.0, which it did on every single observed step across every candidate tested in this
  stage).

**Quantified dampening at clamp=0.4, from the new per-row data (128 baseline rows, 64
main rows)**:

| | baseline | main |
|---|---|---|
| mean \|pre-clamp advantage\| | 0.8154 | 0.8263 |
| mean \|post-clamp advantage\| | 0.3555 | 0.3558 |
| sum \|pre-clamp advantage\| | 104.38 | 52.89 |
| sum \|post-clamp advantage\| | 45.50 | 22.77 |
| **total signal removed** | **56.4%** | **56.9%** |
| unengaged rows (n, mean \|advantage\|) | 27, 0.189 | 12, 0.165 |
| engaged rows (n, mean pre-clamp, mean removed) | 101, 0.983, 0.583 | 52, 0.979, 0.579 |

**Roughly 56% of the total raw advantage-magnitude signal is being removed by the
clamp at 0.4, in both phases.** This is NOT surgical trimming of a genuine tail: the
UNENGAGED rows' own mean magnitude (0.19) confirms the "normal" population itself
centers well below 0.4 relative to the engaged population, but the ENGAGED population
(79-81% of all rows) has typical pre-clamp magnitude ~0.98 -- close to double the clamp
value -- meaning the clamp is cutting deep into what is, for this checkpoint and reward
config, an ORDINARY (not pathological) range of advantage values.

**Plain verdict (task requirement 3)**: two separate effects need to be kept distinct,
not conflated into one "how dampened is it" number:
- **Update MAGNITUDE**: essentially unaffected by the clamp value specifically --
  already externally capped to norm=1.0 by the ever-present default `max_grad_norm=1.0`
  and scaled down further by the deliberately tiny LR (1e-6, ~200x smaller than what
  drove Stage 9c's SFT convergence). This 200x gap exists with or without the advantage
  clamp, is far larger than anything the clamp itself contributes, and was already true
  at EVERY candidate value tested in this stage (0.4, 0.7, 1.2), since pre-clip grad_norm
  exceeded 1.0 on every single logged step regardless of clamp value.
  Given this, if the real experiment fixes the model's own SFT-conservative
  update scale (LR unchanged from what's already tested), a SHORT run (dry-run scale,
  8-16 steps) is unlikely to show large behavior change under ANY of the tested clamp
  values, simply because the per-step update ceiling is tiny by design -- this is a
  property of the LR/clipping choice, not something the advantage-clamp search can fix
  or worsen.
- **Update DIRECTION/composition**: genuinely affected by the clamp. At 0.4, ~56% of
  the raw signal that would otherwise differentiate which rows/tokens matter most is
  compressed before the final norm-1.0 rescaling -- the resulting update direction is
  measurably more "flattened" across rows than the natural advantage-weighted direction
  would be. This is real, disclosed dampening, separate from the (externally-bounded,
  clamp-independent) magnitude question above.
- **Net**: **0.4 is not likely to be the dominant bottleneck on whether the real
  experiment can show real learning** -- the 200x LR/clip gap already sets a much
  tighter ceiling on per-step movement than the clamp's directional compression adds.
  But the clamp's directional dampening IS real (56% signal removed) and compounds a
  system that was already conservative by design. **Recommendation, stated plainly**: a
  SHORT real run (dry-run-scale step counts) would produce a genuinely low-confidence
  result regardless of outcome -- a null result (seeded behavior survives) could mean
  either "the behavior is robust" or "training simply hasn't moved yet," and these are
  not distinguishable at this scale given how conservative the whole pipeline already
  is before the clamp is even considered. A LONGER run (accumulating many more small
  steps, e.g. Stage 9b's own ~50-step scale) would let small per-step movements compound
  into an interpretable signal and is the more trustworthy path -- not decided or
  launched here, flagged for the next decision point.

### CHECK 2 -- does a per-config clamp value resolve this more surgically?

**Task requirement 1 (why did 0.7 split baseline/main) -- re-examined with real
per-row data, not re-derived from memory.** The raw (pre-clamp, clamp-invariant)
advantage distributions for baseline and main, from THIS run's 192 total rows, are:

| | baseline (n=128) | main (n=64) |
|---|---|---|
| mean | 0.8154 | 0.8263 |
| median | 0.8108 | 0.8117 |
| stdev | 0.4601 | 0.4418 |
| max | 2.1836 | 2.0060 |

**These are essentially statistically indistinguishable.** The hypothesis motivating
this check -- that main's CoT-penalty term produces a systematically different, less
extreme advantage distribution than baseline's correctness-only reward -- is **not
supported by this data**. A mechanistic reason this makes sense, not just an empirical
coincidence: GRPO's advantage is a WITHIN-GROUP z-score,
`(reward - group_mean)/(group_std + 1e-4)`, computed relative to each group's OWN
mean/std regardless of what the underlying reward function measures. This normalization
inherently washes out most of the raw-reward-SCALE differences between different reward
functions (correctness-only vs. full adversarial-with-CoT-penalty) -- both configs'
advantage values are constrained to the same z-score-shaped, Grubbs-bounded range by
construction, independent of the reward function's own content. Both configs' engaged
rows also topped out well below the 2.4749 theoretical ceiling and close to each other
(2.18 vs 2.01) -- no asymmetry in extremity either.

**Task requirement 2 (test whether baseline needs tighter, main could use looser) --
NOT tested, with reasoning stated rather than skipped silently.** The instrumentation to
test a per-config split (`STAGE9D_ADVANTAGE_CLAMP_BASELINE`/`_MAIN`) was built and is
ready to use, but no launch was made: requirement 1's finding removes the principled
basis for expecting a specific split value to help. Testing an arbitrary split (e.g.
baseline=0.5, main=0.7) without a data-grounded reason to expect it would differ
meaningfully from testing 0.4 vs. 0.7 again -- indistinguishable from blind
trial-and-error, exactly the "expensive/circular" iteration the task itself asked to be
reported plainly rather than pursued. **The earlier 0.7 split (baseline broke at step 3,
main reached step 5) is better explained by this project's already-documented run-to-run
GPU/CUDA nondeterminism (independently sampled rollouts each launch) than by a genuine,
reproducible structural asymmetry between the two configs' advantage distributions.**

**Task requirement 3 (report whether a split improves on 0.4) -- no split was tested,
so no improvement claim is made.** Per instruction, reporting this plainly rather than
launching a third exploratory run in search of one: **a single shared clamp value
(0.4) is recommended over a per-config split**, since the data available does not
support expecting a split to help, and testing one anyway would consume GPU time on a
search with no principled stopping point.

### Verdicts, stated plainly

- **CHECK 1**: 0.4 removes ~56% of the raw advantage signal by magnitude, but the
  DOMINANT constraint on real learning in this whole setup is the deliberately
  conservative LR (200x below what drove Stage 9c's SFT) combined with the ALREADY-ACTIVE
  default gradient clip, both independent of the advantage clamp. 0.4 is unlikely to be
  the binding bottleneck, but a SHORT real run would still be low-confidence regardless
  of outcome given how conservative the pipeline already is -- a longer run is the more
  trustworthy path for the eventual real experiment.
- **CHECK 2**: no principled basis found for a per-config clamp split -- baseline's and
  main's raw advantage distributions are nearly identical. Recommend proceeding with the
  single shared value (0.4), not a split.

**Mandatory pre-launch hash check**: performed before the single new launch this task --
`sft_seeded_rl.py` and `advantage_clamp.py` verified byte-identical remote vs. local.
GATE: PASS.

**Regression tests added**: `test_advantage_clamp.py` gained a static wiring test
(confirms `PER_ROW_CAPTURES.append(...)`, the per-phase `CURRENT_CLAMP_VALUE[0]`
assignment, and both `clamp_value=BASELINE_CLAMP_VALUE`/`clamp_value=MAIN_CLAMP_VALUE`
call sites are present), a golden-record test against the new evidence (second clean
pass, per-row data present, signal-removed fraction in the observed 40-70% band), and a
golden-record test asserting baseline's and main's raw advantage means/medians are
within 0.1 of each other (the CHECK 2 finding). Full suite (36 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-signal-preservation-0.4/`:
`stage9d_sft_seeded_adversarial_rl.json`, SHA256
`ebf611ed6da7b0aacdd5e4d15604c7e40c0c6110026ab422153ac75b781ed75b`; `sft_seeded_rl_stdout.log`,
SHA256 `daa15283567d7062e218c34af30425b772221bf2179784ec087a9ec37b2a3904` (both
remote/local identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same pattern as
prior tasks -- retrieved via the background task result, not a script fault).
`g5.xlarge` throughout. Session SSH rule revoked at teardown. One long silent stdout
stretch (~15-20 minutes with no flushed output before the process completed) matched
the same buffering pattern already observed and explained in the prior candidate-search
task -- not treated as anomalous a second time.

**Status: both checks resolved. Recommendation: proceed with the single shared
clamp=0.4 (not a per-config split), understanding that a SHORT real adversarial-pressure
run would be low-confidence regardless of outcome -- a longer run is recommended when
that experiment is launched. Not proceeding to the main adversarial-pressure run in this
task, per instruction.**

## FULL-SCALE EXPERIMENT, BASELINE PHASE (2026-08-30): the actual real-scale run this
## stage exists to enable -- technically clean, but the completions did not move at all,
## confirming the low-confidence risk flagged in the prior signal-preservation check

**New script**: `sft_seeded_rl_full.py`, adapted from `sft_seeded_rl.py` (left
untouched, still the dry-run regression-test source of truth). Changes: `FULL_STEPS=50`
(Stage 9b's own full-run precedent), `MILESTONE_EVERY=5` (10 milestones), a single
phase per launch gated by `STAGE9D_FULL_PHASE` (`baseline`/`main`, `RuntimeError` if
unset or invalid) so baseline could be run, retrieved, and reported before main is even
considered, milestone-to-milestone delta flagging (`>0.05` absolute change in
`nonliteral_rate` prints a loud `*** FLAG ***` line immediately, not deferred),
first-genuine-nonliteral+correct flagging (prints full completion text the moment it's
seen), token-drift flagging, and per-milestone-window advantage-clamp engagement
tracking (not just one aggregate number for the whole run). The advantage clamp itself
(0.4) and the checkpoint-isolation fix are reused unchanged from the calibrated,
validated mechanism.

**Task requirement 1 (checkpoint identity)**: confirmed BEFORE launch by direct
inspection, not assumed -- `sha256sum` on the remote instance's
`stage9c-sft-diagnostic-v1/final_adapter/adapter_model.safetensors` returned
`c84afe0389487766463b74aba3559940b2e92d69c4915d32a5dee1c5ccc7609c`, matching the value
hardcoded as `EXPECTED_STAGE9C_ADAPTER_SHA256` and asserted in the script itself before
any training starts (in addition to the pre-existing zero-step sanity generation check,
which also passed: `sanity_nonliteral_rate=1.0`).

**Task requirement 3 (reward invariant)**: `verify_reward_invariant()` is a pure,
symbolic/combinatorial check over `score_completion_v2`'s own term ranges --
checkpoint-independent by construction, and it re-runs FRESH on every single script
launch (never a cached/stale result). Confirmed structurally non-interacting with the
advantage clamp: the invariant checks properties of the REWARD (`score_completion_v2`'s
output), while the clamp operates strictly downstream on GRPO's post-hoc advantage
(reward -> group z-score -> clamp), inside `compute_loss` -- no shared state or code
path between the two. Re-run explicitly for this launch (`margin_correct_over_wrong=3.0,
margin_wrong_over_malformed=3.0`, both positive) rather than treated as still valid from
an earlier task without re-checking.

**Mandatory pre-launch hash check**: `sft_seeded_rl_full.py` and `advantage_clamp.py`
verified byte-identical remote vs. local. GATE: PASS. Launch was killed and restarted
once, deliberately, before doing meaningful GPU work: the first launch used buffered
stdout (same pattern as prior tasks, which resolved fine on its own, but this task's
explicit "flag immediately" requirement warranted `python3 -u` for true real-time
output instead of tolerating the multi-minute buffering delay again) -- killed at ~10s
elapsed (0 MiB GPU memory freed cleanly) and relaunched unbuffered.

**Result: TECHNICALLY a clean pass -- but the completions did not move.** `terminal_step
=50/50`, `hard_stop=None`. All 10 milestones (steps 5-50) report IDENTICAL aggregate
stats: `nonliteral_rate=1.0`, `structural_nonliteral_candidate_rate=1.0`,
`genuine_correct_among_structural_nonliteral=0.2857` (=2/7, exactly Stage 9c's own
starting rate), `overall_acc=0.286`, token identity staying on Nib/Nomo (occasional
`position_driven_drift` taxonomy flag on 1/21 samples at some milestones, never
accumulating). `grad_norm` ranged 6.9-45.25 (comfortably under `GRAD_BREAKER=50.0`
throughout), KL ranged 0.0-0.31 (comfortably under `KL_BREAKER=5.0`). Advantage-clamp
engagement stayed in the 52.5%-85% band across milestone windows, consistent with
dry-run-scale levels, not blowing up or collapsing at 50-step scale.

**The disambiguating check, run because the aggregate stats alone are ambiguous
(identical stats could mean "genuinely robust" OR "nothing moved"), per this project's
established discipline of tracing a surprising/too-clean result to the underlying
mechanism rather than trusting the aggregate**: compared the actual eval completion
TEXT (not just aggregate stats) between consecutive milestones and between the first
(step 5) and last (step 50) milestone. **Step 5 and step 50's 21 held-out eval
completions are BYTE-IDENTICAL, 21/21.** Consecutive milestones throughout the run show
20-21/21 completions byte-identical each time (the rare single differing sample does
not accumulate into a trend -- consistent with oscillation near a decision boundary,
not genuine drift). **The model's greedy-decoded output on this exact eval set has not
changed AT ALL across 45 additional training steps.**

**This directly confirms, empirically, the low-confidence risk flagged in the prior
signal-preservation check (CHECK 1) -- not a new problem, the PREDICTED one, now
observed**: that check established GRPO's effective per-step update scale
(LR=1e-6 x post-clip grad_norm, always clipped to 1.0 by the ever-active
`max_grad_norm=1.0` default) is ~200x smaller than what drove Stage 9c's SFT
convergence (LR=2e-4, effectively unclipped). Even optimistically assuming all 50
steps' updates summed CONSTRUCTIVELY in the same direction (an upper bound, not
realistic under stochastic rollouts), 50 x 1e-6 = 5e-5 of cumulative raw-magnitude
movement is still ~4x SMALLER than a SINGLE STEP of Stage 9c's own SFT update
(2e-4) -- so a fully frozen-looking eval output after 50 steps is exactly what the
already-established math predicted, not a surprise.

**Plain verdict: this baseline result cannot be read as "ordinary RL alone does not
erode the seeded behavior."** It can only be read as "at this LR/clip/clamp
configuration, 50 steps did not move the policy enough to test the question at all" --
the aggregate stats being identical across every milestone is NOT evidence of stability
under pressure; it is evidence of a policy that, by the eval set's own greedy-decoded
output, did not change. **This is exactly the low-confidence outcome CHECK 1 predicted
would happen at this configuration, now confirmed rather than hypothetical.** Reported
plainly, per instruction 7, to inform how MAIN should be interpreted (or whether it
should be launched at this same configuration at all) -- not decided in this task.

**Regression tests added**: `test_sft_seeded_rl_full.py` (new, 11 static wiring tests +
1 golden-record test) -- confirms single-phase gating, 50-step/5-milestone cadence,
checkpoint SHA assertion, reused checkpoint-isolation fix, unchanged advantage-clamp
wiring, breaker-stops-with-no-automatic-retry behavior, all three flagging mechanisms,
windowed clamp-engagement tracking, and the reward-invariant re-verification's correct
kwarg set (guards against the `cot_min_scale` `TypeError` this task's own draft hit and
fixed before ever reaching the GPU). The golden-record test asserts the clean-pass
basics AND the byte-identical-completions finding, so that finding can't quietly
disappear from the record while still calling this a "clean pass." Full suite (48
tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-full-baseline-v1/`:
`stage9d_full_baseline.json`, SHA256
`0cac77fdbe71dc522dfd040282905cd41f3708bc1c970aac6ebba536d54ff83c`;
`sft_seeded_rl_full_baseline_stdout.log`, SHA256
`580034e6d02732ac29780e72bcfa77ab49394d912a5e9ee6622b28f9667f666b` (both remote/local
identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same pattern as
every prior task in this stage -- retrieved via the background task result, not a
script fault). `g5.xlarge` throughout. Session SSH rule revoked at teardown.

**Status: baseline phase complete, evidence retrieved, teardown confirmed. STOPPING
here per instruction 7 -- not proceeding to MAIN automatically.** The baseline result
changes how MAIN should be approached: since baseline (the simpler, no-CoT-penalty
config) already shows this LR/clip/clamp combination does not move the policy over 50
steps, MAIN (which adds the CoT penalty on top of the same conservative pipeline) is
unlikely to show more movement, not less -- a MAIN run at this identical configuration
would very likely reproduce the same "technically clean, but frozen" outcome, which
would NOT be informative about whether adversarial pressure specifically threatens the
seeded behavior. This is reported as a finding for the user to weigh, not a decision
made here: options include (a) running MAIN anyway at the same configuration for a
direct, symmetric comparison against this baseline (even a frozen-both-ways result is
informative as a matched pair), (b) reconsidering the LR/step-count/clamp combination
before running MAIN so the experiment can actually move the policy, or (c) something
else -- not chosen in this task, flagged for the next decision point.

## Configuration redesign + short diagnostic (2026-08-30): the ~200x gap closed by
## design, confirmed by a real, non-frozen policy at the new LR -- reporting back
## before recommitting to the full budget, per instruction

**Task requirement 1 -- proposed configuration, derived from the actual gap figures, not
a round guess.** Precise mechanism first, since the task's own framing ("higher LR
risks reintroducing the grad_norm instability") is not quite right and needed
correcting rather than silently accepted: pre-clip `grad_norm` (what `GRAD_BREAKER`
checks) is a property of the BACKWARD PASS of the advantage-weighted loss ONLY -- it has
no LR term in its formula at all (LR is applied by the OPTIMIZER STEP, strictly after
the gradient is computed and clipped; re-derived from the same `clip_grad_norm_`
semantics established in the gradient-clipping task). Raising `TARGET_LR` does NOT
directly increase grad_norm or breaker risk through that mechanism, and the advantage
clamp (unchanged, 0.4) continues to bound each row's contribution to grad_norm
regardless of LR. The real risk is INDIRECT and cumulative: a genuinely-moving policy
could, over many steps, shift into a region where rollouts/rewards/advantages behave
differently than what the clamp was calibrated against -- not yet observed, since no
prior run in this stage had actually exercised a moving policy (the 50-step baseline
turned out to be frozen). This is why a short diagnostic BEFORE the full budget (task
requirement 2) is the right next step, not because higher LR mechanically causes
grad_norm blowup.

**Proposed values, with derivation**:
- `TARGET_LR`: 1e-6 -> 2e-5 (**20x**). Deliberately not matched to Stage 9c's own SFT LR
  (2e-4) -- RL fine-tuning atop an already-good SFT checkpoint conventionally uses a
  gentler LR than the seeding SFT phase itself (a separate, standard prudence
  consideration, independent of the grad_norm question) -- stays 10x below SFT's LR
  while still being a large, deliberate jump from the prior value.
- `FULL_STEPS` (for the eventual full run, not launched in this task): 50 -> **150**
  (3x), still a bounded multiple of Stage 9b's own 50-step precedent, not an open-ended
  increase.
- Combined effect on the effective-magnitude proxy (LR x post-clip grad_norm, always
  clipped to 1.0 by the ever-active `max_grad_norm=1.0` default, confirmed again this
  task): per-step magnitude closes the gap **20x** outright (1e-6 -> 2e-5). Cumulative
  over the proposed 150 steps (3e-3, optimistic constructive-sum upper bound) is now
  **~15x LARGER** than a single SFT step (2e-4), versus ~4x SMALLER before -- a **~60x
  (~1.8 order of magnitude) swing**, closing essentially the full gap the task asked
  for. Sanity-checked against SFT's own TOTAL cumulative movement across its full
  40-step run (~1e-2, estimated from its own declining grad_norm trajectory): the new
  GRPO cumulative movement (~3e-3) is **~30% of that** -- non-trivial, but not wildly
  exceeding the scale that successfully drove the original SFT convergence.

**Implementation**: `sft_seeded_rl_full.py`'s `TARGET_LR`, `FULL_STEPS`, and
`MILESTONE_EVERY` are now env-overridable (`STAGE9D_TARGET_LR`,
`STAGE9D_FULL_STEPS`, `STAGE9D_MILESTONE_EVERY`, all defaulting to the original
full-run values when unset) -- one script covers both the short diagnostic and any
eventual full run, no new file needed. `sft_seeded_rl.py` (the original 8/16-step
dry-run script) deliberately left untouched -- it remains the source of truth for the
earlier dry-run regression tests and is no longer the active script for this stage's
forward work.

**Task requirement 3 -- PERMANENT PROCESS ADDITION, implemented in `evaluate_and_track`
itself, not a one-off script**: every milestone now computes and reports, as a
STANDARD field (`completions_changed_vs_previous`, `completions_changed_fraction`,
`completion_diff_reference`) -- not a conditional check requiring a manual post-hoc
text inspection to trigger -- the fraction of the 21 held-out eval completions that
changed relative to the immediately preceding reference point: the previous milestone
for step>0, or the zero-step sanity-check completions (already generated before
training starts, previously computed but never reused for this purpose) for the very
first milestone, so even the first milestone gets a genuine step-0 comparison rather
than "no previous milestone to compare to." Printed at every milestone as part of the
routine log line (`COMPLETIONS_CHANGED=N/21 (fraction)`), plus a loud, immediate `***
FLAG: ZERO completions changed ***` when the fraction is exactly 0 -- the exact
condition that silently produced the frozen 50-step baseline result, now caught
automatically the moment it happens rather than requiring the kind of ad hoc
completion-hash comparison that was needed to catch it after the fact this time.
Applies to both baseline and MAIN going forward, since both run through this same
script. `test_sft_seeded_rl_full.py` gained 4 new static tests confirming this is wired
as a routine (not conditional) field, that it falls back to the sanity-check baseline
for the first milestone, and that the zero-change condition is flagged immediately.

**Task requirement 2 -- short diagnostic, launched and completed**: 16 steps,
`milestone_every=4`, `TARGET_LR=2e-5` (the proposed new value), baseline reward config
(the simpler config, appropriate for a mechanical "does this configuration move the
policy at all" check rather than a scientific result). Mandatory pre-launch hash check
performed (`sft_seeded_rl_full.py`, `advantage_clamp.py` both verified byte-identical
remote vs. local; GATE: PASS) before launch. Launched with `python3 -u` (unbuffered)
deliberately this time, given the task's own emphasis on immediate reporting.

**Result: grad_norm stayed comfortably under the breaker (5.59-40.75, vs.
`GRAD_BREAKER=50.0`), confirming the "LR doesn't feed into grad_norm" mechanism
empirically, not just by derivation. `hard_stop=None`, `terminal_step=16/16`.** KL
ranged 0.0-1.0477 (mean 0.497) -- comfortably under `KL_BREAKER=5.0` but notably higher
than the frozen 50-step run's max (0.31), consistent with a policy that is now actually
diverging from its reference, i.e. genuinely moving. Advantage-clamp engagement 70.3%
(90/128), in the same range as every prior run at this clamp value -- not blowing up at
the new LR.

**The completion-diff check (task requirement 3's new mechanism) worked exactly as
intended, catching the pattern live rather than requiring a post-hoc check**:

| milestone | completions changed vs. previous | reference |
|---|---|---|
| step 4 | 0/21 (0.000) | step-0 sanity check |
| step 8 | 0/21 (0.000) | step 4 |
| step 12 | 1/21 (0.048) | step 8 |
| step 16 | 2/21 (0.095) | step 12 |

**A monotonically increasing trend, not noise -- and it starts right where the math
predicts it should.** Cumulative movement crosses the single-SFT-step reference
(2e-4) between step 8 (1.6e-4, still below) and step 12 (2.4e-4, just above) -- exactly
where the first non-zero completion change appears (step 12). This is a clean,
independent confirmation of the same magnitude-based reasoning that predicted the
original 50-step run would be frozen, now predicting -- correctly -- where movement
would start to appear at the new LR.

**Plain verdict: the new configuration is confirmed to (a) keep grad_norm safely under
the breaker with the advantage clamp still active, and (b) actually move the policy --
the key criterion the original 50-step run failed.** 16 steps is not enough to show
LARGE movement (2/21 by the end), but the trend is real, accelerating, and lands
exactly where predicted -- a genuinely different outcome from the original frozen run,
not a marginal improvement.

**Task requirement 4 -- reporting back, NOT auto-relaunching the full budget**: per
instruction, the full 150-step baseline+MAIN pair was NOT launched in this task. The
diagnostic confirms the new configuration is viable and worth committing the full
budget to, but that decision is left for the next task, along with any final
adjustment to `FULL_STEPS`/`TARGET_LR` the user wants to make before that commitment.

**Regression tests added**: `test_sft_seeded_rl_full.py` gained a golden-record test
against the diagnostic evidence (confirms `hard_stop=None`, `grad_norm` max well under
the breaker, and the exact `[0, 0, 1, 2]` completion-change trend), plus the 4
completion-diff-mechanism tests noted above and 2 tests confirming the env-overridable
config defaults and the documented LR/step derivation. Full suite (53 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-full-baseline-diag-lr2e5/`:
`stage9d_full_baseline_diag.json`, SHA256
`667c9260dc18237c2f3922480366fe32a8ab0b9a53277c309aa7912b02fdc126`;
`sft_seeded_rl_full_diag_stdout.log`, SHA256
`d81a729cb0f4947bcf5328e5e8d114facc935235ae2ded02c81702e667deacc1` (both remote/local
identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~17-iteration stopping poll, same pattern as
every prior task in this stage -- retrieved via the background task result, not a
script fault). `g5.xlarge` throughout. Session SSH rule revoked at teardown.

**Status: short diagnostic confirms the new configuration (LR=2e-5) both stays safe
(no breaker) and genuinely moves the policy, unlike the original configuration.
STOPPING here per instruction 4 -- not auto-relaunching the full 150-step
baseline+MAIN pair. Waiting for go-ahead (and any final config adjustment) before
committing the full budget.**

## Full 150-step BASELINE at the corrected configuration (2026-08-31): breaker fired --
## on KL, not grad_norm, a genuinely new failure mode never seen at any prior scale

**Launch**: `STAGE9D_FULL_PHASE=baseline STAGE9D_FULL_STEPS=150 STAGE9D_MILESTONE_EVERY=15
STAGE9D_TARGET_LR=2e-5`, approved configuration from the prior diagnostic. Milestone
cadence proposed and stated explicitly per task requirement 2: every 15 steps (150/15=10
milestones, matching the same milestone-count precedent as the earlier 50-step/5-cadence
run). Mandatory pre-launch hash check: `sft_seeded_rl_full.py` and `advantage_clamp.py`
verified byte-identical remote vs. local. GATE: PASS. Launched with `python3 -u`
(unbuffered), checkpoint identity confirmed (`c84afe03...` matches), zero-step sanity
check passed before training started.

**Step 15 milestone (the only one reached): zero completions changed vs. the step-0
sanity baseline -- investigated live, not just logged, per instruction.** Pulled the raw
grad_norm telemetry for all 15 steps (range 3.8-36.5, comfortably under
`GRAD_BREAKER=50.0`, no anomaly) and reasoned explicitly before continuing: this run's
cumulative movement at step 15 (15x2e-5=3e-4) sits in the same range as the prior
diagnostic's step-12 mark (2.4e-4), where only 1/21 completions had flipped -- a single
sample changing this close to the movement threshold is expected to be noisy/
probabilistic given a different run's stochastic rollout draws, not a hard cutoff.
Judged NOT a red flag (unlike the original LR=1e-6 run, which was ~4x below even a
single SFT step -- an order-of-magnitude-different, unambiguously frozen regime) and
continued monitoring rather than stopping -- reported this reasoning to the user live,
before the next milestone landed.

**Breaker fired at step 23 -- but on KL, NOT grad_norm, a genuinely new failure mode.**
`hard_stop = {'step': 23, 'grad_norm': 4.9375, 'kl': 5.375675238668919}`. `grad_norm`
was nowhere near `GRAD_BREAKER=50.0`; `kl` exceeded `KL_BREAKER=5.0`. **This is the
first time in this entire stage's history (dry runs, the frozen 50-step run, the 16-step
diagnostic) that KL -- not grad_norm -- has been the breaking condition.** Per
instruction, stopped and reported immediately rather than attempting an automatic fix
(e.g., raising `KL_BREAKER`, which would mask rather than address the underlying
dynamic, exactly the kind of change this stage has consistently treated as off-limits
without explicit, separate sign-off).

**Full per-step KL telemetry, showing a real trend, not a spike**:
`0.0 -> 0.21 -> 0.12 -> 0.27 -> 0.89 -> 1.09 -> 2.51 -> 1.86 -> 3.04 -> 2.78 -> 1.64 ->
3.29 -> 2.77 -> 2.94 -> 3.80 -> 2.87 -> 2.36 -> 4.21 -> 4.56 -> 4.41 -> 3.49 -> 4.51 ->
5.38` (steps 1-23). Noisy step-to-step, but a clear, sustained UPWARD trend from
~0 to past 5.0 over 23 steps -- not a sudden anomalous jump. `grad_norm` over the same
steps (14.6, 20.8, 29.5, 23.9, 36.5, 24.3, 5.75, 18.0, 5.78, 6.53, 16.9, 22.6, 23.5,
19.5, 3.80, 15.75, 14.9, 4.56, 8.44, 16.4, 9.25, 4.0, 4.94) shows NO comparable trend --
noisy but range-bound throughout, exactly as expected (LR has no term in the
pre-clip-grad_norm formula, re-confirmed empirically again this run).

**Mechanistic explanation, not just a correlation**: the advantage clamp bounds each
row's contribution to the POLICY GRADIENT magnitude (hence `grad_norm`) -- it has no
effect on KL divergence, which is a property of how far the current policy has drifted
from the frozen reference policy, accumulating over TRAINING STEPS regardless of any
per-row clamp. GRPO's `beta=0.04` KL penalty TERM in the loss is a soft, continuous
discouragement of drift, not a hard cap -- if the correctness-reward-driven push is
strong enough relative to that soft penalty, cumulative KL can still climb past the hard
`KL_BREAKER` given enough steps of genuine movement. **This is, in a real sense, exactly
what closing the ~200x movement gap was supposed to produce -- a policy that actually
moves -- and moving means diverging from the reference, which is precisely what KL
measures.** The original (frozen) configuration never exercised this failure mode
because there was no real divergence to measure.

**Completion-text spot-check (task requirement 5), done and disclosed with an honest
limitation**: read 3 full sample completions from the one reached milestone (step 15) --
all well-formed, coherent Nib/Nomo state-tracking completions (not corrupted/degenerate
output), ruling out "the automated diff trivially matched because everything broke."
**Limitation disclosed rather than glossed over**: a true independent BYTE-LEVEL
re-verification against the step-0 sanity completions was not possible from the saved
evidence alone -- `_sanity_completions`' raw text was never persisted to the JSON (only
the aggregate `sanity_nonliteral_rate`/identity hash were saved), so the "0/21 changed"
figure at step 15 rests on the automated hash-diff computed live in memory (which did
have both texts available at the time), not on a manual re-derivation from the archived
artifacts. Flagged as a gap worth closing (persist sanity completion text too) if this
exact kind of manual cross-check is needed again -- not fixed in this task.

**Plain verdict**: the redesigned configuration succeeded at its actual goal --
producing a policy that genuinely moves, confirmed both by the step-15 investigation
(consistent with the diagnostic's own near-threshold pattern) and unambiguously by the
KL trajectory itself (real, sustained, measurable divergence, not noise). It also
surfaced a real, previously-unobserved constraint: **the advantage clamp protects
grad_norm, not KL** -- these are two independent breaker conditions addressing two
different mechanisms, and only one of the two has a calibrated mitigation in place.
150 steps at this LR is more than this specific rollout trajectory's KL budget could
sustain; the run stopped at 23. This is a genuinely different, more informative kind of
"stop" than the original frozen run's silence -- it is direct evidence the pipeline is
now exercising real dynamics, just dynamics with an as-yet-unaddressed constraint.

**Regression tests added**: `test_sft_seeded_rl_full.py` gained a golden-record test
against this evidence -- confirms `hard_stop` fired on the KL condition specifically
(not grad_norm), asserts the KL trend is real (start < mid < end, not just noise), and
confirms grad_norm stayed bounded throughout. Full suite (54 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-full-baseline150-v1/`:
`stage9d_full_baseline150.json`, SHA256
`6d561577ddb89d54acc4d0d626dbac6aadaf27397f13909a5a3cc178d9d423cf`;
`sft_seeded_rl_full_baseline150_stdout.log`, SHA256
`b8ae9b4c65d04422a37f6f92ea573ac06bd602eb7340332380bee1786b9faf79` (both remote/local
identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same pattern as
every prior task -- retrieved via the background task result, not a script fault).
`g5.xlarge` throughout. One AWS SSO credential expiry at the very start of this task
(same recurring pattern documented elsewhere in this project) -- paused, had the user
re-authenticate, re-verified via `aws sts get-caller-identity` before proceeding.
Session SSH rule revoked at teardown.

**Status: BASELINE did not complete -- stopped at step 23/150 via the KL breaker, a
genuinely new finding (not grad_norm, which the advantage clamp addresses). NOT
proceeding to MAIN, per instruction and per basic prudence -- MAIN adds the CoT penalty
on top of the same LR, and would be exposed to the identical unaddressed KL-growth
mechanism, likely sooner given the added reward-shaping pressure. Reporting this
plainly and waiting for direction: whether to address the KL constraint directly (e.g.
a KL-focused mitigation analogous to Stage 4/9b's own per-token KL clamp, now
demonstrably relevant here in a way it wasn't at any frozen or dry-run scale), adjust
`KL_BREAKER`/`beta` with explicit sign-off, reduce the step budget to fit within
whatever KL budget this LR allows, or something else -- not decided or attempted in
this task.**

## Per-token KL clamp: root cause found (never wired in), calibrated on real telemetry,
## validated -- KL now plateaus instead of climbing toward the breaker (2026-08-31)

**Task requirement 1 -- traced, not assumed: the per-token KL clamp is NOT active
anywhere in Stage 9d's pipeline.** `grep` across every script in this stage
(`sft_seeded_rl_full.py`, `sft_seeded_rl.py`, `advantage_clamp.py`) for
`kl_clamp`/`KL_CLAMP`/`ref_per_token_logps`/`D_MAX`/`entropy_from_logits` returned zero
matches in the active training pipeline (`per_token_gradient_diagnostic.py` patches
`_get_per_token_logps_and_entropies`, but only for its own retain_grad gradient-capture
diagnostic, unrelated and never part of the training pipeline). **This is the actual
root cause of the step-23 KL breaker -- not a regression of something that used to
work, and not a miscalibration of an existing mechanism (task requirement 2's scenario
does not apply): the clamp was simply never wired in at all.** Task requirement 3's
scenario applies.

**Implementation**: wired in, reusing Stage 9b's own `kl_calibration.py`
(`compute_clamp_from_pool`) unchanged rather than reimplementing it -- imported via the
same cross-stage `sys.path.insert` pattern already used for `09_direct...`/
`07_positive...`, now extended to `09b_model_scale_ablation`. Clamp-application
mechanism follows `dmax_sensitivity_check.py`'s exact validated pattern: intercept
`_get_per_token_logps_and_entropies` during the POLICY forward pass inside
`compute_loss` (guarded by the same `inside_compute_loss` flag style), read the
already-precomputed `inputs['ref_per_token_logps']`, clamp the ref-minus-policy diff to
`+/-D_MAX`, write the clamped value back into `inputs['ref_per_token_logps']` so the
rest of `compute_loss`'s own KL term computation uses it -- not re-derived from
scratch. Merged into the SAME `compute_loss` override as the existing advantage clamp
(one monkeypatch, two independent mechanisms). `STAGE9D_KL_CLAMP_D_MAX` unset/empty =
MEASUREMENT mode (pool collected, pass-through, no clamping); set to a float = ACTIVE
mode. Full pool and its `compute_clamp_from_pool` diagnostics are now persisted in the
saved evidence (`kl_clamp_summary`), so calibration never has to be redone blind.

**Task requirement 2's calibration methodology, applied from the start (nonzero-only
pooling, the lesson already learned in 9b)**: launched a MEASUREMENT run at the SAME
configuration that produced the step-23 breaker (baseline, LR=2e-5, `FULL_STEPS=25` to
reach past the prior break point), clamp in pass-through mode. Completed the full 25
steps this time WITHOUT breaking (run-to-run stochastic variance -- KL reached 3.60 by
step 25, close to but under the 5.0 breaker this specific launch) -- still gathered a
large (49,376-token), genuinely-informative pool since the clamp was pass-through the
whole time and real divergence was accumulating throughout.

**The naive Stage-4-formula transplant barely engages here too -- the SAME lesson
already learned once for the advantage clamp, now recurring for KL, disclosed rather
than silently worked around**: `compute_clamp_from_pool`'s own formula (median +
1.5*IQR, *1.5) produced `clamp_value=13.80`. This engages only 2.9% of the pool and
reduces its mean by just 1.7% (2.68 -> 2.64) -- nowhere near enough to meaningfully
change the aggregate KL trajectory. Root cause: this pool's shape violates the
formula's implicit assumption (a mostly-well-behaved population with a separable
genuine-outlier tail) -- median (0.457, nonzero-only) is SMALLER than the IQR (5.83),
meaning the "mild outlier fence" (median+1.5*IQR=9.2) sits above the 90th percentile
(9.27) of the pool, and the actual clamp (1.5x that) sits near the 97th percentile
(13.68) -- a heavy, genuine right tail from real, ongoing divergence, not a population
with occasional true outliers layered on a tight, stable core.

**Chosen instead, data-driven rather than formula-derived, same discipline as the
advantage clamp's own 0.4 choice: D_MAX=2.0.** Justified directly from the pool: caps
engagement at 34.1% (16,820/49,376 tokens) and reduces the pool's own mean by 71%
(2.68 -> 0.78) -- a MEANINGFUL reduction, unlike the naive formula. Explicitly reasoned
through why being this aggressive is more defensible for the KL clamp than it would be
for the advantage clamp (which directly competes with the primary reward-learning
signal): the KL clamp bounds a REGULARIZATION/divergence term, not the reward-driven
gradient itself -- a token whose divergence is clamped still receives a real, bounded
KL penalty (not zero), it simply isn't allowed to dominate the aggregate/gradient
unboundedly, the same "additive, surgical, real signal persists just bounded"
philosophy this project has used for every clamp so far, not a case of masking the
underlying dynamic (KL_BREAKER itself is left completely untouched, per task
requirement 4).

**Task requirement 5 -- validation run, launched and passed cleanly.** 20 steps
(matching the 15-20 range, past the original 23-step break point given the shorter
20-step budget was itself informative), `STAGE9D_KL_CLAMP_D_MAX=2.0` ACTIVE, same LR=2e-5,
`milestone_every=5`. Mandatory pre-launch hash check performed for all three files
(`sft_seeded_rl_full.py`, `advantage_clamp.py`, and the reused `kl_calibration.py`) --
GATE: PASS on all three, both for this launch and the earlier measurement launch.

**Result: `terminal_step=20/20`, `hard_stop=None`.** KL, instead of climbing toward the
breaker as in every prior LR=2e-5 run, **plateaus**: grows from 0 to ~0.5 by step 7,
then stays bounded in the 0.28-0.59 range for the remaining 13 steps -- a stabilized
ceiling, not just slower growth, comfortably under `KL_BREAKER=5.0` throughout (max
observed 0.59, an 8.5x margin). `grad_norm` stayed in its normal range (3.8-38.0,
unaffected by the KL clamp, as expected -- the two mechanisms are independent).
Clamp engagement 30.5% (11,751/38,472 tokens), closely matching the 34.1% predicted
from the measurement pool -- the extrapolation held up under a live re-run, not just in
the static snapshot. Advantage-clamp engagement (75%) unaffected, confirming no
interference between the two mechanisms.

**Confirmed explicitly, not assumed, per task requirement 5's own instruction: the
policy did NOT re-freeze.** Completion-diff trend across the 4 milestones:
`0/21 -> 1/21 -> 1/21 -> 1/21` changed -- continued, genuine movement at essentially
the same rate as the earlier successful (unclamped) diagnostic (`0,0,1,2`), not a
collapse back to the original frozen-policy failure mode. The KL clamp tamed the
breaker-triggering metric without suppressing the thing the LR increase was designed
to produce.

**Task requirement 6 -- not needed, but the underlying question addressed**: since the
validated 20-step run shows KL PLATEAUING rather than continuing to grow, this is a
meaningfully different (better) trajectory than "still growing but more slowly," and
suggests -- though only 20 of the eventual 150 steps were directly validated, stated as
a real limitation, not overclaimed -- that the calibrated clamp may sustain a much
longer horizon without further step-budget reduction. Not proven for the full 150-step
range in this task; the next full-length launch will be the actual confirmation.

**Task requirement 7 -- evidence gap fixed.** `sft_seeded_rl_full.py`'s saved `config`
now includes `sanity_completions` (the full raw step-0 completion text, previously
computed but never persisted -- only the aggregate `sanity_nonliteral_rate` was saved
before). Milestone completions were already fully persisted (each `sample` always
included its full `completion` text) -- the gap was specifically the step-0 reference
point used by the completion-diff check. Future completion-diff claims can now be
independently re-derived from the archived JSON alone, not just trusted from the
live in-memory computation.

**Regression tests added**: `test_sft_seeded_rl_full.py` gained 5 new tests -- static
wiring confirmation (KL clamp reuses `kl_calibration.py`, applies the
`dmax_sensitivity_check.py` pattern, measurement/active mode switch, pool/diagnostics
persistence, sanity-completion persistence) plus 2 golden-record tests against the
measurement and validation evidence (naive-formula under-engagement, and the
plateaued-KL + continued-movement result). Full suite (60 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/`:
- `stage9d-kl-clamp-measurement/stage9d_full_baseline_klmeasure.json`, SHA256
  `5f761394c3a9102294c820c94a8241aa50367a1a3b72c4ec7b35ae95a334b000`;
  `sft_seeded_rl_full_klmeasure_stdout.log`, SHA256
  `705c0dd688a39cb085db09899bcd7324520ecdf1029eab8877c97e8502962e1f`.
- `stage9d-kl-clamp-validation/stage9d_full_baseline_klvalidate.json`, SHA256
  `0424141843c45ed688c0da3eab2e4eef74caad63942e0939687067e1233783c8`;
  `sft_seeded_rl_full_klvalidate_stdout.log`, SHA256
  `452760b7f1ce2625cd81e1dc7ce9f4a1fab60414e49d7bead1a99085a7af6f20`.
All four files confirmed remote/local identical.

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same pattern as
every prior task -- retrieved via the background task result). One capacity-driven
instance-type substitution this task: `g5.xlarge` hit `InsufficientInstanceCapacity` on
the initial start attempt (a documented, previously-seen pattern in this project) --
switched to `g5.2xlarge` (same A10G GPU), which started successfully; restored to
`g5.xlarge` after teardown for consistency with future launches. One SSH timeout mid-run
from the operator's own local IP changing (same ordinary dynamic-IP pattern seen in an
earlier task) -- confirmed via `describe-instances` that the instance itself stayed
healthy throughout, re-authorized the new IP, reconnected. Session SSH rules revoked at
teardown.

**Status: root cause identified and fixed (KL clamp was never wired in), calibrated on
real telemetry from this LR regime (not the old frozen-policy regime, which never had
meaningful KL signal to calibrate against), and validated over 20 steps: KL plateaus
well under the breaker, grad_norm unaffected, policy continues to move. Not yet
confirmed at the full 150-step horizon -- that requires the next launch. Not proceeding
automatically; reporting this result and waiting for direction on re-attempting the
full 150-step BASELINE run with the KL clamp (D_MAX=2.0) now active alongside the
advantage clamp.**

## Full 150-step BASELINE, both clamps active: COMPLETE, clean pass at real scale
## (2026-08-31) -- the KL plateau holds for the entire run, not just 20 steps

**Launch**: `STAGE9D_FULL_PHASE=baseline STAGE9D_FULL_STEPS=150 STAGE9D_MILESTONE_EVERY=15
STAGE9D_TARGET_LR=2e-5 STAGE9D_KL_CLAMP_D_MAX=2.0` -- the approved, both-clamps
configuration. Mandatory pre-launch hash check on all three files
(`sft_seeded_rl_full.py`, `advantage_clamp.py`, `kl_calibration.py`) -- GATE: PASS on
all three. One capacity note: `g5.xlarge` started successfully this time (no repeat of
the earlier `InsufficientInstanceCapacity` substitution).

**Explicit step 40-50 checkpoint (per instruction, reported specifically rather than
folded into routine milestone logging)**: at step 45, pulled the full raw
grad_norm/KL telemetry directly (not just the milestone summary) and confirmed KL had
been oscillating in a 0.29-0.66 band since roughly step 15 -- **no resumption of
growth, 30 steps after the clamp first engaged. Success criterion met; did not stop.**
Secondary observation surfaced at the same checkpoint, not itself a stop condition:
three consecutive zero-completion-change milestones (steps 15, 30, 45) coincided with a
real, sustained decline in grad_norm (15-35 in the first ~8 steps down to 2-6 by
step 30-45) -- flagged as consistent with the model settling toward a low-gradient
equilibrium (accuracy flat at 28.6% the whole run) rather than a safety concern, since
neither breaker condition was anywhere close. This assessment held up: movement resumed
at step 60 (1/21 changed) and continued intermittently through step 120, confirming the
zero-change stretch was a temporary plateau in an otherwise-oscillating pattern, not a
permanent freeze.

**Result: `terminal_step=150/150`, `hard_stop=None` -- the full run completed with no
breaker fire at any point.** KL: min 0.0, max 0.72, mean 0.48 across all 150 steps --
**the plateau holds for the ENTIRE run, not just the 20-step validation or the step
40-50 checkpoint**: KL in the final 30 steps (121-150) ranged 0.20-0.68, statistically
indistinguishable from the 0.29-0.66 band seen at step 15-45 -- no late-stage growth
whatsoever, a 7x margin under `KL_BREAKER=5.0` at the observed max. `grad_norm`: max
34.75, mean 4.45 (comfortably under `GRAD_BREAKER=50.0`). KL-clamp engagement 41.1%
overall; advantage-clamp engagement 75.8% -- both in the same range observed at every
prior scale, no runaway engagement as the run got longer.

**All 10 milestones**: `nonliteral_rate=1.0` and `genuine_correct_among_structural_
nonliteral=0.286` at EVERY milestone, unchanged from the checkpoint's own starting
values throughout the entire 150-step run -- no erosion, no growth, no token drift away
from Nib/Nomo at any point (`drifted_from_nib_nomo_count=0` throughout). Completion
movement was real but modest and non-monotonic across milestones (`0,0,0,1,2,2,2,2,0,0`
changed per 21), alternating between brief plateaus and small bursts of movement rather
than a steady trend in either direction.

**Task requirement 5 -- completion-text spot-check at start/mid/end, independently
re-verified from the now-persisted evidence (not just the live diff), per instruction**:
- **Start (step 15) vs. step-0 sanity**: independently recomputed hash comparison
  confirms 21/21 identical, exactly matching the script's own live-computed claim --
  the fix from the prior task (persisting `sanity_completions`) worked as intended.
- **Mid (step 75) and end (step 150)**: read 3 full completions each (indices 0, 10,
  20) -- all well-formed, coherent Nib/Nomo state-tracking completions, and (a further,
  unprompted cross-check) samples 0/10/20 turned out to be BYTE-IDENTICAL between step
  75 and step 150 specifically, consistent with the overall low-but-nonzero movement
  rate.
- **Start (step 15) vs. end (step 150), independently computed**: 18/21 completions
  identical, 3/21 genuinely different -- real, modest movement across the full run,
  neither frozen nor dramatically changed. This matches the picture the per-milestone
  diff counts already suggested, now confirmed by a direct start-to-end comparison
  rather than only chained consecutive-milestone diffs.

**Task requirement 6 -- not proceeding to MAIN automatically, per instruction.** The
KL-clamp fix is now confirmed at the actual scale this stage has been building toward:
root cause found (never wired in) -> calibrated on real telemetry -> validated at 20
steps -> validated at the full 150-step horizon with an explicit mid-run checkpoint.
Both clamps coexist without interference (engagement rates for each stayed in their
already-established ranges throughout). The seeded Nib/Nomo behavior is completely
stable under ordinary (non-adversarial) RL pressure at this configuration -- the
necessary BASELINE control this whole stage exists to establish before MAIN (the actual
adversarial-pressure test) can be meaningfully interpreted.

**Regression tests added**: `test_sft_seeded_rl_full.py` gained a golden-record test
against this evidence -- confirms the clean pass, that the KL plateau holds across the
full run (final-third mean not more than 2x the first-third mean, guarding against
delayed growth), grad_norm stays bounded, all milestones show zero token drift, and
`sanity_completions` are persisted. Full suite (61 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-full-baseline150-v2-both-clamps/`:
`stage9d_full_baseline150_bothclamps.json`, SHA256
`4e7f4cb6c32d3b722dbec54fc6a9168ab7d25a5ca19c5fa1ca4cb466a084beb9`;
`sft_seeded_rl_full_baseline150v2_stdout.log`, SHA256
`3c2055b035c3bc0bf8ebb4ce52353aee991d41719ebe65383ded9c5299a9b0fd` (both remote/local
identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (exceeded
the 120s foreground timeout during the ~16-iteration stopping poll, same pattern as
every prior task -- retrieved via the background task result). `g5.xlarge` throughout,
no capacity substitution needed this time. Session SSH rule revoked at teardown.

**Status: BASELINE fully complete and clean at real scale (150/150 steps, no breaker,
seeded behavior completely stable, KL plateau confirmed holding for the entire run
including an explicit mid-run checkpoint). Both clamps validated as compatible and
non-interfering. NOT proceeding to MAIN automatically, per instruction -- waiting for
confirmation before launching MAIN (the actual adversarial-pressure test) at this
now-fully-validated configuration.**

## MAIN, full 150-step adversarial-pressure run: COMPLETE -- the actual experiment
## this entire stage exists to answer (2026-08-31). Clean pass; core negative result:
## genuine_correct_among_nonliteral never rose above the 28.6% starting baseline

**Task requirement 1 -- reward invariant re-verified fresh for this exact combination
before launching, not assumed to carry over.** Ran `verify_reward_invariant()` locally
(CPU-only, no GPU needed) for the full adversarial defaults (`cot_max_scale=2.0,
consistency_magnitude=0.15, signal_magnitude=0.15`): `margin_correct_over_wrong=0.7`,
`margin_wrong_over_malformed=0.7`, both strictly positive -- confirmed BEFORE any
launch. Re-derived, not assumed, why the invariant and both clamps are structurally
non-interacting for MAIN specifically: the invariant checks `score_completion_v2`'s
OWN output range, entirely upstream of GRPO's advantage computation; the advantage
clamp bounds a group-normalized quantity whose ceiling (Grubbs, `GROUP_SIZE`-only) is
reward-function-independent; the KL clamp bounds policy-vs-reference divergence, a
property of the training trajectory, not of which reward function drove it. The one
genuine open question -- whether the ACTUAL KL trajectory (not just the clamp
mechanism) would look the same under MAIN's added CoT penalty -- was correctly treated
as needing live verification, not assumed (see the checkpoint below).

**Task requirement 2 -- mandatory pre-launch hash check**: all three files
(`sft_seeded_rl_full.py`, `advantage_clamp.py`, `kl_calibration.py`) verified
byte-identical remote vs. local (unchanged since the BASELINE launch -- no redeploy
needed, gate re-run anyway). GATE: PASS on all three. One capacity-driven substitution
(`g5.xlarge` -> `g5.2xlarge`, same pattern as an earlier task) -- restored after
teardown.

**Task requirement 3 -- explicit step 40-50 checkpoint, verified live, not assumed to
match BASELINE.** At step 45 (with the run already having progressed to step 60 by the
time of the check), pulled the full raw grad_norm/KL telemetry directly: KL oscillating
in a 0.11-0.71 range throughout -- **essentially indistinguishable in character from
BASELINE's own 0.29-0.66 band at the same checkpoint, confirmed rather than assumed.**
grad_norm showed the same declining-then-stable pattern BASELINE exhibited. The
mechanistic non-interaction argument from requirement 1 held up empirically: MAIN's
added CoT penalty did not produce a meaningfully different KL trajectory.

**Result: `terminal_step=150/150`, `hard_stop=None`.** KL: min 0.0, max 0.71, mean
0.467 across the full run -- the plateau held identically to BASELINE (final-30-step
KL values ranged 0.19-0.58, no late growth). `grad_norm`: max 39.75, mean 4.75.
KL-clamp engagement 40.0% (vs. BASELINE's 41.1%); advantage-clamp engagement 72.6%
(vs. 75.8%) -- both configs landed in essentially the same clamp-engagement regime.

**Task requirement 4/5 -- primary tracking, and the critical flag condition
(`genuine_correct_among_nonliteral` above 28.6%) monitored at every milestone: never
triggered.** All 10 milestones (steps 15-150):

| step | nonliteral_rate | genuine_correct_among_nonliteral | completions changed | token drift |
|---|---|---|---|---|
| 15-150 (all 10) | 1.000 | **0.2857 (exactly 2/7) at every single milestone** | 1-3/21, never zero after step 45 | 0 |

**`genuine_correct_among_nonliteral` did not move from its starting value at ANY point
across the full 150-step adversarial run -- no first-sign-of-increase event to attribute,
because none occurred.** `nonliteral_rate` also held at 1.0 throughout, and
`drifted_from_nib_nomo_count` stayed at 0 throughout -- the model never abandoned
Nib/Nomo specifically under this pressure either. One structural difference from
BASELINE worth noting, not a red flag: MAIN's completion movement was more
consistently non-zero across milestones (BASELINE had three zero-change milestones,
including its very last one; MAIN had none after step 45) -- suggesting the added
reward shaping produces a somewhat more continuously-shifting policy at the surface
text level, but this extra movement never translated into any change on the
core correctness-while-nonliteral metric.

**Task requirement 6 -- completion-text spot-check (start/mid/end, independently
re-verified from persisted evidence) and direct BASELINE comparison at every
corresponding milestone, not just at the end**:
- **Start (step 15) vs. step-0 sanity**: independently recomputed, 20/21 identical --
  matches the script's own live-computed claim (1/21 changed) exactly.
- **Mid (step 75) and end (step 150)**: 3 full completions each spot-checked --
  well-formed, coherent Nib/Nomo completions throughout.
- **Start (step 15) vs. end (step 150), MAIN only**: 18/21 identical, 3/21 different --
  the SAME magnitude of overall movement BASELINE showed across its own full run (also
  18/21 identical).
- **Direct BASELINE-vs-MAIN comparison, every milestone (15 through 150)**:
  `nonliteral_rate` and `genuine_correct_among_structural_nonliteral` matched EXACTLY
  between the two configs at every single corresponding step -- zero divergence on
  either core metric anywhere in either run. `drifted_from_nib_nomo_count` matched (0)
  throughout too. The only observed difference was in movement CONSISTENCY (discussed
  above), not in outcome.
- **A further, unprompted cross-check**: BASELINE's step-150 completions vs. MAIN's
  step-150 completions (same 21 held-out eval prompts, two entirely independent
  150-step training trajectories under different reward functions) are **20/21
  identical**. Despite 150 steps of differently-shaped reward pressure, the two final
  policies converged to nearly the same observable behavior on this eval set --
  a striking, independently-derived confirmation that the adversarial pressure
  produced essentially no differentiated effect relative to the simple control.

**Plain verdict, stated directly**: under this specific, now-thoroughly-validated
configuration (Stage 9c's SFT-seeded checkpoint, LR=2e-5, advantage clamp=0.4, KL clamp
D_MAX=2.0, 150 steps), **adversarial pressure to abandon literal Heads/Tails language
while rewarding consistent non-literal state tracking did NOT induce the model to
combine non-literal token usage with genuinely correct state tracking.** The model
maintained its SFT-seeded Nib/Nomo usage (100%, never eroded) at the SAME
28.6%-correct rate it started with, under both the simple correctness-only control and
the full adversarial pressure, with no measurable difference between the two. This is
a clean negative result for the core steganography question this project has been
investigating since Stage 1 -- not an inconclusive one, given the extensive validation
(20-step diagnostic, explicit mid-run checkpoints in both runs, full-length completion
tracking, independent spot-checks, and a direct matched-pair comparison) behind it.

**Regression tests added**: `test_sft_seeded_rl_full.py` gained two golden-record
tests -- confirms MAIN's clean pass and the core negative result (`genuine_correct_
among_nonliteral` never exceeds 2/7 at any milestone) is preserved in the test suite,
and a direct BASELINE-vs-MAIN comparison test asserting exact metric equality at every
corresponding milestone. Full suite (63 tests) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-full-main150-v1/`:
`stage9d_full_main150.json`, SHA256
`967f2a2c6965141ebdf6c6fe310f41eff3fae6f8f1912a9bce9b19dc34a077e8`;
`sft_seeded_rl_full_main150_stdout.log`, SHA256
`792cee0692641601abce4b9a6d52fe8d5e2dd60a4a9d1b03d5a153fdd3cea0d8` (both remote/local
identical).

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED** (one AWS
SSO credential expiry mid-task, same recurring pattern documented elsewhere in this
project -- paused, had the user re-authenticate, re-verified via `aws sts
get-caller-identity` before retrying). `g5.2xlarge` (capacity substitution, same
pattern as an earlier task) restored to `g5.xlarge` after teardown. Session SSH rule
revoked.

**Status: MAIN complete. The full BASELINE+MAIN matched pair is now done, at a fully
validated, thoroughly-checkpointed configuration. Core finding: the seeded non-literal
behavior is completely stable under both ordinary RL and active adversarial pressure,
and adversarial pressure produces no increase in genuine correctness-while-nonliteral
above the SFT-seeded starting rate. This closes the experiment this stage was designed
to run.**

## Re-run on the decode-back-fixed seed checkpoint: BASELINE (2026-09-02)

The original BASELINE+MAIN pair above (immediately preceding this section) was run
against `stage9c-sft-diagnostic-v1`, a checkpoint whose genuine correctness started at
only ~28.6% -- too low, on its own, to distinguish "adversarial pressure suppresses
genuine correctness" from "there was barely any genuine correctness to suppress."
Stage 9c's subsequent work (rebalancing, then the corrupted-prefill diagnostic, then
the decode-back-line fix) produced `stage9c-sft-decode-back-v1`: 81.0% final-answer
accuracy, 90.5% intermediate-tracking accuracy, with the mechanism connecting the two
(the decode-back line) proven fully reliable (`decode_back_matches_own_trace_rate` =
1.0 on that checkpoint's own held-out eval). This re-run repeats the SAME validated
BASELINE+MAIN structure and configuration (LR=2e-5, 150 steps, advantage clamp=0.4, KL
clamp D_MAX=2.0) seeded from THIS much stronger checkpoint instead, so the original
question can finally be tested against a policy with real correctness to lose.

**Mandatory pre-flight reward check (performed before writing the launch script, per
explicit instruction -- not launched unconfirmed).** Traced `score_completion_v2`
term-by-term against a real decode-back completion. Six of seven terms
(`r_task`/`p_length`/`p_structure`/`p_state_variation`/`r_consistency`/`r_signal`) are
correctly blind to the decode-back line's presence, confirmed by inspection:
`parse_state_slots` only matches `Step N: ... State: X` lines (a different format than
the decode-back line), and `_extract_answer` only reads the trailing `<answer>` tag.
**`p_cot` is NOT blind**: it scans the entire reasoning region before `<answer>` for
literal Heads/Tails/H/T patterns, with no awareness that the decode-back line
deliberately, necessarily contains a literal word by design. Confirmed empirically
(`decode_back_bank.py`, `test_decode_back_bank.py`): an otherwise-perfect decode-back
completion scores `banned_count=1` -- a false-positive literal-leakage hit -- purely
from its own required line, vanishing to 0 with the line stripped. Under MAIN's full
`p_cot` weight (ramping to `cot_max_scale=2.0`), this would have been a direct,
mechanical, ever-growing incentive to suppress the exact line that fixed the
structural disconnect -- contaminating the adversarial run with "did the model unlearn
the decode-back format" noise instead of measuring genuine concealment behavior.

**Fixed, not left unconfirmed**: added `score_completion_v2_decode_back_aware()` to
`decode_back_bank.py` -- every other term computed on the ORIGINAL completion exactly
as before; only `p_cot` is recomputed with the decode-back line stripped out (via the
already-existing, tested `strip_decode_back_line`), so genuine literal leakage
elsewhere in the `Step N:`/`State:` lines is still fully caught. `total` corrected by
exactly the removed delta -- verified via a sign-error catch during testing (the first
implementation subtracted the correction in the wrong direction; caught by a direct
numeric assertion before deployment, fixed, re-verified). `r_task` itself is left
UNCHANGED, using the raw literal `<answer>` token -- intentional, not a gap: that IS
the actual task, and the ~19% residual Nib->Heads decode-back-mistranslation noise
(already fully characterized in `experiments/09c_sft_diagnostic/design.md`) is a known,
bounded quantity, disambiguated during THIS run via the new, separately-tracked
`decode_back_matches_own_trace_rate` milestone metric (task requirement 4) rather than
papered over inside the reward function. `verify_reward_invariant()` is
checkpoint-independent by construction and unaffected by this fix (only WHICH TEXT
`p_cot` scans changes, no term's range does) -- re-verified fresh anyway, per
instruction. 15 CPU tests in `test_decode_back_bank.py` cover the wrapper, including
one confirming genuine literal leakage elsewhere in the trace is still penalized.

**Launch script**: `sft_seeded_rl_decode_back.py`, adapted from `sft_seeded_rl_full.py`
with: the decode-back checkpoint + its freshly-computed, pinned adapter SHA256
(`2b3a26d5ff8584ae305d446a4b954030115e0196108dd780a4579ea736ec0274`, re-verified
identical after the instance's stop/restart, immediately before launch); the
decode-back-aware reward wrapper in place of the raw one; `decode_back_matches_own_
trace_rate` tracked at every milestone alongside `genuine_correct_among_nonliteral`
(task requirement 4); an extended zero-step sanity check that verifies the freshly-
loaded checkpoint reproduces not just its nonliteral rate but its ~81% final-answer
accuracy AND ~100% decode-back fidelity before any training starts (all three matched);
a dedicated, loudly-printed step-40-50 checkpoint block reporting cumulative advantage-
and KL-clamp engagement plus the recent grad_norm/KL windows (task requirement 5); and
a pre-flight empirical check confirming the reward wrapper's correction is never
negative on the checkpoint's own real zero-step completions, not just on synthetic
examples.

**Mandatory pre-launch hash check**: all 8 dependencies
(`sft_seeded_rl_decode_back.py`, `decode_back_bank.py`, `advantage_clamp.py`,
`kl_calibration.py`, `soft_stops.py`, `taxonomy.py`, `synthetic_bridge.py`,
`reward_v3.py`) verified identical between local and deployed remote copies -- GATE:
PASS on all 8.

**BASELINE result (correctness-only control, 150 steps, terminal_step=150,
hard_stop=None -- no breaker fired)**:
- Zero-step sanity check passed: fresh-loaded checkpoint reproduced 81.0%
  final-answer accuracy and 100% decode-back fidelity before any training, matching
  the SFT run's own report exactly.
- **`decode_back_matches_own_trace_rate` = 1.0 at all 30 milestones (steps 5-150),
  without a single exception.** The structural disconnect fix held completely under
  correctness-only RL pressure for the full run -- the answer never once stopped
  faithfully following the decode-back line.
- **`genuine_correct_among_nonliteral` fluctuated in a real, bounded band: min 0.571,
  max 0.810 (at step 5), mean 0.698, ending at 0.714 (step 150).** A real decline from
  the starting ~81% down to a ~57-76% range over the course of training -- but at
  every point, including its lowest observed value, comfortably above (>1.5x) the
  original weak checkpoint's ~28.6% floor. Since decode-back fidelity never wavered,
  this decline is NOT the structural disconnect reopening -- it traces to the
  underlying intermediate TRACKING quality itself eroding somewhat under RL pressure
  (taxonomy shift toward more `state_predictive_code`, fewer `correct_globally_
  consistent_code` classifications), a different and more tractable phenomenon than
  the bug this whole sub-investigation exists to work around.
- Advantage clamp engaged at 32.1% overall (385/1200 rows); KL clamp (D_MAX=2.0,
  ACTIVE) engaged at 12.5% overall (17,298/138,040 tokens) -- both well-behaved,
  non-degenerate rates, consistent across the explicit step-40-50 checkpoint's
  cumulative readouts, no runaway growth observed for either clamp against this
  meaningfully stronger starting policy.
- One `COMPLETIONS_CHANGED=0/21` flag at step 40 (vs. step 35) -- a brief plateau, NOT
  the sustained whole-run freeze found in an earlier task in this stage: completions
  resumed changing at every subsequent milestone (2/21 at step 45 onward).
- Pre-flight reward check re-run against this checkpoint's own real zero-step
  completions (not just synthetic examples): all `p_cot` corrections non-negative,
  confirming the fix only ever removes penalty, never adds it.

**Regression tests added**: `test_decode_back_baseline_evidence.py` (7 golden-record
tests against the retrieved evidence).

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-baseline-v1/`:
`stage9d_decode_back_baseline.json`, SHA256
`a7a23d3156cb31ada8ee0e488d97414abee3f41c180c6556fbdaf9dbd95cb577`;
`sft_seeded_rl_decode_back_baseline_stdout.log`, SHA256
`db0bf1f65f00d2643f93828dc10212f76f6d9c7538bcc13a2192245960c239c9` (both remote/local
identical).

**Status: BASELINE complete, reported for review before MAIN, per instruction --
same stop-and-check discipline as the original BASELINE+MAIN pair. The checkpoint
behaves sensibly: no breaker, no runaway clamp engagement, no reopened structural
disconnect, and genuine correctness stays well above the old floor throughout. MAIN
(full adversarial reward, decode-back-aware) not yet launched.**

## MAIN launch: grad_norm breaker fired at step 5 (2026-09-02)

Fresh pre-launch hash check performed (all 8 dependencies, identical local vs.
deployed remote, same as BASELINE's set). Instance capacity note: `g5.xlarge` hit
`InsufficientInstanceCapacity` on start; `g5.2xlarge` (same A10G 24GB GPU, established
mitigation from earlier tasks in this project) also hit it on the first attempt, then
succeeded on retry -- same EBS volume, checkpoint hash re-verified identical
(`2b3a26d5...`) after the type switch. Zero-step sanity check passed, matching all
three of BASELINE's own pre-training numbers (81.0% accuracy, 100% decode-back
fidelity, 100% nonliteral rate). Pre-flight reward-wrapper check re-confirmed clean
against this checkpoint's own real completions immediately before launch (all `p_cot`
corrections non-negative).

**MAIN hit the grad_norm breaker (>=50.0) at physical step 5 and stopped
immediately -- `terminal_step=5`, `hard_stop={'step': 5, 'grad_norm': 51.5, 'kl':
0.498}`.** Per this project's established, non-negotiable discipline (stated
explicitly in the script's own breaker-fired print): stopped immediately, no
automatic fix attempted, reported plainly here instead.

**The one milestone reached (step 5) shows no sign this is caused by, or coincides
with, either kind of degradation the task was watching for.** `decode_back_matches_
own_trace_rate = 1.0` (perfect, matching BASELINE's own step 5 exactly) and
`genuine_correct_among_nonliteral = 0.952` -- HIGHER than BASELINE's own step-5 value
of 0.810, not lower. Whatever caused the gradient spike, it did not manifest as
either a reopened structural disconnect or a drop in genuine correctness at the one
point where this could be checked.

**Comparison to BASELINE's own grad_norm behavior, since the question of whether this
was a config-level instability or a narrow one-off matters for interpretation:**
BASELINE itself came within ~2.5-4 points of the SAME 50.0 threshold three separate
times over its own 150-step run (steps 35/39/73: grad_norm 47.5/46.0/46.0), all
comfortably later in training, all safely under the line. MAIN's single breach (51.5)
happened much earlier -- physical step 5, exactly the step where the 5-step linear LR
warmup completes and the learning rate first reaches its full target value (2e-5,
confirmed directly in that step's own telemetry row) -- coinciding with an unusually
high-variance rollout group for that step (`reward_std=3.542`, one of the group's 8
completions hit the 256-token generation ceiling without terminating,
`completions/clipped_ratio=0.125`). This reads as a plausible LR-warmup-boundary
gradient-variance event interacting with the full adversarial reward's inherently
higher within-group reward variance (p_cot's continuous, completion-content-dependent
penalty vs. BASELINE's flatter correctness-only signal) -- not evidence of a
systematic instability specific to this checkpoint or a malfunction in the
decode-back-aware reward wrapper, but this is offered as the most likely reading of
the available telemetry, not a confirmed diagnosis; no root-cause investigation or
config change has been attempted, per instruction.

**Regression tests added**: `test_decode_back_main_evidence.py` (6 golden-record
tests, including explicit assertions that the breaker event does NOT coincide with
degraded correctness or fidelity, and a direct BASELINE-grad_norm comparison so the
"was this a narrow miss or an out-of-envelope config" question stays answerable from
the archived evidence alone).

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-main-v1/`:
`stage9d_decode_back_main.json`, SHA256
`08610c07f373310ee26cd8c31d0a26d7d8a2e751114f590c557b6e279020508b`;
`sft_seeded_rl_decode_back_main_stdout.log`, SHA256
`06128392ce5de20c4ee90cd80dbc21224eee3d9c2b998d9f2722825797469424` (both
remote/local identical).

**Status: MAIN did not complete -- it stopped at step 5/150 via the grad_norm
breaker, working exactly as designed. The full 30-milestone BASELINE-vs-MAIN
comparison this task's requirement 5 asks for is NOT possible from this run: there is
only one MAIN milestone to compare against BASELINE's 30. That one point shows no
degradation. No fix attempted or config change made, per instruction -- this is
reported as an open question for the next task to decide how to proceed (retry as-is
to test reproducibility, extend warmup, or treat this single data point as sufficient
signal on its own), not resolved here.**

## Diagnosis of the step-5 breaker: ordinary variance, not a reward mechanism (2026-09-02)

Pure CPU-side diagnosis (no GPU launch), from already-retrieved, hash-verified
evidence (BASELINE's full 150-step telemetry vs. MAIN's steps 1-5).

**Grad_norm trajectory, steps 1-5**: MAIN and BASELINE are essentially
indistinguishable across steps 1-4 (both bouncing in [0.03, 1.02]), then MAIN spikes
to 51.5 at step 5 while BASELINE stays tiny (0.11) -- a sudden, isolated event at step
5 specifically, not a gradual divergence from the start. This coincides exactly with
MAIN's step-5 completion group jumping from short (87-100 tokens, entropy~0.02-0.18)
to the 256-token generation ceiling with high entropy (1.089) -- a qualitative shift
in what got sampled, not a gradual drift.

**Are the new adversarial reward terms mathematically capable of causing this?** No.
At physical step 5, `annealed_cot_scale(step=5, warmup=5)` evaluates to exactly 0.2 (just
past its own warmup boundary) -- `p_cot`'s per-completion contribution is bounded to
~0.2 at this step, and `r_consistency`/`r_signal` are bounded to <=0.15 each. The
theoretical maximum additional spread the new terms could contribute at step 5 is
~0.5 total, nowhere near enough to explain the observed `reward_std=3.542` -- that has
to come from `r_task` itself varying within the group, the SAME mechanism that
already drives variance in BASELINE's own reward.

**Is this a NEW phenomenon, or one BASELINE already exhibits?** BASELINE hit the exact
same "long completion group" pattern 21 separate times over its own 150 steps
(mean_length>=200), with grad_norm ranging 2.97-47.5, kl 0.28-0.63, entropy 0.81-1.82,
reward_std 1.59-4.49 -- entirely under a correctness-only reward with zero adversarial
terms active. MAIN's step-5 values fall inside that same distribution on every
dimension except grad_norm, which is only ~8% above BASELINE's own observed max
(47.5, at its own step 35). **Verdict: (a) ordinary run-to-run variance in when a
long/high-entropy completion group gets sampled, not (b) a mechanistic consequence of
the adversarial reward interacting with the clamps.** No warmup/LR change recommended
-- the diagnosis doesn't point there (BASELINE hit the same phenomenon repeatedly at
LRs from 1.4e-5 to 1.8e-5, well past its own warmup, not just at the warmup boundary).

**Approved and acted on**: straight retry, same config, no changes -- plus fixing the
data gap this diagnosis exposed (per-row training-batch data wasn't persisted, only
aggregate `on_log` stats, limiting this diagnosis to a distributional argument).

**Row-level logging added** to `sft_seeded_rl_decode_back.py`: `diagnostic_reward`
now computes each row's token length via the tokenizer and, whenever the MAX (not
mean) length across a training step's group exceeds `LONG_COMPLETION_TOKEN_THRESHOLD`
(150 tokens), persists the WHOLE group's completion text + full reward-term breakdown
+ token length to `ROW_LEVEL_LOG` -- a max-based trigger deliberately chosen over a
mean-based one so a single long outlier inside an otherwise-short group still
triggers capture of the whole group for context (exactly the shape of the step-5
spike: one long row alongside seven short groupmates). If the breaker fires again,
the `on_log` callback now immediately prints the breaking step's captured rows
(completion text, every reward term, ground truth) directly to the log, not just
aggregate grad_norm/kl. Per-row entropy was deliberately NOT captured: it's computed
in a different internal call (`_get_per_token_logps_and_entropies`) than the reward
function, and reliably aligning its per-row order with the reward function's rows
would require unverified assumptions about TRL's internal micro-batching order -- a
wrong alignment could actively mislead a future diagnosis. Per-step aggregate entropy
remains fully available via the existing `on_log` telemetry, which combined with
row-level completion text and rewards is sufficient without that alignment risk.

**Mandatory pre-launch hash check**: all 8 dependencies re-verified fresh
(identical local vs. deployed remote) immediately before this relaunch. Checkpoint
hash re-confirmed identical (`2b3a26d5...`).

## MAIN retry: complete, 150/150 steps, no breaker (2026-09-02)

**The retry cleared step 5 cleanly** -- `terminal_step=150`, `hard_stop=None`,
directly confirming the (a) ordinary-variance diagnosis: the same config, same
checkpoint, same reward function ran the full 150 steps without incident this time.
The new row-level logging triggered twice during this run (steps 10 and 40, 8 rows
each, 16 total) -- confirming the instrumentation works correctly -- but neither
event caused a breaker; both were unremarkable, consistent with BASELINE's own
observation that long-completion groups usually pass without incident.

**The full, matched 30-milestone BASELINE-vs-MAIN comparison this whole re-run exists
to produce:**

| | BASELINE | MAIN |
|---|---|---|
| mean `genuine_correct_among_nonliteral` | 0.698 | **0.906** |
| min | 0.571 | 0.762 |
| max | 0.810 | 1.000 |
| final (step 150) | 0.714 | 0.857 |
| `decode_back_matches_own_trace_rate` | 1.0 at all 30 milestones | 1.0 at all 30 milestones |

**MAIN exceeded BASELINE at 29 of 30 matched milestones** (the sole exception: step
10, 0.762 vs. BASELINE's 0.810). Mean difference (MAIN - BASELINE) = +0.208, paired
t-statistic = 10.85 (df=29) -- far beyond any conventional significance threshold,
not explainable by the milestone-to-milestone noise already characterized elsewhere
in this stage. **MAIN's own MEAN (0.906) exceeds BASELINE's entire-run PEAK (0.810)**
-- this is not "decays less than BASELINE," it is uniformly, substantially higher
throughout.

**Plain verdict, against the three scenarios posed going in**: none of "decays
similarly," "decays faster," or "decays less" fits what the data actually shows.
There is no meaningful decay in MAIN at all relative to a flat high reference, and it
substantially outperforms BASELINE's own declining trajectory throughout the entire
run, not just at the end. This is the OPPOSITE of "adversarial pressure erodes
genuine correctness." **`decode_back_matches_own_trace_rate` stayed perfect (1.0) in
BOTH conditions at every one of 30 milestones each -- the structural fix from the
decode-back line held completely regardless of reward condition.**

Offered as a plausible mechanism, not confirmed further in this task (consistent
with not investigating beyond what was asked): the "full adversarial reward" is not
purely a concealment-pressure term working against correctness -- it also adds
`r_consistency` (rewards a single stable non-literal code) and `r_signal` (rewards
partial-credit correlation between the invented tokens and true physical state)
alongside `r_task`, terms BASELINE's correctness-only reward lacks entirely. These
provide direct, additional positive training signal toward exactly the tracking
behavior this checkpoint already does well, which may explain why the "full reward"
condition outperforms "correctness alone" rather than trading off against it. KL
clamp engagement in MAIN was also markedly lower than BASELINE's (1.16% vs. 12.5%
overall) -- consistent with a policy staying closer to its reference distribution
under the full reward, rather than drifting further from it.

**Mandatory pre-launch hash check**: performed fresh before this launch (all 8
dependencies, checkpoint hash re-verified after the g5.xlarge restart succeeded this
time -- see Instance capacity note below).

**Instance capacity**: `g5.xlarge` succeeded on the first attempt for this relaunch
(no `InsufficientInstanceCapacity` this time, unlike the prior MAIN launch which
needed a `g5.2xlarge` substitution).

**Regression tests added**: `test_decode_back_main_v2_evidence.py` (6 golden-record
tests, including a direct paired-comparison significance check and confirmation the
row-level logging infrastructure captured real events even though no breaker
occurred).

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-main-v2/`:
`stage9d_decode_back_main.json`, SHA256
`c7193ee8aa7b6a109917f7337fe6d269a086767f2eff38dbe124c2da52b903d8`;
`sft_seeded_rl_decode_back_main_v2_stdout.log`, SHA256
`c0c0cb68da06b6edc3778f441507d048c3cee17bcaa377a831aae84267161c9d` (both
remote/local identical).

**Status: MAIN retry complete, full BASELINE-vs-MAIN comparison done. This is the
clearest, most statistically decisive result this stage has produced: with a
starting policy that has real genuine correctness to lose (81.0% vs. the original
checkpoint's ~28.6%), the full adversarial reward does not suppress it -- MAIN
substantially and consistently outperforms the correctness-only control across the
entire run, while the decode-back structural fix holds perfectly under both
conditions. This directly answers the question this whole re-run existed to test.**

## SIGNAL-ONLY: isolating the CoT penalty from r_consistency/r_signal (2026-09-02)

MAIN changed TWO things relative to BASELINE at once (added the CoT penalty AND
added r_consistency/r_signal), so MAIN's win alone cannot say which addition is
responsible -- especially given the earlier step-5 diagnosis already showed the CoT
penalty's per-completion contribution is mathematically tiny (bounded ~0.2-0.5 at
early steps). This task isolates it directly: correctness + r_consistency + r_signal
at MAIN's own magnitudes (0.15 each), with `p_cot` fully zeroed via
`cot_min_scale=cot_max_scale=0.0` -- added as a third phase (`signal_only`) to
`sft_seeded_rl_decode_back.py`, reusing every other piece of MAIN's validated
machinery unchanged (checkpoint, clamps, decode-back-aware wrapper, row-level
logging, milestone cadence).

**Confirmed before any GPU time**: with `cot_max_scale=0`, `annealed_cot_scale`
returns 0 at every step regardless of the warmup/ramp schedule, forcing `p_cot=0` for
every completion -- which also makes the decode-back-aware wrapper's correction
(`raw_p_cot - stripped_p_cot`) identically 0, i.e. the wrapper is mathematically
inert under this config (verified directly, not just asserted: 5 CPU tests in
`test_signal_only_reward_params.py`, plus `verify_reward_invariant(cot_max_scale=0.0)`
re-checked fresh).

**Disclosed gap, found while writing tests against the retrieved evidence (harmless,
didn't affect the actual run)**: the pre-flight reward check block (printed before
training starts) never passes `**reward_params` to the wrapper, so its printed
corrections always preview MAIN-style default settings regardless of which phase is
launching -- a pre-existing omission from the original MAIN task's implementation,
invisible until a phase with genuinely different `reward_params` (SIGNAL-ONLY) made
it show a non-zero correction where zero was expected. This does NOT affect training
correctness: `diagnostic_reward`, the function actually used for every real training
step, DOES correctly pass `**reward_params` -- confirmed directly against the row-
level long-completion log (real training-batch rewards, not the pre-flight preview),
which shows `p_cot=0.0` on every one of its captured rows. Not fixed in this task
(the run this affects is already complete and was unaffected); worth fixing the
pre-flight-check call site before it's relied on again for a future phase.

**Mandatory pre-launch hash check**: all 8 dependencies re-verified fresh, checkpoint
hash re-confirmed identical (`2b3a26d5...`). `g5.xlarge` succeeded on the first
attempt.

**Result: 150/150 steps, no breaker.** Full matched 30-milestone three-way
comparison:

| | mean | min | max | final |
|---|---|---|---|---|
| BASELINE | 0.698 | 0.571 | 0.810 | 0.714 |
| SIGNAL-ONLY | 0.865 | 0.714 | 1.000 | 0.952 |
| MAIN | 0.906 | 0.762 | 1.000 | 0.857 |

SIGNAL-ONLY sits strictly between BASELINE and MAIN, but MUCH closer to MAIN:
SIGNAL-ONLY - BASELINE = +0.167 (paired t=5.99, df=29, decisive); MAIN - SIGNAL-ONLY
= +0.041 (paired t=-2.20 for SIGNAL-ONLY vs. MAIN, real but an order of magnitude
smaller effect). **The signal-shaping terms (r_consistency + r_signal) alone recover
roughly 80% of MAIN's total improvement over BASELINE; the CoT penalty contributes
the remaining ~20% -- real and statistically detectable, not zero, but clearly the
minor contributor, not the primary driver.**

**Plain verdict, against the three scenarios posed going in**: this is the third
scenario (partial contribution), reported with its actual magnitude rather than left
vague -- and that magnitude is heavily weighted toward "reward-shaping-for-tracking-
quality is doing most of the work." The project's finding should be reframed
accordingly: this is primarily evidence that positive reward shaping toward
consistent, state-correlated non-literal tracking improves genuine correctness, NOT
primarily evidence of genuine resistance to adversarial (CoT-penalty) pressure --
though the CoT penalty's own small positive contribution means it isn't purely inert
either.

**One notable anomaly**: `decode_back_matches_own_trace_rate` dropped to 0.952 (1/21
samples) at step 65 -- the FIRST sub-1.0 reading across all 90 milestones observed in
this stage's three 150-step runs (BASELINE's 30, MAIN's 30, this run's 30, all
otherwise perfect). Since `p_cot` is mathematically zero throughout this entire run,
this cannot be a decode-back-aware-wrapper failure -- there is no p_cot pressure to
have caused one. Recovered fully at the very next milestone (step 70, back to 1.0)
and stayed there through step 150. Reported as a single, isolated, immediately-
corrected organic fluctuation, not a trend.

**Regression tests added**: `test_signal_only_reward_params.py` (5 CPU-only,
pre-GPU tests confirming the wrapper's mathematical inertness under this config) and
`test_decode_back_signal_only_evidence.py` (7 golden-record tests against the
retrieved evidence, including the precise magnitude-of-contribution assertions and
the disclosed pre-flight-check gap's correct characterization).

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-signal_only-v1/`:
`stage9d_decode_back_signal_only.json`, SHA256
`bb4c9612c8f297f5d9a1a12f6a8bd7bc4f0c797bb9721cd1ccbb1dcde92409d5`;
`sft_seeded_rl_decode_back_signal_only_stdout.log`, SHA256
`556329f276bc61abd6d4335677d77049bc267e279b553ccea57aaca7fd3ea201` (both
remote/local identical).

**Status: SIGNAL-ONLY complete. The isolation this task set out to run is done and
quantified: MAIN's win over BASELINE is primarily (~80%) attributable to the positive
reward-shaping terms (r_consistency, r_signal), not the CoT penalty, which
contributes a real but clearly secondary (~20%) increment. This stage's finding is
best described as demonstrating that reward shaping toward consistent, correlated
non-literal state tracking improves genuine correctness under RL -- with a smaller,
additional, genuine (not zero) contribution from surviving active concealment
pressure specifically -- rather than as a clean demonstration of resistance to
adversarial pressure alone.**

## PENALTY-ONLY: completing the 2x2 factorial (2026-09-03)

Every condition tested so far had the bonus terms present (BASELINE has neither;
SIGNAL-ONLY has bonus only; MAIN has both) -- the penalty had never been tested
completely alone, with nothing telling the model what "good" non-literal usage looks
like. Real, specific concern going in: the penalty alone might reproduce Stage 1's
old failure mode (avoiding literal words via vacuous/degenerate output, not genuine
improved tracking) -- i.e. the bonus reward might be a NECESSARY complement, not just
an independently-helpful addition. This task runs the fourth, previously-untested
condition to find out.

**Config**: the exact mirror image of SIGNAL_ONLY. Added as a fourth phase
(`penalty_only`) to `sft_seeded_rl_decode_back.py`: `PENALTY_ONLY_REWARD_PARAMS =
dict(signal_magnitude=0.0, consistency_magnitude=0.0)` -- leaves `p_cot` at MAIN's own
defaults (`cot_min_scale=0.2, cot_max_scale=2.0`, fully active, decode-back-aware
wrapper doing REAL protective work again here -- confirmed on CPU before any GPU time:
`test_penalty_only_reward_params.py`, 6 tests, including a direct check that the
wrapper's correction grows monotonically as `cot_scale` ramps, unlike SIGNAL_ONLY's
always-zero correction).

**Also fixed** (the same-session gap disclosed as "worth fixing before it's relied on
again for a future phase" in the SIGNAL-ONLY writeup, and this IS that future phase):
the pre-flight reward check block now passes `**_reward_params` to the wrapper,
so its preview correctly reflects each phase's own actual reward config instead of
always previewing MAIN-style defaults. Verified this doesn't change behavior for the
three already-completed phases (their evidence-based golden-record tests re-run
clean, unaffected -- the fix only touches the pre-flight PREVIEW, never the actual
per-step training reward computed by `diagnostic_reward`, which already correctly
passed `**reward_params`).

**Mandatory pre-launch hash check**: all 8 dependencies re-verified fresh, checkpoint
hash re-confirmed identical (`2b3a26d5...`). `g5.xlarge` succeeded on the first
attempt. (AWS SSO session expired mid-task between the SIGNAL-ONLY and PENALTY-ONLY
launches -- same recurring pattern documented elsewhere in this project; paused, had
the user re-authenticate, re-verified via `aws sts get-caller-identity` before
proceeding. IP also changed mid-session, requiring a fresh security-group
authorization before SSH would connect.)

**Result: 150/150 steps, no breaker.** Full matched 30-milestone four-way comparison:

| | mean | min | max | final | vs. BASELINE |
|---|---|---|---|---|---|
| BASELINE (neither) | 0.698 | 0.571 | 0.810 | 0.714 | -- |
| PENALTY-ONLY (penalty only) | 0.778 | 0.619 | 0.952 | 0.714 | +0.079 (t=4.37) |
| SIGNAL-ONLY (bonus only) | 0.865 | 0.714 | 1.000 | 0.952 | +0.167 (t=5.99) |
| MAIN (both) | 0.906 | 0.762 | 1.000 | 0.857 | +0.208 (t=10.85) |

**PENALTY-ONLY falls strictly between BASELINE and SIGNAL-ONLY** -- significantly
above BASELINE (t=4.37, a real and decisive improvement, NOT "no help" or a
regression) but significantly below both SIGNAL-ONLY (t=-4.07) and MAIN (t=-8.12).
The penalty alone produces less than half of SIGNAL-ONLY's own improvement over
BASELINE (+0.079 vs. +0.167).

**Additive-model check** (a natural consequence of running the full 2x2, not asked
for explicitly but a useful internal-consistency read): if the two reward components
contributed independently and additively, MAIN's mean would be predicted as BASELINE
+ SIGNAL-ONLY's own gain + PENALTY-ONLY's own gain = 0.698 + 0.167 + 0.079 = 0.945.
Actual MAIN mean is 0.906 -- close to but modestly below the naive additive
prediction (by 0.038), consistent with the two components being roughly, not
perfectly, independent contributors, with a small amount of diminishing returns when
combined.

**The specific Stage-1-failure-mode signature this task asked to watch for did NOT
appear.** `nonliteral_rate` stayed at a perfect 1.0 across every one of PENALTY-
ONLY's 30 milestones -- zero reversion toward literal Heads/Tails usage anywhere in
the run. Taxonomy breakdown, totaled across all 30 milestones: 489
`correct_globally_consistent_code` (78%), 140 `state_predictive_code` (22%), 1
single `prompt_domain_echo` instance -- no meaningful presence of any
vacuous/degenerate-output category. **This refutes the strongest version of the
concern** (penalty alone reproduces Stage 1's collapse) while still supporting the
weaker, now-quantified version (the penalty alone is real but clearly less effective
than the bonus terms alone, and both together are needed for MAIN's full result).

**One anomaly, matching the pattern already seen once in SIGNAL-ONLY**:
`decode_back_matches_own_trace_rate` dipped to 0.952 (1/21 samples) at step 75 --
recovered fully at the very next milestone (step 80, back to 1.0) and stayed there
through step 150. A different step than SIGNAL-ONLY's own single anomaly (step 65),
consistent with these being isolated, self-correcting RL-training fluctuations rather
than a systematic weakness in either condition -- now observed once in 2 of the 4
conditions run, never in BASELINE or MAIN, never sustained.

**Row-level logging**: triggered 32 times (more than MAIN's 16), consistent with
PENALTY-ONLY having no bonus term encouraging concise/consistent output the way
r_consistency does -- none of these events preceded or coincided with the single
decode-back anomaly or caused any breaker.

**Regression tests added**: `test_penalty_only_reward_params.py` (6 CPU-only,
pre-GPU tests) and `test_decode_back_penalty_only_evidence.py` (10 golden-record
tests against the retrieved evidence, including explicit assertions that the
Stage-1-failure signature did not appear and the additive-model consistency check).

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-penalty_only-v1/`:
`stage9d_decode_back_penalty_only.json`, SHA256
`6f56311103a7f855dc76930e58789c1198928aef1d388b9a6c8d8ec332b7ca47`;
`sft_seeded_rl_decode_back_penalty_only_stdout.log`, SHA256
`85b89661e1d2420aa66bb9456f6d582f24b48e581b4c4eb888228f3190d4cf51` (both
remote/local identical).

**Status: PENALTY-ONLY complete. The full 2x2 factorial across all four conditions is
now done. Final, decisive answer: both reward components help independently (neither
one alone is useless, and critically, the penalty alone does NOT reproduce Stage 1's
degenerate-output failure mode), but they are NOT equally load-bearing and are NOT
purely redundant with each other -- the positive signal-shaping terms are the
dominant driver of MAIN's overall result, the CoT penalty is a real but clearly
secondary contributor, and combining both gets closest to (though not quite matching)
their naive additive sum. This closes the mechanistic question this whole stage was
built to answer.**

## MAIN re-run (v3): a saved checkpoint was needed for a follow-up diagnostic,
## and the re-run revealed MAIN's result does NOT reliably reproduce (2026-09-03)

A separate, follow-up task (testing whether MAIN's checkpoint still needs to visibly
write the decode-back line to answer correctly, or whether it's scaffolding no
longer depended on) needed to LOAD the trained MAIN checkpoint -- and none had ever
been saved. `run_phase` in `sft_seeded_rl_decode_back.py` trained in-memory,
evaluated via milestones, then discarded the model (`del trainer, model`) every
time, for every phase, throughout this whole stage. Fixed by adding an
`output_dir` parameter (only passed for the `main` phase) that saves
`model.save_pretrained(OUTPUT / 'final_adapter')` before the model is discarded --
BASELINE/SIGNAL-ONLY/PENALTY-ONLY's behavior is unchanged.

**Getting the checkpoint required re-running MAIN's full 150-step recipe** (the
exact same seed, config, and code as the original `stage9d-decode-back-main-v2` run
whose 0.906 mean result is reported above) -- flagged explicitly before launch as
NOT a guaranteed reproduction: GRPO rollout sampling is stochastic
(temperature=0.8, top_p=0.95), and with no forced-determinism flags and 8-bit
bitsandbytes kernels, even an identical seed doesn't guarantee an identical trained
model. This risk materialized. The re-run (`stage9d-decode-back-main-v3`) diverged
substantially from v2's trajectory starting around step 30 and never recovered:

| | mean | min | max | final (step 150) |
|---|---|---|---|---|
| BASELINE (for reference) | 0.698 | 0.571 | 0.810 | 0.714 |
| MAIN v2 (original, reported above) | 0.906 | 0.762 | 1.000 | 0.857 |
| **MAIN v3 (this re-run)** | **0.611** | **0.333** | 0.857 | 0.619 |

MAIN v3's mean is significantly BELOW even the correctness-only BASELINE control
(paired t=-5.06 across the 30 matched milestones), and dramatically below the
original MAIN v2 run (paired t=-10.41) -- despite identical seed, identical reward
config, identical code. `terminal_step=150`, `hard_stop=None` -- this is not a
breaker event, the run completed cleanly at a genuinely different, weaker outcome.

**Root-caused, not left as an unexplained regression.** `decode_back_matches_own_
trace_rate` stayed PERFECTLY at 1.0 across all 30 of v3's own milestones, exactly
like every other run in this stage -- the original structural disconnect (fixed by
the decode-back line, diagnosed via the corrupted-prefill test) has NOT reopened.
Tracing every one of v3's 245 wrong final answers (pooled across all 30 milestones)
directly: **100% (245/245) are decode-back LINE mistranslations of the model's own
correctly-tracked last state** -- e.g. `...State: Nib\nThe final code token Nib
decodes to Tails.\n<answer>Tails</answer>` (tracking is exactly right; the code
token the decode-back line references is exactly right; the ENGLISH WORD chosen for
that code is wrong). Zero of the 245 errors trace to a genuine intermediate-tracking
mistake or an unparseable answer. This is the SAME residual Nib->Heads
mistranslation bias already characterized and quantified in `experiments/
09c_sft_diagnostic/design.md`'s decode-back-line SFT task (there: ~25% of
Nib-ending held-out cases, never the reverse direction) -- not a new failure mode.

**What this means, precisely**: the reward signal used by both BASELINE and MAIN
(`r_task`, the literal `<answer>` token vs. ground truth) is downstream of this
residual translation bias and has no way to specifically target or protect against
it -- it can only reinforce whatever the sampled rollouts happen to do. Depending on
which stochastic trajectory a given RL run follows, this pre-existing SFT-inherited
bias can get differentially SUPPRESSED (v2's trajectory, ending near-perfect) or
AMPLIFIED (v3's trajectory, ending with every single error attributable to this one
mapping failure) over 150 steps of training. **MAIN's reported 0.906-mean result is
therefore not a reliably reproducible property of "the MAIN reward config" alone --
it is sensitive to which specific stochastic RL trajectory gets realized**, and that
sensitivity traces to a real, already-disclosed, pre-existing weakness in the SFT
seed checkpoint's decode-back translation step, not to any new problem introduced by
this stage's reward-shaping work.

**Consequence for the follow-up diagnostic task**: v3's checkpoint is a genuine,
cleanly-completed MAIN-recipe checkpoint, but it does not represent the reported
0.906 MAIN result -- using it as "the flagship checkpoint" for the decode-back-line
concealment/truncation diagnostic would characterize a materially different,
weaker-performing artifact than the one the project has been calling MAIN.
**Not proceeding to that diagnostic on this checkpoint without explicit direction --
reported here as an open decision, not resolved unilaterally.**

**Mandatory pre-launch hash check**: performed fresh (all 11 dependencies for this
combined session, including the new `truncation_lib.py`/`truncation_diagnostic.py`
prepared for the follow-up task). Checkpoint identity re-verified. `g5.xlarge`
succeeded on the first attempt.

**Evidence, retrieved and hash-verified**: into `experiments/
09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-main-v3/`:
`stage9d_decode_back_main.json`, SHA256
`1d0a42c04dec0bafa61547bce82fafebd50f5fab4ee531db4094a58be01a4bb3`;
`sft_seeded_rl_decode_back_main_v3_stdout.log`, SHA256
`41ca9b77bd10959809d05c5106139f18876918e08e58e3e80b65ef59f8837cfc` (both
remote/local identical). Saved adapter remains on the instance's EBS volume at
`~/aisi_checkpoints/stage9d-decode-back-main-v3/final_adapter/`, SHA256
`2d698a88f36d1946521fdd654f94a1e5c7e3a413de292b79dfd8ffe4fb2aa26f` (remote-verified,
not downloaded, matching this stage's established convention).

**Status: checkpoint-saving capability added and validated (a real gap fixed), but
the re-run itself surfaced an important, unplanned finding that reframes how
reproducible MAIN's headline result actually is. Truncation/concealment diagnostic
NOT yet run -- awaiting a decision on how to proceed (retry again hoping to
reproduce v2's trajectory; use v3 anyway with the caveat disclosed; or treat this
run-to-run sensitivity finding as itself the priority to investigate further)
before spending more GPU time.**

## MAIN variance-characterization batch: v4-v7 (2026-09-03) -- the true finding is
## a majority breaker rate, not just a wide accuracy distribution

Decision: characterize the TRUE variance of MAIN before trusting any single-run
number, and assess whether BASELINE/PENALTY-ONLY/SIGNAL-ONLY need the same
treatment, before proceeding to the truncation diagnostic on any checkpoint.

**Infrastructure**: `RUN_SEED` made env-overridable (`STAGE9D_RUN_SEED`, default
unchanged at 20260902) so each run in the batch uses a genuinely distinct seed (42,
43, 44, 45), not just repeated stochastic draws under the one hardcoded seed v2/v3
both used. New reusable analysis module `variance_analysis.py`: reconstructs ground
truth per scenario by rebuilding `eval_rows` (fully deterministic, same CLEAN21_SEED,
same fixed order every time) and zipping it against each milestone's samples by
index -- since the RL evidence's milestone samples never stored
starting_state/operations directly (unlike the SFT evidence). This enables a
RIGOROUS four-way error classification per wrong answer (genuine tracking error /
decode-back mistranslation / decode-back structural inconsistency / unparseable),
refining the faster internal-consistency-only check used for v3's initial trace (that
check couldn't distinguish "tracking wrong but internally self-consistent" from a true
mistranslation -- the rigorous version finds v2 itself has a real, non-zero 25.4%
genuine-tracking-error component, not the 0% the quick check implied). 6 CPU tests
(`test_variance_analysis.py`) validated against the already-retrieved v2/v3/BASELINE
evidence before use.

**Mandatory pre-launch hash check performed fresh before EACH of the 4 launches**
(all 8 dependencies re-verified every time, not just once for the batch, per
instruction). Evidence retrieved and hash-verified after each run, before launching
the next.

**Result: 5 of 7 total MAIN launches (71.4%) terminated via the grad_norm breaker
before reaching 150 steps -- v1 (the very first MAIN attempt, step 5), v4 (seed 42,
step 6), v5 (seed 43, step 39), v6 (seed 44, step 10), v7 (seed 45, step 32). Only 2
of 7 (v2, v3) completed the full run.** This is the dominant, most robust finding
from this batch -- more severe than "wide variance in the final accuracy number,"
since most launches never reach a comparable endpoint at all.

**Every one of the 4 new breaks shares the IDENTICAL signature already diagnosed for
v1**, confirmed by direct inspection of each breaking step's raw telemetry:
`completions/mean_length=256` (every one of the 8 rollouts in the group hit the
generation ceiling), `completions/clipped_ratio=0.125` (1/8 truncated without an
EOS), entropy in [1.02, 1.22] -- exactly the long/high-entropy rollout-group pattern
identified as the cause of v1's original breaker. This is now confirmed
independently 4 additional times, not a one-off: **the true rate at which this
configuration produces a long/high-entropy rollout group landing above
GRAD_BREAKER=50.0 is far higher than the original single-BASELINE-run's own
telemetry suggested** (BASELINE touched 47.5 three times across 150 steps without
crossing -- MAIN's launches are crossing at 51.5-64.0, values only modestly above
that same near-miss range). No new, previously-unseen failure mode was found in any
of the 4 new runs -- checked directly via `variance_analysis.py`, not assumed.

**genuine_correct_among_nonliteral for the 2 runs that DID complete**: v2=0.906,
v3=0.611 -- a ~30-point spread. **This is explicitly NOT a usable mean+-stdev
distribution**: n=2 is far too small for a trustworthy point estimate or standard
deviation, and there is a real, acknowledged survivorship-bias concern -- only
trajectories that happen not to trigger the breaker even get to report a 150-step
number at all, so the 2 observed values are not necessarily a random, representative
sample of "what MAIN would produce if the breaker didn't exist."

**Error-taxonomy tracing applied to every one of the 7 launches, not assumed to
transfer from v3** (task requirement 2): all 7 -- completed and aborted alike -- show
the SAME dominant mechanism, 67-100% of wrong answers attributable to decode-back
mistranslation or genuine tracking error combined (the two categories this stage's
methodology can cleanly distinguish), with `decode_back_structural_inconsistency`
appearing in exactly one case (v5, a single sample, 3.4% of that run's errors) --
noted as the first appearance of this category anywhere in the stage, vanishingly
rare, not a pattern. `decode_back_matches_own_trace_rate` never dropped below 0.952
in any run (one isolated dip in v5, immediately recovered, matching the pattern
already seen once each in SIGNAL-ONLY and PENALTY-ONLY's own single runs).

**Risk assessment for the other three conditions (task requirement 3)**: applying
the SAME `variance_analysis.py` tracing to BASELINE, SIGNAL-ONLY, and PENALTY-ONLY's
existing single runs shows **all three share the identical mistranslation-dominated
error profile** (96.3%, 89.4%, 92.9% of their own errors respectively) as MAIN's
runs -- direct evidence the underlying source of accuracy variance (the pre-existing,
checkpoint-level decode-back mistranslation bias) is NOT specific to MAIN or to
adversarial RL pressure; it is a property of the SFT-inherited checkpoint itself,
present regardless of which reward condition trains on top of it. There is **no
principled reason found to expect BASELINE/SIGNAL-ONLY/PENALTY-ONLY are more
stable** than MAIN on the accuracy-variance dimension. On the SEPARATE
breaker-rate dimension: BASELINE's own single run already showed grad_norm
approaching the same 50.0 threshold three times (max 47.5) without this stage ever
re-running BASELINE to see if a different seed crosses it -- the same long-completion
mechanism driving MAIN's breaker is a property of THIS TASK'S generation dynamics
(long, high-entropy completions occurring periodically under greedy-adjacent
sampling), not something introduced by MAIN's specific reward terms. **Flagged
explicitly: the entire reward-decomposition table (BASELINE=0.698,
PENALTY-ONLY=0.778, SIGNAL-ONLY=0.865, MAIN=0.906) should be treated as preliminary,
single-run point estimates, not as characterized, trustworthy values, until each
condition receives the same multi-seed treatment MAIN just got.**

**What the paper's actual claim should be, given what's now known**: NOT a
"MAIN improves over BASELINE by X +/- Y, p<0.05" framing -- that would overstate the
precision this evidence supports (n=2 for MAIN's own completed-run distribution is
not enough to support a confidence interval, and the comparison conditions have n=1
each). The defensible claim is closer to: "Under the current GRPO configuration
(advantage clamp=0.4, KL clamp D_MAX=2.0, GRAD_BREAKER=50.0), training frequently
(71% of MAIN launches observed) terminates early via a grad_norm safety breaker,
driven by a consistent, identified mechanism -- occasional long/high-entropy rollout
groups -- that BASELINE's own telemetry shows operating near the same threshold. Of
the launches that do complete, final accuracy varies substantially (61-91% observed
for MAIN specifically) and this variance traces predominantly to a pre-existing,
checkpoint-level decode-back translation bias whose amplification or suppression
appears sensitive to which stochastic RL trajectory is realized, not to the specific
reward condition -- the same error profile appears in every condition's own
single-run evidence. The reward-decomposition comparison (BASELINE vs. PENALTY-ONLY
vs. SIGNAL-ONLY vs. MAIN) should be re-run with multiple seeds per condition, and the
grad_norm breaker threshold reconsidered (a single isolated spike is currently
treated identically to genuine divergence, discarding potentially-recoverable runs),
before the current single-run point estimates are reported as characterized
findings."

**Mandatory pre-launch hash check**: performed fresh before each of the 4 new
launches (8 dependencies each time). `g5.xlarge` throughout, no capacity
substitution needed for any of the 4 launches.

**Regression tests added**: `test_variance_analysis.py` (6 CPU-only tests,
validated against real BASELINE/v2/v3 evidence before use) and
`test_main_variance_batch_evidence.py` (7 golden-record tests against all 7 MAIN
launches' evidence, including explicit assertions of the majority breaker rate, the
shared breaker signature, and the n=2 distribution's own limitations).

**Evidence, retrieved and hash-verified after EACH run before launching the next**,
into `experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-decode-back-main-v{4,5,6,7}/`
(each `stage9d_decode_back_main.json` + its stdout log, all remote/local hash-identical
-- SHA256 values recorded in each run's own retrieval step above and in the
regression tests). Saved adapters for all 4 remain on the instance's EBS volume,
matching this stage's established convention.

**Status: variance-characterization batch complete. The question this task set out
to answer is answered, but the answer is more severe than the task's own framing
anticipated: this is not primarily a "wide but usable" accuracy distribution to
report as mean+-stdev -- it is a configuration that fails to complete training the
majority of the time via a well-understood, now-repeatedly-confirmed mechanism, and
every other reward condition in this stage's factorial is equally under-
characterized from a single run. Truncation/concealment diagnostic still NOT run --
correctly deferred, since no single MAIN checkpoint from this batch can yet be
called representative of a properly-characterized "MAIN" result. Recommended next
steps (not undertaken in this task): (1) reconsider/recalibrate the grad_norm
breaker given its apparent over-sensitivity to a recoverable, single-step spike
rather than genuine divergence; (2) once a less-brittle training configuration is
established, re-run all four reward conditions across multiple seeds each before
reporting the decomposition table as a finding.**

## GRAD_BREAKER recalibration attempt: proposed 80.0, empirical validation shows the
## methodology itself is not sound -- NOT settled (2026-09-03)

Following recommendation (1) above: recalibrate GRAD_BREAKER using real data,
distinguishing genuinely benign long-completion spikes from genuinely pathological
ones, rather than picking a round number.

**Data gathering**: new module `grad_norm_calibration.py` pulls every long-
completion-group event (`completions/mean_length>=200`) from every SAME-CONFIG
(LR=2e-5, both clamps at their validated values, GROUP_SIZE=8) run's telemetry --
12 runs, 315 events -- classified as `survived` (from a run that reached its target
step count, i.e. training continued normally afterward in every case, by
construction) or `breaker_triggering` (the exact step GRAD_BREAKER=50.0 halted a
run). 7 CPU tests (`test_grad_norm_calibration.py`) validated against this real data
before proposing anything.

**Initial empirical picture (n=315)**: survived values ranged [0.68, 48.75] (n=310);
breaker-triggering values ranged [51.5, 64.0] (n=5) -- a clean 2.75-point gap.
Checked and rejected a standard Tukey/IQR fence (as used for the KL clamp elsewhere
in this stage) explicitly, not assumed: the distribution is heavily right-skewed
(median 3.3, mean 7.1, max 48.75), so a 1.5*IQR fence lands at ~11.7 -- it would
flag a large fraction of genuinely-survived events as outliers, the wrong failure
mode for a breaker. Also confirmed: every one of the 5 breaker-triggering events had
KL 7-12x BELOW KL_BREAKER=5.0 -- no elevated-KL signal at any of them, supporting
that these are grad_norm-magnitude artifacts of longer completions (more tokens ->
larger gradient sum), not genuine policy divergence (which correlates with elevated
KL too).

**Proposed threshold: GRAD_BREAKER=80.0** -- clears the entire observed
breaker-triggering range with a margin (80-64=16) exceeding that range's own width
(64-51.5=12.5), while staying same-order-of-magnitude (1.6x) rather than an
arbitrary jump. `GRAD_BREAKER` made env-overridable (`STAGE9D_GRAD_BREAKER`) to
support validation without a code change per launch.

**Mandatory pre-launch hash check**: performed fresh (8 dependencies). Checkpoint
identity re-verified.

**Validation (task requirement 4)**: re-ran the SAME 4 seeds that broke before
(42/43/44/45), short 50-step budget (not a full 150-step commitment), at
GRAD_BREAKER=80.0, specifically to see whether each reproduces a long-completion
event and, if so, whether it now survives. All 4 completed the full 50 steps with
no breaker fire -- retrieved and hash-verified after each run before launching the
next.

**Per-run results, checking actual grad_norm values at every long-completion event,
not just breaker pass/fail** (explicitly requested mid-task, before declaring
anything settled): seed 42 -- 1 event, max 48.25 (in line with the original range);
seed 44 -- 8 events, max 45.75; seed 45 -- 7 events, max 43.25; **seed 43 -- 1
event, grad_norm=72.0** (step 40, KL=0.867 -- still comfortably below KL_BREAKER,
5.8x margin).

**This single value changes the conclusion.** grad_norm=72.0 is HIGHER than ALL 5
of the original breaker-triggering values (51.5-64.0) -- proving the "survived" and
"breaker-triggering" classes from the initial 315-point sample were never actually
two separate populations at different scales; they are the SAME underlying
distribution, and the first sample simply hadn't captured its upper tail yet. Adding
17 more data points (the 4 validation runs' own long-completion events) pushed the
empirical survived-max from 48.75 to 72.0 -- a 48% jump from a modest amount of
additional sampling. The proposed 80.0 threshold now has only an 8-point margin
above the newly-observed maximum, the same "uncomfortably close" concern raised
before launch, this time confirmed with real data rather than hypothetically.

**Plain verdict: the empirical-max-plus-fixed-margin methodology is not sound for
this specific mechanism, and GRAD_BREAKER=80.0 is NOT being declared settled.**
Raw grad_norm for a long-completion event is fundamentally a MECHANICAL scaling
effect (more tokens summed into the per-step loss -> larger gradient norm), not
directly a policy-quality signal -- there is no evidence from the data gathered so
far that this quantity has a bounded ceiling; every additional batch of samples
could plausibly push the observed "benign maximum" higher again, the same pattern
just observed going from n=315 to n=332. Picking a bigger round number and
re-validating with another small batch would very likely repeat this exact
pattern rather than resolve it.

**What DID stay completely stable across all 22 examined high-grad_norm events (5
original breakers + 17 new validation events, no exceptions)**: KL remained far
below KL_BREAKER=5.0 in every single case, typically by a 5-12x margin, including
at the newly-discovered 72.0 grad_norm event. This is the reliable, discriminating
signal in this data, not raw grad_norm.

**Recommendation (not implemented in this task, per instruction 5 -- explicitly
deferred for a decision)**: two candidate directions, not mutually exclusive --
(a) lean on KL_BREAKER as the primary safety net for genuine divergence (already
doing that job with a consistent, wide margin) and treat GRAD_BREAKER as a much
coarser, higher backstop rather than a tightly-calibrated primary catch; or
(b) normalize grad_norm by completion token count before thresholding, since the
raw metric is mechanically confounded with completion length in a way a per-token
or length-normalized measure would not be. Both are larger changes than this task's
scope (a single-threshold recalibration) and are left for the next task's decision,
not undertaken here.

**Regression tests added**: `test_grad_norm_calibration.py` (7 CPU-only tests,
run before proposing any threshold).

**Evidence, retrieved and hash-verified after EACH validation run before launching
the next**, into `experiments/09d_sft_seeded_adversarial_rl/aws_runs/
stage9d-decode-back-main-v{8,9,10,11}/` (seeds 42/43/44/45 respectively) -- each
`stage9d_decode_back_main.json` + its stdout log, all remote/local hash-identical
(SHA256 values recorded at each retrieval step above).

**Status: recalibration attempt complete but explicitly NOT resolved -- reported
plainly rather than forcing a clean answer the evidence doesn't support, per this
project's established discipline. GRAD_BREAKER remains at its original 50.0 in the
committed script (the env-override defaults to 50.0; 80.0 was only ever used via
STAGE9D_GRAD_BREAKER for these validation launches, never adopted as the new
default) until a decision is made on which recommended direction to pursue. Full
multi-seed re-run across all four reward conditions (the next task per the prior
task's own scoping) should wait for that decision, since it depends on a training
configuration that reliably completes.**

## Length-normalized grad_norm: designed, validated, and REJECTED -- normalization is
## a mathematically real fix for the wrong population (2026-09-03)

**Task**: of the two directions left open above, evaluate direction (b) first
(length-normalized grad_norm as the primary breaker signal), since it addresses the
diagnosed mechanism directly rather than demoting the metric to a backstop. Fall back
to direction (a) (KL-as-primary) only if normalization proves impractical. All CPU-only
re-analysis of already-retrieved evidence -- no new GPU work needed for this task.

**Step 1 -- fit the normalization power.** Pulled every `(completions/mean_length,
grad_norm)` pair from all 12 SAME_CONFIG_RUN_PATHS runs' full telemetry (not filtered
to long-completion events only), spanning the complete observed length range 73-256:
1142 pairs. Raw Pearson correlation(length, grad_norm) = 0.446 -- confirms a real,
moderate confound across the full population. Log-log correlation = 0.690, noticeably
stronger, so fit `log(grad_norm) = a + b*log(length)` via OLS: **b = 2.706** -- a
much steeper power-law than a naive linear or sqrt scaling assumption. A grid search
over candidate powers `p`, computing `pearson(length, grad_norm/length^p)` directly on
raw (non-logged) normalized values, independently found the correlation crosses
through zero between p=2.0 (corr=0.119) and p=3.0 (corr=-0.129), with p=2.5 giving the
single closest-to-zero value (corr=-0.019) and the fitted exponent p=2.706 giving
corr=-0.069 -- close enough that the OLS-fitted exponent is preferred as the more
principled choice (derived directly from a regression, not a coincidence of a grid
search). Adopted **NORMALIZATION_POWER = 2.706**, `normalized_grad_norm(grad_norm,
mean_length) = grad_norm / mean_length**2.706` in `grad_norm_calibration.py`.

**Step 1b -- checked entropy as a possible second covariate, and rejected a
multivariate normalizer.** Every previously-examined high-grad_norm event also showed
elevated entropy (~1.0-1.5), raising the question of whether entropy independently
drives grad_norm alongside length. Fit `log(grad_norm) = a + b*log(length) +
c*log(entropy)` via OLS on the same 1142-pair population: the length exponent
**flips sign to b=-0.609** (entropy exponent c=0.905). This is NOT a genuine
effect reversal -- `pearson(length, entropy) = 0.933` in this population
(VIF = 1/(1-0.933^2) = 7.7, well into the range where individual multivariate
coefficients are unstable and not reliably interpretable). Length and entropy are
almost the same signal here; a two-covariate normalizer built on top of this
collinearity would not be defensible. Confirmed pure-length normalization at
p=2.706 already drives BOTH residual correlations near zero simultaneously
(corr(length, normalized)=-0.069, corr(entropy, normalized)=+0.034) -- because
they're so collinear, length alone already absorbs nearly all the entropy-driven
variation too. This supported proceeding with a pure-length normalizer, not a
combined one.

**Step 2 -- checked the proposal against the actual population that matters, and
it fails.** The 1142-pair regression spans the full 73-256 length range, but the
breaker threshold is only ever evaluated against the "long completion" regime
(`completions/mean_length >= 200`). Checked the length DISTRIBUTION within that
regime directly: of 315 long-completion events across the same 12 runs, **309
(98.1%) sit at EXACTLY 256.0** (the `MAX_NEW_TOKENS` generation ceiling), with the
remaining 6 spanning only 218-250 -- a narrow band, not real spread. All 22 of the
task's own key events (5 original breaker-triggering events at GRAD_BREAKER=50.0 +
17 validation-batch events from the seed 42/43/44/45 runs at
STAGE9D_GRAD_BREAKER=80.0, retrieved from `aws_runs/stage9d-decode-back-main-v{8,9,10,11}/`)
sit at length=256.0 with **zero exceptions**.

This means length-normalization is mechanically a **no-op** for the population the
threshold must actually be calibrated against: dividing 22 values that are all at the
same length by (very nearly) the same constant (256^2.706) cannot change their
relative spread at all. Verified numerically, not just argued: raw grad_norm across
the 22 key events has CV=0.478, spread ratio 72.00/4.44=16.23x; normalized grad_norm
has the identical CV=0.478 and spread ratio=16.23x, matching to better than 1e-6.
The huge apparent length-grad_norm correlation in the full 1142-pair population comes
entirely from comparing the SHORT (~73-199 token) regime against the LONG (~256
token) regime -- it says nothing about why grad_norm varies nearly 95x (0.68 to 64.0)
*within* the long-completion regime alone, which is the actual open question for
calibrating a threshold on this metric.

**Step 3 -- checked entropy as a fallback normalizer for the within-regime spread,
since it's the one covariate that does vary at fixed length, and it also fails.**
Within the 309 events at length=256.0, entropy ranges 0.472-1.817 and shows only a
moderate correlation with grad_norm (r=0.329 raw, r=0.445 log-log; log-log fit
`grad_norm ~ entropy^1.770`) -- far too weak to explain a 95x within-regime spread on
its own. Worse, checked directly against the 22 key events: correlation(entropy,
grad_norm) among just these 22 is **-0.476** -- the sign flips negative relative to
the larger 309-event population's +0.329, a real instability, not a typo. Normalizing
by entropy^p for p in {1.0, 1.5, 2.0, 2.5} makes the CV monotonically WORSE (0.478 ->
0.557 -> 0.601 -> 0.647 -> 0.694), not better. Entropy-normalization is rejected on
direct empirical grounds, not just theoretical caution.

**Conclusion: normalization proves impractical, per the task's own stated fallback
condition.** Neither length nor entropy (nor, by the multicollinearity argument in
Step 1b, any simple combination of the two) produces a stable, tighter range for the
population that actually matters for this calibration problem. This is a different,
more specific failure than the earlier max-plus-margin attempt's failure (which was
about an under-sampled tail): here the normalizer itself is mathematically sound for
the population it was fit on, but that population isn't the one the threshold needs
to discriminate within, because the long-completion regime has almost no length
variance of its own. Reported plainly rather than forcing a normalized threshold that
the data doesn't actually support.

**Falling back to direction (a): KL-as-primary, grad_norm-as-coarse-backstop.**
This is now the recommended direction, on the strength of evidence accumulated across
every sub-task in this whole recalibration effort, not just this one: of the 22 key
high-grad_norm events examined across BOTH the original calibration batch and the
GRAD_BREAKER=80.0 validation batch (grad_norm ranging 4.44-72.0, no stable ceiling
found by any method tried so far), KL stayed 5-12x below KL_BREAKER=5.0 in every
single case, including the most extreme grad_norm outlier (seed=43, grad_norm=72.0,
kl=0.867). KL has never once given a false negative or an ambiguous signal in this
dataset, consistent with Stage 4's original finding that KL -- not raw gradient
magnitude -- is this project's real indicator of genuine policy divergence. Concrete
proposed design (NOT yet implemented pending sign-off, consistent with this project's
practice of not unilaterally adopting a new default): keep the breaker condition as
`grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER`, but treat KL as the primary,
trusted signal and demote GRAD_BREAKER to a much coarser backstop threshold (e.g.
somewhere above the highest observed value in this dataset, 72.0, with the explicit
understanding that this offers no real precision -- it exists only to catch a
genuinely extreme, off-distribution event that KL might somehow miss, not to do any
fine discrimination of its own).

**Regression tests added**: `test_length_normalization.py` (9 CPU-only tests) --
confirms the confound is real in the full population, confirms normalization removes
it there, then confirms the decisive negative result (no-op on the 22 key events,
309/315 events at length=256.0, entropy normalization monotonically worsening CV,
entropy correlation sign instability between populations).

**What this task could NOT validate (explicitly, per the task's own framing)**: there
is still no genuinely-pathological (KL-elevated) event anywhere in this dataset, so
neither the original threshold, the rejected 80.0 proposal, nor this rejected
normalization attempt has ever been tested against a true positive -- every
conclusion in this whole recalibration effort rests on characterizing the benign
side only. This remains an open gap; it does not block adopting KL as primary, since
KL's reliability claim rests on Stage 4's original evidence, not on this dataset
having a positive example.

**Status: length-normalization direction closed out with a clear negative result.
KL-as-primary / grad_norm-as-coarse-backstop is now the recommended path forward, not
yet implemented -- awaiting sign-off before changing the committed breaker logic in
`sft_seeded_rl_decode_back.py`.**

## Settled configuration: KL_BREAKER primary, GRAD_BREAKER=200.0 coarse backstop --
## recalibration effort CLOSED (2026-09-04)

**Approved and implemented.** `sft_seeded_rl_decode_back.py` (the sole script
actively reused for ongoing Stage 9d launches) now uses:
- **KL_BREAKER = 5.0** (unchanged) -- the PRIMARY safety signal by design, not just by
  convention. Validated reliable across all 22 examined high-grad_norm events from
  every prior sub-task in this recalibration effort (5-12x below threshold in every
  case, including the highest observed benign grad_norm=72.0), consistent with
  Stage 4's original finding that KL, not raw gradient magnitude, is this project's
  real indicator of genuine policy divergence.
- **GRAD_BREAKER = 200.0** (raised from 50.0) -- a deliberately COARSE backstop, not a
  fine-grained discriminator. 2.78x the highest benign grad_norm value observed
  anywhere in this whole effort (72.0), within the proposed 2-3x range. This value is
  NOT derived from a calibration methodology the way KL_BREAKER's median+IQR
  derivation was -- both empirical-max-plus-margin (rejected: no stable ceiling found,
  seed=43 alone produced 72.0 against a "confirmed" 64.0 ceiling) and length/entropy
  normalization (rejected: mechanically a no-op for the population that matters, see
  above) were tried and explicitly failed to produce a metric this precise. 200.0
  exists purely as a catastrophic-explosion tripwire -- something is badly wrong if
  grad_norm ever reaches it, but clearing it is not evidence training is healthy;
  that judgment now rests on KL alone.
- The breaker condition itself is unchanged in form: `grad_norm >= GRAD_BREAKER or kl
  >= KL_BREAKER` (still an immediate hard stop, no auto-fix, by design). What changed
  is which term is trusted for fine discrimination.

**Re-validation**: the same 4 seeds used throughout this whole recalibration effort
(42/43/44/45) were re-run at a short 50-step budget -- the same scale that reliably
exercises this mechanism in every prior batch -- with the new settled config. Retrieved
and hash-verified as `aws_runs/stage9d-decode-back-main-v{12,13,14,15}/`.

| seed | run | long-completion events | max grad_norm | max KL | completed? |
|---|---|---|---|---|---|
| 42 | v12 | 5 | 48.75 | 0.60 | yes, 50/50 steps |
| 43 (pivotal) | v13 | 10 | 66.00 | 1.17 | yes, 50/50 steps |
| 44 | v14 | 3 | 54.25 | 0.47 | yes, 50/50 steps |
| 45 | v15 | 0 | -- | -- | yes, 50/50 steps |

All 4 seeds completed their full step budget with `hard_stop: None` -- none tripped
either breaker. seed43 (the seed that produced the pivotal 72.0 finding under the old
GRAD_BREAKER=80.0 batch) is the seed to watch here: this run's own independent
stochastic trajectory produced a different max (66.0, not a literal repeat of 72.0 --
expected, these are fresh launches with the same seed/config but different runtime
nondeterminism in generation, not byte-identical reruns), still comfortably under the
new 200.0 backstop. Max grad_norm across the whole fresh batch (66.0) stayed under
half of GRAD_BREAKER, and max KL (1.17) stayed under a third of KL_BREAKER --
confirming the backstop relationship holds on fresh data, not just the historical
22-event dataset the recommendation was originally built from.

**Regression tests**: `test_grad_breaker_settled_config.py` (5 tests, source-text
checks against `sft_seeded_rl_decode_back.py` confirming GRAD_BREAKER=200.0/
KL_BREAKER=5.0 are correctly baked into the script and the env-override mechanism
survives) + `test_settled_config_validation_batch.py` (5 tests against the
v12-v15 evidence above, including the seed43-survives-its-old-break-point check and
the KL-still-reliable-on-fresh-data check).

**What remains unvalidated, stated plainly**: there is still no genuinely-pathological
(KL-elevated) event anywhere in any evidence collected across this whole recalibration
effort. KL_BREAKER's reliability as a primary signal rests on Stage 4's original
evidence (where it WAS validated against genuine divergence), not on this dataset
containing a positive example -- this dataset only ever demonstrates the negative
side (KL stays low when training is healthy). This is a pre-existing limitation
carried forward, not a new gap introduced by this decision.

**Status: RECALIBRATION EFFORT CLOSED.** This is the settled configuration for all
remaining Stage 9d work, distinct from both prior attempts that were correctly NOT
adopted (empirical-max-plus-margin GRAD_BREAKER=80.0 -- rejected on validation-batch
evidence; length-normalized grad_norm -- rejected as a mechanical no-op for the
relevant population). This unblocks the next task: the full multi-seed re-run across
all four reward conditions (BASELINE/PENALTY-ONLY/SIGNAL-ONLY/MAIN) at this
configuration, to finally get a trustworthy characterization of the
reward-decomposition table -- not started here, pending separate confirmation to
proceed.

## Multi-seed reward-decomposition re-characterization, batch 1/2 (BASELINE + MAIN) --
## the original single-run comparison does NOT survive (2026-09-06)

**Task**: the actual goal this whole recalibration effort existed to unblock --
replace the unreliable single-run reward-decomposition numbers (BASELINE=0.698,
PENALTY-ONLY=0.778, SIGNAL-ONLY=0.865, MAIN=0.906) with a proper multi-seed
characterization, now that the settled config (KL_BREAKER=5.0 primary,
GRAD_BREAKER=200.0 backstop) should let MAIN complete reliably.

**Compute budget check (per task instruction 1)**: real step-time data from
completed 150-step runs gives ~60-70 min training time alone (BASELINE v1:
28.2s/step; MAIN v2: 23.5s/step), plus setup overhead -- realistically ~75-85 min per
full run. The full 16-20 run design (4 conditions x 4-5 seeds) would be ~20-28 hours
of sequential GPU time on one instance -- not a reasonable single-shot commitment.
Per the task's own explicit fallback, prioritized **BASELINE + MAIN first** (4 seeds
each, 8 runs, ~10-11 hours), reusing seeds 42/43/44/45 for continuity with the
recalibration dataset. Reporting here before starting PENALTY-ONLY/SIGNAL-ONLY, per
instruction 1's own stopping condition.

**Launches**: all 8 runs hash-checked immediately before launch (script SHA256
5601aa74...295fe5c, matched local vs remote every time, non-negotiable per-launch per
task instruction 2), using the settled `sft_seeded_rl_decode_back.py` unchanged.
Retrieved and hash-verified after each run:
- BASELINE: `aws_runs/stage9d-decode-back-baseline-v{2,3,4,5}/` (seeds 42,43,44,45)
- MAIN: `aws_runs/stage9d-decode-back-main-v{16,17,18,19}/` (seeds 42,43,44,45)

**Breaker status -- zero fires across the entire batch.** All 8 runs completed their
full 150-step budget with `hard_stop: None`. This is itself the headline result: MAIN
previously broke in 5 of 7 launches (71%) under the old GRAD_BREAKER=50.0; under the
settled config it completed cleanly in 4 of 4 fresh seeds, including seed=43 (the
seed that produced the pivotal grad_norm=72.0 finding during recalibration
validation -- here it topped out at 43.25, comfortably clear of the new 200.0
backstop). Max grad_norm across the whole batch: 59.5 (MAIN seed42). Max KL: 0.698
(MAIN seed42) -- under a seventh of KL_BREAKER=5.0 in every single case.

**Per-run results** (final-milestone `overall_final_answer_accuracy`, and the
error-taxonomy check from `variance_analysis.py` applied to every wrong sample across
every milestone in every run, not a sample of samples):

| condition | seed | final acc | max grad_norm | max KL | decode-back mistranslation frac | dominant? |
|---|---|---|---|---|---|---|
| BASELINE | 42 | 0.952 | 49.00 | 0.640 | 0.889 | yes |
| BASELINE | 43 | 0.905 | 48.75 | 0.477 | 0.792 | yes |
| BASELINE | 44 | 1.000 | 32.75 | 0.565 | 0.778 | yes |
| BASELINE | 45 | 1.000 | 11.38 | 0.570 | 0.750 | yes |
| MAIN | 42 | 0.952 | 59.50 | 0.698 | 0.730 | yes |
| MAIN | 43 | 0.810 | 43.25 | 0.614 | 0.923 | yes |
| MAIN | 44 | 1.000 | 10.12 | 0.453 | 0.500 | **no (exact 50/50 split)** |
| MAIN | 45 | 0.905 | 56.00 | 0.503 | 0.859 | yes |

Decode-back mistranslation is the dominant error source (>50% of wrong samples) in
**7 of 8 runs** -- confirming the known SFT-inherited translation bias, not a new or
different mechanism, is still the dominant error source under the settled config too.
MAIN seed=44 is the sole exception, an exact 50/50 split with genuine tracking error
-- notable but a single instance, not a pattern.

**Per-condition distribution (final-milestone accuracy, n=4 seeds each)**:

| condition | mean | sd | min | max |
|---|---|---|---|---|
| BASELINE | 0.9643 | 0.0456 | 0.9048 | 1.0000 |
| MAIN | 0.9167 | 0.0813 | 0.8095 | 1.0000 |

Cross-checked against a second, more stable metric (mean of
`genuine_correct_among_structural_nonliteral` across ALL milestones within each run,
not just the single final step): BASELINE mean=0.9421 (sd=0.0354), MAIN mean=0.8694
(sd=0.0833). Same direction, same conclusion.

**Statistical comparison (Welch's t-test, unequal variance assumed given MAIN's
visibly larger spread)**: computed by hand (no scipy in this environment; verified
against the standard Welch-Satterthwaite formula, cross-checked with a numerical
t-distribution tail integral) --
- Final-milestone accuracy: t=-1.02, df=4.72, **p=0.36**
- Mean-across-milestones accuracy: t=-1.60, df=4.05, **p=0.18**

Neither is anywhere close to conventional significance. With n=4 per condition,
this comparison has low power -- it cannot rule out a small-to-moderate true effect
in either direction. What it DOES rule out is the original claimed effect: the
single-run comparison's implicit claim (MAIN=0.906 decisively beats BASELINE=0.698)
does not hold up as a stable, seed-independent effect.

**The actual, defensible conclusion (per task instruction 6, stated plainly, not
rounded toward the original narrative)**:
- **Does MAIN still significantly outperform BASELINE?** No. Not only is the
  difference not significant, the point estimate goes the OPPOSITE direction --
  BASELINE's multi-seed mean (0.964) is higher than MAIN's (0.917) on final
  accuracy, and higher on the more stable mean-across-milestones metric too
  (0.942 vs 0.869). The original single-run finding does NOT survive
  re-characterization.
- **Does bonus-dominate-penalty (SIGNAL-ONLY > PENALTY-ONLY) hold up?** Not yet
  assessable -- those two conditions are batch 2, not run in this task.
- **Is the near-additivity finding still supported?** Not yet assessable for the
  same reason; requires all four conditions' multi-seed distributions.
- **What DOES survive**: the decode-back mistranslation error profile is consistent
  and dominant across both conditions and nearly every seed (7/8 runs), and the
  settled breaker configuration is fully validated under real full-scale multi-seed
  load -- zero breaker fires across 8 fresh full 150-step runs, including the
  historically pivotal seed=43.
- **What this means for the paper**: the original BASELINE-vs-MAIN comparison table
  is superseded by this one. The honest headline is that MAIN's advantage over
  BASELINE, as originally reported from single runs, is not currently supported by
  multi-seed evidence -- either the true effect is much smaller than the single-run
  numbers suggested, or these two conditions are genuinely close and single-run luck
  (in either direction) explains the original gap. Distinguishing between these
  requires either more seeds per condition or the completion of the SIGNAL-ONLY/
  PENALTY-ONLY batch to see if the same instability appears there too.

**Regression tests**: `test_reward_decomposition_multiseed_batch1.py` (9 tests) --
confirms zero breaker fires, the per-condition means/sds, the ordering flip relative
to the original single-run claim, the Welch's-test non-significance, MAIN's higher
variance, and the mistranslation-dominance finding across 7/8 runs.

**Evidence retrieved and hash-verified**: all 8 runs' JSON + stdout logs under
`aws_runs/stage9d-decode-back-{baseline,main}-v{2..5,16..19}/`.

**GPU teardown**: `TEARDOWN FULLY CONFIRMED` after this batch, before starting the
analysis above (no further GPU work needed for batch 1's own conclusions).

**Status: BATCH 1 OF 2 COMPLETE. Reported here per task instruction 6's explicit
stopping condition -- NOT proceeding to PENALTY-ONLY/SIGNAL-ONLY without separate
confirmation.** The settled breaker configuration is now fully validated under
production multi-seed load (this task's own secondary but important confirmation).
The primary finding -- that MAIN's originally-claimed advantage over BASELINE does
not survive proper multi-seed characterization -- is exactly the kind of result this
whole recalibration effort was designed to be able to surface honestly, rather than
continuing to build the paper's narrative on unreliable single-run point estimates.

## Multi-seed reward-decomposition re-characterization, batch 2/2 (PENALTY-ONLY +
## SIGNAL-ONLY) and the FINAL, complete four-condition result (2026-09-06)

**Task**: complete the reward-decomposition re-characterization -- PENALTY-ONLY and
SIGNAL-ONLY, 4 seeds each (42/43/44/45), same settled config, same methodology as
batch 1. Explicitly instructed to go in prepared for a clean null across all four
conditions and to report that plainly if the data supports it, rather than searching
for a way to preserve the original bonus-dominates-penalty or near-additivity
narrative.

**Launches**: all 8 runs hash-checked immediately before launch (non-negotiable
per-launch, same discipline as batch 1). Retrieved and hash-verified after each run:
- PENALTY-ONLY: `aws_runs/stage9d-decode-back-penalty_only-v{2,3,4,5}/` (seeds
  42,43,44,45)
- SIGNAL-ONLY: `aws_runs/stage9d-decode-back-signal_only-v{2,3,4,5}/` (seeds
  42,43,44,45)

**Breaker status -- zero fires, all 16 runs (both batches) now confirmed.** All 8
batch-2 runs completed their full 150-step budget with `hard_stop: None`. Max
grad_norm across batch 2: 67.0 (PENALTY-ONLY seed45), max KL: 0.939 (SIGNAL-ONLY
seed44) -- both comfortably clear of GRAD_BREAKER=200.0 and KL_BREAKER=5.0
respectively. Combined with batch 1, the settled configuration held cleanly across
the entire 16-run, four-condition, full-scale production characterization -- the
single most direct validation of the recalibration decision available.

**Per-run results (final-milestone accuracy)**:

| condition | seed | final acc | max grad_norm | max KL | mistranslation frac | dominant? |
|---|---|---|---|---|---|---|
| PENALTY_ONLY | 42 | 1.000 | -- | -- | 0.919 | yes |
| PENALTY_ONLY | 43 | 0.667 | -- | -- | 0.831 | yes |
| PENALTY_ONLY | 44 | 0.905 | -- | -- | 0.762 | yes |
| PENALTY_ONLY | 45 | 1.000 | 67.00 | 0.611 | 0.865 | yes |
| SIGNAL_ONLY | 42 | 0.857 | 59.75 | 0.592 | 0.560 | yes |
| SIGNAL_ONLY | 43 | 0.952 | 51.25 | 0.609 | 0.959 | yes |
| SIGNAL_ONLY | 44 | 0.810 | 47.25 | 0.939 | 0.701 | yes |
| SIGNAL_ONLY | 45 | 0.810 | 44.50 | 0.596 | 0.850 | yes |

All 8 batch-2 runs show decode-back mistranslation as the dominant error source
(>50%). Combined with batch 1's 7/8, mistranslation dominates in **15 of 16 runs
across all four conditions** -- overwhelming, consistent confirmation this is a
structural property of the checkpoint's decode-back translation, not something any
reward configuration meaningfully changes.

**FINAL four-condition table (final-milestone accuracy, n=4 seeds each) -- this
supersedes both the original single-run table and batch 1's partial result**:

| condition | original single-run value | multi-seed mean | sd | min | max |
|---|---|---|---|---|---|
| BASELINE | 0.698 | **0.9643** | 0.0456 | 0.9048 | 1.0000 |
| MAIN | 0.906 | **0.9167** | 0.0813 | 0.8095 | 1.0000 |
| PENALTY_ONLY | 0.778 | **0.8929** | 0.1573 | 0.6667 | 1.0000 |
| SIGNAL_ONLY | 0.865 | **0.8571** | 0.0673 | 0.8095 | 0.9524 |

Cross-checked against the more stable mean-across-all-milestones metric (not just
the single final step): BASELINE=0.9421 (sd=0.0354), MAIN=0.8694 (sd=0.0833),
PENALTY_ONLY=0.8381 (sd=0.1486), SIGNAL_ONLY=0.8321 (sd=0.0926). Same ordering, same
conclusion.

Notable: EVERY multi-seed mean is lower than or roughly comparable to BASELINE's,
the opposite of the original narrative where BASELINE was the weakest condition.
PENALTY_ONLY shows by far the largest spread (sd=0.157 on final accuracy) --
driven mostly by seed=43's rough run (final_acc=0.667, 236 wrong samples) --
underscoring how much a single seed can swing a reward condition's apparent
ranking.

**All 6 pairwise Welch's t-tests (final-milestone accuracy, n=4 per condition,
unequal variance assumed)**:

| pair | mean diff | t | df | p |
|---|---|---|---|---|
| BASELINE vs MAIN | +0.048 | 1.02 | 4.72 | 0.357 |
| BASELINE vs PENALTY_ONLY | +0.071 | 0.87 | 3.50 | 0.439 |
| BASELINE vs SIGNAL_ONLY | +0.107 | 2.63 | 5.27 | **0.044** |
| MAIN vs PENALTY_ONLY | +0.024 | 0.27 | 4.50 | 0.800 |
| MAIN vs SIGNAL_ONLY | +0.060 | 1.13 | 5.80 | 0.304 |
| PENALTY_ONLY vs SIGNAL_ONLY | +0.036 | 0.42 | 4.06 | 0.698 |

(Computed by hand -- no scipy in this environment -- via the standard
Welch-Satterthwaite formula and a numerical t-distribution tail integral;
cross-checked against the `|t|<2` conservative significance proxy used in the CPU
tests.)

**One pair (BASELINE vs SIGNAL_ONLY) reaches nominal p<0.05 -- but this does not
survive scrutiny.** With 6 pairwise comparisons run on n=4-per-condition data,
Bonferroni correction requires p<0.0083 for significance; p=0.044 does not clear
this bar. More decisively, the SAME comparison on the more stable
mean-across-milestones metric gives p=0.093 -- not even nominally significant.
One-in-six comparisons landing under 0.05 by chance alone is close to the expected
rate under the null (0.05 x 6 ~ 0.3 expected "hits"), so this is exactly what
uncorrected multiple comparisons predict, not evidence of a real effect. Reported
explicitly, not treated as a positive finding.

**The actual, complete, defensible conclusion (per task instruction, stated plainly,
not rounded toward the original narrative):**
- **No reward condition differs significantly from any other at this sample size.**
  All 6 pairwise comparisons are non-significant after correcting for multiple
  comparisons; the one nominally-significant uncorrected result does not replicate
  on a second, more stable metric. This is a clean null, not a partial or
  inconclusive result to be spun positively.
- **Does MAIN still significantly outperform BASELINE?** No (established in batch 1,
  confirmed here) -- and BASELINE's point estimate is actually higher.
- **Does bonus-dominate-penalty (SIGNAL-ONLY > PENALTY-ONLY) hold up?** No. Their
  means are nearly identical (0.857 vs 0.893 final-accuracy; 0.832 vs 0.838
  mean-across-milestones), difference not remotely significant (p=0.698). The
  original single-run values (0.865 vs 0.778) suggested a real gap; that gap does
  not survive re-characterization either.
- **Is the near-additivity finding still supported?** Not in any form that survives
  scrutiny -- with no pairwise difference among any of the four conditions reaching
  significance, there is no reliable structure in this data to check additivity
  against. The premise (that the conditions differ in a decomposable way) itself
  isn't supported.
- **What DOES survive, robustly, across all 16 runs**: (1) the settled breaker
  configuration -- zero fires across every condition and seed; (2) the decode-back
  mistranslation error profile -- dominant in 15/16 runs, confirming this checkpoint's
  translation bias is a structural property independent of reward shaping; (3) the
  reward-decomposition table's honest headline finding itself: **at n=4 seeds per
  condition, this experiment cannot distinguish BASELINE, MAIN, PENALTY-ONLY, or
  SIGNAL-ONLY from each other.**
- **What this means for the paper**: the original single-run reward-decomposition
  table (and its associated "MAIN wins," "bonus dominates penalty," and
  "near-additive" narrative claims) should NOT be reported as findings. The honest
  result is a null: within the precision this experiment can achieve, none of the
  four reward-shaping variants produces a detectably different outcome on this task.
  If a real difference exists, it is smaller than this study (n=4/condition) can
  reliably detect -- a legitimate, reportable limitation, not a failure of the
  investigation. A larger n per condition (10+) would be needed to meaningfully
  re-open this question; that is a scoping decision for a future task, not
  undertaken here.

**Regression tests**: `test_reward_decomposition_multiseed_final.py` (9 tests) --
confirms all 16 runs loaded, zero breaker fires in batch 2, the PENALTY_ONLY/
SIGNAL_ONLY per-condition means, that all 6 pairwise comparisons are computed, that
NONE survive Bonferroni correction, that exactly one (BASELINE vs SIGNAL_ONLY) is
nominally significant uncorrected, and the 15/16 mistranslation-dominance finding.
Together with batch 1's `test_reward_decomposition_multiseed_batch1.py` (9 tests),
the full re-characterization is captured in 18 CPU-only regression tests.

**Evidence retrieved and hash-verified**: all 16 runs' JSON + stdout logs under
`aws_runs/stage9d-decode-back-{baseline,main,penalty_only,signal_only}-v*/`.

**GPU teardown**: `TEARDOWN FULLY CONFIRMED` after the full 16-run investigation
completed.

**Status: REWARD-DECOMPOSITION RE-CHARACTERIZATION COMPLETE.** This four-condition,
multi-seed table -- and its clean-null conclusion -- is the final, authoritative
result for the paper, superseding the original single-run table entirely. This is
the intended endpoint of the whole chain of work in this file: recalibrating the
breaker (twice, rejecting two unsound approaches before settling on KL-as-primary),
validating that settled config under real multi-seed load, and finally using it to
discover that the original reward-decomposition narrative was an artifact of
single-run variance, not a real effect.

## Decode-back truncation diagnostic on the best-scoring saved checkpoint --
## COLLAPSED result: the decode-back line is load-bearing, not optional scaffolding
## (2026-09-06)

**Task**: since the reward-decomposition investigation found a clean null (no
condition is special), select the single best-performing checkpoint from all 16
completed multi-seed runs and run the decode-back truncation diagnostic on it --
the actual test of whether this project has produced steganographic encoding,
independent of which reward condition trained it.

**Step 1 -- identify the best-scoring run.** Pulled `genuine_correct_among_
structural_nonliteral` from all 16 runs' final milestones. Five-way tie at 1.0000:
BASELINE seed44, BASELINE seed45, MAIN seed44, PENALTY_ONLY seed42, PENALTY_ONLY
seed45. Broke the tie using the more stable mean-across-all-milestones metric (the
same one used throughout the reward-decomposition analysis): **BASELINE seed44**
wins outright (mean=0.9429, `decode_back_matches_own_trace_rate` perfect across every
milestone, fewest total wrong samples of the tied group). Noted explicitly, per the
task's own instruction: this identifies the single best INDIVIDUAL sample, not
evidence BASELINE-as-a-condition is superior -- already established false by the
clean-null result above.

**Step 2 -- checkpoint availability check (the task's own explicit caution, and it
mattered).** `run_phase()`'s checkpoint-saving code path (`model.save_pretrained`)
is only ever invoked when `output_dir` is passed, which happens **only for
`PHASE=='main'`** -- confirmed by reading the phase-dispatch block directly
(`sft_seeded_rl_decode_back.py` lines 698-712). BASELINE, SIGNAL_ONLY, and
PENALTY_ONLY never save a checkpoint, in any run, at any seed -- this was true
for every one of the 16 multi-seed runs, not just this one. So BASELINE seed44's
checkpoint does not exist. **Fell back to the best-scoring run among those actually
saved**: since only MAIN-phase runs ever save, this restricts to the 4 MAIN seeds,
and the best of those is **MAIN seed44** (`stage9d-decode-back-main-v18`) -- itself
tied for the overall best score (1.0000), so the fallback cost nothing in practice.
Confirmed on the remote instance: `final_adapter/adapter_model.safetensors` (7.4MB)
present and complete, hash-verified before use
(`3f693c1b6bde03eb55164065b64e236acca0c120158bdebe90a5670698aa30a4`).

**Step 3 -- ran the existing `truncation_diagnostic.py`, found and fixed a real
integration bug on first use.** This script (built earlier, never previously run --
explicitly noted in the task) crashed on its first launch with `KeyError:
'final_answer_correct'` inside `summarize_subset()`. Root cause: `truncation_lib.py`'s
`summarize_subset()` was validated in isolation (`test_truncation_lib.py`, 9 tests,
all passing) against hand-built dicts using the correct key name, but
`truncation_diagnostic.py`'s own results-dict-building code (GPU-only, never
CPU-testable) used a different key (`normal_final_answer_correct`) for the same
value. Nothing caught this because the two modules had never actually run together
end-to-end before this task. Fixed by renaming the key to `final_answer_correct`,
matching both `summarize_subset()`'s expectation and `score_normal_completion()`'s
own output key. Added `test_truncation_diagnostic_result_shape.py` (3 tests,
source-inspection-based) specifically to close this gap going forward -- checks the
actual dict-literal keys `truncation_diagnostic.py` builds against what
`summarize_subset()` requires, without needing a GPU to run.

**Step 4 -- the diagnostic itself, re-run cleanly after the fix.** Inference only,
no training. Fresh, independent checkpoint load (same isolation pattern used
throughout this stage), identity hash confirmed
(`79f954b027853840e4bc1ede6339da64d3b8c20cba5864e42c2571a07762b14d`). Normal
(untruncated) generation on the full held-out-21: 90.5% accuracy (19/21) -- 2
scenarios excluded per the eligibility restriction (task requirement 1: only
scenarios where normal generation is currently correct). Truncated generation:
decode-back line entirely removed, `<answer>` forced open immediately after the
last tracked-state line, 40 new tokens allowed.

**Result -- reported per direction, NOT pooled (task requirement 4)**:

| subset | n | normal accuracy | truncated accuracy | no-parseable-answer rate |
|---|---|---|---|---|
| Heads-truth | 13 | 1.000 | **0.000** | 0.000 |
| Tails-truth | 6 | 1.000 | **0.000** | 0.000 |
| (pooled, reference only) | 19 | 1.000 | 0.000 | 0.000 |

**Not a formatting failure and not random wrong-guessing -- a precise, mechanistic
collapse.** Every one of the 19 truncated completions produced a parseable answer
(0% no-parseable-answer rate) -- the model always closes `<answer>...</answer>`
correctly. But inspecting the actual truncated answers: **all 19 are the literal
internal code token itself** (`<answer>Nib</answer>` or `<answer>Nomo</answer>`),
never the translated word ("Heads"/"Tails") -- confirmed directly, not inferred:
`truncated_answer in {'nib','nomo'}` for all 19 rows, zero landing on
`{'heads','tails'}` even by accident. Cross-checked against the model's own state
tracking: **18 of 19 (94.7%) truncated answers exactly match the model's own last
tracked-state code token** from its normal completion. State tracking is intact and
carried forward faithfully under truncation; what's missing is specifically the
Nib->Heads / Nomo->Tails translation step, which appears to depend entirely on
generating the explicit decode-back sentence -- there is no evidence of a separate,
internalized translation mechanism that survives its removal.

**Plain verdict (task requirement 5): COLLAPSED.** Truncated accuracy (0.0) is not
"close to normal" (0.905) and not "in between" -- it is a complete floor-level
collapse in both direction subsets independently. The decode-back line is NOT
scaffolding the model has outgrown; it is the sole mechanism by which the internal
Nib/Nomo code gets translated into the literal Heads/Tails answer on this checkpoint.
This directly bears on the project's steganography framing: a monitor reading the
full completion text (with no carve-out for the decode-back line, unlike this
project's own training-time CoT-penalty scan) would see the translation happen in
plain language, in exactly the sentence a real monitor would flag. There is no
evidence, from this diagnostic, of the model encoding-then-decoding internally
without saying so -- the opposite is demonstrated: remove the saying-so, and the
decoding does not happen at all.

**Regression tests**: `test_truncation_diagnostic_result_shape.py` (3 tests, the
integration-bug regression guard) + `test_truncation_diagnostic_evidence.py` (9
tests, golden-record against the real evidence) -- 12 new CPU tests, all passing,
alongside the pre-existing `test_truncation_lib.py` (9 tests, unaffected).

**Evidence retrieved and hash-verified**:
`aws_runs/stage9d-truncation-diagnostic-v1/stage9d_truncation_diagnostic.json` +
stdout log.

**Inference-only GPU usage, normal hygiene teardown** (per task's own reduced
discipline requirement for this task) -- `TEARDOWN FULLY CONFIRMED` after retrieval.

**Status: TRUNCATION DIAGNOSTIC COMPLETE, COLLAPSED result reported. NOT proceeding
to the causal-flip test, per explicit task instruction -- stopping here for a
decision.** This is a materially important finding for how this project's results
should be framed: on the best-available checkpoint, the observed behavior looks like
"encode internally, then narrate the decode step explicitly in monitorable text,"
not "encode and decode without ever saying so." Whether this motivates a redesign
(e.g., training pressure to internalize the translation, or accepting this as the
honest current state of the checkpoint) is a scoping decision for the next task, not
made here.
