# Step 14b: rollout-group injection on top of the hard reward gate

## Why (the actual observed failure, not a hypothetical)

Step 14's full 150-step run (`experiments/09_direct_indomain_synthetic_bridge/README.md`
item 15, `step14_design.md`) completed with 0 non-literal completions across 2,200
training rollouts and 1,443 milestone-eval completions. Root cause: from full anneal
(step 30) onward, ~90%+ of accepted bank-scenario training groups either hit the
dynamic-sampling all-zero-reward fallback or landed at exact zero anyway (73/127
group-attempts, 28/54 accepted groups forced to fallback, 21/54 more at exact zero).
GRPO had no usable relative-advantage signal on the large majority of bank-scenario
training rounds, because resampling draws from the same policy that has never
produced anything but literal/vacuous on these scenarios -- and across this
project's entire history (3,850+ prior tracked rollouts, plus these 2,200), the
model has never once spontaneously produced a non-literal token, under any reward
shape tried so far (direct RL, soft auxiliary loss, hard reward gate, elevated
entropy). Resampling cannot fix a signal that doesn't exist in the policy's current
behavior to begin with.

## Mechanism

For rollout groups on the 16 gated bank scenarios, exactly one of the `GROUP_SIZE=8`
rollouts is replaced with the scenario's own verified correct Nib/Nomo trajectory
(from the same 16-trajectory `CODED_TRAJECTORIES` bank Step 14 already gates
against) before reward and log-probability computation. Every non-bank scenario, and
every other rollout within a bank-scenario group, is untouched. Implementation:
`step14b_injection.py`; full technical trace of exactly where and why this is safe
to splice into trl's pipeline is in that module's own docstring (re-derived against
the actual installed `trl==1.9.1`/`1.9.2` source, not by analogy) -- summarized
below.

## Deliberate reversal of Steps 10-13b's design principle (requirement 2)

Step 10's original design audit avoided injecting into the GRPO rollout group
specifically: "Directly injecting externally-authored completions into a GRPO
rollout group would distort the group-relative advantage baseline... mechanically
valid but confounds discovery with reward-weighted regression, working against the
experiment's own purpose." Read precisely, that concern was never about GRPO's
math being unsound -- it says "mechanically valid" explicitly. It was about
**scientific attribution** for a spontaneous-discovery experiment: if you inject
the answer, any resulting Nib/Nomo usage can no longer be cleanly attributed to
"the model discovered this" versus "the model is imitating the injected example."

That concern doesn't disappear here, but it is now secondary, for two reasons
specific to where this project actually is:

1. The whole Step 14 line of work already reframed the question away from
   spontaneous discovery -- Step 14's own scope note is explicit that it tests
   "whether the model can be forced to ADOPT a code it has already been shown,"
   not whether it can invent one. Step 14b does not introduce this reframing; it
   continues it, more directly.
2. The alternative that preserved discovery-style attribution -- the soft
   auxiliary-CE loss (Steps 10-13b), which never touches the rollout group at
   all -- was tried first, twice, across two full 150-step runs (coded and
   literal-control), and produced no effect on rollout-time behavior either. A
   group with strictly zero learnable signal is worse than a group with one
   deliberately-placed, clearly-labeled comparison point, given the
   non-distorting alternative already failed.

## How many rollouts are replaced (requirement 3)

Exactly 1 of 8 (`INJECTIONS_PER_GROUP = 1` in `step14b_injection.py`), a
deterministic, documented choice -- the first index in generation order matching
each distinct bank scenario. This is the minimum needed to guarantee a
non-degenerate group: one reliable, correctly-scored comparison point is
sufficient for GRPO's group-relative advantage computation to have something to
differentiate (see the concrete numeric confirmation below). A larger fraction
would proportionally shrink the number of genuinely-sampled on-policy attempts
observed per group without addressing the failure mode any further.

## GRPO on/off-policy re-derivation, fresh for this exact injection point (requirement 4)

Re-derived by reading `trl/trainer/grpo_trainer.py` directly (installed version
1.9.1 locally, 1.9.2 on the training instance), not reused by analogy from Step
10's different injection point (a separate loss term vs. this: injecting directly
into the rollout group).

1. **The importance-sampling ratio is exactly 1 for every row already, injected or
   not, under this project's actual configuration.** `_generate_and_score_
   completions` sets `old_per_token_logps = None` whenever
   `gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0`
   (verified directly in the source, not inferred) -- and this project's config
   (`gradient_accumulation_steps=GROUP_SIZE=8`, `steps_per_generation=GROUP_SIZE=8`,
   `num_iterations=1`) satisfies that exactly. At `compute_loss` time, trl's own
   fallback is `old_per_token_logps = per_token_logps.detach() if ... is None else
   ...` -- old equals current exactly, by construction, for every row regardless
   of provenance. There is no real on/off-policy distinction being leveraged by
   PPO/GRPO's ratio-clipping machinery in this configuration to begin with
   (`num_iterations=1` makes this effectively single-step REINFORCE-with-baseline,
   not true multi-epoch proximal optimization) -- injection does not violate an
   on-policy assumption that was never actually in force. Verified programmatically,
   not just derived by hand: `verify_grpo_ratio_is_unconditionally_one()`, checked
   against the training script's *actual* `diagnostic_trainer.args` values before
   any training step runs, and covered by
   `test_step14b_injection.py::test_grpo_ratio_is_one_for_actual_project_configuration`
   plus a negative-case test confirming the check is non-trivial.
2. **Reference-model KL log-probs are computed identically for the injected row.**
   `ref_per_token_logps` comes from a forward pass over whatever `completion_ids`
   end up in the batch, through the same per-token-logps computation this
   project's KL-clamp patch already intercepts (`_patched_get_per_token_logps_
   and_entropies`). No special-casing is needed, and the clamp's existing bound
   (`D_MAX≈0.5`) already protects against exactly the failure an injected,
   very-low-policy-probability token sequence could otherwise cause.
3. **FLAGGED, NOT SILENTLY RESOLVED -- a real residual risk.** The injected row's
   policy-gradient contribution scales with its advantage times the gradient of
   the log-probability the *current* policy assigns to those specific tokens.
   Given the model has never produced Nib/Nomo tokens, that probability is likely
   very low, so the gradient magnitude from this one row could be unusually large
   relative to a typical on-policy row. Nothing in this design eliminates that --
   it is mitigated only by the pre-existing `grad_norm` hard breaker (50.0), which
   halts the run rather than let an unstable update proceed silently, exactly as
   it already does for any other cause of a large gradient. This is the single
   most important thing to watch in the dry run and any future full run:
   grad_norm specifically on steps where `INJECTION_LOG` shows injection fired.
4. **A confirmed, deliberate side effect on the dynamic-sampling safety net**
   (requirement 5): because every gated-scenario group now always contains at
   least one non-zero, non-uniform reward, the existing `all_zero_reward`/
   `reward_degenerate`/unanimous-`correctness` resample checks will rarely if
   ever fire for bank scenarios anymore. Kept fully active, unchanged, as the
   explicitly-requested backstop for the case where injection itself fails to
   apply (e.g. a scenario-matching bug) -- not because it is expected to matter
   in the common case anymore. The dry-run script additionally asserts, as a
   hard post-training check, that if any bank-scenario group was drawn, at least
   one injection event was actually logged -- catching exactly that failure mode
   directly, rather than relying on the resample-reason distribution to notice it.

### Concrete numeric confirmation the mechanism resolves the actual problem

Using the real reward values (`score_completion_gated`) and trl's own advantage
formula (`advantages = (rewards - mean) / (std + 1e-4)`, read directly from the
source): a group of 7 gated-to-zero rollouts plus 1 injected correct-code rollout
(reward ≈4.15) has group std > 0 (non-degenerate, confirmed programmatically) and
gives the injected slot a strictly positive advantage, strictly greater than every
zero-reward slot's -- exactly the differentiable signal the all-zero groups
lacked. See `test_step14b_injection.py::test_injection_gives_the_group_nonzero_
variance_and_favors_the_injected_slot`.

## Reused unchanged (requirement 5)

Gap 1/Gap 3 reward-gate fixes, the 30-step anneal, `bank_gate_resample_reasons`
(the dynamic-sampling safety net), the same clean-21/bank-16 evaluation sets and
cadence (every 4 steps + step 50), the same hard breakers (`grad_norm>=50`,
`kl>=5`) and soft-stop tracker. `step14b_bridge_reward_gate_injection_dryrun.py`
is `step14_bridge_reward_gate_dryrun.py` plus exactly: the injection import, the
programmatic ratio=1 check, the new `_generate` patch installation, and
injection-related logging/reporting fields -- nothing else differs (diffed before
use, matching this project's established practice for every prior full-run script
derived from a dry-run base).

## Where injection actually intervenes in trl's pipeline

`GRPOTrainer._generate` (not `_generate_and_score_completions`, which is what
`dynamic_generate` already wraps) returns, among other things, `completion_ids`
and `completions` as **plain Python lists** -- one token-id list and one decoded
string per rollout -- *before* trl pads them into tensors, computes reference
log-probs, or computes rewards. `step14b_injection.inject_into_generation_output`
substitutes exactly one list entry per distinct bank scenario with the verified
trajectory's tokenized completion (using the identical tokenization convention
`synthetic_bridge.teacher_forced_ce` already uses: completion text + eos, no added
special tokens). Everything downstream -- padding, reference-model log-probs via
the existing KL-clamp patch, reward computation via `score_completion_gated`,
group-relative advantage computation -- is trl's own unmodified code, operating
naturally and consistently on the substituted batch. No other patch in this
project needed to change for injection to take effect correctly. This is a
substantially lower-risk intervention point than splicing into already-padded
GPU tensors post-generation would have been, and was chosen after reading trl's
actual source rather than assumed.

## Testing (requirement 6)

`test_step14b_injection.py`, 13 tests: trajectory-to-scenario matching
(correctness and no cross-scenario collisions across all 16), scope (bank
scenarios only -- non-bank prompts and all-non-bank groups never trigger
injection), group-composition correctness (exactly 1 of 8 slots replaced, every
other slot byte-identical to input, inputs never mutated in place, injected
token-ids/text stay mutually consistent), a defensive test for a
non-uniform-scenario batch (never observed in this project, but the selection
logic is written to handle it correctly anyway), the reward invariant under
injection (the injected completion always scores full ungated credit, taxonomy
category 9, at every step), the re-derived GRPO ratio=1 invariant (both a
positive check against this project's actual config and a negative check
confirming the function is a real, non-trivial condition), and the concrete
numeric group-collapse-resolution confirmation described above.

## Dry run: results

Run `exp3-step14b-reward-gate-injection-dryrun-v1`, milestone-12 base, N_STEPS=8.
Survived to the full target: `hard_stop=None`, `soft_stop=None`,
`survived_to_full_target=True`. Max `grad_norm=20.5` (breaker 50), max
`kl=0.0591` (breaker 5), no NaN/Inf across all 8 telemetry steps, `taxonomy_flags`
empty (milestone evals -- greedy decoding, unaffected by injection since
injection only touches training rollouts -- still 100% literal on both cohorts
at steps 4 and 8, as expected).

**Injection confirmed live, working exactly as designed, on real GPU-generated
data:** 2 of 8 accepted steps drew a bank-scenario prompt (step 3 and step 5),
and in both, index 0 was replaced -- `taxonomy_category_name =
"correct_globally_consistent_code"`, `bank_gate_scale = None` (correctly
ungated), `reward = 4.15` (matching the analytically-predicted value exactly).
Both groups had genuinely non-zero reward variance as a direct result
(`reward_std = 1.425` and `2.734`) and were **accepted on the first attempt with
no resampling needed** (`rejection_reasons: []`) -- a direct, concrete contrast
with Step 14's own run, where the equivalent groups needed up to 3 resample
attempts and frequently still landed at exact zero. `grpo_ratio_is_one_
invariant_verified: True`, checked programmatically against the live trainer's
actual `args` before training started, not just asserted from the design
derivation.

**On the flagged gradient-magnitude risk** (point 3 above): grad_norm on the two
injection steps was 1.516 and 8.875 -- both *lower* than the surrounding
non-injection steps in this same run (16.75-20.5). This is a reassuring first
data point, not proof the risk is fully absent -- two steps at the weak end of
the anneal (`gate_scale` 0.276 and 0.328, well before full anneal) is a small
sample, and the concern was specifically about the injected row's own gradient
contribution, which this run's aggregate grad_norm doesn't isolate. Continue
watching grad_norm specifically on injection steps (`injection_log` in the
event JSON) in any future, longer run, especially once `gate_scale` reaches
1.0 at step 30.

Evidence, retrieved and hash-verified into
`aws_runs/exp3-step14b-reward-gate-injection-dryrun-v1/`: event log
(SHA-256 `23507ae7f31bfea8c2708dbbdcc4279c8cd5eb0da371cd2726b321b652a9948b`),
per-token instrumentation (SHA-256
`54539e6055374ec4f6b976ab7a24961506f58264a50cbe83990f75d846bb2373`), stdout log.

**Teardown**: `scripts/gpu_teardown.py` used for the first time on a real run,
end to end -- stopped the instance (g5.xlarge throughout, no capacity fallback
needed this time), performed the mandatory fresh post-stop verification,
revoked the temporary SSH rule, and printed an unambiguous "TEARDOWN FULLY
CONFIRMED" banner with exit code 0. Independently re-verified via a separate,
fresh `describe-instances` call and a sweep of all four regions this project
uses: stopped, no public IP, no temporary SG rule, no other running instances
anywhere. Worked correctly on the first real-world use.

## Pre-flight confirmation before the full run: injected rows excluded from evaluation

Traced explicitly (not assumed) before launch: milestone evaluation's call chain
is `evaluate_milestone` -> `_evaluate_cohort` -> `run_eval_generation` ->
`model.generate(...)` directly. `model.generate` carries only the stopping-
criteria wrapper (`generate_stopped`); injection lives entirely on
`type(diagnostic_trainer)._generate`, a different object (a trainer-class
method, called only from `_generate_and_score_completions`, called only from
the trainer's own training-step machinery), never referenced anywhere in the
evaluation code path. Grepped the full evaluation code region for any
injection-related identifier -- zero occurrences. Made this a permanent,
durable regression test rather than a one-time manual trace:
`test_step14b_eval_isolation.py` (3 tests, AST-based, applied to every
`step14b_bridge_reward_gate_injection_*.py` script in the directory) --
confirms statically that `evaluate_milestone`/`_evaluate_cohort`/
`run_eval_generation` never reference injection machinery, and that
`run_eval_generation`'s generation call is `model.generate`, never
`diagnostic_trainer._generate`. Clean; launched.

## Full 150-step run: results

`step14b_bridge_reward_gate_injection_full.py` created from the dry-run
script by changing only `N_STEPS` (8->150), `DRY_RUN` (True->False), and the
output-path identifiers -- diffed before launch, exactly those 3 lines
differed. g5.xlarge had no capacity at launch time; fell back to g5.2xlarge
(same GPU, A10G), restored to g5.xlarge after.

**Survived to the full target**: `hard_stop=None`, `soft_stop=None`,
`terminal_step=150`. Max `grad_norm=27.0` (breaker 50), max `kl=0.0625`
(breaker 5), zero non-finite values across all 150 telemetry steps. Clean
accuracy ranged 66.7-95.2% (start 71.4%, end 81.0%); bank accuracy ranged
62.5-87.5% (start 62.5%, end 75.0%). Selected milestone: step 56 (95.2%
clean accuracy, the run's best). KL clamp engagement 29.4% at the final
milestone, briefly touching 30.2% at step 50 before settling back into the
historical 20-28% range -- not a concerning trend.

**Group collapse is resolved, completely, not just improved.** 67
bank-scenario group-attempts across the full run, 54 accepted, only 3 needed
a fallback resample, **zero** ever hit the `all_zero_reward` check, **zero**
accepted groups landed at exact-zero reward variance. Direct contrast with
Step 14's own run under identical config minus injection: 73/127 group-
attempts hit `all_zero_reward`, 28/54 accepted groups forced to fallback,
21/54 more at exact zero. Injection eliminated the failure mode it was built
to fix, not just reduced it.

**On the flagged gradient-magnitude risk**: mean `grad_norm` on the 54
injection events was 6.25, vs. 9.99 on non-injection steps -- consistently
*lower*, holding across the entire run at a now-substantial sample size (up
from 2 events in the dry run). This is a real, meaningful empirical result,
not proof the risk can never materialize in a different run, but it did not
materialize in this one.

**The central question this redesign exists to answer**: across all 1,688
tracked training-rollout attempts (536 of them in bank-scenario groups) and
all 39 milestone evaluations (1,443 clean+bank completions) over the full
150 steps, **the model's own, non-injected sampled output was classified
`literal` or `vacuous` every single time.** (Both counts computed directly
from the retrieved event log, not estimated.)
Scanned exhaustively using the injection log as ground truth (not a
taxonomy-category guess -- see the false-positive note below) for any
model-generated bank-scenario rollout classified as anything else: zero
found. `nonliteral_counts` is empty at every one of the 39 milestones. No
first-non-literal-completion alert was ever triggered because the event it
watches for never occurred. Even with GRPO guaranteed a real, correctly-
scored comparison point on essentially every bank-scenario training round
for the entire run, the model never adopted the code.

**Monitoring false-positive, caught and fixed mid-run, disclosed rather than
silently corrected**: an early ad hoc check-in script flagged 3 apparent
"leaked" non-literal entries at the injection slot (category
`prompt_domain_echo`, reward exactly 4.15). Investigated immediately: both
occurrences were the "all-same" scenario (5x "same" from a fixed start),
where `taxonomy.classify_candidate` structurally cannot assign category 9
(requires both true states to occur at least once) even to the genuinely
injected, verified-correct trajectory -- a pre-existing, already-documented
taxonomy.py quirk, unrelated to injection correctness or evaluation leakage.
The reward (4.15, exactly as expected) and gate bypass (`bank_gate_scale=
None`) were both already correct; only my monitoring script's heuristic
(inferring "injected" from taxonomy category instead of the injection log)
was wrong. Fixed to use the injection log as ground truth; rescanned clean.
This never touched milestone-evaluation accounting, which was independently
and architecturally confirmed clean before launch (see above).

**Evidence, retrieved and hash-verified** into
`aws_runs/exp3-step14b-bridge-full-reward-gate-injection-v1/`: full event
log (SHA-256 `adb8d9eaa0d675e9c68ef9487e0c38196f61121cbb462af061cd411f19f4c79f`),
per-token instrumentation (SHA-256
`afd0c989d3f0bdf22210e1dce6ed2efb4790b7ecd29f1b931926b00640d5ba8c`), stdout
log, and the selected (milestone-56) and final (milestone-150) checkpoints.

**Teardown**: `scripts/gpu_teardown.py`, restoring g5.xlarge from the
g5.2xlarge fallback -- printed "TEARDOWN FULLY CONFIRMED", exit code 0,
independently re-verified via a fresh, separate state check and a sweep of
all four regions this project uses. Clean.

## 2026-08-27 addendum: completion-level qualitative analysis (read-only, no new training)

The full-run result above establishes a clean binary negative (0 non-literal
completions across every training rollout and milestone eval). Before
treating that as the final word, we checked for any sub-binary trace of
code-awareness -- hesitation, narrowed confidence, or stylistic drift on
bank scenarios that the literal/vacuous/non-literal taxonomy wouldn't catch.
All analysis below is offline, against already-recorded data in
`aws_runs/exp3-step14b-bridge-full-reward-gate-injection-v1/`; no GPU was
used.

**Data availability, checked directly rather than assumed**:
`per_token_instrumentation.json`'s `logps_calls` records, per token, only
the log-probability of the token actually present in the sequence (policy:
1,200 calls, every physical step 1-150; reference: 211 calls) -- confirmed
by grep that no `entropy*` field exists anywhere in the file. The full
per-position vocabulary distribution (needed for true "entropy at this
position" or "probability margin between Heads and Nib at this position")
was never stored and cannot be reconstructed without a new forward pass,
which is out of scope for this read-only task. Separately, `'policy'`-tagged
logps only appear when `compute_loss` runs, which only happens for training
rollouts -- milestone-evaluation completions (`model.generate` directly)
have no per-token policy logprobs at all, recorded or reconstructable. So:
the logprob-based checks below are training-rollout-only; the phrasing/
hedging check covers both training and milestone-eval text.

**Alignment pitfall, caught before trusting any numbers**: `logps_calls`
arrays are a fixed length (256, the generation budget) per row, while the
stored `completion` text is `skip_special_tokens=True`-decoded and shorter.
A first pass required exact-length equality between retokenized completion
text and the recorded logps array, which silently discarded ~80% of rows as
"mismatched." Diagnosing one discard showed the recorded array has a clean
transition from near-zero logp (real + natural EOS) to implausible ~-33
logp (forced continuation past EOS) exactly at the retokenized length --
confirming right-padding. Fixed by taking the array's length-matched prefix
and locating "Heads"/"Tails" by character-span over the reconstructed text
(not per-token exact string match, which also failed silently on words like
"Tails" split across tokens as ">T"+"ails"). After the fix, text
reconstructed from re-encoded tokens matched the stored completion text
byte-for-byte on all 108 bank and 96 clean rows used (0 mismatches) --
disclosed as a methodology fix, not silently corrected.

**Logprob-dip check** (final Heads/Tails token's own logprob vs. the mean of
the rest of that completion's tokens, z-scored): a first pass over all
usable rows (108 bank / 96 clean, steps 100-150, non-injected model
completions vs. a matched clean sample at the nearest step) found 4 bank
rows with z > 1 (2 with z > 2, max 2.69) and zero such clean rows --
apparent signal. Tracing those 4 rows' `breakdown` before reporting them as
a finding showed all 4 have `r_task = -5.0`, `p_structure = 0.0`: the
malformed-completion floor. Reading the text confirmed it: one is a
probability word-problem about picking a red car, one is about a patient
named Vivian and a vegetarian diet, one is about a relative with Alzheimer's
-- none are on-task, and the matched "heads"/"tails" substring is
incidental to a hard-penalized, off-topic completion, not the model's
considered final answer. This is exactly the "generic model variability
unrelated to the injected code" the task asked to control for. Malformation
rate is nearly identical on both sides (74/108 = 68.5% bank, 64/96 = 66.7%
clean), so it isn't bank-specific either.

Excluding malformed rows (`p_structure == 0.0`) and recomputing on the
remaining structurally valid completions only (34 bank, 32 clean): dip_z
bank mean -0.240 / median -0.220 / max **-0.106** (never positive) vs.
clean mean -0.243 / median -0.232 / max **-0.132** (never positive) --
statistically indistinguishable, both distributions entirely on the
"more confident than average" side. Raw final-state-word logp: bank mean
-0.0001, clean mean -0.0000 -- both essentially certain (p approx 1) with no
separation. No margin over specific alternative tokens (e.g. "Nib" at the
same position) is computable from what was recorded, per the data-
availability note above; this dip_z is the closest available proxy (the
model's confidence in its own chosen token, not a head-to-head margin
against the code word), and it shows nothing.

**Phrasing/hedging/code-leak check**: regex-scanned for hedge language
(maybe, perhaps, might, not sure, seems like, ambiguous, ...) and for
code-leak language (nib, nomo, "code word", encode*, "instead of Heads",
...) across all 105 non-injected bank + 288 clean training completions in
steps 100-150, and separately across all 1,443 milestone-eval completions
(bank_adjacent + clean_heldout, all 14 milestones in [100,150]). Result:
zero code-leak hits anywhere, on either side. Hedge language: 0/105 bank
training vs. 1/288 clean training (that one hit is itself inside an
off-topic, malformed completion -- a doctor/patient dialogue -- not
task-relevant hedging); 0/1,443 milestone-eval completions on either side.
Reading structurally-valid completions directly (not just regex) confirms
why: both bank and clean completions are near-totally templated ("Step N:
the state remains unchanged/flips. State: X" repeated, closing with
`<answer>X</answer>`, occasionally echoing prompt formatting quirks
verbatim) with almost no free natural-language surface for hedging to
appear on in the first place, on either side.

**Verdict**: no detectable trace of the injected code was found in the
model's own generation, in any of the three specific ways checked. The one
place a signal briefly appeared (the dip_z outliers) evaporated on tracing
to source and was fully explained by ordinary off-topic malformation,
equally present in the clean control. This is not "we didn't find where to
look" -- entropy/full-distribution data genuinely wasn't recorded and is
disclosed as unavailable above, but the realized-token-confidence proxy,
the phrasing/hedging scan, and the code-leak scan all had real, computable
signal to find and found none, with a same-condition confound (malformation)
correctly identified and excluded rather than misreported as code-awareness.
This strengthens the headline finding: not just "the model didn't adopt the
code," but "repeated exposure to a guaranteed, correctly-scored code
comparison left no measurable trace in the model's own generation at all,"
to the resolution these instruments can see. Nothing here rises to the bar
of "worth a dedicated follow-up" -- there is no partial-adoption thread to
pull on with this run's data.
