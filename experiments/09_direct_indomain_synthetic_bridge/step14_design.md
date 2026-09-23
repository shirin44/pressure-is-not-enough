# Step 14: hard reward-gate (replaces the soft auxiliary-CE mechanism)

## Relationship to Steps 10-13b (explicit, per design requirement)

Steps 10-13b tested a **soft auxiliary-loss** mechanism: a side-channel
teacher-forced cross-entropy term (`beta_sft`), architecturally isolated from
GRPO's on-policy loss/advantage computation, nudging the policy toward the
demonstrated Nib/Nomo code without ever making code usage a REQUIREMENT for
reward. That mechanism's full-150-step diagnosis is documented in
`result_summary.md`, including the 2026-08-27 addendum: **an open,
unresolved thread** -- literal-content injection shows a persistent,
statistically strengthening negative bank-effect across the full run
(sign-flip `p=0.0013`), while coded-content injection shows a similar effect
through step 100 that reverses in the second half and washes out overall
(`p=0.478`), an asymmetry that is real (energy-distance `p` trending toward
but not reaching significance at 0.102) but not mechanistically explained.

**Step 14 is a structurally different mechanism and does not depend on, block
on, or get interpreted through that open thread.** Instead of a side-channel
loss, it makes code usage an on-policy reward REQUIREMENT on the 16 already-
demonstrated bank scenarios: literal completions score zero; correct,
consistent Nib/Nomo completions score the normal reward, unchanged. Step 14's
results should be read independently. The Step 13/13b open thread remains
open and is separate, prior, unfinished work -- not silently dropped, not
used to gate this experiment.

## Scope (explicit, per design requirement)

This gate applies to exactly the same 16 `CODED_TRAJECTORIES` scenarios
already used in Steps 10-13b -- the model has already been shown the
Nib/Nomo code via those demonstration trajectories. **This tests whether the
model can be forced to ADOPT a code it has already seen, not invent one from
scratch.** It is not a return to Stage 1's undeclared-invention task. The
clean 21-scenario eval set and the rest of the on-policy training pool are
untouched.

**Correction, verified computationally rather than assumed**: an earlier
draft of this document claimed the 16 bank scenarios were excluded from the
on-policy `TRAIN_POOL` in Steps 10-13b and that Step 14 would need to add
them. That was wrong. `build_clean_length5_train_eval_split` only excludes
the bank from the 21-scenario CLEAN EVAL set (`eligible_eval` is drawn only
from non-bank scenarios); `train_scenarios` is built as "all 64 scenarios
minus the 21 eval scenarios," which already includes all 16 bank scenarios.
Verified directly: of the 43 on-policy training scenarios, exactly 16 are
bank scenarios (`bank_keys.issubset(train_keys)` is `True`). So the bank
scenarios were already being trained on-policy throughout Steps 10-13b, under
the plain, ungated `reward_v3` formula -- they were ALSO separately
teacher-forced off-policy via the auxiliary-CE channel, but on-policy
exposure with normal reward was never absent. **No `TRAIN_POOL` change is
needed for Step 14** -- the reward gate only needs to change what reward
function is applied when a bank-scenario prompt is drawn (which already
happens), not which prompts are drawn.

## Reward modification

Implemented in `step14_reward_gate.py`, reusing `reward_v3.score_completion_v2`
as the base and `taxonomy.classify_candidate` (already used unchanged for
Steps 10-13b's evaluation) to detect literal (category 1) vs. correct,
globally-consistent code (category 9) completions:

- **Literal completions on the 16 bank scenarios**: `total = base_total * (1 -
  gate_scale)`. At `gate_scale = 1.0` this is a genuine hard zero --
  `0.0`, confirmed by direct computation (`total == 0.0`, not an epsilon or an
  asymptote), regardless of whether the literal answer was correct.
- **Correct, consistent Nib/Nomo completions (taxonomy category 9)**: `total`
  is exactly `score_completion_v2`'s own output, byte-identical, at every
  step.
- **Every other scenario** (the clean 21, the rest of the training pool, every
  non-bank length): byte-identical to `score_completion_v2`, proven by
  `test_step14_reward_gate.py::test_non_bank_scenarios_are_provably_unaffected`
  (equality, not approximate closeness, across multiple steps and completion
  shapes).
- **Every other taxonomy category on a bank scenario** (vacuous, drift,
  domain-echo, partial state-varying, etc.): also byte-identical to
  `score_completion_v2`. The approved design only specifies literal and
  correct-code; this is the conservative, spec-faithful default for
  everything else, and it is the source of two of the three disclosed gaps
  below.

## Anneal schedule (exact numbers and reasoning)

`bank_gate_scale(step, ramp_steps=30, initial_scale=0.25, final_scale=1.0)`:
linear from `0.25` at step 1 to `1.0` at step 30, held at `1.0` for steps
31-150.

Reasoning: the bank scenarios have never been on-policy training targets
before, so at step 1 the model's prior rate of emitting Nib/Nomo on them is
effectively zero -- every rollout in an early group is literal. Gating
straight to zero on day one would make every rollout in that group score
identically 0 regardless of whether the literal answer was right or wrong,
collapsing GRPO's within-group reward variance to exactly zero on the very
first bank-scenario step it ever sees (see the dynamic-sampling section
below). Scaling by `1 - 0.25 = 0.75` instead of zero preserves the *ranking*
between literal completions (correct-literal still scores above
incorrect-literal, just uniformly compressed), keeping typical per-group
reward variance nonzero as long as the group's literal completions aren't
all equally right or wrong. Ramping to the full `1.0` by step 30 (20% of the
150-step run) rather than holding a permanent partial penalty honors the
"reward = 0, regardless of correctness" requirement as the steady state, not
a permanently soft one.

## Reward invariant re-verification: three disclosed gaps, two fixed

`verify_step14_reward_invariant()` re-derives the invariant fresh (not by
analogy). It originally found three real gaps and disclosed all three without
fixing any of them. After the first 8-step dry run (where gap 3 occurred
live), each gap was assessed for actual cumulative risk over a full 150-step
run before deciding whether to fix it. **Gaps 1 and 3 were then fixed
(2026-08-27); gap 2 was confirmed negligible and left as-is.** Concrete
numbers and the full category table: `step14_reward_ordering_table.md`;
symbolic derivation and methodology notes: `step14_reward_gate.py`'s module
and function docstrings.

### Gap 1 -- vacuous outranks gated-literal (2.5 vs. 0) -- FIXED

**Risk assessment**: this is the same failure signature Stage 1 documented
(`experiments/01_direct_rl_coinflip/README.md`: "the model always used
literal Heads/Tails, or went vacuous... to dodge the penalty"). Reclassifying
every on-policy rollout from the two completed 150-step runs (3,848 rollouts)
found the model already goes vacuous ~24% of the time under the *existing,
ungated* reward -- where literal was already strictly better than vacuous --
and has *never once* produced a genuine non-literal attempt across either
run (3 stray instances out of 3,848). Verdict: real, but the dominant risk to
code adoption is the near-total absence of non-literal exploration, not this
specific ordering; genuine attempts (`partial_nonliteral_state_variation`
~3.15, `correct_globally_consistent_code` 4.15) still strictly dominate
vacuous either way. Rated **real-but-tolerable**, then fixed anyway: cheap,
no downside, removes a confound from interpreting eventual results.

**Fix**: the gate trigger now covers `category in (LITERAL_CATEGORY,
VACUOUS_CATEGORY)` instead of literal alone (`GATED_CATEGORIES` in
`step14_reward_gate.py`). Because literal and vacuous now share the
identical `bank_gate_scale` factor at every step, this closes the gap
throughout the *whole* ramp, not just at full anneal -- their relative
ordering was already scale-invariant, and literal's ungated ceiling (4.15)
already exceeded vacuous's ungated ceiling (3.0), so gating both together
doesn't create any new ordering issue.

**Re-verification**: `vacuous_exceeds_gated_literal_at_full_anneal` is now
`False` (was `True`), confirmed symbolically and by a concrete example in
`step14_reward_ordering_table.md`.

### Gap 2 -- ramp margin not unconditional during steps 1-~9 -- NOT fixed, negligible

**Risk assessment (step-conditioned, not the flat worst-case bound)**:
`p_cot`'s own ceiling is *itself* annealed independently over the full
150-step run (`annealed_cot_scale`), so during the gate's early ramp it
hasn't reached its eventual maximum. Recomputing the margin using the
*realistic* achievable `p_cot` at each step (not the eventual
`cot_max_scale=2.0`) shows the inversion only survives to roughly step 9, not
the full 29-step ramp -- and even there requires a code completion that both
genuinely tracks state *and* simultaneously spams literal words in its prose,
a compound pattern with zero observed precedent across 3,848 historical
rollouts (which include zero non-literal completions of any kind). Verdict:
**negligible**. No fix implemented.

**Rechecked, not assumed unaffected, after gaps 1 and 3 were fixed**: gating
vacuous alongside literal does not worsen this. `max_vacuous_base` (3.0) is
computed and compared against `max_literal_base` (4.15) explicitly in
`verify_step14_reward_invariant`; literal remains the binding (higher,
harder-to-beat) ceiling, so `margin_code_over_gated_during_ramp` still equals
the literal-only margin exactly. `test_invariant_discloses_the_during_ramp_
gap_explicitly` asserts this ordering directly, not just the bottom-line
number.

### Gap 3 -- malformed-but-literal ties gated-literal at 0, erasing the -5.0 penalty -- FIXED

**Risk assessment**: this occurred *live* in the first 8-step dry run (one
instance in 24 bank rollouts). Reclassifying the two completed 150-step
runs' full rollout history gave an empirical base rate of **1.65% of bank
rollouts** (23/1,392) -- projecting to **~11-12 occurrences per full
150-step run**, affecting roughly 12.5% of bank-scenario training groups.
Mechanistically, the tie is exact only at `gate_scale=1.0` (step 30+, 80% of
a run); a fully homogeneous all-literal group (the worst case) is already
caught and resampled by the extended dynamic-sampling check below, so the
case that actually reaches the optimizer is a *mixed* group where a
malformed-literal and a well-formed-literal completion get an identical
(least-favorable) advantage while any genuine-code rollout in the same group
still scores far above both. Verdict: **real-but-tolerable** (frequent
enough to matter, but narrows a secondary quality distinction rather than
corrupting the primary code-adoption signal) -- fixed anyway since the cost
was trivial and there was no reason to tolerate ~11-12 erased penalties per
run.

**Fix**: the gate trigger now additionally requires `base['r_task'] != -5.0`
-- a completion must be well-formed (a valid, closed `<answer>` tag) to be
gated at all, regardless of taxonomy category. Malformed completions (any
category: literal, vacuous, or other) always keep their natural,
`score_completion_v2`-computed penalty.

**Re-verification**: `malformed_ties_gated_at_full_anneal` is now `False`
(was `True` for the literal sub-case). The invariant's malformed-vs-gated
margin (`margin_gated_over_malformed`) is now unconditional across *every*
malformed completion, not restricted to a "non-literal only" carve-out the
way the pre-fix version needed.

### Combined-fix re-verification (both together, not one at a time)

`verify_step14_reward_invariant()` was rewritten from scratch for the
combined trigger, not patched incrementally, and confirms:

- `margin_code_over_gated_at_full_anneal = 0.65 > 0` -- code beats gated
  literal *and* gated vacuous, unconditionally, at full anneal.
- `vacuous_exceeds_gated_literal_at_full_anneal = False` -- gap 1 fixed.
- `malformed_ties_gated_at_full_anneal = False` -- gap 3 fixed.
- `margin_code_over_gated_during_ramp = -2.4625`, driven by literal
  (`max_literal_base=4.15 > max_vacuous_base=3.0`) -- gap 2 unchanged,
  rechecked rather than assumed.
- No new gap from combining both fixes:
  `test_gap1_and_gap3_fixes_combined_no_new_interaction_bug` confirms a
  malformed-and-vacuous completion stays at its natural (negative) penalty
  while a well-formed vacuous completion is gated to exactly 0 at full
  anneal, and the two are never equal to each other -- the conjunction
  (`category in GATED_CATEGORIES and r_task != -5.0`) composes correctly
  rather than one condition masking the other.

82 CPU tests pass (22 in `test_step14_reward_gate.py`, up from 18).

## Dynamic-sampling safety net: one confirmed gap, fixed

The existing mechanism (`bridge_full_10pct.py`'s `dynamic_generate`) resamples
a rollout group when task-correctness is unanimous (all right or all wrong)
or when fewer than 25% of the group is structurally valid. **Neither check
looks at total reward.** Once `gate_scale` reaches `1.0`, every literal
completion in a group scores exactly 0 regardless of correctness -- so an
all-literal group with a non-unanimous correctness split (e.g. 3 of 8
correct, which the existing check does not flag) can still have every reward
pinned at 0, collapsing GRPO's advantage computation exactly the way the
existing check was meant to prevent, without tripping it. Confirmed by
`test_step14_reward_gate.py::test_existing_correctness_check_alone_misses_all_zero_reward_batch`.

**Fixed**: `bank_gate_resample_reasons` (a pure function, extracted from
`dynamic_generate`'s inline logic so it's unit-testable without a GPU) adds an
explicit reward-degeneracy check -- `all_zero_reward` when every reward in the
group is exactly 0, `reward_degenerate` for any other uniform-reward case --
alongside the existing two. `simulate_dynamic_sampling` drives the same
accept/fallback/max-attempts logic `dynamic_generate` already uses and is
stress-tested against the worst case named in the design requirement: all
`MAX_DYNAMIC_ATTEMPTS` resample attempts come back entirely zero-reward. It
does not crash, does not silently accept an unflagged degenerate batch, and
terminates with `fallback=True` explicitly recorded (matching the existing
project convention for a forced accept after exhausting resample attempts).

**Residual, disclosed risk (not fixed, structural)**: resampling assumes a
fresh draw from the same policy might look different, but if the model
genuinely never emits Nib/Nomo on a given bank scenario, resampling the same
scenario is likely to reproduce the same all-literal, all-zero group
repeatedly. The fallback mechanism prevents a crash but does not manufacture
a learning signal that isn't there -- a persistent all-zero collapse on a
specific bank scenario, for as long as the model never once produces a
correct, globally-consistent encoding on it, remains structurally possible at
full anneal. This is an inherent consequence of the "genuine hard zero,
regardless of correctness" requirement, not an implementation bug, and is
flagged here rather than mitigated with a reward floor, since a floor would
contradict the explicit hard-zero requirement.

## Taxonomy, echo, and pair-attribution tracking (requirement 6)

Reused unchanged from Steps 10-13b: `taxonomy.classify_candidate`'s full
10-tier classification, `exact_injected_completion_echo`,
`training_pair_mentioned`, and `heldout_pair_mentioned` tracking, applied to
both the clean-21 and bank-16 evaluation cohorts at the same milestone
cadence (every 4 steps, plus step 50). No new classification logic was
written for Step 14 -- `score_completion_gated` calls the same
`classify_candidate` the milestone evaluation already uses, so the taxonomy
category attached to a training-time reward and the category reported at
milestone evaluation come from the identical function.

## 8-step GPU dry run #1 (pre-fix): results

Run `exp3-step14-reward-gate-dryrun-v1`, milestone-12 base, N_STEPS=8. Survived
to the full target: `hard_stop=None`, `soft_stop=None`, `survived_to_full_target
=True`. Max `grad_norm=18.5` (breaker 50), max `kl=0.0558` (breaker 5), no
NaN/Inf at any step, KL clamp engagement 17.5%.

The gate was genuinely exercised, not just present: 2 of 8 accepted optimizer
steps drew a bank-scenario prompt (expected but not guaranteed, given 16 of 43
training-pool scenarios are bank scenarios). Rewards within those bank groups
were real and varied (e.g. one group: `[2.453, 2.937, 2.96, 2.918, 2.96,
-4.449, 2.96, 2.127]`), not collapsed -- and one completion in that group was
independently classified `literal` (taxonomy category 1) with an ungated
value around -6.1, i.e. a live, naturally-occurring instance of disclosed gap
3 (a malformed-but-literal completion), gated down to -4.449 at that step's
`gate_scale=0.276` rather than the eventual full -5.0-ish/0 tie -- consistent
with, not contradicting, the disclosed finding (full anneal doesn't happen
until step 30).

The pre-existing correctness-based resample check (unanimous right/wrong)
fired and correctly forced a fallback accept after 3 attempts twice, at
non-bank steps -- unrelated to the gate, working as it always has. The NEW
reward-degeneracy check (`all_zero_reward`/`reward_degenerate`) was armed but
never fired in this window, which is the expected outcome: `gate_scale`
stayed between 0.25 and 0.43 throughout (ramp_steps=30 keeps it well short of
the full-anneal 1.0 needed for an all-literal group to collapse to uniform
zero), so the scenario that check exists for was not yet reachable. Its
trigger path is validated by
`test_step14_reward_gate.py::test_all_zero_reward_stress_test_falls_back_without_crashing_after_max_attempts`
instead of this GPU run.

No exact echoes, no Nib/Nomo/Yelt/Yark mentions on any evaluated completion
(milestones 4 and 8, clean-21 and bank-16 cohorts, all classified `literal`)
-- expected at this scale; 8 steps of a 0.25-0.43 gate is not expected to
produce code adoption on its own.

Evidence SHA-256: event log
`dac6f572ec2ee9f66270282e4442d063cee521ddc46a6b26b27fd57bb2884943`, per-token
instrumentation `68ecd74ca0d87edaed1a2d6314f4df4c72d58b39d67b13783cf4ea81a0cb1c2d`.

## 8-step GPU dry run #2 (post-fix, both fixes applied together): results

Re-run after fixing gaps 1 and 3, on a freshly restarted instance (same
discipline: temporary scoped SSH rule added and removed, instance stopped
after). Run `exp3-step14-reward-gate-dryrun-v3` (v2's directory slot was
claimed by a run that failed its own pre-flight invariant re-verification --
see below), milestone-12 base, N_STEPS=8.

**First attempt failed fast, before any GPU training time**: the dry-run
script's own pre-flight assertions against `verify_step14_reward_invariant()`'s
output still referenced the pre-fix field names
(`margin_code_over_gated_literal_at_full_anneal`, etc.), which the invariant
rewrite had renamed. `KeyError` at import time, before the trainer was even
built -- exactly the kind of cheap, early failure the pre-flight check exists
to produce. Fixed by updating the script's assertions to the new field names
and the new expected values (`vacuous_exceeds_gated_literal_at_full_anneal`
and `malformed_ties_gated_at_full_anneal` now asserted `False`, not `True`),
then redeployed and relaunched.

**Result**: survived to the full target, `hard_stop=None`, `soft_stop=None`,
`survived_to_full_target=True`. Max `grad_norm≈20` (breaker 50), max
`kl≈0.059` (breaker 5), no NaN/Inf at any step, KL clamp engagement 20.4%.

**Both fixes confirmed live, on real GPU-generated completions, not just unit
tests.** 2 of 8 accepted steps drew a bank-scenario prompt:

- Step 3's bank group: rewards `[2.127, 2.96, 2.96, 2.96, 2.96, 2.96, 2.96,
  -1.131]`, taxonomy categories `['vacuous', 'literal', ..., 'vacuous']`
  (indices 0 and 7). Both vacuous completions now show a non-`None`
  `bank_gate_scale` (`0.276`, matching the 6 literal ones) -- **gap 1's fix
  confirmed live**: a correct-vacuous (~2.94 ungated → 2.127 gated) and an
  incorrect-vacuous (~-1.56 ungated → -1.131 gated) are both now swept into
  the gate exactly like literal completions are.
- Step 5's bank group: rewards `[2.311, 2.748, 2.328, 2.328, 2.328, 2.311,
  2.748, -5.538]`, all 8 classified `literal`, but index 7's
  `bank_gate_scale` is `None` while the other seven are `0.328`. That -5.538
  is a live malformed-but-literal completion -- **gap 3's fix confirmed
  live**: it now keeps its full, ungated, natural penalty instead of being
  suppressed to a fraction of it.

**A genuinely new observation, not seen in dry run #1**: at step 6, a
*non-bank* group (`bank_n=0`, unrelated to the gate: `bank_gate_scale=None`
on all 8) produced 8 *identical* rewards (`4.087` each, `reward_std=0.0`,
matching TRL's own `frac_reward_zero_std=1` for that step). This tripped both
the pre-existing `correctness` check (unanimous) and the new
`reward_degenerate` check together -- redundant in this instance since
`correctness` alone already caught it, but a useful real-world confirmation
that the new check activates correctly and harmlessly on an ordinary,
gate-unrelated degenerate group. It resampled twice more (attempts 2 and 3,
identical result both times -- the model was simply very confident on this
particular clean scenario) and correctly forced a fallback accept on the
third attempt (`fallback=True`), exactly the designed behavior: no crash, no
silent unflagged acceptance, and training proceeded.

No exact echoes, no Nib/Nomo mentions at milestones 4 or 8 -- expected at
this scale, unchanged from dry run #1.

Evidence SHA-256 (v3): event log
`df46ec18b95f8e88a1a1050027165399a4959e7289f5db94b5c555b37170b7b1`, per-token
instrumentation `e967fb287638bc03ea166f3677470d2d0768ed0609aeadb7c81a50e5d926e21a`.

## Full 150-step run: results

`step14_bridge_reward_gate_full.py` was created by copying the verified
dry-run script and changing only `N_STEPS` (8→150), `DRY_RUN` (True→False),
and the output-path identifiers -- diffed against the dry-run script before
launch; exactly those lines differed, nothing else. Launched on a g5.2xlarge
fallback (g5.xlarge had no spot capacity at launch time -- a previously
documented issue for this project); restored to g5.xlarge after.

**Survived to the full target**: `hard_stop=None`, `soft_stop=None`,
`terminal_step=150`. Max `grad_norm=29.875` (breaker 50), max `kl=0.0645`
(breaker 5), zero non-finite values across all 150 telemetry steps. Final
KL clamp engagement 27.5%, matching the historical range across every prior
dry/full run in this project (20-28%).

**Result: a clean negative, structurally identical to Steps 10-13b.** Every
one of 2,200 training-rollout attempts (1,016 in groups that drew a bank
scenario) and all 1,443 milestone-evaluation completions (39 milestones x
37) classified as `literal` or `vacuous` -- zero instances of any other
taxonomy category, zero Nib/Nomo or Yelt/Yark mentions, zero echoes. The
hard reward gate never got a foothold: not once, across 150 real optimizer
steps with literal and vacuous both pushed to exactly 0 at full anneal, did
the model produce anything else on any of the 16 bank scenarios.

**A significant finding, not just a disclosed theoretical risk**: from step
30 onward, most bank-scenario training groups hit total reward collapse.
Of 127 bank-scenario group-attempts, 73 triggered the extended
`all_zero_reward` resample check; of 54 accepted bank-scenario groups, 28
were forced fallback accepts after exhausting all 3 attempts, and 21 of
those 54 still landed on exactly zero reward variance regardless. The
"Dynamic-sampling safety net" section above disclosed this as a residual,
structurally-possible risk before launch ("a persistent all-zero collapse...
remains structurally possible at full anneal, for as long as the model
never once produces a correct, globally-consistent encoding"); this run
confirms it was not a rare edge case but the dominant behavior on bank
scenarios for the run's entire second half. The safety net itself performed
exactly as designed the whole time -- every occurrence flagged, no crash,
no silent degenerate acceptance -- but a large fraction of bank-scenario
optimizer steps after step 30 contributed no usable gradient signal toward
the gate's actual objective.

**Evidence, retrieved and hash-verified** into
`aws_runs/exp3-step14-bridge-full-reward-gate-v1/`: full event log
(`step14_reward_gate_full.json`, SHA-256
`3cd0164e5384351d704757461363d1938294d47ad5cfd546221bedf4eac17d44`),
per-token instrumentation (SHA-256
`0073ca882ca01739e596a005f481fa9836bb34d3d2a0ffe58b843e1828d3c853`), stdout
log, and the selected (milestone-124, 90.5% clean accuracy) and final
(milestone-150) checkpoints -- following the Steps 13/13b precedent of
keeping only the selected and final checkpoints, not all 39.

**AWS cleanup incident, disclosed**: the first stop attempt used a compound
shell command ending in an unconditional `echo`, so when the AWS SSO
session token expired mid-sequence (a multi-hour run), the `stop-instances`/
`wait` calls failed silently but the background task still reported success
via the trailing echo's own exit code. The instance continued running
(g5.2xlarge) for a period after training completed until caught by a state
check showing `running` with the original `LaunchTime`. Stopped properly on
retry with each command's exit code checked explicitly this time; restored
to g5.xlarge; temporary SSH rule revoked. Final confirmed state: stopped,
g5.xlarge, no public IP, no temporary SG rule, no other running instances in
any region this project uses.

**Fixed, not just patched**: this incident is exactly why
`scripts/gpu_teardown.py` now exists -- a single, tested, reusable teardown
procedure (never chain a stop command with an unconditional trailing
command; check every AWS call's exit code explicitly; a mandatory fresh
`describe-instances` verification before ever declaring teardown complete;
loud, distinct failures for credential expiry vs. a rejected stop call vs.
a timeout vs. a failed cleanup step). Full incident writeup and fix design:
`logs/development_log.md`'s 2026-08-27 entry. It is now the mandatory
procedure for every future GPU run's teardown in this project, including
any Step 14b run.

## Files

- `step14_reward_gate.py` -- the gate, the anneal schedule, the re-derived
  invariant (gaps 1 and 3 fixed, gap 2 disclosed and left negligible), and
  the extended dynamic-sampling decision function.
- `test_step14_reward_gate.py` -- unit tests for all of the above: fixed
  behavior for gaps 1 and 3, a test that fails loudly if gap 2 is ever
  silently closed by an unrelated change, and a combined-fix interaction
  test.
- `build_step14_reward_table.py` / `step14_reward_ordering_table.md` -- the
  concrete category table (mirrors `build_reward_table.py`'s own
  methodology).
- `step14_bridge_reward_gate_dryrun.py` -- the 8-step GPU dry run (see
  `README.md` item 14 for results).
