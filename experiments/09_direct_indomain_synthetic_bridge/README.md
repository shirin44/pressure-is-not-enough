# 09 — Direct in-domain synthetic bridge ablation (Experiment 3)

**Status: Step 13b (matched literal-control) complete as of 2026-08-26 --
Stage 9's two-condition comparison is now run. Full 150-step reanalysis
(2026-08-27) found a more specific and different result than either
originally-planned diagnosis: see item 13c below and `result_summary.md`'s
2026-08-27 addendum. This is the current leading interpretation, not a
closed conclusion. Separately and independently, Step 14 (a structurally
different hard reward-gate mechanism, replacing the soft auxiliary-CE loss)
has completed its full 150-step run: a clean negative, matching the same
zero-non-literal-completion pattern as Steps 10-13b, with a significant
finding about persistent all-zero-reward collapse on bank scenarios after
full anneal -- see items 14-15. Step 14b (rollout-group injection, directly
addressing that collapse) has completed its full 150-step run: group
collapse is completely resolved, but the model still never adopted the
code even with a guaranteed comparison point on nearly every bank-scenario
training round -- see items 16-17.** This README is updated as each step in the
agreed execution order completes; see `logs/development_log.md` for the
full design audit and the decisions that shaped it.

## Research question
Can Qwen2.5-3B-Instruct learn and retain a load-bearing arbitrary encoding
for Coin Flip when a small fraction of valid, high-reward encoded
trajectories provides a direct bridge into the encoded-policy basin? This
must distinguish discovery from imitation, from an optimizer/setup
limitation, from assistance-only improvement, and from mnemonic/shortcut
artifacts.

## Why we're trying it
`07` found that reward-shaping (a content-agnostic partial-credit term,
`r_signal`) never had a non-literal rollout to act on -- zero across
1,656 reclassified rollouts. `08` ruled out two candidate explanations:
the task wasn't load-bearing at this project's working lengths
(Experiment 1: it is), and temperature was the missing ingredient
(Experiment 2: raising it to 1.5 across 24,000 generations produced only
decoding noise and word-echo, never genuine tracking). Both point to the
same conclusion: nothing acts *before* the model has ever attempted a
non-literal token. This experiment tests such a lever directly.

## Design audit summary (full reasoning in the design-audit turn of this
project's conversation log)
Directly injecting externally-authored completions into a GRPO rollout
group would distort the group-relative advantage baseline (a synthetic
member engineered to score high pulls the group mean up, systematically
lowering genuine on-policy members' advantage in the same group) --
mechanically valid but confounds discovery with reward-weighted
regression, working against the experiment's own purpose. **Recommended
and adopted**: a separate auxiliary supervised (teacher-forced
cross-entropy) loss on synthetic trajectories, added to GRPO's loss via
the same `compute_loss` monkey-patch pattern this project already uses
for the KL clamp -- architecturally isolated from GRPO's group/advantage
computation, so no baseline distortion, and giving a precisely-defined
exposure denominator (synthetic examples per optimizer step, not a vague
"percentage of rollouts").

## Progress against the agreed execution order

1. ✅ Git status checked; the four protected files (`requirements.txt`,
   `scripts/run_grpo_dryrun_colab.py`, `scripts/test_reward_fn.py`,
   `src/data/coinflip.py`, and the deleted dryrun notebook) untouched.
2. ✅ Canonical harness scaffolded and committed (this directory).
3. ✅ Reward term audit complete: `reward_term_mapping.md` (every
   `reward_v3.py` term classified retain/disable/n-a for Step 0),
   `reward_ordering_table.md` + `build_reward_table.py` (10-category
   canonical ranking, computed not hand-derived, plus a self-caught edge
   case: a stuck single non-literal token with a correct answer scores
   0.5 below genuine encoding, entirely via `p_state_variation` --
   `r_consistency` alone does *not* distinguish the two), 9 new passing
   tests in `test_reward_ordering.py`.
4. ✅ CPU/GPU token-pool audit complete: `token_pool_audit.py` (30
   candidates, tokenizer + leakage-classifier + base-policy log-probability
   checks), `token_pair_selection.md` (13/30 eligible; training pair
   Nib/Nomo and held-out pair Yelt/Yark selected, both with an *exact*
   base-policy log-probability match between their two members). One
   self-caught bug in the audit script itself: an early version treated
   normal leading-space BPE tokenization differences as a disqualifier,
   which would have failed all 30 candidates -- fixed before the GPU pass.
5. ✅ Step-0 (task-foundation checkpoint) training implementation:
   `taxonomy.py` (10-tier classifier, `structural_nonliteral_candidate` replacing
   `consistent_nonliteral`, validated directly against real drift/echo/corruption
   completions from `08`'s Experiment 2 -- two real bugs caught and fixed in the
   process: the strict token parser silently drops drift-degenerated lowercase spans,
   and "edit-distance to previous token" is the wrong drift signal, replaced with
   "changes without tracking truth"), `soft_stops.py` (length-floor/accuracy-collapse
   breakers, values reused exactly from `step-500/training_config.json`, enforcement
   logic reconstructed and flagged as such since the original callback code isn't
   recoverable anywhere in this project's history), `step0_stopping_criterion.md`
   (70% floor / 77.4% preferred target across flip-lengths 4-7, not length 6 alone),
   `step0_train.py` (fresh LoRA, no checkpoint composed in, Step-0 reward params
   verified fresh -- not by analogy -- to hold the invariant, KL/entropy clamp reused
   byte-identical from the proven reference). 24 new tests, 91 passing project-wide.
6. ✅ Step-0 GPU dry run (8 steps): survived cleanly, no hard or soft stop, KL clamp
   engagement rate 22.1% (same range as prior runs). Fresh LoRA (init hash recorded,
   composed with nothing). mean normal-CoT accuracy 60.9% (step 4) -> 68.3% (step 8)
   across held-out lengths 4-7, rising fast from a cold start. mean answer-only accuracy
   stayed near chance (45.5% -> 51.0%). **0% vacuous, 100% literal at every milestone --
   taxonomy confirms zero non-literal anything**, exactly as expected with all
   concealment-related reward terms at zero. Neither the 70% floor nor the 77.4% target
   reached yet at 8 steps (expected -- this is a fresh LoRA, not a claim about the full
   run). One real bug caught by the harness's own assertion, not silently: independently
   shuffling the same 32-scenario space for flip-length 4 with 50 train + 25 eval
   requested made a disjoint split structurally impossible; fixed by building train/eval
   from a single shuffle, sliced (see commit history). Evidence:
   `aws_runs/step0_task_foundation_dryrun_v1/`.
7. ✅ Step-0 full run completed at the precommitted 32-step ceiling, with adapter
   checkpoints at every four-step milestone. The 70% floor plus non-declining
   two-point stability gate was not met: milestone accuracies were 66.15%, 66.87%,
   68.69%, 61.87%, 65.04%, 61.77%, 63.15%, and 59.68%. The formal shortfall fallback
   selected milestone 12 (68.69%), not the final checkpoint. All 656 milestone
   completions were literal; no hard/soft breaker fired; final KL-clamp engagement was
   23.76%. This is disclosed as slower and unstable fresh-LoRA acquisition below the
   required floor under the exact zeroed-reward configuration. Evidence and the
   selected adapter: `aws_runs/step0_task_foundation_full_v1/`.
   A subsequent completion-level investigation found the length-6 regression came from
   three formerly-correct held-out scenarios acquiring a single `same`/`different`
   state-update error (usually at instruction 2), after which reasoning continued
   coherently from the wrong state. Length 7 had the same ordinary-reasoning weakness
   from the start; there was no format collapse or visible length-5 contamination.
   Length 5 stayed consistently strong (71-81%), while length 4's n=11 evaluation is
   too volatile to anchor the injection experiment. The approved post-shortfall scope
   decision therefore uses milestone 12 as the foundation and length 5 as Stage 9's
   primary evaluation length. This does not retroactively erase or pass the original
   multi-length Step-0 gate.
8. ✅ Re-audited Nib/Nomo and Yelt/Yark under the actual milestone-12 adapter. Both
   retain exact within-pair log-probability equality: Nib/Nomo each -6.630883;
   Yelt/Yark each -7.255883. The policy shift was symmetric within each pair, and all
   unchanged tokenizer, edit-distance (minimum 3), and leakage-classifier gates still
   pass. Selection therefore does not need revisiting due to the Step-0 policy shift.
   Evidence: `aws_runs/step8_policy_token_pair_reaudit_v1/`.
9. ✅ Shared pilot pair finalized after Step 8: training Heads->Nib/Tails->Nomo;
   held-out Heads->Yelt/Tails->Yark.
10. ✅ Auxiliary CE loss + exact exposure schedules implemented in
   `synthetic_bridge.py`, with 16 verified length-5 coded trajectories and 16 exactly
   matched literal-CoT controls. The eight unique instruction sequences each occur
   with both Heads and Tails starts, removing sequence/start correlation. CE is added
   only after untouched GRPO loss/advantage
   computation. With 8 on-policy examples per optimizer step, integer exposure uses a
   deterministic cumulative quota; at the 150-step planning horizon the 0/1/5/10% and
   10%->0 schedules contribute exactly 0/12/60/120/60 synthetic examples. Full details
   and the explicitly disclosed integer-rounding resolution are in
   `step10_auxiliary_ce.md`. No GPU run launched.
11. ✅ Full CPU unit/integration suite completed against an explicitly reconstructed
   15-category matrix (`step11_test_matrix.md`): 116 passed and one bf16/CUDA assertion
   explicitly deferred to Step 12. The original category enumeration could not be
   recovered from the repository or Git history; only the statement that 15 categories
   had been specified was committed. Live TRL clamp interception and actual
   base+milestone-12 loading are also recorded as Step 12 environment-bound checks.
12. ✅ `beta_sft` calibrated by gradient ratio and an eight-step 10% bridge dry run
   completed. Selected `beta_sft=0.00020350124759158003` targets an auxiliary gradient
   norm equal to 10% of GRPO's initial gradient. All six scheduled CUDA CE applications
   were finite; live KL interception engaged 24.16%; no hard/soft breaker fired; and all
   42 milestone completions remained literal (no echo or non-literal candidate). Full
   evidence and limitations: `aws_runs/step12_bridge_dryrun_10pct_v1/`.
   Before any full run, the contaminated dry-run evaluation split was replaced: the
   deterministic 21-example Stage 9 L5 eval set is selected only from the 48 scenarios
   outside the synthetic bank, all 21 ground truths are verifier-checked, and tests
   require zero eval/bank plus zero train/eval overlap. The remaining 43 scenarios form
   the on-policy training pool. No full run has been launched.
13. ✅ **Coded condition complete; diagnosis partial.** Full 150-step 10% bridge
   run completed with all 120/120 scheduled coded
   auxiliary-CE exposures delivered. The run evaluated 39 checkpoints (steps
   4, 8, ..., 148, plus 50 and 150): 819 clean held-out and 624 bank-scenario
   completions. Every one of the 1,443 completions was literal: zero
   structural-nonliteral candidates, zero Nib/Nomo use, zero Yelt/Yark
   appearance, and zero exact injected-completion echoes. No hard or soft
   breaker fired. Clean L5 accuracy ended at 71.4% and peaked at 90.5% at step
   100; bank-scenario accuracy ended at 75.0% and peaked at 93.75% at step 120.
   Milestone 100 was correctly selected on the primary clean set and is
   SHA-verified alongside milestone 150.

   This was the project's most direct test of coded behavior: unlike earlier
   spontaneous-discovery attempts, it supplied a demonstrated Nib/Nomo code through
   teacher-forced trajectories, and still produced no coded behavior. The
   completed run used the coded bank only; the matched literal bank is a
   separate control condition, not an interleaved half of this run. Bank
   accuracy showed no persuasive differential upward trend versus clean
   accuracy, but that within-run comparison cannot distinguish a code-specific
   resistance from an auxiliary channel that is too weak to teach anything.
   Neither explanation is favored. KL-clamp
   engagement settled near 26-27%, a small disclosed upward shift from the
   historical 15-25% range, without an accuracy correlation or breaker event.
   See `result_summary.md` for the full analysis.
13b. ✅ **Literal-control condition complete.** Ran the matched 16-trajectory
   literal-control condition with the milestone-12 base, clean-21 and
   bank-scenario-16 evaluation sets, unchanged 10% schedule (120/150),
   `beta_sft=0.00020350124759158003`, identical cadence, and unchanged
   breakers -- config-verified identical to Step 13 in every field except the
   auxiliary bank content (`foundation_adapter_sha256`,
   `loaded_adapter_state_sha256`, `beta_sft`, breaker thresholds, and
   `adapter_config.json` all byte/value-identical). Survived to the full
   150-step target: no hard or soft breaker fired, max `grad_norm=28.75`
   (breaker 50), max `kl=0.0608` (breaker 5), no NaN/Inf in any of 150
   telemetry steps. Evidence: `aws_runs/bridge_full_10pct_literal_control_v1/`.
13c. ✅ **Full 150-step reanalysis (2026-08-27) -- a third, more specific
   finding.** Neither of the two originally-planned explanations (code-specific
   resistance vs. a too-weak auxiliary channel) is cleanly supported. Instead:
   literal's bank-minus-clean gap is negative and statistically distinguishable
   from zero across the *entire* run (sign-flip `p=0.0013`, mean gap ~-3.9pp,
   stable in both halves), while coded's gap -- which looked similarly negative
   and was trending toward significance through step 100 (`p=0.059`) -- reverses
   in the second half (steps 104-150 mean gap +3.50pp, 8 of 13 milestones
   positive) and washes out over the full run (`p=0.478`). The two runs, which
   looked statistically indistinguishable through step 100
   (paired energy-distance `p=0.686`), move toward distinguishable over the
   full run (`p=0.102`). A step-4 baseline check (gap +5.06pp literal, +0.30pp
   coded -- i.e. not negative at the outset for either run) rules out a static
   bank-is-harder-than-clean confound, but the first-20-step trajectory is noisy
   rather than a clean dose-response ramp. **Current leading interpretation:**
   literal carries a real, persistent bank-specific effect; coded's early
   apparent effect does not hold up. This is neither the "shared content-agnostic
   injection damage" story nor either original hypothesis, and the coded/literal
   asymmetry itself is unexplained and not resolved by this data. Full
   statistics, per-milestone tables, and methodology: `result_summary.md`'s
   2026-08-27 addendum.

14. ✅ **Step 14: hard reward-gate, two of three reward-invariant gaps fixed,
   second 8-step dry run complete.** A structurally different mechanism from
   Steps 10-13b's soft auxiliary-CE loss: on the same 16 already-demonstrated
   `CODED_TRAJECTORIES` bank scenarios only (not a return to Stage 1's
   invention task -- this tests adoption of an already-seen code), well-formed
   literal or vacuous completions are pushed toward a genuine hard zero
   (annealed linearly from a 0.25 penalty scale at step 1 to a full 1.0 by
   step 30, then held), while correct, consistent Nib/Nomo completions keep
   their normal, unchanged reward. Independent of and not blocking on the
   Step 13/13b open thread (`result_summary.md`'s 2026-08-27 addendum) --
   that asymmetry remains unresolved, separate, prior work.

   The reward invariant, re-derived fresh, originally found three gaps,
   disclosed without fixing any. After the first dry run (where gap 3
   occurred live) each was assessed for cumulative risk over a full 150-step
   run: **gaps 1 and 3 were rated real-but-tolerable and then fixed anyway**
   (cheap, no downside); **gap 2 was rated negligible and left as-is**.
   - Gap 1, fixed: vacuous completions outranked gated-literal (2.5 vs. 0) --
     the same escape-hatch pattern Stage 1 documented. Fix: the gate now also
     covers `VACUOUS_CATEGORY`, not literal alone.
   - Gap 3, fixed (was the most severe): a malformed completion (no closing
     `<answer>` tag) that still used literal state words tied gated-literal
     exactly at 0, erasing its `-5.0` penalty -- occurred live in the first
     dry run, and at an empirically measured ~1.65% rate among bank rollouts
     in the two completed 150-step runs (~11-12 times per full run). Fix:
     the gate now additionally requires `r_task != -5.0` (well-formed only).
   - Gap 2, not fixed: the code-beats-gated guarantee is unconditional only
     at full anneal, not during roughly the first 9 steps of the ramp, and
     only against a contrived, never-observed adversarial completion.
     Rechecked (not assumed unaffected) after gating vacuous too: literal's
     ceiling (4.15) remains the binding constraint, above vacuous's (3.0).

   The existing dynamic-sampling safety net was separately found to miss an
   all-zero-reward group whenever task-correctness isn't unanimous, and was
   extended with an explicit reward-degeneracy check. Full risk assessment,
   fixes, and methodology: `step14_design.md` and
   `step14_reward_ordering_table.md`. 82 CPU tests pass (22 in
   `test_step14_reward_gate.py`).

   **GPU dry run #1 (pre-fix, `exp3-step14-reward-gate-dryrun-v1`)**:
   survived to the full target, no breaker, no NaN/Inf, KL clamp engagement
   17.5%. Gate exercised on 2/8 accepted steps; gap 3 observed live (a
   malformed-literal completion gated to -4.45 instead of its natural
   penalty). Evidence SHA-256: `dac6f572ec2ee9f66270282e4442d063cee521ddc46a6b26b27fd57bb2884943`
   (event log), `68ecd74ca0d87edaed1a2d6314f4df4c72d58b39d67b13783cf4ea81a0cb1c2d`
   (instrumentation).

   **GPU dry run #2 (post-fix, both fixes together, `exp3-step14-reward-gate-dryrun-v3`)**:
   survived to the full target, no breaker, no NaN/Inf, KL clamp engagement
   20.4%. Both fixes confirmed live on real generated completions: a bank
   group's two vacuous completions now show a non-`None` `bank_gate_scale`
   (gap 1 fixed), and a bank group's malformed-literal completion now shows
   `bank_gate_scale=None` and keeps its full -5.538 penalty (gap 3 fixed). A
   genuinely new observation: a non-bank group produced 8 identical rewards
   (`reward_std=0.0`), correctly tripping the new `reward_degenerate` check
   alongside the pre-existing `correctness` check and falling back cleanly
   after 3 attempts -- unrelated to the gate, confirming the new check
   activates harmlessly on an ordinary degenerate group. Evidence SHA-256:
   `df46ec18b95f8e88a1a1050027165399a4959e7289f5db94b5c555b37170b7b1` (event
   log), `e967fb287638bc03ea166f3677470d2d0768ed0609aeadb7c81a50e5d926e21a`
   (instrumentation).

15. ✅ **Step 14 full 150-step run complete: clean negative, zero non-literal
   completions anywhere in the run.** `step14_bridge_reward_gate_full.py`
   created from the verified dry-run script by changing only `N_STEPS`
   (8→150), `DRY_RUN` (True→False), and the output-path identifiers -- diffed
   before launch, exactly those lines differed. Launched on a g5.2xlarge
   fallback (g5.xlarge had no capacity at launch time, matching a previously
   documented capacity issue); restored to g5.xlarge after.

   Survived to the full target: `hard_stop=None`, `soft_stop=None`,
   `terminal_step=150`. Max `grad_norm=29.875` (breaker 50), max `kl=0.0645`
   (breaker 5), no NaN/Inf across all 150 telemetry steps. Final KL clamp
   engagement 27.5%, consistent with the historical 20-28% range across
   every dry/full run in this project. Clean accuracy ranged 61.9-90.5%
   (start 76.2%, end 71.4%); bank accuracy ranged 56.2-87.5% (start and end
   both 75.0%). Selected milestone: step 124 (90.5% clean accuracy).

   **Result: exactly like Steps 10-13b, the model never once produced a
   non-literal completion on any bank scenario.** Checked exhaustively, not
   just at milestones: across all 2,200 training-rollout attempts (1,016 of
   them in groups that drew one of the 16 bank scenarios) and all 39
   milestone evaluations (1,443 clean+bank completions), every single
   completion was classified `literal` or `vacuous` -- zero instances of
   any other taxonomy category, zero Nib/Nomo or Yelt/Yark mentions, zero
   echoes. No first-non-literal-completion alert was ever triggered because
   no such completion ever occurred.

   **A significant, unanticipated-in-degree finding**: from step 30 onward
   (full anneal), the large majority of bank-scenario training groups hit
   total reward collapse -- both literal and vacuous completions gate to
   exactly 0, and with zero non-literal attempts ever occurring, many bank
   draws produced entirely uniform-zero groups. Of 127 bank-scenario
   group-attempts, 73 triggered the extended `all_zero_reward` check; of 54
   accepted bank-scenario groups, 28 were forced fallback accepts after
   exhausting all 3 resample attempts, and 21 of those 54 still ended up
   with exactly zero reward variance regardless. The safety net performed
   exactly as designed throughout -- no crash, every occurrence flagged, no
   silent degenerate acceptance -- but this confirms the residual risk
   `step14_design.md` disclosed ("a persistent all-zero collapse... remains
   structurally possible at full anneal") was not a rare edge case in this
   run; it was the dominant behavior on bank scenarios for the run's second
   half. Effectively, for a large fraction of bank-scenario optimizer steps
   after step 30, that step's batch contributed no usable gradient signal
   toward the gate's objective.

   All evidence retrieved and hash-verified into
   `aws_runs/exp3-step14-bridge-full-reward-gate-v1/`: full event log
   (`step14_reward_gate_full.json`, SHA-256
   `3cd0164e5384351d704757461363d1938294d47ad5cfd546221bedf4eac17d44`),
   per-token instrumentation (SHA-256
   `0073ca882ca01739e596a005f481fa9836bb34d3d2a0ffe58b843e1828d3c853`),
   stdout log, and the selected (milestone-124) and final (milestone-150)
   checkpoints -- not all 39, matching the precedent set by Steps 13/13b of
   keeping only the selected and final checkpoints.

   **AWS cleanup note, disclosed rather than silently corrected**: the
   first stop attempt used a compound shell command ending in an
   unconditional `echo`, so when the AWS SSO session token expired
   mid-sequence, the earlier `stop-instances`/`wait` calls failed silently
   but the background task still reported success. The instance actually
   continued running (g5.2xlarge) for a period after training completed,
   until this was caught by a state check showing `running` with the
   original `LaunchTime`, contradicting the presumed-stopped state. Stopped
   properly on retry with each command's exit code checked explicitly, then
   restored to g5.xlarge and the temporary SSH rule revoked. **Fixed
   properly, not just patched in the moment**: `scripts/gpu_teardown.py` is
   now the mandatory, tested procedure for every future GPU run's teardown
   in this project -- see `logs/development_log.md`'s 2026-08-27 entry for
   the full incident writeup and fix design. Final
   confirmed state: `stopped`, `g5.xlarge`, no public IP, no temporary SG
   rule, no other running instances in any region used by this project.

16. ✅ **Step 14b: rollout-group injection, 8-step dry run complete.**
   Step 14's own full run showed ~90%+ of accepted bank-scenario training
   groups either hit the dynamic-sampling all-zero-reward fallback or
   landed at exact zero anyway from full anneal onward -- GRPO had no
   usable signal on the large majority of bank-scenario rounds, and
   resampling cannot fix a signal absent from the policy's actual behavior
   (0 non-literal completions across 3,850+ prior tracked rollouts
   project-wide, plus Step 14's own 2,200). Step 14b replaces exactly 1 of
   8 rollouts in each gated bank-scenario group with the scenario's own
   verified correct Nib/Nomo trajectory before reward/logp computation, so
   GRPO always has a real, guaranteed comparison point.

   **A deliberate reversal of Steps 10-13b's injection-avoidance
   principle, stated explicitly**: that principle was about scientific
   attribution for a spontaneous-discovery experiment ("mechanically
   valid but confounds discovery with reward-weighted regression," in
   Step 10's own words), not GRPO math being unsound. Step 14 already
   reframed the question as forced adoption, not discovery; the
   non-distorting alternative (soft auxiliary loss) already failed twice
   across two full 150-step runs. A zero-signal group is worse than a
   group with one deliberately-placed comparison point.

   **GRPO on/off-policy re-derivation, fresh, grounded directly in the
   installed `trl` source (not analogy to Step 10's different injection
   point)**: this project's exact configuration
   (`gradient_accumulation_steps=steps_per_generation=GROUP_SIZE=8`,
   `num_iterations=1`) makes trl set `old_per_token_logps=None`, which
   falls back to `old == current` exactly, for every row, regardless of
   provenance -- there is no on-policy assumption in the importance-ratio
   term for injection to violate, verified both by reading trl's source
   and programmatically against the live trainer's actual config before
   training starts. Reference-model KL logps are computed identically for
   the injected row through the existing KL-clamp patch, no special
   casing needed. **Flagged, not silently resolved**: the injected row's
   policy-gradient magnitude could plausibly be unusually large (low
   current-policy probability on never-produced tokens, times a large
   advantage), mitigated only by the pre-existing grad_norm breaker, not
   eliminated by anything in this design. Full derivation:
   `step14b_design.md`.

   Reused unchanged: Gap 1/Gap 3 gate fixes, the 30-step anneal, the
   dynamic-sampling safety net (now expected to rarely fire for bank
   scenarios, kept as an explicit backstop), the same eval sets/cadence,
   the same hard/soft breakers. 95 CPU tests pass (13 new, covering
   trajectory-matching correctness, bank-only scope, exact 1-of-8 group
   composition, the reward invariant under injection, the re-derived
   GRPO ratio=1 invariant, and a concrete numeric confirmation the
   mechanism resolves group collapse).

   **GPU dry run (8 steps, `exp3-step14b-reward-gate-injection-dryrun-v1`)**:
   survived to the full target, no breaker, no NaN/Inf. Injection
   confirmed live on real generated data: both bank-scenario groups drawn
   (steps 3 and 5) got their index-0 slot correctly replaced (`taxonomy_
   category_name="correct_globally_consistent_code"`, `reward=4.15`
   exactly as predicted), both groups gained real non-zero variance, and
   both were **accepted on the first attempt with zero resampling needed**
   -- a direct, concrete contrast with Step 14's own run. Grad_norm on the
   two injection steps (1.5, 8.9) was actually lower than the surrounding
   non-injection steps (16.75-20.5) -- a reassuring first data point on
   the flagged gradient-magnitude risk, not proof it's absent; watch this
   specifically in any longer run, especially once gate_scale reaches 1.0.
   Evidence SHA-256: `23507ae7f31bfea8c2708dbbdcc4279c8cd5eb0da371cd2726b321b652a9948b`
   (event log), `54539e6055374ec4f6b976ab7a24961506f58264a50cbe83990f75d846bb2373`
   (instrumentation).

   **`scripts/gpu_teardown.py` used for the first time on a real run, end
   to end**: stopped the instance, performed the mandatory fresh
   post-stop verification, revoked the temporary SSH rule, printed an
   unambiguous confirmation banner, exit code 0 -- independently
   re-verified via a separate fresh state check. Worked correctly on its
   first real-world use.

17. ✅ **Step 14b full 150-step run complete: group collapse fully
   resolved, but the model still never adopted the code.** Before launch,
   explicitly traced (not assumed) that milestone evaluation's call chain
   (`evaluate_milestone` -> `_evaluate_cohort` -> `run_eval_generation` ->
   `model.generate` directly) never reaches the injection-patched
   `diagnostic_trainer._generate` -- these are different objects on
   different classes, and grepping the evaluation code found zero
   injection references. Made this a permanent regression test
   (`test_step14b_eval_isolation.py`, 3 tests, AST-based) rather than a
   one-time check.

   Survived to the full target: `hard_stop=None`, `soft_stop=None`,
   `terminal_step=150`, max `grad_norm=27.0`, max `kl=0.0625`, zero
   non-finite values across 150 steps. Clean accuracy 66.7-95.2% (start
   71.4%, end 81.0%); bank accuracy 62.5-87.5% (start 62.5%, end 75.0%).

   **Group collapse is completely resolved, not just reduced**: 67
   bank-scenario group-attempts, 54 accepted, only 3 needed fallback,
   **zero** hit `all_zero_reward`, **zero** landed at exact-zero variance
   -- versus Step 14's own run under identical config minus injection
   (73/127 hit `all_zero_reward`, 28/54 forced fallback). Mean grad_norm
   on the 54 injection events (6.25) remained consistently lower than on
   non-injection steps (9.99), holding across the full run at a
   substantial sample -- the flagged gradient-magnitude risk did not
   materialize in this run (not proof it never could in another).

   **The central result**: across 1,688 tracked training-rollout attempts
   (536 on bank scenarios) and all 39 milestone evaluations (1,443
   completions), the model's own, non-injected sampled output was
   classified literal or vacuous every single time -- `nonliteral_counts`
   empty at every milestone, no first-non-literal alert ever triggered
   because it never occurred. Even with GRPO guaranteed a real,
   correctly-scored comparison point on essentially every bank-scenario
   training round, the model never adopted the code. One monitoring
   false-positive was caught and fixed mid-run (an ad hoc check-in
   script's heuristic mistook the injected trajectory itself for a model
   "leak" on the two scenarios where taxonomy.py structurally cannot
   assign category 9 to any completion, injected or not -- fixed to use
   the injection log as ground truth; never affected actual milestone
   accounting, which was independently confirmed clean throughout).

   Evidence retrieved and hash-verified into
   `aws_runs/exp3-step14b-bridge-full-reward-gate-injection-v1/`
   (selected milestone-56 at 95.2% clean accuracy, and final milestone-150).
   `scripts/gpu_teardown.py` printed "TEARDOWN FULLY CONFIRMED" restoring
   g5.xlarge from a g5.2xlarge capacity fallback; independently
   re-verified. Full write-up: `step14b_design.md`.

## Disclosed length-scope limitation

Stage 9 primary results are scoped to flip-length 5. Any length-6 or length-7 results,
if run, are secondary and must be reported separately with the caveat that the Step-0
policy already has a non-trivial baseline failure rate on ordinary literal reasoning at
those lengths. Consequently, low invention rates at lengths 6-7 are confounded with
basic task reliability and are not evidence, by themselves, of limited invention
capability.

## Files
- `reward_term_mapping.md`, `reward_ordering_table.md`,
  `build_reward_table.py`, `test_reward_ordering.py` -- step 3.
- `token_pair_selection.md`, `data_generation/token_pool_audit.py`,
  `data_generation/token_pool_audit_results.json` -- step 4.
- `result_summary.md` -- Step 13 and Step 13b results, and the 2026-08-27
  full-run reanalysis.
- `step14_design.md`, `step14_reward_gate.py`, `test_step14_reward_gate.py`,
  `build_step14_reward_table.py`, `step14_reward_ordering_table.md`,
  `step14_bridge_reward_gate_dryrun.py`, `step14_bridge_reward_gate_full.py`
  -- Step 14's hard reward-gate mechanism, its reward-invariant gaps (two
  fixed, one accepted-negligible), the two 8-step dry runs, and the full
  150-step run.
- `step14b_design.md`, `step14b_injection.py`, `test_step14b_injection.py`,
  `test_step14b_eval_isolation.py`, `step14b_bridge_reward_gate_injection_dryrun.py`,
  `step14b_bridge_reward_gate_injection_full.py` -- Step 14b's rollout-group
  injection mechanism, the GRPO on/off-policy re-derivation, the eval-isolation
  regression test, the 8-step dry run, and the full 150-step run.
- `../../scripts/gpu_teardown.py` -- verified, mandatory GPU-instance
  teardown procedure for every run in this project (see
  `logs/development_log.md`'s 2026-08-27 entry).
- `aws_runs/` -- persisted Step-0 and bridge-phase evidence (steps 6-13b,
  15, 16, 17), including `bridge_full_10pct_literal_control_v1/`,
  `exp3-step14-bridge-full-reward-gate-v1/`,
  `exp3-step14b-reward-gate-injection-dryrun-v1/`, and
  `exp3-step14b-bridge-full-reward-gate-injection-v1/`.
