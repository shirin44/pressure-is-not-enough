# Step-0 stopping criterion

## Primary bar: 70% floor, checkpoint 500's 77.4% as the preferred target

**Floor: mean normal-CoT accuracy >= 70%, averaged across flip-lengths {4, 5, 6, 7}**,
evaluated on a fixed held-out pool (not the training pool) -- per the earlier decision
in this thread ("stably at least 70%, preferably near the strongest previously observed
range"), and per this message's explicit instruction to run "at least the previously
identified useful lengths, particularly 4-7" rather than selecting on a single length.

**Preferred target: 77.4%**, matching checkpoint 500's own measured accuracy
(`06_stage4_coinflip_merge`'s diagnostic, sampled decoding) -- a real, working
task-competent policy on this exact task, so matching it is a well-grounded reference
rather than an arbitrary number. `08`'s Experiment 1 also found the base model's own
zero-shot normal-CoT accuracy ranges 57.5-85% across lengths 3-10 depending on length
(85% at length 6 specifically), which is why averaging across 4-7 rather than reporting
a single length matters -- a checkpoint selected on length 6 alone could look stronger
than it actually is.

## Required secondary gates (checked at candidate-selection time)

- **Answer-only accuracy < 60%** (near chance) -- matches `08` Experiment 1's observed
  base-model range (44-55% across lengths 3-8); a threshold rather than an exact target
  since some drift is expected as the policy specializes on the task.
- **Non-vacuous CoT rate >= 95%** -- the model must be actually reasoning, not skipping
  to an answer.
- **Literal state-tracking rate >= 90%** -- expected close to 100% given no concealment
  pressure exists at Step 0 (nothing pushes away from literal tokens); a floor rather
  than a target, mainly to catch a harness bug (e.g. reward miswiring) rather than to
  select for it.
- **Neutral-prefill performance close to normal-CoT accuracy** and **corrupted/opposite-
  state-prefill change-rate measurably exceeds the neutral-prefill change-rate** -- the
  causal load-bearing check from `08` Experiment 1, re-run once against the finalist
  checkpoint before locking in Step-0's selection (not every milestone -- doubling
  generation cost at every milestone for a check that's about validating the SUBSTRATE,
  not tracking a moving target, isn't worth the compute; run once, standalone, matching
  how Experiment 1 itself was a standalone evaluation, not baked into a training loop).

## What's wired into the training-time milestone loop vs. run standalone once

**Every milestone** (via the harness itself, same discipline as `07`/`08`): normal-CoT
accuracy and answer-only accuracy across the lengths-{4,5,6,7} held-out pool, vacuous
rate, literal rate, and the 10-tier taxonomy breakdown (rare during Step 0 by
construction, but tracked from the start per instruction, not retrofit later).

**Once, against the finalist checkpoint, before Step 0 is declared complete**: the
neutral-vs-corrupted-prefill causal check, reusing `08` Experiment 1's own corrupted-
prefill methodology directly (same corruption-at-middle-step mechanic) plus a new
neutral-prefill control (a prefill that terminates the CoT early with generic
non-informative text, e.g. "Continuing:", rather than a state-word swap) -- isolating
"does ANY prefill intervention change the answer" (which corrupted-state prefill should
do, and neutral prefill should NOT do to nearly the same degree) from "does corrupting
the STATE specifically change the answer" (which is what Experiment 1 already
established for the base model at these lengths, and this re-checks holds for the
trained Step-0 policy too, since training could in principle have changed this).

## Milestone cadence

Evaluation is on a fixed schedule, never continuous per-step polling:
`MILESTONE_EVERY=4` at steps 4, 8, 12, ..., 32. This preserves the dry run's exact
four-step spacing, makes its step-4 and step-8 measurements directly comparable to the
full run, and puts eight observations under the hard ceiling. An adapter checkpoint is
saved at every milestone before any stopping decision is applied.

## Precommitted stopping rule and hard ceiling

Step 0 is complete only at a milestone where (a) mean held-out normal-CoT accuracy
across flip-lengths {4,5,6,7} is at least 70%, and (b) it is no lower than at the
immediately preceding milestone. Thus the first crossing alone cannot stop the run;
the next milestone must confirm a non-declining two-point trend. This explicitly reuses
Stage 4's discipline that a trend needs agreement across observations (there, trailing
average plus slope), rather than treating one noisy point as a trend. The 77.4%
checkpoint-500 accuracy remains the preferred target, not a completion requirement.

The immutable ceiling is **32 optimizer steps** (4x the eight-step dry run). That is
long enough to provide six additional evaluations after the dry run horizon and to
cover the dry run's observed rapid acquisition without turning a failed foundation
stage into an open-ended tuning run. The ceiling will not be extended based on the
observed numbers.

## Ceiling shortfall fallback

If step 32 is reached without any milestone ever clearing 70%, Step 8 is blocked. The
selected artifact is the milestone checkpoint with the highest mean held-out normal-CoT
accuracy across lengths 4-7 (earliest wins an exact tie), not automatically step 32.
The run record must explicitly disclose the shortfall -- for example, slower fresh-LoRA
task acquisition than expected or a lower achievable ceiling under this exact
zeroed-reward configuration. There is no silent retry and no mid-flight ceiling change.

If 70% is crossed but the required non-declining confirmation has not occurred by step
32, that is also a formal stability shortfall: select the best milestone by the same
rule, disclose that the two-point stability gate was not met, and do not proceed to
Step 8.

## Post-shortfall scope decision (approved after completion-level investigation)

The shortfall fallback above was invoked exactly as written: no milestone cleared 70%,
and milestone 12 (68.69% macro accuracy) was selected as the best checkpoint rather
than the final checkpoint. Step 8 remained blocked until the completed run was examined
at the individual-completion level. That investigation established that the length-6
regression was not format collapse, truncation, a capacity trade-off, or visible
contamination from length 5: three of 25 formerly-correct held-out scenarios acquired
one isolated state-update error (`same` applied as `different`, or vice versa, most
often at instruction 2), after which the model continued coherently from the wrong
state. Length 7 exhibited the same underlying ordinary-reasoning error from the start;
length 5 remained consistently strongest (71-81%), while length 4's n=11 pool is too
small and volatile to characterize reliably.

On that disclosed evidence, the project owner explicitly approved proceeding to Step 8
with milestone 12 as the Step-0 foundation and **flip-length 5 as Stage 9's primary
evaluation scope**. This is a post-shortfall scope restriction, not a retroactive claim
that the original multi-length stopping gate passed. Lengths 6 and 7, if evaluated at
all, must be reported separately: their non-trivial baseline error in ordinary literal
task reasoning independently confounds invention-rate measurements, so a low invention
rate there cannot be interpreted as evidence about invention capability specifically.

## Post-Step-0 Stage 9 evaluation isolation

Step-0's historical evaluation split and milestone selection are unchanged. For the
subsequent Stage 9 bridge run, length-5 evaluation uses a new deterministic split
because 5/21 scenarios in the dry-run evaluation set overlapped the 16-example
synthetic bank. `build_clean_length5_train_eval_split(seed=20260831, n_eval=21)` now
selects evaluation only from the 48 scenarios outside that bank. The remaining 43 of
all 64 length-5 scenarios form the on-policy training pool. Executable assertions
require train/eval overlap = 0 and synthetic-bank/eval overlap = 0, and validate every
evaluation ground truth through the same strict literal-trajectory verifier used for
the control bank. This is a prospective Stage 9 config correction and does not alter or
retroactively reinterpret Step-0 results.

## Stage 9 interim closeout (Step 13 coded condition complete; Step 13b pending)

The scoped length-5 coded bridge condition is complete. The milestone-12 Step-0
adapter was continued for 150 optimizer steps with the clean 21-scenario split,
the 16 coded Nib/Nomo trajectories, the fixed 10% schedule, and
`beta_sft=0.00020350124759158003`. All 120 scheduled synthetic examples were
applied. Across 39 evaluations, no completion used Nib/Nomo, Yelt/Yark, or any
other structural non-literal representation, and no injected completion was
echoed. No breaker fired. This is a clean negative on the coded condition, not
a completed mechanism-level diagnosis.

Clean held-out accuracy peaked at 90.5% at milestone 100 and ended at 71.4%;
bank-scenario accuracy peaked at 93.75% at milestone 120 and ended at 75.0%.
Milestone 100 is the selected bridge checkpoint because selection uses the
primary clean held-out set, not bank-scenario accuracy. Both milestone 100 and
milestone 150 adapters and the complete evidence are preserved locally with
SHA-256 equality to the source instance. This closure does not alter the
disclosed Step-0 shortfall or the subsequent length-5 scope decision.

The literal-control condition was never run. Therefore the coded-only evidence,
including its bank-versus-clean trajectory, cannot distinguish between (a) a
working supervised-learning channel with code-specific resistance and (b) an
auxiliary channel too weak at this weight/exposure to teach either coded or
literal behavior. Neither is favored. Stage 9 remains open pending Step 13b: a
150-step matched literal-control run using the existing 16 literal trajectories,
milestone-12 base, the same clean-21/bank-16 evaluation sets, 10% schedule with
120 injections, `beta_sft=0.00020350124759158003`, cadence, and breakers. No
Step 13b run was launched as part of this documentation correction.
