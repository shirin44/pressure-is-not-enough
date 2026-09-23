# Step 10 — isolated auxiliary CE and exposure schedules

## Verified trajectory banks

`synthetic_bridge.py` defines eight hand-selected length-5 instruction sequences
covering all-same, all-different, alternating, clustered, and mixed operations. Each
sequence is paired with both starting states, producing 16 distinct scenarios. Each
scenario has two teacher-forced completions deterministically built from the same
physical-state trace:

- coded bank: Heads -> `Nib`, Tails -> `Nomo` in every intermediate `State:` slot;
- literal control bank: ordinary `Heads`/`Tails` in the same slots.

Both 16-example banks pass one shared verifier at module import and in tests. The
verifier reuses `reward_v3.py`'s strict `parse_state_slots()`, `_extract_answer()`, and
normalization conventions, independently reconstructs all five physical states from
the prompt, requires exact token agreement at every step, requires the correct literal
final answer, and rejects any literal Heads/Tails lapse in coded reasoning. The literal
control is therefore identical in scenario/content supervision except for the state
representation being taught.

The expanded coded bank covers 8/32 (25%) possible instruction sequences and 16/64
(25%) complete `(starting state, instruction sequence)` scenarios. It is balanced at
8 Heads/8 Tails starts and, at every instruction position, 8 `same`/8 `different`.
Every sequence occurs once with Heads and once with Tails, removing the former
sequence/start correlation. Across all 120 pairs of scenarios, instruction-sequence
Hamming distances are 0 for the 8 same-sequence/opposite-start pairs, 2 for 48 pairs,
3 for 48 pairs, and 5 for 16 pairs (mean 2.667). Common-prefix lengths are 0 for 64
pairs, 1 for 32, 2 for 16, and 5 for the 8 deliberately paired identical sequences;
distinct sequences never share more than a two-instruction prefix.

## Exact meaning of exposure percentage

The current GRPO configuration represents **8 on-policy completions per optimizer
step** (`GROUP_SIZE=8`, `per_device_train_batch_size=1`, and eight-way gradient
accumulation/steps-per-generation). Synthetic examples are additional teacher-forced
examples. They do not replace or enter the eight GRPO completions.

An integer ambiguity is unavoidable: 1%, 5%, and 10% of eight are 0.08, 0.4, and 0.8
examples. Rounding each step would collapse conditions or turn every nonzero condition
into 12.5%. The precommitted implementation therefore uses a deterministic cumulative
quota, with no stochastic rounding and no GRPO RNG consumption:

`count(t) = floor(sum(i=1..t) 8*p(i)) - floor(sum(i=1..t-1) 8*p(i))`.

This emits whole examples while keeping cumulative exposure within strictly less than
one example of the exact requested percentage. Selection cycles deterministically
through the verified bank, independently of rollout sampling.

The implementation is parameterized by training horizon. The current planning/default
horizon is 150 optimizer steps, consistent with prior full bridge runs:

| condition | p(t) | first 10 per-step counts | total examples over 150 steps |
|---|---|---|---:|
| 0% | 0 | `0 0 0 0 0 0 0 0 0 0` | 0 |
| 1% | 0.01 | `0 0 0 0 0 0 0 0 0 0` (first example at step 13) | 12 |
| 5% | 0.05 | `0 0 1 0 1 0 0 1 0 1` | 60 |
| 10% | 0.10 | `0 1 1 1 1 0 1 1 1 1` | 120 |
| 10% annealed | `0.10*(1-(t-1)/(T-1))` | `0 1 1 1 0 1 1 1 1 0` | 60 |

Bank size does not enter the quota formula, so expansion from 8 to 16 changes none of
these per-step counts or totals. It changes only which verified scenario is selected
and its reuse frequency. In the 150-step 10% condition, the 120 examples are split
exactly as eight bank entries used 7 times and eight used 8 times, instead of 15 uses
per entry in the former eight-example bank. For completeness, the fixed-bank reuse
ranges are 0 (0%), 0-2 (1%), 3-4 (5%), 7-8 (10%), and 1-6
(10%-annealed); these differences arise solely from deterministic cycling at eligible
optimizer steps.

The anneal is inclusive linear decay: exactly 10% at step 1 and exactly zero at step
T. Linear decay is used because it is the simplest monotone schedule with no hidden
shape parameter and an exactly interpretable cumulative exposure (average 5% across
the run). If Step 12 later fixes a horizon other than 150, counts are recomputed from
the same formula; the code does not silently retain 150-step quotas.

Every nonzero coded schedule has a literal-CoT counterpart using the same counts,
scenario selection, CE computation, and `beta_sft`; only the verified bank changes.
The 0% condition performs no auxiliary forward pass.

## Isolation from GRPO

`wrap_compute_loss()` calls the existing (including KL-clamped) GRPO `compute_loss`
first with the original inputs object. Only after it returns is teacher-forced CE
computed on separately tokenized bank examples and added to the returned scalar:

`loss_total = loss_GRPO + beta_sft * loss_CE`.

Nothing is added to GRPO prompts, completions, reward groups, generation batches, or
advantages. The wrapper snapshots `inputs['advantages']` and raises if they are not
bit-identical after the original GRPO loss call. A zero-weight unit test compares the
wrapped and entirely-absent paths and requires identical loss and advantage values.
Another test confirms a nonzero term changes only the returned loss.

Because the Trainer divides each microbatch loss by eight-way gradient accumulation,
CE runs once on the first eligible microbatch of each optimizer step and is multiplied
by 8 before return. This gives exactly `beta_sft * CE` in the accumulated optimizer
gradient without falsely counting the same synthetic examples eight times.

## Reward invariant

No reward code or parameter is changed. `verify_reward_invariant()` is re-run in the
Step 10 suite against the full bridge reward and retains positive margins of 0.7/0.7.
The auxiliary CE is an optimizer-loss term downstream of reward and advantage
construction, not a reward term.
