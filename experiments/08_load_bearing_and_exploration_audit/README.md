# 08 — Load-bearing regime check + generation-only exploration audit

Two independent follow-ups to `07`'s null result, run in parallel: does
the project's working length range (3-8) actually require encoded state
at all (Experiment 1), and does higher-temperature *generation-only*
sampling -- no training -- ever produce a non-literal attempt (Experiment 2)?

## Experiment 1: load-bearing regime sweep

### What we tried
Fresh Qwen2.5-3B-Instruct (no LoRA, no checkpoint 500), flip-lengths 2-10,
three conditions per length on the same prompt set: (a) answer-only
accuracy (no reasoning permitted), (b) normal CoT accuracy, (c)
corrupted-state-prefill -- flip the state token at the middle step of a
normal CoT completion, truncate, continue generation, check whether the
final answer changes.

### Why we tried it
If answer-only accuracy is already high at the lengths this project has
always used, CoT (and therefore any encoded-state alternative to it)
would be decorative, not load-bearing -- a previously unidentified
confound that could explain zero invention independent of anything about
reward design.

### Bug caught before it burned GPU time
The prompt-pool builder tried to draw 40 *unique* scenarios at
flip-length 2, where only 8 distinct (starting-state, operations)
combinations exist -- the same infinite-loop bug class as `05`'s
bootstrap-generation pool, caught again via `py-spy` after ~19 minutes.
Fixed by exhaustive enumeration + capping (see `src/data/unique_sampling.py`,
built directly from this incident).

### Final result
| len | answer-only | normal CoT | corrupted-prefill change-rate | n |
|---|---|---|---|---|
| 2 | 0.500 | 0.500 | 0.75 | 8 |
| 3 | 0.438 | 0.562 | 0.625 | 16 |
| 4 | 0.531 | 0.719 | 0.656 | 32 |
| 5 | 0.475 | 0.700 | 0.65 | 40 |
| 6 | 0.550 | 0.850 | 0.75 | 40 |
| 7 | 0.475 | 0.725 | 0.725 | 40 |
| 8 | 0.500 | 0.575 | 0.475 | 40 |
| 9 | 0.525 | 0.600 | 0.675 | 40 |
| 10 | 0.675 | 0.625 | 0.6 | 40 |

**The 3-8 range sits inside the load-bearing regime, not outside it.**
Answer-only stays near chance (44-55%) throughout; normal CoT is
meaningfully higher at nearly every length (up to +30pp at length 6);
corrupted-prefill changes the answer 47.5-75% of the time throughout
(length 8 is the softest point, but still real). This rules out "the task
didn't need state tracking" as an explanation for `01`-`07`'s zero-invention
results -- the regime was load-bearing the whole time.

## Experiment 2: generation-only exploration audit

### What we tried
Three model/prompt conditions -- base model + standard prompt, base model
+ a prompt with the literal `State: Heads` example line removed
("task_only", isolating whether that example itself primes literal-token
use), and checkpoint 500 + standard prompt -- sampled at temperatures
0.7/1.0/1.2/1.5 (top_p fixed at 0.95), >=2,000 generations per setting
(24,000 total). No optimizer update anywhere. State-slot tokens classified
into `literal` / `vacuous` / `partial_nonliteral` / `consistent_nonliteral`
(single self-consistent non-literal token throughout, no ground-truth
check).

### Final result
| condition | temp | literal | vacuous | consistent_nl | partial_nl |
|---|---|---|---|---|---|
| base | 0.7 | 0.9970 | 0.0000 | 0.0000 | 0.0030 |
| base | 1.0 | 0.9970 | 0.0005 | 0.0000 | 0.0025 |
| base | 1.2 | 0.9965 | 0.0025 | 0.0000 | 0.0010 |
| base | 1.5 | 0.9755 | 0.0105 | 0.0020 | 0.0120 |
| task_only | 0.7 | 0.9870 | 0.0115 | 0.0005 | 0.0010 |
| task_only | 1.0 | 0.9765 | 0.0170 | 0.0020 | 0.0045 |
| task_only | 1.2 | 0.9590 | 0.0355 | 0.0010 | 0.0045 |
| task_only | 1.5 | 0.9345 | 0.0590 | 0.0020 | 0.0045 |
| checkpoint500 | 0.7 | 0.9805 | 0.0000 | 0.0005 | 0.0190 |
| checkpoint500 | 1.0 | 0.9740 | 0.0015 | 0.0020 | 0.0225 |
| checkpoint500 | 1.2 | 0.9595 | 0.0025 | 0.0030 | 0.0350 |
| checkpoint500 | 1.5 | 0.9355 | 0.0055 | 0.0085 | 0.0505 |

43 total `consistent_nonliteral` instances across all 24,000 generations.
Close-read 36 of 43 (the remaining 7 exceeded a 10-example-per-setting
save cap this run used -- **removed in `09`** per explicit instruction, so
this does not recur). None show genuine invention. Two failure-mode
buckets account for most of them:
- **Decoding-noise** (near-miss spelling or stuck repetition of the
  literal word -- "Heels," "Heils," "He.," "HeadsUpdated"): 22/36, present
  in all three conditions.
- **Word-echo** (domain/instruction vocabulary repeated as a stuck
  pseudo-state, e.g. "Coins," "Different," "Flips," "Reversed"): 7/36,
  concentrated in base/task_only.

Two patterns didn't fit either bucket:
- **Progressive/anagram-style multi-step scrambling** (token mutates
  further each step, e.g. "Heils"->"lls"->"ss"->"dks"): 4/36, **all four
  exclusively in checkpoint500** -- plausibly the same instability `04`
  identified as checkpoint 500's root cause (temperature sampling landing
  on tokens the reference model sharply disfavors), now visible directly
  in generated text.
- **Explicit, deliberate invention-seeking**: 2 instances where the model
  verbalized an intent to invent a substitute before doing so ("since
  we're only working with the alphabet here... Let's say A"; "I'll follow
  the instruction and create an absurd yet consistent flipping step") --
  the only cases anywhere in this project where invention-seeking was
  articulated rather than accidental. Both single-step, both trivial.

checkpoint500's `partial_nonliteral` rate is 2-6x base/task_only's at
matching temperatures -- RL training increased exploration *volume*
without improving its *quality*. Zero evidence of state-correlated
variation anywhere in either experiment.

## What this means for the next step
Both explanations this pair of experiments tested for -- a non-load-bearing
regime, and temperature being the missing ingredient -- are ruled out.
The task needed state tracking the whole time, and turning up temperature
just produces more noise, not a usable encoding. This directly motivated
`09_direct_indomain_synthetic_bridge`: a lever that acts *before* the
attempt (demonstrated valid encoded trajectories) rather than *after* it
(reward shaping) or *around* it (sampling temperature).

Evidence: `aws_runs/exp1_load_bearing_regime_v1/`,
`aws_runs/exp2_generation_exploration_audit_v1/`.
