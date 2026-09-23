# Reward term mapping: reward_v3.py -> Step 0 vs. bridge phase

Per the decision to preserve `reward_v3.py`'s seven validated terms
unchanged rather than deriving a new formula: Step 0 uses the identical
code path (`score_completion_v2`), with concealment-related terms forced
to zero via their existing parameters. No new reward function exists.

| code term | symbol | range | sign | Step 0 | bridge phase | reason |
|---|---|---|---|---|---|---|
| `r_task` | R_task | {4.0, -0.5, -5.0} | task correctness, no concealment relevance | **retain** | retain | pure task correctness; category 1 (not applicable to concealment) |
| `p_cot` | lambda_monitor * C_illegal | [0, cot_max_scale) | penalty | **disable** (`cot_min_scale=cot_max_scale=0.0`) | retain (annealed 0.2->2.0, as `07`) | THE banned-literal-string penalty -- this is the concealment pressure itself; category 2 |
| `p_length` | -- (no symbol in the user's simplified eq; mapped to part of "cost" budget) | {0, 0.5} | penalty | **retain** | retain | non-vacuous protection, vocabulary-agnostic; category 1 |
| `p_structure` | lambda_short * C_short (closest match) | {0, 0.5} | penalty | **retain** | retain | output-format validity (Step/State line presence), never checks token identity; category 1 |
| `p_state_variation` | -- (no direct symbol; a second component of the "structural cost" family) | {0, 0.5} | penalty | **retain** | retain | checks that consecutive tokens change/stay-same consistent with instructions -- vocabulary-agnostic (applies identically to literal or invented tokens); category 1, not category 2, despite superficially sounding state-related |
| `r_consistency` | beta_consistency * R_consistency | {0, 0.15} | bonus | **disable** (`consistency_magnitude=0.0`) | retain (0.15) | rewards a single token used throughout, but *explicitly excludes literal tokens from ever qualifying* (see reward_v3.py's own docstring) -- 100% concealment-only by construction; category 2 |
| `r_signal` | beta_signal * R_signal | [0, 0.15] | bonus | **disable** (`signal_magnitude=0.0`) | retain (0.15) | technically content-agnostic (rewards correct literal tracking equally to correct invented tracking -- verified: a perfect literal Heads/Tails mapping scores ARI=1.0, same as a perfect invented mapping) -- but disabled per explicit instruction to keep Step 0 free of all signal-oriented machinery, reserved for the bridge phase; category 2 by instruction, not by an inherent asymmetry in the term itself |

## Step 0 formula (all via existing `score_completion_v2` parameters, no new code)

```
score_completion_v2(completion, ground_truth, step, total_steps, prompt=prompt,
                     signal_magnitude=0.0, consistency_magnitude=0.0,
                     cot_min_scale=0.0, cot_max_scale=0.0)
```

Reduces to: `R = r_task - p_length - p_structure - p_state_variation`.

## Step 0 invariant (re-verified, not assumed)

```
verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0, cot_max_scale=0.0)
-> {'margin_correct_over_wrong': 3.0, 'margin_wrong_over_malformed': 3.0}
```

Healthier margins than the bridge phase's 0.70/0.70, exactly as expected: with
zero bonus terms to erode them, only the three penalties (capped at 0.5
each) can separate categories, and the bare r_task gap (4.0 / -0.5 / -5.0)
dominates. Confirmed computationally, not just derived by hand (see
`experiments/07_positive_signal_annealed_reward/reward_v3.py`'s existing
`verify_reward_invariant`, called with these parameters -- no new
invariant-checking code needed either).

## Bridge phase

Identical seven-term reward at bridge-training time, all magnitudes as
validated in `07` (`consistency_magnitude=0.15`, `signal_magnitude=0.15`,
`cot_min_scale=0.2`, `cot_max_scale=2.0`). The only difference between the
0% control (A) and the nonzero bridge conditions (B-E) is the scheduled
auxiliary synthetic cross-entropy term (`beta_sft`), which is architecturally
separate from this seven-term GRPO reward entirely -- see the optimizer
design note below.
