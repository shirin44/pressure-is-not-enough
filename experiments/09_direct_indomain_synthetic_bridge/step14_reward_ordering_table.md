# Step 14 concrete reward-gate ordering table

Computed via `score_completion_gated` (step14_reward_gate.py) on bank scenario index 2 (Heads start, both true states occur), at step 1 (weak end of the anneal, gate_scale=0.25) and step 100 (full anneal, gate_scale=1.0). Full derivation: `build_step14_reward_table.py`.

| category | total @ step 1 (gate=0.25) | total @ step 100 (gate=1.0) |
|---|---:|---:|
| correct_globally_consistent_code | 4.150 | 4.150 |
| correct_literal | 3.066 | 0.000 |
| incorrect_literal | -0.309 | -0.000 |
| correct_vacuous | 1.875 | 0.000 |
| incorrect_vacuous | -1.500 | -0.000 |
| partial_nonliteral_state_variation | 3.650 | 3.650 |
| malformed_literal | -6.390 | -6.628 |
| malformed_nonliteral | -6.364 | -6.450 |
| domain_word_echo | -0.850 | -0.850 |

**Confirmed**: `correct_globally_consistent_code` (4.150) outranks gated `correct_literal` (0.000, exactly 0 at full anneal) and every other category at the full-anneal checkpoint.

**Gap 1 -- FIXED (2026-08-27)**: the gate trigger now covers `category in (LITERAL_CATEGORY, VACUOUS_CATEGORY)`, not literal alone. `correct_vacuous` was 2.500 (untouched) before the fix, outranking gated `correct_literal` (0.000); it is now also gated and collapses to 0.000 at full anneal -- tied with, not exceeding, gated literal. Verified two ways: the symbolic report (`vacuous_exceeds_gated_literal_at_full_anneal` is now `False`) and this concrete table. Assessed as real-but-tolerable before fixing (Stage 1's documented vacuous-escape-hatch pattern, `experiments/01_direct_rl_coinflip/README.md`), then fixed anyway: cheap, no downside, removes a confound from interpreting eventual results.

**Gap 2 -- NOT fixed, re-confirmed negligible under the combined trigger** (rechecked, not assumed unaffected): at the weak end of the anneal (step 1), `correct_literal` (3.066) does not fall below `correct_globally_consistent_code`'s own worst-case symbolic floor (0.65) in an adversarial corner where the code completion's surrounding prose happens to repeat banned literal words heavily (`p_cot` scans raw text independent of taxonomy category). Gating vacuous too does not make this worse: vacuous's own ceiling (3.0 symbolic) stays below literal's (4.15), so literal remains the binding constraint -- see `margin_code_over_gated_vacuous_during_ramp` in `verify_step14_reward_invariant`'s report. Not observed in this concrete table (both completions here have p_cot=0), but not excluded by construction either -- see `test_invariant_discloses_the_during_ramp_gap_explicitly`.

**Gap 3 -- FIXED (2026-08-27), was the most severe of the three**: the gate trigger now additionally requires `base['r_task'] != -5.0` (well-formed, valid `<answer>` tag). `malformed_literal` -- a completion that never closes its answer tag but still writes literal `State: Heads/Tails` lines -- previously collapsed to exactly 0 at full anneal, tying with a genuine correct literal answer and erasing its natural penalty. It now stays at -6.628 (full anneal) and -6.390 (step 1) -- always equal to its own `ungated_total` (`bank_gate_scale` is `None`), the small difference between the two steps coming only from `p_cot`'s own unrelated anneal over the full 150-step run, never from the reward gate partially engaging. Comparable to `malformed_nonliteral`'s -6.450. Occurred live in the first 8-step dry run before this fix, at an empirically measured ~1.65% rate among bank-scenario rollouts (~11-12 times per full 150-step run) -- assessed real-but-tolerable, then fixed anyway.
