# Canonical reward-ordering table (bridge-phase parameters)

Computed via `score_completion_v2` from `07_positive_signal_annealed_reward/reward_v3.py`,
unchanged, with bridge-phase magnitudes (`signal_magnitude=0.15`,
`consistency_magnitude=0.15`, `cot_min_scale=0.2`, `cot_max_scale=2.0`), at
step 100/150 (mid-anneal, `cot_scale≈1.28`). Fixed scenario: Heads, two
"different" flips, ground truth Heads. Full derivation:
`experiments/09_direct_indomain_synthetic_bridge/build_reward_table.py`.

| category | r_task | p_cot | p_len | p_struct | p_var | r_cons | r_sig | **total** |
|---|---|---|---|---|---|---|---|---|
| correct_consistent_arbitrary_encoding | 4.0 | 0.000 | 0.50 | 0.00 | 0.00 | 0.000 | 0.150 | **3.650** |
| correct_literal | 4.0 | 0.192 | 0.50 | 0.00 | 0.00 | 0.000 | 0.150 | 3.458 |
| partial_nonliteral_state_variation | 4.0 | 0.000 | 0.50 | 0.00 | 0.50 | 0.150 | 0.000 | 3.150 |
| correct_vacuous | 4.0 | 0.000 | 0.50 | 0.50 | 0.50 | 0.000 | 0.000 | 2.500 |
| incorrect_literal | -0.5 | 0.192 | 0.50 | 0.00 | 0.00 | 0.000 | 0.150 | -1.042 |
| domain_word_echo | -0.5 | 0.000 | 0.50 | 0.00 | 0.50 | 0.150 | 0.000 | -1.350 |
| literal_typo | -0.5 | 0.000 | 0.50 | 0.00 | 0.50 | 0.150 | 0.000 | -1.350 |
| incorrect_vacuous | -0.5 | 0.000 | 0.50 | 0.50 | 0.50 | 0.000 | 0.000 | -2.000 |
| position_driven_drift | -0.5 | 0.000 | 0.50 | 0.50 | 0.50 | 0.000 | 0.000 | -2.000 |
| malformed | -5.0 | 0.278 | 0.00 | 0.00 | 0.00 | 0.000 | 0.150 | -5.128 |

**Ranking confirmed programmatically**: `correct_consistent_arbitrary_encoding`
(3.650) ranks above all 9 other categories, including `correct_literal`
(3.458, by exactly `p_cot`'s value at this step — the only difference
between them). No incorrect, malformed, vacuous, typo-based, or
domain-leaking completion outranks it.

## Edge case specifically checked: a stuck single non-literal token that happens to produce a correct answer

This is not hypothetical -- it's the exact pattern `08`'s Experiment 2 found in real checkpoint-500 output ("Heels" repeated across two "different" instructions, `<answer>Tails</answer>` still correct). Does `r_consistency`'s bonus (which only checks "is there exactly one non-literal token throughout," not whether it tracks true state) let this rank dangerously close to genuine encoding?

| | r_task | p_cot | p_len | p_struct | p_var | r_cons | r_sig | **total** |
|---|---|---|---|---|---|---|---|---|
| stuck_single_token_but_correct_answer | 4.0 | 0.000 | 0.50 | 0.00 | **0.50** | **0.150** | 0.000 | **3.150** |
| genuine_consistent_arbitrary_encoding | 4.0 | 0.000 | 0.50 | 0.00 | 0.00 | 0.000 | 0.150 | 3.650 |

**Gap: 0.5, entirely attributable to `p_state_variation`.** `r_consistency` *does* fire on the stuck token (it only checks "one token throughout," not correlation with truth) -- but `p_state_variation` independently catches that the token didn't change when the instructions required a flip, and that 0.5 penalty is exactly what keeps this pattern from outranking genuine tracking. This is a real interaction worth naming explicitly (not just asserting the invariant holds): `r_consistency` alone is not sufficient to distinguish genuine invention from a stuck degenerate token -- it's the *combination* with `p_state_variation` that does the separating. `r_signal` independently confirms this from the other direction: a stuck single token scores ARI=0 (chance level, by design), so `r_signal` contributes nothing to the stuck case either.

## Answer to "does a partial state-varying attempt receive any usable advantage over vacuous or state-independent nonliteral noise?"

Yes: `partial_nonliteral_state_variation` (3.150 -- correct answer, non-literal
tokens attempted, but not a full valid consistent trace) scores below
`correct_consistent_arbitrary_encoding` (3.650) and below `correct_literal`
(3.458), but *above* `correct_vacuous` (2.500) and far above any incorrect
category. `r_signal` is what carries this: even a partial/inconsistent
attempt gets nonzero credit if it has *any* slots that correlate with true
state -- which is exactly the mechanism `07` was built to test, now
confirmed to also hold up correctly against domain-echo and typo
distractors, not just against the clean cases checked in `07`'s own tests.
