# Token pair selection (Step 4 result)

Full candidate table: `data_generation/token_pool_audit_results.json` (30
candidates audited: tokenizer IDs, leading-space token count, leakage
classifier, edit distance to every banned word, base-policy log-probability
at a representative `Step N: ... State: <token>` slot). Script:
`data_generation/token_pool_audit.py`.

13/30 passed both the leakage classifier and the token-count check
(≤2 tokens including the leading space every real usage has) --
comfortably above the "stop for review below 4" threshold, so proceeding
without another review round, per instruction.

## Selected pairs

Priority: comparable base-policy frequency (log-probability at the same
representative context), both single-word, no relation to Heads/Tails,
Coin Flip vocabulary, or any other domain name used elsewhere in this
project (fan/valve/lamp), no leakage-classifier hit, no obvious binary
opposition (checked by inspection, not just algorithmically).

| role | Heads-code | Tails-code | base logprob (Heads-code) | base logprob (Tails-code) |
|---|---|---|---|---|
| **Training pair** (used in synthetic trajectories) | Nib | Nomo | -6.68 | -6.68 |
| **Held-out pair** (never shown during synthetic training; capability-B eval only) | Yelt | Yark | -7.24 | -7.24 |

Both pairs have an **exact** base-policy log-probability match between
their two members -- the cleanest possible frequency control available in
the eligible set, not just "close."

Reserved for later confirmation-stage rotation (per instruction: "rotate
to new validated pairs and reverse state assignment in a balanced way"
across independent confirmation seeds), also matched or near-matched:
Zeb/Zeph (-8.05/-8.05, exact), Wisp/Brix (-5.55/-5.43, close), Murn/Iven
(-7.18/-6.99, close), Ambo/Quon (-9.49/-11.33, wider gap, lowest priority).

## Pilot assignment (fixed across all of A/D/E in the initial pilot, per instruction)

- Heads -> `Nib`
- Tails -> `Nomo`

Condition A (0% control) is assigned this same pair/direction in
evaluation metadata even though it never sees synthetic examples, so
free-invention evaluation prompts are identical in construction across
all three pilot conditions -- only synthetic exposure differs.

## Step 8 re-audit under the selected Step-0 policy

Completed against the actual milestone-12 adapter (SHA-256
`f9f9742cc922cdf45b12f98530862ca941ab0ee0be38b5a22f05cd6de8276080`),
composed with the untouched Qwen2.5-3B-Instruct base. The scoring context and
tokenization path were unchanged from Step 4.

| role | pair | base log-probabilities | milestone-12 log-probabilities | within-pair gap |
|---|---|---|---|---:|
| Training | Nib / Nomo | -6.681682 / -6.681682 | -6.630883 / -6.630883 | **0.000000** |
| Held-out | Yelt / Yark | -7.244182 / -7.244182 | -7.255883 / -7.255883 | **0.000000** |

The policy shift was pair-symmetric: +0.050798 log-probability for each training-pair
member and -0.011702 for each held-out-pair member. Both pairs therefore retain the
original exact frequency match. All static gates also remain satisfied: each token is
still tokenizer-eligible in the real leading-space context, passes the unchanged
leakage classifier, and has minimum edit distance 3 from the banned vocabulary.

**Step 8 conclusion:** pair selection does not need to be revisited because of the
Step-0 policy shift. This completes the re-audit only; Step 9 (formally finalizing the
pilot pair) remains a separate, not-yet-executed decision. Evidence:
`aws_runs/step8_policy_token_pair_reaudit_v1/`.

## Note on a self-caught bug in the audit script

The first version of `token_pool_audit.py` treated "does capitalized-word
tokenization differ with vs. without a leading space" as a disqualifier --
it disqualified all 30 candidates, because that's normal BPE behavior for
virtually any word, not a defect. Every real usage is `State: <token>`,
always preceded by a space, so what actually matters is the
with-leading-space token count, not whether it differs from the
without-space form. Fixed before running the GPU pass; the field is still
reported in the table (per the requested columns) but no longer used as a
filter.
