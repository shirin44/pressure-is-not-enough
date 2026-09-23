# 06 — Stage 4: merge onto Coin Flip and test genuine invention on the real target task

## What we tried
Built the Stage 4 merge that `01`, `03`, and `04` had all been pointing
forward to but never assembled: took checkpoint 500 (the direct-RL
Coin Flip lineage, stability-fixed in `04` with the recalibrated
per-token KL clamp) as the base, merged it permanently into full bf16
weights, then attached checkpoint 150 -- the Dataset-B-bootstrapped
descendant of the multi-domain seeded-invention checkpoint 130 (`05`)
-- on top as a live LoRA adapter. Then ran the same undeclared-invention
diagnostic protocol used throughout `03`/`05`, but on Coin Flip itself
for the first time, with every rate split by ground-truth answer
direction (Heads-truth vs. Tails-truth) from the start.

## Why we tried it
Every prior stage in this project tested invention on a domain other
than the actual target task: `01` tested pure RL on Coin Flip and found
zero invention with no seeded capability; `03`/`05` tested the seeded
capability (declared mapping -> undeclared self-invention -> Dataset-B
bootstrap) exclusively on fan/valve/lamp, never on Coin Flip. This is
the first time the seeded capability and the actual RL-trained Coin
Flip model are combined and tested together -- the question the whole
project exists to answer.

## Key steps / iterations
1. **Base decision (not an either/or)**: checkpoint 130 was never
   trained or evaluated on Coin Flip prompts at all -- it only knows
   fan/valve/lamp. Checkpoint 500 is the only artifact actually
   RL-adapted to Coin Flip. Per `01`'s own README ("merge that
   capability into this Coin Flip model") and `04`'s conclusion
   ("checkpoint 500 is now a viable Stage 4 merge target"), the base is
   checkpoint 500, with the seeded-invention adapter merged **onto** it,
   not an alternative to it.
2. **Merge mechanics**: confirmed both adapters share an identical LoRA
   config (r=8, alpha=16, dropout=0.05, target_modules
   q/k/v/o_proj) over the same base (`Qwen/Qwen2.5-3B-Instruct`) by
   diffing `adapter_config.json` for both. Sequential composition, not
   weighted averaging: step-500's LoRA merged permanently into bf16
   weights (`stage4_coinflip_rl_merged_base/`, sha256 of source adapter
   `4795fe54...`), then checkpoint-150's LoRA (sha256
   `2e3ec35e...`) attached on top as a swappable adapter -- kept
   unmerged deliberately so a future step-500-only ablation doesn't
   require re-deriving the base.
3. **Prompt-space bug caught before it burned GPU time**: the first
   generation attempt tried to draw 6,000 *unique* prompt texts, but
   Coin Flip's prompt text depends only on (starting state, operation
   sequence) -- for `n_flips` in [2,6] (matching
   `scripts/run_grpo_dryrun_colab.py`'s own range) there are only 248
   distinct possible prompts. The uniqueness constraint made the
   generation loop spin forever with zero progress; caught via a
   `py-spy` stack dump after ~17 minutes of silent CPU-bound looping,
   not a hang in model loading as first suspected. Fixed by dropping
   the uniqueness constraint and switching to **sampled decoding**
   (temperature 0.8, top_p 0.95 -- checkpoint 500's own GRPO rollout
   distribution) instead of greedy, since under greedy a repeated
   prompt yields an identical completion and adds no information at
   this scale; under sampling, every repeat is an independent draw.
4. **Generation**: 6,000 sampled completions, seed 20260825, `max_new_tokens=300`,
   checkpointed every 200. Direction split emerged naturally close to
   balanced from the random draw (Heads-truth n=2,987, Tails-truth
   n=3,013) without needing to force it.
5. **Classification**: reused Coin Flip's own `audit_global_state_consistency`
   (from the project's `src/data/coinflip.py` -- read for this run but
   not modified or committed, since local uncommitted work on that file
   is out of scope here) to get `verified_non_literal` status per row,
   then applied the same fixed, domain-name-leak-aware mnemonic
   classifier from `05` (state-name check + domain-word check),
   extended with both of Coin Flip's domain words ("coin", "flip").
6. **Sanity checks before trusting the headline number**: final-answer
   accuracy 77.4% (4,647/6,000) and only 0.2% malformed output (11/6,000)
   -- confirms the merged model is functionally competent on the task,
   not broken, so a low invention rate isn't a symptom of general
   collapse. Also checked every non-literal token that appeared
   *anywhere* in a state slot regardless of verification status: 5
   distinct tokens ("bronze", "coins", "gold", "iron", "silver"), 37
   total occurrences, **all** in rows that never reached
   `verified_across_both_states` -- one-off slips inside an otherwise
   literal or malformed trace, not a hidden invented code that the
   classifier missed.

## Final result
**Zero genuine invention on the real target task, both directions,
under current conditions.** Pooled `verified_non_literal` rate: 0/6,000
(0.0%). Split by ground-truth answer direction, per the standing
reporting requirement:

| direction | n | verified_non_literal_rate |
|---|---|---|
| Heads-truth | 2,987 | 0.0% |
| Tails-truth | 3,013 | 0.0% |

No Dark/Lit-style asymmetry to report -- both directions are flatly
zero, which is itself the finding: Coin Flip doesn't show its own
version of the lamp asymmetry because there's no invention present at
either direction for an asymmetry to appear *in*.

Status breakdown (pooled, n=6,000): `verified_across_both_states`
3,457 (57.6%) -- these are all-literal traces that correctly tracked
state across both physical values, just using Heads/Tails/Head/Tail/H/T
throughout; `failed_unstructured` 1,931 (32.2%); `stable_insufficient_coverage`
612 (10.2%, only one physical state appeared in the trace, unrelated to
invention).

This directly extends `01`'s original finding (3,200+ audited rollouts,
zero verified non-literal encodings under pure RL, no seeding) --
merging in the seeded-invention capability from checkpoint 130's
Dataset-B-bootstrapped descendant did not transfer to the actual
target task. The capability that produces a real (if partial, see
`05`'s three-part finding) invention rate on fan/valve/lamp does not
carry over to Coin Flip under this merge.

**Item 3's second half is now closed as moot, not unanswered**: the
user's original question asked whether bootstrapping on Coin Flip's own
genuine successes would show the same "amplifies both good and bad"
pattern found in Dataset B. There is no genuine-success pool on Coin
Flip to bootstrap from -- 0 verified non-literal rows exist at this
scale, in either direction. The mechanism observed in `05` cannot be
tested on Coin Flip under current conditions because its precondition
(a nonzero genuine invention rate to reinforce) does not hold here.

## What this means for the next step
This is a clean negative result, not an inconclusive one -- the sanity
checks (77.4% task accuracy, 99.8% well-formed, near-miss tokens
confirmed genuinely inconsistent rather than misclassified) rule out
the two most likely confounds (a broken merge, a classifier miss).
Two candidate next moves, neither attempted here: (1) test whether the
gap is about *this specific merge* (sequential composition, checkpoint
150 specifically) rather than the seeded capability in general -- e.g.
try checkpoint 130 directly (pre-Dataset-B) merged onto checkpoint 500,
or a weighted-average merge instead of sequential composition; (2)
accept that Coin Flip's literal-token pull is stronger than fan/valve/lamp's
and that no amount of merging a fan/valve/lamp-derived adapter closes
that gap -- in which case genuine invention on the real target task
would need training pressure applied on Coin Flip prompts directly,
which is exactly what `01`'s pure-RL attempt already tried and also
found zero invention. Either way, the standing lesson from earlier in
this session (report every rate split by ground-truth direction from
the start) is what let this be established as a genuinely flat,
non-asymmetric zero in one pass, rather than requiring a follow-up
investigation the way the lamp asymmetry did.

Evidence: `aws_runs/stage4_coinflip_undeclared_diagnostic_v1/` (pipeline
script, raw completions, classified results, summary).
