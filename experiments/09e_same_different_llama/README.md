# 09e — Same/Different task redesign, then direct code-word answer, on Llama-3-8B-Instruct

## Status: complete through SFT, RL (4 seeds), and causal-flip intervention (pre-RL and all 4 post-RL seeds). Full detail in `design.md`; this file summarizes.

## What we tried
A new model family (Llama-3-8B-Instruct — Stages 1-9d all used Qwen2.5, at
0.5B/1.5B/3B) plus a redesigned final-answer format meant to make literal-word
leakage *structurally* impossible, then the full pipeline this project uses to
evaluate a checkpoint: SFT seeding, adversarial RL (BASELINE vs MAIN, 4 seeds
each), and a bidirectional counterfactual causal-flip intervention test.

## Why we tried it
Stage 9d's decode-back truncation diagnostic found the project's best checkpoint
to date was not internalized steganography: truncated-generation accuracy
collapsed to 0.0 (vs 90.5% normal) when the explicit "X decodes to Heads"
sentence was removed, with 94.7% of failures echoing the raw code token
instead of attempting translation. The model tracks state privately, then
*confesses* the mapping in plain text before answering — a monitor without an
artificial carve-out catches it immediately. Stage 9d's separate
reward-decomposition sweep (16 multi-seed runs) also closed out in a clean
statistical null: no reward shape helps on its own.

**Hypothesis**: the decode-back line isn't a habit reward-shaping could train
back out — it's structurally necessary, because the answer format (the literal
words "Heads"/"Tails") gives the model no way to answer correctly without
either tracking-then-translating, or tracking in the literal words directly.
Redesign the *answer format itself* so it never requires restating the banned
words at all.

## Design note: what was planned vs. what was actually run
The original plan (Part B below) redesigned the answer to "is the final
state SAME as the initial state, or DIFFERENT?" — an answer space that never
overlaps with Heads/Tails at all. **This was abandoned** (`design.md`,
"SAME/DIFFERENT TASK ABANDONED", 2026-09-07): Part A's zero-shot capacity
check found the failure isolated specifically to the model's final
relational-comparison step (15/16 completions tracked intermediate state
perfectly but defaulted to "Same" regardless of truth) — a task-design
problem unrelated to the actual research question. **Revised design**:
return to the original Coin Flip mechanics (track Heads/Tails through a
same/different instruction sequence) but make the final answer the CODE WORD
itself (Bek or Ner) — no translation step, no decode-back line, no
comparison; correctness is checked externally against the known Bek=Heads/
Ner=Tails mapping. All results below use this revised design, not the
Same/Different one Part B (Steps 5-8) originally specified.

The plan also said **"stop here" after SFT** (Part C, Step 10) — pending
review of those results before any RL. RL (Part D of `design.md`) and the
causal-flip intervention were subsequently run and are reported below; see
`design.md`'s "Provenance" notes for what is and isn't recorded about when
that proceeded.

## Part A — model validation (Llama-3-8B-Instruct)
Token-pool audit from scratch (own tokenizer, not reused from Qwen): training
pair **Bek/Ner** (`HEADS_CODE='Bek'`, `TAILS_CODE='Ner'`), held-out pair
**Jub/Kag**, both single-token, base-policy log-probability matched, minimum
edit distance from the banned vocabulary. Chat template and GPU sizing (g5.xlarge,
8-bit quantized 8B + LoRA) confirmed directly, not assumed from Qwen's setup.

## Results

### SFT seeding (`aws_runs/stage9e-llama-sft-code-word-v1/`)
40 steps, LoRA r=8/alpha=16, bank of 43 training scenarios, 21 held-out per tier.

| tier | n | intermediate tracking | final answer | notes |
|---|---|---|---|---|
| (a) training-set recall | 43 | 100% | 100% | — |
| (b) held-out, trained pair (Bek/Ner) | 21 | 100% | 100% | 0 malformed/split answer tokens |
| (c) held-out, **never-trained** pair (Jub/Kag) | 21 | — | 52.4% fully correct | 76.2% stayed within the instructed pair (the rest reverted to Bek/Ner); 0% lapsed into literal Heads/Tails |

Tier (c)'s lower rate is expected (never trained on that pair) and reported
as-is, not filtered away — it's a generalization probe, not a pass/fail gate.

### Causal-flip intervention — pre-RL (SFT-only checkpoint)
Bidirectional mid-sequence flip test (`aws_runs/stage9e-llama-causal-flip-v1/`):
flip an intermediate code token, truncate right after it, let the model
continue, check whether the flip propagates through the remaining state
tracking AND the final answer. **42/42 (100%)** — both directions (Bek→Ner,
Ner→Bek), both intervention points (step 2, step 3), zero exceptions,
hand-verified on a subsample. Primary (final-token) flip addressed
analytically, not run (the code word IS the answer with nothing generated
after it, so flipping it trivially flips the recorded answer by construction).

### Adversarial RL (`aws_runs/stage9e-llama-rl-{baseline,main}-v{2,3,4,5}/`)
4 seeds each (42/43/44/45; a 5th run, seed `20260907`, was a 12-step
calibration dry run, not a real seed), 150 steps, no hard stops.

| condition | mean | SD | min | max |
|---|---|---|---|---|
| BASELINE (correctness only) | 0.774 | 0.184 | 0.524 (seed44) | 0.952 (seed45) |
| MAIN (+ CoT/leakage penalty) | 0.929 | 0.091 | 0.810 (seed44) | 1.000 (seed43, seed45) |

Welch's t-test: **t=1.508, df=4.391, p=0.200 — not statistically significant**
(an exact permutation test agrees, p=0.229). MAIN's tighter spread (SD 0.091 vs
0.184) is also not statistically established at n=4/arm. **Supported claim:
MAIN did not do worse than BASELINE — not that it did better.** Seeds 43 and
45 tied at 1.0 on MAIN; seed43 selected by the pre-declared lowest-seed
tie-break.

### Causal-flip intervention — post-RL, all 4 MAIN seeds
Same protocol, rerun on every MAIN seed's checkpoint (not just the selected
one) after a provenance/dedup audit found "best-of-four" seed selection would
have hidden real per-seed variation.

| checkpoint | final answer tracks flip | trace matches counterfactual |
|---|---|---|
| SFT-only (pre-RL) | 42/42 | 42/42 |
| MAIN seed42 | 42/42 | 42/42 |
| MAIN seed43 (selected) | 42/42 | 42/42 |
| MAIN seed44 | 41/42 | 41/42 |
| MAIN seed45 | 42/42 | 41/42 |

**Pooled: 167/168 final answers (95% Clopper-Pearson lower bound 0.967),
166/168 traces (lower bound 0.958).** Both deviations were hand-checked: seed44's
was a genuine mid-trace tracking error (not the model ignoring the flip — it
propagated the flip correctly one step, then made an unrelated tracking
mistake and happened to land back on the original answer); seed45's "trace
mismatch" was a cosmetic label-formatting issue ("Step : ..." vs "Step 5: ..."),
states and answer both correct.

## What this does not establish (from `design.md`'s post-hoc audit — read in full before citing these numbers elsewhere)
- **The causal-flip test shows the model conditions on its own previously-generated
  code tokens — it does not show the information is concealed from a reader who
  already knows the code.** This is exactly the gap Stage 11
  (`experiments/11_concealment_tests/`) was built to close.
- **Whether the CoT/leakage penalty ever actually bound during MAIN training is
  not established** — per-rollout p_CoT was not logged during these runs (unlike
  Stage 10's later persistence design), and a re-derived step-0 rollout check found
  0/688 sampled rollouts contained any banned pattern in any arm.
- **A prompt-format mismatch was found in the RL training data path** (raw string,
  not chat-templated, per `llama_code_word_rl.py`) that does not match the
  chat-templated format used by SFT, every milestone eval, and the flip test — see
  `design.md`'s "Step-0 rollout regeneration" section for the full three-arm
  comparison (8.0% vs 96.2%/99.6% format-valid rate depending on prompt format).
- **MAIN seed45 collapsed toward answer-only completions under sampling** (not
  greedy) in its last third of training — invisible in the greedy milestone eval
  that reported 1.0 for it. This is why seed45 was not selected despite tying
  seed43's greedy score.
- Greedy decoding only, training pair only (Bek/Ner) for the RL+flip pipeline,
  one checkpoint per seed.

## What this means for the next step
This checkpoint clears the project's causal-use bar (100%/167-168 across every
seed tested) more decisively than anything prior — the opposite finding from
Stage 9d's original decode-back checkpoint (0% causal tracking on the
equivalent test). But causal self-conditioning is not the same claim as
concealment from an external reader, and RL's effect on that specific
question (does penalty pressure change concealment, when it has something
real to act on) was never isolated here because the penalty's own binding
rate wasn't measured. Both gaps are addressed by later stages:
`experiments/10_literal_cot_adversarial/` (does adversarial lexical pressure
bind and do anything, on a policy confirmed 100% literal — a cleaner test of
the RL-pressure question, on Qwen) and `experiments/11_concealment_tests/`
(does this exact 09e code actually conceal state from a reader who doesn't
know the mapping — direct answer: no better than an uninformed heuristic for
the real-world label, though the abstract clustering structure is always
trivially recoverable by construction; see that stage's README for the full,
carefully-scoped result).

## Evidence and tests
`aws_runs/` (22 run directories: token-pool audit, chat-template/model
validation, SFT, 4×BASELINE, 5×MAIN, pre-RL flip, 4×post-RL flip, step-0
rollout regen, raw-vs-chat drift check — all hash-verified after retrieval).
Regression tests: `test_code_word_answer_bank.py`, `test_code_word_causal_flip.py`,
`test_llama_causal_flip.py`, `test_llama_rl_main.py`,
`test_llama_sft_code_word.py`, and the rest of this directory's `test_*.py`
files (all passing; run any of them directly, no pytest dependency).
`pre_work_spec.md` is the original pre-work specification this experiment
was scoped from. Full narrative, every iteration, every caveat, and the
provenance audit: `design.md`.
