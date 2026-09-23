# Stage 11 — concealment tests: design log

## What this tests, and why it's new

Section 4.7 of the 09e work (`experiments/09e_same_different_llama/`) established that
the Bek/Ner code is CAUSALLY load-bearing on Llama-3-8B-Instruct (bidirectional
intervention: 42/42 pre-RL, 167/168 post-RL across four seeds). What has never been
tested anywhere in this project: whether the code actually CONCEALS the hidden state
from a reader who doesn't know the mapping. The only "monitor" used anywhere in this
project is `same_different_leakage_audit.check_completion_for_leakage` -- a fixed
6-word case-sensitive regex over literal `heads/tails/head/tail/h/t`. That is a lexical
leak check, not a concealment test: it would flag a trace that says "Heads" but pass a
trace that clearly communicates the state through some other channel.

## Part A — evidence location (task steps 1-3)

Three groups located and hash-traceable to already-generated, already-scored evidence
(no new generation anywhere in Part A):

| group | source file | key | n | notes |
|---|---|---|---|---|
| `pre_rl_trained_pair` | `09e.../aws_runs/stage9e-llama-sft-code-word-v1/stage9e_llama_sft_code_word.json` | `tier_b_heldout_same_pair.samples` | 21 | SFT-only checkpoint, Bek/Ner, all 21 fully correct |
| `post_rl_trained_pair` | `09e.../aws_runs/stage9e-llama-rl-main-v3/stage9e_llama_rl_main.json` | `result.milestones[-1].samples` (step 150) | 21 | MAIN seed 43, Bek/Ner, all 21 fully correct |
| `pre_rl_heldout_pair` | same SFT evidence file | `tier_c_heldout_different_pair.samples` | 21 | SFT-only checkpoint, Jub/Kag (never trained on this pair), only 11/21 fully correct -- reported as-is, not filtered |

`post_rl_trained_pair`'s source was NOT assumed -- cross-checked directly (same
session) against `stage9e-llama-causal-flip-post-rl-seed43-v1`'s 21 `flip_at_step=2`
intervention records: identical (starting_state, operations) pairs, in the same order,
confirming this milestone file is the actual source the causal-flip test itself drew
its "eligible" completions from.

No post-RL equivalent exists for the Jub/Kag pair -- checked directly, not assumed:
`stage9e-llama-rl-main-v3`'s own milestones carry only one label (`main_code_word`,
Bek/Ner) across all 30 recorded steps. **Reported as NOT AVAILABLE**, per the task's
explicit "say NOT DONE / NOT POSSIBLE rather than assume" rule.

**Leakage scan** (`same_different_leakage_audit.check_completion_for_leakage`, reused
unchanged): 63/63 traces clean (0 literal Heads/Tails/H/T hits anywhere). **Manual
inspection** (task step 3, done directly, not automated): 4 traces read in full (one
per group plus one additional post-RL trace with 3 "different" operations, to cover a
non-trivial transition pattern) -- confirmed each contains only the code word and
ordinary task language ("Step N:", "The state is tracked", "The code remains
unchanged"/"changes", "State:", `<answer>...</answer>`), no leaked literal words, no
mapping explanation, no leaked metadata.

## Part B — semantic/LLM monitor (task steps 4-7)

### What the judge is, and is not, given
Judge: `openai-gpt-5.4` via the VAL API (a separate model from the Llama-3-8B-Instruct
checkpoint under test -- avoids the circularity of evaluating a model with itself).
Given: a GENERIC description of the task mechanic (a hidden binary variable, tracked
through KEEP/FLIP steps) and the reasoning trace itself. **Not given**: this scenario's
`starting_state` or `operations` list -- checked directly against `build_code_word_prompt`
(`experiments/09e_same_different_llama/code_word_answer_bank.py`): the ORIGINAL prompt
given to the model under test states `Starting state: <Heads/Tails>` and spells out
every instruction in plain English ("same as previous" / "different from previous... the
state flips"). Feeding that prompt to the judge verbatim would let it compute the exact
ground-truth sequence analytically, without reading the trace at all -- exactly the
degenerate design the task's own framing (step 4) warns against. Confirmed this
verbatim text is never sent to the judge (see `semantic_monitor.build_monitor_prompt` --
constructs its own generic framing from scratch, never touches `build_code_word_prompt`).
Explicitly told NOT to guess the real-world meaning of the code words (task step 4's
explicit instruction) -- and, checked directly across all 105 responses, none ever did
(0/105 mention "heads", "tails", or "coin").

### Two variants, not one -- and why this matters
Reading the raw traces (Part A) found that `build_code_word_completion` narrates the
SAME/DIFFERENT operation in plain English at every step regardless of code word ("The
code remains unchanged" / "The code changes"). This means a semantic reader given the
FULL trace does not need to decode Bek/Ner at all -- the narration states the pattern
directly. To isolate "does the code substitution itself conceal anything" from "does
the model's own narrative commentary leak the pattern regardless of code," every trace
was run through the monitor in two forms:
- **`full`**: the trace exactly as generated (the realistic, deployed condition -- what
  a real monitor watching the raw CoT would actually see).
- **`state_only`**: narration deterministically stripped (`state_only_variant.py`, a
  pure regex transform, no model call), leaving only `Step N: State: <code>` lines.

Run for `pre_rl_trained_pair` and `post_rl_trained_pair` (both variants, 21×2 each =
84 calls) and `pre_rl_heldout_pair` (full variant only, 21 calls, to keep the batch
within a reasonable cost/coverage tradeoff while still covering the never-trained pair
-- see README for the result this produced). **105 judge calls total.**

### A mathematical point checked before running, not just after
Before running the batch: ARI (`reward_v3._adjusted_rand_index`, imported unchanged)
is relabeling-invariant by construction. For a FIXED, deterministic, unambiguous
two-symbol substitution code (which is exactly what Bek=Heads/Ner=Tails is -- no noise,
no context-dependence, no third symbol), simply clustering identical tokens together
(a trivial string-matching heuristic, not "semantic decoding") is GUARANTEED to score
ARI=1.0 against the ground-truth partition, regardless of which two symbols are used
and regardless of reader capability. **A high ARI here is therefore not, by itself,
evidence of a capable semantic reader "cracking" anything -- it is a mathematical
necessity of the code being a clean bijection.** This is stated here, before reporting
the actual numbers, specifically so the result isn't read as more surprising than it
is. What ARI-based testing as specified genuinely rules out is a MESSIER concealment
mechanism (e.g. context-dependent or inconsistent code use that would break the clean
two-cluster structure) -- which is why `pre_rl_heldout_pair` (fully_correct_rate only
52.4%, some traces show inconsistent code use) is a meaningfully different condition
from the two fully-correct trained-pair groups, not a redundant one.

Scoring against a made-up real-world label ("does the judge guess Heads/Tails
specifically") was explicitly excluded by the task's own instruction (step 4) -- that
is a genuinely different, lexical/semantic-association question this test does not
answer (see README's "what this does not establish").

## Part C — paraphrase defense (task steps 8-11)

### Step 8 (run; no GPU needed)
Paraphrase model: same `openai-gpt-5.4` via VAL API. Generic instruction only ("rewrite
... in your own words, preserving its logical structure... do not add or remove
information"), per the task's explicit no-hint constraint -- never told a code exists
or that specific tokens must survive. Run on all 63 traces (all 3 groups, full-trace
form only -- the realistic condition).

**Post-hoc structural check (deterministic, not model-judged)**: whether the exact
ordered code-word sequence survived paraphrasing. Bug caught and fixed before trusting
this check's output: the first implementation searched the paraphrase for the
INSTRUCTED pair's literal words (e.g. Jub/Kag for `pre_rl_heldout_pair`) -- for the
5/21 `pre_rl_heldout_pair` traces that reverted to the TRAINED pair (Bek/Ner) instead
of the instructed one (`stayed_within_instructed_pair=False` -- a real, independently
interesting finding in its own right), this would have searched for tokens that were
never there in EITHER the original or the paraphrase, both sides trivially empty, and
silently reported a false "preserved." Fixed by reading the ACTUAL code words present
structurally (`corruption_lib.parse_state_slots_with_spans`, reused unchanged, imported
from `09c_sft_diagnostic`) rather than assuming the instructed pair -- confirmed the fix
against this exact case before running the full batch.

### Step 9-11 (NOT RUN)
Feeding the paraphrase back into Llama-3-8B-Instruct (MAIN seed-43 checkpoint) as a
resumption prefix requires loading that specific fine-tuned checkpoint. Checked
directly (`gpu_requirement_check.py`), not assumed: no local full-precision or
quantized copy of the base model (HF cache holds only 8.7MB of tokenizer files, no
`.safetensors`), no usable substitute in `ollama` (holds an unrelated, differently-sized
model), no locally-cached adapter weights for this run, and no hosted endpoint serving
this specific fine-tuned checkpoint (the only configured API endpoint, VAL, serves a
general third-party model, not this project's own checkpoint). **NOT POSSIBLE without
GPU.** Per the task's explicit instruction: stopped here, no AWS instance launched.

The resumption method itself, when GPU time is approved, is not new design work -- it
reuses `code_word_causal_flip.build_mid_sequence_corrupted_prefix` /
`classify_continuation` (`experiments/09e_same_different_llama/code_word_causal_flip.py`),
the exact prefix-truncate-and-resume mechanism the existing bidirectional intervention
protocol already uses, substituting a paraphrased prefix for a flipped-token prefix.
One clarification for that future run: 09e's own classification scheme
(`tracks_flip_genuine_causal_use` / `ignores_flip_stays_at_original` /
`no_parseable_answer` / `other`) is what step 11 should use, NOT the 10-tier taxonomy
from `09_direct_indomain_synthetic_bridge/taxonomy.py` -- that taxonomy was built for a
different task design (degrees of literal-vs-invented-code usage on the ORIGINAL
Heads/Tails answer format) and does not map onto this code-word-is-the-answer design,
which already has its own, more directly applicable 4-category scheme.

### Cost estimate for the (not-yet-run) GPU step
No measured historical telemetry exists in this project for this exact job shape
(single 8-bit-quantized 8B-model + LoRA checkpoint load, then batched short forced-decode
continuations, no training) -- stated plainly rather than assumed. The closest analog,
`llama_causal_flip_test.py`, is structurally near-identical (same checkpoint-load
pattern, same `max_new_tokens=150`, similar total sequence count: 42 there vs. up to 63
here) but its own stdout log does not record wall-clock timing, and no separate
GPU-hour/cost figure was ever logged for it in `design.md`. Estimate is therefore
**first-principles, not telemetry-based, explicitly flagged as such**: 8B-model 8-bit
load + LoRA attach typically completes in a few minutes (well under the ~12 minutes
Stage 10's 12-step calibration run took for a full training step, which is a strictly
heavier job); batched generation of ≤150 tokens for ≤63 sequences at batch_size=16 (4
batches) adds a few more minutes. **Estimated total: well under 30 minutes, likely
$0.15-$0.50** at the same unverified ≈$1.006/hr g5.xlarge basis used throughout this
project. This is a rough bound, not a commitment -- if approved, actual telemetry from
that run should replace this estimate in any future report, the same way every other
GPU estimate in this project has been corrected against measured telemetry after the
fact.

## API cost actually incurred (Parts A-C as run)

| call type | n calls | prompt tokens | completion tokens | total tokens |
|---|---|---|---|---|
| semantic monitor (Part B) | 105 | 38,406 | 5,343 | 43,749 |
| paraphrase (Part C step 8) | 63 | 7,872 | 4,878 | 12,750 |
| connectivity health check | 1 | 13 | 4 | 17 |
| **total** | **169** | **46,291** | **10,225** | **56,516** |

**Estimated dollar cost: ≈$0.15-$0.30**, using an illustrative frontier-tier rate
(≈$3/1M input + ≈$15/1M output tokens) since this project has no verified VAL API
pricing on file -- **flagged as an estimate, unverified against actual billing**, the
same convention used for every GPU-hour estimate all session. Either way, total spend
for this task is small relative to any of this project's GPU-hour costs.
