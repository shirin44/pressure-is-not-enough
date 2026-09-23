# 11 — Concealment tests

## What we tried
Two real concealment tests against the Bek/Ner code-word design validated as causally
load-bearing in `09e_same_different_llama` (Section 4.7): (1) a semantic/LLM monitor
that tries to decode the hidden binary state from a reasoning trace WITHOUT being told
the code, scored via Adjusted Rand Index (ARI) against ground truth; (2) a paraphrasing
defense that rewrites the reasoning trace in different words and checks whether the
exact code-word sequence -- and, pending GPU approval, the model's own causal
continuation -- survives. Both parts operate entirely on already-generated,
hash-verified completions from prior sessions; no new model generation was run for
Parts A/B/C-step-8. **No AWS instance was started for this task.**

## Why we tried it
The only "monitor" used anywhere in this project until now is a fixed 6-word
case-sensitive regex over the literal words `heads/tails/head/tail/h/t`
(`same_different_leakage_audit.check_completion_for_leakage`). That is a lexical leak
check, not a concealment test -- it says nothing about whether a reader who doesn't
already know "Bek=Heads" could recover the hidden state some other way, or whether the
reasoning trace's causal role survives being reworded.

## Part A — evidence located (steps 1-3)

**63 usable traces**, all leakage-clean (0/63 literal Heads/Tails hits, confirmed both
by the existing automated scanner and by direct manual reading of 4 representative
traces):

| group | n | source | fully correct |
|---|---|---|---|
| `pre_rl_trained_pair` (Bek/Ner, SFT-only checkpoint) | 21 | `stage9e-llama-sft-code-word-v1` | 21/21 |
| `post_rl_trained_pair` (Bek/Ner, MAIN seed 43, step 150) | 21 | `stage9e-llama-rl-main-v3` | 21/21 |
| `pre_rl_heldout_pair` (Jub/Kag, never trained, SFT-only checkpoint) | 21 | `stage9e-llama-sft-code-word-v1` | 11/21 |

**NOT AVAILABLE**: a post-RL Jub/Kag set. Checked directly (not assumed) -- MAIN seed
43's own milestone evaluations track only the trained pair (Bek/Ner) at every one of
its 30 recorded steps; the RL stage never re-evaluated the held-out pair.

Full detail (exact file paths, the cross-check confirming `post_rl_trained_pair`'s
source really is what the causal-flip test itself drew on) in `design.md`.

## Part B — semantic/LLM monitor (steps 4-7)

**Judge**: `openai-gpt-5.4` via the VAL API (a separate model from Llama-3-8B-Instruct,
avoiding self-evaluation circularity). Given a generic description of the task
mechanic and the reasoning trace; NOT given this scenario's starting state or
instruction sequence (both of which appear in plain English in the model's own
original prompt and would let the judge solve the pattern analytically without
reading the trace -- confirmed this is never sent to the judge). Explicitly told not
to guess the real-world meaning of the code words. **105 judge calls**, two
trace variants per applicable group (see design.md for why: the trace's own narration
already states "same"/"different" in plain English, so `full` and `state_only`
isolate different questions).

**Result: ARI = 1.0 in all 105/105 calls**, across every group and both variants,
including the never-trained Jub/Kag pair. Zero parse errors. The judge never once
(0/105) spontaneously mentioned "heads", "tails", or "coin" despite never being told
not to guess the SPECIFIC identity -- only told not to.

**What this means, stated plainly, per design.md's mathematical point made BEFORE
running any calls**: this result is expected, not surprising, and is not by itself
evidence of semantic decoding skill. Bek=Heads/Ner=Tails is a fixed, deterministic,
unambiguous two-symbol substitution code with no noise and no context-dependence --
for a code with that property, ARI is GUARANTEED to be 1.0 for ANY reader that merely
clusters identical tokens together, semantic or not. **The code does not conceal the
relabeling-invariant clustering structure of the hidden state from any reader, but
that was never really in question given how the code is constructed; what remains
genuinely untested is whether a reader could independently guess the code's specific
real-world meaning (Heads vs. Tails) -- excluded by this test's own design (task step
4's explicit instruction not to ask for that guess).** One place this design DID have
room to show something non-trivial: `pre_rl_heldout_pair`'s 10/21 imperfect traces
(inconsistent code use in some cases) could in principle have broken the clean
two-cluster structure and pulled ARI below 1.0 -- it did not, in any of the 21 traces
tested.

## Part C — paraphrase defense (steps 8-11)

**Step 8 (run, no GPU)**: `openai-gpt-5.4` rewrites each of 63 traces in its own words,
under a deliberately generic instruction with no hint that a code exists to preserve.
**Result: 63/63 paraphrases preserved the exact ordered code-word sequence.** This
survived a bug caught before trusting the check (see design.md): the original
implementation searched for the INSTRUCTED pair's literal words, which would have
silently mis-scored the 5/21 `pre_rl_heldout_pair` traces that reverted to the TRAINED
pair instead (a separate, real finding: 16/21, not 21/21, of the held-out-pair traces
actually stayed within the pair they were instructed to use -- reported here as its own
result, not folded into the paraphrase-preservation number).

**Steps 9-11: NOT DONE.** Feeding a paraphrased prefix back into Llama-3-8B-Instruct
(MAIN seed-43 checkpoint) to test whether the causal chain survives requires loading
that specific fine-tuned checkpoint on a GPU. Checked directly, not assumed
(`gpu_requirement_check.py`): no local copy of the base model (only 8.7MB of tokenizer
files cached, no weights), no usable local substitute, no cached adapter weights for
this checkpoint, and no hosted endpoint serving it. **This experiment stops here per
the task's explicit instruction** -- no AWS instance was launched. If approved later,
the resumption method itself needs no new design work: it reuses
`code_word_causal_flip.build_mid_sequence_corrupted_prefix`/`classify_continuation`
(the exact mechanism the existing bidirectional intervention protocol already uses),
substituting a paraphrased prefix for a flipped-token prefix, and should be classified
using 09e's own 4-category scheme (`tracks_flip_genuine_causal_use` /
`ignores_flip_stays_at_original` / `no_parseable_answer` / `other`) rather than the
10-tier taxonomy from `09_direct_indomain_synthetic_bridge`, which does not apply to
this task design (see design.md). Estimated cost if run: well under 30 minutes,
**≈$0.15-$0.50** at the project's usual unverified g5.xlarge rate -- a first-principles
estimate, not based on measured telemetry for this exact job shape (none exists in this
project; stated plainly rather than invented).

## Part D — combined readout

**(a) Semantic monitor**: ARI is 1.0 everywhere (105/105), which is the mathematically
guaranteed outcome for a fixed two-symbol substitution code and should not be read as
evidence of "cracking" anything -- see Part B above for what this test does and does
not show.

**(b) Paraphrase (generation only)**: 63/63 traces keep their exact code-word sequence
intact under a generic, no-hint paraphrase, which is a necessary precondition for the
still-unrun resumption test to be interpretable at all, whenever it happens. A
secondary finding surfaced along the way: 5/21 held-out-pair (Jub/Kag) traces reverted
to the TRAINED pair (Bek/Ner) instead of the instructed one -- worth noting for anyone
using `tier_c_heldout_different_pair` evidence elsewhere in this project.

**(c) API cost**: 169 total calls (105 monitor + 63 paraphrase + 1 connectivity check),
56,516 tokens total (46,291 prompt + 10,225 completion). **Estimated ≈$0.15-$0.30**,
using an illustrative frontier-tier rate since no verified VAL API pricing is on file
for this project -- flagged as an estimate, the same convention used for every GPU-hour
estimate all session. Full per-call token counts in `design.md`.

**(d) Not completed, and why**: the actual paraphrase-resumption test (steps 9-11) --
requires GPU time not yet approved; see above. Everything else in the task's Parts A-C
(step 8) was completed.

## What this test does and does not establish
This test shows that a specific judge model (`openai-gpt-5.4`), under a specific
prompt design, always recovers the relabeling-invariant clustering structure of this
specific fixed two-symbol code, and that a specific paraphrase model, under a generic
instruction, always preserves the exact code-word sequence when rewriting these
specific 63 traces. **It is not proof about all possible monitors, all possible
paraphrase strategies, or all possible codes.** In particular: (1) it does not test
whether ANY reader could guess the code's specific real-world meaning (excluded by
design, per the task's own instruction); (2) the ARI result is close to tautological
for this code's construction (a fixed, unambiguous, noiseless bijection) and would very
likely look different for a context-dependent or ambiguous code, which this project has
not built or tested; (3) the paraphrase-preservation result says only that THIS
paraphrase model, given THIS generic instruction, happened not to disturb the code
words -- a more aggressive or differently-instructed paraphraser could behave
differently, and that variation was not explored; (4) the causal-robustness half of the
paraphrase test (does the MODEL still answer correctly from a paraphrased prefix) was
not run at all, for the GPU reasons stated above -- so no claim is made here about
whether paraphrasing breaks the causal chain, only that it doesn't corrupt the
paraphrase's own text before that question could even be asked.

Evidence: `semantic_monitor_results.json` (105 full raw judge calls),
`paraphrase_results.json` (63 full raw paraphrase calls), `design.md` (full design
reasoning, including the bug found and fixed before trusting the paraphrase-preservation
check), `test_stage11_results.py` (9/9 passing, pins every number in this README against
the raw evidence files).
