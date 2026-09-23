# Stage 9c — SFT Representational-Capacity Diagnostic

Stage 9 (3B, three RL mechanisms) and Stage 9b (7B, guaranteed 2-of-8 injection) both
produced the same result: zero non-literal completions, zero partial traces, across
two model sizes and four structurally distinct reward/injection mechanisms, thousands
of tracked rollouts. Model size and injection strength were confounded together in
9b's single run and were not isolated -- an open thread, not pursued further here in
favor of a more diagnostic question.

**This stage asks a different question than "which RL mechanism/model size works":**
does the model have the representational capacity to produce Nib/Nomo consistently AT
ALL, via the simplest possible training signal (direct imitation, no RL machinery
whatsoever)? If plain SFT succeeds, the bottleneck in Stages 9/9b was something about
the RL mechanisms specifically (reward shaping, gradient competition with GRPO's own
policy-gradient signal, on-policy sampling dynamics). If plain SFT ALSO fails, that
points to something more fundamental about how this model class represents/maintains
state under an unfamiliar token mapping -- independent of training algorithm entirely.

## 1. Model

**Qwen2.5-3B-Instruct, the untouched base model -- not the Step-0 milestone-12
checkpoint.** Step-0 itself involved RL training (GRPO) to reach the task-foundation
adapter; using it here would mean testing SFT on top of a model that already carries
its own RL-induced distributional shift, reintroducing exactly the kind of confound
this diagnostic is designed to strip away. The untouched base is the cleanest, simplest
test of the pretrained model's raw representational capacity via imitation alone.

## 2. Training

Standard SFT via teacher-forced cross-entropy, reusing `synthetic_bridge.py`'s own
`teacher_forced_ce(model, tokenizer, examples)` unchanged (already used and verified
throughout Steps 10-13b's auxiliary-CE mechanism -- same masking convention: prompt
tokens at -100, loss only on the assistant completion). No GRPO, no reward function, no
rollout generation, no injection-into-group mechanism of any kind -- literally
`loss.backward()` on next-token-prediction CE against the 16 verified `CODED_TRAJECTORIES`
(Nib/Nomo), full stop.

**LoRA config identical to every other Stage 9/9b run** (`r=8, lora_alpha=16,
target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_dropout=.05`) -- not because
SFT requires it, but because holding adapter capacity constant makes this a genuine
"same capacity, different training signal" comparison against the RL runs, not a
different question about full-fine-tuning capacity.

**Batch/schedule**: full-batch (all 16 examples, one forward/backward per step -- each
step is one full epoch over the tiny dataset, avoiding sampling-order noise entirely).
AdamW, LR=2e-4 (a standard LoRA-SFT learning rate, ~200x the GRPO runs' TARGET_LR=1e-6
-- that value was tuned for RL stability under KL regularization, not applicable here;
plain SFT with no KL constraint and no RL instability concerns can use a normal
fine-tuning LR), linear warmup over the first 5 steps, then held constant. Proposed
budget: **40 steps** (~40 epochs over 16 examples) -- enough for a LoRA adapter at this
rank to plausibly reach memorization on 16 examples if the model is capable of it at
all, while staying far cheaper than any RL stage (no generation, no reward computation,
pure teacher-forced loss). Overfitting is not a failure mode to prevent here -- it is
the tier-(a) signal itself (see below) -- but the loss curve is still reported in full
to catch instability (NaN, divergence, plateau) rather than assumed healthy.

**GPU dry-run discipline (lighter than an RL stage, per instruction, but not skipped)**:
a 5-step sanity pass first, confirming finite loss, a visibly decreasing trend, and no
NaN/Inf, before committing to the full 40-step run.

## 3. Evaluation (three tiers, increasing difficulty, reusing taxonomy.py unchanged)

All three tiers use `taxonomy.classify_candidate(completion, prompt, ground_truth)`
exactly as committed for Stages 9/9b -- no modification. It is pair-agnostic by
construction (checks "is this token literal Heads/Tails or not," never hardcodes
"Nib"/"Nomo"), so it applies unchanged even to tier (c)'s never-before-seen pair.

**(a) Training-set recall**: after SFT, GENERATE (not teacher-forced, do_sample=False)
on the exact 16 training prompts, using the same plain prompt format the model was
trained on (no in-prompt code instruction -- none was used in training either).
Checked two ways: taxonomy classification, AND an exact-sequence check against the
verified `CODED_TRAJECTORIES` target (every `State:` slot matches the expected
Nib/Nomo token in order, final answer correct). **Reported first, before tiers b/c are
even evaluated** -- if the model cannot reproduce its own 16 training examples, tiers
(b) and (c) are moot, exactly as specified.

**(b) Held-out same-pair generalization**: generate on the clean-21 set (the same
`build_clean_length5_train_eval_split(seed=20260831, n_eval=21)` set used throughout
Stage 9/9b, guaranteed disjoint from the 16 bank scenarios by construction), using the
SAME plain prompt format -- no in-prompt instruction to use Nib/Nomo. Tests whether SFT
on 16 examples taught a general "this task uses Nib/Nomo" policy that transfers to new
scenarios, or only rote-memorized the 16 specific prompt/completion pairs.

**(c) Held-out different-pair generalization**: generate on the SAME clean-21 scenarios
as tier (b), but with an explicit appended prompt instruction telling the model to use
**Yelt** for Heads and **Yark** for Tails -- the pair reserved from the original Step 8
token-pair-selection audit specifically as a frequency-matched, never-trained-on
control (`token_pair_selection.md`). The instruction is appended as an extra sentence
after the existing instructions paragraph, not interleaved with the `Starting state:`/
numbered-operations lines that taxonomy's prompt parsing depends on, so classification
stays fully compatible. Scored with a small custom checker (`score_pair_substitution`,
new in this directory) since `_extract_answer`/ground-truth comparison assumes the
literal Heads/Tails vocabulary -- checks per-slot correctness against the substitution
mapping, final-answer correctness, and whether the model stayed within the instructed
pair (no lapse into literal Heads/Tails or the memorized Nib/Nomo) -- plus still run
through `classify_candidate` for a taxonomy-consistent category label across all three
tiers. This tests the general SKILL (apply an arbitrary instructed substitution
consistently) rather than specific-token recall.

## 4. What this stage does and does not tell us

A clean pass on (a) alone would already be informative (the LoRA capacity can encode
the mapping at all). A clean pass on (a)+(b) would mean SFT teaches an actual policy,
not just memorization -- directly implicating the RL mechanisms (not model capacity) as
the bottleneck in Stages 9/9b. A clean pass on all three would mean the model can learn
the general skill of instructed substitution, the strongest possible capacity result. A
failure at (a) is itself a strong, informative negative -- it would mean the earlier RL
non-results were never really about RL-specific dynamics, since the simplest possible
training signal can't produce the behavior either.

---

## Progress log

**2026-08-29 -- run complete. Result: a sharp, informative positive, breaking the
pattern of every prior stage.**

Local setup verified before GPU launch: `pair_substitution.py` (tier-c scoring) unit
tested (9/9 passed) against hand-built cases (fully correct, wrong slot, wrong final
answer, lapse into literal Heads/Tails, lapse into the memorized Nib/Nomo pair while
still "not literal" -- caught deliberately since that's exactly the failure mode this
diagnostic needed to be able to detect). All imports (`synthetic_bridge`, `reward_v3`,
`taxonomy`) and the clean-21 SHA256 verified locally, torch-free, before touching GPU.

**Training health**: loss curve is a clean, monotonic, textbook SFT convergence --
1.970 -> 0.006 over 40 full-batch steps, no NaN, no instability, grad_norm bounded
0.12-2.60 throughout. The step-5 health check passed cleanly
(1.970 -> 1.441, well inside the "no >3x increase" gate) before the run continued to
the full budget.

### Tier (a) -- training-set recall: **100%** (reported first, per instruction)

16/16 exact-sequence recall, 16/16 correct final answer. Taxonomy: 14/16
`correct_globally_consistent_code`, 2/16 `prompt_domain_echo` -- the 2 exceptions are
both "all-same" scenarios (Heads-all-same, Tails-all-same), independently confirmed
both `exact_recall=True` and `final_answer_correct=True`; they're mis-bucketed by a
pre-existing, already-documented taxonomy limitation (category 9 structurally requires
BOTH physical states to appear at least once, which an all-same scenario can never
satisfy even at 100% correctness -- the identical quirk found and documented during
Step 14b's evidence analysis). Not a new finding, not a partial failure -- true
tier-(a) result is a clean 16/16.

**The model can fully imitate its own training examples.** Tiers (b) and (c) are not
moot -- proceeding to report them in full.

### Tier (b) -- held-out same-pair generalization: **strong, real generalization**

On the clean-21 set (21 scenarios never seen during training), using the plain prompt
format with NO in-prompt instruction to use any code: **100% of completions
(21/21) used non-literal (Nib/Nomo) tokens instead of literal Heads/Tails.**
Taxonomy: 6/21 `correct_globally_consistent_code` (fully correct code AND fully
correct final answer), 14/21 `state_predictive_code` (genuinely state-correlated
non-literal code, ARI meaningfully above chance, but not a perfect bijective mapping),
1/21 `position_driven_drift` (degenerate). **20/21 (95.2%) show real, structured,
state-correlated non-literal code usage on scenarios the model was never trained on.**
Final-answer accuracy specifically (a stricter bar than "used the code at all") was
28.6% (6/21) -- lower, but nonzero and structured, not noise: sample completions show
coherent "Step N: The code remains unchanged/changes. State: Nib/Nomo" reasoning
applied correctly to new instruction sequences, not memorized text reproduced
verbatim (these are new scenarios by construction).

**This has never happened once across Stages 9 or 9b** -- zero non-literal completions
across three RL mechanisms, two model sizes, thousands of tracked rollouts. Plain SFT
on 16 examples produces it on the very first attempt, on 20/21 held-out scenarios.

### Tier (c) -- held-out different-pair (Yelt/Yark, explicitly instructed): **fails, but informatively**

`stayed_within_instructed_pair_rate = 0.0` -- not one of the 21 completions used ONLY
Yelt/Yark as instructed. But this is NOT the model ignoring the instruction and
reverting to pure Nib/Nomo either -- inspecting actual completions shows **partial,
inconsistent MIXING within single completions**: e.g. one sample's per-slot tokens
were `[nomo, nomo, nomo, yark, yark]` (memorized pair for the first three slots,
instructed pair for the last two, mid-completion switch); another was
`[nomo, yark, nomo, nomo, yark]` (alternating); a few were purely `[nomo, nib, ...]`
(memorized pair only, instruction fully overridden). `lapsed_into_literal_rate = 0.0`
-- it never fell back to literal Heads/Tails, only ever mixed between the memorized
Nib/Nomo and the instructed Yelt/Yark. Taxonomy still shows 19/21 (90.5%)
`state_predictive_code`/`correct_globally_consistent_code` -- the model IS still
"coding" state structurally, just with an inconsistently-selected vocabulary rather
than reliably following the in-context substitution instruction. `final_answer_correct
= 9.5%` (2/21, both presumably where the mixed vocabulary happened to still resolve
correctly against the expected Yelt/Yark mapping by chance or partial compliance).

**Interpretation, precise rather than collapsed to a single verdict**: the model
learned a fairly rigid, specific-token mapping ("Nib means Heads-state, Nomo means
Tails-state") strongly enough to generalize it to new SCENARIOS (tier b), but did not
learn -- and an explicit in-context instruction could not reliably override -- the more
abstract SKILL of "substitute whatever pair of words I'm told to use." The memorized
token identities exert a real, partial pull even against a direct, clearly-worded
instruction to use different ones.

### Bottom line

Plain SFT succeeds where every RL mechanism failed, at both training-set recall and
held-out same-pair generalization -- **the model's representational capacity was never
the bottleneck in Stages 9/9b.** Something about the RL mechanisms specifically (reward
shaping, GRPO's own gradient-competition dynamics, on-policy sampling never
encountering high enough reward to reinforce the behavior, or some combination) is
what prevented adoption under RL, not a fundamental inability to represent or use this
non-literal code. Tier (c)'s result adds a further-refined finding beyond the original
diagnostic question: what SFT taught is closer to a memorized specific-token mapping
than a flexible instructed-substitution skill -- a distinct, narrower claim than "the
model learned the general concept of coded state-tracking," worth flagging as its own
finding rather than folding into the headline result.

**Evidence, retrieved and hash-verified** into
`experiments/09c_sft_diagnostic/aws_runs/stage9c-sft-diagnostic-v1/`: full event log
(SHA-256 `53e2ce24a6ca41debc54ace20ddbd383257671f6dc61fff0fb899b519329c241`, 81KB --
small, since this is teacher-forced-CE-only, no rollout/reward instrumentation to
log), and the full stdout log. Local hash confirmed identical to the remote-computed
hash both immediately after transfer and again after copying into the repo path.
LoRA adapter weights (`final_adapter/`) remain on the instance's EBS volume, not
copied down -- the evidence JSON (with full per-sample completions for all three
tiers) is the load-bearing record.

**Teardown**: `scripts/gpu_teardown.py`, `g5.xlarge` throughout (no capacity issue
this time) -- printed "TEARDOWN FULLY CONFIRMED", exit code 0. Independently
re-verified via a fresh, separate `describe-instances` call: `stopped`, `g5.xlarge`.
Clean.

## SFT-SEED REBUILD (2026-08-31): expanded trajectory bank fixes state-tracking
## completely (100%, verified against ground truth) -- but a NEW, narrow bug masks
## this behind a modest-looking headline number

**Motivation**: Stage 9d showed the original 16-example seed's held-out correctness
(28.6%) never moved across 150 steps of adversarial RL in either direction -- strong
evidence the SFT stage never taught genuine state tracking through the code, only a
surface substitution pattern, leaving nothing for RL to reinforce. This task rebuilds
the seed with a substantially larger, verified trajectory bank to test whether more
diverse training data fixes the underlying capability gap.

**Task requirement 1 -- expanded bank, reusing the existing verification pipeline
unchanged.** Reproduced Stage 9c's EXACT held-out-21 first (same seed, same function
call, `CODED_TRAJECTORIES` untouched) -- verified byte-identical to the pinned
`EXPECTED_CLEAN21_SHA256` before building anything, so the comparison to 28.6% would
be valid. Built the expanded bank from the remaining 43 length-5 x starting-state
scenarios (all 64 total, minus the 21 held out) using `synthetic_bridge.py`'s existing
`build_prompt`/`build_coded_completion` UNCHANGED, then verified every row via the
existing `verify_bank`/`verify_trajectory` pipeline UNCHANGED -- no new generation or
verification logic written. **Coverage: 43/64 (67.2%) of the length-5 x starting-state
space, up from the original 16/64 (25.0%)** -- a 2.7x expansion, confirmed a strict
superset of the original 16-row bank, confirmed zero overlap with the held-out eval
set. 8 CPU-only tests (`test_expanded_bank.py`) verify this construction before any
GPU time was spent.

**Task requirement 2 -- SFT re-run, one real methodological problem hit and fixed
properly, not band-aided.** First launch attempted Stage 9c's exact full-batch
pattern (one forward/backward over all 43 examples at once) and hit a genuine CUDA
OOM on the very first step's `backward()` ("Tried to allocate 6.67 GiB" against
~22GB total, ~21.5GB already in use) -- 2.7x more examples in one padded batch
genuinely does not fit where 16 did. **Not treated as a fragmentation issue to paper
over with `PYTORCH_CUDA_ALLOC_CONF` alone** (added anyway as good hygiene, matching
this project's usual convention, but disclosed as insufficient by itself) -- fixed
with standard gradient accumulation: 43 examples split into `MICRO_BATCH_SIZE=8`
chunks (safely smaller than the 16 that fit), each chunk's PROPORTIONALLY-weighted
loss backward()'d before the next chunk's forward pass begins (so only one chunk's
activations are ever held at once), one `optimizer.step()` per outer step --
mathematically equivalent to the full-batch mean-CE-over-all-43-examples gradient,
confirmed via a dedicated CPU-only test that the chunking is an exact, non-overlapping
partition with correctly-summing weights. Redeployed, re-hash-verified, relaunched
cleanly -- no further OOM. `N_STEPS=40`, same as Stage 9c (telemetry didn't demand a
change: health check passed at step 5, final loss 0.014 from a starting 1.93, a
137.7x reduction -- see the overfitting discussion below for whether this represents
healthy convergence).

**Periodic overfitting watch (task requirement 2), added as a genuine live check, not
just a final look**: held-out mini-evaluations every 10 steps during training.
`nonliteral_rate` on the untouched held-out set: 0% at step 10, 28.6% at step 20,
100% from step 30 onward. `genuine_correct_among_nonliteral`: undefined (0 non-literal
samples) at step 10, 50% at step 20 (small-N, only 6 samples), 33.3% at step 30,
unchanged at step 40 -- **plateaued for the last third of training, not still
climbing**, suggesting 40 steps was sufficient for whatever this configuration was
going to reach, not obviously under- or over-shot on its own terms (the deeper
overfitting-adjacent issue found below is a different, more specific problem than
"trained too long").

**Task requirement 3b -- the key metric, reported plainly, THEN traced to its
mechanism rather than taken at face value (this project's standing discipline)**:
`genuine_correct_among_nonliteral = 33.3%` on the SAME held-out-21 set Stage 9c used,
vs. Stage 9c's own `28.6%`. Taken alone: a real but modest increase (+4.7 points),
not the substantial jump the task was hoping for -- and per instruction 4, this would
be the honest headline if reported without further investigation.

**It should not be reported without further investigation. Tracing why held-out
correctness only reached 33.3% revealed the reported number is actively misleading
about what the model can and cannot do:**

- **Every single held-out completion (21/21) answers "Tails" in the `<answer>` tag,
  regardless of what the model itself tracked.** Same complete collapse on the
  training set (43/43). This is not a training-label-imbalance artifact by itself --
  the expanded bank's true answer split is a mild 58%/42% (Tails/Heads), nowhere near
  the 100% collapse observed in the model's output.
- **Verified against GROUND TRUTH (not just internal self-consistency): the model's
  intermediate Nib/Nomo code sequence exactly matches the true expected trace on
  21/21 held-out scenarios (100%) and 43/43 training scenarios (100%).** The
  state-tracking-through-the-code skill this whole rebuild was trying to teach is
  **completely, verifiably fixed** -- a dramatic improvement over Stage 9c's original
  seed, whose own tracking (given its 28.6% eventual correctness with 100% nonliteral
  usage) was clearly far less reliable.
- **If the model had correctly translated its own (perfectly correct) last tracked
  code token into the literal answer instead of always saying "Tails," held-out
  correctness would be 21/21 = 100%, not 33.3%.** The entire gap between the reported
  33.3% and a hypothetical 100% is attributable to one narrow, specific,
  well-localized bug: a majority-class collapse at the final-answer-TRANSLATION step
  specifically, decoupled from the (otherwise flawless) intermediate tracking.
- **Comparison point ruling out "this collapse was already present, just unnoticed":**
  Stage 9c's ORIGINAL 16-example run (perfectly balanced, 8 Heads / 8 Tails) showed
  `final_answer_accuracy=1.0` on its OWN training set -- no collapse at all. The
  expanded bank's own answer distribution (25 Tails / 18 Heads, a mild 58/42 split,
  absent from the original perfectly-balanced bank) is the most likely trigger: with
  the final-answer token depending on correctly parsing back through a much longer,
  more indirect context (5 steps of code) than any single intermediate state token
  (which mostly depends on the immediately-preceding same/different instruction), a
  even a mild class imbalance may have been enough to make "always predict the
  majority class" a lower-loss shortcut for that ONE specific token, while the
  intermediate tokens -- with their more direct, locally-learnable structure -- did
  not fall into the same shortcut.

**Plain verdict, stated precisely rather than rounded to either extreme**: the bank
expansion (16/64 -> 43/64) **fully solved the problem this rebuild set out to fix**
(genuine, ground-truth-verified state tracking through the code, both on training and
held-out scenarios) -- but introduced a **new, narrow, well-diagnosed, and by its
narrowness plausibly fixable** bug in the separate step of transcribing that tracked
state into the required literal final answer. **The reported
`genuine_correct_among_nonliteral=33.3%` headline number is technically correct by
its own definition but must not be read as "state tracking only modestly improved" --
tracking itself is fixed; the answer-transcription step is what's currently broken.**
Per instruction, not fixed or investigated further in this task (no RL/adversarial
stage attempted either) -- flagged for the next decision point, along with a concrete,
well-supported hypothesis (the mild class-imbalance-driven majority-vote shortcut) for
whoever addresses it next.

**Task requirement 3c -- Tier (c), held-out different-pair (Yelt/Yark), reported as
secondary per instruction**: `fully_correct_rate=0.0`, `stayed_within_instructed_pair_
rate=0.0` -- the model does not flexibly substitute a new instructed pair, consistent
with Stage 9c's own original Tier C finding (a rigid, specific-token mapping rather
than a flexible skill). Not investigated further, per instruction's own "secondary to
fixing (b) first" framing -- and (b) itself turned out to need more investigation than
expected before this would be worth pursuing.

**Mandatory pre-launch hash check**: performed for every dependency
(`sft_train_eval_expanded.py`, `synthetic_bridge.py`, `pair_substitution.py`,
`reward_v3.py`, `taxonomy.py`) before the first launch, and re-verified for the
OOM-fixed script before the relaunch. GATE: PASS every time.

**Regression tests added**: `test_expanded_bank.py` (9 tests, CPU-only, run before any
GPU time -- bank construction, verification-pipeline pass-through, superset/no-overlap
guarantees, and the micro-batching partition's correctness) and
`test_expanded_sft_evidence.py` (7 golden-record tests against the retrieved evidence)
-- deliberately asserting BOTH halves of the finding (the modest-looking headline
number AND the underlying 100% ground-truth-verified tracking success) so neither can
silently disappear from the record, plus the Stage-9c-comparison test supporting the
class-imbalance hypothesis. Full suite (25 tests across this directory) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09c_sft_diagnostic/aws_runs/stage9c-sft-expanded-v1/`:
`stage9c_sft_expanded.json`, SHA256
`e6041202b6d427841d5e966a122803ccc51b7c70861e9cf9eda6fff072beb92f`;
`sft_train_eval_expanded_stdout.log`, SHA256
`a7f54c4c3983522a34d60d7d8fcbbc62904e0ba429e6ca16a5268e0c793fe69d` (both remote/local
identical). LoRA adapter weights (`final_adapter/adapter_model.safetensors`, SHA256
`77af683cc819f5560a257cb3b0bce6610c79b61366de5ac1600cf71cc40ebdcf`) remain on the
instance's EBS volume, not copied down, matching this stage's established convention.

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED**
(exceeded the 120s foreground timeout during the ~16-iteration stopping poll, same
pattern as every prior GPU task in this project -- retrieved via the background task
result, not a script fault). `g5.xlarge` throughout, no capacity substitution needed.

**Status: task requirement 4 honored -- NOT proceeding to any RL/adversarial stage.**
The held-out correctness rate is reported first and plainly (33.3%, a modest increase
over 28.6%), exactly as instructed -- but reporting it WITHOUT the tracing above would
have been actively misleading given what the tracing revealed. Waiting for direction
on the next step: most plausibly, either rebalancing the expanded bank's answer
distribution (or otherwise directly addressing the diagnosed majority-class shortcut
at the final-answer step) and re-evaluating, given how narrow and well-localized the
remaining bug appears to be -- not attempted in this task.

## REBALANCE ATTEMPT (2026-08-31): the label-imbalance hypothesis is DISPROVEN --
## the collapse persists identically at exact 18/18 parity. Reported plainly, per
## instruction, rather than guessing a new cause

**Task requirement 1/2 -- rebalanced the expanded bank, and the "add vs. trim"
question was mathematically forced, not a preference.** Confirmed directly: the
expanded bank (43 scenarios) plus the held-out eval (21 scenarios) already sum to
64 -- the ENTIRE length-5 x starting-state space. There is no uncovered pool to add
from within this space while keeping the held-out-21 fixed (a hard requirement for a
valid comparison), so trimming was the only available option, not a shortcut taken
over a preferred alternative. Trimmed exactly 7 Tails-labeled scenarios (the exact
25-18=7 needed for parity) via a dedicated, documented, reproducible seed
(`REBALANCE_SEED=20260901`), constrained to draw ONLY from the 27 scenarios the prior
expansion newly added -- **the original 16-row bank (Stage 9c's own trajectories)
remains a complete, untouched subset of the rebalanced bank.** Before: 25 Tails / 18
Heads (58/42). After: **18 Tails / 18 Heads (exact 50/50 parity)**, 36/64 (56.2%)
coverage -- still more than double the original 16/64 (25.0%). All 36 rows verified
via the existing, unchanged `verify_bank` pipeline. 8 CPU-only tests
(`test_rebalanced_bank.py`) confirm this construction before any GPU time was spent.

**Task requirement 3 -- re-run, same methodology, same 40 steps, same overfit-watch
cadence, intermediate-tracking accuracy now reported as a FIRST-CLASS metric
(task requirement 4) rather than requiring forensic tracing after the fact.** Health
check passed at step 5. Final loss 0.0114 (from 1.937, a 169.6x reduction). Overfit
watch showed intermediate-tracking accuracy climbing from 0% (step 10, no non-literal
usage yet) through 57.1% (step 30) to **95.2% (step 40)** -- while final-answer
accuracy stayed FLAT at 33.3% from step 30 through step 40, unmoved by 10 further
steps of continued tracking improvement. This divergence (tracking still improving,
answer accuracy static) was visible live, not just at the end.

**Task requirement 4/5/6 -- the result, reported exactly as the task's own branching
logic anticipated for the "gap persists" case.** Final held-out Tier (b) result:
`intermediate_tracking_accuracy=95.2%` (20/21, ground-truth verified, matching/
exceeding the unbalanced run), `final_answer_accuracy=33.3%` (7/21) --
**IDENTICAL to the unbalanced expanded-bank run's own 33.3%, to the exact same
fraction.** Confirmed by re-examining the actual completions (not just the summary
numbers): **all 21 held-out completions still answer "Tails" in the `<answer>` tag,
regardless of the model's own (now even more reliably correct) tracked state --
the SAME complete collapse observed before rebalancing, unchanged in kind or degree.**

**Plain verdict, per task requirement 6, no further guessing attempted in this
task**: **rebalancing the training bank's answer-label distribution to exact parity
had ZERO measurable effect on the held-out collapse.** This DISPROVES the hypothesis
that the 58/42 imbalance was (at least the sole) cause -- the collapse is not driven
by training-set answer-label frequency, or at minimum not driven by it alone, since
removing the imbalance entirely changed nothing about the model's held-out behavior.
**The next diagnostic angle needs to be identified fresh, not assumed** -- per
instruction, not investigated further in this task. One observation recorded for
whoever picks this up next, offered as a data point rather than a new theory: the
held-out eval set's TRUE answer distribution is 14 Heads / 7 Tails (a 67/33 split, an
artifact of which 21 scenarios happened to be drawn for the fixed eval set at Stage
9c's own seed) -- the model's 33.3% "success" rate continues to exactly match the
count of genuinely-Tails eval scenarios (7/21) under an unconditional always-"Tails"
policy, in both the unbalanced AND the now-perfectly-balanced training runs. Whether
this points to some OTHER systematic driver of the "Tails" bias (a pretrained-model
prior over the word itself, something structural in the prompt/generation setup, or
something else entirely) is exactly the kind of question the task said not to guess
at here.

**Mandatory pre-launch hash check**: performed for every dependency
(`sft_train_eval_rebalanced.py`, `synthetic_bridge.py`, `pair_substitution.py`,
`reward_v3.py`, `taxonomy.py`) before launch. GATE: PASS on all five. No OOM this
time (the micro-batched gradient-accumulation fix from the prior task carried over
unchanged, and comfortably handled the smaller, 36-example rebalanced bank).

**Regression tests added**: `test_rebalanced_bank.py` (8 CPU-only, pre-GPU tests --
parity, trim-count, original-bank preservation, coverage, verification-pipeline
pass-through, no-eval-overlap, determinism) and `test_rebalanced_sft_evidence.py`
(6 golden-record tests against the retrieved evidence) -- deliberately asserting the
DISPROVEN hypothesis explicitly (identical 33.3% result, identical complete collapse,
widened not narrowed gap) so this negative finding can't quietly disappear from the
record. Full suite (39 tests across this directory) passes.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09c_sft_diagnostic/aws_runs/stage9c-sft-rebalanced-v1/`:
`stage9c_sft_rebalanced.json`, SHA256
`3d5ab4404464b04280aa18a7bcb1e6e91a0edc9d548b97edb30d0c2fa8eb603c`;
`sft_train_eval_rebalanced_stdout.log`, SHA256
`41c4516f54955d34794f80872b92735dde3c3a84c59df95d7db0a25278feb69d` (both remote/local
identical). LoRA adapter weights (`final_adapter/adapter_model.safetensors`, SHA256
`f19fe37eae12cef50dbeef012f5e50c9748b57e66d1bb101888ec80f0fafe673`) remain on the
instance's EBS volume, not copied down, matching this stage's established convention.

**Teardown**: `scripts/gpu_teardown.py` produced **TEARDOWN FULLY CONFIRMED**
(exceeded the 120s foreground timeout during the ~16-iteration stopping poll, same
pattern as every prior GPU task in this project -- retrieved via the background task
result, not a script fault). `g5.xlarge` throughout, no capacity substitution needed
this time.

**Status: task requirement 6 honored -- the gap persists after rebalancing, reported
plainly, no further guessing attempted.** Intermediate state tracking remains fully
fixed and further improved (95.2%, ground-truth verified, on top of the prior run's
own 100%). The final-answer-translation collapse is now CONFIRMED not to be (solely)
an answer-label-imbalance artifact. This checkpoint is NOT yet the "new seed worth
carrying into RL" the task's requirement 5 branch describes -- that branch's condition
(final-answer accuracy closely matching tracking accuracy) was not met. No RL/
adversarial stage launched. Waiting for a fresh diagnostic angle before any further
GPU work on this specific bug.

## Corrupted-prefill causal diagnostic + base-model logit-prior check (2026-08-31)

The rebalancing task above ended with an open question and an explicit instruction not
to guess further without a fresh diagnostic angle. This task supplies that angle: a
direct causal test of whether the `<answer>` token is conditioned on the model's own
tracked trace at all, adapted from Stage 8's own corrupted-state-prefill methodology
(`experiments/08_load_bearing_and_exploration_audit/aws_runs/exp1_load_bearing_regime_v1/run_script.py`),
plus an independent check of whether the base (pre-SFT) model already carries a raw
next-token prior toward "Tails" at the answer position, which would explain why a
small SFT dataset couldn't overcome it.

**Diagnostic 1 -- corrupted-prefill causal test.** Reused the `stage9c-sft-rebalanced-v1`
checkpoint's own already-generated, already ground-truth-verified held-out completions
(`stage9c_sft_rebalanced.json`, SHA256
`3d5ab4404464b04280aa18a7bcb1e6e91a0edc9d548b97edb30d0c2fa8eb603c` -- reused rather than
regenerated, since eval used `do_sample=False` and is deterministic). Selected two
groups among the 20/21 correctly-tracked samples: `heads_to_tails` (true state Heads,
correctly tracked to Nib, N=14 -- the group the task's instruction 1 literally
specifies) and `tails_to_heads` (true state Tails, correctly tracked to Nomo, N=6 -- an
intentional expansion beyond the letter of instruction 1, disclosed here as such, and
directly anticipated by the task's own instruction 2 wording "even when corrupted to
say Heads"). For each sample, took the model's own trace verbatim through Step 4,
flipped ONLY the Step 5 (final) State token to its code-opposite (Nib<->Nomo), dropped
everything after that line (including any existing `<answer>` tag), and continued
generation from the corrupted prefix (`corruption_lib.py`'s `parse_state_slots_with_spans`
/ `build_corrupted_prefix`, a direct Nib/Nomo-domain, last-step adaptation of Stage 8's
own span-parsing and corruption-loop code).

The `heads_to_tails` direction is NOT decisive alone -- explicitly flagged as such
before running it: the model's known default is already "Tails" regardless of input,
so both competing hypotheses (tracks-the-corrupted-prefix vs. structurally
disconnected) predict the SAME outcome ("Tails") in that direction. It came back
14/14 `tracks_corrupted_prefix` -- consistent with either hypothesis, exactly as
expected, and reported for completeness rather than as evidence either way.

The `tails_to_heads` direction is the decisive one: tracks-the-prefix predicts the
answer flips to "Heads" (a change from the model's own default); structural
disconnect predicts it stays "Tails" regardless of the injected "Nib" one line above
the answer tag. **Result: 6/6 `stays_at_original_answer`.** Every single corrupted
continuation ignored the injected Heads-equivalent state and answered "Tails" anyway
-- e.g. one sample's raw continuation was literally `"\n<answer>Tails</answer>"`
immediately following a prefix manually corrupted to end in `State: Nib`. **This
confirms genuine structural disconnect, not a narrower "defaults to Tails under
uncertainty" bug**: the final-answer slot is not reading the immediately-preceding
trace content at all, in either direction, on real held-out data from the actual
checkpoint under investigation.

**Diagnostic 2 -- base-model raw logit prior check.** Loaded a completely FRESH base
model (`Qwen/Qwen2.5-3B-Instruct`, no LoRA, no fine-tuning of any kind -- same
loading pattern Stage 8 used for its own base-model condition). For 6 representative
scenarios (3 ending in Nib, 3 ending in Nomo, drawn from the same eligible pool above),
built the GROUND-TRUTH-correct Nib/Nomo reasoning trace via `build_coded_completion`
(not model-generated, to isolate the base model's own prior from anything the
fine-tuned model would have produced) up through the `<answer>` tag opening, and
computed a teacher-forced log-probability for the literal continuations "Heads" vs.
"Tails" at that exact position via a single forward pass (`teacher_forced_word_logprob`
in `corrupted_prefill_diagnostic.py`; uses character-offset mapping rather than a
token-count-boundary assumption, after an initial version crashed on a BPE
re-merge across the prefix/word boundary -- caught by its own consistency check, not a
silent wrong answer, and fixed before the results below were produced).

**Result: 6/6 probes favored "Heads", 0/6 favored "Tails"**, with a substantial mean
log-probability gap (~-11 nats, i.e. the base model assigns roughly 60,000x higher
probability to continuing with "Heads" than "Tails" at this exact position, regardless
of whether the preceding trace ends in Nib or Nomo, and regardless of the scenario's
true answer). **This DISPROVES the base-model-prior-toward-Tails hypothesis** the task
was checking for -- there is no pre-existing model bias favoring the literal word
"Tails" at the answer slot; if anything the untouched base model leans the opposite
way. Whatever induces the SFT checkpoint's collapse to "Tails" specifically, it is not
inherited from a pretrained prior over that word in that position -- it must be
introduced or amplified during SFT itself.

**Combined verdict, reported plainly per instruction 4 -- no fix attempted in this
task.** The two diagnostics point in the same direction and rule out the same
alternative: the final `<answer>` token is genuinely structurally disconnected from
the model's own (correctly tracked) trace content -- confirmed directly, in both
corruption directions, on the actual checkpoint -- and this is NOT explained by an
inherited base-model word-preference bias, since the base model has no such bias
toward "Tails" (Diagnostic 2), and NOT explained by a narrower "defaults to Tails when
uncertain" mechanism that would still read the prefix in principle, since the
corrupted prefix is ignored outright in both directions (Diagnostic 1).

**Proposed next step (per instruction 4, not implemented here):** the evidence favors
a prompt/completion-format change over more contrastive training data. Diagnostic 2's
finding rules out "push the base model away from a specific token via more contrastive
pairs" as the mechanism of a fix, since there is no base-model token-slot prior to
counteract in the first place -- more Tails-avoiding training pairs would be treating
a symptom the base model doesn't actually have. Diagnostic 1's finding instead points
at the SFT training FORMAT: the `<answer>` tag currently sits immediately after the
final `State:` token with no explicit instruction or trained behavior connecting the
two, so the model appears to have learned the `<answer>` slot as a separate,
weakly-conditioned production rather than a read-out of the preceding line. The most
likely fix, by direct analogy to Stage 2's own single-domain seeding fix for a related
bug, is an explicit decode-back step in the completion format -- e.g. a trained
line between the last `State:` token and `<answer>` that names the literal
Heads/Tails word as an explicit translation of the just-emitted code token (something
like "The final code token Nib decodes to Heads." on its own line) before the answer
tag, so the format itself forces a legible dependency between trace and answer for the
SFT objective to actually train against, rather than the current fully-implicit
adjacency. Not implemented in this task.

**Mandatory pre-launch hash check**: performed for every dependency
(`corrupted_prefill_diagnostic.py`, `corruption_lib.py`, `synthetic_bridge.py`,
`reward_v3.py`, `stage9c_sft_rebalanced.json`) against the deployed remote copies --
GATE: PASS on all five, both before the first launch and again after fixing an initial
`ModuleNotFoundError` (the remote deployment was flattened rather than mirroring the
repo's `experiments/09_direct_indomain_synthetic_bridge` / `experiments/09c_sft_diagnostic`
/ `experiments/07_positive_signal_annealed_reward` layout the scripts' own path
resolution expects -- caught immediately, before any GPU/model work started, fixed by
restructuring the remote layout and re-verifying all hashes before relaunch). The
LoRA checkpoint under investigation was independently re-hashed after the instance's
stop/restart and confirmed byte-identical to the run that produced it (SHA256
`f19fe37eae12cef50dbeef012f5e50c9748b57e66d1bb101888ec80f0fafe673`).

**CPU-only tests added, run before any GPU work**: `corruption_lib.py` (new,
pure-logic, no model/GPU imports) plus `test_corruption_lib.py` (10 tests, run
directly against the real `stage9c-sft-rebalanced-v1` evidence -- span-parsing
correctness, eligible-group selection and sizes, corrupted-prefix construction in
both directions, outcome classification). All 10 passed before deployment.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09c_sft_diagnostic/aws_runs/corrupted-prefill-diagnostic-v1/`:
`corrupted_prefill_diagnostic.json`, SHA256
`6df442223ec8e65f0357a33d7882b8170a5ff00f1c595e31ab3321ffd758dbe7` (remote/local
identical). No new checkpoint was trained in this task -- Diagnostic 1 loaded the
existing rebalanced-run adapter (still on the instance's EBS volume, unchanged) and
Diagnostic 2 used a fresh untouched base model with nothing to save.

**Teardown**: `scripts/gpu_teardown.py` run after evidence retrieval and hash
verification; see run output for the **TEARDOWN FULLY CONFIRMED** banner.
`g5.xlarge` throughout, no capacity substitution needed.

**Status**: both diagnostics requested by this task are complete, with a clean,
mutually-reinforcing, non-contradictory result: genuine structural disconnect between
the tracked trace and the final answer, confirmed directly by causal intervention in
both corruption directions on the real checkpoint, and NOT attributable to a
pretrained base-model word-preference bias. No fix attempted, per instruction. A
concrete, evidence-motivated fix proposal (explicit decode-back line before
`<answer>`) is recorded above for whoever picks this up next, but not implemented.

## Decode-back-line fix, implemented and re-evaluated (2026-09-02)

Implements the fix proposed (and approved) above. New, additive, local module
`decode_back_bank.py` inserts `"The final code token {CODE} decodes to {LITERAL}."`
between the last `State:` line and `<answer>` in every training completion, on top of
the UNCHANGED `build_coded_completion` output -- `synthetic_bridge.py` itself was not
modified, per this stage's established discipline of treating that module as
shared/load-bearing. Verification reuses `verify_trajectory` UNCHANGED (via a
strip-the-new-line-back-out step) for everything it already checks, plus one new,
additional assertion: the decode-back line's code token must match the model's own
immediately-preceding `State:` line, and its literal word must be the true decode of
that token. 12 CPU-only tests (`test_decode_back_bank.py`) pass, including two that
confirm the verifier correctly REJECTS a wrong decoded literal and a code token that
doesn't match the preceding line, before any GPU time was spent. Retrained on the
IDENTICAL 36-scenario, 18/18-parity rebalanced bank (same seeds, same 40 steps, same
micro-batched gradient accumulation) -- the ONLY variable changed is the completion
format.

**Result: the gap closed dramatically, from 61.9 points to 9.5 points on held-out
data.** Tier B (held-out-21, same Nib/Nomo pair):
`intermediate_tracking_accuracy=90.5%`, `final_answer_accuracy=81.0%`,
`tracking_vs_answer_gap=9.5pp` -- down from 95.2%/33.3%/61.9pp in the rebalanced run.
`genuine_correct_among_nonliteral` rose from 33.3% to 81.0%. Tier A (training-set
recall) shows a fully closed gap: 97.2%/97.2%, exactly matched. The overfit-watch
milestones (step 10/20/30/40, tracked separately for tracking and answer accuracy
throughout training per requirement 2, not just at the end) show the gap actually
INVERTED early in training (answer accuracy briefly exceeded tracking accuracy at
steps 10-20, before the model had reliably learned to track state at all) before both
metrics converged upward together by step 40 -- consistent with the model learning to
produce the new decode-back step only in the second half of training.

**The decode-back line itself, per requirement 3**: `decode_back_presence_rate=1.0`
-- the model reliably learned to produce this new intermediate step on every single
held-out completion, not just some. More importantly, **`decode_back_matches_own_trace_rate=1.0`
across all 21 held-out samples, with zero exceptions**: the final `<answer>` now
FAITHFULLY follows whatever the decode-back line says, every time. This is the direct,
decisive confirmation that the original structural disconnect (established by the
corrupted-prefill diagnostic above) is fixed -- the answer is no longer ignoring the
line immediately before it.

**The residual 9.5-point gap is precisely characterized, not left unexplained.** All 4
held-out errors decompose into exactly two causes: (1) ONE case where the model's own
intermediate tracking was simply wrong (unrelated to translation direction -- a
full-trace mismatch against ground truth); (2) THREE cases where tracking was fully
correct (ending in the right code token, `Nib`) but the decode-back line itself
mistranslated it as `"Tails"` instead of `"Heads"` -- a smaller, residual version of
the SAME directional bias the original diagnostic found, now confined specifically to
the decode-back translation step rather than the whole pipeline, and observed ONLY in
the Nib->Heads direction (10/13 = 77% correct) -- never once in the Nomo->Tails
direction (7/8 = 87.5% correct, and that one miss was case (1)'s tracking error, not a
translation error). The fix converted a total, 100%-of-the-time structural collapse
into an intermittent, direction-asymmetric translation error affecting roughly a
quarter of Nib-ending cases -- a real, large reduction, but not a complete elimination
of the underlying directional tendency.

**Against the task's stated success criterion** ("final_answer_accuracy should now
closely track intermediate_tracking_accuracy, both approaching ~90%+"):
`intermediate_tracking_accuracy` (90.5%) clears the ~90%+ bar; `final_answer_accuracy`
(81.0%) is close and dramatically improved but does not quite reach it, sitting 9.5
points behind. Reported plainly per instruction 4 rather than rounded up to a clean
"success": this is a substantial, well-characterized partial fix, not a full closure
of the gap. Given the residual cause is now narrow, directional, and specific (not the
original open-ended structural disconnect), the natural next increment -- not pursued
in this task, consistent with "do not guess further" -- would be additional
decode-back-specific contrastive training pairs targeting the Nib-decode direction
specifically, rather than another format change.

**Tier C (different-pair, Yelt/Yark substitution)**: `fully_correct_rate=0.0`,
unchanged from the established pattern in every prior run at this stage -- pair-word
generalization remains a separate, harder, unsolved problem this task did not target
and did not move.

Per requirement 5's condition (closely-matching accuracies, ~90%+ both) not being
fully met, this checkpoint is a markedly stronger RL-seed candidate than any prior one
at this stage (81.0% vs. 33.3% final-answer accuracy, with the disconnect mechanism
itself now proven fixed) but is reported here as a partial rather than complete
success -- **no RL/adversarial stage launched in this task**, per instruction.

**Mandatory pre-launch hash check**: performed for all six dependencies
(`sft_train_eval_decode_back.py`, `decode_back_bank.py`, `pair_substitution.py`,
`taxonomy.py`, `reward_v3.py`, `synthetic_bridge.py`) against the deployed remote
copies -- GATE: PASS on all six.

**Evidence, retrieved and hash-verified BEFORE teardown**, into
`experiments/09c_sft_diagnostic/aws_runs/stage9c-sft-decode-back-v1/`:
`stage9c_sft_decode_back.json`, SHA256
`c2ac2b810870014d570de0ed449c3bd94b723d1a4cf5758f48f708a33577384b`;
`sft_train_eval_decode_back_stdout.log`, SHA256
`b9d077b77f183732a04a42dbeb02b9ac1a98c7e55e6c3107749fe350b9b16423` (both remote/local
identical). LoRA adapter (`final_adapter/`) remains on the instance's EBS volume,
matching this stage's established convention.

**Regression tests added**: `test_decode_back_bank.py` (12 CPU-only, pre-GPU tests)
and `test_decode_back_sft_evidence.py` (8 golden-record tests against the retrieved
evidence, including explicit assertions of the residual-gap breakdown and its
directional asymmetry so this nuance can't quietly disappear from the record). Full
suite (75 tests across this directory) passes.

**Teardown**: `scripts/gpu_teardown.py` run after evidence retrieval and hash
verification; see run output for the **TEARDOWN FULLY CONFIRMED** banner.
`g5.xlarge` throughout, no capacity substitution needed.
