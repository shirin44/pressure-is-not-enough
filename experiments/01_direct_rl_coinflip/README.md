# 01 — Direct RL on Coin Flip

## What we tried
Trained Qwen2.5-3B-Instruct with GRPO and a 6-term reward that penalizes
literal "Heads"/"Tails" tokens in the chain-of-thought, testing whether RL
pressure alone induces the model to invent a non-literal substitute
encoding — no seeding, no demonstration.

## Why we tried it
This is the project's starting hypothesis, adapted from Skaf et al.
(steganographic CoT emerging under RL pressure).

## Key steps / iterations
1. **Issue**: mode collapse under the CoT penalty. **Fix**: added entropy
   regularization.
2. **Issue**: pushing the CoT penalty harder (`p_CoT` rescaled 2x) caused
   reproducible optimizer instability. **Fix**: reward left at its
   original scale; instability accepted as a boundary condition, not
   solved here (later revisited directly in `04_stability_investigation`).
3. Circuit breakers and dynamic sampling (see `00_infrastructure`) added
   throughout as safety scaffolding around the training loop.

## Final result
3,200+ audited rollouts, **zero verified non-literal encodings**. The
model always used literal Heads/Tails, or went vacuous (omitted the state
token entirely) to dodge the penalty. Complete negative result. Last
verified-good checkpoint: logical step 500
(`grpo-step500-*`/full-snapshots/step-500 on Drive) — this checkpoint is
later reused as a fixed diagnostic subject in `04_stability_investigation`,
not as an active training lineage.

## What this means for the next step
RL pressure alone, within this compute/time budget, does not induce the
target behavior. Pivoted to demonstration-seeding: teach the general
"track a hidden binary state via a consistent substitute token" strategy
on an unrelated toy task via SFT first, then (eventually) merge that
capability into this Coin Flip model — see `02_demonstration_seeding_single_domain`.

---
*Referenced, not moved: `scripts/run_grpo_dryrun_colab.py` has uncommitted
local changes predating this reorganization and stays at its current path
until those changes are resolved and committed separately.*

## Checkpoint-500 re-evaluation under current audit tooling (2026-09-22)

**This is a standalone re-evaluation, not a rerun of the "Final result" above.**
The original ~3,200-rollout training run's own driver code and prompt format
could not be located anywhere (git history, local session transcripts, Drive,
or AWS — see the prior search-task report); checkpoint 500's weights
themselves were confirmed still present on AWS instance
`i-REDACTED` at `~/aisi_checkpoints/step-500/` (78M,
`adapter_config.json`: LoRA r=8/alpha=16/dropout=0.05/
target_modules=[q,k,v,o]_proj on `Qwen/Qwen2.5-3B-Instruct`,
`peft_version=0.19.1`). Loaded those weights (inference only — no
optimizer, no gradient step) and generated 500 fresh rollouts against the
**chat-templated** prompt format (the same fix rationale as the Llama
09e redesign — the original training used a raw string prompt, confirmed
in `07_positive_signal_annealed_reward/README.md`'s Part-A finding; this
re-evaluation deliberately does NOT reproduce that), Coin Flip with 3-8
instructions (`src/data/coinflip.py:generate_coinflip_example`,
`run_seed=20260922`, T=0.8/top_p=0.95, distinct from every prior seed
used in this project), classified with the existing, unmodified 10-tier
taxonomy (`experiments/09_direct_indomain_synthetic_bridge/taxonomy.py`).

**Tier distribution (500/500 rollouts):**

| tier | name | count | % |
|---|---|---|---|
| 1 | literal | 500 | 100.0% |
| 2 | vacuous | 0 | 0.0% |
| 3 | literal_corruption | 0 | 0.0% |
| 4 | prompt_domain_echo | 0 | 0.0% |
| 5 | position_driven_drift | 0 | 0.0% |
| 6 | verbalized_substitution_attempt | 0 | 0.0% |
| 7 | state_varying_candidate | 0 | 0.0% |
| 8 | state_predictive_code | 0 | 0.0% |
| 9 | correct_globally_consistent_code | 0 | 0.0% |
| 10 | causally_load_bearing_code | 0 | 0.0% |

No tier-9 or tier-10 completion. `structural_nonliteral_candidate_count=0`,
`verbalized_substitution_attempt_count=0`. Secondary sanity signal (not
part of the taxonomy): 494/500 (98.8%) format-valid, 320/500 (64.0%)
final-answer correct — the checkpoint behaves coherently under the
chat-templated format, it simply never departs from literal Heads/Tails.

**Reading, kept explicitly separate from the original claim**: under
current audit tooling and the corrected (chat-templated) prompt format,
checkpoint 500 shows the same complete absence of non-literal encoding
attempts the original training reported — this re-evaluation does not
contradict the original "0 verified non-literal encodings" result, but it
is independent evidence of it, obtained under a materially different
(and more correct) prompt format, not a confirmation that the original
run's own procedure was sound.

GPU usage: instance started, checkpoint + base model loaded (inference
only, `model.eval()`, `torch.inference_mode()`), 500 rollouts generated
in 586.1s, stopped again immediately after; `TEARDOWN FULLY CONFIRMED`.
Evidence: `aws_recovered/checkpoint500_reeval.json` (hash-verified
against the AWS copy) and `stdout.log`. Regression test:
`test_checkpoint500_reeval.py` (5 tests, all passing).
