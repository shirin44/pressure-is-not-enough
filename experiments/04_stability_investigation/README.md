# 04 — Checkpoint-500 KL-instability investigation

## What we tried
Diagnosed why resuming checkpoint 500 (the `01_direct_rl_coinflip`
lineage) for further GRPO training reproducibly destabilizes
(`grad_norm`/`kl` exploding) within a handful of optimizer steps.

## Why we tried it
Checkpoint 500 is the fixed artifact any eventual Stage 4 merge (seeded
capability → Coin Flip model) would need to build on. Before that could
even be planned, needed to confirm it could be safely resumed at all.

## Key steps / iterations
1. **Seed diversity** ruled out: two different seeds both blew up within
   3-4 steps regardless of LR schedule aggressiveness (`grad_norm` 1736 &
   151, `kl` 102 & 17.2).
2. **Entropy clamp + gentler warmup**: delayed but did not prevent it
   (step 7 vs. step 3-4) — entropy at the blowup step was unremarkable,
   arguing against entropy magnitude as the driver.
3. **Per-token/per-rollout instrumentation** (validated against TRL's own
   logged `kl` metric to ~7 significant figures): proved the PPO-style
   importance ratio is exactly 1.0 by mathematical construction in this
   config, not the driver; traced the real blowup to specific tokens where
   the policy drifts into meta-commentary/instructional-preamble content
   the reference model finds highly improbable. A recurrence-counting bug
   was caught and fixed along the way (`checkpoint500_kl_evidence_reader.ipynb`).
4. **Greedy-decoding check**: 0/200 flagged for the same pattern under
   deterministic decoding — points at temperature-sampling variance, not a
   latent defect baked into the checkpoint.
5. **Per-token KL clamp mitigation**: built, mechanism unit-verified in 4
   cases (pathological token bounded exactly, healthy tokens pass through
   with zero distortion, correct gradient behavior in both regions) — but
   **not yet executed**; two operational bugs (an IOPub output-flood kill,
   a missing GPU/version guard) caught and fixed first.

## Final result
Root mechanism identified: temperature sampling occasionally lands on a
token the reference model sharply disfavors, not a defect in the
checkpoint or an importance-ratio effect. A targeted mitigation is built
and verified at the mechanism level, but whether it actually reaches
50-step survival is **unknown — not yet run**.

## What this means for the next step
Run `checkpoint500_kl_clamp_mitigation.ipynb` once GPU access returns. If
it survives, checkpoint 500 becomes a viable Stage 4 merge target; if not,
lowering the rollout-sampling temperature is the next lever, not further
hyperparameter guessing.

---
**Note on this folder's notebook output cells**: five notebooks here
(`checkpoint500_seed_diversity_diagnostic`, `_entropy_clamp_diagnostic`,
`_per_token_kl_instrumentation`, `_kl_evidence_reader`,
`_greedy_preamble_check`) had their real execution results accidentally
overwritten by a rebuild-script verification pass during the 2026-08-22
repository reorganization, and have been conservatively reconstructed from
the conversation record (see the notice at the top of each affected
notebook). The headline numbers in the table above are cross-checked
against `logs/development_log.md`, which was never affected by the
incident and remains the authoritative source for exact figures not
reproduced in the reconstructed notebooks. `checkpoint500_kl_clamp_mitigation.ipynb`
was never executed in the first place, so nothing was lost there.
