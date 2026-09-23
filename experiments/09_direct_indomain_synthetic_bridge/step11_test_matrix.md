# Step 11 — reconstructed 15-category unit/integration matrix

## Provenance

The original category list could not be recovered. The repository and every reachable
Git revision contain only `Full unit/integration test suite (15 categories specified)`.
The Stage 9 README says the details were in the design-audit conversation turn, but no
copy of that turn or a category enumeration is stored in the repository. This matrix is
therefore explicitly a **reconstruction from the concrete artifacts produced in Steps
1–10**, following the same disclosure discipline used for the reconstructed Step-0
soft-stop enforcement. It must not be cited as the verbatim original specification.

Counts below are collected pytest cases (parameterizations count separately). “Before”
means the 107-test suite after the 16-scenario bank expansion; “added” is Step 11.

| # | Reconstructed category and experiment-specific purpose | Before | Added | Final CPU result |
|---:|---|---:|---:|---:|
| 1 | **Exact Step-0 reward configuration.** Proves `p_cot`, `r_consistency`, and `r_signal` are zero throughout Step-0 while correctness/structure remain active and the 3.0/3.0 ordering margins hold. Concealment pressure here would invalidate the foundation control. | 5 | 0 | 5 pass |
| 2 | **Bridge reward ordering remains valid.** Proves genuine encoding outranks stuck, partial, vacuous, incorrect, and malformed behavior, and importing auxiliary CE does not mutate the 0.7/0.7 bridge reward invariant. | 5 | 0 | 5 pass |
| 3 | **Trajectory verification acceptance and rejection.** Valid targets must track all five physical states and answer correctly; corrupted targets must fail rather than teach a wrong code. Existing rejection covered a wrong state and literal lapse; Step 11 added wrong-final-answer and missing/out-of-order-slot cases. | 2 | 2 | 4 pass |
| 4 | **Coded/literal matching and bank diversity.** Proves all 16 controls are exact token-substitution matches, every sequence has both starts, and positional/Hamming/prefix diversity has the audited values. This removes the sequence/start confound. | 2 | 0 | 2 pass |
| 5 | **Fixed exposure quotas.** Proves the cumulative-floor formula for 0%, 1%, 5%, and 10%, including exact per-step patterns and totals at the 150-step horizon. Wrong integerization would collapse the intended dose ablation. | 5 | 0 | 5 pass |
| 6 | **Full-horizon annealing.** Proves all 150 exact-rational fractions have a constant decrement, endpoints are exactly 10% and 0%, all cumulative errors remain below one example, and total exposure is 60. | 2 | 0 | 2 pass |
| 7 | **Deterministic selection and expanded-bank reuse.** Proves bank growth changes reuse—not quotas—gives the exact 7/8-use split at 10%, wraps correctly, and consumes no Python RNG state that could perturb GRPO sampling. | 0 | 2 | 2 pass |
| 8 | **GRPO loss/advantage isolation.** Proves zero-weight auxiliary CE is bit-identical to absence and nonzero CE changes only the returned loss, never advantages or GRPO outputs. This is the central reason auxiliary CE was chosen over rollout injection. | 2 | 0 | 2 pass |
| 9 | **Optimizer-step/accumulation integration.** Proves CE is applied once per eligible optimizer step, not once per microbatch, and is scaled by eight-way accumulation exactly once. | 1 | 0 | 1 pass |
| 10 | **Teacher-forcing masks and numerical precision.** Proves prompt/padding labels are `-100` and completion labels remain trained. Step 12 subsequently closed the deferred real CUDA/bf16-model-path forward/backward and gradient-scale check using persisted evidence. | 0 | 2 | 2 pass after Step 12 |
| 11 | **Token-pair eligibility under milestone 12.** Proves summarization logic plus the persisted Nib/Nomo and Yelt/Yark results: exact within-pair Step-0 log-probability matches, edit distance >=3, leakage/static gates, and adapter SHA binding. | 1 | 1 | 2 pass |
| 12 | **Ten-tier taxonomy on known real and constructed examples.** Reconfirms literal, vacuous, corruption, Experiment-2 echo/drift, partial, stuck, verbalized, and genuine-code cases, including the corrected “changes without tracking truth” drift rule. | 16 | 0 | 16 pass |
| 13 | **Train/eval isolation at every length.** Proves the persisted Step-0 audit contains lengths 4–7 with zero overlap and exact 21/11, 43/21, 50/25, and 50/25 train/eval sizes. A post-Step-12 test additionally requires the prospective Stage 9 L5 eval set to be deterministic, verifier-valid, disjoint from both its 43-example train pool and the 16-example synthetic bank, and to cover the full 64-scenario partition. | 0 | 2 | 2 pass |
| 14 | **KL clamp and hard/soft breakers.** Existing tests exercise reconstructed soft-stop behavior and values; Step 11 binds those values plus grad>=50, KL>=5, clamp 0.1518003887/D-max 0.5047181100, healthy 23.76% engagement, and no fired breaker to the persisted run. Live TRL/CUDA clamp interception remains a Step 12 integration check. | 8 | 1 | 9 pass; live GPU path deferred |
| 15 | **Checkpoint identity and no composition.** Proves milestone 12 is a LoRA over Qwen2.5-3B-Instruct, has matching fresh-init identity in config/final evidence, contains no `PeftModel.from_pretrained` composition path, declares no source checkpoint, and its weight SHA matches the Step-8 policy audit. Actual base+adapter loading requires the Step 12 model environment. | 0 | 1 | 1 pass; live load deferred |

After the post-Step-12 split correction, these category rows total 60 passing Stage 9 cases. Another 58 passing
legacy cases cover infrastructure and earlier experiments outside this matrix,
producing the repository-wide total of 118 passing cases.

## Step 12 closure

The 8-step 10% dry run turned all three environment-bound portions into executed assertions:

1. Six scheduled auxiliary CE calls ran on CUDA through the bf16 model path; CE used
   stable float32 accumulation, all losses/gradients were finite, and returned terms
   were exactly 8x their optimizer-effective values.
2. Live TRL interception covered 64 policy-loss calls and 10,032 tokens, clamping 2,424
   (24.16%); no hard or soft breaker fired.
3. The untouched Qwen base loaded exactly one `default` adapter whose file SHA-256 was
   `f9f9742cc922cdf45b12f98530862ca941ab0ee0be38b5a22f05cd6de8276080`, matching Step 8.
