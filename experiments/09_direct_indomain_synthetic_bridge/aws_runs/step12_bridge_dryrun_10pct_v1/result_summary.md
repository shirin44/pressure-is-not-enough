# Step 12 bridge dry run — result summary

## Scope

This was the precommitted eight-optimizer-step dry run only. It used the 10% constant
exposure condition and length 5 as the primary task/evaluation length. No full bridge
run was launched. At an eight-step horizon, the exact cumulative-floor quota is
`0,1,1,1,1,0,1,1`: six synthetic examples total. The often-quoted 7–8 reuses per bank
entry applies to 120 exposures over a 150-step full run, not this dry run.

## Foundation load

The untouched `Qwen/Qwen2.5-3B-Instruct` base loaded exactly one PEFT adapter named
`default`, from Step-0 milestone 12. The adapter file SHA-256 was
`f9f9742cc922cdf45b12f98530862ca941ab0ee0be38b5a22f05cd6de8276080`,
identical to the adapter audited in Step 8. There were no meta parameters and no second,
merged, or composed adapter. Optimizer and scheduler were fresh; policy weights were a
continuation from milestone 12, not a fresh LoRA.

## `beta_sft` calibration

Loss magnitude alone is not a sound scale for centered-advantage GRPO, so calibration
used trainable-parameter gradient norms on the first real rollout group:

| quantity | value |
|---|---:|
| GRPO loss | -0.0006904486 |
| auxiliary CE | 1.2505816 |
| GRPO gradient norm | 0.00567208 |
| unweighted CE gradient norm | 2.78724762 |

Candidate weights were derived—not guessed—to make the weighted auxiliary gradient 5%,
10%, or 20% of the measured GRPO gradient: `0.0001017506`, `0.0002035012`, and
`0.0004070025`. The middle candidate, **`beta_sft=0.00020350124759158003`**, was selected.
It makes the signal material while retaining a 10x GRPO-gradient margin at calibration.

## CUDA CE and optimization health

The six scheduled auxiliary calls ran at steps 2, 3, 4, 5, 7, and 8 on `cuda:0` through
the bf16 model path. Cross-entropy accumulated in float32, as expected for numerical
stability. CE values were 1.85777, 1.83703, 1.86569, 2.20400, 2.33747, and 1.96015.
Every CE, GRPO loss, weighted term, and calibration gradient was finite. Returned
microbatch auxiliary terms were exactly eight times the optimizer-effective terms,
closing the gradient-accumulation scaling check.

Per-step trainer gradient norms were `0.0366, 0.0483, 11.125, 10.438, 1.039, 24.875,
22.5, 0.4805`; the maximum was below the hard threshold of 50. KL values were `0,
0.00297, 0.05254, 0.04254, 0.02040, 0.05473, 0.04927, 0.00746`, all below 5. No hard
or reconstructed soft breaker fired.

## Live KL clamp

The installed TRL interception executed on 64 real policy-loss calls. It inspected
10,032 tokens and clamped 2,424, for **24.16% engagement**. Milestone engagement was
22.10% at step 4 and 24.16% at step 8, inside the expected 15–25% healthy range.

## Milestones and taxonomy

| step | normal-CoT accuracy | answer-only | exposed scenarios | unseen scenarios | taxonomy | KL clamp engagement |
|---:|---:|---:|---:|---:|---|---:|
| 4 | 80.95% | 52.38% | 4/5 (80.0%) | 13/16 (81.25%) | 21 literal | 22.10% |
| 8 | 80.95% | 57.14% | 4/5 (80.0%) | 13/16 (81.25%) | 21 literal | 24.16% |

All 42 milestone completions were classified. None was a
`structural_nonliteral_candidate`; none exactly echoed an injected completion. Thus the
dry run establishes mechanism/numerical safety but does not yet demonstrate code
learning, memorization, or held-out scenario generalization. An eight-step run with only
six synthetic exposures is too short to distinguish those outcomes.

Five of the 21 evaluation scenarios overlap the synthetic scenario bank. This was
detected before launch and is reported separately above; only the other 16 scenarios
can support scenario-level generalization claims in future runs. A single aggregate
over all 21 must not be presented as wholly held out from auxiliary supervision.

**Prospective correction after this dry run:** the full-run configuration now uses
`build_clean_length5_train_eval_split(seed=20260831, n_eval=21)`, drawing evaluation
only from the 48 non-bank scenarios. Its executable audit reports 21 eval, 43 train,
zero train/eval overlap, zero bank/eval overlap, and 21/21 ground truths accepted by the
strict literal-trajectory verifier. The table above remains an accurate historical
record of this dry run and is not silently rewritten as though it used the corrected
split.

## Infrastructure

AWS lacked `g5.xlarge` and `g6.xlarge` capacity in the instance's availability zone, so
the same AISI instance and volume were temporarily run as `g5.2xlarge` (A10G). After
evidence retrieval it was stopped, restored to `g5.xlarge`, and the temporary `/32` SSH
ingress rule was revoked.
