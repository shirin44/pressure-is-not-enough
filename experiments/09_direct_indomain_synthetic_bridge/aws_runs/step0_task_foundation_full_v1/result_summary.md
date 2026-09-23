# Step-0 full run v1 — ceiling shortfall

The run reached its immutable 32-step ceiling without a milestone satisfying the
precommitted 70% floor plus non-declining two-point stability gate. The shortfall
fallback is therefore formally invoked. Step 8 is blocked.

| Step | Mean normal-CoT accuracy | Answer-only | Vacuous | Literal | KL clamp engagement |
|---:|---:|---:|---:|---:|---:|
| 4 | 66.15% | 43.81% | 0% | 100% | 14.59% |
| 8 | 66.87% | 44.19% | 0% | 100% | 21.25% |
| 12 | **68.69%** | 40.64% | 0% | 100% | 23.60% |
| 16 | 61.87% | 46.00% | 0% | 100% | 23.59% |
| 20 | 65.04% | 45.38% | 0% | 100% | 24.53% |
| 24 | 61.77% | 45.72% | 0% | 100% | 23.39% |
| 28 | 63.15% | 42.91% | 0% | 100% | 24.07% |
| 32 | 59.68% | 43.00% | 0% | 100% | 23.76% |

The selected checkpoint is **milestone 12**, the highest held-out macro accuracy
across flip-lengths 4-7 (68.69%); it is intentionally not the final checkpoint. Its
per-length accuracies were 81.82% (L4), 80.95% (L5), 64.00% (L6), and 48.00% (L7).
The adapter is preserved under `milestone-12/`; all eight milestone checkpoints remain
on the run host under the paths recorded in `step0_task_foundation.json`.

This is disclosed as slower and unstable fresh-LoRA task acquisition, with an observed
ceiling below the required floor under this exact zeroed-concealment-reward
configuration. It is not silently retried and the 32-step ceiling is not extended.

Safety checks were clean: the per-length train/eval overlap count was zero for lengths
4, 5, 6, and 7; all 656 milestone completions were literal (zero structural non-literal
classifications); no hard or soft breaker fired; maximum logged gradient norm was
27.125 (<50), maximum logged KL was 0.05950 (<5), and final cumulative KL-clamp
engagement was 23.76% (inside the predeclared ~15-25% healthy range).

Evidence files: `step0_task_foundation.json`, `per_token_instrumentation.json`, and
`stdout.log`. The exact launched source is the stage-level `step0_train.py` at SHA-256
`8369adb3cb83b2427dc06e8e4e1e34d650f198b75ea50be4a4f76c50f0942ded`.
