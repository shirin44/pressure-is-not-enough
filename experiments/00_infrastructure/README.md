# Infrastructure

Reused across every numbered experiment below. Code lives in `src/`, not
physically duplicated here — this is a pointer index, not a copy.

## Data generator + reward function
`src/data/coinflip.py` — Coin Flip task generator and the 6-term reward
function (task correctness, CoT-word penalty, length penalty, structure
penalty, transition-variation penalty, consistency bonus).
Tests: `experiments/00_infrastructure/test_coinflip_data.py`,
`scripts/test_reward_fn.py`.

> `src/data/coinflip.py` and `scripts/test_reward_fn.py` currently have
> uncommitted local changes predating this reorganization and were left
> untouched — see repo root, not moved as part of this pass.
> `scripts/run_grpo_dryrun_colab.py` (the direct-RL dry-run entry point,
> logically part of `01_direct_rl_coinflip`) has the same status and stays
> at its current path for the same reason.

## Dynamic sampling
DAPO-style rejection of degenerate rollout groups (correctness-degenerate,
or fewer than 25% structurally-valid completions), 3-attempt fallback. Not
a standalone module — implemented inline as the `dynamic_generate` wrapper
pattern in every GRPO training notebook. Representative example:
`01_direct_rl_coinflip/notebooks/checkpoint500_corrected_full_reward_dryrun.ipynb`.

## Monitoring / safeguards
`src/training/resume_guard.py` — refuses a literal checkpoint resume when
the optimizer/scheduler learning rate is nonfinite or near-zero, or when a
known scheduler horizon is exhausted. Proceeding then requires either
`weights_only_fresh_schedule` or an explicit reviewed override. Every new
resume notebook must use this before model construction. Tests:
`experiments/00_infrastructure/test_resume_guard.py`.

Circuit breakers (`grad_norm>=50` or `kl>=5` hard stop) are likewise
inline, not a standalone module — representative example: the `Safety`
callback in any `04_stability_investigation/notebooks/checkpoint500_*.ipynb`.

`src/data/unique_sampling.py` — guards against the "draw a random scenario,
dedupe against a `seen` set, retry `while len(pool) < n_requested`" loop
hanging forever when the underlying combinatorial space is smaller than
`n_requested` (hit twice: the Coin Flip bootstrap-generation prompt pool in
`05_self_bootstrapping`, and the flip-length sweep in the Coin Flip
load-bearing-regime experiment, both only caught via a `py-spy` stack dump
after several minutes of silent hanging). `check_unique_sample_feasible()`
raises immediately instead of hanging; `sample_unique()` safely draws from
an already-enumerated finite space (shuffle + slice, never probe-and-dedupe).
Every new prompt-pool generator that requests N unique scenarios should use
this before its draw loop, not after discovering the hang. Tests:
`experiments/00_infrastructure/test_unique_sampling.py`.

Full Drive snapshots (used for resumable checkpoints throughout) are
written to `/content/drive/MyDrive/AISI/checkpoints/full-snapshots/step-XXX/`,
each containing adapter/model state, optimizer and scheduler state,
`trainer_state.json`, exact reward source, a complete configuration
snapshot, raw monitoring history, and a versioned manifest. Only resume
from a snapshot whose `manifest.json` contains `"roundtrip_verified": true`.
