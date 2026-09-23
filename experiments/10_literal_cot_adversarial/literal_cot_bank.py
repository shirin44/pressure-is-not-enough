"""Stage 10: prompt/held-out-split construction for the literal-CoT adversarial-pressure
experiment. Reuses `generate_coinflip_example` from src/data/coinflip.py UNCHANGED (the same
function `stage07_signal_annealed.py` used against checkpoint 500, and the same function
Stage 9e Llama's design deliberately mirrors for the Coin Flip domain) -- not reimplemented.

HARD REQUIREMENT (per task instruction, citing the Llama audit finding
grpo_trainer.py's data_utils.py:160-188 / :1707): prompts for GRPOTrainer's train_dataset
must be CONVERSATIONAL (list-of-dict), never a raw string, so TRL's own conversational-prompt
branch applies the tokenizer's chat template automatically at generation time. This module
never returns a raw string as a 'prompt' value; build_conversational_prompt() is the only
prompt-shaping function used by the runner.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'src' / 'data'))
from coinflip import generate_coinflip_example  # noqa: E402

FLIPS_RANGE = (3, 8)
N_HELDOUT = 20


def build_conversational_prompt(prompt_text: str) -> list[dict[str, str]]:
    """The ONLY prompt shape fed to GRPOTrainer's train_dataset or to eval's chat_wrap
    equivalent. A list-of-dict message, never a plain string."""
    return [{'role': 'user', 'content': prompt_text}]


def unique_pool(size: int, start_seed: int, excluded: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Deterministic, disjoint-from-`excluded` pool of Coin Flip scenarios, 3-8 instructions
    (FLIPS_RANGE), cycling seeds from start_seed. Same disjoint-pool pattern used throughout
    this project (Stage 1's checkpoint-500 diagnostics, Stage 7's TRAIN_POOL/HELDOUT_POOL)."""
    if size < 0:
        raise ValueError('size must be non-negative')
    rows: list[dict[str, Any]] = []
    seen = set(excluded)
    seed = start_seed
    while len(rows) < size:
        n_flips = 3 + (seed % 6)  # cycles 3..8, same construction as Stage 1/7's own unique_pool
        prompt, truth = generate_coinflip_example(n_flips, seed)
        seed += 1
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({'prompt': prompt, 'ground_truth': truth, 'n_flips': n_flips})
    return rows


def build_train_heldout_split(*, run_seed: int, n_train: int, n_heldout: int = N_HELDOUT) -> tuple[list[dict], list[dict]]:
    """Disjoint TRAIN_POOL / HELDOUT_POOL, verified disjoint by construction and by an
    explicit assertion (not just by convention)."""
    train_rows = unique_pool(n_train, run_seed)
    heldout_rows = unique_pool(n_heldout, run_seed + 1_000_000, frozenset(r['prompt'] for r in train_rows))
    train_prompts = {r['prompt'] for r in train_rows}
    heldout_prompts = {r['prompt'] for r in heldout_rows}
    assert train_prompts.isdisjoint(heldout_prompts), 'train/held-out pool overlap -- must never happen'
    assert len(train_rows) == n_train and len(heldout_rows) == n_heldout
    return train_rows, heldout_rows
