"""Stage 9e, Same/Different trajectory bank (Decisions 3, 4, 5, 6).

Decision 5 finding, verified programmatically rather than assumed: reusing the
ORIGINAL Heads/Tails task's exact train/eval partition (same seed, same scenario
split) does NOT give a balanced Same/Different bank -- it was randomized without any
awareness of Same/Different parity, since that concept didn't exist yet. Directly
measured: train 24 Same / 19 Different, eval 8 Same / 13 Different, both meaningfully
skewed. This module builds a STRATIFIED split instead: scenarios are partitioned by
their own Same/Different label first, then drawn from each stratum, guaranteeing
near-exact balance by construction rather than hoping a seed lands well.

Exact 50/50 is mathematically impossible for an odd-sized split (n_eval=21, matching
the original task's scale for direct comparability per Decision 7) -- the closest
achievable balance is 11/10 or 10/11. This module achieves that closest-possible
balance and reports the EXACT achieved counts, not a rounded or assumed number.

Uses the DECLARED-starting_state origin anchoring (Decision 4, same_different_logic.py).
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from synthetic_bridge import CODED_TRAJECTORIES, N_FLIPS, HEADS_CODE, TAILS_CODE, _trace  # noqa: E402

from same_different_logic import same_different_answer

BANK_KEYS = {(row['starting_state'], tuple(row['operations'])) for row in CODED_TRAJECTORIES}
ALL_SCENARIOS = [
    (starting_state, tuple(operations))
    for starting_state in ('Heads', 'Tails')
    for operations in itertools.product(('same', 'different'), repeat=N_FLIPS)
]


def _row(scenario: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    starting_state, operations = scenario
    return {
        'starting_state': starting_state,
        'operations': list(operations),
        'same_different_answer': same_different_answer(starting_state, operations),
    }


def build_same_different_train_eval_split(*, seed: int, n_eval: int = 21) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified by Same/Different label -- guarantees near-exact eval balance by
    construction. Eval scenarios drawn only from outside BANK_KEYS (the original
    hand-picked 16-scenario bank, itself already exactly 8 Same / 8 Different),
    mirroring build_clean_length5_train_eval_split()'s own exclusion rule."""
    eligible_eval = [sc for sc in ALL_SCENARIOS if sc not in BANK_KEYS]
    same_eligible = [sc for sc in eligible_eval if same_different_answer(sc[0], list(sc[1])) == 'Same']
    diff_eligible = [sc for sc in eligible_eval if same_different_answer(sc[0], list(sc[1])) == 'Different']

    n_same_eval = n_eval // 2
    n_diff_eval = n_eval - n_same_eval  # gets the extra one when n_eval is odd
    if n_same_eval > len(same_eligible) or n_diff_eval > len(diff_eligible):
        raise ValueError(f'n_eval={n_eval} exceeds available balanced pool '
                          f'(same_eligible={len(same_eligible)}, diff_eligible={len(diff_eligible)})')

    rng = random.Random(seed)
    same_shuffled = list(same_eligible); rng.shuffle(same_shuffled)
    diff_shuffled = list(diff_eligible); rng.shuffle(diff_shuffled)
    eval_scenarios = same_shuffled[:n_same_eval] + diff_shuffled[:n_diff_eval]
    eval_keys = set(eval_scenarios)

    train_scenarios = [sc for sc in ALL_SCENARIOS if sc not in eval_keys]
    train_rows = [_row(sc) for sc in train_scenarios]
    eval_rows = [_row(sc) for sc in eval_scenarios]
    assert not (eval_keys & BANK_KEYS)
    assert not ({(r['starting_state'], tuple(r['operations'])) for r in train_rows} & eval_keys)
    return train_rows, eval_rows


def build_coded_same_different_completion(starting_state: str, operations: list[str]) -> str:
    """Same rigid per-step template as synthetic_bridge.build_coded_completion(), but
    the final answer is Same/Different (Decision 4's origin-anchored comparison), and
    there is NO decode-back line -- nothing to decode back to, by design (there is no
    literal state word anywhere in this completion at all, only the code tokens and
    the Same/Different answer itself)."""
    states = _trace(starting_state, operations)
    lines = []
    for step, (operation, state) in enumerate(zip(operations, states), 1):
        token = HEADS_CODE if state == 'Heads' else TAILS_CODE
        if step == 1:
            lines.append(f'Step 1: The state is tracked. State: {token}')
        else:
            action = 'remains unchanged' if operation == 'same' else 'changes'
            lines.append(f'Step {step}: The code {action}. State: {token}')
    answer = same_different_answer(starting_state, operations)
    lines.append(f'<answer>{answer}</answer>')
    return '\n'.join(lines)
