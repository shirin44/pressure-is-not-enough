"""CPU-only tests for the rebalanced trajectory bank built in
sft_train_eval_rebalanced.py (2026-08-31). Reproduces the exact rebalancing logic --
run before spending any GPU time, per this project's established discipline."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_bridge import (CODED_TRAJECTORIES, build_clean_length5_train_eval_split,
    build_prompt, build_coded_completion, verify_bank)

CLEAN21_SEED = 20260831
REBALANCE_SEED = 20260901


def _build_rebalanced_bank():
    train_scenarios_meta, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
    original_bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in CODED_TRAJECTORIES}
    tails_rows = [r for r in train_scenarios_meta if r['final_answer'] == 'Tails']
    heads_rows = [r for r in train_scenarios_meta if r['final_answer'] == 'Heads']
    n_to_remove = len(tails_rows) - len(heads_rows)
    tails_new_only = [r for r in tails_rows if (r['starting_state'], tuple(r['operations'])) not in original_bank_keys]
    rng = random.Random(REBALANCE_SEED)
    to_remove = rng.sample(tails_new_only, n_to_remove)
    to_remove_keys = {(r['starting_state'], tuple(r['operations'])) for r in to_remove}
    rebalanced_meta = [r for r in train_scenarios_meta if (r['starting_state'], tuple(r['operations'])) not in to_remove_keys]

    bank = []
    for r in rebalanced_meta:
        starting_state, operations = r['starting_state'], r['operations']
        completion = build_coded_completion(starting_state, operations)
        bank.append({'starting_state': starting_state, 'operations': list(operations),
            'prompt': build_prompt(starting_state, operations), 'completion': completion,
            'final_answer': r['final_answer'], 'coded': True})
    return bank, eval_rows, to_remove


def test_no_uncovered_length5_pool_exists():
    # Confirms the mathematical constraint that makes trimming necessary: the expanded
    # bank (43) plus the held-out eval (21) already exhaust the full 64-scenario
    # length-5 space.
    train_scenarios_meta, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
    assert len(train_scenarios_meta) + len(eval_rows) == 64


def test_rebalanced_bank_achieves_exact_parity():
    bank, _, _ = _build_rebalanced_bank()
    dist = Counter(r['final_answer'] for r in bank)
    assert dist['Tails'] == dist['Heads']
    assert len(bank) == 36


def test_rebalanced_bank_removes_exactly_7_scenarios_from_the_expanded_43():
    bank, _, removed = _build_rebalanced_bank()
    assert len(removed) == 7
    assert all(r['final_answer'] == 'Tails' for r in removed)


def test_rebalanced_bank_preserves_the_entire_original_16_row_bank():
    bank, _, _ = _build_rebalanced_bank()
    original_keys = {(r['starting_state'], tuple(r['operations'])) for r in CODED_TRAJECTORIES}
    bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in bank}
    assert original_keys.issubset(bank_keys)


def test_rebalanced_bank_coverage_still_more_than_double_the_original():
    bank, _, _ = _build_rebalanced_bank()
    assert len(bank) / 64 > 2 * (16 / 64)  # 36/64=56.2% vs 16/64=25.0%


def test_rebalanced_bank_passes_the_existing_verification_pipeline():
    bank, _, _ = _build_rebalanced_bank()
    reports = verify_bank(bank)
    assert len(reports) == 36
    assert all(r['verified'] for r in reports)


def test_rebalanced_bank_has_no_overlap_with_held_out_eval():
    bank, eval_rows, _ = _build_rebalanced_bank()
    bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in bank}
    eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
    assert not (bank_keys & eval_keys)


def test_rebalanced_bank_is_deterministic():
    bank1, _, _ = _build_rebalanced_bank()
    bank2, _, _ = _build_rebalanced_bank()
    sha1 = hashlib.sha256(json.dumps(bank1, sort_keys=True).encode()).hexdigest()
    sha2 = hashlib.sha256(json.dumps(bank2, sort_keys=True).encode()).hexdigest()
    assert sha1 == sha2 == 'e3097a0fa7cc8f334b669031a6007343ba8960f6178d75eb5ab20490854ad84a'


if __name__ == '__main__':
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t()
            print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append((t.__name__, e))
            print(f'FAILED: {t.__name__}: {e}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
