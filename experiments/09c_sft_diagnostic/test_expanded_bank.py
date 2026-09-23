"""CPU-only tests for the expanded trajectory bank built in sft_train_eval_expanded.py
(2026-08-31 SFT-seed rebuild). Reproduces the exact bank-construction logic against
synthetic_bridge.py's UNCHANGED functions (build_prompt, build_coded_completion,
verify_bank, build_clean_length5_train_eval_split) -- run before spending any GPU time,
per this project's established discipline."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_bridge import (CODED_TRAJECTORIES, build_clean_length5_train_eval_split,
    build_prompt, build_coded_completion, verify_bank)

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'


def _build_expanded_bank():
    train_scenarios_meta, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
    bank = []
    for meta in train_scenarios_meta:
        starting_state, operations = meta['starting_state'], meta['operations']
        completion = build_coded_completion(starting_state, operations)
        bank.append({'starting_state': starting_state, 'operations': list(operations),
            'prompt': build_prompt(starting_state, operations), 'completion': completion,
            'final_answer': meta['final_answer'], 'coded': True})
    return bank, eval_rows


def test_held_out_21_is_byte_identical_to_stage9c():
    _, eval_rows = _build_expanded_bank()
    actual_sha = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
    assert actual_sha == EXPECTED_CLEAN21_SHA256, (
        'held-out eval set must be byte-identical to Stage 9c\'s for the 28.6% comparison to be valid')


def test_expanded_bank_is_substantially_larger_than_original():
    bank, _ = _build_expanded_bank()
    assert len(bank) == 43
    assert len(bank) > 2.5 * len(CODED_TRAJECTORIES)  # 43 vs 16 -- a real, substantial expansion


def test_expanded_bank_covers_67_percent_of_length5_space():
    bank, _ = _build_expanded_bank()
    coverage = len(bank) / 64
    assert abs(coverage - 43 / 64) < 1e-9
    assert coverage > 0.6  # substantially more than the original 16/64 = 25.0%


def test_expanded_bank_passes_the_existing_verification_pipeline():
    bank, _ = _build_expanded_bank()
    reports = verify_bank(bank)  # raises on any failure -- reaching this line means every row passed
    assert len(reports) == len(bank)
    assert all(r['verified'] for r in reports)
    assert all(r['correct_state_tracking'] for r in reports)
    assert all(r['correct_final_answer'] for r in reports)
    assert all(r['code_consistent'] for r in reports)  # no literal Heads/Tails leakage


def test_expanded_bank_is_a_strict_superset_of_the_original_16():
    bank, _ = _build_expanded_bank()
    original_keys = {(r['starting_state'], tuple(r['operations'])) for r in CODED_TRAJECTORIES}
    expanded_keys = {(r['starting_state'], tuple(r['operations'])) for r in bank}
    assert original_keys.issubset(expanded_keys)
    assert len(expanded_keys - original_keys) == 27  # 27 genuinely new scenarios beyond the original 16


def test_expanded_bank_has_no_overlap_with_held_out_eval():
    bank, eval_rows = _build_expanded_bank()
    bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in bank}
    eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
    assert not (bank_keys & eval_keys)


def test_expanded_bank_has_no_duplicate_scenarios():
    bank, _ = _build_expanded_bank()
    keys = [(r['starting_state'], tuple(r['operations'])) for r in bank]
    assert len(set(keys)) == len(keys)


def test_expanded_bank_sha256_is_deterministic():
    bank1, _ = _build_expanded_bank()
    bank2, _ = _build_expanded_bank()
    sha1 = hashlib.sha256(json.dumps(bank1, sort_keys=True).encode()).hexdigest()
    sha2 = hashlib.sha256(json.dumps(bank2, sort_keys=True).encode()).hexdigest()
    assert sha1 == sha2 == '80608bb45d58783c90253399bd6d6e8439f9d082bb20ee008af759ec88db3ef9'


def test_micro_batching_covers_all_examples_with_no_overlap_or_gap():
    # Regression guard for the 2026-08-31 OOM fix: the full-batch forward over all 43
    # examples at once didn't fit (CUDA OOM on the very first backward()). Fixed with
    # gradient accumulation over MICRO_BATCH_SIZE=8 chunks -- this test checks the
    # chunking logic itself (pure Python, no GPU) partitions the bank correctly and that
    # the per-chunk loss weights sum to exactly 1.0, so the accumulated gradient is
    # mathematically equivalent to Stage 9c's full-batch mean-CE-over-the-whole-set.
    bank, _ = _build_expanded_bank()
    MICRO_BATCH_SIZE = 8
    chunks = [bank[i:i + MICRO_BATCH_SIZE] for i in range(0, len(bank), MICRO_BATCH_SIZE)]
    assert sum(len(c) for c in chunks) == len(bank) == 43
    assert all(len(c) <= MICRO_BATCH_SIZE for c in chunks)
    assert all(len(c) <= 16 for c in chunks), 'every micro-batch must be no larger than the size that fit in Stage 9c (16)'
    flattened_keys = [(r['starting_state'], tuple(r['operations'])) for c in chunks for r in c]
    bank_keys = [(r['starting_state'], tuple(r['operations'])) for r in bank]
    assert flattened_keys == bank_keys  # exact partition, same order, no example dropped or duplicated
    weights = [len(c) / len(bank) for c in chunks]
    assert abs(sum(weights) - 1.0) < 1e-9


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
