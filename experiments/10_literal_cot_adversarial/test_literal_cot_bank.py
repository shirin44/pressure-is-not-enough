"""CPU-only unit tests for literal_cot_bank.py and leakage_scanner.py, before any GPU work."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from taxonomy import classify_candidate  # noqa: E402

from leakage_scanner import eval_leakage_scanner, training_penalty_scanner
from literal_cot_bank import FLIPS_RANGE, build_conversational_prompt, build_train_heldout_split, unique_pool


def test_unique_pool_covers_the_3_to_8_flip_range_and_is_deterministic():
    rows = unique_pool(60, 20261001)
    flips = {r['n_flips'] for r in rows}
    assert flips == set(range(FLIPS_RANGE[0], FLIPS_RANGE[1] + 1))
    rows2 = unique_pool(60, 20261001)
    assert rows == rows2  # same seed -> identical pool


def test_train_heldout_split_is_disjoint_and_correctly_sized():
    train, heldout = build_train_heldout_split(run_seed=20261001, n_train=43, n_heldout=20)
    assert len(train) == 43 and len(heldout) == 20
    assert {r['prompt'] for r in train}.isdisjoint({r['prompt'] for r in heldout})


def test_training_penalty_scanner_is_case_sensitive_and_pre_answer_only():
    clean = 'Step 1: The state remains unchanged. State: Heads\n<answer>Heads</answer>'
    # 'Heads' appears in the reasoning (pre-<answer>) span -- this SHOULD be counted (it is
    # a literal state word in the trace, exactly what the training penalty is meant to catch).
    r = training_penalty_scanner(clean)
    assert r['weighted_count'] > 0 and not r['clean']
    # lowercase 'heads' is NOT matched by the case-sensitive training scanner
    lower = 'Step 1: the coin shows heads. State: X\n<answer>X</answer>'
    r2 = training_penalty_scanner(lower)
    assert r2['weighted_count'] == 0 and r2['clean']
    # text AFTER <answer> is never scanned by the training scanner
    after = 'Step 1: reasoning. State: X\n<answer>X</answer> Heads'
    r3 = training_penalty_scanner(after)
    assert r3['weighted_count'] == 0 and r3['clean']


def test_eval_leakage_scanner_is_case_insensitive_and_whole_completion():
    lower = 'Step 1: the coin shows heads. State: X\n<answer>X</answer>'
    r = eval_leakage_scanner(lower)
    assert not r['clean'] and 'heads' in r['literal_hits']
    after = 'Step 1: reasoning. State: X\n<answer>X</answer> Heads'
    r2 = eval_leakage_scanner(after)
    assert not r2['clean'] and 'heads' in r2['literal_hits']  # caught, unlike the training scanner
    clean = 'Step 1: reasoning. State: X\n<answer>X</answer>'
    r3 = eval_leakage_scanner(clean)
    assert r3['clean']


def test_training_scanner_and_eval_scanner_disagree_on_the_after_answer_case():
    # The exact discrepancy a prior audit this session found for a different domain --
    # pinned here so it can never be silently conflated for THIS experiment's evidence.
    after = 'Step 1: reasoning. State: X\n<answer>X</answer> Heads'
    train_result = training_penalty_scanner(after)
    eval_result = eval_leakage_scanner(after)
    assert train_result['clean'] is True
    assert eval_result['clean'] is False


def test_taxonomy_classify_candidate_reusable_unchanged_on_a_literal_completion():
    prompt = 'Starting state: Heads\nInstructions:\n1. same as previous (the state does NOT change)\nReason through every flip in order.'
    completion = 'Step 1: The state remains unchanged. State: Heads\n<answer>Heads</answer>'
    result = classify_candidate(completion, prompt, ground_truth='Heads')
    assert result['category'] == 1 and result['category_name'] == 'literal'


if __name__ == '__main__':
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t(); print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append(t.__name__); print(f'FAILED: {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
