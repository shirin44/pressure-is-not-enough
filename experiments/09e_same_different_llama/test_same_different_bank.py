"""CPU-only tests for the Same/Different trajectory bank (Decisions 3-6), run before
any GPU work touches the bank."""
from collections import Counter

from same_different_bank import (
    ALL_SCENARIOS, BANK_KEYS, build_coded_same_different_completion,
    build_same_different_train_eval_split,
)
from same_different_logic import same_different_answer

SAME_DIFFERENT_SEED = 20260907


def test_all_scenarios_covers_the_full_length5_universe():
    assert len(ALL_SCENARIOS) == 64  # 2 starting states x 2^5 operation sequences


def test_full_universe_is_exactly_balanced():
    answers = [same_different_answer(s, list(ops)) for s, ops in ALL_SCENARIOS]
    counts = Counter(answers)
    assert counts['Same'] == 32
    assert counts['Different'] == 32


def test_bank_keys_are_already_exactly_balanced():
    answers = [same_different_answer(s, ops) for s, ops in BANK_KEYS]
    counts = Counter(answers)
    assert counts['Same'] == 8
    assert counts['Different'] == 8


def test_stratified_split_achieves_closest_possible_balance():
    # Decision 5: exactly 50/50 is impossible for odd n_eval=21 -- the achievable
    # closest balance is 10/11 or 11/10, verified directly, not assumed.
    train_rows, eval_rows = build_same_different_train_eval_split(seed=SAME_DIFFERENT_SEED, n_eval=21)
    assert len(train_rows) == 43
    assert len(eval_rows) == 21
    eval_counts = Counter(r['same_different_answer'] for r in eval_rows)
    assert {eval_counts['Same'], eval_counts['Different']} == {10, 11}
    train_counts = Counter(r['same_different_answer'] for r in train_rows)
    assert {train_counts['Same'], train_counts['Different']} == {21, 22}


def test_split_does_not_reuse_the_unbalanced_original_seed_naively():
    # Explicitly confirms this is NOT the same skewed split the original
    # (non-stratified) CLEAN21_SEED produces when reused as-is for this task.
    _, eval_rows = build_same_different_train_eval_split(seed=SAME_DIFFERENT_SEED, n_eval=21)
    counts = Counter(r['same_different_answer'] for r in eval_rows)
    assert counts != Counter({'Different': 13, 'Same': 8})  # the original unbalanced result


def test_no_overlap_between_train_eval_and_bank_scenarios():
    train_rows, eval_rows = build_same_different_train_eval_split(seed=SAME_DIFFERENT_SEED, n_eval=21)
    train_keys = {(r['starting_state'], tuple(r['operations'])) for r in train_rows}
    eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
    assert not (eval_keys & BANK_KEYS)
    assert not (train_keys & eval_keys)


def test_coded_completion_has_no_decode_back_line_and_no_literal_state_word():
    completion = build_coded_same_different_completion('Heads', ['different', 'same', 'different', 'same', 'different'])
    assert 'heads' not in completion.lower()
    assert 'tails' not in completion.lower()
    assert 'decodes to' not in completion.lower()  # confirms no decode-back sentence at all
    assert completion.strip().endswith('</answer>')


def test_coded_completion_answer_matches_the_verified_logic():
    starting_state, operations = 'Tails', ['same', 'different', 'different', 'same', 'same']
    completion = build_coded_same_different_completion(starting_state, operations)
    expected = same_different_answer(starting_state, operations)
    assert f'<answer>{expected}</answer>' in completion


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
