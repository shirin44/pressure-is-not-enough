"""CPU-only tests for the direct code-word answer bank (revised Stage 9e design,
Llama-3-8B-Instruct). The Same/Different task is abandoned; this is the current
design -- see design.md."""
from collections import Counter

from code_word_answer_bank import (
    ALL_SCENARIOS, BANK_KEYS, CODE_FOR, HEADS_CODE, TAILS_CODE,
    build_code_word_completion, build_code_word_train_eval_split, verify_code_word_trajectory,
)

CODE_WORD_SEED = 20260907


def test_all_scenarios_covers_the_full_length5_universe():
    assert len(ALL_SCENARIOS) == 64


def test_all_sixty_four_scenarios_fully_verified():
    results = [verify_code_word_trajectory(s, list(ops)) for s, ops in ALL_SCENARIOS]
    assert all(r['all_verified'] for r in results)
    assert all(r['intermediate_correct'] for r in results)
    assert all(r['final_answer_correct'] for r in results)
    assert all(r['no_literal_leak'] for r in results)


def test_final_answer_format_uses_leading_space_for_single_token_preservation():
    completion = build_code_word_completion('Heads', ['same', 'same', 'same', 'same', 'same'])
    assert completion.strip().endswith(f'<answer> {HEADS_CODE}</answer>')
    assert f'<answer>{HEADS_CODE}</answer>' not in completion  # the no-space form, rejected


def test_completion_contains_no_decode_back_line_of_any_kind():
    completion = build_code_word_completion('Tails', ['different', 'same', 'different', 'same', 'different'])
    assert 'decodes to' not in completion.lower()
    assert 'the final code' not in completion.lower()


def test_code_mapping_is_consistent_and_bidirectional():
    assert CODE_FOR['Heads'] == HEADS_CODE
    assert CODE_FOR['Tails'] == TAILS_CODE
    assert HEADS_CODE != TAILS_CODE


def test_stratified_split_achieves_closest_possible_balance():
    train_rows, eval_rows = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=21)
    assert len(train_rows) == 43
    assert len(eval_rows) == 21
    eval_counts = Counter(r['final_answer_state'] for r in eval_rows)
    assert {eval_counts['Heads'], eval_counts['Tails']} == {10, 11}
    train_counts = Counter(r['final_answer_state'] for r in train_rows)
    assert {train_counts['Heads'], train_counts['Tails']} == {21, 22}


def test_split_does_not_reuse_the_unbalanced_original_seed_naively():
    _, eval_rows = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=21)
    counts = Counter(r['final_answer_state'] for r in eval_rows)
    assert counts != Counter({'Heads': 14, 'Tails': 7})  # the unstratified CLEAN21_SEED result


def test_no_overlap_between_train_eval_and_bank_keys():
    train_rows, eval_rows = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=21)
    train_keys = {(r['starting_state'], tuple(r['operations'])) for r in train_rows}
    eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
    assert not (eval_keys & set(BANK_KEYS))
    assert not (train_keys & eval_keys)


def test_final_answer_code_matches_final_answer_state_mapping():
    train_rows, eval_rows = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=21)
    for row in train_rows + eval_rows:
        assert row['final_answer_code'] == CODE_FOR[row['final_answer_state']]


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
