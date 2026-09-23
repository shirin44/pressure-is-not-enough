"""Golden-record regression test against the real, persisted code-word answer bank
(aws_runs/stage9e-code-word-bank-v1/stage9e_code_word_bank.json), Stage 9e revised
design. All 64 length-5 scenarios verified, stratified-balanced train/eval split,
zero leakage, every entry's token positions logged and confirmed single-token."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-code-word-bank-v1' / 'stage9e_code_word_bank.json').read_text())


def test_full_universe_verified():
    v = EVIDENCE['full_universe_verification']
    assert v['n_scenarios'] == 64
    assert v['n_verified'] == 64


def test_bank_sizes_match_expected():
    assert len(EVIDENCE['train_rows']) == 43
    assert len(EVIDENCE['eval_rows']) == 21


def test_zero_leakage_across_the_full_bank():
    la = EVIDENCE['leakage_audit']
    assert la['n_total'] == 64
    assert la['n_clean'] == 64
    assert la['all_clean'] is True
    assert la['failures'] == []


def test_every_row_has_confirmed_single_token_positions():
    all_rows = EVIDENCE['train_rows'] + EVIDENCE['eval_rows']
    assert len(all_rows) == 64
    for row in all_rows:
        tp = row['token_positions']
        assert tp['all_occurrences_single_token_confirmed'] is True
        assert tp['occurrence_count_matches_expected'] is True
        assert tp['n_occurrences_found'] == 6  # 5 steps + 1 final answer, length 5


def test_every_row_final_occurrence_is_role_tagged_final_answer():
    all_rows = EVIDENCE['train_rows'] + EVIDENCE['eval_rows']
    for row in all_rows:
        occurrences = row['token_positions']['occurrences']
        assert occurrences[-1]['role'] == 'final_answer'
        assert all(o['role'] == f'intermediate_step_{i + 1}' for i, o in enumerate(occurrences[:-1]))


def test_every_row_verification_passed():
    all_rows = EVIDENCE['train_rows'] + EVIDENCE['eval_rows']
    for row in all_rows:
        v = row['verification']
        assert v['all_verified'] is True
        assert v['intermediate_correct'] is True
        assert v['final_answer_correct'] is True
        assert v['no_literal_leak'] is True


def test_train_and_eval_balance_is_closest_possible_given_odd_counts():
    from collections import Counter
    train_counts = Counter(r['final_answer_state'] for r in EVIDENCE['train_rows'])
    eval_counts = Counter(r['final_answer_state'] for r in EVIDENCE['eval_rows'])
    assert {train_counts['Heads'], train_counts['Tails']} == {21, 22}
    assert {eval_counts['Heads'], eval_counts['Tails']} == {10, 11}


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
