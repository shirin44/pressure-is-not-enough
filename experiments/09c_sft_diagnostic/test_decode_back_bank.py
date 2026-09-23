"""CPU-only tests for decode_back_bank.py, run before spending any GPU time on the
decode-back-line SFT re-run, per this project's established discipline."""
import hashlib
import json
import re

import pytest

from decode_back_bank import (
    build_decode_back_completion, build_rebalanced_decode_back_bank, parse_decode_back_line,
    score_completion_v2_decode_back_aware, strip_decode_back_line, verify_decode_back_bank,
    verify_decode_back_trajectory)


def test_decode_back_completion_inserts_exactly_one_new_line_before_answer():
    c = build_decode_back_completion('Heads', ['same', 'different', 'different', 'same', 'different'])
    assert c.count('\nThe final code token ') == 1
    assert re.search(r'The final code token \S+ decodes to \S+\.\n<answer>', c)


def test_decode_back_line_decodes_the_true_final_state_for_every_16_original_scenarios():
    from synthetic_bridge import CODED_TRAJECTORIES, _trace
    for row in CODED_TRAJECTORIES:
        c = build_decode_back_completion(row['starting_state'], row['operations'])
        code, literal = parse_decode_back_line(c)
        true_final = _trace(row['starting_state'], row['operations'])[-1]
        assert literal == true_final
        assert code == ('Nib' if true_final == 'Heads' else 'Nomo')


def test_strip_decode_back_line_recovers_byte_identical_original_completion():
    from synthetic_bridge import build_coded_completion
    starting_state, operations = 'Tails', ['different', 'same', 'same', 'different', 'same']
    original = build_coded_completion(starting_state, operations)
    with_line = build_decode_back_completion(starting_state, operations)
    assert strip_decode_back_line(with_line) == original


def test_parse_decode_back_line_returns_none_when_missing():
    assert parse_decode_back_line('Step 1: foo. State: Nib\n<answer>Heads</answer>') is None


def test_verify_decode_back_trajectory_passes_on_a_correct_row():
    from synthetic_bridge import build_prompt
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    row = {'starting_state': starting_state, 'operations': operations,
           'prompt': build_prompt(starting_state, operations),
           'completion': build_decode_back_completion(starting_state, operations),
           'final_answer': 'Heads', 'coded': True}
    report = verify_decode_back_trajectory(row)
    assert report['verified'] is True
    assert report['decode_back_line_verified'] is True
    assert report['decode_back_code_token'] == 'nib'
    assert report['decode_back_literal_word'] == 'heads'


def test_verify_decode_back_trajectory_rejects_a_wrong_decoded_literal():
    from synthetic_bridge import build_prompt
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    row = {'starting_state': starting_state, 'operations': operations,
           'prompt': build_prompt(starting_state, operations),
           'completion': build_decode_back_completion(starting_state, operations)
               .replace('decodes to Heads', 'decodes to Tails'),
           'final_answer': 'Heads', 'coded': True}
    with pytest.raises(AssertionError, match='literal word'):
        verify_decode_back_trajectory(row)


def test_verify_decode_back_trajectory_rejects_a_code_token_mismatched_with_the_preceding_line():
    from synthetic_bridge import build_prompt
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    row = {'starting_state': starting_state, 'operations': operations,
           'prompt': build_prompt(starting_state, operations),
           'completion': build_decode_back_completion(starting_state, operations)
               .replace('The final code token Nib', 'The final code token Nomo'),
           'final_answer': 'Heads', 'coded': True}
    with pytest.raises(AssertionError):
        verify_decode_back_trajectory(row)


def test_verify_decode_back_trajectory_rejects_missing_decode_back_line():
    from synthetic_bridge import build_coded_completion, build_prompt
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    row = {'starting_state': starting_state, 'operations': operations,
           'prompt': build_prompt(starting_state, operations),
           'completion': build_coded_completion(starting_state, operations),
           'final_answer': 'Heads', 'coded': True}
    with pytest.raises(AssertionError, match='missing or malformed'):
        verify_decode_back_trajectory(row)


def test_rebalanced_decode_back_bank_matches_the_prior_rebalanced_bank_exactly_in_scope():
    # Same 36 scenarios, same 18/18 parity, same 7 removed -- the ONLY difference vs.
    # the prior rebalanced-bank task is the decode-back line's presence.
    bank, eval_rows, removed = build_rebalanced_decode_back_bank(clean21_seed=20260831, n_eval=21)
    from collections import Counter
    dist = Counter(r['final_answer'] for r in bank)
    assert len(bank) == 36
    assert dist == Counter({'Heads': 18, 'Tails': 18})
    assert len(removed) == 7
    assert all(r['final_answer'] == 'Tails' for r in removed)
    assert len(eval_rows) == 21


def test_rebalanced_decode_back_bank_has_no_overlap_with_held_out_eval():
    bank, eval_rows, _ = build_rebalanced_decode_back_bank(clean21_seed=20260831, n_eval=21)
    bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in bank}
    eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
    assert not (bank_keys & eval_keys)


def test_rebalanced_decode_back_bank_passes_full_verification_pipeline():
    bank, _, _ = build_rebalanced_decode_back_bank(clean21_seed=20260831, n_eval=21)
    reports = verify_decode_back_bank(bank)
    assert len(reports) == 36
    assert all(r['verified'] and r['decode_back_line_verified'] for r in reports)


def test_reward_wrapper_zeroes_the_false_positive_p_cot_penalty_from_the_decode_back_line():
    from synthetic_bridge import build_prompt
    from reward_v3 import score_completion_v2
    starting_state, operations = 'Heads', ['same', 'different', 'different', 'same', 'different']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    raw = score_completion_v2(completion, 'Tails', step=40, total_steps=150, prompt=prompt)
    fixed = score_completion_v2_decode_back_aware(completion, 'Tails', step=40, total_steps=150, prompt=prompt)
    assert raw['banned_count'] == 1.0  # the bug: the decode-back line's own word counted as a leak
    assert fixed['banned_count'] == 0.0
    assert fixed['p_cot'] == 0.0
    assert fixed['total'] > raw['total']  # the false-positive penalty was removed, not doubled


def test_reward_wrapper_leaves_every_other_term_unchanged():
    from synthetic_bridge import build_prompt
    from reward_v3 import score_completion_v2
    starting_state, operations = 'Tails', ['different', 'same', 'same', 'different', 'same']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    raw = score_completion_v2(completion, 'Heads', step=100, total_steps=150, prompt=prompt)
    fixed = score_completion_v2_decode_back_aware(completion, 'Heads', step=100, total_steps=150, prompt=prompt)
    for key in ('r_task', 'p_length', 'p_structure', 'p_state_variation', 'r_consistency', 'r_signal'):
        assert raw[key] == fixed[key], key


def test_reward_wrapper_still_penalizes_genuine_literal_leakage_in_the_trace():
    from synthetic_bridge import build_prompt
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_decode_back_completion(starting_state, operations).replace('State: Nib', 'State: Heads', 1)
    prompt = build_prompt(starting_state, operations)
    fixed = score_completion_v2_decode_back_aware(completion, 'Heads', step=100, total_steps=150, prompt=prompt)
    assert fixed['banned_count'] >= 1.0
    assert fixed['p_cot'] > 0.0


def test_rebalanced_decode_back_bank_is_deterministic():
    bank1, _, _ = build_rebalanced_decode_back_bank(clean21_seed=20260831, n_eval=21)
    bank2, _, _ = build_rebalanced_decode_back_bank(clean21_seed=20260831, n_eval=21)
    sha1 = hashlib.sha256(json.dumps(bank1, sort_keys=True).encode()).hexdigest()
    sha2 = hashlib.sha256(json.dumps(bank2, sort_keys=True).encode()).hexdigest()
    assert sha1 == sha2 == 'c21722dd2be3bb600cef1cbbcdf4d13cfc43568aa0f5ce46cfb41d6780f9e4f2'


if __name__ == '__main__':
    import inspect
    import sys
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
