"""CPU-only tests for the causal-flip corruption/counterfactual logic, run before
any GPU work touches it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from synthetic_bridge import _trace  # noqa: E402

from code_word_answer_bank import build_code_word_completion, build_code_word_prompt
from code_word_causal_flip import (
    build_mid_sequence_corrupted_prefix, classify_continuation, counterfactual_trace,
)


def test_counterfactual_matches_true_trace_before_the_flip_point():
    starting_state, operations = 'Heads', ['different', 'same', 'different', 'same', 'different']
    true_states = _trace(starting_state, operations)
    cf = counterfactual_trace(starting_state, operations, flip_at_step=3)
    assert cf[:2] == true_states[:2]  # unchanged before the flip


def test_counterfactual_is_flipped_at_the_intervention_point():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    true_states = _trace(starting_state, operations)
    cf = counterfactual_trace(starting_state, operations, flip_at_step=2)
    assert cf[1] != true_states[1]
    assert cf[1] == ('Tails' if true_states[1] == 'Heads' else 'Heads')


def test_counterfactual_propagates_remaining_operations_from_the_flip():
    # All-same sequence: once flipped, every subsequent state should STAY at the
    # flipped value (since 'same' never changes it) -- a clean, hand-verifiable case.
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    cf = counterfactual_trace(starting_state, operations, flip_at_step=2)
    assert cf[1:] == [cf[1]] * 4  # steps 2-5 all equal the flipped value


def test_counterfactual_final_differs_from_true_final_when_flip_survives_to_the_end():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    true_states = _trace(starting_state, operations)
    cf = counterfactual_trace(starting_state, operations, flip_at_step=2)
    assert cf[-1] != true_states[-1]


def test_build_mid_sequence_corrupted_prefix_flips_the_correct_line_only():
    starting_state, operations = 'Tails', ['different', 'same', 'different', 'same', 'different']
    completion = build_code_word_completion(starting_state, list(operations))
    prompt = build_code_word_prompt(starting_state, list(operations))
    result = build_mid_sequence_corrupted_prefix(prompt, completion, flip_at_step=2)
    assert result['ok'] is True
    assert result['flipped_token'] != result['original_token']
    # step 1's line must be UNCHANGED in the corrupted prefix
    step1_line = completion.split('\n')[0]
    assert step1_line in result['corrupted_prefix']
    # nothing after step 2's line should survive (steps 3-5 gone). Note: the PROMPT's
    # own instructions legitimately mention '<answer>...</answer>' as a formatting
    # example, so checking the full prefix for that substring would be a false
    # positive -- check the COMPLETION portion specifically instead.
    assert 'Step 3' not in result['corrupted_prefix']
    completion_portion = result['corrupted_prefix'].split(step1_line, 1)[1]
    assert '<answer>' not in completion_portion


def test_classify_continuation_detects_genuine_propagation():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    cf_states = counterfactual_trace(starting_state, operations, flip_at_step=2)
    # Build a synthetic "continuation" that a genuinely-causal model would produce:
    # steps 3-5 all show the flipped code, and the answer matches the flipped final state.
    code_for = {'Heads': 'Bek', 'Tails': 'Ner'}
    lines = [f'Step {i}: The code remains unchanged. State: {code_for[cf_states[i-1]]}' for i in range(3, 6)]
    continuation = '\n'.join(lines) + f'\n<answer> {code_for[cf_states[-1]]}</answer>'
    result = classify_continuation(starting_state, operations, 2, continuation)
    assert result['propagation_matches_counterfactual'] is True
    assert result['final_answer_classification'] == 'tracks_flip_genuine_causal_use'


def test_classify_continuation_detects_disconnect_ignoring_the_flip():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    true_states = _trace(starting_state, operations)
    code_for = {'Heads': 'Bek', 'Tails': 'Ner'}
    # A model that IGNORES the corruption and continues as if uncorrupted.
    lines = [f'Step {i}: The code remains unchanged. State: {code_for[true_states[i-1]]}' for i in range(3, 6)]
    continuation = '\n'.join(lines) + f'\n<answer> {code_for[true_states[-1]]}</answer>'
    result = classify_continuation(starting_state, operations, 2, continuation)
    assert result['propagation_matches_original_ignoring_flip'] is True
    assert result['final_answer_classification'] == 'ignores_flip_stays_at_original'


def test_flip_at_step_two_changes_the_predicted_final_answer_for_all_same_sequence():
    # Sanity: for an all-'same' sequence, flipping any step always changes the final
    # answer (since nothing flips it back) -- confirms this intervention point is a
    # meaningful, decisive test, not a no-op.
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    result_via_helper = counterfactual_trace(starting_state, operations, 2)
    true_states = _trace(starting_state, operations)
    assert result_via_helper[-1] != true_states[-1]


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
