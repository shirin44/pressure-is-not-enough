"""Stage 9e, RL reward re-derivation for the direct code-word answer design (task
item 1). Confirms the CoT penalty (literal-word ban) and correctness reward interact
sensibly with NO scaffolding-line carve-out needed -- unlike Stage 9d's decode-back
design (which needed score_completion_v2_decode_back_aware to exempt its own
decode-back sentence from the CoT scan), this design has no such line at all, so
PLAIN score_completion_v2 is used unchanged. Confirms this concretely: a clean
code-word completion (mentions Bek/Ner only, never Heads/Tails) gets ZERO CoT
penalty; a completion that leaks a literal word IS penalized, exactly as intended."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from reward_v3 import score_completion_v2, verify_reward_invariant  # noqa: E402

from code_word_answer_bank import CODE_FOR, HEADS_CODE, build_code_word_completion

MAIN_REWARD_PARAMS = {}  # score_completion_v2 defaults: cot_max_scale=2.0, consistency/signal=0.15
BASELINE_REWARD_PARAMS = dict(cot_min_scale=0.0, cot_max_scale=0.0, consistency_magnitude=0.0, signal_magnitude=0.0)


def test_symbolic_invariant_holds_for_both_conditions():
    main_margins = verify_reward_invariant(**{k: v for k, v in MAIN_REWARD_PARAMS.items()
                                               if k in ('consistency_magnitude', 'signal_magnitude', 'cot_max_scale')})
    baseline_margins = verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0, cot_max_scale=0.0)
    assert main_margins['margin_correct_over_wrong'] > 0
    assert baseline_margins['margin_correct_over_wrong'] > 0


def test_clean_code_word_completion_gets_zero_cot_penalty_under_main():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    ground_truth = CODE_FOR['Heads']
    breakdown = score_completion_v2(completion, ground_truth, 150, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    assert breakdown['p_cot'] == 0.0
    assert breakdown['r_task'] == 4.0


def test_leaking_literal_heads_in_the_trace_is_penalized_under_main():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    leaked = completion.replace('Step 1: The state is tracked. State: Bek',
                                 'Step 1: The state is tracked, it is Heads. State: Bek')
    ground_truth = CODE_FOR['Heads']
    clean_breakdown = score_completion_v2(completion, ground_truth, 150, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    leaked_breakdown = score_completion_v2(leaked, ground_truth, 150, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    assert leaked_breakdown['p_cot'] > 0.0
    assert leaked_breakdown['total'] < clean_breakdown['total']


def test_baseline_condition_applies_zero_cot_penalty_even_when_leaked():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    leaked = completion.replace('Step 1: The state is tracked. State: Bek',
                                 'Step 1: The state is tracked, it is Heads. State: Bek')
    ground_truth = CODE_FOR['Heads']
    breakdown = score_completion_v2(leaked, ground_truth, 150, 150, prompt='dummy', **BASELINE_REWARD_PARAMS)
    assert breakdown['p_cot'] == 0.0


def test_main_at_full_anneal_penalizes_more_than_early_warmup():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    leaked = completion.replace('Step 1: The state is tracked. State: Bek',
                                 'Step 1: it is Heads not Tails. State: Bek')
    ground_truth = CODE_FOR['Heads']
    early = score_completion_v2(leaked, ground_truth, 1, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    late = score_completion_v2(leaked, ground_truth, 150, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    assert late['p_cot'] > early['p_cot']


def test_correctness_still_dominates_a_single_leaked_word_at_full_anneal():
    # A single leaked word should not flip a correct answer into a net-negative
    # reward under full pressure -- confirms the invariant's margin is real in
    # practice for this design, not just in the abstract combinatorial check.
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    leaked = completion.replace('Step 1: The state is tracked. State: Bek',
                                 'Step 1: it is Heads. State: Bek')
    ground_truth = CODE_FOR['Heads']
    breakdown = score_completion_v2(leaked, ground_truth, 150, 150, prompt='dummy', **MAIN_REWARD_PARAMS)
    assert breakdown['total'] > 0  # still net positive: correct answer, one leaked word


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
