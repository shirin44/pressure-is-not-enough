"""The one sanity check this experiment's whole premise depends on: does reward_v3's lexical
penalty actually produce nonzero pressure against a completion that uses literal Heads/Tails
(the exact behavior checkpoint 500 produces 100% of the time, per
experiments/01_direct_rl_coinflip/aws_recovered/checkpoint500_reeval.json)? In every prior
stage this session audited, the starting policy's completions structurally could not trigger
the penalty (code-word designs, or a checkpoint that had already gone non-literal/vacuous).
This is the first design where the penalty has real, mechanical opportunity to bind from
step 0 -- confirmed here by direct execution, not assumed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from reward_v3 import score_completion_v2, verify_reward_invariant  # noqa: E402

from literal_cot_bank import build_conversational_prompt, unique_pool

MAIN_REWARD_PARAMS = {}
BASELINE_REWARD_PARAMS = dict(cot_min_scale=0.0, cot_max_scale=0.0, consistency_magnitude=0.0, signal_magnitude=0.0)


def _row():
    return unique_pool(1, 20261001)[0]


def test_symbolic_invariant_holds_for_both_conditions():
    main_margins = verify_reward_invariant()
    baseline_margins = verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0, cot_max_scale=0.0)
    assert main_margins['margin_correct_over_wrong'] > 0
    assert baseline_margins['margin_correct_over_wrong'] > 0


def test_literal_completion_gets_NONZERO_cot_penalty_under_main_at_full_anneal():
    row = _row()
    literal_completion = f"Step 1: The state remains unchanged. State: {row['ground_truth']}\n<answer>{row['ground_truth']}</answer>"
    b = score_completion_v2(literal_completion, row['ground_truth'], 150, 150, prompt=row['prompt'], **MAIN_REWARD_PARAMS)
    assert b['banned_count'] > 0, 'a literal Heads/Tails completion must trip the banned-pattern scanner'
    assert b['p_cot'] > 0, 'the whole premise of this experiment requires p_cot > 0 for checkpoint 500-style output'


def test_literal_completion_gets_ZERO_cot_penalty_under_baseline():
    row = _row()
    literal_completion = f"Step 1: The state remains unchanged. State: {row['ground_truth']}\n<answer>{row['ground_truth']}</answer>"
    b = score_completion_v2(literal_completion, row['ground_truth'], 150, 150, prompt=row['prompt'], **BASELINE_REWARD_PARAMS)
    assert b['p_cot'] == 0.0


def test_penalty_grows_with_more_literal_words_and_with_anneal_progress():
    row = _row()
    short = f"State: {row['ground_truth']}\n<answer>{row['ground_truth']}</answer>"
    long_literal = f"Step 1: it is {row['ground_truth']}, definitely {row['ground_truth']}, not the other one. State: {row['ground_truth']}\n<answer>{row['ground_truth']}</answer>"
    b_short = score_completion_v2(short, row['ground_truth'], 150, 150, prompt=row['prompt'])
    b_long = score_completion_v2(long_literal, row['ground_truth'], 150, 150, prompt=row['prompt'])
    assert b_long['banned_count'] > b_short['banned_count']
    assert b_long['p_cot'] >= b_short['p_cot']
    early = score_completion_v2(long_literal, row['ground_truth'], 1, 150, prompt=row['prompt'])
    late = score_completion_v2(long_literal, row['ground_truth'], 150, 150, prompt=row['prompt'])
    assert late['p_cot'] > early['p_cot']


def test_correctness_still_dominates_at_full_anneal_for_a_realistic_checkpoint500_style_completion():
    row = _row()
    literal_completion = f"Step 1: The state remains unchanged. State: {row['ground_truth']}\n<answer>{row['ground_truth']}</answer>"
    b = score_completion_v2(literal_completion, row['ground_truth'], 150, 150, prompt=row['prompt'])
    assert b['total'] > 0, 'a single correct literal completion should still net positive even at full anneal pressure'


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
