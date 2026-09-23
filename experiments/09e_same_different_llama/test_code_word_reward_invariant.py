"""Stage 9e, reward invariant re-derivation for the direct code-word answer format
(task item 7). Same pattern already established for the (abandoned) Same/Different
task: verify_reward_invariant() is symbolic and task-independent (trivially
unchanged); what needs concrete re-verification is whether score_completion_v2's
PRACTICAL r_task computation correctly classifies correct/wrong/malformed completions
when the ground_truth passed in is the CODE WORD itself (e.g. 'Bek') rather than the
literal word ('Heads') -- confirmed here directly, not assumed.

The scoring call site's own responsibility (not reward_v3.py's) is to translate the
row's true final_answer_state into the corresponding code word BEFORE calling
score_completion_v2 -- e.g. ground_truth = CODE_FOR[row['final_answer_state']] -- so
that score_completion_v2 itself never needs to know about the Heads/Tails <-> Bek/Ner
mapping at all. Confirmed this translation-before-scoring pattern works correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from reward_v3 import score_completion_v2, verify_reward_invariant  # noqa: E402

from code_word_answer_bank import CODE_FOR, HEADS_CODE, TAILS_CODE, build_code_word_completion


def test_symbolic_invariant_still_holds_unchanged():
    margins = verify_reward_invariant()
    assert margins['margin_correct_over_wrong'] > 0
    assert margins['margin_wrong_over_malformed'] > 0


def test_r_task_scores_a_correct_code_word_completion_as_4():
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    ground_truth = CODE_FOR['Heads']  # translate BEFORE scoring, per this module's own design
    breakdown = score_completion_v2(completion, ground_truth, 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == 4.0


def test_r_task_scores_a_wrong_code_word_completion_as_negative_half():
    # Completion says Bek (Heads), but ground truth for this scenario is Tails (Ner).
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_code_word_completion(starting_state, operations)
    wrong_ground_truth = CODE_FOR['Tails']
    breakdown = score_completion_v2(completion, wrong_ground_truth, 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == -0.5


def test_r_task_scores_a_malformed_completion_as_negative_5():
    completion = 'Step 1: The state is tracked. State: Bek\nNo answer tags at all.'
    breakdown = score_completion_v2(completion, HEADS_CODE, 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == -5.0


def test_a_literal_word_answer_is_scored_as_wrong_not_correct_even_if_state_correct():
    # If the model ever DID slip and answer with the literal word instead of the
    # code word, this must NOT be scored as correct -- confirms the reward function
    # enforces the code-word convention, not just "any correct-looking answer."
    completion = 'Step 1: The state is tracked. State: Bek\n<answer> Heads</answer>'
    breakdown = score_completion_v2(completion, HEADS_CODE, 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == -0.5  # 'heads' != 'bek' (normalized), scored wrong not correct


def test_case_and_whitespace_variants_of_the_code_word_still_score_correctly():
    for variant in ('bek', 'BEK', ' Bek ', 'Bek.'):
        completion = f'Step 1: The state is tracked. State: Bek\n<answer> {variant}</answer>'
        breakdown = score_completion_v2(completion, HEADS_CODE, 1, 150, prompt='dummy prompt')
        assert breakdown['r_task'] == 4.0, variant


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
