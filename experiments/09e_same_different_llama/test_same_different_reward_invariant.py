"""Stage 9e, reward invariant re-derivation for the Same/Different task (task item 6).

verify_reward_invariant() itself is entirely SYMBOLIC -- a combinatorial check over
the auxiliary-term extremes and the fixed r_task constants (correct=4.0, wrong=-0.5,
malformed=-5.0), independent of model, tokenizer, or answer vocabulary. Re-running it
is trivially unchanged (same margins) regardless of task. The actual thing that needs
re-verification, per the task's own framing ("correctness is now Same/Different-based,
not Heads/Tails-based -- re-verify rather than assume identical"), is whether
score_completion_v2's PRACTICAL r_task computation still correctly classifies
correct/wrong/malformed completions when ground_truth is 'Same'/'Different' instead of
'Heads'/'Tails' -- verified here concretely, not just assumed from the symbolic proof."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from reward_v3 import score_completion_v2, verify_reward_invariant  # noqa: E402


def test_symbolic_invariant_still_holds_unchanged():
    # Trivially unchanged (task-independent), but re-run per the task's instruction
    # to "re-derive fresh," not silently assumed from Qwen's own prior run.
    margins = verify_reward_invariant()
    assert margins['margin_correct_over_wrong'] > 0
    assert margins['margin_wrong_over_malformed'] > 0


def test_r_task_scores_a_correct_same_different_completion_as_4():
    completion = 'Step 1: The state is tracked. State: Bek\n<answer>Same</answer>'
    breakdown = score_completion_v2(completion, 'Same', 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == 4.0


def test_r_task_scores_a_wrong_same_different_completion_as_negative_half():
    completion = 'Step 1: The state is tracked. State: Bek\n<answer>Different</answer>'
    breakdown = score_completion_v2(completion, 'Same', 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == -0.5


def test_r_task_scores_a_malformed_completion_as_negative_5_regardless_of_task():
    completion = 'Step 1: The state is tracked. State: Bek\nNo answer tags here at all.'
    breakdown = score_completion_v2(completion, 'Same', 1, 150, prompt='dummy prompt')
    assert breakdown['r_task'] == -5.0


def test_case_and_whitespace_variants_of_same_different_still_score_correctly():
    # normalize_state_token casefolds and strips -- confirmed concretely for the new
    # answer vocabulary, not assumed to transfer from Heads/Tails.
    for variant in ('same', 'SAME', ' Same ', 'Same.'):
        completion = f'Step 1: The state is tracked. State: Bek\n<answer>{variant}</answer>'
        breakdown = score_completion_v2(completion, 'Same', 1, 150, prompt='dummy prompt')
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
