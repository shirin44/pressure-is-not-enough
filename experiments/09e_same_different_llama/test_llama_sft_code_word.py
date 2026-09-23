"""Golden-record regression test against the real Llama-3-8B-Instruct SFT code-word
seeding evidence (aws_runs/stage9e-llama-sft-code-word-v1/, 2026-09-07). Direct
teacher-forced imitation, no RL, no reward, LoRA -- mirrors Stage 9c's Qwen
methodology, with Qwen's settled breaker thresholds added as a provisional,
disclosed-as-unverified safety net (real per-step KL(adapted||base) computed via
disable_adapter(), not faked).

Headline result: dramatically stronger than Stage 9c's original Qwen run (28.6%
held-out accuracy before its own translation-bug fix). Tier (a) training recall and
Tier (b) held-out same-pair both reach 100% -- tier (b)'s stop condition ("if weak,
diagnose before proceeding") is not triggered. Tier (c) held-out different-pair
(never-trained Jub/Kag) reaches 52.4% strict accuracy, but completion-level tracing
(matching this project's established discipline) shows this understates genuine
capability: of the 10 "failures," 5 are pure instruction-following lapses (the model
tracked and answered with perfect internal consistency using its TRAINED Bek/Ner
vocabulary instead of the instructed Jub/Kag pair -- verified state-correct in every
case), and only 5 are genuine tracking/mapping confusion within the correctly-used
instructed pair. Substance-correct rate (right Heads/Tails state regardless of
vocabulary compliance): 16/21 = 76.2%."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-llama-sft-code-word-v1' / 'stage9e_llama_sft_code_word.json').read_text())


def test_training_completed_all_40_steps_no_hard_stop():
    assert EVIDENCE['hard_stop'] is None
    assert len(EVIDENCE['training_telemetry']) == 40
    assert EVIDENCE['training_telemetry'][-1]['step'] == 40


def test_no_qwen_breaker_ever_fired_during_sft():
    for row in EVIDENCE['training_telemetry']:
        assert row['qwen_grad_breaker_would_fire'] is False
        assert row['qwen_kl_breaker_would_fire'] is False


def test_loss_decreased_substantially():
    first = EVIDENCE['training_telemetry'][0]['loss']
    last = EVIDENCE['training_telemetry'][-1]['loss']
    assert last < first * 0.01  # loss dropped by more than 100x


def test_overfit_watch_shows_genuine_progressive_learning_not_instant_memorization():
    watch = EVIDENCE['overfit_watch']
    assert len(watch) == 4  # steps 10, 20, 30, 40
    accs = [w['final_answer_accuracy'] for w in watch]
    assert accs[0] < 0.3  # not memorized immediately
    assert accs[-1] == 1.0  # converges by the end
    assert accs == sorted(accs)  # monotonically non-decreasing


def test_tier_a_training_recall_is_perfect():
    s = EVIDENCE['tier_a_training_set_recall']['summary']
    assert s['n'] == 43
    assert s['intermediate_tracking_accuracy'] == 1.0
    assert s['final_answer_accuracy'] == 1.0
    assert s['answer_is_single_token_rate'] == 1.0
    assert s['split_or_malformed_answer_token_count'] == 0


def test_tier_b_heldout_same_pair_is_perfect_stop_condition_not_triggered():
    s = EVIDENCE['tier_b_heldout_same_pair']['summary']
    assert s['n'] == 21
    assert s['intermediate_tracking_accuracy'] == 1.0
    assert s['final_answer_accuracy'] == 1.0
    assert s['answer_is_single_token_rate'] == 1.0
    assert s['split_or_malformed_answer_token_count'] == 0
    # far above any reasonable "weak" threshold -- the task's own diagnose-before-
    # proceeding condition for tier (b) is not triggered
    assert s['final_answer_accuracy'] > 0.65


def test_tier_c_strict_accuracy_matches_measured_value():
    s = EVIDENCE['tier_c_heldout_different_pair']['summary']
    assert s['n'] == 21
    assert abs(s['fully_correct_rate'] - 11 / 21) < 1e-9
    assert abs(s['final_answer_correct_rate'] - 11 / 21) < 1e-9


def test_tier_c_failures_decompose_into_instruction_lapse_vs_genuine_tracking_error():
    samples = EVIDENCE['tier_c_heldout_different_pair']['samples']
    n_fully_correct = 0
    n_lapsed_to_training_pair = 0
    n_stayed_in_pair_but_wrong = 0
    for r in samples:
        s = r['score']
        actual_final = s['actual_tokens'][-1] if s['actual_tokens'] else None
        if s['fully_correct']:
            n_fully_correct += 1
        elif actual_final in ('bek', 'ner'):
            n_lapsed_to_training_pair += 1
        elif s['stayed_within_instructed_pair']:
            n_stayed_in_pair_but_wrong += 1
    assert n_fully_correct == 11
    assert n_lapsed_to_training_pair == 5
    assert n_stayed_in_pair_but_wrong == 5


def test_every_instruction_lapse_case_was_internally_state_correct():
    # The key qualifying finding: all 5 "lapsed to training pair" cases used Bek/Ner
    # from step 1 (not a late slip), and the training-pair word they used always
    # correctly implied the true Heads/Tails state -- a pure vocabulary-compliance
    # lapse, not a tracking failure.
    samples = EVIDENCE['tier_c_heldout_different_pair']['samples']
    n_checked = 0
    for r in samples:
        s = r['score']
        actual_final = s['actual_tokens'][-1] if s['actual_tokens'] else None
        if actual_final in ('bek', 'ner') and not s['fully_correct']:
            n_checked += 1
            true_final_state = 'Heads' if s['expected_tokens'][-1] == 'jub' else 'Tails'
            implied_state = 'Heads' if actual_final == 'bek' else 'Tails'
            assert implied_state == true_final_state
            # used the training pair from the very first tracked step, not just the answer
            assert all(t in ('bek', 'ner') for t in s['actual_tokens'])
    assert n_checked == 5


def test_substance_correct_rate_when_setting_aside_vocabulary_compliance():
    samples = EVIDENCE['tier_c_heldout_different_pair']['samples']
    n_substance_correct = 0
    for r in samples:
        s = r['score']
        actual_final = s['actual_tokens'][-1] if s['actual_tokens'] else None
        if s['fully_correct']:
            n_substance_correct += 1
        elif actual_final in ('bek', 'ner'):
            true_final_state = 'Heads' if s['expected_tokens'][-1] == 'jub' else 'Tails'
            implied_state = 'Heads' if actual_final == 'bek' else 'Tails'
            if implied_state == true_final_state:
                n_substance_correct += 1
    assert n_substance_correct == 16  # 16/21 = 76.2%


def test_zero_lapses_into_literal_heads_tails_in_tier_c():
    s = EVIDENCE['tier_c_heldout_different_pair']['summary']
    assert s['lapsed_into_literal_rate'] == 0.0


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
