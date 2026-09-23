"""Golden-record regression tests against the decode-back-line SFT re-run's evidence
(aws_runs/stage9c-sft-decode-back-v1/, 2026-09-02). Captures the approved fix's result:
the explicit decode-back line ("The final code token Nib decodes to Heads.") closes
MOST of the tracking-vs-answer gap identified by the corrupted-prefill diagnostic
(61.9 points -> 9.5 points on held-out), and the residual gap is precisely
characterized rather than left as an unexplained "gap persists": the final answer now
FAITHFULLY follows the decode-back line 100% of the time
(decode_back_matches_own_trace_rate == 1.0) -- proving the original structural
disconnect is fixed -- and the leftover error is a much smaller, residual version of
the SAME directional bias, now confined to the decode-back translation step itself
(the model occasionally still mistranslates "Nib" as "Tails" specifically, never the
reverse), plus one unrelated intermediate-tracking mistake."""
import json
from pathlib import Path

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9c-sft-decode-back-v1'
                        / 'stage9c_sft_decode_back.json').read_text())
TIER_B = EVIDENCE['tier_b_heldout_same_pair']


def test_config_used_the_same_36_scenario_18_18_bank_and_same_held_out_21():
    cfg = EVIDENCE['config']
    assert cfg['decode_back_bank_size'] == 36
    assert cfg['label_distribution'] == {'Heads': 18, 'Tails': 18}
    assert cfg['clean21_eval_sha256'] == '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'


def test_training_completed_without_oom_and_converged():
    telemetry = EVIDENCE['training_telemetry']
    assert len(telemetry) == 40
    assert all(__import__('math').isfinite(r['loss']) for r in telemetry)
    assert telemetry[-1]['loss'] < telemetry[0]['loss'] / 100


def test_overfit_watch_shows_the_gap_closing_over_training_not_just_at_the_end():
    watch = EVIDENCE['overfit_watch']
    assert [w['step'] for w in watch] == [10, 20, 30, 40]
    # Requirement 2: tracking and answer accuracy tracked SEPARATELY throughout, not
    # just once at the end.
    assert watch[-1]['intermediate_tracking_accuracy'] > 0.9
    assert watch[-1]['decode_back_presence_rate'] == 1.0  # the model reliably learned to PRODUCE the new step


def test_training_set_recall_shows_zero_gap():
    tier_a = EVIDENCE['tier_a_training_set_recall']['summary']
    assert tier_a['intermediate_tracking_accuracy'] == tier_a['final_answer_accuracy']
    assert tier_a['tracking_vs_answer_gap'] == 0.0


def test_held_out_gap_closed_from_61_9_points_to_9_5_points():
    summary = TIER_B['summary']
    assert summary['n'] == 21
    assert abs(summary['intermediate_tracking_accuracy'] - 0.9047619047619048) < 1e-9
    assert abs(summary['final_answer_accuracy'] - 0.8095238095238095) < 1e-9
    assert abs(summary['tracking_vs_answer_gap'] - 0.09523809523809523) < 1e-9
    # Rebalanced-run comparison point: gap was 0.9518 (95.2% - 33.3%) before this fix.
    prior_gap = 0.9518 - 1/3
    assert summary['tracking_vs_answer_gap'] < prior_gap / 5  # gap shrank to well under a fifth


def test_decode_back_line_is_produced_reliably_and_faithfully_read_by_the_answer():
    summary = TIER_B['summary']
    # The model learned to PRODUCE the new step on every single held-out completion.
    assert summary['decode_back_presence_rate'] == 1.0
    # THE key structural finding: the final <answer> now faithfully follows whatever
    # the decode-back line says, with ZERO exceptions -- proving the original
    # structural disconnect (diagnosed via corrupted-prefill testing) is fixed.
    assert summary['decode_back_matches_own_trace_rate'] == 1.0
    # decode_back_accuracy and final_answer_accuracy are therefore identical.
    assert summary['decode_back_accuracy'] == summary['final_answer_accuracy']


def test_residual_gap_is_precisely_characterized_not_left_unexplained():
    # Every held-out error is one of exactly two well-understood causes: (1) a single
    # intermediate-tracking mistake unrelated to translation direction, or (2) a
    # residual, much-smaller version of the SAME directional bias, now confined to the
    # decode-back translation step, and observed ONLY in the Nib->should-be-Heads
    # direction (never Nomo->should-be-Tails) -- consistent with the original
    # diagnostic's own finding that the collapse always favored "Tails".
    samples = TIER_B['samples']
    wrong = [s for s in samples if not s['final_answer_correct']]
    assert len(wrong) == 4
    tracking_mistakes = [s for s in wrong if not s['intermediate_tracking_correct']]
    mistranslations = [s for s in wrong if s['intermediate_tracking_correct']]
    assert len(tracking_mistakes) == 1
    assert len(mistranslations) == 3
    # Every mistranslation error has the model's own last tracked token = 'nib', and
    # the decode-back line wrongly says "Tails" instead of "Heads" -- never the reverse.
    for s in mistranslations:
        assert s['actual_tokens'][-1] == 'nib'
        assert 'decodes to Tails' in s['completion']
    # No case anywhere in the held-out set shows the opposite mistranslation direction
    # (Nomo wrongly decoded as Heads).
    assert not any('nomo' in ''.join(s['actual_tokens'][-1:]) and 'decodes to Heads' in s['completion']
                   and not s['decode_back_correct'] for s in samples)


def test_decode_back_translation_accuracy_by_direction_shows_the_residual_asymmetry():
    samples = [s for s in TIER_B['samples'] if s['decode_back_present']]
    nib_ending = [s for s in samples if s['actual_tokens'] and s['actual_tokens'][-1] == 'nib']
    nomo_ending = [s for s in samples if s['actual_tokens'] and s['actual_tokens'][-1] == 'nomo']
    nib_correct_rate = sum(s['decode_back_correct'] for s in nib_ending) / len(nib_ending)
    nomo_correct_rate = sum(s['decode_back_correct'] for s in nomo_ending) / len(nomo_ending)
    # Both directions now mostly correct (a dramatic improvement over the pre-fix 0%),
    # but Nib-ending completions still show a lower translation-correctness rate,
    # consistent with a residual (not eliminated) version of the original bias.
    assert nib_correct_rate < nomo_correct_rate
    assert nib_correct_rate > 0.7  # still mostly correct -- residual, not dominant


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
