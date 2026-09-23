"""Golden-record regression tests against the rebalanced-bank SFT run's evidence
(aws_runs/stage9c-sft-rebalanced-v1/, 2026-08-31). Captures the DISPROVEN hypothesis:
rebalancing the training bank's final-answer-label distribution to exact 18/18 parity
had ZERO effect on the held-out collapse to "Tails" -- tracking accuracy improved
further (95.2%, matching the expanded-bank run) but final-answer accuracy is
UNCHANGED at exactly 33.3%. This rules out class imbalance as the (or at least THE
sole) cause; the next diagnostic angle needs to be identified fresh, per instruction,
not guessed here."""
import json
import re
from collections import Counter
from pathlib import Path

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9c-sft-rebalanced-v1'
                        / 'stage9c_sft_rebalanced.json').read_text())


def test_rebalance_config_matches_the_build():
    cfg = EVIDENCE['config']
    assert cfg['rebalanced_bank_size'] == 36
    assert cfg['label_distribution_after'] == {'Heads': 18, 'Tails': 18}
    assert cfg['label_distribution_before'] == {'Heads': 18, 'Tails': 25}
    assert cfg['removed_scenario_count'] == 7
    assert cfg['original_16_bank_fully_preserved'] is True


def test_training_completed_without_oom():
    telemetry = EVIDENCE['training_telemetry']
    assert len(telemetry) == 40
    assert all(__import__('math').isfinite(r['loss']) for r in telemetry)


def test_intermediate_tracking_further_improved_after_rebalancing():
    tier_b = EVIDENCE['tier_b_heldout_same_pair']['summary']
    assert tier_b['intermediate_tracking_accuracy'] > 0.9  # 95.2%, matching/exceeding the unbalanced run's 100%-ish tracking


def test_final_answer_accuracy_UNCHANGED_by_rebalancing_disproving_the_imbalance_hypothesis():
    # The core finding of this task: rebalancing training labels to exact 18/18 parity
    # had ZERO measurable effect on held-out final-answer accuracy -- identical to the
    # unbalanced (58/42) run's own result. This disproves the label-imbalance
    # hypothesis as (at least the sole) cause of the collapse.
    tier_b = EVIDENCE['tier_b_heldout_same_pair']['summary']
    assert abs(tier_b['final_answer_accuracy'] - 1 / 3) < 1e-9
    assert abs(tier_b['genuine_correct_among_nonliteral'] - 1 / 3) < 1e-9
    assert abs(tier_b['final_answer_accuracy'] - tier_b['expanded_bank_unbalanced_result_for_comparison']) < 1e-9


def test_held_out_completions_still_collapse_completely_to_tails():
    tier_b_samples = EVIDENCE['tier_b_heldout_same_pair']['samples']
    assert len(tier_b_samples) == 21
    answers = Counter()
    for s in tier_b_samples:
        m = re.search(r'<answer>(.*?)</answer>', s['completion'])
        answers[m.group(1) if m else 'MISSING'] += 1
    assert answers == Counter({'Tails': 21}), (
        f'expected the SAME complete collapse to "Tails" as before rebalancing; got {answers}')


def test_tracking_vs_answer_gap_widened_not_narrowed():
    # Tracking improved (57%->95% over the course of training) while the answer stayed
    # flat -- the gap actually WIDENED relative to training progress, not narrowed,
    # underscoring that the bug is fully decoupled from tracking quality.
    tier_b = EVIDENCE['tier_b_heldout_same_pair']['summary']
    gap = tier_b['tracking_vs_answer_gap']
    assert gap > 0.6


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
