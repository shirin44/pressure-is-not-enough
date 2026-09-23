"""Golden-record regression tests against the Stage 9d decode-back-seeded
PENALTY-ONLY run (aws_runs/stage9d-decode-back-penalty_only-v1/, 2026-09-03): the
exact mirror image of SIGNAL-ONLY -- correctness + p_cot at MAIN's own defaults, with
r_consistency and r_signal fully zeroed. Completes the 2x2 factorial (bonus: yes/no x
penalty: yes/no) across all four conditions. Captures the quantified answer: PENALTY-
ONLY (0.778) falls between BASELINE (0.698) and SIGNAL-ONLY (0.865), significantly
above BASELINE but significantly below both SIGNAL-ONLY and MAIN -- the penalty alone
does real, statistically significant work, but roughly half as much as the bonus
terms alone. Also captures that Stage 1's old failure mode (literal-word reversion /
vacuous output under an unguided penalty) did NOT reappear: nonliteral_rate stayed
perfect throughout, and the taxonomy shows no degenerate-output signature."""
import json
import statistics
from pathlib import Path

BASELINE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())
SIGNAL_ONLY = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-signal_only-v1'
                           / 'stage9d_decode_back_signal_only.json').read_text())
PENALTY_ONLY = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-penalty_only-v1'
                            / 'stage9d_decode_back_penalty_only.json').read_text())
MAIN = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v2'
                    / 'stage9d_decode_back_main.json').read_text())


def _matched(*runs):
    ms_list = [{r['step']: r['genuine_correct_among_structural_nonliteral'] for r in d['result']['milestones']} for d in runs]
    steps = sorted(set.intersection(*[set(m) for m in ms_list]))
    return steps, [[m[s] for s in steps] for m in ms_list]


def test_config_zeroed_bonus_terms_and_kept_p_cot_at_main_defaults():
    cfg = PENALTY_ONLY['config']
    assert 'r_consistency/r_signal fully zeroed' in cfg['reward_params']
    assert 'cot_min_scale=0.2, cot_max_scale=2.0' in cfg['reward_params']


def test_ran_the_full_150_steps_with_no_breaker():
    result = PENALTY_ONLY['result']
    assert result['terminal_step'] == 150
    assert result['hard_stop'] is None


def test_penalty_only_falls_between_baseline_and_signal_only():
    steps, (b_vals, po_vals, so_vals) = _matched(BASELINE, PENALTY_ONLY, SIGNAL_ONLY)
    assert len(steps) == 30
    b_mean, po_mean, so_mean = statistics.mean(b_vals), statistics.mean(po_vals), statistics.mean(so_vals)
    assert b_mean < po_mean < so_mean


def test_penalty_only_vs_baseline_is_a_real_significant_improvement():
    steps, (po_vals, b_vals) = _matched(PENALTY_ONLY, BASELINE)[0], _matched(PENALTY_ONLY, BASELINE)[1]
    diffs = [p - b for p, b in zip(po_vals, b_vals)]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    assert t_stat > 3.0  # decisively positive, not noise


def test_penalty_only_is_significantly_weaker_than_both_signal_only_and_main():
    _, (po_vals, so_vals) = _matched(PENALTY_ONLY, SIGNAL_ONLY)
    _, (po_vals2, m_vals) = _matched(PENALTY_ONLY, MAIN)
    for a_vals, b_vals in ((po_vals, so_vals), (po_vals2, m_vals)):
        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        n = len(diffs)
        t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
        assert t_stat < -3.0  # decisively negative in both comparisons


def test_penalty_effect_is_smaller_than_bonus_effect_but_not_negligible():
    b_mean = statistics.mean(_matched(BASELINE, PENALTY_ONLY, SIGNAL_ONLY)[1][0])
    po_mean = statistics.mean(_matched(BASELINE, PENALTY_ONLY, SIGNAL_ONLY)[1][1])
    so_mean = statistics.mean(_matched(BASELINE, PENALTY_ONLY, SIGNAL_ONLY)[1][2])
    penalty_gain = po_mean - b_mean
    bonus_gain = so_mean - b_mean
    assert 0 < penalty_gain < bonus_gain
    assert penalty_gain > bonus_gain * 0.3  # a real, non-negligible fraction, not near-zero


def test_no_stage1_failure_signature_nonliteral_rate_stays_perfect_throughout():
    milestones = PENALTY_ONLY['result']['milestones']
    assert all(m['nonliteral_rate'] == 1.0 for m in milestones)


def test_no_stage1_failure_signature_taxonomy_shows_no_degenerate_output_pattern():
    from collections import Counter
    tax_total = Counter()
    for m in PENALTY_ONLY['result']['milestones']:
        tax_total.update(m['taxonomy_counts'])
    # 'correct_globally_consistent_code' must dominate; no meaningful presence of a
    # degenerate/vacuous-output category.
    total = sum(tax_total.values())
    assert tax_total['correct_globally_consistent_code'] / total > 0.7


def test_decode_back_fidelity_has_exactly_one_anomalous_milestone_immediately_recovered():
    milestones = PENALTY_ONLY['result']['milestones']
    below_one = [(m['step'], m['decode_back_matches_own_trace_rate']) for m in milestones
                 if m['decode_back_matches_own_trace_rate'] < 1.0]
    assert below_one == [(75, 0.9523809523809523)]
    step80 = next(m for m in milestones if m['step'] == 80)
    assert step80['decode_back_matches_own_trace_rate'] == 1.0


def test_actual_training_reward_had_zero_bonus_terms_throughout_the_real_run():
    rows = PENALTY_ONLY['result']['row_level_long_completion_log']
    assert len(rows) > 0
    assert all(r['reward_breakdown']['r_consistency'] == 0.0 for r in rows)
    assert all(r['reward_breakdown']['r_signal'] == 0.0 for r in rows)
    # p_cot IS genuinely active in at least some rows (unlike SIGNAL_ONLY, where it's
    # always exactly 0) -- confirms the mirror-image design landed correctly.
    assert any(r['reward_breakdown']['p_cot'] > 0.0 for r in rows)


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
