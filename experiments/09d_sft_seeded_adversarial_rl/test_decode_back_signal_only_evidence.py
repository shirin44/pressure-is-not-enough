"""Golden-record regression tests against the Stage 9d decode-back-seeded SIGNAL-ONLY
run (aws_runs/stage9d-decode-back-signal_only-v1/, 2026-09-02): correctness +
r_consistency + r_signal with p_cot fully zeroed, isolating whether MAIN's decisive
win over BASELINE (0.906 vs 0.698, t=10.85) is attributable to the CoT penalty or to
the positive-signal reward terms. Captures the quantified answer: SIGNAL-ONLY (0.865)
sits between BASELINE and MAIN but MUCH closer to MAIN -- the signal-shaping terms
alone recover ~80% of MAIN's total improvement over BASELINE, with the CoT penalty
contributing a real but modest remaining increment. Also captures the one notable
anomaly: the first-ever sub-1.0 decode_back_matches_own_trace_rate reading across all
three 150-step runs in this stage, at step 65, immediately recovered."""
import json
import statistics
from pathlib import Path

BASELINE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())
MAIN = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v2'
                    / 'stage9d_decode_back_main.json').read_text())
SIGNAL_ONLY = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-signal_only-v1'
                           / 'stage9d_decode_back_signal_only.json').read_text())


def _matched_vals():
    b_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in BASELINE['result']['milestones']}
    m_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in MAIN['result']['milestones']}
    s_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in SIGNAL_ONLY['result']['milestones']}
    steps = sorted(set(b_ms) & set(m_ms) & set(s_ms))
    return steps, [b_ms[s] for s in steps], [m_ms[s] for s in steps], [s_ms[s] for s in steps]


def test_config_zeroed_p_cot_and_kept_signal_consistency_active():
    cfg = SIGNAL_ONLY['config']
    assert 'p_cot fully zeroed' in cfg['reward_params']
    assert 'r_consistency=0.15' in cfg['reward_params']
    assert 'r_signal=0.15' in cfg['reward_params']


def test_ran_the_full_150_steps_with_no_breaker():
    result = SIGNAL_ONLY['result']
    assert result['terminal_step'] == 150
    assert result['hard_stop'] is None


def test_signal_only_sits_between_baseline_and_main_much_closer_to_main():
    steps, b_vals, m_vals, s_vals = _matched_vals()
    assert len(steps) == 30
    b_mean, m_mean, s_mean = statistics.mean(b_vals), statistics.mean(m_vals), statistics.mean(s_vals)
    assert b_mean < s_mean < m_mean
    gap_to_baseline = s_mean - b_mean
    gap_to_main = m_mean - s_mean
    # The signal-shaping terms alone recover the large majority of MAIN's total gain.
    assert gap_to_baseline > 3 * gap_to_main


def test_signal_only_vs_baseline_is_a_decisive_statistical_result():
    steps, b_vals, m_vals, s_vals = _matched_vals()
    diffs = [sv - bv for sv, bv in zip(s_vals, b_vals)]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    assert t_stat > 5.0


def test_signal_only_vs_main_is_a_real_but_much_smaller_effect():
    steps, b_vals, m_vals, s_vals = _matched_vals()
    diffs = [sv - mv for sv, mv in zip(s_vals, m_vals)]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    # SIGNAL-ONLY is significantly BELOW MAIN (negative t), but the gap is small
    # relative to the SIGNAL-ONLY-vs-BASELINE gap (checked in the test above).
    assert t_stat < -2.0
    assert abs(statistics.mean(diffs)) < 0.10


def test_decode_back_fidelity_has_exactly_one_anomalous_milestone_immediately_recovered():
    dbmr = [r['decode_back_matches_own_trace_rate'] for r in SIGNAL_ONLY['result']['milestones']]
    assert len(dbmr) == 30
    below_one = [(r['step'], r['decode_back_matches_own_trace_rate']) for r in SIGNAL_ONLY['result']['milestones']
                 if r['decode_back_matches_own_trace_rate'] < 1.0]
    assert below_one == [(65, 0.9523809523809523)]
    # Recovered at the very next milestone.
    step70 = next(r for r in SIGNAL_ONLY['result']['milestones'] if r['step'] == 70)
    assert step70['decode_back_matches_own_trace_rate'] == 1.0


def test_actual_training_reward_had_zero_p_cot_throughout_the_real_run():
    # The pre-flight check block (unlike diagnostic_reward, the function actually used
    # for training) never passes **reward_params to the wrapper -- a pre-existing,
    # harmless gap (disclosed in design.md) that means its printed corrections always
    # preview MAIN-style defaults regardless of phase, not this phase's own settings.
    # What matters for correctness is the ACTUAL per-step training reward, which DOES
    # use SIGNAL_ONLY_REWARD_PARAMS via diagnostic_reward -- confirmed here by checking
    # every row captured in the row-level long-completion log (real training-batch
    # rewards, not the pre-flight preview) shows p_cot exactly 0.
    rows = SIGNAL_ONLY['result']['row_level_long_completion_log']
    assert len(rows) > 0
    assert all(r['reward_breakdown']['p_cot'] == 0.0 for r in rows)
    assert all(r['reward_breakdown']['p_cot_correction_applied'] == 0.0 for r in rows)


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
