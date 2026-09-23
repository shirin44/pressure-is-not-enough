"""Golden-record regression tests against the Stage 9d decode-back-seeded MAIN retry
(aws_runs/stage9d-decode-back-main-v2/, 2026-09-02): the successful re-run after the
first MAIN attempt hit the grad_norm breaker at step 5, diagnosed as ordinary
long-completion/high-entropy variance (the same phenomenon BASELINE hit 21 times
independently) rather than a mechanistic consequence of the adversarial reward terms.
This retry ran the full 150 steps cleanly with the new row-level long-completion
logging active. Captures the headline, unexpected finding: MAIN does NOT show the
"decay similarly / decay faster / decay less" pattern anticipated going in -- it shows
NO decay at all, and substantially, statistically significantly OUTPERFORMS BASELINE's
own declining trajectory at 29 of 30 matched milestones, while decode-back trace
fidelity remains perfect in both conditions throughout."""
import json
import statistics
from pathlib import Path

BASELINE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())
MAIN = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v2'
                    / 'stage9d_decode_back_main.json').read_text())


def test_retry_ran_the_full_150_steps_with_no_breaker():
    result = MAIN['result']
    assert result['terminal_step'] == 150
    assert result['hard_stop'] is None


def test_row_level_logging_captured_long_completion_events_without_needing_them_for_a_breaker():
    rows = MAIN['result']['row_level_long_completion_log']
    assert len(rows) > 0  # the new instrumentation did trigger at least once this run
    steps_triggered = {r['physical_step'] for r in rows}
    for step in steps_triggered:
        step_rows = [r for r in rows if r['physical_step'] == step]
        assert len(step_rows) == 8  # a full GRPO group captured, not just the long outlier
        assert max(r['token_length'] for r in step_rows) > MAIN['result']['row_level_log_threshold_tokens']


def test_decode_back_trace_fidelity_perfect_in_both_runs_at_every_milestone():
    b_dbmr = [r['decode_back_matches_own_trace_rate'] for r in BASELINE['result']['milestones']]
    m_dbmr = [r['decode_back_matches_own_trace_rate'] for r in MAIN['result']['milestones']]
    assert len(b_dbmr) == len(m_dbmr) == 30
    assert all(x == 1.0 for x in b_dbmr)
    assert all(x == 1.0 for x in m_dbmr)


def test_main_substantially_and_consistently_exceeds_baseline_at_matched_milestones():
    b_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in BASELINE['result']['milestones']}
    m_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in MAIN['result']['milestones']}
    steps = sorted(set(b_ms) & set(m_ms))
    assert len(steps) == 30
    diffs = [m_ms[s] - b_ms[s] for s in steps]
    n_main_higher = sum(1 for d in diffs if d > 0)
    assert n_main_higher == 29  # MAIN higher at all but one matched milestone
    assert abs(statistics.mean(diffs) - 0.20793650793650792) < 1e-9


def test_paired_comparison_is_a_statistically_decisive_result_not_milestone_noise():
    b_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in BASELINE['result']['milestones']}
    m_ms = {r['step']: r['genuine_correct_among_structural_nonliteral'] for r in MAIN['result']['milestones']}
    steps = sorted(set(b_ms) & set(m_ms))
    diffs = [m_ms[s] - b_ms[s] for s in steps]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    # A paired t-statistic this large (df=29) is far beyond any conventional
    # significance threshold -- this is not noise within the already-characterized
    # milestone-to-milestone variance of this stage.
    assert t_stat > 8.0


def test_main_mean_exceeds_even_baselines_own_peak_not_just_its_declining_tail():
    b_vals = [r['genuine_correct_among_structural_nonliteral'] for r in BASELINE['result']['milestones']]
    m_vals = [r['genuine_correct_among_structural_nonliteral'] for r in MAIN['result']['milestones']]
    assert statistics.mean(m_vals) > max(b_vals)


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
