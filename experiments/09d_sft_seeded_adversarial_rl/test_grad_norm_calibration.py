"""CPU-only tests for grad_norm_calibration.py, run against the real, already-
retrieved evidence before proposing any new GRAD_BREAKER value or spending GPU time
validating it. Confirms the empirical basis for the proposed recalibration: a clean
gap between the survived and breaker-triggering distributions, and that a naive
IQR/Tukey-fence approach (as used for other clamps in this stage) would badly
under-estimate the legitimate tail of this specific, heavily right-skewed
mechanism -- explicitly checked, not assumed, before choosing a different
methodology (empirical-max-plus-margin) for this particular calibration."""
import statistics
from pathlib import Path

from grad_norm_calibration import extract_long_completion_events, load_evidence, summarize

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = load_evidence(AWS_RUNS)
EVENTS = extract_long_completion_events(EVIDENCE)


def test_loaded_all_twelve_same_config_runs():
    assert len(EVIDENCE) == 12


def test_found_a_large_number_of_long_completion_events():
    # Sanity floor -- this mechanism should be common enough to characterize, not a
    # handful of cherry-picked points.
    assert len(EVENTS) > 250


def test_survived_and_breaker_triggering_classes_do_not_overlap():
    s = summarize(EVENTS)
    assert s['n_survived'] > 300
    assert s['n_breaker_triggering'] == 5
    assert s['survived_max'] < s['breaker_min']
    gap = s['breaker_min'] - s['survived_max']
    assert gap > 2.0  # a real, non-trivial gap, not two classes touching at the boundary


def test_survived_max_matches_the_hand_computed_value():
    s = summarize(EVENTS)
    assert abs(s['survived_max'] - 48.75) < 1e-9
    assert s['breaker_values'] == [51.5, 53.25, 58.25, 59.5, 64.0]


def test_breaker_triggering_events_show_no_elevated_kl_signal():
    # Every breaker-triggering event's own KL should sit far below KL_BREAKER=5.0 --
    # supporting that these are grad_norm-magnitude artifacts of longer completions,
    # not genuine policy divergence (which would show elevated KL too).
    KL_BREAKER = 5.0
    breaker_events = [e for e in EVENTS if e['class'] == 'breaker_triggering']
    assert len(breaker_events) == 5
    for e in breaker_events:
        assert e['kl'] is not None
        assert e['kl'] < KL_BREAKER / 5  # at least 5x margin below the KL breaker


def test_naive_iqr_tukey_fence_would_badly_underestimate_the_benign_tail():
    # Explicitly checked (not assumed) before rejecting the IQR-based methodology
    # used for the KL clamp elsewhere in this stage: the distribution here is heavily
    # right-skewed (median << mean << max), so a standard Tukey fence would flag a
    # meaningful fraction of genuinely-survived events as "outliers" -- the wrong
    # failure mode for a breaker threshold.
    survived = sorted(e['grad_norm'] for e in EVENTS if e['class'] == 'survived')
    q1, _, q3 = statistics.quantiles(survived, n=4)
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    n_flagged = sum(1 for v in survived if v > fence)
    assert n_flagged > 10  # a real, non-trivial number of benign events would be misflagged
    assert fence < max(survived) / 2  # the fence sits well below the true observed max


def test_proposed_threshold_clears_the_entire_observed_breaker_range_with_margin():
    PROPOSED_GRAD_BREAKER = 80.0
    s = summarize(EVENTS)
    assert PROPOSED_GRAD_BREAKER > s['breaker_max']
    margin_above_breaker_max = PROPOSED_GRAD_BREAKER - s['breaker_max']
    observed_breaker_range_width = s['breaker_max'] - s['breaker_min']
    # the margin above the highest observed breaker-triggering value should be at
    # least as wide as the observed breaker range itself, not a token increment
    assert margin_above_breaker_max >= observed_breaker_range_width


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
