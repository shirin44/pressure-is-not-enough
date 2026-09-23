"""CPU-only tests for the length-normalized grad_norm proposal (2026-09-03), the
first of the two fixes recommended after the empirical-max-plus-margin recalibration
(GRAD_BREAKER=80.0) was validated and found NOT viable (test_grad_breaker_validation_
batch.py). Confirms the actual, decisive finding: length-normalization is a
mathematically real fix for the confound observed across the FULL telemetry
population (length 73-256), but a NO-OP for the specific population the breaker
threshold must be calibrated against, because 309/315 long-completion events (and all
22 of the key events) sit at exactly the same length (256.0, the generation ceiling).
Entropy -- the other candidate covariate -- is checked too and found to make the
CV worse, not better, among the 22 key events. Net result: neither candidate
normalizer produces a stable, tighter range for this specific calibration problem,
which is why the fallback (KL-as-primary, grad_norm-as-coarse-backstop) is recommended
instead. See design.md for the full write-up."""
from pathlib import Path

from grad_norm_calibration import (
    NORMALIZATION_POWER,
    coefficient_of_variation,
    extract_all_telemetry_pairs,
    extract_long_completion_events,
    key_22_events,
    length_normalization_report,
    load_evidence,
    load_validation_batch_evidence,
    normalized_grad_norm,
)

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = load_evidence(AWS_RUNS)
VALIDATION_EVIDENCE = load_validation_batch_evidence(AWS_RUNS)
EVENTS_22 = key_22_events(EVIDENCE, VALIDATION_EVIDENCE)


def _pearson(xs, ys):
    import statistics
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy)


def test_loaded_all_four_validation_batch_runs():
    assert len(VALIDATION_EVIDENCE) == 4


def test_key_22_events_has_exactly_22():
    # 5 original breaker-triggering events + 17 validation-batch long-completion
    # events (1 + 1 + 8 + 7 across seeds 42/43/44/45) -- matches the task's own count.
    assert len(EVENTS_22) == 22


def test_the_confound_is_real_across_the_full_length_range():
    # Across the FULL telemetry population (length 73-256, not filtered to long
    # completions only), length and grad_norm are genuinely correlated -- this is
    # the confound the normalization proposal is trying to address.
    pairs = extract_all_telemetry_pairs(EVIDENCE)
    assert len(pairs) > 1000
    lengths = [p['mean_length'] for p in pairs]
    gn = [p['grad_norm'] for p in pairs]
    assert _pearson(lengths, gn) > 0.4


def test_length_normalization_removes_the_confound_in_the_full_population():
    pairs = extract_all_telemetry_pairs(EVIDENCE)
    lengths = [p['mean_length'] for p in pairs]
    norm = [normalized_grad_norm(p['grad_norm'], p['mean_length']) for p in pairs]
    # the whole point of the fitted power: residual correlation collapses near zero
    assert abs(_pearson(lengths, norm)) < 0.1


def test_but_the_long_completion_regime_has_almost_no_length_variance():
    # This is the wrinkle: the population the breaker threshold must actually be
    # calibrated against (long-completion events) sits almost entirely at the exact
    # 256-token generation ceiling, not spread across the full 73-256 range.
    events = extract_long_completion_events(EVIDENCE)
    n_at_256 = sum(1 for e in events if e['mean_length'] == 256.0)
    assert n_at_256 / len(events) > 0.95


def test_all_22_key_events_sit_at_the_generation_ceiling():
    assert all(e['mean_length'] == 256.0 for e in EVENTS_22)


def test_length_normalization_is_a_no_op_for_the_22_key_events():
    # The decisive test: since every one of the 22 key events is at the SAME length,
    # dividing by length^p is a uniform rescale -- it cannot change the RELATIVE
    # spread at all. CV before and after should match to high precision.
    report = length_normalization_report(EVENTS_22)
    assert report['n_at_length_256'] == 22
    assert report['cv_unchanged']
    assert abs(report['raw_spread_ratio'] - report['normalized_spread_ratio']) < 1e-6


def test_entropy_normalization_does_not_help_either():
    # Entropy is the one covariate that DOES vary among the 22 events at fixed
    # length, so it's the natural next candidate -- but normalizing by entropy^p
    # makes the coefficient of variation WORSE, not better, and does so
    # monotonically as p increases, confirming this isn't a power-tuning problem.
    raw = [e['grad_norm'] for e in EVENTS_22]
    ent = [e['entropy'] for e in EVENTS_22]
    raw_cv = coefficient_of_variation(raw)
    cvs_by_power = []
    for p in (1.0, 1.5, 2.0, 2.5):
        norm = [g / (e ** p) for g, e in zip(raw, ent)]
        cvs_by_power.append(coefficient_of_variation(norm))
    assert all(cv > raw_cv for cv in cvs_by_power)
    assert cvs_by_power == sorted(cvs_by_power)  # monotonically worsening


def test_entropy_correlation_sign_is_unstable_across_populations():
    # Within the full 309-event long-completion population, entropy correlates
    # POSITIVELY with grad_norm (r~0.33). Within the smaller 22-key-event subset,
    # the sign flips negative -- a real, reportable instability, not a typo. This is
    # part of why entropy is rejected as a normalizer, not just the CV result alone.
    events = [e for e in extract_long_completion_events(EVIDENCE) if e['mean_length'] == 256.0 and e['entropy']]
    full_corr = _pearson([e['entropy'] for e in events], [e['grad_norm'] for e in events])
    key22_corr = _pearson([e['entropy'] for e in EVENTS_22], [e['grad_norm'] for e in EVENTS_22])
    assert full_corr > 0.2
    assert key22_corr < 0


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
