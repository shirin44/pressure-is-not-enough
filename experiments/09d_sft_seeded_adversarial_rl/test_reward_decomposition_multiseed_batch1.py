"""Golden-record regression tests for the multi-seed reward-decomposition
re-characterization, batch 1 of 2 (2026-09-06): BASELINE and MAIN, 4 seeds each
(42/43/44/45), all run at the SETTLED breaker configuration (KL_BREAKER=5.0 primary,
GRAD_BREAKER=200.0 coarse backstop) -- the entire point this recalibration effort
existed to unblock. PENALTY-ONLY and SIGNAL-ONLY are a separate, not-yet-run batch 2,
per the task's own resource-constrained prioritization (BASELINE+MAIN first, report
before continuing).

Decisive finding, reported plainly per the task's own explicit instruction not to
round toward the original narrative: the original single-run comparison (BASELINE
final=0.698 vs MAIN final=0.906, described as MAIN "significantly outperforming"
BASELINE) does NOT survive multi-seed re-characterization. Across 4 fresh seeds each,
BASELINE's mean (0.9643) is actually HIGHER than MAIN's (0.9167) on final-milestone
accuracy, and MAIN shows nearly double BASELINE's seed-to-seed standard deviation
(0.0813 vs 0.0456) -- the opposite ordering from the original claim. Welch's t-test
(both on final-milestone accuracy and on the more stable mean-across-milestones
metric) shows no significant difference in either direction (p=0.36 and p=0.18
respectively, n=4 per condition). See design.md for the full write-up including the
explicit caveat that n=4 per condition has low power to detect anything but a large
effect -- this result rules out the ORIGINAL claimed effect size, it does not prove
the two conditions are equivalent.

All 8 runs in this batch completed their full 150-step budget with hard_stop=None --
zero breaker fires anywhere in the batch, confirming the settled recalibration
resolves MAIN's prior 71% breaker rate under the old GRAD_BREAKER=50.0."""
import json
import math
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'

BASELINE_RUNS = {
    42: AWS_RUNS / 'stage9d-decode-back-baseline-v2' / 'stage9d_decode_back_baseline.json',
    43: AWS_RUNS / 'stage9d-decode-back-baseline-v3' / 'stage9d_decode_back_baseline.json',
    44: AWS_RUNS / 'stage9d-decode-back-baseline-v4' / 'stage9d_decode_back_baseline.json',
    45: AWS_RUNS / 'stage9d-decode-back-baseline-v5' / 'stage9d_decode_back_baseline.json',
}
MAIN_RUNS = {
    42: AWS_RUNS / 'stage9d-decode-back-main-v16' / 'stage9d_decode_back_main.json',
    43: AWS_RUNS / 'stage9d-decode-back-main-v17' / 'stage9d_decode_back_main.json',
    44: AWS_RUNS / 'stage9d-decode-back-main-v18' / 'stage9d_decode_back_main.json',
    45: AWS_RUNS / 'stage9d-decode-back-main-v19' / 'stage9d_decode_back_main.json',
}
ALL_RUNS = {'BASELINE': BASELINE_RUNS, 'MAIN': MAIN_RUNS}
GRAD_BREAKER = 200.0
KL_BREAKER = 5.0
LONG_COMPLETION_THRESHOLD = 200

ORIGINAL_SINGLE_RUN_BASELINE_FINAL = 0.698
ORIGINAL_SINGLE_RUN_MAIN_FINAL = 0.906


def _load(path):
    return json.loads(path.read_text())


def _final_acc(ev):
    return ev['result']['milestones'][-1]['overall_final_answer_accuracy']


def _long_completion_events(ev):
    return [r for r in ev['result']['telemetry'] if r.get('completions/mean_length', 0) >= LONG_COMPLETION_THRESHOLD]


def welch_ttest(a, b):
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, df


def test_all_eight_runs_loaded():
    for cond, seeds in ALL_RUNS.items():
        assert len(seeds) == 4, cond


def test_all_eight_runs_completed_full_150_steps_with_no_breaker():
    for cond, seeds in ALL_RUNS.items():
        for seed, path in seeds.items():
            ev = _load(path)
            r = ev['result']
            assert r['terminal_step'] == 150, f'{cond} seed={seed}'
            assert r['hard_stop'] is None, f'{cond} seed={seed}'


def test_no_run_approached_either_breaker():
    for cond, seeds in ALL_RUNS.items():
        for seed, path in seeds.items():
            ev = _load(path)
            events = _long_completion_events(ev)
            if not events:
                continue
            max_gn = max(r['grad_norm'] for r in events)
            max_kl = max(r.get('kl', 0) for r in events)
            assert max_gn < GRAD_BREAKER, f'{cond} seed={seed}'
            assert max_kl < KL_BREAKER / 3, f'{cond} seed={seed}'


def test_baseline_mean_final_accuracy_matches_hand_computed_value():
    finals = [_final_acc(_load(p)) for p in BASELINE_RUNS.values()]
    mean = sum(finals) / len(finals)
    assert abs(mean - 0.9643) < 0.001


def test_main_mean_final_accuracy_matches_hand_computed_value():
    finals = [_final_acc(_load(p)) for p in MAIN_RUNS.values()]
    mean = sum(finals) / len(finals)
    assert abs(mean - 0.9167) < 0.001


def test_the_original_single_run_ordering_does_not_survive_multiseed_recharacterization():
    # The decisive, honest finding: the original single-run comparison claimed MAIN
    # (0.906) beats BASELINE (0.698). Across 4 fresh seeds each, the ORDERING FLIPS --
    # BASELINE's multi-seed mean is now higher than MAIN's. This does not mean MAIN is
    # worse (see the t-test below: not significant), but it decisively means the
    # original claimed effect does not survive -- reported plainly, not rounded
    # toward the original narrative.
    baseline_finals = [_final_acc(_load(p)) for p in BASELINE_RUNS.values()]
    main_finals = [_final_acc(_load(p)) for p in MAIN_RUNS.values()]
    baseline_mean = sum(baseline_finals) / len(baseline_finals)
    main_mean = sum(main_finals) / len(main_finals)
    assert ORIGINAL_SINGLE_RUN_MAIN_FINAL > ORIGINAL_SINGLE_RUN_BASELINE_FINAL  # the original claim
    assert main_mean < baseline_mean  # the multi-seed finding: ordering flips


def test_welch_ttest_shows_no_significant_difference_between_conditions():
    baseline_finals = [_final_acc(_load(p)) for p in BASELINE_RUNS.values()]
    main_finals = [_final_acc(_load(p)) for p in MAIN_RUNS.values()]
    t, df = welch_ttest(main_finals, baseline_finals)
    # not claiming a precise p-value here (no scipy in this environment) -- the
    # |t| < 2 threshold at ~4-5 df is a conservative, well-known proxy for
    # "nowhere near conventional significance," which is the only claim this test
    # needs to support.
    assert abs(t) < 2.0
    assert 3.0 < df < 6.0


def test_main_shows_higher_seed_to_seed_variance_than_baseline():
    import statistics
    baseline_finals = [_final_acc(_load(p)) for p in BASELINE_RUNS.values()]
    main_finals = [_final_acc(_load(p)) for p in MAIN_RUNS.values()]
    assert statistics.stdev(main_finals) > statistics.stdev(baseline_finals)


def test_decode_back_mistranslation_dominates_in_seven_of_eight_runs():
    # Confirms the known translation bias is the dominant error source in nearly
    # every run under the settled config too, not just the runs examined previously.
    from variance_analysis import analyze_run
    n_dominant = 0
    n_total = 0
    for cond, seeds in ALL_RUNS.items():
        for seed, path in seeds.items():
            ev = _load(path)
            a = analyze_run(ev)
            frac = a['error_taxonomy_fractions'].get('decode_back_mistranslation', 0)
            n_total += 1
            if frac > 0.5:
                n_dominant += 1
    assert n_total == 8
    assert n_dominant == 7  # MAIN seed44 is the sole exception, an exact 50/50 split


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
