"""Golden-record regression tests for the COMPLETE multi-seed reward-decomposition
re-characterization (2026-09-06): all four conditions (BASELINE, MAIN, PENALTY_ONLY,
SIGNAL_ONLY), 4 seeds each (42/43/44/45), all at the settled breaker configuration
(KL_BREAKER=5.0 primary, GRAD_BREAKER=200.0 backstop). Batch 1 covered BASELINE+MAIN
(test_reward_decomposition_multiseed_batch1.py); this file adds PENALTY_ONLY and
SIGNAL_ONLY (batch 2) and the complete 6-pairwise-comparison significance table that
supersedes both the original single-run table AND batch 1's partial result.

Headline finding, reported plainly per the task's own explicit instruction not to
search for a way to salvage the original bonus-dominates-penalty or near-additivity
narrative if the data doesn't support it: **no reward condition differs
significantly from any other at this sample size (n=4 per condition)**. Of 6
pairwise Welch's t-tests on final-milestone accuracy, one (BASELINE vs SIGNAL_ONLY)
reaches nominal p<0.05 (p=0.044) -- but it does not survive Bonferroni correction for
6 comparisons (requires p<0.0083), and is not even nominally significant on the more
stable mean-across-milestones metric (p=0.093). This is a clean null, not a partial
finding to be spun positively -- exactly the kind of result this whole
recalibration-then-recharacterization effort was designed to be able to surface
honestly."""
import json
import math
from itertools import combinations
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'

PENALTY_ONLY_RUNS = {
    42: AWS_RUNS / 'stage9d-decode-back-penalty_only-v2' / 'stage9d_decode_back_penalty_only.json',
    43: AWS_RUNS / 'stage9d-decode-back-penalty_only-v3' / 'stage9d_decode_back_penalty_only.json',
    44: AWS_RUNS / 'stage9d-decode-back-penalty_only-v4' / 'stage9d_decode_back_penalty_only.json',
    45: AWS_RUNS / 'stage9d-decode-back-penalty_only-v5' / 'stage9d_decode_back_penalty_only.json',
}
SIGNAL_ONLY_RUNS = {
    42: AWS_RUNS / 'stage9d-decode-back-signal_only-v2' / 'stage9d_decode_back_signal_only.json',
    43: AWS_RUNS / 'stage9d-decode-back-signal_only-v3' / 'stage9d_decode_back_signal_only.json',
    44: AWS_RUNS / 'stage9d-decode-back-signal_only-v4' / 'stage9d_decode_back_signal_only.json',
    45: AWS_RUNS / 'stage9d-decode-back-signal_only-v5' / 'stage9d_decode_back_signal_only.json',
}
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
ALL_RUNS = {
    'BASELINE': BASELINE_RUNS, 'MAIN': MAIN_RUNS,
    'PENALTY_ONLY': PENALTY_ONLY_RUNS, 'SIGNAL_ONLY': SIGNAL_ONLY_RUNS,
}
GRAD_BREAKER = 200.0
KL_BREAKER = 5.0
LONG_COMPLETION_THRESHOLD = 200
BONFERRONI_ALPHA = 0.05 / 6


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


def _finals(cond):
    return [_final_acc(_load(p)) for p in ALL_RUNS[cond].values()]


def test_all_sixteen_runs_loaded():
    assert sum(len(v) for v in ALL_RUNS.values()) == 16


def test_penalty_only_and_signal_only_completed_full_150_steps_with_no_breaker():
    for cond in ('PENALTY_ONLY', 'SIGNAL_ONLY'):
        for seed, path in ALL_RUNS[cond].items():
            ev = _load(path)
            r = ev['result']
            assert r['terminal_step'] == 150, f'{cond} seed={seed}'
            assert r['hard_stop'] is None, f'{cond} seed={seed}'


def test_no_run_in_batch_2_approached_either_breaker():
    for cond in ('PENALTY_ONLY', 'SIGNAL_ONLY'):
        for seed, path in ALL_RUNS[cond].items():
            ev = _load(path)
            events = _long_completion_events(ev)
            if not events:
                continue
            max_gn = max(r['grad_norm'] for r in events)
            max_kl = max(r.get('kl', 0) for r in events)
            assert max_gn < GRAD_BREAKER, f'{cond} seed={seed}'
            assert max_kl < KL_BREAKER / 3, f'{cond} seed={seed}'


def test_penalty_only_mean_final_accuracy_matches_hand_computed_value():
    finals = _finals('PENALTY_ONLY')
    mean = sum(finals) / len(finals)
    assert abs(mean - 0.8929) < 0.001


def test_signal_only_mean_final_accuracy_matches_hand_computed_value():
    finals = _finals('SIGNAL_ONLY')
    mean = sum(finals) / len(finals)
    assert abs(mean - 0.8571) < 0.001


def test_all_six_pairwise_comparisons_are_computed():
    pairs = list(combinations(ALL_RUNS.keys(), 2))
    assert len(pairs) == 6
    for a_name, b_name in pairs:
        t, df = welch_ttest(_finals(a_name), _finals(b_name))
        assert df > 0  # sanity: a real, finite Welch-Satterthwaite df was computed


def test_no_pairwise_comparison_survives_bonferroni_correction():
    # The decisive, honest finding: with 6 pairwise comparisons at n=4 per condition,
    # NONE survive a Bonferroni-corrected significance threshold (0.05/6 = 0.0083).
    # One comparison (BASELINE vs SIGNAL_ONLY) reaches nominal p<0.05 uncorrected
    # (checked in the next test) but that is expected by chance alone across 6 tests
    # and does not indicate a real, robust effect.
    import numpy as np

    def t_pdf(x, df):
        coef = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return coef * (1 + x ** 2 / df) ** (-(df + 1) / 2)

    def two_tailed_p(t, df):
        tt = abs(t)
        xs = np.linspace(tt, tt + 200, 500_000)
        ys = np.array([t_pdf(x, df) for x in xs])
        return 2 * np.trapezoid(ys, xs)

    for a_name, b_name in combinations(ALL_RUNS.keys(), 2):
        t, df = welch_ttest(_finals(a_name), _finals(b_name))
        p = two_tailed_p(t, df)
        assert p >= BONFERRONI_ALPHA, f'{a_name} vs {b_name} unexpectedly survived Bonferroni: p={p}'


def test_baseline_vs_signal_only_is_the_only_nominally_significant_uncorrected_pair():
    import numpy as np

    def t_pdf(x, df):
        coef = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
        return coef * (1 + x ** 2 / df) ** (-(df + 1) / 2)

    def two_tailed_p(t, df):
        tt = abs(t)
        xs = np.linspace(tt, tt + 200, 500_000)
        ys = np.array([t_pdf(x, df) for x in xs])
        return 2 * np.trapezoid(ys, xs)

    nominally_sig = []
    for a_name, b_name in combinations(ALL_RUNS.keys(), 2):
        t, df = welch_ttest(_finals(a_name), _finals(b_name))
        p = two_tailed_p(t, df)
        if p < 0.05:
            nominally_sig.append((a_name, b_name, p))
    assert len(nominally_sig) == 1
    assert set(nominally_sig[0][:2]) == {'BASELINE', 'SIGNAL_ONLY'}


def test_decode_back_mistranslation_dominates_in_fifteen_of_sixteen_runs():
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
    assert n_total == 16
    assert n_dominant == 15  # MAIN seed44 (batch 1) is the sole exception


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
