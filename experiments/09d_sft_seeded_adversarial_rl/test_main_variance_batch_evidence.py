"""Golden-record regression tests against the full MAIN variance-characterization
batch (2026-09-03): 4 new seeded re-runs (v4-v7, seeds 42/43/44/45) launched after
v3's dramatic divergence from v2 revealed MAIN's headline result is not reliably
reproducible. Captures the actual, more severe finding this batch surfaced: the
grad_norm breaker fires in the MAJORITY of MAIN launches (5 of 7 total, 71.4%), via
the SAME consistent mechanism in every case (a long/high-entropy rollout group,
completions/mean_length=256, matching the original v1 breaker's own signature) --
this is the dominant, most robust result, more severe than "high variance in the
final accuracy number" since most launches never reach a comparable endpoint at all.
Also captures that every one of the 7 launches (completed or aborted) shows the SAME
mistranslation-dominated error profile already established for v3, with no new
mechanism found in any new run."""
import json
from pathlib import Path

from variance_analysis import analyze_run

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
RUNS = {f'v{v}': json.loads((AWS_RUNS / f'stage9d-decode-back-main-v{v}' / 'stage9d_decode_back_main.json').read_text())
        for v in range(1, 8)}


def test_all_seven_main_launches_are_present():
    assert set(RUNS) == {f'v{v}' for v in range(1, 8)}


def test_breaker_fired_in_the_majority_of_launches():
    hard_stops = {name: ev['result']['hard_stop'] for name, ev in RUNS.items()}
    broken = [name for name, hs in hard_stops.items() if hs is not None]
    completed = [name for name, hs in hard_stops.items() if hs is None]
    assert len(broken) == 5
    assert set(broken) == {'v1', 'v4', 'v5', 'v6', 'v7'}
    assert set(completed) == {'v2', 'v3'}


def test_every_breaker_step_is_within_the_first_39_steps():
    breaking_steps = [ev['result']['hard_stop']['step'] for ev in RUNS.values() if ev['result']['hard_stop']]
    assert all(1 <= s <= 39 for s in breaking_steps)
    assert breaking_steps == [5, 6, 39, 10, 32]


def test_every_breaker_shares_the_same_long_completion_signature():
    # Confirmed by direct inspection of each breaking step's raw telemetry row in the
    # retrieved stdout logs (not re-derived here, since telemetry isn't persisted
    # in the JSON evidence) -- all four NEW breaks (v4/v5/v6/v7) showed
    # completions/mean_length=256, clipped_ratio=0.125, entropy in [1.02, 1.22],
    # matching v1's own original breaker signature exactly. This test guards the
    # breaking-step values themselves, which ARE persisted.
    expected_grad_norms = {'v1': 51.5, 'v4': 53.25, 'v5': 64.0, 'v6': 58.25, 'v7': 59.5}
    for name, expected in expected_grad_norms.items():
        assert RUNS[name]['result']['hard_stop']['grad_norm'] == expected


def test_only_two_runs_completed_so_the_genuine_correct_distribution_has_n_equals_2():
    completed = {name: ev for name, ev in RUNS.items() if ev['result']['hard_stop'] is None}
    assert len(completed) == 2
    means = {name: analyze_run(ev)['genuine_correct_among_nonliteral']['mean'] for name, ev in completed.items()}
    assert abs(means['v2'] - 0.9063492063492063) < 1e-9
    assert abs(means['v3'] - 0.611111111111111) < 1e-9
    # The two completed runs differ by ~30 points -- too few points for a trustworthy
    # mean+-stdev claim, which this test exists to keep visible, not to paper over.
    assert abs(means['v2'] - means['v3']) > 0.25


def test_every_run_completed_or_aborted_shows_the_same_dominant_error_mechanism():
    # No new/different failure mode found in any of the 4 new runs -- checked, not
    # assumed, per task instruction 2.
    for name, ev in RUNS.items():
        result = analyze_run(ev)
        if result['total_wrong_samples'] == 0:
            continue
        fractions = result['error_taxonomy_fractions']
        mistranslation_or_tracking = fractions.get('decode_back_mistranslation', 0) + fractions.get('genuine_tracking_error', 0)
        assert mistranslation_or_tracking > 0.9, f'{name} shows an unexplained error category'


def test_decode_back_fidelity_never_drops_far_from_perfect_in_any_run():
    for name, ev in RUNS.items():
        result = analyze_run(ev)
        assert result['decode_back_matches_own_trace_rate']['min'] >= 0.95


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
