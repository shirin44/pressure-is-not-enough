"""Golden-record regression tests against the GRAD_BREAKER=80.0 validation batch
(aws_runs/stage9d-decode-back-main-v{8,9,10,11}/, 2026-09-03): 4 short (50-step) re-
runs of the same seeds that broke under the old GRAD_BREAKER=50.0, launched to
confirm the proposed 80.0 threshold. Captures the actual, more important finding:
one validation event (seed=43, grad_norm=72.0) came in HIGHER than every one of the
original 5 breaker-triggering values, proving the initial 315-point "survived" vs
"breaker-triggering" separation was an artifact of under-sampling the upper tail,
not a genuine gap in the underlying distribution. This is why GRAD_BREAKER=80.0 is
NOT adopted as a new default -- the empirical-max-plus-margin methodology itself
does not hold up against additional data for this specific, mechanically
length-confounded metric."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
VALIDATION_RUNS = {
    'seed42': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v8' / 'stage9d_decode_back_main.json').read_text()),
    'seed43': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v9' / 'stage9d_decode_back_main.json').read_text()),
    'seed44': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v10' / 'stage9d_decode_back_main.json').read_text()),
    'seed45': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v11' / 'stage9d_decode_back_main.json').read_text()),
}
OLD_BREAKER_TRIGGERING_MAX = 64.0
LONG_COMPLETION_THRESHOLD = 200


def _long_completion_grad_norms(ev):
    return [r['grad_norm'] for r in ev['result']['telemetry']
            if r.get('completions/mean_length', 0) >= LONG_COMPLETION_THRESHOLD]


def test_all_four_validation_runs_completed_the_full_50_steps():
    for name, ev in VALIDATION_RUNS.items():
        assert ev['result']['terminal_step'] == 50, name
        assert ev['result']['hard_stop'] is None, name


def test_all_four_seeds_reproduced_at_least_one_long_completion_event():
    # Confirms these seeds are still genuinely exercising the mechanism under test,
    # not just happening to avoid it entirely this time.
    for name, ev in VALIDATION_RUNS.items():
        assert len(_long_completion_grad_norms(ev)) >= 1, name


def test_seed43_produced_a_confirmed_benign_value_above_the_entire_old_breaker_range():
    # The key finding: this value did NOT trigger the (raised) breaker, training
    # continued normally for the rest of the run, yet it EXCEEDS every one of the
    # original 5 breaker-triggering grad_norm values from the prior batch.
    grad_norms = _long_completion_grad_norms(VALIDATION_RUNS['seed43'])
    assert max(grad_norms) > OLD_BREAKER_TRIGGERING_MAX
    assert abs(max(grad_norms) - 72.0) < 1e-9


def test_the_72_grad_norm_event_still_showed_no_elevated_kl_signal():
    KL_BREAKER = 5.0
    ev = VALIDATION_RUNS['seed43']
    row = next(r for r in ev['result']['telemetry']
               if r.get('completions/mean_length', 0) >= LONG_COMPLETION_THRESHOLD and abs(r['grad_norm'] - 72.0) < 1e-9)
    assert row['kl'] < KL_BREAKER / 5  # same >=5x margin pattern as every other examined event


def test_proposed_threshold_still_held_this_batch_but_margin_shrank_substantially():
    PROPOSED_GRAD_BREAKER = 80.0
    all_new_grad_norms = [g for ev in VALIDATION_RUNS.values() for g in _long_completion_grad_norms(ev)]
    new_max = max(all_new_grad_norms)
    assert new_max < PROPOSED_GRAD_BREAKER  # no new breaker fired in this batch
    remaining_margin = PROPOSED_GRAD_BREAKER - new_max
    assert remaining_margin < 10.0  # far smaller than the original ~16-point margin computed against n=315


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
