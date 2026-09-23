"""Golden-record regression tests against the settled-configuration validation batch
(aws_runs/stage9d-decode-back-main-v{12,13,14,15}/, 2026-09-04): the SAME 4 seeds
(42/43/44/45) used throughout this whole recalibration effort, re-run at the newly
adopted settled config (GRAD_BREAKER=200.0, KL_BREAKER=5.0 primary) to confirm they
now survive past their previous break/near-break points. seed43 is the pivotal case --
it produced grad_norm=72.0 under the OLD GRAD_BREAKER=80.0 validation batch (the
highest benign value ever observed, the finding that sank the max-plus-margin
approach); here it must complete cleanly under the new backstop.

Distinct from test_grad_breaker_validation_batch.py (the OLD, GRAD_BREAKER=80.0
batch, correctly NOT adopted) -- this is the batch for the FINAL, adopted
configuration, per design.md's "Settled configuration" section."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
SETTLED_RUNS = {
    'seed42': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v12' / 'stage9d_decode_back_main.json').read_text()),
    'seed43': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v13' / 'stage9d_decode_back_main.json').read_text()),
    'seed44': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v14' / 'stage9d_decode_back_main.json').read_text()),
    'seed45': json.loads((AWS_RUNS / 'stage9d-decode-back-main-v15' / 'stage9d_decode_back_main.json').read_text()),
}
GRAD_BREAKER = 200.0
KL_BREAKER = 5.0
LONG_COMPLETION_THRESHOLD = 200
OLD_VALIDATION_BATCH_MAX_GRAD_NORM = 72.0  # seed43, under the old GRAD_BREAKER=80.0 batch


def _long_completion_events(ev):
    return [r for r in ev['result']['telemetry']
            if r.get('completions/mean_length', 0) >= LONG_COMPLETION_THRESHOLD]


def test_all_four_seeds_completed_the_full_50_steps_with_no_hard_stop():
    for name, ev in SETTLED_RUNS.items():
        assert ev['result']['terminal_step'] == 50, name
        assert ev['result']['hard_stop'] is None, name


def test_seed43_now_survives_past_its_previous_break_point():
    # The pivotal case: seed43 produced grad_norm=72.0 under the old GRAD_BREAKER=80.0
    # batch. Under the new GRAD_BREAKER=200.0, it must complete cleanly.
    ev = SETTLED_RUNS['seed43']
    events = _long_completion_events(ev)
    assert len(events) >= 1
    assert ev['result']['hard_stop'] is None
    assert max(r['grad_norm'] for r in events) < GRAD_BREAKER


def test_no_run_in_this_batch_approached_the_new_grad_breaker():
    all_grad_norms = [r['grad_norm'] for ev in SETTLED_RUNS.values() for r in _long_completion_events(ev)]
    assert len(all_grad_norms) > 0
    assert max(all_grad_norms) < GRAD_BREAKER
    # comfortably clear, consistent with GRAD_BREAKER being a coarse backstop, not a
    # threshold this batch was expected to graze
    assert max(all_grad_norms) < GRAD_BREAKER / 2


def test_kl_never_approached_its_threshold_in_any_of_this_batch_either():
    # Confirms KL_BREAKER remains capable of catching genuine divergence: in every
    # single long-completion event across this NEW batch (not just the historical 22),
    # KL stayed far below KL_BREAKER=5.0 -- the backstop relationship (grad_norm noisy
    # and confounded, KL clean and stable) holds on fresh data too, not just the
    # dataset used to derive the recommendation.
    all_kls = [r['kl'] for ev in SETTLED_RUNS.values() for r in _long_completion_events(ev) if r.get('kl') is not None]
    assert len(all_kls) > 0
    assert max(all_kls) < KL_BREAKER / 3


def test_this_batchs_max_grad_norm_is_consistent_with_the_historical_benign_range():
    # Not a strict ceiling claim (grad_norm has no stable ceiling, which is the whole
    # reason KL was promoted to primary) -- just confirms this fresh batch isn't some
    # wildly different regime from the data the recalibration decision was based on.
    all_grad_norms = [r['grad_norm'] for ev in SETTLED_RUNS.values() for r in _long_completion_events(ev)]
    assert max(all_grad_norms) < OLD_VALIDATION_BATCH_MAX_GRAD_NORM * 1.5


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
