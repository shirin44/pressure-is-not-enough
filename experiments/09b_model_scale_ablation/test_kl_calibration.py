import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kl_calibration import compute_clamp_from_pool, floor_d_max, solve_kl_clamp_bound


def test_zero_dominated_pool_degenerate_with_all_values():
    # Mirrors the real observed failure: mostly zeros, few small nonzero values, so
    # median/Q3 land at exactly 0 when zeros are included.
    pool = [0.0] * 100 + [0.001, 0.002, 0.003]
    result = compute_clamp_from_pool(pool, nonzero_only=False)
    assert result['median'] == 0.0
    assert result['clamp_value'] == 0.0
    assert result['degenerate'] is True


def test_same_pool_nonzero_only_is_not_degenerate():
    pool = [0.0] * 100 + [0.001, 0.002, 0.003]
    result = compute_clamp_from_pool(pool, nonzero_only=True)
    assert result['n'] == 3
    assert result['median'] == 0.002
    assert result['clamp_value'] > 0
    assert result['degenerate'] is False


def test_empty_pool_after_excluding_zeros_is_degenerate():
    pool = [0.0] * 50
    result = compute_clamp_from_pool(pool, nonzero_only=True)
    assert result['n'] == 0
    assert result['degenerate'] is True
    assert result['clamp_value'] is None


def test_healthy_spread_pool_gives_reasonable_clamp_either_way():
    pool = [0.01, 0.02, 0.015, 0.03, 0.025, 0.018, 0.022, 0.017, 0.019, 0.021]
    result = compute_clamp_from_pool(pool, nonzero_only=True)
    assert result['degenerate'] is False
    assert 0 < result['clamp_value'] < 1.0


def test_solve_kl_clamp_bound_round_trips():
    import math
    for target in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0):
        d = solve_kl_clamp_bound(target)
        reconstructed = math.exp(d) - d - 1
        assert abs(reconstructed - target) < 1e-6


def test_solve_kl_clamp_bound_rejects_nonpositive():
    try:
        solve_kl_clamp_bound(0.0)
        assert False, 'expected ValueError'
    except ValueError:
        pass
    try:
        solve_kl_clamp_bound(-1.0)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_floor_not_applied_when_d_max_valid():
    d, applied, msg = floor_d_max(0.02, floor_value=0.001, source='test')
    assert d == 0.02
    assert applied is False
    assert msg is None


def test_floor_applied_when_d_max_zero():
    d, applied, msg = floor_d_max(0.0, floor_value=0.005, source='min observed nonzero D_MAX')
    assert d == 0.005
    assert applied is True
    assert msg is not None and 'DEGENERATE' in msg


def test_floor_applied_when_d_max_none():
    d, applied, msg = floor_d_max(None, floor_value=0.005, source='test')
    assert d == 0.005
    assert applied is True


def test_floor_applied_when_d_max_negative():
    d, applied, msg = floor_d_max(-0.1, floor_value=0.005, source='test')
    assert d == 0.005
    assert applied is True


def test_end_to_end_before_fix_rejects_degenerate_result():
    # Reproduces the real observed failure mode end to end: a zero-dominated pool (like a
    # fresh, barely-diverged LoRA calibration run) pooled with the OLD all-values method
    # yields a clamp value that cannot be solved for a usable D_MAX (clamp_value <= 0 is
    # not a valid target for solve_kl_clamp_bound -- it must reject, not silently proceed).
    pool = [0.0] * 200 + [0.0001, 0.0002, 0.00015]
    result = compute_clamp_from_pool(pool, nonzero_only=False)
    assert result['degenerate'] is True
    assert result['clamp_value'] == 0.0
    try:
        solve_kl_clamp_bound(result['clamp_value'])
        assert False, 'expected ValueError -- a degenerate clamp_value must not silently solve to a D_MAX'
    except ValueError:
        pass


def test_end_to_end_after_fix_produces_stable_sane_d_max():
    # SAME pool as above, but through the nonzero-only method plus the floor backstop --
    # end-to-end path a live calibration run actually takes.
    pool = [0.0] * 200 + [0.0001, 0.0002, 0.00015]
    result = compute_clamp_from_pool(pool, nonzero_only=True)
    assert result['degenerate'] is False
    d_max = solve_kl_clamp_bound(result['clamp_value'])
    assert d_max > 0
    # Floor backstop only engages on an ACTUALLY degenerate result -- must be a no-op here,
    # since nonzero-only already produced a valid, positive D_MAX.
    floored, applied, msg = floor_d_max(d_max, floor_value=0.050781, source='seed-stability check')
    assert applied is False
    assert floored == d_max
    # Sanity range: should land in the same rough order of magnitude as the real 7B
    # seed-stability check's observed values (0.05-0.06), not wildly off.
    assert 1e-4 < d_max < 10.0


def test_end_to_end_floor_engages_only_for_a_genuinely_empty_nonzero_pool():
    # A pool with NO nonzero values at all (worse than the real observed case, which always
    # had a handful of nonzero tokens) -- nonzero-only pooling itself degenerates here, and
    # this is exactly the case the floor backstop exists for.
    pool = [0.0] * 200
    result = compute_clamp_from_pool(pool, nonzero_only=True)
    assert result['degenerate'] is True
    assert result['clamp_value'] is None
    floored, applied, msg = floor_d_max(None, floor_value=0.050781, source='seed-stability check')
    assert applied is True
    assert floored == 0.050781
    assert 'DEGENERATE' in msg  # loud, not silent


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
