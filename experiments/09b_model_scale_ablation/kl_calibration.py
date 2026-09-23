"""Stage 9b: KL-clamp calibration statistics, shared between injection_dryrun_7b.py's
live Phase 0 and the standalone seed-stability check (calibration_seed_stability_check.py),
so both use the exact same pooling/quantile/floor logic rather than two copies that could
drift apart.

Deviation from the 3B methodology, disclosed (design.md section 4 addendum): the 3B
calibration pooled ALL per-token KL values (zero and nonzero alike) from clean steps of an
already-completed, mature 500-step instrumentation run, where enough real divergence had
accumulated that the pool's median/IQR were never near-degenerate. Calibrating on a FRESH,
barely-diverged 7B LoRA adapter at a tiny learning rate surfaces a failure mode the 3B
methodology never had to handle: with a fresh adapter, the large majority of tokens have
per-token KL of EXACTLY zero (policy identical to reference before any real weight drift),
so pooling zeros alongside the genuinely-informative nonzero tokens can push the median (and
even Q3) to exactly 0, collapsing HEALTHY_BASELINE and CLAMP_VALUE to 0 and producing an
unusable D_MAX=0 clamp (verified: this happened on a real 7B calibration run, D_MAX=0.0,
next to a D_MAX=0.020 result from an otherwise-identical run with a different seed -- a 100x
swing from which few steps happened to land in the "clean" bucket).

Fix: pool only STRICTLY POSITIVE per-token KL values (`nonzero_only=True`, the default).
This addresses the actual mechanism (mass at exactly zero is uninformative about the SPREAD
of real divergence among tokens that did diverge at all, not a representative "typical
value" to center the clamp on) rather than papering over the symptom with an arbitrary
floor. The floor in `floor_d_max` below is a SEPARATE, deliberately-distinct safety backstop
for the case where even nonzero-only pooling still degenerates (e.g. an all-zero-KL step
with no nonzero tokens at all) -- not a substitute for real measurement.
"""
from __future__ import annotations

import math
import statistics


def compute_clamp_from_pool(per_token_kl_pool, *, iqr_multiplier=1.5, clamp_multiplier=1.5, nonzero_only=True):
    """Median + iqr_multiplier*IQR mild-outlier-fence baseline, times clamp_multiplier,
    over `per_token_kl_pool` (already restricted to clean/non-borderline steps by the
    caller). Returns a dict with n, median, q1, q3, iqr, healthy_baseline, clamp_value,
    nonzero_only, and 'degenerate' (True if the pool is empty or clamp_value <= 0)."""
    pool = [v for v in per_token_kl_pool if v > 0] if nonzero_only else list(per_token_kl_pool)
    n = len(pool)
    if n == 0:
        return {'n': 0, 'nonzero_only': nonzero_only, 'median': None, 'q1': None, 'q3': None,
                'iqr': None, 'healthy_baseline': None, 'clamp_value': None, 'degenerate': True}
    pool = sorted(pool)
    median = statistics.median(pool)
    q1 = statistics.median(pool[:n // 2])
    q3 = statistics.median(pool[(n + 1) // 2:])
    iqr = q3 - q1
    healthy_baseline = median + iqr_multiplier * iqr
    clamp_value = clamp_multiplier * healthy_baseline
    return {'n': n, 'nonzero_only': nonzero_only, 'median': median, 'q1': q1, 'q3': q3, 'iqr': iqr,
            'healthy_baseline': healthy_baseline, 'clamp_value': clamp_value,
            'degenerate': not (clamp_value is not None and clamp_value > 0)}


def solve_kl_clamp_bound(target_kl, tol=1e-10, max_iter=200):
    """Solve exp(D) - D - 1 = target_kl for D > 0 via Newton's method. Same formula as
    step14b's own calibration -- unchanged."""
    if target_kl <= 0:
        raise ValueError(f'target_kl must be > 0 to solve for D_MAX; got {target_kl}')
    D = math.log(target_kl + 1) if target_kl > 1 else math.sqrt(2 * target_kl)
    for _ in range(max_iter):
        f = math.exp(D) - D - 1 - target_kl
        fp = math.exp(D) - 1
        if fp == 0:
            break
        D_new = D - f / fp
        if abs(D_new - D) < tol:
            D = D_new
            break
        D = D_new
    return D


def floor_d_max(d_max, floor_value, *, source):
    """Pure safety backstop, distinct from real calibration: if `d_max` is None, zero, or
    negative (a degenerate calibration result), return `floor_value` instead and LOUDLY
    report that the floor was applied -- never silently. If `d_max` is already a valid
    positive value, it is returned unchanged and the floor is not applied. `source` is a
    human-readable string identifying where `floor_value` came from (for the log message),
    e.g. 'min observed nonzero D_MAX across seed-stability check' or '3B calibrated D_MAX
    (sanity-check reference)'.

    Returns (effective_d_max, floor_was_applied: bool, message: str | None)."""
    if d_max is not None and d_max > 0:
        return d_max, False, None
    message = (f'DEGENERATE CALIBRATION RESULT: measured D_MAX={d_max!r} is not usable. '
               f'Applying safety-backstop floor D_MAX={floor_value} (source: {source}). '
               f'This is NOT a real measured calibration for this run -- it is a fallback '
               f'to avoid training with a broken (D_MAX<=0) clamp. Flagged, not silent.')
    return floor_value, True, message
