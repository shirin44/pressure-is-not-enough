"""Small, dependency-free (no scipy) binomial-proportion statistics for scoring the
v2 monitor against the null of an uninformed 50/50 guesser. Exact math only (no
normal-approximation p-value, though a z-statistic is reported alongside for
familiarity) -- appropriate given the task's own "do not overclaim precision at
n=63" instruction."""
from __future__ import annotations

import math
from typing import Any


def _binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def exact_binomial_two_sided_p(successes: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial test p-value: sum of P(X=k) over all k whose
    probability under H0 is <= the probability of the observed count (the standard
    'sum of tails' definition R's binom.test / scipy's binomtest both use)."""
    observed_pmf = _binomial_pmf(successes, n, p)
    total = 0.0
    for k in range(n + 1):
        pk = _binomial_pmf(k, n, p)
        if pk <= observed_pmf * (1 + 1e-9):  # tolerance for float comparison
            total += pk
    return min(total, 1.0)


def wilson_score_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval (z=1.96) -- preferred over the normal approximation
    at small/moderate n, standard choice for a binomial proportion CI."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def summarize_binomial(successes: int, n: int, p0: float = 0.5) -> dict[str, Any]:
    phat = successes / n if n else None
    z_stat = (phat - p0) / math.sqrt(p0 * (1 - p0) / n) if n else None
    return {
        'successes': successes, 'n': n, 'phat': phat,
        'z_statistic': z_stat,
        'exact_binomial_two_sided_p': exact_binomial_two_sided_p(successes, n, p0),
        'wilson_95pct_ci': wilson_score_interval(successes, n),
        'distinguishable_from_0.5_at_alpha_0.05': (
            exact_binomial_two_sided_p(successes, n, p0) < 0.05 if n else None
        ),
    }
