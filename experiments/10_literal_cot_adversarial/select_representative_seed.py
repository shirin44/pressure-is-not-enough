"""Stage 10: PRE-DECLARED checkpoint/seed selection rule and the BASELINE-vs-MAIN
statistical test. Written 2026-09-22, BEFORE any of this experiment's 8 GPU runs were
launched (no aws_runs/ or evidence directory exists yet at the time this file was written --
verifiable from this repo's own commit/file history). This exists specifically to close the
gap found in the Llama 09e audit, where the "best MAIN seed" selection (seed43 tie-broken
against seed45 by lowest-seed-number) was written AFTER the MAIN results were already known,
making the tie-break itself impossible to distinguish from post hoc cherry-picking.

THE RULE (fixed, not to be edited after seeing results; any change after this file is first
committed must be a new dated addendum below, never a silent edit of the rule itself):

  The reported "representative" seed, if one is needed for a figure or table, is the seed
  with the MEDIAN (not max) final-milestone genuine_correct_rate across the four MAIN seeds.
  If two seeds tie for median (i.e. an even-length list with the two middle values equal),
  the LOWER seed number is used.

  With four seeds, "median" of a 4-element list is not a single middle element by the usual
  definition (there are two middle elements, indices 1 and 2 of the sorted-by-score list).
  This rule resolves that ambiguity EXPLICITLY, decided now, not when the four real numbers
  are in hand: sort the four (seed, score) pairs by score ascending; the representative seed
  is the LOWER-INDEX of the two middle-ranked seeds (index 1 of the 0-indexed sorted list),
  i.e. the seed with the SMALLER of the two middle scores; if those two middle scores are
  exactly equal, use the lower seed number between them (same lower-seed-number tie-break
  extended to this case for consistency). This gives one specific, unambiguous seed for any
  possible set of four real numbers, decided before those numbers exist.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def select_representative_main_seed(seed_to_score: dict[int, float]) -> dict[str, Any]:
    """Implements the rule stated in this file's module docstring above, unchanged since
    2026-09-22. `seed_to_score` maps MAIN seed -> final-milestone genuine_correct_rate.
    Returns {'selected_seed', 'selected_score', 'rank_used', 'sorted_pairs', 'tie_broken'}."""
    if len(seed_to_score) != 4:
        raise ValueError(f'This rule is defined for exactly 4 MAIN seeds; got {len(seed_to_score)}')
    pairs = sorted(seed_to_score.items(), key=lambda kv: (kv[1], kv[0]))  # score asc, seed asc as a stable sub-key
    lower_middle, upper_middle = pairs[1], pairs[2]
    tie_broken = False
    if math.isclose(lower_middle[1], upper_middle[1], rel_tol=0, abs_tol=1e-12):
        chosen = min(lower_middle, upper_middle, key=lambda kv: kv[0])  # lower seed number
        tie_broken = True
    else:
        chosen = lower_middle  # the smaller of the two middle scores, per the rule above
    return {'selected_seed': chosen[0], 'selected_score': chosen[1], 'rank_used': 'lower_of_two_middle_scores',
            'sorted_pairs': pairs, 'tie_broken': tie_broken}


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Reused verbatim from experiments/09e_same_different_llama/analyze_main_vs_baseline.py
    (Numerical Recipes continued fraction for the Welch t-test's p-value) -- not reimplemented."""
    def fraction(aa: float, bb: float, xx: float) -> float:
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c, d, h = 1.0, 1.0 - qab * xx / qap, 1.0
        d = 1.0 / max(abs(d), 3e-14) * (1 if d >= 0 else -1)
        h = d
        for m in range(1, 201):
            m2 = 2 * m
            term = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + term * d; d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
            c = 1.0 + term / c; c = c if abs(c) > 3e-14 else 3e-14
            h *= d * c
            term = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
            d = 1.0 + term * d; d = 1.0 / (d if abs(d) > 3e-14 else 3e-14)
            c = 1.0 + term / c; c = c if abs(c) > 3e-14 else 3e-14
            delta = d * c; h *= delta
            if abs(delta - 1.0) < 3e-14:
                return h
        raise RuntimeError('incomplete-beta continued fraction did not converge')
    if not 0.0 <= x <= 1.0:
        raise ValueError(x)
    if x in (0.0, 1.0):
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * fraction(a, b, x) / a
    return 1.0 - front * fraction(b, a, 1.0 - x) / b


def welch_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    sa, sb = va / na, vb / nb
    t = (statistics.mean(a) - statistics.mean(b)) / math.sqrt(sa + sb)
    df = (sa + sb) ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1))
    p = regularized_incomplete_beta(df / (df + t * t), df / 2.0, 0.5)
    return {'t_statistic': t, 'degrees_of_freedom': df, 'p_value_two_sided': p}


if __name__ == '__main__':
    # No aws_runs/ evidence exists yet -- this file only documents/tests the rule, it does not
    # (and at the time of writing, cannot) select a real seed.
    example = {42: 0.60, 43: 0.90, 44: 0.75, 45: 0.90}
    print('Example (illustrative, not real data):', select_representative_main_seed(example))
