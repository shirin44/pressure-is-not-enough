"""Prespecified Stage 9e four-seed BASELINE-vs-MAIN comparison."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_DIRS = [f"stage9e-llama-rl-baseline-v{i}" for i in range(1, 5)]
MAIN_DIRS = [f"stage9e-llama-rl-main-v{i}" for i in range(2, 6)]


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta via the Numerical Recipes continued fraction."""
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
        raise RuntimeError("incomplete-beta continued fraction did not converge")

    if not 0.0 <= x <= 1.0:
        raise ValueError(x)
    if x in (0.0, 1.0):
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * fraction(a, b, x) / a
    return 1.0 - front * fraction(b, a, 1.0 - x) / b


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float, float]:
    va, vb = statistics.variance(a), statistics.variance(b)
    sa, sb = va / len(a), vb / len(b)
    t = (statistics.mean(a) - statistics.mean(b)) / math.sqrt(sa + sb)
    df = (sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
    p = regularized_incomplete_beta(df / (df + t * t), df / 2.0, 0.5)
    return t, p, df


def load_final(dirname: str, filename: str) -> tuple[int, float]:
    path = ROOT / "aws_runs" / dirname / filename
    report = json.loads(path.read_text())
    result = report["result"]
    assert result["terminal_step"] == 150
    assert result["hard_stop"] is None
    final = result["milestones"][-1]
    assert final["step"] == 150
    assert final["leakage_rate"] == 0.0
    assert final["format_valid_rate"] == 1.0
    return int(report["config"]["run_seed"]), float(final["genuine_correct_rate"])


baseline = [load_final(d, "stage9e_llama_rl_baseline.json") for d in BASELINE_DIRS]
main = [load_final(d, "stage9e_llama_rl_main.json") for d in MAIN_DIRS]
baseline_values = [v for _, v in baseline]
main_values = [v for _, v in main]
t_statistic, p_value, degrees_of_freedom = welch_t_test(main_values, baseline_values)
best_seed, best_rate = max(main, key=lambda item: item[1])

summary = {
    "baseline": {
        "per_seed": dict(baseline),
        "mean": statistics.mean(baseline_values),
        "sample_sd": statistics.stdev(baseline_values),
        "min": min(baseline_values),
        "max": max(baseline_values),
    },
    "main": {
        "per_seed": dict(main),
        "mean": statistics.mean(main_values),
        "sample_sd": statistics.stdev(main_values),
        "min": min(main_values),
        "max": max(main_values),
    },
    "welch_t_test_main_vs_baseline": {
        "t_statistic": t_statistic,
        "p_value_two_sided": p_value,
        "degrees_of_freedom": degrees_of_freedom,
    },
    "selected_main_checkpoint": {"seed": best_seed, "genuine_correct_rate": best_rate},
}
print(json.dumps(summary, indent=2))
