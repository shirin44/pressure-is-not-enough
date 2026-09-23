"""Evidence-based recalibration of the GRAD_BREAKER threshold (2026-09-03), following
the same style already used for the advantage clamp (Grubbs'-bound derivation) and
the KL clamp (median+IQR calibration from a real telemetry pool) elsewhere in this
stage -- not picking a round number.

Motivation: 5 of 7 recent MAIN launches (71.4%) hit GRAD_BREAKER=50.0 via an
already-diagnosed, non-mysterious mechanism (a rollout group where all 8 completions
hit the 256-token generation ceiling with elevated entropy ~1.0-1.2 produces an
elevated grad_norm). BASELINE's own single completed run touched grad_norm=47.5 via
the IDENTICAL mechanism three times without ever crossing 50 -- meaning the current
threshold sits right at the edge of a recurring, apparently-benign training dynamic,
not cleanly separating healthy from pathological training.

This module pulls every long-completion-group grad_norm value from every SAME-CONFIG
(LR=2e-5, advantage clamp=0.4, KL clamp D_MAX=2.0, GROUP_SIZE=8) run's telemetry
available in this stage's evidence, split into two classes by a principled,
non-circular operational definition:
  - SURVIVED: the value comes from a COMPLETED run (reached its target step count
    without ever hitting the script's own non-finite-loss safety check) -- these
    grad_norm values are, by construction, NOT associated with any observed
    divergence; training continued normally afterward in every case.
  - BREAKER-TRIGGERING: the value at the exact step GRAD_BREAKER halted a run.
Deliberately avoids the words "benign"/"pathological" as class labels until the
distributions are actually compared -- that's the question being asked, not the
label being assumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LONG_COMPLETION_MEAN_LENGTH_THRESHOLD = 200

# Only same-config (LR=2e-5, both clamps at their validated values, 150-step target,
# GROUP_SIZE=8) runs are included -- mixing in earlier diagnostic-scale runs at
# different LRs/clamp values/step budgets would conflate regimes. grad_norm is a
# property of the backward pass alone (no LR term in its own computation -- LR only
# enters at the optimizer step, strictly after grad_norm is measured), so this
# restriction is about keeping the COMPLETION-LENGTH/ENTROPY dynamics comparable, not
# about grad_norm's own scale.
SAME_CONFIG_RUN_PATHS = [
    ('BASELINE', 'stage9d-decode-back-baseline-v1', 'stage9d_decode_back_baseline.json'),
    ('MAIN_v1', 'stage9d-decode-back-main-v1', 'stage9d_decode_back_main.json'),
    ('MAIN_v2', 'stage9d-decode-back-main-v2', 'stage9d_decode_back_main.json'),
    ('MAIN_v3', 'stage9d-decode-back-main-v3', 'stage9d_decode_back_main.json'),
    ('MAIN_v4', 'stage9d-decode-back-main-v4', 'stage9d_decode_back_main.json'),
    ('MAIN_v5', 'stage9d-decode-back-main-v5', 'stage9d_decode_back_main.json'),
    ('MAIN_v6', 'stage9d-decode-back-main-v6', 'stage9d_decode_back_main.json'),
    ('MAIN_v7', 'stage9d-decode-back-main-v7', 'stage9d_decode_back_main.json'),
    ('SIGNAL_ONLY', 'stage9d-decode-back-signal_only-v1', 'stage9d_decode_back_signal_only.json'),
    ('PENALTY_ONLY', 'stage9d-decode-back-penalty_only-v1', 'stage9d_decode_back_penalty_only.json'),
    # Older, pre-decode-back-checkpoint full runs at the SAME validated LR/clamp
    # config -- included since the completion-length/entropy -> grad_norm mechanism
    # is about generation dynamics, not which checkpoint or reward-wrapper is used.
    ('FULL_BASELINE150_v2', 'stage9d-full-baseline150-v2-both-clamps', 'stage9d_full_baseline150_bothclamps.json'),
    ('FULL_MAIN150_v1', 'stage9d-full-main150-v1', 'stage9d_full_main150.json'),
]


def load_evidence(aws_runs_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for name, subdir, filename in SAME_CONFIG_RUN_PATHS:
        path = aws_runs_dir / subdir / filename
        if path.is_file():
            out[name] = json.loads(path.read_text())
    return out


def extract_long_completion_events(evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per long-completion-group telemetry step across every loaded run,
    tagged with which run it came from and whether it was the exact step that
    triggered that run's breaker (if any)."""
    events = []
    for name, ev in evidence.items():
        result = ev['result']
        hard_stop_step = result['hard_stop']['step'] if result['hard_stop'] else None
        completed = result['hard_stop'] is None
        for row in result['telemetry']:
            length = row.get('completions/mean_length')
            if length is None or length < LONG_COMPLETION_MEAN_LENGTH_THRESHOLD:
                continue
            step = row['physical_step']
            is_breaker_step = (hard_stop_step is not None and step == hard_stop_step)
            events.append({
                'run': name, 'step': step, 'grad_norm': row['grad_norm'], 'kl': row.get('kl'),
                'entropy': row.get('entropy'), 'mean_length': length,
                'clipped_ratio': row.get('completions/clipped_ratio'),
                'class': 'breaker_triggering' if is_breaker_step else ('survived' if completed else 'mid_run_no_break'),
            })
    return events


VALIDATION_BATCH_RUN_PATHS = [
    ('seed42', 'stage9d-decode-back-main-v8', 'stage9d_decode_back_main.json'),
    ('seed43', 'stage9d-decode-back-main-v9', 'stage9d_decode_back_main.json'),
    ('seed44', 'stage9d-decode-back-main-v10', 'stage9d_decode_back_main.json'),
    ('seed45', 'stage9d-decode-back-main-v11', 'stage9d_decode_back_main.json'),
]
# Loaded separately from SAME_CONFIG_RUN_PATHS rather than merged in: these 4 runs
# were launched at STAGE9D_GRAD_BREAKER=80.0 (not the default 50.0) specifically to
# validate the earlier empirical-max-plus-margin proposal, so their own hard_stop/
# completed status is not comparable to the main population's breaker_triggering vs
# survived classification -- only their raw (grad_norm, mean_length, entropy) values
# are reused here.


def load_validation_batch_evidence(aws_runs_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for name, subdir, filename in VALIDATION_BATCH_RUN_PATHS:
        path = aws_runs_dir / subdir / filename
        if path.is_file():
            out[name] = json.loads(path.read_text())
    return out


def extract_all_telemetry_pairs(evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every (mean_length, grad_norm, entropy) telemetry row, WITHOUT the >=200
    long-completion filter -- used only to characterize the length/entropy -> grad_norm
    relationship across the full observed length range (73-256), not to select
    breaker-calibration events."""
    out = []
    for name, ev in evidence.items():
        for row in ev['result']['telemetry']:
            length = row.get('completions/mean_length')
            gn = row.get('grad_norm')
            ent = row.get('entropy')
            if length and gn and ent and length > 0 and gn > 0 and ent > 0:
                out.append({'run': name, 'mean_length': length, 'grad_norm': gn, 'entropy': ent})
    return out


# Fitted via OLS on log(grad_norm) ~ log(mean_length) across all 1142 telemetry rows
# (length 73-256) from the 12 SAME_CONFIG_RUN_PATHS runs: grad_norm ~ length^2.706.
# This is the single most defensible normalization power (comes directly from a
# regression fit, not a grid search for zero residual correlation -- a grid search
# over p in {0.5..4.0} independently landed at p=2.5, giving corr=-0.019 vs the fitted
# exponent's corr=-0.069; both are close enough to zero that the OLS estimate is
# preferred as the more principled choice). See design.md for the full derivation,
# including why a length+entropy MULTIVARIATE fit was rejected (r(length,entropy)=
# 0.933 in the full population -> VIF=7.7, severe multicollinearity: the multivariate
# fit's length exponent flips sign to -0.609 purely as a collinearity artifact, not a
# real effect reversal).
NORMALIZATION_POWER = 2.706


def normalized_grad_norm(grad_norm: float, mean_length: float, power: float = NORMALIZATION_POWER) -> float:
    return grad_norm / (mean_length ** power)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    survived = sorted(e['grad_norm'] for e in events if e['class'] == 'survived')
    breaker = sorted(e['grad_norm'] for e in events if e['class'] == 'breaker_triggering')
    mid_run = sorted(e['grad_norm'] for e in events if e['class'] == 'mid_run_no_break')
    return {
        'n_survived': len(survived), 'survived_min': min(survived) if survived else None,
        'survived_max': max(survived) if survived else None, 'survived_values': survived,
        'n_breaker_triggering': len(breaker), 'breaker_min': min(breaker) if breaker else None,
        'breaker_max': max(breaker) if breaker else None, 'breaker_values': breaker,
        'n_mid_run_no_break': len(mid_run), 'mid_run_values': mid_run,
    }


def key_22_events(evidence: dict[str, dict[str, Any]], validation_evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The 5 original breaker-triggering events + the 17 validation-batch long-
    completion events (the population the length-normalization proposal must be
    checked against, per the task's own framing)."""
    breakers = [e for e in extract_long_completion_events(evidence) if e['class'] == 'breaker_triggering']
    out = [{'run': e['run'], 'grad_norm': e['grad_norm'], 'mean_length': e['mean_length'], 'entropy': e['entropy']}
           for e in breakers]
    for name, ev in validation_evidence.items():
        for row in ev['result']['telemetry']:
            if row.get('completions/mean_length', 0) >= LONG_COMPLETION_MEAN_LENGTH_THRESHOLD:
                out.append({'run': name, 'grad_norm': row['grad_norm'],
                            'mean_length': row['completions/mean_length'], 'entropy': row.get('entropy')})
    return out


def coefficient_of_variation(values: list[float]) -> float:
    import statistics
    return statistics.stdev(values) / statistics.mean(values)


def length_normalization_report(events_22: list[dict[str, Any]], power: float = NORMALIZATION_POWER) -> dict[str, Any]:
    """Checks the task's own stated validation criterion: does normalizing by length^p
    collapse the 22 key events into a tighter, more stable range? Reports raw vs
    normalized spread so the answer is a number, not an impression."""
    raw = [e['grad_norm'] for e in events_22]
    lengths = [e['mean_length'] for e in events_22]
    norm = [normalized_grad_norm(g, l, power) for g, l in zip(raw, lengths)]
    n_at_256 = sum(1 for l in lengths if l == 256.0)
    return {
        'n_events': len(events_22),
        'n_at_length_256': n_at_256,
        'raw_cv': coefficient_of_variation(raw), 'raw_spread_ratio': max(raw) / min(raw),
        'normalized_cv': coefficient_of_variation(norm), 'normalized_spread_ratio': max(norm) / min(norm),
        'cv_unchanged': abs(coefficient_of_variation(raw) - coefficient_of_variation(norm)) < 1e-6,
    }
