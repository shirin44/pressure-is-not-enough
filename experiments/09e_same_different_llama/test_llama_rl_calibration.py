"""Golden-record regression test against the real Llama-3-8B-Instruct RL calibration
pass evidence (aws_runs/stage9e-llama-rl-main-v1/, 2026-09-07), Stage 9e adversarial
RL, task item 3. A SHORT (12-step) diagnostic pass using the MAIN reward config
(correctness + CoT penalty, genuine adversarial pressure) on the SFT-seeded
checkpoint, KL clamp in MEASUREMENT mode -- purely to check whether Qwen's settled
breaker thresholds and clamp values are reasonable for this specific model+task+
reward combination before committing to full-length runs.

Key finding: GRAD_BREAKER=200.0 and KL_BREAKER=5.0 both hold with wide margins
(observed max grad_norm=1.24, max aggregate KL=0.089) -- safe to reuse. But the
per-token KL clamp's borrowed value (Qwen's D_MAX=2.0) would have been COMPLETELY
INERT here -- the real per-token KL pool's own fresh median+IQR calibration gives
clamp_value=0.0407, ~50x smaller. Reused GRAD_BREAKER/KL_BREAKER unchanged; adopted
the freshly-calibrated KL_CLAMP_D_MAX(~0.04) for the real BASELINE/MAIN runs instead
of assuming the borrowed value transfers, per this task's explicit instruction."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-llama-rl-main-v1' / 'stage9e_llama_rl_main.json').read_text())


def test_calibration_ran_twelve_steps_no_hard_stop():
    assert EVIDENCE['result']['terminal_step'] == 12
    assert EVIDENCE['result']['hard_stop'] is None


def test_sanity_check_passed_before_any_calibration_steps():
    cfg = EVIDENCE['config']
    assert cfg['sanity_intermediate_tracking_accuracy'] == 1.0
    assert cfg['sanity_final_answer_accuracy'] == 1.0


def test_qwen_breaker_thresholds_hold_with_wide_margin():
    telemetry = EVIDENCE['result']['telemetry']
    grad_norms = [r['grad_norm'] for r in telemetry if 'grad_norm' in r]
    kls = [r['kl'] for r in telemetry if 'kl' in r]
    assert max(grad_norms) < 200.0 / 100  # more than 100x margin
    assert max(kls) < 5.0 / 10  # more than 10x margin


def test_advantage_clamp_engaged_frequently_flagged_not_hidden():
    summary = EVIDENCE['result']['advantage_clamp_summary']
    assert summary['clamp_value'] == 0.4
    assert summary['engagement_rate'] > 0.5  # notably higher than typical -- disclosed, not concerning on its own
    assert summary['total_count'] > 0


def test_kl_clamp_measurement_mode_shows_qwens_borrowed_value_would_be_inert():
    summary = EVIDENCE['result']['kl_clamp_summary']
    assert summary['d_max'] is None  # measurement mode, confirmed
    assert summary['engaged_token_count'] == 0  # nothing clamped since D_MAX unset
    diag = summary['pool_diagnostics']
    assert diag['degenerate'] is False
    # Qwen's borrowed D_MAX=2.0 sits far above this pool's own calibrated clamp value
    assert diag['clamp_value'] < 2.0 / 20  # more than 20x smaller
    assert 0.01 < diag['clamp_value'] < 0.1  # the actual freshly-calibrated range


def test_milestone_at_final_step_shows_zero_leakage_under_adversarial_pressure():
    milestones = EVIDENCE['result']['milestones']
    assert len(milestones) == 1
    final = milestones[0]
    assert final['step'] == 12
    assert final['genuine_correct_rate'] == 1.0
    assert final['intermediate_tracking_accuracy'] == 1.0
    assert final['leakage_rate'] == 0.0
    assert final['format_valid_rate'] == 1.0


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
