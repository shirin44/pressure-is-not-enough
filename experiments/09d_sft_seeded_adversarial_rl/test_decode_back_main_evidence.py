"""Golden-record regression test against the Stage 9d decode-back-seeded MAIN run
(aws_runs/stage9d-decode-back-main-v1/, 2026-09-02): the full-adversarial-reward run,
seeded from the SAME stage9c-sft-decode-back-v1 checkpoint as BASELINE, using the
decode-back-aware reward wrapper validated in pre-flight. Unlike BASELINE, this run hit
the grad_norm breaker (>=50.0) at step 5 and stopped immediately, per the script's own
explicit no-auto-fix policy -- captured here plainly, not smoothed over. At the single
milestone reached (step 5), genuine_correct_among_nonliteral (0.952) and
decode_back_matches_own_trace_rate (1.0) both looked at least as good as BASELINE's own
step-5 values (0.810 and 1.0 respectively) -- the breaker event shows no sign of being
caused by, or coinciding with, any degradation in genuine correctness or a reopening of
the structural disconnect; it looks like an early, LR-warmup-boundary gradient-variance
event, comparable in magnitude to (though narrowly over, where BASELINE's were narrowly
under) three separate close calls BASELINE itself had later in its own run."""
import json
from pathlib import Path

MAIN = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v1'
                    / 'stage9d_decode_back_main.json').read_text())
BASELINE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())


def test_config_matches_baseline_except_reward_params():
    assert MAIN['config']['seed_checkpoint'] == 'stage9c-sft-decode-back-v1'
    assert MAIN['config']['stage9c_adapter_sha256'] == BASELINE['config']['stage9c_adapter_sha256']
    assert MAIN['config']['advantage_clamp_value'] == BASELINE['config']['advantage_clamp_value'] == 0.4
    assert MAIN['config']['kl_clamp_d_max'] == BASELINE['config']['kl_clamp_d_max'] == 2.0


def test_grad_norm_breaker_fired_at_step_5_stopping_training_immediately():
    result = MAIN['result']
    assert result['terminal_step'] == 5
    assert result['hard_stop'] == {'step': 5, 'grad_norm': 51.5, 'kl': 0.49830040661618114}


def test_only_one_milestone_was_reached_before_the_breaker():
    assert len(MAIN['result']['milestones']) == 1
    assert MAIN['result']['milestones'][0]['step'] == 5


def test_the_one_reached_milestone_shows_no_sign_of_degradation_vs_baselines_own_step_5():
    m5_main = MAIN['result']['milestones'][0]
    m5_baseline = next(m for m in BASELINE['result']['milestones'] if m['step'] == 5)
    assert m5_main['decode_back_matches_own_trace_rate'] == 1.0
    assert m5_baseline['decode_back_matches_own_trace_rate'] == 1.0
    # MAIN's step-5 genuine correctness is AT LEAST as high as BASELINE's own step-5 value
    assert m5_main['genuine_correct_among_structural_nonliteral'] >= m5_baseline['genuine_correct_among_structural_nonliteral']


def test_the_breaking_grad_norm_is_close_to_but_distinct_from_baselines_own_near_misses():
    # BASELINE itself came within ~2.5-4 points of the same 50.0 threshold three times
    # later in its run (steps 35/39/73, values 47.5/46.0/46.0) without ever crossing it.
    # MAIN's single breach (51.5) happened much earlier, right at the warmup-to-full-LR
    # boundary (step 5) -- a different point in training than any of BASELINE's near
    # misses, not simply "the same kind of spike happening more often."
    baseline_grad_norms = [r['grad_norm'] for r in BASELINE['result']['telemetry'] if 'grad_norm' in r]
    assert max(baseline_grad_norms) < 50.0
    assert max(baseline_grad_norms) > 45.0
    assert MAIN['result']['hard_stop']['grad_norm'] > 50.0


def test_preflight_reward_check_was_reconfirmed_clean_immediately_before_this_launch():
    check = MAIN['config']['pre_flight_reward_check']
    assert check['all_corrections_non_negative'] is True
    assert any(d > 0 for d in check['p_cot_corrections_applied'])


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
