"""Golden-record regression tests against the Stage 9d decode-back-seeded BASELINE run
(aws_runs/stage9d-decode-back-baseline-v1/, 2026-09-02): the correctness-only control,
seeded from stage9c-sft-decode-back-v1 (final-answer accuracy 81.0%) instead of the
original weak checkpoint (~28.6%), using the decode-back-aware reward wrapper. Captures
the two headline findings: genuine_correct_among_nonliteral fluctuates in a real but
bounded band under plain correctness-only RL pressure (never collapsing anywhere near
the original weak-checkpoint floor), while decode_back_matches_own_trace_rate stays
PERFECT at every single milestone -- the structural fix holds completely under this
condition, so any decline seen is attributable to tracking-quality drift, not the
disconnect bug returning."""
import json
from pathlib import Path

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())
RESULT = EVIDENCE['result']


def test_config_used_the_decode_back_checkpoint_with_pinned_hash():
    cfg = EVIDENCE['config']
    assert cfg['seed_checkpoint'] == 'stage9c-sft-decode-back-v1'
    assert cfg['stage9c_adapter_sha256'] == '2b3a26d5ff8584ae305d446a4b954030115e0196108dd780a4579ea736ec0274'
    assert cfg['reward_function'] == 'score_completion_v2_decode_back_aware'


def test_preflight_reward_check_confirmed_clean_before_launch():
    check = EVIDENCE['config']['pre_flight_reward_check']
    assert check['all_corrections_non_negative'] is True


def test_zero_step_sanity_check_reproduced_the_sft_checkpoints_own_reported_numbers():
    cfg = EVIDENCE['config']
    assert abs(cfg['sanity_final_answer_accuracy'] - 0.8095238095238095) < 1e-9
    assert cfg['sanity_decode_back_matches_own_trace_rate'] == 1.0


def test_ran_the_full_150_steps_with_no_breaker():
    assert RESULT['terminal_step'] == 150
    assert RESULT['hard_stop'] is None


def test_clamps_engaged_at_stable_non_degenerate_rates():
    adv = RESULT['advantage_clamp_summary']
    kl = RESULT['kl_clamp_summary']
    assert 0.0 < adv['engagement_rate'] < 1.0
    assert kl['mode'] == 'ACTIVE' and kl['d_max'] == 2.0
    assert 0.0 < kl['engagement_rate'] < 1.0


def test_decode_back_trace_fidelity_never_wavered_across_all_30_milestones():
    milestones = RESULT['milestones']
    assert len(milestones) == 30
    assert all(m['decode_back_matches_own_trace_rate'] == 1.0 for m in milestones)


def test_genuine_correctness_fluctuates_but_never_collapses_toward_the_original_weak_floor():
    milestones = RESULT['milestones']
    gc = [m['genuine_correct_among_structural_nonliteral'] for m in milestones]
    ORIGINAL_WEAK_CHECKPOINT_BASELINE = 0.2857142857142857
    assert min(gc) > ORIGINAL_WEAK_CHECKPOINT_BASELINE * 1.5  # stays comfortably above, even at its lowest
    assert max(gc) == milestones[0]['genuine_correct_among_structural_nonliteral']  # peak at/near the start
    assert gc[0] > gc[-1]  # ends somewhat below where it started
    assert gc[-1] > 0.6  # but the final value is still a real, substantial correctness rate


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
