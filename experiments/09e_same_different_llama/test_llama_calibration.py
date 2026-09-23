"""Golden-record regression test against the real Llama-3-8B-Instruct calibration-
pass evidence (aws_runs/stage9e-llama-calibration-v1/, 2026-09-07), Stage 9e Part A,
Step 3. A SHORT (12-step) diagnostic-only GRPO pass, no breaker enforcement, no
checkpoint produced -- purely measures real gradient-norm/KL telemetry and real peak
GPU memory during actual training steps (fresh LoRA on the untouched base model, no
SFT-seed checkpoint, reusing the existing Coin Flip task/reward pipeline as a
placeholder per the task's own instruction).

Key, carefully-qualified finding: raw grad_norm (0.14-0.46) and KL (0.0-0.015) are
both far below Qwen's typical settled-threshold range -- but the telemetry itself
(reward means near the -5.0 malformed-completion floor, clipped_ratio up to 0.875,
low entropy 0.26-0.46) shows this is an UNTRAINED base policy producing largely
malformed, low-diversity rollouts on a task format it has never seen (no SFT seed),
not evidence of a genuine Llama-vs-Qwen gradient-scale difference. This test captures
that qualification directly, not just the raw numbers, so a future reader doesn't
mistake "no breaker would have fired" for "Llama's thresholds are calibrated." """
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-llama-calibration-v1' / 'stage9e_llama_calibration.json').read_text())

QWEN_KL_BREAKER = 5.0
QWEN_GRAD_BREAKER = 200.0


def test_ran_the_full_twelve_steps():
    assert EVIDENCE['terminal_step'] == 12
    assert EVIDENCE['config']['calibration_steps'] == 12
    assert len(EVIDENCE['telemetry']) == 13  # 12 per-step logs + 1 final train_runtime summary row


def test_no_qwen_breaker_would_have_fired():
    a = EVIDENCE['analysis']
    assert a['qwen_grad_breaker_would_have_fired'] is False
    assert a['qwen_kl_breaker_would_have_fired'] is False
    assert a['any_qwen_breaker_would_have_fired'] is False


def test_observed_grad_norm_and_kl_are_far_below_qwens_thresholds():
    a = EVIDENCE['analysis']
    assert a['grad_norm_max'] < QWEN_GRAD_BREAKER / 100
    assert a['kl_max'] < QWEN_KL_BREAKER / 100


def test_reward_means_confirm_this_was_an_untrained_low_engagement_policy():
    # The qualification that matters: low grad_norm/KL here reflects an untrained
    # base model producing largely malformed rollouts, not a calibrated measurement
    # of Llama's true gradient scale under real learning.
    per_step = [r for r in EVIDENCE['telemetry'] if 'rewards/diagnostic_reward/mean' in r]
    assert len(per_step) == 12
    reward_means = [r['rewards/diagnostic_reward/mean'] for r in per_step]
    # score_completion_v2's malformed-completion floor is -5.0 (r_task); every step's
    # mean reward sits close to or below that, consistent with mostly-malformed output
    assert all(m < -2.0 for m in reward_means)
    assert sum(1 for m in reward_means if m <= -5.0) >= 5


def test_entropy_is_low_consistent_with_low_rollout_diversity():
    per_step = [r for r in EVIDENCE['telemetry'] if 'entropy' in r]
    entropies = [r['entropy'] for r in per_step]
    assert all(e < 0.5 for e in entropies)


def test_most_steps_show_high_clipped_ratio_completions_hitting_the_token_ceiling():
    per_step = [r for r in EVIDENCE['telemetry'] if 'completions/clipped_ratio' in r]
    clipped = [r['completions/clipped_ratio'] for r in per_step]
    assert sum(1 for c in clipped if c >= 0.5) >= 8


def test_peak_training_memory_measured_and_fits_with_real_headroom():
    mem = EVIDENCE['memory']
    assert mem['peak_reserved_during_training_mib'] > mem['allocated_after_lora_load_mib']
    headroom = mem['total_gpu_mib'] - mem['peak_reserved_during_training_mib']
    assert headroom > 5000  # real headroom remains, but notably tighter than the static-load check alone suggested
    assert mem['headroom_remaining_mib'] == headroom


def test_real_training_memory_meaningfully_exceeds_the_static_load_footprint():
    # The whole point of measuring this separately from Part A Step 2's static
    # 8-bit-load check (~8,666 MiB): gradients + optimizer state + generation
    # KV-cache add real, non-trivial overhead on top of the static footprint.
    mem = EVIDENCE['memory']
    added_by_training = mem['peak_reserved_during_training_mib'] - mem['allocated_after_lora_load_mib']
    assert added_by_training > 2000


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
