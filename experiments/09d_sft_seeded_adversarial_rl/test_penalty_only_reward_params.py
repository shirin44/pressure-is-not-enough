"""CPU-only tests for the PENALTY-ONLY reward condition added to
sft_seeded_rl_decode_back.py (2026-09-03): correctness + p_cot at MAIN's own defaults,
with r_consistency and r_signal fully zeroed -- the exact mirror image of SIGNAL_ONLY.
Completes the 2x2 factorial (bonus: yes/no x penalty: yes/no). Confirms p_cot is
genuinely active under this config (unlike under SIGNAL_ONLY, where it's forced to 0)
and that the decode-back-aware wrapper is doing real, growing protective work as
cot_scale ramps over training -- run BEFORE any GPU time is spent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))

from decode_back_bank import build_decode_back_completion, score_completion_v2_decode_back_aware  # noqa: E402
from synthetic_bridge import build_prompt  # noqa: E402
from reward_v3 import verify_reward_invariant  # noqa: E402

PENALTY_ONLY_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0)


def _sample_completion():
    starting_state, operations = 'Heads', ['same', 'different', 'different', 'same', 'different']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    return completion, prompt


def test_r_consistency_and_r_signal_are_zero_at_every_step():
    completion, prompt = _sample_completion()
    for step in (1, 5, 40, 75, 150):
        r = score_completion_v2_decode_back_aware(completion, 'Tails', step, 150, prompt=prompt, **PENALTY_ONLY_REWARD_PARAMS)
        assert r['r_consistency'] == 0.0
        assert r['r_signal'] == 0.0


def test_decode_back_aware_wrapper_does_real_growing_protective_work():
    # Opposite of SIGNAL_ONLY: p_cot is fully active here, so the wrapper's correction
    # (the false-positive penalty removed from the decode-back line's own literal word)
    # should be strictly positive and grow as cot_scale ramps over training.
    completion, prompt = _sample_completion()
    corrections = []
    for step in (5, 40, 75, 150):
        r = score_completion_v2_decode_back_aware(completion, 'Tails', step, 150, prompt=prompt, **PENALTY_ONLY_REWARD_PARAMS)
        assert r['p_cot_correction_applied'] > 0.0
        corrections.append(r['p_cot_correction_applied'])
    assert corrections == sorted(corrections)  # monotonically non-decreasing with cot_scale


def test_genuine_literal_leakage_in_the_trace_is_still_penalized():
    completion, prompt = _sample_completion()
    bad = completion.replace('State: Nib', 'State: Heads', 1)
    r = score_completion_v2_decode_back_aware(bad, 'Tails', 150, 150, prompt=prompt, **PENALTY_ONLY_REWARD_PARAMS)
    assert r['p_cot'] > 0.0
    assert r['banned_count'] >= 1.0


def test_reward_invariant_holds_for_the_penalty_only_term_ranges():
    margins = verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0)
    assert margins['margin_correct_over_wrong'] > 0
    assert margins['margin_wrong_over_malformed'] > 0


def test_penalty_only_params_dict_matches_the_deployed_launch_script():
    script = (Path(__file__).resolve().parent / 'sft_seeded_rl_decode_back.py').read_text()
    assert "PENALTY_ONLY_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0)" in script


def test_preflight_check_now_passes_phase_reward_params_not_always_main_defaults():
    # Regression guard for the disclosed gap fixed alongside this phase: the pre-flight
    # check call must pass **_reward_params, not call the wrapper with implicit
    # (MAIN-style) defaults regardless of phase.
    script = (Path(__file__).resolve().parent / 'sft_seeded_rl_decode_back.py').read_text()
    assert "score_completion_v2_decode_back_aware(c, y, 1, FULL_STEPS, prompt=p, **_reward_params)" in script


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
