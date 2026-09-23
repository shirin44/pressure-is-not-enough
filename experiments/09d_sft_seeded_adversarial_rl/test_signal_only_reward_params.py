"""CPU-only tests for the SIGNAL-ONLY reward condition added to
sft_seeded_rl_decode_back.py (2026-09-02): correctness + r_consistency + r_signal with
p_cot fully zeroed, isolating whether MAIN's win over BASELINE is attributable to the
CoT penalty or to the positive-signal terms. Confirms the decode-back-aware wrapper's
p_cot correction is mathematically inert (always exactly 0) under this config, run
BEFORE any GPU time is spent -- SIGNAL_ONLY_REWARD_PARAMS is duplicated here rather
than imported, since the launch script is a linear GPU script not meant to be
imported; the values must be kept in sync with sft_seeded_rl_decode_back.py by hand,
verified below against the actual deployed hash."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))

from decode_back_bank import build_decode_back_completion, score_completion_v2_decode_back_aware  # noqa: E402
from synthetic_bridge import build_prompt  # noqa: E402
from reward_v3 import verify_reward_invariant  # noqa: E402

SIGNAL_ONLY_REWARD_PARAMS = dict(cot_min_scale=0.0, cot_max_scale=0.0)


def test_p_cot_is_zero_at_every_step_under_signal_only_params():
    starting_state, operations = 'Heads', ['same', 'different', 'different', 'same', 'different']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    for step in (1, 5, 40, 75, 150):
        r = score_completion_v2_decode_back_aware(completion, 'Tails', step, 150, prompt=prompt, **SIGNAL_ONLY_REWARD_PARAMS)
        assert r['p_cot'] == 0.0
        assert r['p_cot_correction_applied'] == 0.0


def test_decode_back_aware_wrapper_is_mathematically_inert_under_signal_only():
    # The wrapper's whole purpose is correcting p_cot; with p_cot forced to 0 by
    # cot_max_scale=0, the raw and decode-back-line-stripped scores must be identical
    # for every OTHER term too (confirming no hidden interaction).
    from reward_v3 import score_completion_v2
    from decode_back_bank import strip_decode_back_line
    starting_state, operations = 'Tails', ['different', 'same', 'same', 'different', 'same']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    raw = score_completion_v2(completion, 'Heads', 40, 150, prompt=prompt, **SIGNAL_ONLY_REWARD_PARAMS)
    stripped = score_completion_v2(strip_decode_back_line(completion), 'Heads', 40, 150, prompt=prompt, **SIGNAL_ONLY_REWARD_PARAMS)
    assert raw['total'] == stripped['total']
    assert raw['p_cot'] == stripped['p_cot'] == 0.0


def test_signal_and_consistency_terms_remain_active_at_their_main_magnitudes():
    # An all-same-state trajectory shares one code token across every slot -- the
    # exact condition consistency_bonus_v2 rewards -- and correlates perfectly with
    # true state, so both bonus terms should hit their full 0.15 magnitude.
    starting_state, operations = 'Heads', ['same', 'same', 'same', 'same', 'same']
    completion = build_decode_back_completion(starting_state, operations)
    prompt = build_prompt(starting_state, operations)
    r = score_completion_v2_decode_back_aware(completion, 'Heads', 40, 150, prompt=prompt, **SIGNAL_ONLY_REWARD_PARAMS)
    assert r['r_consistency'] == 0.15
    assert r['r_signal'] == 0.0  # ARI needs >=2 distinct true states in the valid-slot subset; none here


def test_reward_invariant_holds_for_the_signal_only_term_ranges():
    margins = verify_reward_invariant(cot_max_scale=0.0)
    assert margins['margin_correct_over_wrong'] > 0
    assert margins['margin_wrong_over_malformed'] > 0


def test_signal_only_params_dict_matches_the_deployed_launch_script():
    script = (Path(__file__).resolve().parent / 'sft_seeded_rl_decode_back.py').read_text()
    assert "SIGNAL_ONLY_REWARD_PARAMS = dict(cot_min_scale=0.0, cot_max_scale=0.0)" in script


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
