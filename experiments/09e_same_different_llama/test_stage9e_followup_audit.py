"""Golden-record test for the 2026-09-21 GPU follow-up: four-seed post-RL causal flip (SFT-only + MAIN 42/43/44/45)
and the step-0 rollout regeneration from the SFT adapter (aws_runs/stage9e-step0-rollout-regen-v1/).
Pins the numbers reported in design.md so a later edit cannot silently change them."""
import json
from pathlib import Path

AWS = Path(__file__).resolve().parent / 'aws_runs'


def _flip(name):
    return json.loads((AWS / name / 'stage9e_llama_causal_flip.json').read_text())


FLIPS = {'sft': _flip('stage9e-llama-causal-flip-v1'), **{s: _flip(f'stage9e-llama-causal-flip-post-rl-seed{s}-v1') for s in (42, 43, 44, 45)}}
REGEN = json.loads((AWS / 'stage9e-step0-rollout-regen-v1' / 'step0_rollout_regen.json').read_text())


def _tracks(ev):
    return sum(r['final_answer_classification'] == 'tracks_flip_genuine_causal_use' for r in ev['intervention_results'])


def _trace(ev):
    return sum(bool(r['propagation_matches_counterfactual']) for r in ev['intervention_results'])


def test_flip_counts_per_checkpoint():
    assert {k: (_tracks(v), _trace(v), len(v['intervention_results'])) for k, v in FLIPS.items()} == {
        'sft': (42, 42, 42), 42: (42, 42, 42), 43: (42, 42, 42), 44: (41, 41, 42), 45: (42, 41, 42)}


def test_flip_pooled_main_and_no_noop_interventions():
    assert sum(_tracks(FLIPS[s]) for s in (42, 43, 44, 45)) == 167
    assert sum(_trace(FLIPS[s]) for s in (42, 43, 44, 45)) == 166
    for ev in FLIPS.values():
        assert all(r['flip_changes_the_predicted_answer'] for r in ev['intervention_results'])


def test_flip_adapters_are_the_four_distinct_main_checkpoints():
    shas = [FLIPS[s]['config']['adapter_sha256'] for s in (42, 43, 44, 45)]
    assert len(set(shas)) == 4
    assert shas[1].startswith('9de5d373')
    assert FLIPS['sft']['config']['adapter_sha256'].startswith('3eb41be3')


def test_flip_single_failure_seed44_is_a_step5_tracking_slip_not_an_ignored_flip():
    bad = [r for r in FLIPS[44]['intervention_results'] if r['final_answer_classification'] != 'tracks_flip_genuine_causal_use']
    assert len(bad) == 1
    r = bad[0]
    assert r['flip_at_step'] == 3 and list(r['subsequent_tokens']) == ['bek', 'ner']  # step 4 flipped correctly, step 5 wrongly changed
    assert list(r['expected_counterfactual_tokens']) == ['bek', 'bek']


def test_flip_seed45_trace_mismatch_is_a_malformed_step_label():
    bad = [r for r in FLIPS[45]['intervention_results'] if not r['propagation_matches_counterfactual']]
    assert len(bad) == 1 and 'Step : ' in bad[0]['full_corrupted_completion'] and bad[0]['extracted_answer'] == 'bek'


def test_regen_config_matches_rl_sampling_settings():
    c = REGEN['config']
    assert (c['temperature'], c['top_p'], c['top_k'], c['max_new_tokens'], c['n_train_scenarios'], c['n_per_prompt']) == (0.8, 0.95, 0, 256, 43, 16)
    assert c['adapter_sha256'].startswith('3eb41be3')  # the SFT adapter
    assert set(REGEN['arms']) == {'train_format', 'chat_eval_harness', 'chat_sft_exact'}


def test_regen_penalty_never_binds_at_step0_in_any_arm():
    for arm, d in REGEN['arms'].items():
        s = d['summary']
        assert s['n_completions'] == 688
        assert s['nonzero_p_cot (banned pattern in TRAIN span)'] == 0.0, arm
        assert s['banned_pattern_anywhere_whole_completion'] == 0.0 and s['eval_scanner_flag'] == 0.0 and s['case_insensitive_word_anywhere'] == 0.0, arm
        assert s['groups_with_any_nonzero_p_cot'] == 0.0, arm


def test_regen_raw_prompt_arm_is_mostly_malformed_chat_arms_are_not():
    a = {k: v['summary'] for k, v in REGEN['arms'].items()}
    assert a['train_format']['format_valid'] < 0.1 and a['train_format']['correct'] < 0.05 and a['train_format']['hit_cap_256'] > 0.8
    assert a['train_format']['mean_reward_main_step1'] < -4.5
    assert a['chat_eval_harness']['format_valid'] == a['chat_sft_exact']['format_valid'] == 1.0
    assert a['chat_eval_harness']['correct'] < a['chat_sft_exact']['correct']  # the eval harness's double BOS costs accuracy under sampling
    assert a['train_format']['ended_answer_stop'] == 0.0 and a['chat_sft_exact']['ended_answer_stop'] > 0.4


DRIFT = json.loads((AWS / 'stage9e-raw-vs-chat-drift-v1' / 'raw_vs_chat_drift.json').read_text())


def test_drift_controls_are_exactly_zero_and_adapters_are_the_expected_ones():
    sh = DRIFT['C_drift']['adapter_sha256']
    assert sh['sft'].startswith('3eb41be3') and sh['main_seed43_frozen_ref_adapter'] == sh['sft'] == sh['sft_reloaded_noise_floor']
    for c in ('main_seed43_frozen_ref_adapter', 'sft_reloaded_noise_floor'):
        for f in ('chat_sft', 'raw'):
            assert DRIFT['C_drift']['stats'][c][f]['mean_abs_logratio'] == 0.0


def test_rl_barely_moved_chat_format_likelihood_but_moved_raw_format_likelihood():
    st_ = DRIFT['C_drift']['stats']
    for s in ('main_seed42', 'main_seed43', 'main_seed44', 'main_seed45'):
        assert abs(st_[s]['chat_sft']['mean_per_token_logratio(adapter-SFT)']) < 0.003, s
        assert st_[s]['raw']['mean_per_token_logratio(adapter-SFT)'] > 0.03, s
    assert st_['main_seed42']['raw']['mean_per_token_logratio(adapter-SFT)'] > 0.25


def test_sft_greedy_heldout21_chat_perfect_raw_completely_malformed():
    b = DRIFT['B_greedy_heldout21_sft_adapter']
    for k in ('chat_harness', 'chat_single'):
        assert b[k]['summary']['accuracy(final answer correct)'] == 1.0 and b[k]['summary']['malformed_rate(no parseable final answer)'] == 0.0
    assert b['chat_harness']['summary']['identical_to_persisted_sft_tier_b'] == 21
    for k in ('raw', 'raw_stop'):
        assert b[k]['summary']['malformed_rate(no parseable final answer)'] == 1.0 and b[k]['summary']['accuracy(final answer correct)'] == 0.0
    assert b['raw']['summary']['hit_cap_256'] == 1.0


if __name__ == '__main__':
    import inspect
    tests = [o for n, o in list(globals().items()) if n.startswith('test_') and inspect.isfunction(o)]
    failures = []
    for t in tests:
        try:
            t(); print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append(t.__name__); print(f'FAILED: {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
