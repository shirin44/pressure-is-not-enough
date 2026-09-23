"""Golden-record regression test against the Stage 9e Llama MAIN adversarial-RL
evidence (aws_runs/stage9e-llama-rl-main-v{2,3,4,5}/ = seeds 42/43/44/45) and the
post-RL causal-flip test on the selected checkpoint (seed43,
aws_runs/stage9e-llama-causal-flip-post-rl-seed43-v1/).

Pins the numbers reported in design.md AND the caveats an audit of the raw
telemetry found, so neither can drift silently:
  * MAIN vs BASELINE: no established difference (Welch p=0.200); MAIN's spread is
    tighter (SD 0.091 vs 0.184) but that is also not statistically established.
  * Milestone metrics are GREEDY evals on n=21; training rollouts are sampled at
    T=0.8. MAIN seed45's sampled rollouts collapsed to ~8-token answer-only
    completions while its greedy eval stayed at 1.0 -- pinned as a fact so it is
    never quietly dropped from the narrative.
  * Eval leakage scanner and training CoT-penalty scanner share the same banned
    vocabulary, so zero eval leakage is largely guaranteed by construction."""
import json
import statistics as st
from pathlib import Path

AWS = Path(__file__).resolve().parent / 'aws_runs'
MAIN = {s: json.loads((AWS / f'stage9e-llama-rl-main-v{i}' / 'stage9e_llama_rl_main.json').read_text())
        for s, i in ((42, 2), (43, 3), (44, 4), (45, 5))}
BASE = {s: json.loads((AWS / f'stage9e-llama-rl-baseline-v{i}' / 'stage9e_llama_rl_baseline.json').read_text())
        for s, i in ((42, 1), (43, 2), (44, 3), (45, 4))}
FLIP = json.loads((AWS / 'stage9e-llama-causal-flip-post-rl-seed43-v1' / 'stage9e_llama_causal_flip.json').read_text())


def _final(ev):
    return ev['result']['milestones'][-1]


def test_main_config_matches_baseline_except_reward_params():
    for s, ev in MAIN.items():
        c, b = ev['config'], BASE[s]['config']
        assert c['run_seed'] == s and c['phase'] == 'main'
        assert c['kl_clamp_d_max'] == b['kl_clamp_d_max'] == 0.04
        assert c['grad_breaker'] == b['grad_breaker'] == 200.0
        assert c['kl_breaker'] == b['kl_breaker'] == 5.0
        assert c['sft_adapter_sha256'] == b['sft_adapter_sha256']
        assert c['reward_params'] == '{}'  # stored as a string; score_completion_v2 defaults = penalty active
        assert "'cot_max_scale': 0.0" in b['reward_params']


def test_main_all_seeds_complete_no_breaker_zero_leakage_valid_format():
    for s, ev in MAIN.items():
        r = ev['result']
        assert r['terminal_step'] == 150 and r['hard_stop'] is None, s
        assert max(t['grad_norm'] for t in r['telemetry'] if 'grad_norm' in t) < 20.0, s
        assert max(t['kl'] for t in r['telemetry'] if 'kl' in t) < 0.01, s
        assert all(m['leakage_rate'] == 0.0 for m in r['milestones']), s
        assert _final(ev)['format_valid_rate'] == 1.0, s


def test_main_final_genuine_correct_rates_and_summary_stats():
    got = {s: _final(ev)['genuine_correct_rate'] for s, ev in MAIN.items()}
    assert got == {42: 19 / 21, 43: 1.0, 44: 17 / 21, 45: 1.0}
    v = list(got.values())
    assert abs(st.mean(v) - 0.9285714285714286) < 1e-12
    assert abs(st.stdev(v) - 0.0911835340720322) < 1e-12
    assert min(v) == 17 / 21 and max(v) == 1.0


def test_welch_main_vs_baseline_is_not_significant():
    m = [_final(ev)['genuine_correct_rate'] for ev in MAIN.values()]
    b = [_final(ev)['genuine_correct_rate'] for ev in BASE.values()]
    va, vb = st.variance(m) / 4, st.variance(b) / 4
    t = (st.mean(m) - st.mean(b)) / (va + vb) ** 0.5
    df = (va + vb) ** 2 / (va ** 2 / 3 + vb ** 2 / 3)
    assert abs(t - 1.5078271124027665) < 1e-9
    assert abs(df - 4.390823203932072) < 1e-9
    assert abs(t) < 2.7  # below the two-sided 5% critical value for df~4.4 (~2.7)
    assert st.stdev(m) < st.stdev(b)


def test_selected_checkpoint_is_seed43_via_tie_break_with_seed45():
    finals = {s: _final(ev)['genuine_correct_rate'] for s, ev in MAIN.items()}
    top = [s for s, v in finals.items() if v == max(finals.values())]
    assert top == [43, 45]  # a genuine tie; seed43 chosen by lowest-seed tie-break
    assert 'main-v3' in FLIP['config']['checkpoint_dir']


def test_audit_caveat_seed45_sampled_rollouts_collapsed_to_answer_only_while_greedy_eval_stayed_perfect():
    t = [x for x in MAIN[45]['result']['telemetry'] if 'reward' in x]  # drops the trailing train-summary row
    last = t[-50:]
    assert sum(1 for x in last if x.get('completions/min_length', 999) <= 10) / len(last) > 0.9
    assert st.mean(x['reward'] for x in last) < 0  # penalised/malformed in training
    assert _final(MAIN[45])['genuine_correct_rate'] == 1.0  # greedy eval does not show it
    # ...and the selected seed43 does NOT show the collapse
    t43 = [x for x in MAIN[43]['result']['telemetry'] if 'reward' in x][-50:]
    assert sum(1 for x in t43 if x.get('completions/min_length', 999) <= 10) == 0


def test_audit_caveat_leakage_scanners_share_banned_vocabulary():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
    from reward_v3 import ALL_BANNED_PATTERNS
    from same_different_leakage_audit import BANNED_ABBREVIATIONS, BANNED_LITERALS
    train_words = {label.lower() for _, label in ALL_BANNED_PATTERNS}
    eval_words = BANNED_LITERALS | BANNED_ABBREVIATIONS
    assert train_words == eval_words


def test_post_rl_causal_flip_is_42_of_42_both_steps_both_directions():
    a = FLIP['analysis']
    for step in ('step_2', 'step_3'):
        for grp in ('all', 'bek_to_ner_direction', 'ner_to_bek_direction'):
            g = a[step][grp]
            assert g['final_answer_tracks_flip_rate'] == 1.0
            assert g['intermediate_propagation_matches_counterfactual_rate'] == 1.0
    assert a['step_2']['all']['n'] == a['step_3']['all']['n'] == 21
    assert len(FLIP['intervention_results']) == 42
    assert all(r['flip_changes_the_predicted_answer'] for r in FLIP['intervention_results'])
    assert FLIP['config']['adapter_sha256'] == '9de5d3732242eb37c4927a6de1ee570859c878ace550bc019e7bef50c38a172b'


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
