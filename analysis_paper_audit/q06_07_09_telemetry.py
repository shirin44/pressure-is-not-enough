"""Q6/Q7/Q9 from persisted per-step telemetry (one telemetry row per optimizer step; one GRPO group of 8 per step) and per-milestone evals."""
import json, statistics as st, csv
from common import *
rows_out = []; md = []
def mm(v): return (min(v), st.mean(v), max(v))
print('== Q6: per run, over 150 steps (one group of 8 completions per optimizer step) ==')
hdr = 'run | steps | nondeg_frac(mean) | nondeg 10-step-bin min/max | steps w/ zero-std group | grad_norm mean nondeg / deg | reward_std min/mean/max | clipped_ratio mean | adv-clamp engaged rate (run-level) | kl-clamp engaged tokens frac'
print(hdr)
for key in RUNS:
    ev = load_run(key); t = [r for r in ev['result']['telemetry'] if 'frac_reward_zero_std' in r]
    assert [r['physical_step'] for r in t] == list(range(1, 151)), key
    fz = [r['frac_reward_zero_std'] for r in t]; assert set(fz) <= {0.0, 1.0}
    nd = [1 - x for x in fz]
    bins = [st.mean(nd[i:i+10]) for i in range(0, 150, 10)]
    g_nd = [r['grad_norm'] for r in t if r['frac_reward_zero_std'] == 0]; g_d = [r['grad_norm'] for r in t if r['frac_reward_zero_std'] == 1]
    rs = [r['reward_std'] for r in t]; cr = [r['completions/clipped_ratio'] for r in t]
    ac = ev['result']['advantage_clamp_summary']; kc = ev['result']['kl_clamp_summary']
    print(f"{key[0]}-{key[1]} | {len(t)} | {st.mean(nd):.3f} | {min(bins):.2f}/{max(bins):.2f} | {int(sum(fz))} | {st.mean(g_nd) if g_nd else float('nan'):.3f} / {st.mean(g_d) if g_d else float('nan'):.3f} | {min(rs):.2f}/{st.mean(rs):.2f}/{max(rs):.2f} | {st.mean(cr):.3f} | {ac['engagement_rate']:.3f} ({ac['engaged_count']}/{ac['total_count']}) | {kc['engaged_token_count']/kc['total_token_count']:.3f}")
    print('   nondeg per 10-step bin:', [round(b, 1) for b in bins])
print()
print('== Q7: logged KL at each milestone step (telemetry row whose physical_step == milestone step); min/mean/max over 30 milestones ==')
for key in RUNS:
    ev = load_run(key); tel = {r['physical_step']: r for r in ev['result']['telemetry'] if 'kl' in r}
    ks = [tel[m['step']]['kl'] for m in ev['result']['milestones']]
    print(f"{key[0]}-{key[1]} kl@ms min/mean/max = {min(ks):.2e}/{st.mean(ks):.2e}/{max(ks):.2e}; kl@step1={tel[1]['kl']:.2e} kl@step150={tel[150]['kl']:.2e}; max over all steps={max(r['kl'] for r in tel.values()):.2e}")
    for m, k in zip(ev['result']['milestones'], ks): rows_out.append(dict(cond=key[0], seed=key[1], step=m['step'], kl_logged=k, genuine_correct=m['genuine_correct_rate'], intermediate=m['intermediate_tracking_accuracy'], leakage=m['leakage_rate'], format_valid=m['format_valid_rate']))
import math
print('theoretical cap of logged per-token k3 KL when |log-ratio| is clamped to 0.04:', math.exp(0.04) - 0.04 - 1, '(exp(d)-d-1) ; other sign:', math.exp(-0.04) + 0.04 - 1)
with open('milestones_all_runs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
print()
print('== Q9: greedy held-out (n=21) genuine_correct_rate per milestone ==')
sft = load_sft(); print('SFT checkpoint (tier b, 21 held-out, greedy): final_answer_accuracy', sft['tier_b_heldout_same_pair']['summary']['final_answer_accuracy'], 'intermediate', sft['tier_b_heldout_same_pair']['summary']['intermediate_tracking_accuracy'])
for key in RUNS:
    ev = load_run(key); c = ev['config']
    g = [m['genuine_correct_rate'] for m in ev['result']['milestones']]
    print(f"{key[0]}-{key[1]} step0(sanity)={c['sanity_final_answer_accuracy']} | " + ' '.join(f'{x*21:.0f}' for x in g) + f"   (counts of 21 at steps 5..150) min={min(g):.3f}@{[m['step'] for m in ev['result']['milestones']][g.index(min(g))]} final={g[-1]:.3f}")
print('lr rows (steps 1..8, 75, 149, 150) MAIN-43:', [(r['physical_step'], r['learning_rate']) for r in load_run(('MAIN',43))['result']['telemetry'] if r['physical_step'] in (1,2,3,4,5,6,7,8,75,149,150) and 'learning_rate' in r])
