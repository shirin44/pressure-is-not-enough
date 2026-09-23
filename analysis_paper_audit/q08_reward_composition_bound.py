"""Q8: r_task per rollout is NOT persisted (only per-step group mean/std of the total reward). A rigorous LOWER bound on the number of r_task=-5 rollouts per step (group of 8)
follows from the mean: a rollout with r_task=-5 has total <= -5 + (consistency+signal bonus max); any other rollout has total <= 4 + bonus max. Bonuses: BASELINE 0 (both magnitudes 0), MAIN 0.15+0.15 (R3 defaults).
  8*mean <= k*U5 + (8-k)*Uo   =>   k >= (8*Uo - 8*mean) / (Uo - U5).
Reported per 10-step bin per run: mean total reward, lower-bound fraction of r_task=-5 rollouts, mean completion length, share of steps whose mean length >= 200 (raw-format degenerate signature)."""
import math, statistics as st
from common import *
def kmin(mean, cond):
    bonus = 0.0 if cond == 'BASELINE' else 0.3
    U5, Uo = -5.0 + bonus, 4.0 + bonus
    return max(0, min(8, math.ceil((8 * Uo - 8 * mean) / (Uo - U5) - 1e-9)))
print('per run: bins of 10 steps -> [mean reward | lower-bound frac r_task=-5 | mean len]')
tot = {}
for k in RUNS:
    t = [r for r in load_run(k)['result']['telemetry'] if 'reward' in r and 'completions/mean_length' in r][:150]
    assert len(t) == 150
    bins = []
    for i in range(0, 150, 10):
        b = t[i:i + 10]
        bins.append((st.mean(r['reward'] for r in b), st.mean(kmin(r['reward'], k[0]) / 8 for r in b), st.mean(r['completions/mean_length'] for r in b)))
    print(f'{k[0]}-{k[1]}:  ' + ' '.join(f'[{a:5.2f}|{f:.2f}|{l:3.0f}]' for a, f, l in bins))
    first = t[:10]; last = t[-50:]
    print(f'    steps1-10: mean reward {st.mean(r["reward"] for r in first):.2f}, LB frac(-5) {st.mean(kmin(r["reward"], k[0])/8 for r in first):.2f}, steps with mean_len>=200: {sum(r["completions/mean_length"]>=200 for r in first)}/10 | steps101-150: mean reward {st.mean(r["reward"] for r in last):.2f}, LB frac(-5) {st.mean(kmin(r["reward"], k[0])/8 for r in last):.2f}')
    n_first_pos = next((i + 1 for i, r in enumerate(t) if st.mean(x['reward'] for x in t[i:i + 5]) > 0), None)
    print(f'    first step whose next-5-step mean reward > 0: {n_first_pos}')
