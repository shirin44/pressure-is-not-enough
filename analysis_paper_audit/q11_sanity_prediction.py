"""Predict whether llama_causal_flip_test.py's zero-step sanity gate (>=80% of first 5 eligible SFT tier-b completions reproduced greedily) would pass for each RL checkpoint,
using the persisted step-150 greedy samples as a proxy (max_new_tokens differs: 150 in the gate vs 256 in the milestone eval, so this is a prediction, not a measurement)."""
from common import *
sft = load_sft()['tier_b_heldout_same_pair']['samples']
elig = [r for r in sft if r['intermediate_tracking_correct'] and r['final_answer_correct']]
first5 = elig[:5]
key = lambda s: (s['starting_state'], tuple(s['operations']))
for k in RUNS:
    fin = {key(s): s['completion'] for s in load_run(k)['result']['milestones'][-1]['samples']}
    m = sum(fin[key(r)].strip() == r['completion'].strip() for r in first5)
    print(k, f'predicted sanity matches {m}/5 ->', 'PASS' if m >= 4 else 'ABORT (RuntimeError)')
