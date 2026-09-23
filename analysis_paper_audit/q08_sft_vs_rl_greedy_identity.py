"""Q8: on the 21 held-out scenarios (greedy), fraction of completions string-identical between SFT (tier b samples in the SFT evidence) and each run's persisted step-150 milestone samples.
Token ids are not persisted for the RL milestones, only decoded text (skip_special_tokens=True), so 'identical' here = identical decoded string."""
import difflib
from common import *
sft = load_sft()['tier_b_heldout_same_pair']['samples']
key = lambda s: (s['starting_state'], tuple(s['operations']))
sft_by = {key(s): s['completion'] for s in sft}
assert len(sft_by) == 21
print('run | identical@150 /21 | identical at every milestone (mean over 30) | first diff example (sft vs rl)')
for k in RUNS:
    ms = load_run(k)['result']['milestones']
    per = []
    for m in ms:
        assert set(key(s) for s in m['samples']) == set(sft_by), 'scenario set mismatch'
        per.append(sum(sft_by[key(s)] == s['completion'] for s in m['samples']))
    fin = ms[-1]; diffs = [(sft_by[key(s)], s['completion']) for s in fin['samples'] if sft_by[key(s)] != s['completion']]
    print(f"{k[0]}-{k[1]} | {per[-1]}/21 | steps5..150 counts: {per}")
    if diffs:
        a, b = diffs[0]; print('    e.g. SFT:', repr(a[:170])); print('         RL :', repr(b[:170]))
