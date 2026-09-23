"""Q4/Q5: which completions are PERSISTED, and what do the training penalty / eval scanner say about them?
Persisted completions = greedy milestone-eval samples (21 held-out scenarios x 30 milestones x 8 runs) + SFT tiers a/b/c + the 12-step calibration run.
NOT persisted = any TRAINING rollout (sampled T=0.8, top_p=0.95). So this is NOT the training-rollout fraction asked for; it is the only re-scorable set."""
import json, re
from common import *
from reward_v3 import score_completion_v2, ALL_BANNED_PATTERNS, count_flips
from same_different_leakage_audit import check_completion_for_leakage
from code_word_answer_bank import build_code_word_prompt

def scan(c, step, total, prompt_text, gt):
    b = score_completion_v2(c, gt, max(step,1), total, prompt=prompt_text)   # MAIN defaults => real training penalty
    pre = c.split('<answer>', 1)[0]
    return dict(banned_count_train_span=b['banned_count'], p_cot_main=b['p_cot'],
                eval_leak=not check_completion_for_leakage(c)['clean'],
                ci_word_anywhere=bool(re.search(r'(?<![A-Za-z])(heads|tails|head|tail)(?![A-Za-z])', c, re.I)))
out = {}
tot = dict(n=0, train_nonzero=0, eval_leak=0, ci_any=0)
for key in RUNS:
    ev = load_run(key); ms = ev['result']['milestones']
    n = nz = el = ci = 0
    for m in ms:
        for s in m['samples']:
            p = build_code_word_prompt(s['starting_state'], s['operations'])
            r = scan(s['completion'], m['step'], ev['config']['full_steps'], p, 'Bek')
            n += 1; nz += r['banned_count_train_span'] > 0; el += r['eval_leak']; ci += r['ci_word_anywhere']
    out[f'{key[0]}-seed{key[1]}'] = dict(n_completions=n, nonzero_train_pcot=nz, eval_flagged=el, case_insens_word_anywhere=ci, milestones=len(ms))
    for k, v in (('n', n), ('train_nonzero', nz), ('eval_leak', el), ('ci_any', ci)): tot[k] += v
cal = json.loads((AWS/'stage9e-llama-rl-main-v1'/'stage9e_llama_rl_main.json').read_text())
n = sum(len(m['samples']) for m in cal['result']['milestones']); out['CALIBRATION-main-12step'] = dict(n_completions=n)
sft = load_sft()
for tier in ('tier_a_training_set_recall', 'tier_b_heldout_same_pair', 'tier_c_heldout_different_pair'):
    ss = sft[tier]['samples']; nz = 0; el = 0
    for s in ss:
        c = s.get('completion') or s.get('text') or ''
        b = score_completion_v2(c, 'Bek', 1, 150, num_flips=5)
        nz += b['banned_count'] > 0; el += not check_completion_for_leakage(c)['clean']
    out[f'SFT-{tier}'] = dict(n_completions=len(ss), nonzero_train_pcot=nz, eval_flagged=el, keys=list(ss[0].keys())[:8])
print(json.dumps(out, indent=1)); print('TOTAL over 8 RL runs:', tot)
