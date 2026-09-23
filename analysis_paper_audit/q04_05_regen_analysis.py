"""Q4/Q5: analyse the regenerated step-0 rollouts (aws_runs/stage9e-step0-rollout-regen-v1/step0_rollout_regen.json)."""
import json, math, statistics as st
from common import AWS
ev = json.loads((AWS / 'stage9e-step0-rollout-regen-v1' / 'step0_rollout_regen.json').read_text())
print('config:', {k: ev['config'][k] for k in ('adapter_sha256', 'n_per_prompt', 'seed', 'temperature', 'top_p', 'top_k', 'eos_token_id')})
def cp_upper(k, n, alpha=0.05):   # exact Clopper-Pearson upper bound (two-sided 95%), for 0 events -> 1-(alpha/2)^(1/n)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid**i * (1-mid)**(n-i) for i in range(0, k+1))
        if cdf > alpha/2: lo = mid
        else: hi = mid
    return lo
for arm, d in ev['arms'].items():
    rows = d['rows']; s = d['summary']; n = len(rows)
    print(f'\n===== {arm} (n={n}) =====')
    for k in ('format_valid', 'correct', 'no_delimiter', 'ended_eos', 'ended_answer_stop', 'hit_cap_256', 'mean_tokens', 'nonzero_p_cot (banned pattern in TRAIN span)', 'banned_pattern_anywhere_whole_completion', 'eval_scanner_flag', 'case_insensitive_word_anywhere', 'n_groups', 'groups_zero_reward_variance_step1', 'groups_with_any_nonzero_p_cot', 'mean_reward_main_step1'):
        print(f'  {k}: {s[k]}')
    nz = sum(r['banned_count_train_span'] > 0 for r in rows)
    print(f'  nonzero pCoT: {nz}/{n}; 95% CP upper bound on rate = {cp_upper(nz, n):.4f}')
    rew = [r['reward_main_step1'] for r in rows]
    print(f'  reward_main_step1 min/mean/max = {min(rew):.2f}/{st.mean(rew):.2f}/{max(rew):.2f}; p_cot max = {max(r["p_cot_step150"] for r in rows)}')
    # wrong-answer breakdown
    valid_wrong = sum(r['format_valid'] and not r['correct'] for r in rows); print(f'  valid-format but wrong answer: {valid_wrong}/{n}; invalid format: {sum(not r["format_valid"] for r in rows)}/{n}')
    print(f'  literal state words (heads/tails/etc.) appearing anywhere, case-insensitive: {sum(r["ci_word_anywhere"] for r in rows)}/{n}')
    if arm == 'train_format':
        bad = [r for r in rows if not r['format_valid']]
        print('  example malformed rollout (first 320 chars):', repr(bad[0]['completion'][:320]))
        print('  example malformed rollout #2 (first 320 chars):', repr(bad[5]['completion'][:320]))
        ok = [r for r in rows if r['format_valid']]
        print('  example valid rollout:', repr(ok[0]['completion'][:260]) if ok else None)
        print('  tokens of ended_eos rollouts mean:', st.mean(r['n_tokens'] for r in rows if r['ended_eos']) if any(r['ended_eos'] for r in rows) else None)
