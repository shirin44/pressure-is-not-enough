"""Q2-Q4: per-arm detail from aws_runs/stage9e-step0-rollout-regen-v1/step0_rollout_regen.json (persisted; no recomputation of generation)."""
import json, re, statistics as st, collections
from common import AWS
ev = json.loads((AWS / 'stage9e-step0-rollout-regen-v1' / 'step0_rollout_regen.json').read_text())
print('config:', ev['config'])
for arm, d in ev['arms'].items():
    rows = d['rows']; n = len(rows)
    print(f'\n===== {arm}: n={n}, scenarios={len(set(r["scenario"] for r in rows))}, samples/scenario={collections.Counter(collections.Counter(r["scenario"] for r in rows).values())}')
    print('  starting states:', dict(collections.Counter(r['starting_state'] for r in rows)))
    # Q3: both scanners
    train_nz = sum(r['banned_count_train_span'] > 0 for r in rows); whole_nz = sum(r['banned_count_whole_completion'] > 0 for r in rows)
    ev_flag = sum(r['eval_scanner_flag'] for r in rows); ci = sum(r['ci_word_anywhere'] for r in rows)
    print(f'  Q3 training-penalty scanner (case-sens, pre-<answer>): nonzero {train_nz}/{n} | same regexes on WHOLE completion: {whole_nz}/{n} | eval scanner (LA): {ev_flag}/{n} | case-insens word anywhere: {ci}/{n}')
    # Q4
    rt = collections.Counter(r['r_task'] for r in rows)
    toks = [r['n_tokens'] for r in rows]
    wf_answer = sum(r['format_valid'] for r in rows)
    print(f'  Q4 well-formed answer (parseable <answer>..</answer> at end): {wf_answer}/{n} = {wf_answer/n:.3f}')
    print(f'     mean tokens {st.mean(toks):.1f} (median {st.median(toks):.0f}); hit_cap_256 {sum(r["hit_cap"] for r in rows)}/{n}; ended_eos {sum(r["ended_eos"] for r in rows)}/{n}; ended_answer_stop {sum(r["ended_answer_stop"] for r in rows)}/{n}')
    print(f'     r_task distribution: +4.0: {rt.get(4.0,0)}  -0.5: {rt.get(-0.5,0)}  -5.0: {rt.get(-5.0,0)}  ({n} total)')
