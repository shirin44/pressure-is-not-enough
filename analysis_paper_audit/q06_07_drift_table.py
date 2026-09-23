import json
from common import AWS
ev = json.loads((AWS / 'stage9e-raw-vs-chat-drift-v1' / 'raw_vs_chat_drift.json').read_text())
print('adapter sha256:', {k: v[:8] for k, v in ev['C_drift']['adapter_sha256'].items()})
print('%-32s %-9s %12s %12s %10s %10s %10s %14s %14s' % ('adapter', 'format', 'mean logratio', 'mean|.|', 'max|.|', 'frac>0.04', 'n_tok', 'meanLP sft/tok', 'meanLP adp/tok'))
for name, d in ev['C_drift']['stats'].items():
    for fmt, s in d.items():
        print('%-32s %-9s %12.6f %12.6f %10.4f %10.4f %10d %14.4f %14.4f' % (name, fmt, s['mean_per_token_logratio(adapter-SFT)'], s['mean_abs_logratio'], s['max_abs_logratio'], s['frac_tokens_abs_gt_0.04'], s['n_tokens'], s['mean_per_token_logprob_sft'], s['mean_per_token_logprob_adapter']))
print()
for k, v in ev['B_greedy_heldout21_sft_adapter'].items():
    print(k, v['summary'])
print('\nraw greedy example (first 300 chars):', repr(ev['B_greedy_heldout21_sft_adapter']['raw']['rows'][0]['completion'][:300]))
print('raw greedy: completions containing a complete <answer>..</answer>:', sum('</answer>' in r['completion'] for r in ev['B_greedy_heldout21_sft_adapter']['raw']['rows']), '/21; starting with bare code word:', sum(r['completion'].lstrip().startswith(('Bek', 'Ner')) for r in ev['B_greedy_heldout21_sft_adapter']['raw']['rows']), '/21')
print('A:', {k: v for k, v in ev['A_stop_criterion_tokenisation'].items() if not k.startswith('sft_tierb_last6')})
