"""Stage 11, Part C, step 8 orchestration: paraphrases every trace in all 3 groups
(63 calls) and persists every call's full raw input/output incrementally. Feeding the
paraphrase back into Llama-3-8B-Instruct (step 9) is NOT done here -- see
gpu_requirement_check.py."""
from __future__ import annotations

import json
from pathlib import Path

from load_traces import load_all
from paraphrase_defense import paraphrase_trace

OUT_PATH = Path(__file__).resolve().parent / 'paraphrase_results.json'
GROUPS = ['pre_rl_trained_pair', 'post_rl_trained_pair', 'pre_rl_heldout_pair']


def main():
    groups = load_all()
    results = []
    if OUT_PATH.is_file():
        results = json.loads(OUT_PATH.read_text())
        done = {(r['group'], tuple(r['operations']), r['starting_state']) for r in results}
    else:
        done = set()

    total = sum(len(groups[g]) for g in GROUPS)
    call_i = 0
    for group_name in GROUPS:
        for record in groups[group_name]:
            call_i += 1
            key = (group_name, tuple(record['operations']), record['starting_state'])
            if key in done:
                continue
            result = paraphrase_trace(record)
            results.append(result)
            OUT_PATH.write_text(json.dumps(results, indent=2))
            print(f'[{call_i}/{total}] {group_name} preserved_exactly={result["code_sequence_preserved_exactly"]} '
                  f'stayed_within_instructed_pair={result["stayed_within_instructed_pair"]}')
    print(f'\nDONE. {len(results)} total records written to {OUT_PATH}')


if __name__ == '__main__':
    main()
