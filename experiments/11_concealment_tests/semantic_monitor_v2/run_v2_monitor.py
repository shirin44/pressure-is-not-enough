"""Stage 11, semantic_monitor_v2 orchestration: runs the real-world-label monitor over
all 63 traces from Part A (load_traces.load_all(), unmodified, no regeneration) and
persists every call's full raw input/output incrementally."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_traces import load_all  # noqa: E402
from v2_monitor import run_v2_monitor_on_trace  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / 'v2_monitor_results.json'
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
            result = run_v2_monitor_on_trace(record)
            results.append(result)
            OUT_PATH.write_text(json.dumps(results, indent=2))
            print(f'[{call_i}/{total}] {group_name} acc={result["trace_accuracy"]} '
                  f'first_step={result["first_step_correct"]} consistent={result["internally_consistent"]} '
                  f'mapping={result["inferred_mapping"]} parse_error={result["parse_error"]}')
    print(f'\nDONE. {len(results)} total records written to {OUT_PATH}')


if __name__ == '__main__':
    main()
