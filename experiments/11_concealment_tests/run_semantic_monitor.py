"""Stage 11, Part B orchestration: runs the semantic monitor over every trace/variant
combination and persists every call's full raw input/output incrementally (written
after each call, not only at the end -- survives an interruption mid-batch).

Coverage (task step 6, "run this on all prepared traces"):
  - pre_rl_trained_pair  (n=21) x {full, state_only}  = 42 calls
  - post_rl_trained_pair (n=21) x {full, state_only}  = 42 calls
  - pre_rl_heldout_pair  (n=21) x {full}              = 21 calls  (state_only variant
    intentionally not run for this group -- see design.md's cost/coverage tradeoff note;
    the 'full' variant is the realistic, deployed-behavior condition and is run for
    every group without exception)
Total: 105 judge calls.
"""
from __future__ import annotations

import json
from pathlib import Path

from load_traces import load_all
from semantic_monitor import run_monitor_on_trace

OUT_PATH = Path(__file__).resolve().parent / 'semantic_monitor_results.json'

PLAN = [
    ('pre_rl_trained_pair', ['full', 'state_only']),
    ('post_rl_trained_pair', ['full', 'state_only']),
    ('pre_rl_heldout_pair', ['full']),
]


def main():
    groups = load_all()
    results = []
    if OUT_PATH.is_file():
        results = json.loads(OUT_PATH.read_text())
        done = {(r['group'], r['variant'], tuple(r['operations']), r['starting_state']) for r in results}
    else:
        done = set()

    total_calls = sum(len(groups[g]) for g, variants in PLAN for _ in variants)
    call_i = 0
    for group_name, variants in PLAN:
        for variant in variants:
            for record in groups[group_name]:
                call_i += 1
                key = (group_name, variant, tuple(record['operations']), record['starting_state'])
                if key in done:
                    continue
                result = run_monitor_on_trace(record, variant)
                results.append(result)
                OUT_PATH.write_text(json.dumps(results, indent=2))
                print(f'[{call_i}/{total_calls}] {group_name}/{variant} '
                      f'ari={result["ari"]} can_determine={result["judge_said_can_determine_pattern"]} '
                      f'parse_error={result["parse_error"]}')
    print(f'\nDONE. {len(results)} total records written to {OUT_PATH}')


if __name__ == '__main__':
    main()
