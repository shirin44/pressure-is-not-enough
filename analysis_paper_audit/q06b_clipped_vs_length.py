"""Check what completions/clipped_ratio means in these runs: compare with max_length (256 = actually hit the cap) per step."""
import statistics as st
from common import *
for k in RUNS:
    t = [r for r in load_run(k)['result']['telemetry'] if 'completions/max_length' in r]
    n = len(t); th = n // 3
    def seg(rows):
        cap = st.mean(1.0 if r['completions/max_length'] >= 256 else 0.0 for r in rows)
        return f"clipped={st.mean(r['completions/clipped_ratio'] for r in rows):.2f} mean_len={st.mean(r['completions/mean_length'] for r in rows):6.1f} steps_with_any_256={cap:.2f} mean_term_len={st.mean(r['completions/mean_terminated_length'] for r in rows):5.1f}"
    print(k, '| first third:', seg(t[:th]), '| last third:', seg(t[-th:]))
