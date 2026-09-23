"""Cross-check: milestone lines in each run's stdout log vs the JSON evidence (independent persisted copies)."""
import re, glob
from common import *
for k,(d,f) in RUNS.items():
    logs = [p for p in glob.glob(str(AWS/d/'*.log'))]
    ev = load_run(k); ms = {m['step']: m['genuine_correct_rate'] for m in ev['result']['milestones']}
    txt = ''.join(open(p, errors='ignore').read() for p in logs)
    found = {int(s): float(v) for s, v in re.findall(r'MILESTONE STEP (\d+): genuine_correct_rate=([0-9.]+)', txt)}
    ok = all(abs(found.get(s, -1) - v) < 5e-4 for s, v in ms.items())
    flags = txt.count('FLAG: LEAKAGE')
    print(k, [p.split('/')[-1] for p in logs], 'milestones in log', len(found), 'match JSON:', ok, 'leak flags in log:', flags)
