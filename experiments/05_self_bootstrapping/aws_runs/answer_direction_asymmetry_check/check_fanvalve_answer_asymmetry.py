import json
from pathlib import Path
from collections import defaultdict

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

for domain in ("fan", "valve"):
    rows = [r for r in scored if r["domain"] == domain]
    by_answer = defaultdict(lambda: {"n": 0, "verified": 0, "sft_eligible": 0})
    for r in rows:
        ans = r["final_answer"]
        by_answer[ans]["n"] += 1
        by_answer[ans]["verified"] += r["verified"]
        by_answer[ans]["sft_eligible"] += r["sft_eligible"]
    print(f"===== {domain} (n={len(rows)}) verified/sft_eligible rate split by ground-truth final_answer =====")
    for ans, d in by_answer.items():
        print(f"  ground_truth={ans}: n={d['n']}, verified={d['verified']} ({100*d['verified']/d['n']:.2f}%), "
              f"sft_eligible={d['sft_eligible']} ({100*d['sft_eligible']/d['n']:.2f}%)")
    print()
