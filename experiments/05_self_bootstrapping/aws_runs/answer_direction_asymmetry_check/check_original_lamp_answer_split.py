import json
from pathlib import Path
from collections import Counter

HOME = Path.home()
original = json.loads((HOME / "aisi_checkpoints" / "stage35-undeclared-lamp-checkpoint130-v2" / "rollouts.json").read_text())
rows = original["rows"] if isinstance(original, dict) and "rows" in original else original
print(f"total original lamp eval rows: {len(rows)}")

truth_counts = Counter(r["final_answer"] for r in rows)
print(f"ground-truth final_answer distribution across all 100: {dict(truth_counts)}")

verified_rows = [r for r in rows if r.get("both_states_observed") and r.get("nonliteral_consistent") and r.get("decode_back_self_consistent")]
verified_truth_counts = Counter(r["final_answer"] for r in verified_rows)
print(f"ground-truth distribution among the 10 VERIFIED (the source of the 10% figure): {dict(verified_truth_counts)}")

by_truth = {}
for truth in truth_counts:
    subset = [r for r in rows if r["final_answer"] == truth]
    v = sum(1 for r in subset if r.get("both_states_observed") and r.get("nonliteral_consistent") and r.get("decode_back_self_consistent"))
    by_truth[truth] = {"n": len(subset), "verified": v, "rate": v/len(subset) if subset else 0}
print(f"verified rate split by ground truth: {json.dumps(by_truth, indent=2)}")
