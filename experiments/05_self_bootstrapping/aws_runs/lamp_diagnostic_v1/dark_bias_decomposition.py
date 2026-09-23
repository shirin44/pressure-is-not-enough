import json
from pathlib import Path
from collections import Counter

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "lamp_diagnostic_filter_v1" / "scored_rows.json").read_text())

lit_truth = [r for r in scored if r["final_answer"] == "Lit"]
print(f"total Lit-truth rows: {len(lit_truth)}")

buckets = Counter()
for r in lit_truth:
    s = r["score"]
    if not s["structural"]:
        buckets["structural_failure (malformed trace)"] += 1
    elif not s["global_consistent"]:
        buckets["tracking_broken (global_consistent=False -- genuine tracking error)"] += 1
    elif not s["decode_back_self_consistent"]:
        buckets["tracking_PERFECT_but_decode_stage_wrong (decode-bias pattern)"] += 1
    else:
        buckets["other (verified but somehow not counted -- should be 0)"] += 1

for k, v in buckets.most_common():
    print(f"  {k}: {v} ({100*v/len(lit_truth):.1f}%)")

# Cross-check: of the "tracking perfect but decode-stage wrong" bucket, what did the decode-back line say?
print("\n--- of the tracking-perfect-but-decode-wrong bucket, what physical state did the decode-back line claim? ---")
decode_claims = Counter()
for r in lit_truth:
    s = r["score"]
    if s["structural"] and s["global_consistent"] and not s["decode_back_self_consistent"]:
        import re
        m = re.search(r"Final coded state:\s*([A-Za-z]+)\.\s*\1 represents ([A-Za-z]+)\.", r["completion"], re.IGNORECASE)
        claimed = m.group(2) if m else "(no decode-back line found)"
        decode_claims[claimed] += 1
for k, v in decode_claims.most_common():
    print(f"  decode-back line claimed '{k}': {v}")
