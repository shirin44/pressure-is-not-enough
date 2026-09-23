"""Step 1: re-confirm classifier reproduction on the known 100-row baseline (per-field, not just aggregate).
Step 2: run classify_and_filter on the full 8,160-row bootstrap batch, report rate/domain breakdown/target check."""
import json, math, sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
sys.path.insert(0, str(ROOT))
from src.data.undeclared_generalization import UndeclaredExample, score_undeclared_completion

HOME = Path.home()

print("===== STEP 1: CLASSIFIER REPRODUCTION GATE (100 known rows, per-field) =====")
original_rollouts = HOME / "aisi_checkpoints" / "stage35-undeclared-lamp-checkpoint130-v2" / "rollouts.json"
original = json.loads(original_rollouts.read_text())
original_rows = original["rows"] if isinstance(original, dict) and "rows" in original else original

mismatches = []
reproduced_verified = []
for row in original_rows:
    example = UndeclaredExample(example_id=row["example_id"], domain="lamp",
        initial_state=row["initial_state"], operations=tuple(row["operations"]),
        expected_states=tuple(row["expected_states"]), final_answer=row["final_answer"], prompt="")
    fresh = score_undeclared_completion(example, row["text"])
    for key in ("both_states_observed", "nonliteral_consistent", "decode_back_self_consistent", "answer_correct"):
        if key in row and fresh[key] != row[key]:
            mismatches.append({"example_id": row["example_id"], "field": key, "stored": row[key], "fresh": fresh[key]})
    reproduced_verified.append(fresh["both_states_observed"] and fresh["nonliteral_consistent"] and fresh["decode_back_self_consistent"])

reproduced_rate = sum(reproduced_verified) / len(reproduced_verified) if reproduced_verified else 0.0
KNOWN_RATE = 0.10
print(json.dumps({"n": len(reproduced_verified), "reproduced_verified_rate": reproduced_rate,
                   "known_rate": KNOWN_RATE, "per_row_mismatches": len(mismatches)}, indent=2))
if mismatches or not math.isclose(reproduced_rate, KNOWN_RATE, abs_tol=0.005):
    raise RuntimeError(f"CLASSIFIER DID NOT REPRODUCE THE KNOWN RESULT: {mismatches[:5]}")
print("PASSED: classifier reproduces every original row exactly. Trusted for the new 8,160-row batch.\n")

print("===== STEP 2: CLASSIFY + FILTER THE FULL 8,160-ROW BOOTSTRAP BATCH =====")
sys.path.insert(0, str(ROOT / "experiments" / "05_self_bootstrapping" / "data_generation"))
from filter_bootstrap_successes import classify_and_filter, build_sft_dataset, reverify_sft_dataset

raw = json.loads((HOME / "aisi_checkpoints" / "option1-bootstrap-generation-v1" / "raw_completions.json").read_text())
raw_rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
assert len(raw_rows) == 8160, f"expected 8160 rows, got {len(raw_rows)}"

result = classify_and_filter(raw_rows)
report = result["report"]
print(json.dumps(report, indent=2))

print("\n===== TARGET CHECK: aiming for ~800 genuine (sft_eligible) successes =====")
count = report["sft_eligible_count"]
print(json.dumps({
    "sft_eligible_count": count, "target": 800,
    "ratio_to_target": round(count / 800, 3) if count else 0.0,
    "in_reasonable_range_0.5x_to_2x": 400 <= count <= 1600,
}, indent=2))

# Save the scored results for the next step (building the SFT dataset), without building it yet.
out_dir = HOME / "aisi_checkpoints" / "bootstrap_filter_v1"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
(out_dir / "scored_rows.json").write_text(json.dumps(result["scored"], indent=2))
print(f"\nSaved classification report and scored rows to {out_dir} (SFT dataset not yet built).")
