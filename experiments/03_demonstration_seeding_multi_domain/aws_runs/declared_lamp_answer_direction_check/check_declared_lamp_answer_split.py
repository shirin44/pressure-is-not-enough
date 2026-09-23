import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
sys.path.insert(0, str(ROOT))
from src.data.multidomain_seed import generate_multidomain_dataset, DEFAULT_SEED

assert DEFAULT_SEED == 20260812, f"expected seed 20260812, got {DEFAULT_SEED}"
_, evaluations = generate_multidomain_dataset(seed=DEFAULT_SEED)
lamp_eval = evaluations["lamp"]
assert len(lamp_eval) == 100
ground_truth_by_id = {row.example_id: row.final_answer for row in lamp_eval}
print(f"regenerated {len(lamp_eval)} deterministic lamp eval examples (seed={DEFAULT_SEED})")

HOME = Path.home()
progress = json.loads((HOME / "aisi_checkpoints" / "multidomain-stage3-v1-eval" / "step130_eval_progress.json").read_text())
lamp_rows = [r for r in progress["rows"] if r["domain"] == "lamp"]
assert len(lamp_rows) == 100

unmatched = [r["example_id"] for r in lamp_rows if r["example_id"] not in ground_truth_by_id]
print(f"unmatched example_ids: {len(unmatched)} {unmatched[:5]}")

by_truth = defaultdict(lambda: {"n": 0, "answer_correct": 0, "decode_back_correct": 0, "transition_tracking": 0, "global_consistent": 0})
for r in lamp_rows:
    truth = ground_truth_by_id.get(r["example_id"])
    if truth is None:
        continue
    by_truth[truth]["n"] += 1
    by_truth[truth]["answer_correct"] += bool(r["answer_correct"])
    by_truth[truth]["decode_back_correct"] += bool(r["decode_back_correct"])
    by_truth[truth]["transition_tracking"] += bool(r["transition_tracking"])
    by_truth[truth]["global_consistent"] += bool(r["global_consistent"])

print("\n===== DECLARED-mapping lamp control (checkpoint 130, step 130, n=100), split by ground-truth final_answer =====")
for truth, d in by_truth.items():
    print(f"  ground_truth={truth}: n={d['n']}, answer_correct={d['answer_correct']} ({100*d['answer_correct']/d['n']:.1f}%), "
          f"decode_back_correct={d['decode_back_correct']} ({100*d['decode_back_correct']/d['n']:.1f}%), "
          f"transition_tracking={d['transition_tracking']} ({100*d['transition_tracking']/d['n']:.1f}%), "
          f"global_consistent={d['global_consistent']} ({100*d['global_consistent']/d['n']:.1f}%)")

overall_correct = sum(r["answer_correct"] for r in lamp_rows)
print(f"\noverall answer_correct (sanity check vs known 86% figure): {overall_correct}/100 = {overall_correct}%")
