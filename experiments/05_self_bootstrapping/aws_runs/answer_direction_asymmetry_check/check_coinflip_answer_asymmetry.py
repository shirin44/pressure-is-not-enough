import json
from pathlib import Path
from collections import defaultdict

HOME = Path.home()
d = json.load(open(HOME / "aisi_checkpoints" / "step-500" / "step50_to500_full_rollouts.json"))
groups = d["groups"]

by_truth = defaultdict(lambda: {"n": 0, "correct": 0})
raw_answer_by_truth = defaultdict(lambda: defaultdict(int))
for g in groups:
    for r in g.get("rollouts", []):
        truth = r.get("truth")
        if truth not in ("Heads", "Tails"):
            continue
        rb = r.get("reward_breakdown", {})
        r_task = rb.get("r_task")
        answer = rb.get("answer")
        by_truth[truth]["n"] += 1
        if r_task is not None and r_task > 0:
            by_truth[truth]["correct"] += 1
        raw_answer_by_truth[truth][answer if answer else "(none/unparseable)"] += 1

print(f"total groups: {len(groups)}")
print(f"total rollouts: {sum(d['n'] for d in by_truth.values())}")
print("\n===== Coin Flip task accuracy split by ground-truth answer (across the full step50->500 trajectory) =====")
for truth, dd in by_truth.items():
    print(f"  truth={truth}: n={dd['n']}, correct={dd['correct']} ({100*dd['correct']/dd['n']:.2f}%)")

print("\n===== raw model answer distribution, split by ground truth =====")
for truth, counter in raw_answer_by_truth.items():
    total = sum(counter.values())
    print(f"  truth={truth} (n={total}):")
    for val, count in sorted(counter.items(), key=lambda x: -x[1])[:6]:
        print(f"    '{val}': {count} ({100*count/total:.1f}%)")
