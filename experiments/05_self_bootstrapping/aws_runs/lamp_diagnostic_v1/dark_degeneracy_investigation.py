import json, re
from pathlib import Path
from collections import Counter, defaultdict

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "lamp_diagnostic_filter_v1" / "scored_rows.json").read_text())
assert len(scored) == 2000

print("===== ITEM 1: verification/eligibility rate split by GROUND-TRUTH final_answer, across ALL 2000 rows =====")
by_answer = defaultdict(lambda: {"n": 0, "verified": 0, "sft_eligible": 0})
for r in scored:
    ans = r["final_answer"]
    by_answer[ans]["n"] += 1
    by_answer[ans]["verified"] += r["verified"]
    by_answer[ans]["sft_eligible"] += r["sft_eligible"]

for ans, d in by_answer.items():
    print(f"  ground_truth_final_answer={ans}: n={d['n']}, verified={d['verified']} ({100*d['verified']/d['n']:.2f}%), "
          f"sft_eligible={d['sft_eligible']} ({100*d['sft_eligible']/d['n']:.2f}%)")

print("\n===== ITEM 2: raw <answer> tag prediction, regardless of correctness, split by ground truth =====")
raw_answer_by_truth = defaultdict(Counter)
for r in scored:
    truth = r["final_answer"]
    raw = r["score"].get("raw_answer")
    raw_norm = raw.strip().casefold() if raw else "(none/unparseable)"
    raw_answer_by_truth[truth][raw_norm] += 1

for truth, counter in raw_answer_by_truth.items():
    total = sum(counter.values())
    print(f"  when ground truth = {truth} (n={total}), model's raw <answer> tag distribution:")
    for val, count in counter.most_common(8):
        print(f"    '{val}': {count} ({100*count/total:.1f}%)")

print("\n===== ITEM 2b: token-length / tokenizer sanity check (does 'Dark' vs 'Lit' differ trivially) =====")
print(f"  'Dark' char length: {len('Dark')}, 'Lit' char length: {len('Lit')}")
print("  (full BPE tokenization check would need the actual tokenizer -- flagging char length only here;")
print("   the raw <answer> distribution above is the more direct signal for a decoding-bias-toward-Dark effect)")

print("\n===== ITEM 3: sample of 'Lit'-truth failures (verified=False or sft_eligible=False) =====")
lit_failures = [r for r in scored if r["final_answer"] == "Lit" and not r["sft_eligible"]]
print(f"total Lit-truth non-eligible rows: {len(lit_failures)} out of {sum(1 for r in scored if r['final_answer']=='Lit')} Lit-truth rows")
for r in lit_failures[:8]:
    s = r["score"]
    print("=" * 90)
    print(f"example_id={r['example_id']} initial_state={r['initial_state']} operations={r['operations']}")
    print(f"expected_states={r['expected_states']} ground_truth_final_answer={r['final_answer']}")
    print(f"structural={s['structural']} global_consistent={s['global_consistent']} "
          f"decode_back_self_consistent={s['decode_back_self_consistent']} answer_correct={s['answer_correct']} "
          f"raw_answer={s.get('raw_answer')!r}")
    print("--- completion ---")
    print(r["completion"])
    print()
