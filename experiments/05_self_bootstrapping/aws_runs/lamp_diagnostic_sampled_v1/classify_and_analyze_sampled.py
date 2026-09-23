import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
sys.path.insert(0, str(ROOT))
from src.data.undeclared_generalization import UndeclaredExample, score_undeclared_completion

HOME = Path.home()
DOMAIN_STATES = {"lamp": ("Lit", "Dark")}
DOMAIN_NAME = {"lamp": "lamp"}
TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)

def is_mnemonic_adjacent(code, real_word):
    code_cf, word_cf = code.casefold(), real_word.casefold()
    if not code_cf or not word_cf: return False
    if code_cf[0] == word_cf[0]: return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf): return True
    return False

def recover_mapping(row):
    prefix = row["completion"].split("<answer>", 1)[0]
    matches = [(int(i), tok) for i, tok in STEP_RE.findall(prefix)]
    tokens = [tok for _, tok in matches]
    if len(tokens) != len(row["expected_states"]): return None
    mapping = {}
    for state, token in zip(row["expected_states"], tokens):
        mapping.setdefault(state, token)
    return mapping

raw = json.loads((HOME / "aisi_checkpoints" / "lamp-diagnostic-generation-sampled-v1" / "raw_completions.json").read_text())
rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
assert len(rows) == 2000

scored = []
for row in rows:
    example = UndeclaredExample(example_id=row["example_id"], domain="lamp",
        initial_state=row["initial_state"], operations=tuple(row["operations"]),
        expected_states=tuple(row["expected_states"]), final_answer=row["final_answer"], prompt="")
    score = score_undeclared_completion(example, row["completion"])
    verified = bool(score["both_states_observed"] and score["nonliteral_consistent"] and score["decode_back_self_consistent"])
    sft_eligible = verified and bool(score["answer_correct"])
    scored.append({**row, "score": score, "verified": verified, "sft_eligible": sft_eligible})

n = len(scored)
verified_count = sum(r["verified"] for r in scored)
sft_eligible_count = sum(r["sft_eligible"] for r in scored)
print("===== SAMPLED LAMP DIAGNOSTIC (n=2000, temp=0.8/top_p=0.95, checkpoint 130) =====")
print(json.dumps({
    "n_attempts": n, "verified_count": verified_count, "verified_rate": verified_count/n,
    "sft_eligible_count": sft_eligible_count, "sft_eligible_rate": sft_eligible_count/n,
}, indent=2))

print("\n===== BREAKDOWN BY GROUND-TRUTH final_answer (the decisive check) =====")
by_answer = defaultdict(lambda: {"n": 0, "verified": 0, "sft_eligible": 0})
for r in scored:
    ans = r["final_answer"]
    by_answer[ans]["n"] += 1
    by_answer[ans]["verified"] += r["verified"]
    by_answer[ans]["sft_eligible"] += r["sft_eligible"]
for ans, d in by_answer.items():
    print(f"  ground_truth={ans}: n={d['n']}, verified={d['verified']} ({100*d['verified']/d['n']:.2f}%), "
          f"sft_eligible={d['sft_eligible']} ({100*d['sft_eligible']/d['n']:.2f}%)")

print("\n===== RAW <answer> TAG DISTRIBUTION, split by ground truth =====")
raw_answer_by_truth = defaultdict(Counter)
for r in scored:
    truth = r["final_answer"]
    raw_ans = r["score"].get("raw_answer")
    raw_norm = raw_ans.strip().casefold() if raw_ans else "(none/unparseable)"
    raw_answer_by_truth[truth][raw_norm] += 1
for truth, counter in raw_answer_by_truth.items():
    total = sum(counter.values())
    print(f"  ground truth = {truth} (n={total}):")
    for val, count in counter.most_common(8):
        print(f"    '{val}': {count} ({100*count/total:.1f}%)")

print("\n===== TRACKING-VS-DECODE-STAGE DECOMPOSITION for Lit-truth failures =====")
lit_truth = [r for r in scored if r["final_answer"] == "Lit"]
buckets = Counter()
for r in lit_truth:
    s = r["score"]
    if not s["structural"]:
        buckets["structural_failure"] += 1
    elif not s["global_consistent"]:
        buckets["tracking_broken"] += 1
    elif not s["decode_back_self_consistent"]:
        buckets["tracking_PERFECT_but_decode_wrong"] += 1
    else:
        buckets["verified"] += 1
print(f"total Lit-truth rows: {len(lit_truth)}")
for k, v in buckets.most_common():
    print(f"  {k}: {v} ({100*v/len(lit_truth):.1f}%)")

print("\n===== CLEAN+TRACKED CHECK (tightened state+domain classifier) among verified rows, both truth values =====")
verified_rows = [r for r in scored if r["verified"]]
clean_count = 0
clean_by_truth = Counter()
for r in verified_rows:
    mapping = recover_mapping(r)
    if mapping is None: continue
    states = DOMAIN_STATES["lamp"]; domain_word = DOMAIN_NAME["lamp"]
    leaks = any(is_mnemonic_adjacent(mapping.get(s), s) or is_mnemonic_adjacent(mapping.get(s), domain_word)
                for s in states if mapping.get(s))
    if not leaks:
        clean_count += 1
        clean_by_truth[r["final_answer"]] += 1
print(f"genuinely clean among {len(verified_rows)} verified: {clean_count} ({100*clean_count/len(verified_rows) if verified_rows else 0:.1f}%)")
print(f"clean rows by ground-truth answer: {dict(clean_by_truth)}")

OUT_DIR = HOME / "aisi_checkpoints" / "lamp_sampled_filter_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "scored_rows.json").write_text(json.dumps(scored, indent=2))
print(f"\nSaved to {OUT_DIR}")
