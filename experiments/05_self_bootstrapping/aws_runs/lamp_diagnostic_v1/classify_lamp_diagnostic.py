import json, re, sys
from pathlib import Path

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
    if not code_cf or not word_cf:
        return False
    if code_cf[0] == word_cf[0]:
        return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf):
        return True
    return False

def recover_mapping(row):
    prefix = row["completion"].split("<answer>", 1)[0]
    matches = [(int(i), tok) for i, tok in STEP_RE.findall(prefix)]
    tokens = [tok for _, tok in matches]
    if len(tokens) != len(row["expected_states"]):
        return None
    mapping = {}
    for state, token in zip(row["expected_states"], tokens):
        mapping.setdefault(state, token)
    return mapping

raw = json.loads((HOME / "aisi_checkpoints" / "lamp-diagnostic-generation-v1" / "raw_completions.json").read_text())
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
print("===== LAMP DIAGNOSTIC (n=2000, greedy, checkpoint 130) CLASSIFICATION =====")
print(json.dumps({
    "n_attempts": n, "verified_count": verified_count, "verified_rate": verified_count / n,
    "sft_eligible_count": sft_eligible_count, "sft_eligible_rate": sft_eligible_count / n,
    "known_baseline_rate_lamp_n100": 0.10, "ratio_to_baseline": (verified_count/n) / 0.10,
}, indent=2))

verified_rows = [r for r in scored if r["verified"]]
clean_count = 0
leaking_count = 0
clean_rows = []
for r in verified_rows:
    mapping = recover_mapping(r)
    if mapping is None:
        continue
    states = DOMAIN_STATES["lamp"]
    domain_word = DOMAIN_NAME["lamp"]
    leaks = any(is_mnemonic_adjacent(mapping.get(s), s) or is_mnemonic_adjacent(mapping.get(s), domain_word)
                for s in states if mapping.get(s))
    if leaks:
        leaking_count += 1
    else:
        clean_count += 1
        clean_rows.append({"example_id": r["example_id"], "mapping": mapping, "sft_eligible": r["sft_eligible"]})

print(f"\n===== CLEAN+TRACKED CHECK (tightened state+domain classifier) among {len(verified_rows)} verified rows =====")
print(json.dumps({
    "genuinely_clean_verified": clean_count, "mnemonic_or_domain_leaking_verified": leaking_count,
    "clean_rate_within_verified": clean_count / len(verified_rows) if verified_rows else None,
    "comparison_to_fanvalve": "fan/valve: 0/575 (0%) clean among verified. lamp original 100-eval: 10/10 (100%) clean.",
}, indent=2))

clean_and_sft_eligible = sum(1 for r in clean_rows if r["sft_eligible"])
print(f"\nof the {clean_count} genuinely clean verified rows, {clean_and_sft_eligible} are ALSO sft_eligible (correct final answer) -- these are the actual Dataset A' candidates.")
print("\nfull detail of clean rows:")
for r in clean_rows:
    print(r)

OUT_DIR = HOME / "aisi_checkpoints" / "lamp_diagnostic_filter_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "scored_rows.json").write_text(json.dumps(scored, indent=2))
(OUT_DIR / "clean_candidates.json").write_text(json.dumps(clean_rows, indent=2))
print(f"\nSaved to {OUT_DIR}")
