import json, re
from pathlib import Path
from collections import Counter

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "lamp_diagnostic_filter_v1" / "scored_rows.json").read_text())
clean_ids = {r["example_id"] for r in json.loads((HOME / "aisi_checkpoints" / "lamp_diagnostic_filter_v1" / "clean_candidates.json").read_text())}

rows = [r for r in scored if r["example_id"] in clean_ids]
assert len(rows) == 120, f"expected 120, got {len(rows)}"

TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)

def recover_mapping(row):
    prefix = row["completion"].split("<answer>", 1)[0]
    matches = [(int(i), tok) for i, tok in STEP_RE.findall(prefix)]
    tokens = [tok for _, tok in matches]
    mapping = {}
    for state, token in zip(row["expected_states"], tokens):
        mapping.setdefault(state, token)
    return mapping

print("===== DIVERSITY: distinct code pairs =====")
pairs = []
for r in rows:
    mapping = recover_mapping(r)
    pairs.append(tuple(sorted(mapping.values())))
pair_counts = Counter(pairs)
print(f"distinct code pairs: {len(pair_counts)} out of {len(rows)} rows")
for pair, count in pair_counts.most_common():
    if count > 1:
        print(f"  repeated {count}x: {pair}")

all_tokens = [t for r in rows for t in recover_mapping(r).values()]
token_counts = Counter(all_tokens)
print(f"\ndistinct individual tokens: {len(token_counts)} out of {len(all_tokens)} total token uses")
for tok, count in token_counts.most_common(10):
    if count > 1:
        print(f"  token '{tok}' reused {count}x")

print("\n===== LENGTH / INITIAL-STATE / FINAL-ANSWER SPREAD =====")
lengths = Counter(len(r["operations"]) for r in rows)
initials = Counter(r["initial_state"] for r in rows)
finals = Counter(r["final_answer"] for r in rows)
print(f"length distribution: {dict(sorted(lengths.items()))}")
print(f"initial_state distribution: {dict(initials)}")
print(f"final_answer distribution: {dict(finals)}")
