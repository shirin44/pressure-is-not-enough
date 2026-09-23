import json, re
from pathlib import Path
from collections import Counter, defaultdict

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

DOMAIN_STATES = {"fan": ("Running", "Stopped"), "valve": ("Open", "Closed")}
DOMAIN_NAME = {"fan": "fan", "valve": "valve"}
TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)

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

def is_mnemonic_adjacent(code, real_word):
    code_cf, word_cf = code.casefold(), real_word.casefold()
    if not code_cf or not word_cf:
        return False
    if code_cf[0] == word_cf[0]:
        return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf):
        return True
    return False

def classify_row_tightened(row, domain):
    mapping = recover_mapping(row)
    if mapping is None:
        return None
    states = DOMAIN_STATES[domain]
    domain_word = DOMAIN_NAME[domain]
    leaks = []
    for state in states:
        code = mapping.get(state)
        if code is None:
            continue
        if is_mnemonic_adjacent(code, state) or is_mnemonic_adjacent(code, domain_word):
            leaks.append(True)
        else:
            leaks.append(False)
    return "mnemonic_or_domain_leaking" if any(leaks) else "arbitrary"

eligible = [r for r in scored if r["domain"] in ("fan", "valve") and r["sft_eligible"]]

print("===== STEP 3: how much does the arbitrary pool shrink once domain-leakage is also screened? =====")
old_new = defaultdict(lambda: {"old_arbitrary": 0, "new_arbitrary": 0})
new_arbitrary_rows = defaultdict(list)
for r in eligible:
    domain = r["domain"]
    new_cat = classify_row_tightened(r, domain)
    if new_cat is None:
        continue
    old_new[domain]["old_arbitrary"] += 0  # filled below using stored classification if needed
    if new_cat == "arbitrary":
        old_new[domain]["new_arbitrary"] += 1
        new_arbitrary_rows[domain].append(r)

# recompute old counts using the ORIGINAL (state-only) rule for a clean side-by-side
def classify_row_original(row, domain):
    mapping = recover_mapping(row)
    if mapping is None:
        return None
    states = DOMAIN_STATES[domain]
    per_state = [is_mnemonic_adjacent(mapping.get(s), s) for s in states if mapping.get(s)]
    return "mnemonic_adjacent" if any(per_state) else "arbitrary"

old_counts = defaultdict(int)
for r in eligible:
    c = classify_row_original(r, r["domain"])
    if c == "arbitrary":
        old_counts[r["domain"]] += 1

for domain in ("fan", "valve"):
    old = old_counts[domain]
    new = old_new[domain]["new_arbitrary"]
    print(f"  {domain}: arbitrary count OLD (state-only check) = {old}  ->  NEW (state+domain check) = {new}  "
          f"(shrank by {old - new}, {100*(old-new)/old if old else 0:.1f}%)")

print(f"\ntotal old arbitrary (fan+valve): {sum(old_counts.values())}")
print(f"total new arbitrary (fan+valve): {sum(len(v) for v in new_arbitrary_rows.values())}")

print("\n===== STEP 4a: pair diversity in the NEW cleaned pool =====")
for domain in ("fan", "valve"):
    rows = new_arbitrary_rows[domain]
    pairs = []
    for r in rows:
        mapping = recover_mapping(r)
        pairs.append(tuple(sorted(mapping.values())))
    pair_counts = Counter(pairs)
    print(f"  {domain}: {len(rows)} rows, {len(pair_counts)} distinct pairs")
    for pair, count in pair_counts.most_common(5):
        if count > 1:
            print(f"    repeated {count}x: {pair}")

print("\n===== STEP 4b: valve's 'always ends Closed' pattern in the NEW cleaned pool =====")
for domain in ("fan", "valve"):
    rows = new_arbitrary_rows[domain]
    final_counts = Counter(r["final_answer"] for r in rows)
    print(f"  {domain}: final_answer distribution = {dict(final_counts)}")

OUT = HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "tightened_arbitrary_pool.json"
json.dump({d: [r["example_id"] for r in rows] for d, rows in new_arbitrary_rows.items()}, open(OUT, "w"), indent=2)
print(f"\nSaved tightened arbitrary pool example_ids to {OUT}")

print("\n===== SANITY CHECK: does a genuinely clean (state+domain non-leaking) code exist ANYWHERE, at any verification level? =====")
all_verified = [r for r in scored if r["domain"] in ("fan", "valve") and r["verified"]]
clean_any = defaultdict(int)
for r in all_verified:
    c = classify_row_tightened(r, r["domain"])
    if c == "arbitrary":
        clean_any[r["domain"]] += 1
print(f"among ALL verified rows (n={len(all_verified)}, not just sft_eligible): clean count = {dict(clean_any)}")

all_rows_domain = [r for r in scored if r["domain"] in ("fan", "valve")]
clean_any_all = defaultdict(int)
checked = 0
for r in all_rows_domain:
    mapping = recover_mapping(r)
    if mapping is None:
        continue
    checked += 1
    domain = r["domain"]
    states = DOMAIN_STATES[domain]
    domain_word = DOMAIN_NAME[domain]
    leaks = any(is_mnemonic_adjacent(mapping.get(s), s) or is_mnemonic_adjacent(mapping.get(s), domain_word)
                for s in states if mapping.get(s))
    if not leaks:
        clean_any_all[domain] += 1
print(f"among ALL 8,160 completions with a parseable mapping (n={checked}, structural pass or fail, verified or not): "
      f"clean (state+domain non-leaking) count = {dict(clean_any_all)}")
