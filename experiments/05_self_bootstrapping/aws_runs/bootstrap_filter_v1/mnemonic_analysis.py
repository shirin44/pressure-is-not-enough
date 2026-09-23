import json, re
from pathlib import Path
from collections import defaultdict

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

DOMAIN_STATES = {"fan": ("Running", "Stopped"), "valve": ("Open", "Closed")}
TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)

def recover_mapping(row):
    """Recompute state->code mapping the same way the classifier does internally
    (zip expected_states against parsed step tokens), since token_pair alone is
    unordered and doesn't tell us which code belongs to which state."""
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

def classify_row(row, domain):
    mapping = recover_mapping(row)
    if mapping is None:
        return None
    states = DOMAIN_STATES[domain]
    per_state_mnemonic = {}
    for state in states:
        code = mapping.get(state)
        per_state_mnemonic[state] = is_mnemonic_adjacent(code, state) if code else None
    pair_mnemonic = any(v for v in per_state_mnemonic.values() if v is not None)
    final_state = row["final_answer"]
    final_code_mnemonic = per_state_mnemonic.get(final_state)
    return {"pair_mnemonic": pair_mnemonic, "final_code_mnemonic": final_code_mnemonic, "mapping": mapping}

def bug_hit(row):
    s = row["score"]
    m = re.search(r"Final coded state:\s*([A-Za-z]+)\.\s*\1 represents ([A-Za-z]+)\.", row["completion"], re.IGNORECASE)
    if not m:
        return False
    decoded_code, decoded_state = m.group(1), m.group(2)
    raw_answer = s.get("raw_answer") or ""
    return raw_answer.casefold() == decoded_code.casefold() and decoded_state.casefold() != raw_answer.casefold()

print("===== STEP 1+2: bug rate WITHIN mnemonic vs arbitrary code categories, pooled across domains =====")
failures = [r for r in scored if r["domain"] in ("fan", "valve") and r["verified"] and not r["sft_eligible"]]
print(f"total verified-but-failed (fan+valve pooled): {len(failures)}")

pooled = defaultdict(lambda: {"n": 0, "bug": 0})
by_domain_category = defaultdict(lambda: {"n": 0, "bug": 0})
unclassifiable = 0
for r in failures:
    c = classify_row(r, r["domain"])
    if c is None:
        unclassifiable += 1
        continue
    cat = "mnemonic_adjacent" if c["pair_mnemonic"] else "arbitrary"
    hit = bug_hit(r)
    pooled[cat]["n"] += 1
    pooled[cat]["bug"] += hit
    by_domain_category[(r["domain"], cat)]["n"] += 1
    by_domain_category[(r["domain"], cat)]["bug"] += hit

print(f"unclassifiable (parsing mismatch): {unclassifiable}")
for cat in ("mnemonic_adjacent", "arbitrary"):
    d = pooled[cat]
    rate = d["bug"] / d["n"] if d["n"] else 0.0
    print(f"  [pooled, failures-only set] {cat}: n={d['n']}, bug_rate_within_failures={rate:.3f} (note: ALL rows in this set already failed by definition -- see step 2b for the real denominator)")

print()
print("===== STEP 2b: the REAL bug rate -- denominator must be ALL verified rows in each category, not just the already-failed ones =====")
all_verified = [r for r in scored if r["domain"] in ("fan", "valve") and r["verified"]]
pooled2 = defaultdict(lambda: {"n": 0, "bug": 0})
by_domain_category2 = defaultdict(lambda: {"n": 0, "bug": 0})
unclassifiable2 = 0
for r in all_verified:
    c = classify_row(r, r["domain"])
    if c is None:
        unclassifiable2 += 1
        continue
    cat = "mnemonic_adjacent" if c["pair_mnemonic"] else "arbitrary"
    hit = not r["sft_eligible"]  # among ALL verified rows, "hit" = failed to become sft_eligible
    pooled2[cat]["n"] += 1
    pooled2[cat]["bug"] += hit
    by_domain_category2[(r["domain"], cat)]["n"] += 1
    by_domain_category2[(r["domain"], cat)]["bug"] += hit

print(f"total verified rows (fan+valve): {len(all_verified)}, unclassifiable: {unclassifiable2}")
print("\npooled across domains (this is the key number -- is mnemonic-ness the real driver?):")
for cat in ("mnemonic_adjacent", "arbitrary"):
    d = pooled2[cat]
    rate = d["bug"] / d["n"] if d["n"] else 0.0
    print(f"  {cat}: n={d['n']}, verified-but-failed-rate={rate:.3f} ({d['bug']}/{d['n']})")

print("\nbroken out by domain AND category (does mnemonic-ness explain the domain gap, or is domain still doing independent work?):")
for domain in ("fan", "valve"):
    for cat in ("mnemonic_adjacent", "arbitrary"):
        d = by_domain_category2[(domain, cat)]
        rate = d["bug"] / d["n"] if d["n"] else 0.0
        print(f"  {domain} / {cat}: n={d['n']}, verified-but-failed-rate={rate:.3f} ({d['bug']}/{d['n']})")

print()
print("===== STEP 3: among the 421 sft_eligible rows already passing, what fraction use mnemonic-adjacent codes? =====")
eligible = [r for r in scored if r["domain"] in ("fan", "valve") and r["sft_eligible"]]
print(f"total sft_eligible: {len(eligible)}")
elig_counts = defaultdict(int)
elig_unclassifiable = 0
elig_by_domain = defaultdict(lambda: defaultdict(int))
for r in eligible:
    c = classify_row(r, r["domain"])
    if c is None:
        elig_unclassifiable += 1
        continue
    cat = "mnemonic_adjacent" if c["pair_mnemonic"] else "arbitrary"
    elig_counts[cat] += 1
    elig_by_domain[r["domain"]][cat] += 1

print(f"unclassifiable: {elig_unclassifiable}")
n_classified = sum(elig_counts.values())
for cat in ("mnemonic_adjacent", "arbitrary"):
    n = elig_counts[cat]
    print(f"  {cat}: {n} ({100*n/n_classified:.1f}% of classified sft_eligible rows)")
print("by domain:")
for domain in ("fan", "valve"):
    total_d = sum(elig_by_domain[domain].values())
    for cat in ("mnemonic_adjacent", "arbitrary"):
        n = elig_by_domain[domain][cat]
        print(f"  {domain} / {cat}: {n} ({100*n/total_d if total_d else 0:.1f}% of {domain}'s eligible rows)")

# For comparison: what fraction of ALL verified rows (regardless of pass/fail) are mnemonic-adjacent,
# to know the baseline rate the model invents mnemonic codes at all.
print()
print("===== BASELINE: mnemonic-adjacent rate among ALL verified rows (pass or fail), for comparison =====")
all_v_counts = defaultdict(int)
for r in all_verified:
    c = classify_row(r, r["domain"])
    if c is None:
        continue
    cat = "mnemonic_adjacent" if c["pair_mnemonic"] else "arbitrary"
    all_v_counts[cat] += 1
n_all_v = sum(all_v_counts.values())
for cat in ("mnemonic_adjacent", "arbitrary"):
    n = all_v_counts[cat]
    print(f"  {cat}: {n} ({100*n/n_all_v:.1f}% of all verified rows)")
