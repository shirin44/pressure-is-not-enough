import json, random, re, sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_self_bootstrapping" / "data_generation"))
from filter_bootstrap_successes import build_sft_dataset, reverify_sft_dataset

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

DOMAIN_STATES = {"fan": ("Running", "Stopped"), "valve": ("Open", "Closed")}
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

def classify_row(row, domain):
    mapping = recover_mapping(row)
    if mapping is None:
        return None
    states = DOMAIN_STATES[domain]
    per_state_mnemonic = {s: (is_mnemonic_adjacent(mapping.get(s), s) if mapping.get(s) else None) for s in states}
    return "mnemonic_adjacent" if any(v for v in per_state_mnemonic.values() if v is not None) else "arbitrary"

eligible = [r for r in scored if r["domain"] in ("fan", "valve") and r["sft_eligible"]]
for r in eligible:
    r["_category"] = classify_row(r, r["domain"])

fan_arbitrary = [r for r in eligible if r["domain"] == "fan" and r["_category"] == "arbitrary"]
valve_arbitrary = [r for r in eligible if r["domain"] == "valve" and r["_category"] == "arbitrary"]
fan_all = [r for r in eligible if r["domain"] == "fan"]
valve_all = [r for r in eligible if r["domain"] == "valve"]

print(f"corrected counts -- fan/arbitrary/sft_eligible: {len(fan_arbitrary)}, valve/arbitrary/sft_eligible: {len(valve_arbitrary)}")
print(f"fan/sft_eligible total: {len(fan_all)}, valve/sft_eligible total: {len(valve_all)}")

RNG_SEED = 20260825

def balanced_sample(rows_a, rows_b, cap, seed):
    rng = random.Random(seed)
    a = sorted(rows_a, key=lambda r: r["example_id"])
    b = sorted(rows_b, key=lambda r: r["example_id"])
    rng.shuffle(a); rng.shuffle(b)
    return a[:cap] + b[:cap]

# Dataset A: arbitrary-only, domain-balanced at the smaller domain's eligible-arbitrary count (23).
CAP_A = min(len(fan_arbitrary), len(valve_arbitrary))
dataset_a_rows = balanced_sample(fan_arbitrary, valve_arbitrary, CAP_A, RNG_SEED)
print(f"\nDataset A (arbitrary-only, balanced): {len(dataset_a_rows)} total ({CAP_A} fan + {CAP_A} valve)")

# Dataset B: all sft_eligible, domain-balanced at the smaller domain's total (124, fan).
CAP_B = min(len(fan_all), len(valve_all))
dataset_b_rows = balanced_sample(fan_all, valve_all, CAP_B, RNG_SEED)
print(f"Dataset B (all sft_eligible, balanced): {len(dataset_b_rows)} total ({CAP_B} fan + {CAP_B} valve)")

# Report mnemonic/arbitrary composition of each built dataset for transparency
for name, rows in (("A", dataset_a_rows), ("B", dataset_b_rows)):
    m = sum(1 for r in rows if r["_category"] == "mnemonic_adjacent")
    a = sum(1 for r in rows if r["_category"] == "arbitrary")
    print(f"  Dataset {name} composition: mnemonic_adjacent={m}, arbitrary={a}")

def finalize(rows, out_name, out_dir):
    for r in rows:
        r.pop("_category", None)
    dataset = build_sft_dataset(rows)
    # build_sft_dataset already filters to sft_eligible internally, but our rows are already
    # pre-filtered to sft_eligible=True, so this should be a lossless 1:1 pass-through -- verify that.
    assert len(dataset) == len(rows), f"expected lossless pass-through, got {len(dataset)} from {len(rows)} input rows"
    audit = reverify_sft_dataset(dataset)
    print(f"\n{out_name} re-verification: {json.dumps({k:v for k,v in audit.items() if k != 'failures'}, indent=2)}")
    if not audit["accepted"]:
        raise RuntimeError(f"{out_name} FAILED re-verification: {audit['failures'][:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{out_name}.json").write_text(json.dumps({
        "name": out_name, "audit": audit, "count": len(dataset), "rows": dataset,
    }, indent=2))
    print(f"Wrote {len(dataset)} examples to {out_dir / (out_name + '.json')}")

OUT_DIR = HOME / "aisi_checkpoints" / "bootstrap_sft_datasets"
finalize([dict(r) for r in dataset_a_rows], "dataset_A_arbitrary_only", OUT_DIR)
finalize([dict(r) for r in dataset_b_rows], "dataset_B_all_eligible_balanced", OUT_DIR)
