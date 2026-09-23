import json, re
from pathlib import Path
from collections import defaultdict

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

verified = [r for r in scored if r["domain"] in ("fan", "valve") and r["verified"]]
assert len(verified) == 575, f"expected 575 verified rows, got {len(verified)}"

by_domain = defaultdict(lambda: {"n": 0, "mnemonic_or_domain_leaking": 0, "genuinely_arbitrary": 0, "unparseable": 0})
for r in verified:
    domain = r["domain"]
    mapping = recover_mapping(r)
    by_domain[domain]["n"] += 1
    if mapping is None:
        by_domain[domain]["unparseable"] += 1
        continue
    states = DOMAIN_STATES[domain]
    domain_word = DOMAIN_NAME[domain]
    leaks = any(is_mnemonic_adjacent(mapping.get(s), s) or is_mnemonic_adjacent(mapping.get(s), domain_word)
                for s in states if mapping.get(s))
    if leaks:
        by_domain[domain]["mnemonic_or_domain_leaking"] += 1
    else:
        by_domain[domain]["genuinely_arbitrary"] += 1

print("===== Breakdown of the 575 verified rows (the exact set the 7.05%/0.70x-baseline number is based on) =====")
total = {"n": 0, "mnemonic_or_domain_leaking": 0, "genuinely_arbitrary": 0, "unparseable": 0}
for domain, d in by_domain.items():
    print(f"  {domain}: n={d['n']}, mnemonic_or_domain_leaking={d['mnemonic_or_domain_leaking']} "
          f"({100*d['mnemonic_or_domain_leaking']/d['n']:.1f}%), genuinely_arbitrary={d['genuinely_arbitrary']} "
          f"({100*d['genuinely_arbitrary']/d['n']:.1f}%), unparseable={d['unparseable']}")
    for k in total: total[k] += d[k]
print(f"  POOLED: n={total['n']}, mnemonic_or_domain_leaking={total['mnemonic_or_domain_leaking']} "
      f"({100*total['mnemonic_or_domain_leaking']/total['n']:.1f}%), genuinely_arbitrary={total['genuinely_arbitrary']} "
      f"({100*total['genuinely_arbitrary']/total['n']:.1f}%)")
