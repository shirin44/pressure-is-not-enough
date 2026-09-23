import json, re
from pathlib import Path
from collections import Counter

HOME = Path.home()
progress = json.loads((HOME / "aisi_checkpoints" / "dataset_b_sft_dryrun_v1" / "step8_eval_progress.json").read_text())
undeclared_rows = progress["undeclared"]
print(f"total undeclared eval rows: {len(undeclared_rows)}")

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

def recover_mapping(text, expected_states_guess_len):
    prefix = text.split("<answer>", 1)[0]
    matches = [(int(i), tok) for i, tok in STEP_RE.findall(prefix)]
    tokens = [tok for _, tok in matches]
    return tokens  # raw token sequence; need expected_states to build the state->code map properly

# We need expected_states per row to build the mapping -- these aren't stored in the eval progress rows
# (score_undeclared_completion doesn't persist them), so re-derive the SAME deterministic lamp eval set
# used by the training script (seed 20260812) and join on example_id, same technique as the declared-lamp-
# control check earlier this session.
import sys
sys.path.insert(0, str(HOME / "AISI"))
from src.data.multidomain_seed import generate_multidomain_dataset
from src.data.undeclared_generalization import build_undeclared_lamp_evaluation

_, declared_eval = generate_multidomain_dataset(seed=20260812, corrected_fan_wording=True)
undeclared_lamp = build_undeclared_lamp_evaluation(declared_eval["lamp"])
expected_by_id = {row.example_id: row.expected_states for row in undeclared_lamp}

dark_verified = [r for r in undeclared_rows if r.get("both_states_observed") and r.get("nonliteral_consistent")
                 and r.get("decode_back_self_consistent")]
# final_answer isn't stored on r directly by the OLD scoring path, but our adapted script DOES store it now.
dark_verified = [r for r in dark_verified if r.get("final_answer") == "Dark"]
print(f"Dark-truth verified rows at step 8: {len(dark_verified)}")

clean_count = 0
leak_count = 0
unparseable = 0
details = []
for r in dark_verified:
    expected_states = expected_by_id.get(r["example_id"])
    if expected_states is None:
        unparseable += 1
        continue
    prefix = r["text"].split("<answer>", 1)[0]
    matches = [(int(i), tok) for i, tok in STEP_RE.findall(prefix)]
    tokens = [tok for _, tok in matches]
    if len(tokens) != len(expected_states):
        unparseable += 1
        continue
    mapping = {}
    for state, token in zip(expected_states, tokens):
        mapping.setdefault(state, token)
    states = DOMAIN_STATES["lamp"]; domain_word = DOMAIN_NAME["lamp"]
    leaks = any(is_mnemonic_adjacent(mapping.get(s), s) or is_mnemonic_adjacent(mapping.get(s), domain_word)
                for s in states if mapping.get(s))
    if leaks:
        leak_count += 1
    else:
        clean_count += 1
    details.append({"example_id": r["example_id"], "mapping": mapping, "leaks": leaks})

n = len(dark_verified)
print(f"\n===== MNEMONIC/DOMAIN-LEAK COMPOSITION OF STEP-8 DARK-TRUTH VERIFIED SUCCESSES =====")
print(f"total: {n}, unparseable: {unparseable}")
print(f"mnemonic/domain-leaking: {leak_count} ({100*leak_count/n:.1f}%)" if n else "n/a")
print(f"genuinely clean: {clean_count} ({100*clean_count/n:.1f}%)" if n else "n/a")
print(f"\nfull detail:")
for d in details:
    print(d)
