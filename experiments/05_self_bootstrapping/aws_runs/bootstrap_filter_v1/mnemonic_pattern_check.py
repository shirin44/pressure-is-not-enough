import json, re
from pathlib import Path

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

DECODE_RE = re.compile(r"Final coded state:\s*([A-Za-z]+)\.\s*\1 represents ([A-Za-z]+)\.", re.IGNORECASE)

for domain in ("fan", "valve"):
    candidates = [r for r in scored if r["domain"] == domain and r["verified"] and not r["sft_eligible"]]
    code_in_answer_count = 0
    other_count = 0
    for r in candidates:
        s = r["score"]
        m = DECODE_RE.search(r["completion"])
        if not m:
            other_count += 1
            continue
        decoded_code, decoded_state = m.group(1), m.group(2)
        raw_answer = s.get("raw_answer") or ""
        if raw_answer.casefold() == decoded_code.casefold() and decoded_state.casefold() != raw_answer.casefold():
            code_in_answer_count += 1
        else:
            other_count += 1
    n = len(candidates)
    print(f"{domain}: n={n}, code-word-copied-into-answer-tag={code_in_answer_count} "
          f"({100*code_in_answer_count/n:.1f}%), other-failure-pattern={other_count} ({100*other_count/n if n else 0:.1f}%)")
