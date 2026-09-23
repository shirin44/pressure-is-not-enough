import json
from pathlib import Path

HOME = Path.home()
scored = json.loads((HOME / "aisi_checkpoints" / "bootstrap_filter_v1" / "scored_rows.json").read_text())

candidates = [r for r in scored if r["domain"] == "fan" and r["verified"] and not r["sft_eligible"]]
print(f"total fan verified-but-not-sft_eligible: {len(candidates)}")
print()

for r in candidates[:15]:
    s = r["score"]
    print("=" * 100)
    print(f"example_id={r['example_id']}  initial_state={r['initial_state']}  operations={r['operations']}")
    print(f"expected_states={r['expected_states']}  ground_truth_final_answer={r['final_answer']}")
    print(f"score: structural={s['structural']} global_consistent={s['global_consistent']} "
          f"nonliteral_consistent={s['nonliteral_consistent']} decode_back_self_consistent={s['decode_back_self_consistent']} "
          f"answer_correct={s['answer_correct']}")
    print(f"raw_answer_extracted={s['raw_answer']!r}  token_pair={s['token_pair']}")
    print("--- completion text ---")
    print(r["completion"])
    print()
