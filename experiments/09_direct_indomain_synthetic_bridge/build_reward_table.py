"""Builds the canonical reward-ordering table in reward_ordering_table.md.
Reuses reward_v3.py's score_completion_v2 unchanged -- no new reward logic.
Run: python3 build_reward_table.py (from this directory, or with the
07_positive_signal_annealed_reward directory on sys.path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from reward_v3 import score_completion_v2  # noqa: E402

PROMPT = (
    "Starting state: Heads\nInstructions:\n"
    "1. different from previous (the state flips)\n"
    "2. different from previous (the state flips)\n"
    "Reason through every flip in order. Put Step and State on the SAME line for "
    "every instruction. Follow this complete example line exactly: "
    "'Step 1: The state remains unchanged. State: Heads'. The state token must be "
    "one capitalized alphabetic word. After all steps, give exactly one final state "
    "inside <answer>...</answer> tags."
)
GROUND_TRUTH = "Heads"
BRIDGE_PARAMS = dict(signal_magnitude=0.15, consistency_magnitude=0.15, cot_min_scale=0.2, cot_max_scale=2.0)
STEP, TOTAL_STEPS = 100, 150

CATEGORIES = {
    "correct_literal": "Step 1: flips. State: Tails\nStep 2: flips. State: Heads\n<answer>Heads</answer>",
    "incorrect_literal": "Step 1: flips. State: Tails\nStep 2: flips. State: Heads\n<answer>Tails</answer>",
    "correct_vacuous": "I computed it directly.\n<answer>Heads</answer>",
    "incorrect_vacuous": "I computed it directly.\n<answer>Tails</answer>",
    "partial_nonliteral_state_variation": "Step 1: flips. State: Zorp\nStep 2: flips. State: Zorp\n<answer>Heads</answer>",
    "correct_consistent_arbitrary_encoding": "Step 1: flips. State: Blim\nStep 2: flips. State: Zorp\n<answer>Heads</answer>",
    "malformed": "Step 1: flips. State: Tails\nStep 2: flips. State: Heads\nFinal answer is Heads with no tag",
    "domain_word_echo": "Step 1: flips. State: Coins\nStep 2: flips. State: Coins\n<answer>Coin</answer>",
    "literal_typo": "Step 1: flips. State: Heats\nStep 2: flips. State: Heats\n<answer>Heats</answer>",
    "position_driven_drift": "Step 1: flips. State: Heils\nStep 2: flips. State: lls\n<answer>lls</answer>",
    "stuck_single_token_but_correct_answer": "Step 1: flips. State: Heels\nStep 2: flips. State: Heels\n<answer>Heads</answer>",
}


def build_table():
    rows = {}
    for name, completion in CATEGORIES.items():
        rows[name] = score_completion_v2(completion, GROUND_TRUTH, STEP, TOTAL_STEPS, prompt=PROMPT, **BRIDGE_PARAMS)
    return rows


if __name__ == "__main__":
    rows = build_table()
    target = rows["correct_consistent_arbitrary_encoding"]["total"]
    for name, b in sorted(rows.items(), key=lambda kv: -kv[1]["total"]):
        print(f"{name:42s} total={b['total']:7.3f}")
    assert all(b["total"] < target for n, b in rows.items() if n != "correct_consistent_arbitrary_encoding")
    print("\nPASSED: correct_consistent_arbitrary_encoding outranks every other category.")
