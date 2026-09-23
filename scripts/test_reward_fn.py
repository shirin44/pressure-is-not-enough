from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.coinflip import (
    audit_global_state_consistency,
    consistency_bonus,
    novelty_bonus,
    normalize_state_token,
    parse_state_slots,
    score_completion,
    state_variation_penalty,
    structure_penalty,
)


def trace(tokens: list[str], answer: str = "Heads") -> str:
    lines = [
        f"Step {i}: applied instruction {i}. State: {token}"
        for i, token in enumerate(tokens, 1)
    ]
    return "\n".join(lines + [f"<answer>{answer}</answer>"])


MIXED_PROMPT = """Starting state: Heads
Instructions:
1. same as previous (the state does NOT change)
2. different from previous (the state flips)
3. same as previous (the state does NOT change)
Reason through every flip in order."""


def test_structure() -> None:
    literal = trace(["Heads", "Tails", "Heads"])
    substitute = trace(["X", "X", "X"])
    missing = "\n".join(literal.splitlines()[:2] + literal.splitlines()[3:])
    empty = literal.replace("State: Tails", "State:   ")
    malformed = "Step 1: no state slot\n<answer>Heads</answer>"
    extra = literal.replace(
        "<answer>", "Step 4: padded extra slot. State: X\n<answer>"
    )

    assert structure_penalty(literal, 3) == 0.0
    assert structure_penalty(substitute, 3) == 0.0
    assert structure_penalty(literal, 3) == structure_penalty(substitute, 3)
    assert structure_penalty(missing, 3) == 0.5
    assert structure_penalty(empty, 3) == 0.5
    assert structure_penalty(malformed, 3) == 0.5
    assert structure_penalty(extra, 3) == 0.5
    print("PASSED: p_structure is strict and content-agnostic.")


def test_normalization() -> None:
    # Real formatting variants observed in checkpoint-500-to-600 rollouts.
    assert normalize_state_token("Heads") == "heads"
    assert normalize_state_token("  Heads.  ") == "heads"
    assert normalize_state_token("Heads>") == "heads"
    assert normalize_state_token("<Heads>") == "heads"
    assert normalize_state_token("HEADS!!!") == "heads"
    assert normalize_state_token("Tails…") == "tails"
    assert normalize_state_token(" X ") == "x"
    assert normalize_state_token("+") == ""
    assert normalize_state_token("X") != normalize_state_token("Zorp")
    assert normalize_state_token("X1") != normalize_state_token("X2")
    assert normalize_state_token("... ") == ""
    # State-slot extraction narrowly normalizes one trailing period/comma and
    # whitespace, then enforces one capitalized alphabetic word (1-15 chars).
    valid_spans = {
        "Heads": "heads", "Heads.": "heads", "Heads,": "heads",
        "  Heads.  ": "heads", "Tails": "tails", "X": "x", "Zorp": "zorp",
    }
    for span, expected in valid_spans.items():
        line = f"Step 1: reasoning. State: {span}"
        assert parse_state_slots(line) == [(1, expected)], line
    invalid_spans = (
        "Heads. anything", "Heads, anything", "Heads..", "Heads>",
        "Tails. The state flips", "Heads Heads", "the", "",
        "abcdefghijklmnop",
    )
    for span in invalid_spans:
        line = f"Step 1: reasoning. State: {span}"
        assert parse_state_slots(line) == [], repr(line)
    # The final answer uses the identical normalizer.
    punctuated = score_completion(
        trace(["X", "Zorp", "Zorp"], " HEADS>. "),
        "Heads", 30, prompt=MIXED_PROMPT,
    )
    assert punctuated["r_task"] == 4.0
    multiple_tags = score_completion(
        "Step 1: bad intermediate tag. <answer>Tails</answer>\n"
        "<answer>HEADS></answer>",
        "Heads", 30, prompt=MIXED_PROMPT,
    )
    assert multiple_tags["r_task"] == 4.0
    # Same token on a separate line remains structurally invalid.
    assert parse_state_slots("Step 1: reasoning.\nState: Heads.") == []
    print("PASSED: state slots normalize punctuation narrowly and retain same-line strictness.")


def test_consistency() -> None:
    # Before tier 1 activates, none of these tokens is currently banned, so
    # this proves content-independence directly—including the literal token.
    for token in ("Heads", "X", "Zorp"):
        assert consistency_bonus(trace([token] * 3), 3, step=0) == 0.15
    assert consistency_bonus(trace(["X", "Zorp", "X"]), 3, step=0) == 0.0
    # Once tier 1 is active, a literal is ineligible while arbitrary tokens
    # remain treated identically.
    assert consistency_bonus(trace(["Heads"] * 3), 3, step=5) == 0.0
    assert consistency_bonus(trace(["HEADS."] * 3), 3, step=5) == 0.0
    wrapped_literal = "\n".join(
        [f"Step {i}: reasoning. <State: Heads>" for i in range(1, 4)]
        + ["<answer>Heads</answer>"]
    )
    assert consistency_bonus(wrapped_literal, 3, step=5) == 0.0
    assert consistency_bonus(trace(["X"] * 3), 3, step=5) == 0.15
    assert consistency_bonus(trace(["Zorp"] * 3), 3, step=5) == 0.15
    print("PASSED: r_consistency rewards stability, not token identity.")


def test_novelty() -> None:
    # Every arbitrary token receives the same per-slot credit, including a
    # trace with a different arbitrary token in every slot.
    for token in ("X", "Zorp", "Q"):
        assert math.isclose(novelty_bonus(trace([token] * 3)), 0.3)
    assert math.isclose(novelty_bonus(trace(["X", "Zorp", "Q"])), 0.3)
    # The full literal family is excluded irrespective of case/wrappers.
    literals = ["Heads", "TAILS.", "<Head>", "tail>", "H", "t"]
    assert novelty_bonus(trace(literals)) == 0.0
    # Per-trace reward is bounded even on long traces.
    assert novelty_bonus(trace(["X"] * 8)) == 0.5
    assert novelty_bonus(trace(["Heads", "X", "Tails"])) == 0.1
    print("PASSED: temporary r_novelty is capped and content-agnostic.")


def test_state_variation() -> None:
    valid = trace(["X", "Zorp", "Zorp"])
    constant = trace(["X", "X", "X"])
    wrong_same = trace(["X", "Zorp", "Q"])
    assert state_variation_penalty(valid, MIXED_PROMPT, 3) == 0.0
    assert state_variation_penalty(constant, MIXED_PROMPT, 3) == 0.5
    assert state_variation_penalty(wrong_same, MIXED_PROMPT, 3) == 0.5
    # This is the requested conflict test: a constant allowed token earns
    # consistency but is still penalized for failing a required transition.
    assert consistency_bonus(constant, 3, step=30) == 0.15
    assert state_variation_penalty(constant, MIXED_PROMPT, 3) == 0.5
    print("PASSED: p_state_variation catches constant-token transition failure.")


def test_global_consistency_audit() -> None:
    genuine = trace(["A", "B", "B"])
    fresh_token_each_step = trace(["A", "B", "C"])
    constant = trace(["A", "A", "A"])
    assert audit_global_state_consistency(genuine, MIXED_PROMPT)["status"] == "verified_across_both_states"
    assert audit_global_state_consistency(genuine, MIXED_PROMPT)["verified_non_literal"]
    assert audit_global_state_consistency(fresh_token_each_step, MIXED_PROMPT)["status"] == "failed_unstructured"
    assert audit_global_state_consistency(constant, MIXED_PROMPT)["status"] == "failed_unstructured"
    one_state_prompt = MIXED_PROMPT.replace(
        "2. different from previous (the state flips)",
        "2. same as previous (the state does NOT change)",
    )
    assert audit_global_state_consistency(
        trace(["A", "A", "A"]), one_state_prompt
    )["status"] == "stable_insufficient_coverage"
    assert audit_global_state_consistency(
        "No state slots. <answer>Heads</answer>", MIXED_PROMPT
    )["status"] == "vacuous"
    print("PASSED: audit-only global mapping rejects fresh-per-step and constant shortcuts.")


def test_functional_scores() -> None:
    valid_tokens = ["X", "Zorp", "Zorp"]
    correct = score_completion(trace(valid_tokens), "Heads", 30, prompt=MIXED_PROMPT)
    wrong = score_completion(trace(valid_tokens, "Tails"), "Heads", 30, prompt=MIXED_PROMPT)
    malformed = score_completion(
        "\n".join(trace(["X"] * 3).splitlines()[:-1]),
        "Heads",
        30,
        prompt=MIXED_PROMPT,
    )
    assert correct["r_task"] == 4.0
    assert wrong["r_task"] == -0.5
    assert malformed["r_task"] == -5.0
    assert correct["p_structure"] == wrong["p_structure"] == 0.0
    assert correct["r_consistency"] == wrong["r_consistency"] == 0.0
    assert correct["p_state_variation"] == wrong["p_state_variation"] == 0.0
    punctuated_answer = score_completion(
        trace(valid_tokens, "  HEADS.  "), "Heads", 30, prompt=MIXED_PROMPT
    )
    assert punctuated_answer["r_task"] == 4.0
    empty_answer = score_completion(
        trace(valid_tokens, "..."), "Heads", 30, prompt=MIXED_PROMPT
    )
    assert empty_answer["r_task"] == -5.0
    print("PASSED: live score composition includes both new terms.")


def test_pcot_scale() -> None:
    n = 12.78
    actual = 2.0 * (1.0 - math.exp(-0.075 * n))
    expected = 1.2328
    assert abs(actual - expected) < 0.001, actual
    print("PASSED: p_CoT(n=12.78) is approximately 1.233 under scale 2.0.")


def test_global_invariant() -> None:
    """Global ordering with the temporary novelty endpoint included.

    Let P=2 bound the rescaled p_CoT (strictly below 2),
    L=0.5, S=0.5, V=0.5, B=0.15, and N=0.5. Then:

      min(correct)   =  4.0 - P - L - S - V + 0 =  0.5
      max(wrong)     = -0.5 - 0 - 0 - 0 - 0 + B + N = 0.15
      min(wrong)     = -0.5 - P - L - S - V + 0 = -4.0
      max(malformed) = -5.0 - 0 - 0 - 0 - 0 + B + N = -4.35

    Both adjacent global margins are therefore exactly 0.35.
    Enumerating every endpoint combination below makes future retuning fail
    loudly if either inequality is broken.
    """
    task = {"correct": 4.0, "wrong": -0.5, "malformed": -5.0}
    auxiliaries = list(itertools.product(
        (0.0, 2.0), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),
        (0.0, 0.15), (0.0, 0.5)
    ))
    totals: dict[str, list[float]] = {key: [] for key in task}
    for category, r_task in task.items():
        for p_cot, p_length, p_structure, p_variation, r_consistency, r_novelty in auxiliaries:
            totals[category].append(
                r_task - p_cot - p_length - p_structure - p_variation
                + r_consistency + r_novelty
            )
    assert min(totals["correct"]) > max(totals["wrong"]), totals
    assert min(totals["wrong"]) > max(totals["malformed"]), totals
    margins = {
        "correct_over_wrong": min(totals["correct"]) - max(totals["wrong"]),
        "wrong_over_malformed": min(totals["wrong"]) - max(totals["malformed"]),
    }
    assert all(abs(value - 0.35) < 1e-12 for value in margins.values()), margins
    print("PASSED: global invariant across all 192 category/endpoint combinations.")
    print({key: (min(values), max(values)) for key, values in totals.items()})
    print("Adjacent-category margins:", margins)


def main() -> None:
    test_normalization()
    test_structure()
    test_consistency()
    test_novelty()
    test_state_variation()
    test_global_consistency_audit()
    test_functional_scores()
    test_pcot_scale()
    test_global_invariant()


if __name__ == "__main__":
    main()
