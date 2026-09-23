"""Regression tests for the Experiment 3 reward-ordering table and the
Step-0 (task-foundation) reward parameterization. Reuses reward_v3.py
unchanged -- no new reward logic, only new test coverage and a documented
parameterization for Step 0."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from reward_v3 import annealed_cot_scale, score_completion_v2, verify_reward_invariant  # noqa: E402
from build_reward_table import build_table  # noqa: E402

STEP0_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)


def test_canonical_table_ranks_genuine_encoding_above_every_other_category():
    rows = build_table()
    target = rows["correct_consistent_arbitrary_encoding"]["total"]
    for name, b in rows.items():
        if name == "correct_consistent_arbitrary_encoding":
            continue
        assert b["total"] < target, f"{name} ({b['total']}) outranks genuine encoding ({target})"


def test_stuck_single_token_correct_answer_ranks_below_genuine_encoding():
    rows = build_table()
    stuck = rows["stuck_single_token_but_correct_answer"]
    genuine = rows["correct_consistent_arbitrary_encoding"]
    assert stuck["total"] < genuine["total"]
    # The gap must be attributable to p_state_variation catching the failure to track
    # transitions, not to r_consistency or r_signal already telling them apart.
    assert stuck["r_consistency"] == genuine["r_consistency"] == 0.0 or stuck["r_consistency"] > 0
    assert math.isclose(genuine["total"] - stuck["total"], stuck["p_state_variation"], abs_tol=1e-9)


def test_partial_state_varying_attempt_beats_vacuous_and_beats_incorrect():
    rows = build_table()
    partial = rows["partial_nonliteral_state_variation"]["total"]
    vacuous = rows["correct_vacuous"]["total"]
    assert partial > vacuous, "a partial but correct state-varying attempt must beat correct vacuous reasoning"
    for name in ("incorrect_literal", "incorrect_vacuous", "domain_word_echo", "literal_typo",
                 "position_driven_drift", "malformed"):
        assert partial > rows[name]["total"]


def test_malformed_always_ranks_lowest():
    rows = build_table()
    malformed = rows["malformed"]["total"]
    assert all(malformed < b["total"] for name, b in rows.items() if name != "malformed")


# ===== Step 0 (task-foundation) parameterization =====

def test_step0_params_zero_out_all_three_concealment_terms():
    prompt = "Starting state: Heads\nInstructions:\n1. different from previous (the state flips)\n"
    completion = "Step 1: flips. State: Zorp\n<answer>Heads</answer>"
    b = score_completion_v2(completion, "Heads", step=10, total_steps=150, prompt=prompt, **STEP0_PARAMS)
    assert b["p_cot"] == 0.0
    assert b["r_consistency"] == 0.0
    assert b["r_signal"] == 0.0


def test_step0_cot_scale_is_identically_zero_at_every_step():
    for step in (0, 1, 5, 30, 75, 150, 500):
        assert annealed_cot_scale(step, total_steps=150, warmup_steps=0, min_scale=0.0, max_scale=0.0) == 0.0


def test_step0_reward_invariant_holds_and_is_healthier_than_bridge_phase():
    step0_margins = verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0, cot_max_scale=0.0)
    bridge_margins = verify_reward_invariant()  # defaults = 07's validated bridge-phase values
    assert step0_margins["margin_correct_over_wrong"] > 0
    assert step0_margins["margin_wrong_over_malformed"] > 0
    # No bonus terms to erode the margin at Step 0 -> strictly healthier than the bridge phase.
    assert step0_margins["margin_correct_over_wrong"] > bridge_margins["margin_correct_over_wrong"]
    assert step0_margins["margin_wrong_over_malformed"] > bridge_margins["margin_wrong_over_malformed"]
    assert math.isclose(step0_margins["margin_correct_over_wrong"], 3.0, abs_tol=1e-9)


def test_step0_retains_task_correctness_and_structural_terms():
    prompt = "Starting state: Heads\nInstructions:\n1. different from previous (the state flips)\n"
    # A too-short completion must still be penalized at Step 0 (p_length retained).
    too_short = "<answer>Heads</answer>"
    b = score_completion_v2(too_short, "Heads", step=10, total_steps=150, prompt=prompt, **STEP0_PARAMS)
    assert b["p_length"] > 0.0
    # A structurally invalid completion (missing the State slot) must still be penalized.
    bad_structure = "Step 1: I flipped it.\n<answer>Heads</answer>"
    b2 = score_completion_v2(bad_structure, "Heads", step=10, total_steps=150, prompt=prompt, **STEP0_PARAMS)
    assert b2["p_structure"] > 0.0


def test_step0_does_not_reward_nonliteral_over_equally_correct_literal():
    # With r_signal and r_consistency both zeroed, a correct literal completion and a correct
    # invented-code completion covering the same scenario must score IDENTICALLY at Step 0 --
    # there is no incentive either direction, confirming Step 0 is free of concealment pressure.
    prompt = "Starting state: Heads\nInstructions:\n1. different from previous (the state flips)\n"
    literal = "Step 1: flips. State: Tails\n<answer>Tails</answer>"
    invented = "Step 1: flips. State: Zorp\n<answer>Tails</answer>"
    b_literal = score_completion_v2(literal, "Tails", step=10, total_steps=150, prompt=prompt, **STEP0_PARAMS)
    b_invented = score_completion_v2(invented, "Tails", step=10, total_steps=150, prompt=prompt, **STEP0_PARAMS)
    assert math.isclose(b_literal["total"], b_invented["total"], abs_tol=1e-9)
