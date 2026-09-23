"""Tests for Step 14's hard reward-gate: the annealing schedule, the freshly
re-derived reward invariant (including the disclosed vacuous-completion gap), the
"non-bank scenarios are provably unaffected" guarantee, and the extended
dynamic-sampling safety net's all-zero-reward stress test."""
import math

from synthetic_bridge import CODED_TRAJECTORIES, LITERAL_CONTROL_TRAJECTORIES
from step14_reward_gate import (
    BANK_SCENARIO_KEYS,
    bank_gate_resample_reasons,
    bank_gate_scale,
    is_bank_scenario,
    scenario_key_from_prompt,
    score_completion_gated,
    simulate_dynamic_sampling,
    verify_step14_reward_invariant,
)

# A mixed same/different bank scenario (both true states occur), so a correct
# encoding can actually reach taxonomy category 9 -- an all-"same" scenario never
# can, since taxonomy.py's category 9 requires both physical states to occur.
BANK_INDEX = 2
BANK_CODED_ROW = CODED_TRAJECTORIES[BANK_INDEX]
BANK_LITERAL_ROW = LITERAL_CONTROL_TRAJECTORIES[BANK_INDEX]
assert BANK_CODED_ROW["starting_state"] == BANK_LITERAL_ROW["starting_state"]
assert BANK_CODED_ROW["operations"] == BANK_LITERAL_ROW["operations"]

NON_BANK_PROMPT = (
    "Starting state: Heads\nInstructions:\n"
    "1. same as previous (the state does NOT change)\n"
    "2. same as previous (the state does NOT change)\n"
    "3. same as previous (the state does NOT change)\n"
    "4. same as previous (the state does NOT change)\n"
    "5. different from previous (the state flips)\n"
    "Reason through every flip in order. Put Step and State on the SAME line for "
    "every instruction."
)


def test_non_bank_scenario_key_is_not_in_bank():
    assert scenario_key_from_prompt(NON_BANK_PROMPT) not in BANK_SCENARIO_KEYS
    assert not is_bank_scenario(NON_BANK_PROMPT)


def test_bank_scenario_key_matches():
    assert is_bank_scenario(BANK_CODED_ROW["prompt"])
    assert is_bank_scenario(BANK_LITERAL_ROW["prompt"])


# ===== Anneal schedule =====

def test_ramp_starts_at_initial_scale_and_holds_at_final_scale():
    assert bank_gate_scale(1) == 0.25
    assert bank_gate_scale(30) == 1.0
    assert bank_gate_scale(31) == 1.0
    assert bank_gate_scale(150) == 1.0


def test_ramp_is_monotonically_nondecreasing():
    values = [bank_gate_scale(step) for step in range(1, 151)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_ramp_is_linear_at_midpoint():
    # step 15 of 1..30 is exactly the midpoint between initial and final scale.
    expected = 0.25 + (1.0 - 0.25) * (15 - 1) / (30 - 1)
    assert math.isclose(bank_gate_scale(15), expected, abs_tol=1e-12)


def test_ramp_rejects_invalid_inputs():
    for bad_kwargs in ({}, {"initial_scale": 0.5, "final_scale": 0.1}):
        try:
            bank_gate_scale(1, **bad_kwargs) if bad_kwargs else bank_gate_scale(0)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# ===== score_completion_gated =====

def test_literal_on_bank_scenario_is_gated_toward_zero():
    b = score_completion_gated(
        BANK_LITERAL_ROW["completion"], BANK_LITERAL_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_LITERAL_ROW["prompt"],
    )
    assert b["taxonomy_category"] == 1
    assert b["is_bank_scenario"] is True
    assert b["bank_gate_scale"] == 1.0  # step 100 >= ramp_steps=30
    assert b["total"] == 0.0
    assert b["ungated_total"] != 0.0


def test_literal_on_bank_scenario_at_step1_is_suppressed_not_zeroed():
    b = score_completion_gated(
        BANK_LITERAL_ROW["completion"], BANK_LITERAL_ROW["final_answer"],
        step=1, total_steps=150, prompt=BANK_LITERAL_ROW["prompt"],
    )
    assert b["bank_gate_scale"] == 0.25
    assert math.isclose(b["total"], b["ungated_total"] * 0.75, abs_tol=1e-9)
    assert b["total"] != 0.0


def test_correct_code_on_bank_scenario_is_unchanged_at_every_step():
    for step in (1, 15, 30, 100, 150):
        b = score_completion_gated(
            BANK_CODED_ROW["completion"], BANK_CODED_ROW["final_answer"],
            step=step, total_steps=150, prompt=BANK_CODED_ROW["prompt"],
        )
        assert b["taxonomy_category"] == 9, b["taxonomy_category_name"]
        assert b["bank_gate_scale"] is None
        assert b["total"] == b["ungated_total"]


def test_vacuous_on_bank_scenario_is_gated_toward_zero():
    """GAP 1 FIX: vacuous now ramps toward zero exactly like literal."""
    vacuous = "I computed it directly.\n<answer>" + BANK_LITERAL_ROW["final_answer"] + "</answer>"
    b = score_completion_gated(
        vacuous, BANK_LITERAL_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_LITERAL_ROW["prompt"],
    )
    assert b["taxonomy_category_name"] == "vacuous"
    assert b["is_bank_scenario"] is True
    assert b["bank_gate_scale"] == 1.0
    assert b["total"] == 0.0
    assert b["ungated_total"] > 0.0  # correct_vacuous is positive (~2.5) before gating


def test_malformed_literal_on_bank_scenario_is_not_gated():
    """GAP 3 FIX: a malformed completion (no closing <answer> tag) that still writes
    literal state words keeps its natural, very negative penalty -- it is no longer
    swept into the gate just because its State: lines happen to be literal."""
    malformed_but_literal = (
        "Step 1: flips. State: Heads\nStep 2: flips. State: Tails\n"
        "Final answer is Heads with no tag"
    )
    b = score_completion_gated(
        malformed_but_literal, BANK_CODED_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_CODED_ROW["prompt"],
    )
    assert b["taxonomy_category_name"] == "literal"
    assert b["bank_gate_scale"] is None
    assert b["total"] == b["ungated_total"]
    assert b["total"] < -5.0


def test_malformed_vacuous_on_bank_scenario_is_not_gated():
    """The same fix applied to the vacuous side: a malformed completion with zero
    state-slot tokens at all must also keep its natural penalty, not get swept in
    just because it's classified vacuous."""
    malformed_and_vacuous = "I thought about it for a while.\nNo final tag here."
    b = score_completion_gated(
        malformed_and_vacuous, BANK_CODED_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_CODED_ROW["prompt"],
    )
    assert b["taxonomy_category_name"] == "vacuous"
    assert b["bank_gate_scale"] is None
    assert b["total"] == b["ungated_total"]
    assert b["total"] < 0.0


def test_non_bank_scenarios_are_provably_unaffected():
    """Byte-identical to score_completion_v2 -- not just 'close' -- for any scenario
    outside the 16-scenario bank, at every step, regardless of completion content."""
    from reward_v3 import score_completion_v2

    completions = [
        "Step 1: same. State: Heads\nStep 2: same. State: Heads\nStep 3: same. State: Heads\n"
        "Step 4: same. State: Heads\nStep 5: flips. State: Tails\n<answer>Tails</answer>",
        "Step 1: The code is tracked. State: Zorp\nStep 2: same. State: Zorp\n"
        "Step 3: same. State: Zorp\nStep 4: same. State: Zorp\nStep 5: flips. State: Blipp\n"
        "<answer>Tails</answer>",
        "<answer>Heads</answer>",  # malformed/too-short
    ]
    for step in (1, 15, 30, 100, 150):
        for completion in completions:
            gated = score_completion_gated(
                completion, "Tails", step=step, total_steps=150, prompt=NON_BANK_PROMPT
            )
            base = score_completion_v2(completion, "Tails", step, 150, prompt=NON_BANK_PROMPT)
            assert gated["is_bank_scenario"] is False
            assert gated["bank_gate_scale"] is None
            assert gated["total"] == base["total"], (step, completion)
            assert gated["ungated_total"] == base["total"]


# ===== Reward invariant =====

def test_invariant_code_always_beats_gated_at_full_anneal():
    """The only UNCONDITIONAL guarantee: once the gate reaches final_scale=1.0,
    every gated completion's (literal OR vacuous) reward is exactly 0 regardless of
    its own base total, and category-9's worst-case floor (0.65, computed with
    r_consistency forced to 0 and r_signal forced to signal_magnitude -- see module
    docstring) is strictly positive."""
    report = verify_step14_reward_invariant()
    assert report["margin_code_over_gated_at_full_anneal"] > 0
    assert report["gated_at_full_anneal"] == 0.0


def test_invariant_discloses_the_during_ramp_gap_explicitly():
    """Documented, flagged, NOT fixed (assessed negligible): during the transient
    ramp (gate scale below 1.0), p_cot scans raw completion text for banned literal
    words independent of taxonomy category, so a contrived category-9 completion
    with literal-word-heavy surrounding prose can still hit p_cot's full penalty
    while a best-case gated completion suppressed only by the weak initial gate
    scale is not guaranteed to fall below it. Rechecked (not assumed unaffected)
    after gating vacuous too: literal's ceiling (4.15) remains the binding
    constraint, strictly above vacuous's ceiling (3.0), so widening the gate did not
    introduce a new or worse version of this gap."""
    report = verify_step14_reward_invariant()
    assert report["max_literal_base"] > report["max_vacuous_base"]
    assert report["margin_code_over_gated_during_ramp"] == report["margin_code_over_gated_literal_during_ramp"]
    assert report["code_over_gated_guaranteed_during_ramp"] is False
    assert report["margin_code_over_gated_literal_during_ramp"] < 0
    assert report["margin_code_over_gated_vacuous_during_ramp"] < 0
    # Still less severe than the literal case -- consistent with the recheck's conclusion.
    assert report["margin_code_over_gated_vacuous_during_ramp"] > report["margin_code_over_gated_literal_during_ramp"]


def test_invariant_gated_always_beats_malformed():
    """GAP 3 FIX: malformed is now excluded from the gate entirely (any category),
    so this holds unconditionally across ALL malformed completions, not just a
    'non-literal only' restricted subset the way the pre-fix version needed."""
    report = verify_step14_reward_invariant()
    assert report["margin_gated_over_malformed"] > 0
    assert report["malformed_ties_gated_at_full_anneal"] is False


def test_invariant_vacuous_no_longer_exceeds_gated_literal():
    """GAP 1 FIX, confirmed via the symbolic report."""
    report = verify_step14_reward_invariant()
    assert report["vacuous_exceeds_gated_literal_at_full_anneal"] is False


def test_gap1_and_gap3_fixes_combined_no_new_interaction_bug():
    """Combined-fix regression test (not just each fix in isolation): a malformed
    AND vacuous completion (the intersection of both fixes' scope) must still keep
    its natural penalty, while a well-formed vacuous completion must still be gated
    to 0 at full anneal -- confirms the `category in GATED_CATEGORIES and r_task !=
    -5.0` conjunction composes correctly rather than one condition masking the
    other."""
    malformed_vacuous = "Not sure.\nNo tag."
    well_formed_vacuous = "I computed it directly.\n<answer>" + BANK_CODED_ROW["final_answer"] + "</answer>"

    b_malformed = score_completion_gated(
        malformed_vacuous, BANK_CODED_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_CODED_ROW["prompt"],
    )
    b_wellformed = score_completion_gated(
        well_formed_vacuous, BANK_CODED_ROW["final_answer"],
        step=100, total_steps=150, prompt=BANK_CODED_ROW["prompt"],
    )
    assert b_malformed["taxonomy_category_name"] == "vacuous"
    assert b_malformed["bank_gate_scale"] is None
    assert b_malformed["total"] < 0.0  # kept its natural malformed-ish penalty

    assert b_wellformed["taxonomy_category_name"] == "vacuous"
    assert b_wellformed["bank_gate_scale"] == 1.0
    assert b_wellformed["total"] == 0.0

    # And the two must NOT be equal -- that would silently reproduce gap 3's tie.
    assert b_malformed["total"] != b_wellformed["total"]
    assert b_malformed["total"] < b_wellformed["total"]


# ===== Dynamic-sampling safety net =====

def test_existing_correctness_check_alone_misses_all_zero_reward_batch():
    """Demonstrates the exact gap design requirement 5 describes: a group with a
    non-unanimous correctness split (so the OLD check would not fire) can still have
    every reward pinned at 0 once the gate is fully annealed."""
    rewards = [0.0] * 8
    reasons_without_reward_check = []
    correct_count = 3  # neither 0 nor 8 -- old correctness check would not fire
    if correct_count in (0, 8):
        reasons_without_reward_check.append("correctness")
    assert reasons_without_reward_check == [], "test setup must reproduce the gap"
    # The extended check catches it:
    reasons = bank_gate_resample_reasons(
        rewards, correct_count=correct_count, structural_count=8, group_size=8
    )
    assert "all_zero_reward" in reasons


def test_resample_reasons_are_empty_for_a_healthy_varied_group():
    rewards = [3.65, -1.04, 0.0, 2.5, -5.13, 3.15, -2.0, 0.3]
    reasons = bank_gate_resample_reasons(
        rewards, correct_count=3, structural_count=6, group_size=8
    )
    assert reasons == []


def test_all_zero_reward_stress_test_falls_back_without_crashing_after_max_attempts():
    """Simulates the worst case named in design requirement 5: every one of
    MAX_DYNAMIC_ATTEMPTS resample attempts comes back entirely zero-reward (the model
    has not yet produced anything but literal completions on this bank scenario, and
    the gate is fully annealed). The safety net must not crash, must not silently
    accept a degenerate batch unflagged, and must terminate."""
    max_attempts = 3
    group_size = 8
    attempts_rewards = [[0.0] * group_size for _ in range(max_attempts)]
    correct_counts = [4, 2, 5]  # varies -- old check alone would never resample this
    structural_counts = [8, 8, 8]
    result = simulate_dynamic_sampling(
        attempts_rewards, correct_counts=correct_counts, structural_counts=structural_counts,
        group_size=group_size, max_attempts=max_attempts,
    )
    assert result["accepted"] is True
    assert result["attempt"] == max_attempts
    assert result["fallback"] is True, "must be explicitly flagged, not silently accepted"
    assert "all_zero_reward" in result["reasons"]


def test_dynamic_sampling_accepts_early_once_a_healthy_attempt_appears():
    attempts_rewards = [[0.0] * 8, [3.65, -1.04, 0.0, 2.5, -5.13, 3.15, -2.0, 0.3], [0.0] * 8]
    correct_counts = [4, 3, 4]
    structural_counts = [8, 6, 8]
    result = simulate_dynamic_sampling(
        attempts_rewards, correct_counts=correct_counts, structural_counts=structural_counts,
        group_size=8, max_attempts=3,
    )
    assert result["attempt"] == 2
    assert result["fallback"] is False
    assert result["reasons"] == []
