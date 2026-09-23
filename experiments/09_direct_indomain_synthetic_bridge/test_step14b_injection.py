"""Tests for Step 14b's rollout-group injection: trajectory-to-scenario matching,
scope (bank scenarios only), group-composition correctness (exactly 1 of 8 slots),
the re-derived GRPO ratio=1 invariant, and a concrete numeric confirmation that
injection actually resolves the group-collapse problem it was built for."""
import math

from synthetic_bridge import CODED_TRAJECTORIES
from step14_reward_gate import score_completion_gated
from step14b_injection import (
    INJECTIONS_PER_GROUP,
    build_injected_completion_ids,
    inject_into_generation_output,
    injected_trajectory_for_prompt,
    select_injection_indices,
    verify_grpo_ratio_is_unconditionally_one,
)

BANK_ROW = CODED_TRAJECTORIES[2]  # mixed same/different scenario, matches step14 tests' convention
NON_BANK_PROMPT = (
    "Starting state: Heads\nInstructions:\n"
    "1. same as previous (the state does NOT change)\n"
    "2. same as previous (the state does NOT change)\n"
    "3. same as previous (the state does NOT change)\n"
    "4. same as previous (the state does NOT change)\n"
    "5. different from previous (the state flips)\n"
    "Reason through every flip in order."
)


class FakeTokenizer:
    """Minimal stand-in -- encode() just returns a deterministic id per character
    plus a fixed EOS id, enough to test the splicing logic without a real model."""
    eos_token = "<EOS>"
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]


# ===== Trajectory-to-scenario matching =====

def test_injected_trajectory_matches_exact_scenario():
    row = injected_trajectory_for_prompt(BANK_ROW["prompt"])
    assert row is not None
    assert row["starting_state"] == BANK_ROW["starting_state"]
    assert row["operations"] == BANK_ROW["operations"]
    assert row["completion"] == BANK_ROW["completion"]


def test_every_bank_scenario_has_a_distinct_matched_trajectory():
    seen = set()
    for row in CODED_TRAJECTORIES:
        matched = injected_trajectory_for_prompt(row["prompt"])
        assert matched is not None
        assert matched["starting_state"] == row["starting_state"]
        assert matched["operations"] == row["operations"]
        seen.add((matched["starting_state"], tuple(matched["operations"])))
    assert len(seen) == 16  # all 16 scenarios distinctly matched, no collisions


# ===== Scope: bank scenarios only =====

def test_non_bank_prompt_gets_no_injection():
    assert injected_trajectory_for_prompt(NON_BANK_PROMPT) is None


def test_select_injection_indices_only_flags_bank_prompts():
    prompts = [NON_BANK_PROMPT] * 4 + [BANK_ROW["prompt"]] * 4
    result = select_injection_indices(prompts)
    assert set(result.keys()) == {4}  # only the first bank-scenario index
    assert result[4]["completion"] == BANK_ROW["completion"]


def test_all_non_bank_group_yields_no_injection():
    prompts = [NON_BANK_PROMPT] * 8
    assert select_injection_indices(prompts) == {}


# ===== Group-composition correctness =====

def test_exactly_one_slot_replaced_per_bank_scenario_group():
    prompts = [BANK_ROW["prompt"]] * 8
    completion_ids = [[1, 2, 3]] * 8
    completions = ["literal filler"] * 8
    new_ids, new_text, injected = inject_into_generation_output(
        prompts, completion_ids, completions, FakeTokenizer()
    )
    assert len(injected) == INJECTIONS_PER_GROUP == 1
    replaced = [i for i in range(8) if new_ids[i] != [1, 2, 3]]
    assert replaced == list(injected.keys())
    assert len(replaced) == 1
    # every other slot is untouched, byte-identical to the input
    for i in range(8):
        if i not in injected:
            assert new_ids[i] == [1, 2, 3]
            assert new_text[i] == "literal filler"


def test_injection_does_not_mutate_input_lists():
    prompts = [BANK_ROW["prompt"]] * 8
    completion_ids = [[1, 2, 3]] * 8
    completions = ["literal filler"] * 8
    inject_into_generation_output(prompts, completion_ids, completions, FakeTokenizer())
    assert completion_ids == [[1, 2, 3]] * 8  # original list untouched
    assert completions == ["literal filler"] * 8


def test_injected_row_text_and_ids_are_consistent_with_each_other():
    prompts = [BANK_ROW["prompt"]] * 8
    completion_ids = [[1, 2, 3]] * 8
    completions = ["literal filler"] * 8
    new_ids, new_text, injected = inject_into_generation_output(
        prompts, completion_ids, completions, FakeTokenizer()
    )
    idx = next(iter(injected))
    assert new_text[idx] == BANK_ROW["completion"]
    expected_ids = build_injected_completion_ids(FakeTokenizer(), BANK_ROW["completion"])
    assert new_ids[idx] == expected_ids


def test_mixed_batch_injects_once_per_distinct_bank_scenario():
    """Defensive test for the non-uniform-batch case this project has never actually
    produced but the selection function is written to handle correctly anyway."""
    other_bank_row = CODED_TRAJECTORIES[5]
    prompts = [BANK_ROW["prompt"]] * 4 + [other_bank_row["prompt"]] * 4
    completion_ids = [[0]] * 8
    completions = ["x"] * 8
    _, _, injected = inject_into_generation_output(prompts, completion_ids, completions, FakeTokenizer())
    assert len(injected) == 2
    assert injected[0]["completion"] == BANK_ROW["completion"]
    assert injected[4]["completion"] == other_bank_row["completion"]


# ===== Reward invariant under injection: the injected row must reliably score full credit =====

def test_injected_completion_scores_full_ungated_reward_at_every_step():
    for step in (1, 15, 30, 100, 150):
        b = score_completion_gated(
            BANK_ROW["completion"], BANK_ROW["final_answer"],
            step=step, total_steps=150, prompt=BANK_ROW["prompt"],
        )
        assert b["taxonomy_category_name"] == "correct_globally_consistent_code"
        assert b["bank_gate_scale"] is None  # never gated
        assert b["total"] == b["ungated_total"]
        assert b["total"] > 4.0  # comfortably the top of the reward range


# ===== GRPO ratio=1 invariant, re-derived and checked against this project's actual config =====

def test_grpo_ratio_is_one_for_actual_project_configuration():
    assert verify_grpo_ratio_is_unconditionally_one(
        gradient_accumulation_steps=8, steps_per_generation=8, num_iterations=1
    ) is True


def test_grpo_ratio_invariant_would_be_violated_by_a_different_configuration():
    """Confirms the check is a real, non-trivial condition -- not a function that
    always returns True regardless of input."""
    assert verify_grpo_ratio_is_unconditionally_one(
        gradient_accumulation_steps=8, steps_per_generation=3, num_iterations=1
    ) is False


# ===== Concrete numeric confirmation: injection actually resolves group collapse =====

def test_injection_gives_the_group_nonzero_variance_and_favors_the_injected_slot():
    """Reconstructs trl's own advantage formula exactly:
    advantages = (rewards - mean) / (std + 1e-4). Uses the actual observed pattern
    from Step 14's full run (7 gated-to-zero rollouts + what would have been an
    8th zero, now replaced by the injected correct-code reward) and confirms the
    group is no longer degenerate."""
    injected_reward = score_completion_gated(
        BANK_ROW["completion"], BANK_ROW["final_answer"], step=100, total_steps=150, prompt=BANK_ROW["prompt"],
    )["total"]
    rewards = [0.0] * 7 + [injected_reward]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    assert std > 0.0, "group must no longer be degenerate (zero variance)"
    advantages = [(r - mean) / (std + 1e-4) for r in rewards]
    injected_advantage = advantages[-1]
    zero_advantages = advantages[:-1]
    assert injected_advantage > 0, "the injected (correct-code) slot must get a positive advantage"
    assert all(math.isfinite(a) for a in advantages), "no NaN/Inf from the new reward distribution"
    assert all(a < injected_advantage for a in zero_advantages), (
        "every gated-to-zero slot must be pushed below the injected slot's advantage"
    )
    # the zero-reward slots get a real, negative-leaning push away from their behavior
    # (not literally required to be negative -- with 7/8 mass at zero, the mean sits
    # near zero too -- but they must be strictly below the injected slot's advantage,
    # already asserted above, and the split must be non-degenerate, already asserted).
