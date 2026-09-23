import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from synthetic_bridge import (
    CODED_TRAJECTORIES,
    CODED_VERIFICATION,
    DEFAULT_TOTAL_STEPS,
    ExposureSchedule,
    LITERAL_CONTROL_TRAJECTORIES,
    LITERAL_CONTROL_VERIFICATION,
    ON_POLICY_EXAMPLES_PER_OPTIMIZER_STEP,
    AuxiliaryCEController,
    build_clean_length5_train_eval_split,
    verify_bank,
    verify_trajectory,
    wrap_compute_loss,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from reward_v3 import verify_reward_invariant  # noqa: E402


def test_coded_bank_is_length5_correct_and_never_lapses_literal():
    assert len(CODED_TRAJECTORIES) == 16
    reports = verify_bank(CODED_TRAJECTORIES)
    assert reports == CODED_VERIFICATION
    assert all(r["verified"] and r["n_flips"] == 5 and r["slot_count"] == 5 for r in reports)
    assert all(r["correct_final_answer"] and r["correct_state_tracking"] and r["code_consistent"]
               for r in reports)


def test_literal_control_uses_same_scenarios_and_same_verification_pipeline():
    assert len(LITERAL_CONTROL_TRAJECTORIES) == len(CODED_TRAJECTORIES) == 16
    assert [x["prompt"] for x in LITERAL_CONTROL_TRAJECTORIES] == [x["prompt"] for x in CODED_TRAJECTORIES]
    reports = verify_bank(LITERAL_CONTROL_TRAJECTORIES)
    assert reports == LITERAL_CONTROL_VERIFICATION
    assert all(r["verified"] and r["literal_control"] and r["correct_state_tracking"] for r in reports)


def test_every_sequence_has_both_starts_and_literal_control_is_exactly_matched():
    starts_by_sequence = {}
    for coded, literal in zip(CODED_TRAJECTORIES, LITERAL_CONTROL_TRAJECTORIES):
        sequence = tuple(coded["operations"])
        starts_by_sequence.setdefault(sequence, set()).add(coded["starting_state"])
        assert literal["starting_state"] == coded["starting_state"]
        assert literal["operations"] == coded["operations"]
        assert literal["prompt"] == coded["prompt"]
        assert literal["final_answer"] == coded["final_answer"]
        assert literal["completion"] == coded["completion"].replace("Nib", "Heads").replace("Nomo", "Tails")
    assert len(starts_by_sequence) == 8
    assert all(starts == {"Heads", "Tails"} for starts in starts_by_sequence.values())
    assert Counter(x["starting_state"] for x in CODED_TRAJECTORIES) == {"Heads": 8, "Tails": 8}
    assert [Counter(x["operations"][position] for x in CODED_TRAJECTORIES)
            for position in range(5)] == [{"same": 8, "different": 8}] * 5

    sequences = [tuple(x["operations"]) for x in CODED_TRAJECTORIES]
    hamming = Counter(sum(left != right for left, right in zip(a, b))
                      for a, b in combinations(sequences, 2))
    prefixes = Counter(next((i for i, (left, right) in enumerate(zip(a, b)) if left != right), 5)
                       for a, b in combinations(sequences, 2))
    assert hamming == {0: 8, 2: 48, 3: 48, 5: 16}
    assert prefixes == {0: 64, 1: 32, 2: 16, 5: 8}


def test_clean_length5_eval_is_deterministic_verified_and_bank_disjoint():
    train, evaluation = build_clean_length5_train_eval_split(seed=20260831, n_eval=21)
    repeated_train, repeated_eval = build_clean_length5_train_eval_split(seed=20260831, n_eval=21)
    assert (train, evaluation) == (repeated_train, repeated_eval)
    assert len(train) == 43 and len(evaluation) == 21
    bank = {(row["starting_state"], tuple(row["operations"])) for row in CODED_TRAJECTORIES}
    train_keys = {(row["starting_state"], tuple(row["operations"])) for row in train}
    eval_keys = {(row["starting_state"], tuple(row["operations"])) for row in evaluation}
    assert len(train_keys | eval_keys) == 64
    assert train_keys.isdisjoint(eval_keys)
    assert eval_keys.isdisjoint(bank)
    assert all(row["final_answer"] in {"Heads", "Tails"} for row in evaluation)


def test_verifier_rejects_one_wrong_coded_state_and_literal_lapse():
    wrong = dict(CODED_TRAJECTORIES[0])
    wrong["completion"] = wrong["completion"].replace("State: Nib", "State: Nomo", 1)
    with pytest.raises(AssertionError, match="state tracking mismatch"):
        verify_trajectory(wrong)
    lapse = dict(CODED_TRAJECTORIES[0])
    lapse["completion"] = lapse["completion"].replace("State: Nib", "State: Heads", 1)
    with pytest.raises(AssertionError):
        verify_trajectory(lapse)


@pytest.mark.parametrize("condition,total", [
    ("0pct", 0), ("1pct", 12), ("5pct", 60), ("10pct", 120),
    ("10pct_annealed", 60),
])
def test_exact_150_step_exposure_totals(condition, total):
    schedule = ExposureSchedule(condition)
    assert schedule.total_steps == DEFAULT_TOTAL_STEPS == 150
    assert schedule.on_policy_examples_per_step == ON_POLICY_EXAMPLES_PER_OPTIMIZER_STEP == 8
    counts = schedule.counts()
    assert sum(counts) == total
    # Whole examples only; cumulative floor error is always less than one example.
    for step in range(1, schedule.total_steps + 1):
        assert schedule.examples_at(step) in (0, 1)
        target = schedule.cumulative_target(step)
        actual = sum(counts[:step])
        assert 0 <= float(target) - actual < 1


def test_constant_condition_per_step_sequences_are_exact():
    assert ExposureSchedule("0pct").counts()[:20] == (0,) * 20
    assert ExposureSchedule("1pct").counts()[:20] == (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    assert ExposureSchedule("5pct").counts()[:10] == (0, 0, 1, 0, 1, 0, 0, 1, 0, 1)
    assert ExposureSchedule("10pct").counts()[:10] == (0, 1, 1, 1, 1, 0, 1, 1, 1, 1)


def test_annealed_schedule_is_linear_from_ten_percent_to_exact_zero():
    schedule = ExposureSchedule("10pct_annealed", total_steps=150)
    assert schedule.fraction_at(1) == pytest.approx(0.10)
    assert schedule.fraction_at(75) == pytest.approx(0.10 * (75 / 149))
    assert schedule.fraction_at(150) == 0
    fractions = [schedule.fraction_at(step) for step in range(1, 151)]
    deltas = [fractions[i] - fractions[i + 1] for i in range(149)]
    assert len(set(deltas)) == 1  # exact Fraction arithmetic: truly linear, no float drift
    assert schedule.counts()[:20] == (
        0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0)
    assert schedule.total_examples() == 60


class FakeTensor:
    def __init__(self, values): self.values = tuple(values)
    def detach(self): return self
    def clone(self): return FakeTensor(self.values)
    def equal(self, other): return self.values == other.values


class DummyController:
    def __init__(self, aux): self.aux = aux; self.calls = 0
    def auxiliary_loss(self, model, optimizer_step):
        self.calls += 1
        return self.aux


def _original_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Stand-in for GRPO: reads advantages but does not mutate them.
    loss = sum(inputs["advantages"].values) / 10
    return (loss, {"grpo": True}) if return_outputs else loss


def test_zero_weight_wrapper_is_bit_identical_to_auxiliary_entirely_absent():
    absent_inputs = {"advantages": FakeTensor([0.125, -0.5, 0.375])}
    wrapped_inputs = {"advantages": FakeTensor([0.125, -0.5, 0.375])}
    absent = _original_loss(None, None, absent_inputs)
    controller = DummyController(None)  # exact behavior of beta_sft=0
    wrapped = wrap_compute_loss(_original_loss, controller, lambda: 1)
    present_zero = wrapped(None, None, wrapped_inputs)
    assert present_zero == absent
    assert wrapped_inputs["advantages"].equal(absent_inputs["advantages"])
    assert wrapped_inputs["advantages"].values == absent_inputs["advantages"].values


def test_nonzero_aux_is_added_after_grpo_without_touching_advantages_or_outputs():
    inputs = {"advantages": FakeTensor([1.0, -1.0])}
    controller = DummyController(0.75)
    wrapped = wrap_compute_loss(_original_loss, controller, lambda: 7)
    loss, outputs = wrapped(None, None, inputs, return_outputs=True)
    assert loss == 0.75
    assert outputs == {"grpo": True}
    assert inputs["advantages"].values == (1.0, -1.0)


def test_auxiliary_controller_applies_once_per_optimizer_step_and_uses_quota_count():
    seen = []
    def fake_ce(model, tokenizer, examples):
        seen.append([row["prompt"] for row in examples])
        return 2.0
    controller = AuxiliaryCEController(
        schedule=ExposureSchedule("10pct", total_steps=10),
        bank=CODED_TRAJECTORIES, beta_sft=0.25, tokenizer=None,
        gradient_accumulation_steps=8, ce_function=fake_ce,
    )
    assert controller.auxiliary_loss(None, 1) is None  # quota is zero at step 1
    assert controller.auxiliary_loss(None, 2) == 4.0  # .25 * 8 accumulation * CE=2
    assert controller.auxiliary_loss(None, 2) is None  # never duplicated across microbatches
    assert len(seen) == 1 and len(seen[0]) == 1


def test_full_reward_invariant_is_unchanged_by_separate_auxiliary_module():
    margins = verify_reward_invariant()
    assert margins["margin_correct_over_wrong"] > 0
    assert margins["margin_wrong_over_malformed"] > 0
    # Full bridge-phase reward defaults retain the already-audited 0.7/0.7 margins;
    # Step 10 imports but never edits or parameterizes reward_v3.py.
    assert math.isclose(margins["margin_correct_over_wrong"], 0.7, abs_tol=1e-9)
    assert math.isclose(margins["margin_wrong_over_malformed"], 0.7, abs_tol=1e-9)
