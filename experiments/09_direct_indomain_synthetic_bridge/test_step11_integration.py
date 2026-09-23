"""Step 11 CPU integration checks tied to persisted Stage 9 evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from soft_stops import (
    ABSOLUTE_WORD_FLOOR,
    FLOOR_CONSECUTIVE_STEPS,
    SEVERE_ACCURACY_THRESHOLD,
    SEVERE_ACCURACY_WINDOW,
    TREND_MIN_OBSERVATIONS,
)
from synthetic_bridge import (
    CODED_TRAJECTORIES,
    ExposureSchedule,
    select_synthetic_examples,
    teacher_forced_ce,
    verify_trajectory,
)


HERE = Path(__file__).resolve().parent
FULL_RUN = HERE / "aws_runs" / "step0_task_foundation_full_v1"
STEP8 = HERE / "aws_runs" / "step8_policy_token_pair_reaudit_v1" / "results.json"


def _json(path: Path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("damage", ["wrong_answer", "missing_slot"])
def test_trajectory_verifier_rejects_additional_broken_targets(damage):
    row = copy.deepcopy(CODED_TRAJECTORIES[0])
    if damage == "wrong_answer":
        row["completion"] = row["completion"].replace("<answer>Heads</answer>", "<answer>Tails</answer>")
        expected = "incorrect final answer"
    else:
        row["completion"] = row["completion"].replace("Step 3:", "Step 8:")
        expected = "missing, duplicate, or out-of-order"
    with pytest.raises(AssertionError, match=expected):
        verify_trajectory(row)


def test_bank_expansion_changes_reuse_not_exposure_quota():
    expected_totals = {"0pct": 0, "1pct": 12, "5pct": 60, "10pct": 120,
                       "10pct_annealed": 60}
    for condition, total in expected_totals.items():
        schedule = ExposureSchedule(condition)
        assert schedule.total_examples() == total
        selected = []
        for step, count in enumerate(schedule.counts(), 1):
            selected.extend(select_synthetic_examples(CODED_TRAJECTORIES, count, step))
        assert len(selected) == total
    ten_pct = ExposureSchedule("10pct")
    uses = Counter(
        row["prompt"]
        for step, count in enumerate(ten_pct.counts(), 1)
        for row in select_synthetic_examples(CODED_TRAJECTORIES, count, step)
    )
    assert Counter(uses.values()) == {7: 8, 8: 8}


def test_selection_is_deterministic_and_does_not_consume_python_rng():
    random.seed(90817)
    before = random.getstate()
    first = select_synthetic_examples(CODED_TRAJECTORIES, 3, 16)
    second = select_synthetic_examples(CODED_TRAJECTORIES, 3, 16)
    after = random.getstate()
    assert first == second
    assert before == after
    assert first == [CODED_TRAJECTORIES[15], CODED_TRAJECTORIES[0], CODED_TRAJECTORIES[1]]


class _Tokenizer:
    eos_token = "~"
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "P:" + messages[0]["content"] + "|A:"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]


class _CaptureModel:
    device = "cpu"

    def __init__(self):
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(loss=0.625)


def test_teacher_forced_ce_masks_prompt_and_padding_without_framework_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        tensor=lambda values, device: values,
    ))
    examples = [CODED_TRAJECTORIES[0], CODED_TRAJECTORIES[1]]
    tokenizer, model = _Tokenizer(), _CaptureModel()
    loss = teacher_forced_ce(model, tokenizer, examples)
    assert loss == 0.625
    labels = model.call["labels"]
    attention = model.call["attention_mask"]
    for index, row in enumerate(examples):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], False, True)
        prefix_len = len(tokenizer.encode(prefix, False))
        assert labels[index][:prefix_len] == [-100] * prefix_len
        for label, attended in zip(labels[index], attention[index]):
            if not attended:
                assert label == -100
        assert all(label != -100 for label, attended in zip(labels[index][prefix_len:],
                                                             attention[index][prefix_len:]) if attended)


def test_gpu_bfloat16_auxiliary_ce_is_finite_with_correct_gradient_scaling():
    """Step 12 persisted evidence closes the former CUDA/bf16 deferral."""
    evidence = _json(HERE / "aws_runs" / "step12_bridge_dryrun_10pct_v1" /
                     "step12_bridge_dryrun.json")
    calibration = evidence["final_report"]["beta_calibration"]
    assert calibration["ce_device"] == "cuda:0"
    assert calibration["ce_dtype"] == "torch.float32"  # stable accumulation on bf16 model path
    assert calibration["all_grpo_grads_finite"] is True
    assert calibration["all_ce_grads_finite"] is True
    calls = [row for row in evidence["final_report"]["auxiliary_ce_calls"]
             if row["synthetic_count"]]
    assert [row["optimizer_step"] for row in calls] == [2, 3, 4, 5, 7, 8]
    assert all(row["ce_device"] == "cuda:0" and row["ce_loss_dtype"] == "torch.float32"
               and row["all_finite"] for row in calls)
    assert all(row["returned_aux_term"] == pytest.approx(
        8 * row["effective_optimizer_aux_term"], rel=1e-6) for row in calls)


@pytest.mark.skipif(
    not (FULL_RUN / "milestone-12" / "adapter_model.safetensors").is_file(),
    reason="needs the local milestone-12 LoRA adapter weights, deliberately never "
           "committed to git (*.safetensors is gitignored repo-wide) -- present only "
           "where these weights were actually retrieved from AWS",
)
def test_persisted_step8_pairs_pass_gates_and_bind_to_local_milestone12_adapter():
    evidence = _json(STEP8)
    weights = FULL_RUN / "milestone-12" / "adapter_model.safetensors"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == evidence["adapter_sha256"]
    assert evidence["candidate_count"] == 30
    assert evidence["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert evidence["pairs"]["training"]["tokens"] == ["Nib", "Nomo"]
    assert evidence["pairs"]["held_out"]["tokens"] == ["Yelt", "Yark"]
    for pair in evidence["pairs"].values():
        assert pair["step0_policy_abs_gap"] == 0.0
        assert pair["both_static_eligible"] is True
        assert pair["both_pass_leakage_classifier"] is True
        assert min(pair["min_edit_distances"]) >= 3


def test_persisted_train_eval_splits_are_disjoint_at_every_evaluated_length():
    event = _json(FULL_RUN / "step0_task_foundation.json")
    audit = event["config"]["split_audit_by_length"]
    assert set(audit) == {"4", "5", "6", "7"}
    assert {length: row["overlap_count"] for length, row in audit.items()} == {
        "4": 0, "5": 0, "6": 0, "7": 0,
    }
    assert {length: (row["train"], row["eval"]) for length, row in audit.items()} == {
        "4": (21, 11), "5": (43, 21), "6": (50, 25), "7": (50, 25),
    }


def test_clamp_and_hard_soft_breakers_match_persisted_run_configuration():
    event = _json(FULL_RUN / "step0_task_foundation.json")
    config, report = event["config"], event["final_report"]
    assert config["kl_clamp_value"] == pytest.approx(0.1518003886715766)
    assert config["kl_clamp_d_max"] == pytest.approx(0.504718110048141)
    assert config["grad_breaker"] == 50.0 and config["kl_breaker"] == 5.0
    assert config["soft_stop_config"] == {
        "absolute_word_floor": ABSOLUTE_WORD_FLOOR,
        "floor_consecutive_steps": FLOOR_CONSECUTIVE_STEPS,
        "trend_min_observations": TREND_MIN_OBSERVATIONS,
        "severe_accuracy_window": SEVERE_ACCURACY_WINDOW,
        "severe_accuracy_threshold": SEVERE_ACCURACY_THRESHOLD,
    }
    assert report["hard_stop"] is None and report["soft_stop"] is None
    assert 0.15 <= report["kl_clamp_engagement_rate"] <= 0.25


def test_milestone12_is_fresh_lora_on_expected_uncomposed_base_and_matches_reaudit():
    event = _json(FULL_RUN / "step0_task_foundation.json")
    adapter = _json(FULL_RUN / "milestone-12" / "adapter_config.json")
    source = (HERE / "step0_train.py").read_text()
    assert event["purpose"].endswith("fresh_lora_no_concealment_pressure")
    assert event["config"]["fresh_lora_init_sha256"] == event["final_report"]["fresh_lora_init_sha256"]
    assert adapter["base_model_name_or_path"] == "Qwen/Qwen2.5-3B-Instruct"
    assert adapter["peft_type"] == "LORA" and adapter["init_lora_weights"] is True
    assert "source_checkpoint': None" in source
    assert "composed_with_existing_adapter': False" in source
    assert "PeftModel.from_pretrained" not in source
