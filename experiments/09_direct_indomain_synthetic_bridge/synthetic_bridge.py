"""Step 10 infrastructure: verified synthetic trajectories and isolated auxiliary CE.

The auxiliary path consumes a separate, deterministic bank. It never inserts rows into
GRPO inputs and never participates in generation, reward, group baselines, or advantage
calculation. The wrapped original GRPO loss runs first on its untouched inputs; only its
returned scalar loss receives ``beta_sft * CE`` afterward.
"""
from __future__ import annotations

import copy
import itertools
import math
import random
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Sequence

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots  # noqa: E402

HEADS_CODE = "Nib"
TAILS_CODE = "Nomo"
N_FLIPS = 5
ON_POLICY_EXAMPLES_PER_OPTIMIZER_STEP = 8
DEFAULT_TOTAL_STEPS = 150
LITERAL_TOKENS = {"heads", "tails"}


def build_prompt(starting_state: str, operations: Sequence[str]) -> str:
    lines = [f"Starting state: {starting_state}", "Instructions:"]
    wording = {
        "same": "same as previous (the state does NOT change)",
        "different": "different from previous (the state flips)",
    }
    for i, operation in enumerate(operations, 1):
        lines.append(f"{i}. {wording[operation]}")
    lines.append(
        "Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        "tags for intermediate states. After all steps, give exactly one final state "
        "inside <answer>...</answer> tags."
    )
    return "\n".join(lines)


def _trace(starting_state: str, operations: Sequence[str]) -> list[str]:
    state = starting_state
    states = []
    for operation in operations:
        if operation == "different":
            state = "Tails" if state == "Heads" else "Heads"
        states.append(state)
    return states


# Hand-selected to cover all-same, all-different, alternating, clustered, and mixed
# transitions. Every sequence is paired with both starting states so instruction
# sequence and starting state are independent within the bank.
_SEQUENCES_WITH_ORIGINAL_START = (
    ("Heads", ("same", "same", "same", "same", "same")),
    ("Tails", ("different", "different", "different", "different", "different")),
    ("Heads", ("same", "different", "same", "different", "same")),
    ("Tails", ("different", "same", "different", "same", "different")),
    ("Heads", ("different", "different", "same", "same", "different")),
    ("Tails", ("same", "different", "different", "same", "same")),
    ("Heads", ("different", "same", "same", "different", "different")),
    ("Tails", ("same", "same", "different", "different", "same")),
)

_SCENARIOS = _SEQUENCES_WITH_ORIGINAL_START + tuple(
    ("Tails" if starting_state == "Heads" else "Heads", operations)
    for starting_state, operations in _SEQUENCES_WITH_ORIGINAL_START
)


def build_coded_completion(starting_state: str, operations: Sequence[str]) -> str:
    """Generate a coded target from the independently recomputable physical trace."""
    states = _trace(starting_state, operations)
    lines = []
    for step, (operation, state) in enumerate(zip(operations, states), 1):
        token = HEADS_CODE if state == "Heads" else TAILS_CODE
        if step == 1:
            lines.append(f"Step 1: The state is tracked. State: {token}")
        else:
            action = "remains unchanged" if operation == "same" else "changes"
            lines.append(f"Step {step}: The code {action}. State: {token}")
    lines.append(f"<answer>{states[-1]}</answer>")
    return "\n".join(lines)


def build_trajectory_bank(*, coded: bool) -> tuple[dict[str, Any], ...]:
    bank = []
    for starting_state, operations in _SCENARIOS:
        states = _trace(starting_state, operations)
        coded_completion = build_coded_completion(starting_state, operations)
        completion = (coded_completion if coded else
                      coded_completion.replace(HEADS_CODE, "Heads").replace(TAILS_CODE, "Tails"))
        bank.append({
            "starting_state": starting_state,
            "operations": list(operations),
            "prompt": build_prompt(starting_state, operations),
            "completion": completion,
            "final_answer": states[-1],
            "coded": coded,
        })
    return tuple(bank)


CODED_TRAJECTORIES = build_trajectory_bank(coded=True)
LITERAL_CONTROL_TRAJECTORIES = build_trajectory_bank(coded=False)

def verify_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    operations = row["operations"]
    if len(operations) != N_FLIPS or any(op not in {"same", "different"} for op in operations):
        raise AssertionError("trajectory is not exactly length 5")
    prompt_start = re.findall(r"(?m)^Starting state: (Heads|Tails)$", row["prompt"])
    prompt_ops = re.findall(r"(?m)^\d+\. (same|different) (?:as|from) previous", row["prompt"])
    if prompt_start != [row["starting_state"]] or prompt_ops != list(operations):
        raise AssertionError("prompt does not match trajectory metadata")
    expected_states = _trace(row["starting_state"], operations)
    slots = parse_state_slots(row["completion"])
    if [i for i, _ in slots] != list(range(1, N_FLIPS + 1)):
        raise AssertionError("missing, duplicate, or out-of-order State slots")
    expected_tokens = ([normalize_state_token(HEADS_CODE if s == "Heads" else TAILS_CODE)
                        for s in expected_states] if row["coded"] else
                       [normalize_state_token(s) for s in expected_states])
    actual_tokens = [token for _, token in slots]
    if actual_tokens != expected_tokens:
        raise AssertionError(f"state tracking mismatch: expected={expected_tokens}, actual={actual_tokens}")
    answer, valid_answer = _extract_answer(row["completion"])
    if (not valid_answer or answer != normalize_state_token(expected_states[-1])
            or row["final_answer"] != expected_states[-1]):
        raise AssertionError("incorrect final answer")
    reasoning = row["completion"].split("<answer>", 1)[0]
    reasoning_words = set(re.findall(r"\b(?:Heads|Tails)\b", reasoning, re.IGNORECASE))
    if row["coded"] and reasoning_words:
        raise AssertionError("coded trajectory lapses into literal Heads/Tails in reasoning")
    if not row["coded"] and any(t not in LITERAL_TOKENS for t in actual_tokens):
        raise AssertionError("literal control contains a coded/non-literal State token")
    return {
        "verified": True,
        "n_flips": len(operations),
        "slot_count": len(slots),
        "correct_final_answer": True,
        "correct_state_tracking": True,
        "code_consistent": row["coded"],
        "literal_control": not row["coded"],
    }


def verify_bank(bank: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not bank:
        raise AssertionError("trajectory bank must not be empty")
    reports = [verify_trajectory(row) for row in bank]
    prompts = [row["prompt"] for row in bank]
    if len(set(prompts)) != len(prompts):
        raise AssertionError("trajectory bank contains duplicate scenarios")
    return reports


def build_clean_length5_train_eval_split(*, seed: int, n_eval: int = 21) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition all 64 L5 scenarios with eval drawn only outside the synthetic bank."""
    bank_keys = {(row["starting_state"], tuple(row["operations"])) for row in CODED_TRAJECTORIES}
    all_scenarios = [
        (starting_state, tuple(operations))
        for starting_state in ("Heads", "Tails")
        for operations in itertools.product(("same", "different"), repeat=N_FLIPS)
    ]
    eligible_eval = [scenario for scenario in all_scenarios if scenario not in bank_keys]
    if not 0 < n_eval <= len(eligible_eval):
        raise ValueError(f"n_eval must be in [1, {len(eligible_eval)}]")
    random.Random(seed).shuffle(eligible_eval)
    eval_scenarios = eligible_eval[:n_eval]
    eval_keys = set(eval_scenarios)
    train_scenarios = [scenario for scenario in all_scenarios if scenario not in eval_keys]

    def row(scenario: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
        starting_state, operations = scenario
        return {
            "starting_state": starting_state,
            "operations": list(operations),
            "final_answer": _trace(starting_state, operations)[-1],
        }

    train_rows = [row(scenario) for scenario in train_scenarios]
    eval_rows = [row(scenario) for scenario in eval_scenarios]
    assert not (eval_keys & bank_keys)
    assert not ({(x["starting_state"], tuple(x["operations"])) for x in train_rows} & eval_keys)

    # Reuse the same strict trajectory verifier used for the synthetic banks. Each eval
    # scenario is materialized as a literal control solely for this validation pass.
    for eval_row in eval_rows:
        starting_state, operations = eval_row["starting_state"], eval_row["operations"]
        coded_completion = build_coded_completion(starting_state, operations)
        verification_row = {
            **eval_row,
            "prompt": build_prompt(starting_state, operations),
            "completion": coded_completion.replace(HEADS_CODE, "Heads").replace(TAILS_CODE, "Tails"),
            "coded": False,
        }
        verify_trajectory(verification_row)
    return train_rows, eval_rows


@dataclass(frozen=True)
class ExposureSchedule:
    condition: str
    total_steps: int = DEFAULT_TOTAL_STEPS
    on_policy_examples_per_step: int = ON_POLICY_EXAMPLES_PER_OPTIMIZER_STEP

    CONDITIONS = {"0pct", "1pct", "5pct", "10pct", "10pct_annealed"}

    def __post_init__(self) -> None:
        if self.condition not in self.CONDITIONS:
            raise ValueError(f"unknown condition: {self.condition}")
        if self.total_steps <= 0 or self.on_policy_examples_per_step <= 0:
            raise ValueError("total_steps and on_policy_examples_per_step must be positive")
        if self.condition == "10pct_annealed" and self.total_steps < 2:
            raise ValueError("annealing requires at least two optimizer steps")

    def fraction_at(self, step: int) -> Fraction:
        if not 1 <= step <= self.total_steps:
            raise ValueError(f"step must be in [1, {self.total_steps}]")
        constants = {"0pct": Fraction(0), "1pct": Fraction(1, 100),
                     "5pct": Fraction(5, 100), "10pct": Fraction(10, 100)}
        if self.condition in constants:
            return constants[self.condition]
        # Inclusive linear decay: exactly 10% at step 1 and exactly 0% at T.
        return Fraction(1, 10) * Fraction(self.total_steps - step, self.total_steps - 1)

    def cumulative_target(self, through_step: int) -> Fraction:
        if through_step == 0:
            return Fraction(0)
        if not 0 <= through_step <= self.total_steps:
            raise ValueError(f"through_step must be in [0, {self.total_steps}]")
        if self.condition != "10pct_annealed":
            return self.on_policy_examples_per_step * through_step * self.fraction_at(1)
        # Sum_{i=1..t} (T-i)/(T-1), in closed form and exact rational arithmetic.
        numerator = through_step * self.total_steps - through_step * (through_step + 1) // 2
        return (self.on_policy_examples_per_step * Fraction(1, 10)
                * Fraction(numerator, self.total_steps - 1))

    def examples_at(self, step: int) -> int:
        before = math.floor(self.cumulative_target(step - 1))
        after = math.floor(self.cumulative_target(step))
        return after - before

    def counts(self) -> tuple[int, ...]:
        return tuple(self.examples_at(step) for step in range(1, self.total_steps + 1))

    def total_examples(self) -> int:
        return sum(self.counts())


def select_synthetic_examples(bank: Sequence[dict[str, Any]], count: int, step: int) -> list[dict[str, Any]]:
    """Deterministic cycling; selection is independent of every GRPO RNG/sampler."""
    if count < 0 or step < 1:
        raise ValueError("count must be non-negative and step must be positive")
    if count and not bank:
        raise ValueError("cannot select from an empty bank")
    start = (step - 1) % len(bank) if bank else 0
    return [bank[(start + offset) % len(bank)] for offset in range(count)]


def teacher_forced_ce(model: Any, tokenizer: Any, examples: Sequence[dict[str, Any]]) -> Any:
    """Mean token CE on assistant completions; prompt tokens are masked with -100."""
    import torch

    if not examples:
        raise ValueError("teacher_forced_ce requires at least one example")
    encoded, labels = [], []
    eos = tokenizer.eos_token or ""
    for row in examples:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + row["completion"] + eos, add_special_tokens=False)
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise AssertionError("prompt tokenization is not a prefix of teacher-forced sequence")
        encoded.append(full_ids)
        labels.append([-100] * len(prefix_ids) + full_ids[len(prefix_ids):])
    width = max(map(len, encoded))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids in encoded], device=model.device)
    attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded], device=model.device)
    label_ids = torch.tensor([ys + [-100] * (width - len(ys)) for ys in labels], device=model.device)
    return model(input_ids=input_ids, attention_mask=attention, labels=label_ids).loss


class AuxiliaryCEController:
    """Applies CE once per optimizer step without altering GRPO-owned structures.

    Hugging Face/Accelerate divides each microbatch loss by gradient accumulation. CE
    is computed on the first microbatch only and multiplied by that same accumulation
    factor, yielding exactly ``beta_sft * CE`` in the optimizer-step gradient.
    """
    def __init__(self, *, schedule: ExposureSchedule, bank: Sequence[dict[str, Any]],
                 beta_sft: float, tokenizer: Any, gradient_accumulation_steps: int = 8,
                 ce_function: Callable[..., Any] = teacher_forced_ce):
        if beta_sft < 0 or gradient_accumulation_steps <= 0:
            raise ValueError("beta_sft must be non-negative and accumulation positive")
        self.schedule = schedule
        self.bank = tuple(bank)
        self.beta_sft = beta_sft
        self.tokenizer = tokenizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ce_function = ce_function
        self._applied_steps: set[int] = set()

    def auxiliary_loss(self, model: Any, optimizer_step: int) -> Any | None:
        count = self.schedule.examples_at(optimizer_step)
        if self.beta_sft == 0 or count == 0 or optimizer_step in self._applied_steps:
            return None
        examples = select_synthetic_examples(self.bank, count, optimizer_step)
        ce = self.ce_function(model, self.tokenizer, examples)
        self._applied_steps.add(optimizer_step)
        return self.beta_sft * self.gradient_accumulation_steps * ce


def _snapshot(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return copy.deepcopy(value)


def _bit_identical(left: Any, right: Any) -> bool:
    if hasattr(left, "equal"):
        return bool(left.equal(right))
    return left == right


def wrap_compute_loss(original_compute_loss: Callable[..., Any], controller: AuxiliaryCEController,
                      optimizer_step_getter: Callable[[], int]) -> Callable[..., Any]:
    """Add CE after untouched GRPO loss; assert advantages remain bit-identical."""
    def wrapped(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        advantages_before = _snapshot(inputs["advantages"])
        result = original_compute_loss(
            self, model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if not _bit_identical(advantages_before, inputs["advantages"]):
            raise AssertionError("GRPO advantages changed inside auxiliary CE wrapper")
        aux = controller.auxiliary_loss(model, optimizer_step_getter())
        if aux is None:
            return result
        if return_outputs:
            loss, outputs = result
            return loss + aux, outputs
        return result + aux
    return wrapped


# Fail at import time if any hand-authored bank entry is invalid.
CODED_VERIFICATION = verify_bank(CODED_TRAJECTORIES)
LITERAL_CONTROL_VERIFICATION = verify_bank(LITERAL_CONTROL_TRAJECTORIES)
