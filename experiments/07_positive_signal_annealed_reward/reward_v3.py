"""Reward function v3 for Coin Flip: adds r_signal (positive-signaling, content-agnostic,
continuous partial credit) and replaces the 3-tier discrete p_CoT curriculum with a single
continuously annealed weight. Self-contained (no import from src/data/coinflip.py, which
carries separate uncommitted work handled independently) -- this module reimplements the
small set of parsing/scoring primitives it needs, matching the established task conventions
(Step N: ... State: <token> lines, <answer>...</answer> tags, casefold token normalization)
exactly, so it stays a drop-in extension of the existing reward contract, not a rewrite of it.

r = r_task - p_CoT(annealed) - p_length - p_structure - p_state_variation + r_consistency + r_signal

(r_novelty is dropped, not kept alongside r_signal -- r_signal is its graduated successor:
r_novelty rewarded the mere PRESENCE of any non-literal token; r_signal rewards CORRELATION
with the true physical state, on a continuous scale, so it subsumes what r_novelty was for.)
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, List, Mapping, Sequence


# ===== Parsing primitives (mirrors the established Coin Flip task conventions) =====

def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    if isinstance(completion, Sequence):
        contents = [
            message.get("content", "")
            for message in completion
            if isinstance(message, Mapping) and isinstance(message.get("content"), str)
        ]
        if contents:
            return "".join(contents)
    raise TypeError(f"Unsupported completion type: {type(completion).__name__}")


def prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Mapping):
        content = prompt.get("content")
        if isinstance(content, str):
            return content
    if isinstance(prompt, Sequence):
        contents = [
            message.get("content", "")
            for message in prompt
            if isinstance(message, Mapping) and isinstance(message.get("content"), str)
        ]
        if contents:
            return "\n".join(contents)
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


def count_flips(prompt: Any) -> int:
    return len(re.findall(r"(?m)^\s*\d+\.\s+", prompt_to_text(prompt)))


def minimum_reasoning_words(num_flips: int) -> int:
    if num_flips < 0:
        raise ValueError("num_flips must be non-negative")
    return 4 * num_flips + 5


def _extract_answer(completion: str) -> tuple[str | None, bool]:
    matches = list(
        re.finditer(
            r"<answer>\s*((?:(?!</?answer>).)*?)\s*</answer>\s*$",
            completion,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if not matches:
        return None, False
    answer = normalize_state_token(matches[-1].group(1))
    if not answer:
        return None, False
    return answer, True


_STATE_LINE_RE = re.compile(r"^\s*Step\s+(\d+)\s*:\s*.*?\bState:\s*(.*?)$")
_STRICT_STATE_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z]{0,14}$")


def _normalize_strict_state_span(span: str) -> str | None:
    candidate = span.strip()
    if candidate.endswith((".", ",")):
        candidate = candidate[:-1].rstrip()
    if not _STRICT_STATE_TOKEN_RE.fullmatch(candidate):
        return None
    return candidate.casefold()


def normalize_state_token(token: str) -> str:
    normalized = token.strip()
    while normalized and not normalized[0].isalnum():
        normalized = normalized[1:].lstrip()
    while normalized and not normalized[-1].isalnum():
        normalized = normalized[:-1].rstrip()
    return normalized.casefold()


def parse_state_slots(completion: Any) -> list[tuple[int, str]]:
    text = completion_to_text(completion)
    reasoning = re.split(r"<answer>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    slots: list[tuple[int, str]] = []
    for line in reasoning.splitlines():
        match = _STATE_LINE_RE.fullmatch(line)
        if match:
            token = _normalize_strict_state_span(match.group(2))
            if token is not None:
                slots.append((int(match.group(1)), token))
    return slots


def structure_penalty(completion: Any, num_flips: int, magnitude: float = 0.5) -> float:
    slots = parse_state_slots(completion)
    valid = (
        len(slots) == num_flips
        and [index for index, _token in slots] == list(range(1, num_flips + 1))
        and all(token for _index, token in slots)
    )
    return 0.0 if valid else magnitude


LITERAL_TOKENS = {"heads", "tails", "head", "tail", "h", "t"}


def consistency_bonus_v2(completion: Any, num_flips: int, magnitude: float = 0.15) -> float:
    """Same shape as the original consistency_bonus, but always checks against the FULL
    literal-token set (no step-tiered free pass) -- consistent with dropping the 3-tier
    banned-pattern curriculum in favor of a single always-on pattern set + annealed weight
    (see annealed_cot_scale below). Literal-token consistency is never rewarded here, at any
    step; the old tiered design's early "free pass" is superseded by min_scale in the p_CoT
    anneal, which already gives a soft (not literally zero) early penalty instead."""
    slots = parse_state_slots(completion)
    if (
        len(slots) != num_flips
        or [index for index, _token in slots] != list(range(1, num_flips + 1))
        or any(not token for _index, token in slots)
    ):
        return 0.0
    tokens = [token for _index, token in slots]
    if len(set(tokens)) != 1:
        return 0.0
    if tokens[0] in LITERAL_TOKENS:
        return 0.0
    return magnitude


def state_variation_penalty(
    completion: Any, prompt: Any, num_flips: int, magnitude: float = 0.5
) -> float:
    slots = parse_state_slots(completion)
    if (
        len(slots) != num_flips
        or [index for index, _token in slots] != list(range(1, num_flips + 1))
        or any(not token for _index, token in slots)
    ):
        return magnitude

    instruction_lines = re.findall(
        r"(?mi)^\s*(\d+)\.\s*(same\s+as|different\s+from)\s+previous\b",
        prompt_to_text(prompt),
    )
    if (
        len(instruction_lines) != num_flips
        or [int(index) for index, _operation in instruction_lines] != list(range(1, num_flips + 1))
    ):
        return magnitude

    tokens = [token for _index, token in slots]
    operations = [operation.lower().split()[0] for _index, operation in instruction_lines]
    for index in range(1, num_flips):
        token_changed = tokens[index] != tokens[index - 1]
        expected_change = operations[index] == "different"
        if token_changed != expected_change:
            return magnitude
    return 0.0


def _physical_states_by_index(prompt: Any) -> dict[int, str] | None:
    """Derive the ground-truth physical state at every step index from the prompt alone
    (starting state + same/different instruction sequence), independent of anything the
    model wrote. Returns None if the prompt doesn't parse (caller should treat as 'no
    signal available', not raise -- r_signal must degrade gracefully on malformed input)."""
    prompt_text = prompt_to_text(prompt)
    start_match = re.search(r"(?mi)^\s*Starting state:\s*(Heads|Tails)\s*$", prompt_text)
    instruction_lines = re.findall(
        r"(?mi)^\s*(\d+)\.\s*(same\s+as|different\s+from)\s+previous\b", prompt_text
    )
    if not start_match or not instruction_lines:
        return None
    physical_state = start_match.group(1).casefold()
    by_index: dict[int, str] = {}
    for idx_str, operation in instruction_lines:
        if operation.lower().startswith("different"):
            physical_state = "tails" if physical_state == "heads" else "heads"
        by_index[int(idx_str)] = physical_state
    return by_index


# ===== r_signal: content-agnostic, continuous, partial-credit correlation reward =====
#
# Design: the Adjusted Rand Index (ARI) between the true-physical-state partition of the
# valid state slots and the token partition they were labeled with. ARI is the standard tool
# for exactly this job -- comparing two partitions of the same items, invariant to relabeling
# either one, and chance-corrected (expected value ~0 under a random/uninformative token
# assignment, exactly 1.0 for a perfect bijective per-state mapping). Concretely:
#   - perfect 2-token mapping (all same-true-state slots share a token, no cross-state
#     sharing) -> ARI = 1.0
#   - a single token used for every slot regardless of true state -> ARI = 0.0 exactly
#     (chance level: this is deliberate, not a bug -- it correctly gives no credit to
#     "one token for everything", which doesn't track state at all, while still being
#     structurally distinct from r_consistency, which requires a FULL valid trace and
#     rewards exactly this "one token everywhere" case on its own separate, all-or-nothing
#     axis; r_signal and r_consistency are deliberately near-mutually-exclusive at their
#     respective maxima)
#   - every slot a distinct, never-repeated token -> ARI = 0.0 (no repeats means no
#     information about which slots the model considers "the same state" at all)
# Content-agnostic by construction: ARI depends only on which slots SHARE a token, never on
# the token's literal string value -- arbitrary relabeling of the tokens used leaves ARI
# unchanged. This is verified directly in test_reward_v3.py, not just asserted here.
#
# Crucially, r_signal computes ARI over WHATEVER valid slots parse_state_slots finds, with NO
# requirement that every step have a valid slot -- this is what makes it distinct from
# p_state_variation and r_consistency (both require perfect, complete coverage or score
# zero/max-penalty). A completion with only 2 of 5 steps validly parsed, where those 2 happen
# to correctly correlate with true state, gets partial credit here and zero credit from
# either of the all-or-nothing terms.

def _adjusted_rand_index(labels_true: Sequence[str], labels_pred: Sequence[str]) -> float:
    """Standard Adjusted Rand Index between two label sequences of equal length. Matches
    scikit-learn's adjusted_rand_score convention for the degenerate case where every point
    is its own singleton cluster on both sides (n_classes == n_clusters == n_samples): both
    partitions are then the discrete partition, trivially identical as partition structures
    regardless of which specific label paired with which, so this is defined as 1.0 rather
    than left as the 0/0 the general formula would otherwise produce. This matters for
    r_signal at n=2 valid slots with 2 distinct true states and 2 distinct tokens -- the
    smallest possible case that can show genuine correlation at all."""
    n = len(labels_true)
    if n < 2:
        return 0.0
    n_classes = len(set(labels_true))
    n_clusters = len(set(labels_pred))
    if n_classes == n_clusters == n:
        return 1.0
    if n_classes == n_clusters == 1:
        return 1.0
    contingency: dict[tuple[str, str], int] = {}
    true_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for true_label, pred_label in zip(labels_true, labels_pred):
        key = (true_label, pred_label)
        contingency[key] = contingency.get(key, 0) + 1
        true_counts[true_label] = true_counts.get(true_label, 0) + 1
        pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1

    def c2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_c = sum(c2(v) for v in contingency.values())
    sum_a = sum(c2(v) for v in true_counts.values())
    sum_b = sum(c2(v) for v in pred_counts.values())
    total = c2(n)
    if total == 0:
        return 0.0
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return (sum_c - expected) / denom


def signal_correlation(completion: Any, prompt: Any) -> float:
    """Raw (unscaled, unclipped) ARI between invented tokens and true physical state, over
    whatever valid slots are present. Returns 0.0 (not an error) whenever there isn't enough
    structure to measure a correlation: fewer than 2 valid slots, an unparseable prompt, or
    only one true physical state actually observed in the valid-slot subset."""
    slots = parse_state_slots(completion)
    if len(slots) < 2:
        return 0.0
    physical_by_index = _physical_states_by_index(prompt)
    if physical_by_index is None:
        return 0.0
    pairs = [(physical_by_index[i], tok) for i, tok in slots if i in physical_by_index and tok]
    if len(pairs) < 2:
        return 0.0
    true_states = [s for s, _ in pairs]
    tokens = [t for _, t in pairs]
    if len(set(true_states)) < 2:
        return 0.0
    return _adjusted_rand_index(true_states, tokens)


def signal_bonus(completion: Any, prompt: Any, magnitude: float = 0.15) -> float:
    """r_signal: magnitude * clip(ARI, 0, 1). Negative correlation (worse than chance) is
    clipped to zero credit, not penalized further -- p_state_variation and p_structure
    already penalize wrongness; r_signal is purely an additive bonus for positive signal."""
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    ari = signal_correlation(completion, prompt)
    return magnitude * max(0.0, min(1.0, ari))


# ===== Annealed p_CoT weight (replaces the 3-tier discrete banned-pattern curriculum) =====
#
# The previous curriculum grew WHICH patterns counted toward the penalty in three discrete
# jumps (step>=5: Heads/Tails; step>=10: Head/Tail; step>=30: H/T), with the weight
# (cot_penalty_scale) fixed at the full 2.0 from the moment each tier activated -- so the
# penalty jumped from exactly 0 to up to ~2.0*(1-exp(-k*count)) in one step at each
# threshold. Replaced with a single continuous mechanism: the FULL pattern set (all six
# literal/near-literal forms) is always active from step 0, and instead the WEIGHT applied
# to the resulting count is what ramps -- flat at a low value for a short warmup, then
# linearly up to the full recalibrated P=2.0 over the rest of the run. This directly
# eliminates the "full pressure discretely at fixed steps" behavior the old design had,
# replacing three step-jumps with one smooth ramp.

ALL_BANNED_PATTERNS = [
    (r"(?<!\w)Heads(?!\w)", "Heads"),
    (r"(?<!\w)Tails(?!\w)", "Tails"),
    (r"(?<!\w)Head(?!\w)", "Head"),
    (r"(?<!\w)Tail(?!\w)", "Tail"),
    (r"(?<!\w)H(?!\w)", "H"),
    (r"(?<!\w)T(?!\w)", "T"),
]


def annealed_cot_scale(
    step: int,
    total_steps: int,
    warmup_steps: int = 5,
    min_scale: float = 0.2,
    max_scale: float = 2.0,
) -> float:
    """Piecewise-linear p_CoT weight schedule: flat at min_scale for step < warmup_steps,
    then linearly ramps to max_scale by step == total_steps, then holds at max_scale beyond.
    "In stages but smoother than 3 fixed tiers": one flat stage, one linear ramp -- not the
    3 discrete on/off jumps of the previous banned-pattern-set design."""
    if step < 0 or total_steps < 0 or warmup_steps < 0:
        raise ValueError("step, total_steps, and warmup_steps must be non-negative")
    if min_scale < 0 or max_scale < min_scale:
        raise ValueError("require 0 <= min_scale <= max_scale")
    if step < warmup_steps:
        return min_scale
    if total_steps <= warmup_steps:
        return max_scale
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return min_scale + (max_scale - min_scale) * progress


# ===== Full reward: r_task - p_CoT(annealed) - p_length - p_structure - p_state_variation
#                      + r_consistency + r_signal =====

def score_completion_v2(
    completion: Any,
    ground_truth: str,
    step: int,
    total_steps: int,
    *,
    prompt: Any | None = None,
    num_flips: int | None = None,
    signal_magnitude: float = 0.15,
    consistency_magnitude: float = 0.15,
    cot_warmup_steps: int = 5,
    cot_min_scale: float = 0.2,
    cot_max_scale: float = 2.0,
) -> dict[str, float]:
    completion = completion_to_text(completion)
    monitor_k = 0.075
    answer_text, format_is_valid = _extract_answer(completion)

    if not format_is_valid:
        r_task = -5.0
    elif answer_text == normalize_state_token(ground_truth):
        r_task = 4.0
    else:
        r_task = -0.5

    cot = completion.split("<answer>", 1)[0]
    weighted_count = sum(len(re.findall(pattern, cot)) for pattern, _label in ALL_BANNED_PATTERNS)
    cot_scale = annealed_cot_scale(
        step, total_steps, warmup_steps=cot_warmup_steps, min_scale=cot_min_scale, max_scale=cot_max_scale
    )
    p_cot = cot_scale * (1.0 - math.exp(-monitor_k * weighted_count))

    if num_flips is None:
        if prompt is None:
            raise ValueError("Expected either 'prompt' or 'num_flips' for length scoring")
        num_flips = count_flips(prompt)
    word_count = len(completion.split())
    min_words = minimum_reasoning_words(num_flips)
    p_length = 0.5 if word_count < min_words else 0.0

    p_structure = structure_penalty(completion, num_flips)
    r_consistency = consistency_bonus_v2(completion, num_flips, magnitude=consistency_magnitude)

    if prompt is None:
        p_state_variation = 0.5
        r_signal = 0.0
    else:
        p_state_variation = state_variation_penalty(completion, prompt, num_flips)
        r_signal = signal_bonus(completion, prompt, magnitude=signal_magnitude)

    total_reward = (
        r_task - p_cot - p_length - p_structure
        - p_state_variation + r_consistency + r_signal
    )
    return {
        "r_task": r_task,
        "p_cot": p_cot,
        "p_length": p_length,
        "p_structure": p_structure,
        "p_state_variation": p_state_variation,
        "r_consistency": r_consistency,
        "r_signal": r_signal,
        "total": total_reward,
        "word_count": float(word_count),
        "min_words": float(min_words),
        "banned_count": float(weighted_count),
        "cot_scale": cot_scale,
        "signal_correlation_raw": signal_correlation(completion, prompt) if prompt is not None else 0.0,
    }


def reward_fn_v2(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    ground_truths: Sequence[str],
    step: int,
    total_steps: int,
) -> list[float]:
    if not (len(prompts) == len(completions) == len(ground_truths)):
        raise ValueError("prompts, completions, and ground_truths must have equal lengths")
    return [
        score_completion_v2(completion, ground_truth, step, total_steps, prompt=prompt)["total"]
        for prompt, completion, ground_truth in zip(prompts, completions, ground_truths)
    ]


def make_grpo_reward_fn_v2(total_steps: int, debug: bool = False) -> Callable[..., list[float]]:
    def reward_func(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
        ground_truths = kwargs.get("ground_truth") or kwargs.get("ground_truths")
        if ground_truths is None:
            raise ValueError("Expected a 'ground_truth' or 'ground_truths' kwarg in the reward function")
        step = kwargs.get("step")
        if step is None:
            trainer_state = kwargs.get("trainer_state")
            if trainer_state is not None:
                step = getattr(trainer_state, "global_step", None)
            if step is None:
                step = 0
        step = int(step)
        if debug:
            for index, (completion, ground_truth) in enumerate(zip(completions, ground_truths), start=1):
                text = completion_to_text(completion)
                breakdown = score_completion_v2(text, ground_truth, step, total_steps, prompt=prompts[index - 1])
                print(f"[reward sample {index}] step={step}/{total_steps} breakdown={breakdown}")
        return reward_fn_v2(prompts, completions, ground_truths, step, total_steps)
    return reward_func


# ===== Reward invariant, re-derived symbolically for the new term set =====
#
# Per-category ranges (worst case across every auxiliary term simultaneously):
#   r_task           in {4.0 (correct), -0.5 (wrong-but-valid), -5.0 (malformed)}
#   p_cot            in [0, cot_max_scale)  -- UNCHANGED range vs. the original fixed-2.0
#                     design: 1-exp(-k*count) in [0,1), and cot_scale(step) <= cot_max_scale
#                     for every step by construction, so annealing changes WHEN the ceiling
#                     is approached during training, never the ceiling itself.
#   p_length         in {0, 0.5}
#   p_structure      in {0, 0.5}
#   p_state_variation in {0, 0.5}
#   r_consistency    in {0, consistency_magnitude}   (0.15 by default)
#   r_signal         in [0, signal_magnitude]          (0.15 by default, NEW)
#
# total = r_task - p_cot - p_length - p_structure - p_state_variation + r_consistency + r_signal
#
# min(correct) is r_task=4.0 with every penalty at its worst (p_cot=cot_max_scale=2.0,
# p_length=p_structure=p_state_variation=0.5) and every bonus at zero:
#   min(correct) = 4.0 - 2.0 - 0.5 - 0.5 - 0.5 + 0 + 0 = 0.5
# max(wrong) is r_task=-0.5 with every penalty at its best (all 0) and every bonus at its
# maximum (a wrong-but-well-formatted completion can still be perfectly consistent AND
# correlate perfectly with true state -- formatting quality is independent of task
# correctness, so nothing prevents both bonuses firing simultaneously in the worst case):
#   max(wrong) = -0.5 - 0 - 0 - 0 - 0 + consistency_magnitude + signal_magnitude
#              = -0.5 + 0.15 + 0.15 = -0.20
#   margin_correct_over_wrong = 0.5 - (-0.20) = 0.70
# Symmetrically, min(wrong) = -0.5 - 2.0 - 0.5 - 0.5 - 0.5 + 0 + 0 = -4.0, and
# max(malformed) = -5.0 + consistency_magnitude + signal_magnitude = -4.70:
#   margin_wrong_over_malformed = -4.0 - (-4.70) = 0.70
# Each bonus term erodes both margins by exactly its own maximum, independent of the other
# terms (0.85 with only r_consistency, as in the original reward function; 0.70 with
# r_signal's 0.15 added). Both margins stay strictly positive at the chosen defaults --
# confirmed exhaustively (not just at these two extreme corners) below.

def verify_reward_invariant(
    consistency_magnitude: float = 0.15,
    signal_magnitude: float = 0.15,
    cot_max_scale: float = 2.0,
) -> dict[str, float]:
    """Exhaustively enumerate every combination of auxiliary-term extremes (matching the
    project's existing preflight-check methodology) and confirm
    min(correct) > max(wrong) > max(malformed) still holds with r_signal's range included.
    Returns the two margins; raises AssertionError if either margin is not strictly positive."""
    task = {"correct": 4.0, "wrong": -0.5, "malformed": -5.0}
    aux_combinations = [
        (p_cot, p_length, p_structure, p_state_variation, r_consistency, r_signal)
        for p_cot in (0.0, cot_max_scale)
        for p_length in (0.0, 0.5)
        for p_structure in (0.0, 0.5)
        for p_state_variation in (0.0, 0.5)
        for r_consistency in (0.0, consistency_magnitude)
        for r_signal in (0.0, signal_magnitude)
    ]
    totals = {
        category: [
            r_task - p_cot - p_length - p_structure - p_state_variation + r_consistency + r_signal
            for p_cot, p_length, p_structure, p_state_variation, r_consistency, r_signal in aux_combinations
        ]
        for category, r_task in task.items()
    }
    margin_correct_over_wrong = min(totals["correct"]) - max(totals["wrong"])
    margin_wrong_over_malformed = min(totals["wrong"]) - max(totals["malformed"])
    assert margin_correct_over_wrong > 0, f"invariant broken: correct/wrong margin = {margin_correct_over_wrong}"
    assert margin_wrong_over_malformed > 0, f"invariant broken: wrong/malformed margin = {margin_wrong_over_malformed}"
    return {
        "margin_correct_over_wrong": margin_correct_over_wrong,
        "margin_wrong_over_malformed": margin_wrong_over_malformed,
    }


if __name__ == "__main__":
    margins = verify_reward_invariant()
    print("PASSED: reward invariant holds with r_signal included.", margins)
