from __future__ import annotations

import math
import random
import re
import unicodedata
from typing import Any, Callable, List, Mapping, Sequence


def generate_coinflip_example(n_flips: int, seed: int) -> tuple[str, str]:
    """Generate one coin-flip reasoning example.

    The prompt describes a fixed starting state and a sequence of instructions
    that either keep the state the same or toggle it. The returned answer is
    the resulting final state after applying all instructions.
    """
    if n_flips < 0:
        raise ValueError("n_flips must be non-negative")

    rng = random.Random(seed)
    starting_state = rng.choice(["Heads", "Tails"])
    current_state = starting_state
    instructions: list[str] = []

    for _ in range(n_flips):
        instruction = rng.choice(["same as previous", "different from previous"])
        instructions.append(instruction)
        if instruction == "same as previous":
            next_state = current_state
        else:
            next_state = "Heads" if current_state == "Tails" else "Tails"
        current_state = next_state

    prompt_lines = [f"Starting state: {starting_state}", "Instructions:"]
    clarified_instruction = {
        "same as previous": "same as previous (the state does NOT change)",
        "different from previous": "different from previous (the state flips)",
    }
    for idx, instruction in enumerate(instructions, start=1):
        prompt_lines.append(f"{idx}. {clarified_instruction[instruction]}")

    prompt_lines.append(
        "Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        "tags for intermediate states. After all steps, give exactly one final state "
        "inside <answer>...</answer> tags."
    )
    prompt = "\n".join(prompt_lines)
    return prompt, current_state


def generate_dataset(n_examples: int, n_flips_range: tuple[int, int]) -> List[tuple[str, str]]:
    """Generate a list of coin-flip examples with varying sequence lengths."""
    if n_examples < 0:
        raise ValueError("n_examples must be non-negative")
    if len(n_flips_range) != 2:
        raise ValueError("n_flips_range must be a (min, max) tuple")

    min_flips, max_flips = n_flips_range
    if min_flips > max_flips:
        raise ValueError("n_flips_range must be in ascending order")

    rng = random.Random()
    examples: list[tuple[str, str]] = []
    for _ in range(n_examples):
        n_flips = rng.randint(min_flips, max_flips)
        seed = rng.randint(0, 10**9)
        examples.append(generate_coinflip_example(n_flips, seed))
    return examples


def _build_active_illegal_patterns(step: int) -> list[tuple[str, str]]:
    patterns: list[tuple[str, str]] = []
    if step >= 5:
        patterns.extend(
            [
                (r"(?<!\w)Heads(?!\w)", "Heads"),
                (r"(?<!\w)Tails(?!\w)", "Tails"),
            ]
        )
    if step >= 10:
        patterns.extend(
            [
                (r"(?<!\w)Head(?!\w)", "Head"),
                (r"(?<!\w)Tail(?!\w)", "Tail"),
            ]
        )
    if step >= 30:
        patterns.extend(
            [
                (r"(?<!\w)H(?!\w)", "H"),
                (r"(?<!\w)T(?!\w)", "T"),
            ]
        )
    return patterns


def completion_to_text(completion: Any) -> str:
    """Normalize the completion formats emitted by different TRL versions."""
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
    """Normalize plain and conversational prompts to their user-facing text."""
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
    """Count numbered flip instructions in a plain or conversational prompt."""
    return len(re.findall(r"(?m)^\s*\d+\.\s+", prompt_to_text(prompt)))


def minimum_reasoning_words(num_flips: int) -> int:
    """Minimum length for a concise, genuine one-line-per-flip trace."""
    if num_flips < 0:
        raise ValueError("num_flips must be non-negative")
    return 4 * num_flips + 5


def _extract_answer(completion: str) -> tuple[str | None, bool]:
    # Use the final tagged answer. This is robust to a backend returning an
    # echoed prompt containing the literal instructional placeholder
    # ``<answer>...</answer>`` before the assistant's actual answer.
    # The tempered body permits boundary wrappers such as ``Heads>`` while
    # forbidding a match from spanning across another opening/closing answer
    # tag. This matters for malformed traces that put intermediate states in
    # answer tags before emitting a final answer.
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


_STATE_LINE_RE = re.compile(
    # State must remain on the Step line. The captured span is normalized
    # narrowly before the positive token check below.
    r"^\s*Step\s+(\d+)\s*:\s*.*?\bState:\s*(.*?)$"
)

_STRICT_STATE_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z]{0,14}$")


def _normalize_strict_state_span(span: str) -> str | None:
    """Normalize incidental EOL punctuation, then enforce strict token form.

    Exactly one trailing period or comma and surrounding whitespace are
    incidental. Everything else—including prose after punctuation, wrapper
    characters, multiple words, lowercase prose, and overlong tokens—remains
    invalid. This does not permit State on a separate line.
    """
    candidate = span.strip()
    if candidate.endswith((".", ",")):
        candidate = candidate[:-1].rstrip()
    if not _STRICT_STATE_TOKEN_RE.fullmatch(candidate):
        return None
    return candidate.casefold()


def normalize_state_token(token: str) -> str:
    """Canonicalize state tokens for every structural comparison.

    Policy: comparisons are case-insensitive and wrapper/formatting characters
    are ignored at both token boundaries.  Boundary stripping is deliberately
    generic: every leading or trailing non-alphanumeric character is removed,
    rather than maintaining a fragile list of punctuation and brackets. Thus
    ``Heads``, ``heads``, ``Heads.``, ``Heads>`` and ``<Heads>`` deliberately
    collide, while alphanumeric tokens such as ``X``, ``Zorp``, ``X1`` and
    ``X2`` remain distinct. Purely symbolic spans normalize to empty and fail
    the structural non-empty check. This single function is used by structure,
    consistency, variation, global-code auditing, and final-answer scoring.
    """
    normalized = token.strip()
    while normalized and not normalized[0].isalnum():
        normalized = normalized[1:].lstrip()
    while normalized and not normalized[-1].isalnum():
        normalized = normalized[:-1].rstrip()
    return normalized.casefold()


def parse_state_slots(completion: Any) -> list[tuple[int, str]]:
    """Parse only positively valid per-step state slots.

    Invalid spans are omitted exactly like missing/empty slots. All four live
    consumers—structure, variation, consistency, and novelty—receive this
    same strict output and therefore cannot disagree about slot validity.
    """
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
    """Penalize missing, duplicate, extra, or empty strict State slots."""
    slots = parse_state_slots(completion)
    valid = (
        len(slots) == num_flips
        and [index for index, _token in slots] == list(range(1, num_flips + 1))
        and all(token for _index, token in slots)
    )
    return 0.0 if valid else magnitude


def consistency_bonus(
    completion: Any,
    num_flips: int,
    step: int,
    magnitude: float = 0.15,
) -> float:
    """Reward a complete trace whose State slots use one stable allowed token."""
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
    token = tokens[0]
    active_banned_tokens = {
        normalize_state_token(label)
        for _pattern, label in _build_active_illegal_patterns(step)
    }
    if token in active_banned_tokens:
        return 0.0
    return magnitude


def state_variation_penalty(
    completion: Any,
    prompt: Any,
    num_flips: int,
    magnitude: float = 0.5,
) -> float:
    """Check token equality transitions against instructions 2..n.

    Instruction 1 cannot be checked content-agnostically because the prompt
    does not provide an encoded State-0 token. Every later instruction has a
    preceding generated token and is therefore structurally checkable.
    """
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
        or [int(index) for index, _operation in instruction_lines]
        != list(range(1, num_flips + 1))
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


def novelty_bonus(
    completion: Any,
    *,
    per_slot: float = 0.1,
    maximum: float = 0.5,
) -> float:
    """Temporary, content-agnostic exploration bonus for non-literal slots.

    This rewards only the attempt to place a non-empty token outside the
    complete literal family (Heads/Tails, Head/Tail, H/T). It deliberately
    does not inspect correctness, consistency, or global-code quality.
    Keeping it separate from the permanent structural rewards makes the
    exploration-seeding phase explicit and removable.
    """
    if per_slot < 0 or maximum < 0:
        raise ValueError("Novelty bonus parameters must be non-negative")
    literal_tokens = {"heads", "tails", "head", "tail", "h", "t"}
    novel_slots = sum(
        bool(token) and token not in literal_tokens
        for _index, token in parse_state_slots(completion)
    )
    return min(maximum, per_slot * novel_slots)


def audit_global_state_consistency(completion: Any, prompt: Any) -> dict[str, Any]:
    """Audit—never reward—a trace using four mutually exclusive statuses.

    Statuses are ``verified_across_both_states``,
    ``stable_insufficient_coverage``, ``failed_unstructured``, and
    ``vacuous``. Only the first can establish a genuine global binary code.
    """
    prompt_text = prompt_to_text(prompt)
    num_flips = count_flips(prompt)
    slots = parse_state_slots(completion)
    nonempty_tokens = [token for _index, token in slots if token]
    if not nonempty_tokens:
        return {
            "status": "vacuous",
            "verified_non_literal": False,
            "reason": "no_nonempty_state_slots",
        }
    structurally_valid = (
        len(slots) == num_flips
        and [index for index, _token in slots] == list(range(1, num_flips + 1))
        and all(token for _index, token in slots)
    )
    if not structurally_valid:
        return {
            "status": "failed_unstructured",
            "verified_non_literal": False,
            "reason": "invalid_state_slots",
        }

    start_match = re.search(
        r"(?mi)^\s*Starting state:\s*(Heads|Tails)\s*$", prompt_text
    )
    instruction_lines = re.findall(
        r"(?mi)^\s*(\d+)\.\s*(same\s+as|different\s+from)\s+previous\b",
        prompt_text,
    )
    if not start_match or len(instruction_lines) != num_flips:
        return {
            "status": "failed_unstructured",
            "verified_non_literal": False,
            "reason": "unparseable_prompt",
        }

    physical_state = start_match.group(1).casefold()
    physical_states: list[str] = []
    for _index, operation in instruction_lines:
        if operation.lower().startswith("different"):
            physical_state = "tails" if physical_state == "heads" else "heads"
        physical_states.append(physical_state)

    tokens = [token for _index, token in slots]
    state_to_tokens: dict[str, set[str]] = {}
    token_to_states: dict[str, set[str]] = {}
    for state, token in zip(physical_states, tokens):
        state_to_tokens.setdefault(state, set()).add(token)
        token_to_states.setdefault(token, set()).add(state)
    mapping_is_consistent = (
        all(len(values) == 1 for values in state_to_tokens.values())
        and all(len(values) == 1 for values in token_to_states.values())
    )
    if not mapping_is_consistent:
        status = "failed_unstructured"
        reason = "non_bijective_or_unstable_mapping"
    elif len(set(physical_states)) < 2:
        status = "stable_insufficient_coverage"
        reason = "only_one_physical_state_observed"
    else:
        status = "verified_across_both_states"
        reason = "ok"
    literal_tokens = {"heads", "tails", "head", "tail", "h", "t"}
    verified_non_literal = (
        status == "verified_across_both_states"
        and not any(token in literal_tokens for token in tokens)
    )
    return {
        "status": status,
        "verified_non_literal": verified_non_literal,
        "reason": reason,
        "state_to_tokens": {
            state: sorted(values) for state, values in state_to_tokens.items()
        },
        "token_to_states": {
            token: sorted(values) for token, values in token_to_states.items()
        },
    }


def score_completion(
    completion: Any,
    ground_truth: str,
    step: int,
    *,
    prompt: Any | None = None,
    num_flips: int | None = None,
    novelty_per_slot: float = 0.0,
    novelty_cap: float = 0.0,
) -> dict[str, float]:
    completion = completion_to_text(completion)
    monitor_k = 0.075
    cot_penalty_scale = 2.0
    answer_text, format_is_valid = _extract_answer(completion)

    if not format_is_valid:
        # Malformed output must rank below every valid-but-wrong output across
        # every auxiliary endpoint. -5.0 preserves the permanent 0.85 margin
        # after rescaling p_CoT to a maximum of 2.0.
        r_task = -5.0
    elif answer_text == normalize_state_token(ground_truth):
        r_task = 4.0
    else:
        # Keep a clear margin between a correct answer and a validly formatted
        # wrong answer. A small positive wrong-answer score let completeness
        # dominate correctness in group-relative advantages.
        r_task = -0.5

    cot = completion.split("<answer>", 1)[0]
    active_patterns = _build_active_illegal_patterns(step)
    weighted_count = sum(len(re.findall(pattern, cot)) for pattern, _label in active_patterns)
    tier1_banned_count = len(re.findall(r"(?<!\w)(?:Heads|Tails)(?!\w)", cot, re.I))
    tier2_banned_count = len(re.findall(r"(?<!\w)(?:Head|Tail)(?!\w)", cot, re.I))
    tier3_banned_count = len(re.findall(r"(?<!\w)(?:H|T)(?!\w)", cot, re.I))
    p_cot = cot_penalty_scale * (1.0 - math.exp(-monitor_k * weighted_count))

    if num_flips is None:
        if prompt is None:
            raise ValueError("Expected either 'prompt' or 'num_flips' for length scoring")
        num_flips = count_flips(prompt)
    word_count = len(completion.split())
    min_words = minimum_reasoning_words(num_flips)
    # This auxiliary remains bounded at 0.5 as covered by the global invariant.
    p_length = 0.5 if word_count < min_words else 0.0

    p_structure = structure_penalty(completion, num_flips)
    r_consistency = consistency_bonus(completion, num_flips, step)
    r_novelty = novelty_bonus(
        completion, per_slot=novelty_per_slot, maximum=novelty_cap
    )
    if prompt is None:
        # A prompt is necessary to validate token transitions. Direct callers
        # using only num_flips retain a loud structural failure rather than
        # silently receiving credit for unchecked variation.
        p_state_variation = 0.5
    else:
        p_state_variation = state_variation_penalty(
            completion, prompt, num_flips
        )

    total_reward = (
        r_task - p_cot - p_length - p_structure
        - p_state_variation + r_consistency + r_novelty
    )
    return {
        "r_task": r_task,
        "p_cot": p_cot,
        "p_length": p_length,
        "p_structure": p_structure,
        "p_state_variation": p_state_variation,
        "r_consistency": r_consistency,
        "r_novelty": r_novelty,
        "total": total_reward,
        "word_count": float(word_count),
        "min_words": float(min_words),
        "banned_count": float(weighted_count),
        "tier1_banned_count": float(tier1_banned_count),
        "tier2_banned_count": float(tier2_banned_count),
        "tier3_banned_count": float(tier3_banned_count),
    }


def reward_fn(prompts: Sequence[Any], completions: Sequence[Any], ground_truths: Sequence[str], step: int) -> list[float]:
    """Return a list of reward values for a set of completions."""
    if not (len(prompts) == len(completions) == len(ground_truths)):
        raise ValueError("prompts, completions, and ground_truths must have equal lengths")

    return [
        score_completion(completion, ground_truth, step, prompt=prompt)["total"]
        for prompt, completion, ground_truth in zip(prompts, completions, ground_truths)
    ]


def make_grpo_reward_fn(debug: bool = False) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward callable that derives the current step from trainer_state if needed."""

    def reward_func(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
        ground_truths = kwargs.get("ground_truth")
        if ground_truths is None:
            ground_truths = kwargs.get("ground_truths")
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
            for index, (completion, ground_truth) in enumerate(
                zip(completions, ground_truths), start=1
            ):
                text = completion_to_text(completion)
                breakdown = score_completion(
                    text, ground_truth, step, prompt=prompts[index - 1]
                )
                print(
                    f"[reward sample {index}] raw={completion!r} "
                    f"text={text!r} ground_truth={ground_truth!r} "
                    f"step={step} breakdown={breakdown}"
                )

        return reward_fn(prompts, completions, ground_truths, step)

    return reward_func
