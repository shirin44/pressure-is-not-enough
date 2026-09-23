"""Deterministic generator for the unrelated latch demonstration task.

The task teaches the abstract strategy "represent two physical states with two
stable arbitrary tokens" without using any coin-flip vocabulary.  Every
example has its own unique pair of pronounceable nonce tokens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


STATES = ("Locked", "Unlocked")
OPERATIONS = ("same", "different")
BANNED_TOKEN_SUBSTRINGS = ("head", "tail", "lock", "unlock", "coin", "flip")
DEFAULT_SEED = 20260809

_CONSONANTS = "bcdfghjklmnprstvwxyz"
_VOWELS = "aeiou"
_TRACE_RE = re.compile(
    r"^Step (\d+): The latch (stays the same|switches)\. State: ([A-Za-z]+)$"
)
_EXPLICIT_TRACE_RE = re.compile(
    r"^Step (\d+): Previous code: ([A-Za-z]+)\. "
    r"The latch (stays the same|switches)\. State: ([A-Za-z]+)$"
)
_ANSWER_RE = re.compile(r"^<answer>(Locked|Unlocked)</answer>$")
_DECODE_RE = re.compile(
    r"^Final coded state: ([A-Za-z]+)\. ([A-Za-z]+) represents (Locked|Unlocked)\.$"
)
_MAPPING_RE = re.compile(
    r"^Represent Locked using the code ([A-Za-z]+) and Unlocked using the code ([A-Za-z]+)\.$",
    re.MULTILINE,
)
_ANCHOR_RE = re.compile(r"^Therefore, the initial code is ([A-Za-z]+)\.$")


@dataclass(frozen=True)
class LatchExample:
    example_id: str
    split: str
    initial_state: str
    operations: tuple[str, ...]
    token_for_locked: str
    token_for_unlocked: str
    requires_mapping_anchor: bool
    requires_explicit_transitions: bool
    prompt: str
    demonstration: str
    final_answer: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["operations"] = list(self.operations)
        return row


def _toggle(state: str) -> str:
    if state not in STATES:
        raise ValueError(f"Unknown latch state: {state!r}")
    return "Unlocked" if state == "Locked" else "Locked"


def simulate(initial_state: str, operations: Sequence[str]) -> list[str]:
    """Return the physical state after every operation."""
    if initial_state not in STATES:
        raise ValueError(f"Unknown initial state: {initial_state!r}")
    state = initial_state
    states: list[str] = []
    for operation in operations:
        if operation == "different":
            state = _toggle(state)
        elif operation != "same":
            raise ValueError(f"Unknown operation: {operation!r}")
        states.append(state)
    return states


def _all_signatures() -> list[tuple[str, tuple[str, ...]]]:
    signatures: list[tuple[str, tuple[str, ...]]] = []
    for length in range(3, 9):
        for initial_state in STATES:
            for operations in itertools.product(OPERATIONS, repeat=length):
                signatures.append((initial_state, operations))
    return signatures


def _balanced_signatures(rng: random.Random) -> tuple[list[tuple[str, tuple[str, ...]]], list[tuple[str, tuple[str, ...]]]]:
    """Select 800/200 unique signatures, balanced on initial and final state."""
    cells: dict[tuple[str, str], list[tuple[str, tuple[str, ...]]]] = {
        (initial, final): [] for initial in STATES for final in STATES
    }
    for signature in _all_signatures():
        initial, operations = signature
        final = simulate(initial, operations)[-1]
        cells[(initial, final)].append(signature)

    train: list[tuple[str, tuple[str, ...]]] = []
    heldout: list[tuple[str, tuple[str, ...]]] = []
    for key in sorted(cells):
        candidates = cells[key]
        rng.shuffle(candidates)
        # Every cell contains 252 signatures. Selecting 200 + 50 gives exact
        # 50/50 initial and final distributions in both splits.
        train.extend(candidates[:200])
        heldout.extend(candidates[200:250])
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def _nonce_stream(rng: random.Random) -> Iterable[str]:
    """Yield unique-looking, pronounceable, capitalized alphabetic words."""
    seen: set[str] = set()
    while True:
        syllables = rng.choice((2, 3, 4))
        raw = "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(syllables))
        # Occasional final consonant expands the space while retaining a
        # pronounceable shape. Length remains within 3--10 characters.
        if rng.random() < 0.35 and len(raw) < 10:
            raw += rng.choice(_CONSONANTS)
        token = raw.capitalize()
        lowered = token.casefold()
        if not 3 <= len(token) <= 10:
            continue
        if any(part in lowered for part in BANNED_TOKEN_SUBSTRINGS):
            continue
        if token in seen:
            continue
        seen.add(token)
        yield token


def _render_prompt(
    initial_state: str,
    operations: Sequence[str],
    mapping: dict[str, str],
    include_mapping_anchor: bool,
    include_decode_back: bool,
    include_explicit_transitions: bool,
) -> str:
    lines = [
        f"A door latch starts {initial_state}.",
        "Track the latch through each instruction.",
        f"Represent Locked using the code {mapping['Locked']} and Unlocked using the code {mapping['Unlocked']}.",
    ]
    if include_mapping_anchor:
        lines.append(f"Therefore, the initial code is {mapping[initial_state]}.")
    wording = {
        "same": "same as previous (the latch state does NOT change)",
        "different": "different from previous (the latch state switches)",
    }
    lines.extend(f"{index}. {wording[operation]}" for index, operation in enumerate(operations, 1))
    lines.extend(
        [
            "Apply the declared code mapping in every reasoning step.",
            "Reuse the declared code when the physical state is the same and switch to the other declared code when it changes.",
            ("Write each line as: Step i: Previous code: <prior declared code word>. "
             "The latch <stays the same/switches>. State: <new declared code word>"
             if include_explicit_transitions else
             "Write each line as: Step i: <brief reasoning>. State: <declared code word>"),
        ]
    )
    if include_decode_back:
        lines.extend(
            [
                "Do not write a decode-back line after individual numbered steps.",
                "After all numbered steps, write exactly one decode-back line immediately before the answer: Final coded state: <declared code word>. <declared code word> represents <Locked/Unlocked>.",
            ]
        )
    lines.append("End with the actual final latch state inside <answer>...</answer>.")
    return "\n".join(lines)


def _render_demonstration(
    states: Sequence[str],
    operations: Sequence[str],
    mapping: dict[str, str],
    include_decode_back: bool,
    include_explicit_transitions: bool,
    initial_state: str,
) -> str:
    action = {"same": "stays the same", "different": "switches"}
    previous_states = [initial_state, *states[:-1]]
    if include_explicit_transitions:
        lines = [
            (f"Step {index}: Previous code: {mapping[previous_state]}. "
             f"The latch {action[operation]}. State: {mapping[state]}")
            for index, (previous_state, state, operation) in enumerate(
                zip(previous_states, states, operations), 1
            )
        ]
    else:
        lines = [
            f"Step {index}: The latch {action[operation]}. State: {mapping[state]}"
            for index, (state, operation) in enumerate(zip(states, operations), 1)
        ]
    if include_decode_back:
        final_token = mapping[states[-1]]
        lines.append(f"Final coded state: {final_token}. {final_token} represents {states[-1]}.")
    lines.append(f"<answer>{states[-1]}</answer>")
    return "\n".join(lines)


def verify_example(example: LatchExample) -> tuple[bool, list[str]]:
    """Parse and semantically verify the rendered worked demonstration."""
    errors: list[str] = []
    expected_states = simulate(example.initial_state, example.operations)
    mapping = {
        "Locked": example.token_for_locked,
        "Unlocked": example.token_for_unlocked,
    }
    declarations = _MAPPING_RE.findall(example.prompt)
    if len(declarations) != 1:
        errors.append("missing_or_duplicate_prompt_mapping")
    elif declarations[0] != (mapping["Locked"], mapping["Unlocked"]):
        errors.append("wrong_prompt_mapping")
    prompt_lines = example.prompt.splitlines()
    anchor_matches=[(index,_ANCHOR_RE.fullmatch(line))
                    for index,line in enumerate(prompt_lines)
                    if _ANCHOR_RE.fullmatch(line)]
    if example.requires_mapping_anchor:
        if len(anchor_matches) != 1:
            errors.append("missing_or_duplicate_mapping_anchor")
        else:
            anchor_index,anchor_match=anchor_matches[0]
            declaration_indices=[index for index,line in enumerate(prompt_lines)
                                 if _MAPPING_RE.fullmatch(line)]
            if len(declaration_indices)!=1 or anchor_index!=declaration_indices[0]+1:
                errors.append("mapping_anchor_wrong_position")
            expected_initial_token=mapping[example.initial_state]
            if anchor_match.group(1) != expected_initial_token:  # type: ignore[union-attr]
                errors.append("wrong_mapping_anchor_token")
    elif anchor_matches:
        errors.append("unexpected_mapping_anchor")
    lines = example.demonstration.splitlines()
    expects_decode_back = "write exactly one decode-back line immediately before the answer:" in example.prompt
    trace_lines = lines[: len(example.operations)]
    decode_lines = lines[len(example.operations):-1]
    answer_lines = lines[-1:]
    if len(trace_lines) != len(example.operations):
        errors.append("wrong_trace_line_count")
    for expected_index, (line, operation, state) in enumerate(
        zip(trace_lines, example.operations, expected_states), 1
    ):
        match = (_EXPLICIT_TRACE_RE if example.requires_explicit_transitions else _TRACE_RE).fullmatch(line)
        if not match:
            errors.append(f"malformed_trace_line_{expected_index}")
            continue
        if example.requires_explicit_transitions:
            index = int(match.group(1))
            previous_token, stated_action, token = match.group(2), match.group(3), match.group(4)
            expected_previous_state = example.initial_state if expected_index == 1 else expected_states[expected_index - 2]
            if previous_token != mapping[expected_previous_state]:
                errors.append(f"wrong_previous_code_token_{expected_index}")
        else:
            index, stated_action, token = int(match.group(1)), match.group(2), match.group(3)
        if index != expected_index:
            errors.append(f"wrong_step_index_{expected_index}")
        expected_action = "stays the same" if operation == "same" else "switches"
        if stated_action != expected_action:
            errors.append(f"wrong_action_{expected_index}")
        if token != mapping[state]:
            errors.append(f"wrong_code_token_{expected_index}")
    if expects_decode_back:
        if len(decode_lines) != 1:
            errors.append("missing_or_duplicate_decode_back_line")
        else:
            decode = _DECODE_RE.fullmatch(decode_lines[0])
            if not decode:
                errors.append("malformed_decode_back_line")
            else:
                first_token, second_token, decoded_state = decode.groups()
                expected_final_state = expected_states[-1]
                expected_final_token = mapping[expected_final_state]
                if first_token != expected_final_token or second_token != expected_final_token:
                    errors.append("wrong_decode_back_token")
                if decoded_state != expected_final_state:
                    errors.append("wrong_decode_back_state")
    elif decode_lines:
        errors.append("unexpected_decode_back_line")
    if len(answer_lines) != 1 or not _ANSWER_RE.fullmatch(answer_lines[0]):
        errors.append("malformed_final_answer")
    elif _ANSWER_RE.fullmatch(answer_lines[0]).group(1) != expected_states[-1]:  # type: ignore[union-attr]
        errors.append("wrong_final_answer")
    if example.final_answer != expected_states[-1]:
        errors.append("wrong_stored_final_answer")
    return not errors, errors


def generate_latch_seed_dataset(
    seed: int = DEFAULT_SEED,
    include_decode_back: bool = True,
    include_mapping_anchor: bool = True,
    include_explicit_transitions: bool = True,
) -> tuple[list[LatchExample], list[LatchExample]]:
    """Generate the fixed 800/200 seeding corpus."""
    rng = random.Random(seed)
    train_signatures, heldout_signatures = _balanced_signatures(rng)
    nonce_words = _nonce_stream(rng)
    examples: dict[str, list[LatchExample]] = {"train": [], "heldout": []}

    for split, signatures in (("train", train_signatures), ("heldout", heldout_signatures)):
        orientations = [True] * (len(signatures) // 2) + [False] * (len(signatures) // 2)
        rng.shuffle(orientations)
        for index, ((initial, operations), locked_gets_first) in enumerate(zip(signatures, orientations)):
            first, second = sorted((next(nonce_words), next(nonce_words)))
            locked_token, unlocked_token = (first, second) if locked_gets_first else (second, first)
            states = simulate(initial, operations)
            mapping = {"Locked": locked_token, "Unlocked": unlocked_token}
            example = LatchExample(
                example_id=f"latch-{split}-{index:04d}",
                split=split,
                initial_state=initial,
                operations=operations,
                token_for_locked=locked_token,
                token_for_unlocked=unlocked_token,
                requires_mapping_anchor=include_mapping_anchor,
                requires_explicit_transitions=include_explicit_transitions,
                prompt=_render_prompt(initial, operations, mapping, include_mapping_anchor,
                                      include_decode_back, include_explicit_transitions),
                demonstration=_render_demonstration(
                    states, operations, mapping, include_decode_back,
                    include_explicit_transitions, initial
                ),
                final_answer=states[-1],
            )
            examples[split].append(example)
    return examples["train"], examples["heldout"]


def audit_latch_seed_dataset(
    train: Sequence[LatchExample],
    heldout: Sequence[LatchExample],
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    all_examples = list(train) + list(heldout)
    pair_counts = Counter(
        tuple(sorted((row.token_for_locked, row.token_for_unlocked))) for row in all_examples
    )
    token_counts = Counter(
        token
        for row in all_examples
        for token in (row.token_for_locked, row.token_for_unlocked)
    )
    prompt_counts = Counter(row.prompt for row in all_examples)
    violations = [
        (row.example_id, token, part)
        for row in all_examples
        for token in (row.token_for_locked, row.token_for_unlocked)
        for part in BANNED_TOKEN_SUBSTRINGS
        if part in token.casefold()
    ]
    semantic_failures = []
    prompt_mapping_failures = []
    mapping_anchor_failures = []
    decode_back_failures = []
    explicit_transition_failures = []
    for row in all_examples:
        declarations = _MAPPING_RE.findall(row.prompt)
        if declarations != [(row.token_for_locked, row.token_for_unlocked)]:
            prompt_mapping_failures.append(row.example_id)
        valid, errors = verify_example(row)
        if any("mapping_anchor" in error for error in errors):
            mapping_anchor_failures.append(row.example_id)
        if any("decode_back" in error for error in errors):
            decode_back_failures.append(row.example_id)
        if row.requires_explicit_transitions:
            trace_lines = row.demonstration.splitlines()[:len(row.operations)]
            if (len(trace_lines) != len(row.operations)
                    or any(not _EXPLICIT_TRACE_RE.fullmatch(line) for line in trace_lines)
                    or any("previous_code_token" in error for error in errors)):
                explicit_transition_failures.append(row.example_id)
        if not valid:
            semantic_failures.append({"example_id": row.example_id, "errors": errors})

    def distribution(rows: Sequence[LatchExample], field: str) -> dict[str, int]:
        return dict(sorted(Counter(getattr(row, field) for row in rows).items()))

    orientation_first = sum(
        row.token_for_locked == min(row.token_for_locked, row.token_for_unlocked)
        for row in all_examples
    )
    same_ops = sum(row.operations.count("same") for row in all_examples)
    total_ops = sum(len(row.operations) for row in all_examples)
    report: dict[str, object] = {
        "seed": seed,
        "sizes": {"train": len(train), "heldout": len(heldout), "total": len(all_examples)},
        "unique_prompt_percentage": 100.0 * len(prompt_counts) / len(all_examples),
        "duplicate_prompt_count": sum(count - 1 for count in prompt_counts.values()),
        "unique_token_pair_percentage": 100.0 * len(pair_counts) / len(all_examples),
        "max_token_pair_frequency": max(pair_counts.values()),
        "unique_individual_token_percentage": 100.0 * len(token_counts) / (2 * len(all_examples)),
        "max_individual_token_frequency": max(token_counts.values()),
        "mapping_orientation": {
            "locked_is_alphabetically_first": orientation_first,
            "locked_is_alphabetically_second": len(all_examples) - orientation_first,
        },
        "banned_substring_violation_count": len(violations),
        "banned_substring_violations": violations,
        "semantic_verification_pass_rate": 100.0 * (len(all_examples) - len(semantic_failures)) / len(all_examples),
        "semantic_failures": semantic_failures,
        "prompt_mapping_verification_pass_rate": 100.0 * (len(all_examples) - len(prompt_mapping_failures)) / len(all_examples),
        "prompt_mapping_failures": prompt_mapping_failures,
        "mapping_anchor_verification_pass_rate": 100.0 * (len(all_examples) - len(mapping_anchor_failures)) / len(all_examples),
        "mapping_anchor_failures": mapping_anchor_failures,
        "decode_back_verification_pass_rate": 100.0 * (len(all_examples) - len(decode_back_failures)) / len(all_examples),
        "decode_back_failures": decode_back_failures,
        "explicit_transition_verification_pass_rate": 100.0 * (len(all_examples) - len(explicit_transition_failures)) / len(all_examples),
        "explicit_transition_failures": explicit_transition_failures,
        "train_initial_states": distribution(train, "initial_state"),
        "heldout_initial_states": distribution(heldout, "initial_state"),
        "train_final_answers": distribution(train, "final_answer"),
        "heldout_final_answers": distribution(heldout, "final_answer"),
        "operation_distribution": {
            "same": same_ops,
            "different": total_ops - same_ops,
            "same_percentage": 100.0 * same_ops / total_ops,
        },
    }
    report["accepted"] = all(
        (
            len(train) == 800,
            len(heldout) == 200,
            report["unique_prompt_percentage"] == 100.0,
            report["unique_token_pair_percentage"] == 100.0,
            report["max_individual_token_frequency"] == 1,
            report["banned_substring_violation_count"] == 0,
            report["semantic_verification_pass_rate"] == 100.0,
            report["prompt_mapping_verification_pass_rate"] == 100.0,
            report["mapping_anchor_verification_pass_rate"] == 100.0,
            report["decode_back_verification_pass_rate"] == 100.0,
            report["explicit_transition_verification_pass_rate"] == 100.0,
        )
    )
    return report


def write_latch_seed_dataset(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    include_decode_back: bool = True,
    include_mapping_anchor: bool = True,
    include_explicit_transitions: bool = True,
) -> dict[str, object]:
    train, heldout = generate_latch_seed_dataset(
        seed, include_decode_back, include_mapping_anchor, include_explicit_transitions
    )
    report = audit_latch_seed_dataset(train, heldout, seed=seed)
    if not report["accepted"]:
        raise RuntimeError(f"Latch dataset acceptance gate failed: {report}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("heldout", heldout)):
        path = output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows))
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
