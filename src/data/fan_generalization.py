"""Deterministic Stage 3.5 zero-shot greenhouse-fan evaluation corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import random
import re
from typing import Sequence


STATES = ("Running", "Stopped")
OPERATIONS = ("same", "different")
DEFAULT_SEED = 20260811
STATE_RE = re.compile(r"^Step (\d+): State: ([A-Z][A-Za-z]{0,14})$", re.MULTILINE)
ANSWER_RE = re.compile(r"<answer>\s*(Running|Stopped)\s*</answer>\s*$", re.IGNORECASE)
RAW_ANSWER_RE = re.compile(r"<answer>\s*([^<\n]+?)\s*</answer>\s*$", re.IGNORECASE)
DECODE_BACK_RE = re.compile(
    r"^Final coded state:\s*([A-Z][A-Za-z]{0,14})\.\s*"
    r"\1 represents (Running|Stopped)\.\s*$",
    re.MULTILINE,
)
LATCH_COIN_WORDS = re.compile(
    # "flip" is intentionally not treated as leakage: it is an ordinary
    # domain-neutral verb for a binary transition. The domain-bearing words
    # below would be genuine latch/coin transfer artifacts.
    r"\b(?:latch|locked|unlocked|heads?|tails?|coin)\b", re.IGNORECASE
)
DECLARATION_RE = re.compile(
    r"^Represent Running using the code ([A-Z][A-Za-z]{2,9}) and "
    r"Stopped using the code ([A-Z][A-Za-z]{2,9})\.$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class FanExample:
    example_id: str
    initial_state: str
    operations: tuple[str, ...]
    expected_states: tuple[str, ...]
    final_answer: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["operations"] = list(self.operations)
        row["expected_states"] = list(self.expected_states)
        return row


@dataclass(frozen=True)
class DeclaredFanExample:
    base: FanExample
    token_for_running: str
    token_for_stopped: str
    prompt: str

    @property
    def example_id(self) -> str:
        return self.base.example_id


def simulate(initial_state: str, operations: Sequence[str]) -> list[str]:
    state = initial_state
    output: list[str] = []
    for operation in operations:
        if operation == "different":
            state = "Stopped" if state == "Running" else "Running"
        elif operation != "same":
            raise ValueError(f"Unknown operation: {operation}")
        output.append(state)
    return output


def _prompt(initial_state: str, operations: Sequence[str]) -> str:
    wording = {
        "same": "same as previously (the fan mode does NOT change)",
        "different": "different from previously (the fan changes to its other mode)",
    }
    lines = [
        f"A greenhouse ventilation fan starts {initial_state}.",
        "Track its mode through every instruction.",
        *[f"{index}. {wording[operation]}" for index, operation in enumerate(operations, 1)],
        "Choose your own two different invented single-word codes for Running and Stopped.",
        "Do not use Running or Stopped in the numbered reasoning lines.",
        "Keep the same code-to-mode mapping throughout this problem.",
        "Write exactly one line per instruction as: Step i: State: <invented code>",
        "After all steps, write exactly: Final coded state: <code>. <code> represents <physical state>.",
        "End with the actual final fan mode inside <answer>...</answer>.",
    ]
    return "\n".join(lines)


def generate_fan_evaluation(seed: int = DEFAULT_SEED, count: int = 100) -> list[FanExample]:
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = {
        (initial, final): [] for initial in STATES for final in STATES
    }
    for length in range(3, 9):
        for initial in STATES:
            for operations in itertools.product(OPERATIONS, repeat=length):
                expected = tuple(simulate(initial, operations))
                if len(set(expected)) != 2:
                    continue
                cells[(initial, expected[-1])].append((initial, operations, expected))
    per_cell = count // 4
    if per_cell * 4 != count:
        raise ValueError("count must be divisible by four")
    selected = []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        selected.extend(cells[key][:per_cell])
    rng.shuffle(selected)
    return [
        FanExample(
            example_id=f"fan-third-domain-{index:03d}",
            initial_state=initial,
            operations=operations,
            expected_states=expected,
            final_answer=expected[-1],
            prompt=_prompt(initial, operations),
        )
        for index, (initial, operations, expected) in enumerate(selected)
    ]


def _nonce_tokens(rng: random.Random):
    consonants, vowels = "bcdfghjklmnprstvwxyz", "aeiou"
    seen: set[str] = set()
    while True:
        token = "".join(rng.choice(consonants) + rng.choice(vowels)
                        for _ in range(rng.choice((2, 3, 4)))).capitalize()
        if token in seen or any(word in token.casefold() for word in
                                ("run", "stop", "lock", "head", "tail", "coin")):
            continue
        seen.add(token)
        yield token


def generate_declared_mapping_control(
    seed: int = DEFAULT_SEED, count: int = 100
) -> list[DeclaredFanExample]:
    """Return the exact third-domain prompts with a unique supplied mapping."""
    base_rows = generate_fan_evaluation(seed, count)
    rng = random.Random(seed + 1)
    tokens = _nonce_tokens(rng)
    output = []
    for index, base in enumerate(base_rows):
        first, second = next(tokens), next(tokens)
        running, stopped = (first, second) if index % 2 == 0 else (second, first)
        declaration = (
            f"Represent Running using the code {running} and Stopped using the code {stopped}."
        )
        prompt = base.prompt.replace(
            "Choose your own two different invented single-word codes for Running and Stopped.",
            declaration,
        ).replace(
            "Keep the same code-to-mode mapping throughout this problem.",
            "Apply this declared code-to-mode mapping throughout this problem.",
        )
        output.append(DeclaredFanExample(base, running, stopped, prompt))
    return output


def verify_declared_control(examples: Sequence[DeclaredFanExample]) -> dict[str, object]:
    failures = []
    for row in examples:
        declarations = DECLARATION_RE.findall(row.prompt)
        if declarations != [(row.token_for_running, row.token_for_stopped)]:
            failures.append(f"{row.example_id}:mapping")
        if LATCH_COIN_WORDS.search(row.prompt):
            failures.append(f"{row.example_id}:leakage")
    return {
        "count": len(examples),
        "unique_prompts": len({row.prompt for row in examples}),
        "unique_token_pairs": len({tuple(sorted((row.token_for_running, row.token_for_stopped)))
                                    for row in examples}),
        "failures": failures,
        "semantic_pass_rate": 100 * (len(examples) - len(failures)) / len(examples),
        "accepted": len(examples) == 100 and not failures,
    }


def score_declared_completion(example: DeclaredFanExample, text: str) -> dict[str, object]:
    score = score_completion(example.base, text)
    expected = {
        "Running": example.token_for_running.casefold(),
        "Stopped": example.token_for_stopped.casefold(),
    }
    score["declared_mapping_adherence"] = (
        bool(score["structural"])
        and score["tokens"] == [expected[state] for state in example.base.expected_states]
    )
    declared_tokens = {
        example.token_for_running.casefold(), example.token_for_stopped.casefold()
    }
    if not score["answer_correct"] and score["raw_answer"] in declared_tokens:
        score["answer_is_code_word"] = True
        score["answer_failure_type"] = "code_word_instead_of_physical_state"
    return score


def verify_corpus(examples: Sequence[FanExample]) -> dict[str, object]:
    failures = []
    for row in examples:
        expected = tuple(simulate(row.initial_state, row.operations))
        if expected != row.expected_states or expected[-1] != row.final_answer:
            failures.append(row.example_id)
        if LATCH_COIN_WORDS.search(row.prompt):
            failures.append(f"{row.example_id}:domain_leakage")
    return {
        "count": len(examples),
        "unique_prompts": len({row.prompt for row in examples}),
        "initial_states": dict(Counter(row.initial_state for row in examples)),
        "final_answers": dict(Counter(row.final_answer for row in examples)),
        "semantic_pass_rate": 100 * (len(examples) - len(failures)) / len(examples),
        "failures": failures,
        "accepted": len(examples) == 100 and not failures,
    }


def score_completion(example: FanExample, text: str) -> dict[str, object]:
    prefix = text.split("<answer>", 1)[0]
    matches = [(int(index), token) for index, token in STATE_RE.findall(prefix)]
    structural = (
        len(matches) == len(example.operations)
        and [index for index, _ in matches] == list(range(1, len(example.operations) + 1))
    )
    tokens = [token.casefold() for _, token in matches]
    literal = {"running", "stopped"}
    leakage = bool(LATCH_COIN_WORDS.search(prefix))
    nonliteral = structural and all(token not in literal for token in tokens) and not leakage
    mapping: dict[str, set[str]] = {state: set() for state in STATES}
    if structural:
        for physical_state, token in zip(example.expected_states, tokens):
            mapping[physical_state].add(token)
    global_consistent = (
        structural
        and all(len(mapping[state]) == 1 for state in STATES)
        and mapping["Running"] != mapping["Stopped"]
    )
    token_pair = None
    if global_consistent:
        token_pair = tuple(sorted((next(iter(mapping["Running"])), next(iter(mapping["Stopped"])))))
    decode_matches = DECODE_BACK_RE.findall(prefix)
    decode_back_valid = len(decode_matches) == 1
    decode_back_token = decode_matches[0][0].casefold() if decode_back_valid else None
    decode_back_state = decode_matches[0][1] if decode_back_valid else None
    final_trace_token = tokens[-1] if structural and tokens else None
    # There is deliberately no declared mapping. The decode-back statement is
    # checked only against the model's own trace and the true physical state.
    decode_back_self_consistent = bool(
        global_consistent
        and decode_back_valid
        and decode_back_token == final_trace_token
        and decode_back_state == example.final_answer
        and decode_back_token in mapping[example.final_answer]
    )
    answer = ANSWER_RE.search(text)
    raw_answer_match = RAW_ANSWER_RE.search(text)
    raw_answer = raw_answer_match.group(1).strip().casefold() if raw_answer_match else None
    answer_correct = bool(answer and answer.group(1).casefold() == example.final_answer.casefold())
    answer_is_code_word = bool(
        raw_answer and raw_answer not in {state.casefold() for state in STATES}
        and raw_answer in set(tokens)
    )
    if answer_correct:
        answer_failure_type = None
    elif answer_is_code_word:
        answer_failure_type = "code_word_instead_of_physical_state"
    elif raw_answer is None:
        answer_failure_type = "malformed_or_missing_answer"
    else:
        answer_failure_type = "other_wrong_answer"
    return {
        "structural": structural,
        "nonliteral": nonliteral,
        "global_consistent": global_consistent,
        "decode_back_valid": decode_back_valid,
        "decode_back_self_consistent": decode_back_self_consistent,
        "decode_back_token": decode_back_token,
        "decode_back_state": decode_back_state,
        "nonliteral_consistent": nonliteral and global_consistent,
        "answer_correct": answer_correct,
        "raw_answer": raw_answer,
        "answer_is_code_word": answer_is_code_word,
        "answer_failure_type": answer_failure_type,
        "latch_coin_leakage": leakage,
        "token_pair": token_pair,
        "tokens": tokens,
        "text": text,
    }
