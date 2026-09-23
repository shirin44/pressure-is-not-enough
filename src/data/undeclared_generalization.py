"""Undeclared Stage 3.5 prompts and self-consistency-only scoring."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from src.data.multidomain_seed import DOMAIN_SPECS, MultiDomainExample


TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)
RAW_ANSWER_RE = re.compile(r"<answer>\s*([^<\n]+?)\s*</answer>\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class UndeclaredExample:
    example_id: str
    domain: str
    initial_state: str
    operations: tuple[str, ...]
    expected_states: tuple[str, ...]
    final_answer: str
    prompt: str


def remove_declared_mapping(row: MultiDomainExample) -> UndeclaredExample:
    """Change only mapping availability; retain task and output requirements."""
    spec = DOMAIN_SPECS[row.domain]
    states = spec["states"]
    lines = [
        f"{spec['subject']} starts {row.initial_state}.",
        "Track its physical state through every instruction.",
        *[f"{i}. {spec['same'] if op == 'same' else spec['different']}"
          for i, op in enumerate(row.operations, 1)],
        f"Choose your own two different invented single-word codes for {states[0]} and {states[1]}.",
        f"Do not use {states[0]} or {states[1]} in the numbered reasoning lines.",
        "Keep the same self-chosen code-to-state mapping throughout this problem.",
        "Write each line as: Step i: <brief reasoning>. State: <invented code word>",
        "After all steps, write exactly: Final coded state: <code>. <code> represents <physical state>.",
        "End with the physical state, not its code, inside <answer>...</answer>.",
    ]
    prompt = "\n".join(lines)
    for forbidden in (
        "Represent Lit using the code", "Represent Dark using the code",
        "Therefore, the initial code is", row.token_for_first_state, row.token_for_second_state,
    ):
        if forbidden in prompt:
            raise AssertionError(f"Declared-mapping leakage in {row.example_id}: {forbidden}")
    return UndeclaredExample(
        row.example_id, row.domain, row.initial_state, row.operations,
        row.expected_states, row.final_answer, prompt,
    )


def build_undeclared_lamp_evaluation(rows: Sequence[MultiDomainExample]) -> list[UndeclaredExample]:
    output = [remove_declared_mapping(row) for row in rows]
    if len(output) != 100 or len({row.prompt for row in output}) != 100:
        raise AssertionError("Expected 100 unique undeclared lamp prompts")
    return output


def score_undeclared_completion(example: UndeclaredExample, text: str) -> dict[str, object]:
    """Score only trace self-consistency; no declared token pair exists."""
    prefix = text.split("<answer>", 1)[0]
    matches = [(int(i), token.casefold()) for i, token in STEP_RE.findall(prefix)]
    structural = (
        len(matches) == len(example.operations)
        and [i for i, _ in matches] == list(range(1, len(example.operations) + 1))
    )
    tokens = [token for _, token in matches]
    states = DOMAIN_SPECS[example.domain]["states"]
    mapping = {state: set() for state in states}
    if structural:
        for state, token in zip(example.expected_states, tokens):
            mapping[state].add(token)
    both_states_observed = all(mapping[state] for state in states)
    global_consistent = bool(
        structural and both_states_observed
        and all(len(mapping[state]) == 1 for state in states)
        and next(iter(mapping[states[0]])) != next(iter(mapping[states[1]]))
    )
    literal = {state.casefold() for state in states}
    nonliteral = structural and bool(tokens) and all(token not in literal for token in tokens)
    decode_re = re.compile(
        rf"^Final coded state:\s*({TOKEN})\.\s*\1 represents ({'|'.join(states)})\.\s*$",
        re.MULTILINE,
    )
    decode = decode_re.findall(prefix)
    decode_valid = len(decode) == 1
    decode_token = decode[0][0].casefold() if decode_valid else None
    decode_state = decode[0][1] if decode_valid else None
    final_trace_token = tokens[-1] if structural and tokens else None
    decode_back_self_consistent = bool(
        global_consistent and decode_valid
        and decode_token == final_trace_token
        and decode_state == example.final_answer
        and decode_token in mapping[example.final_answer]
    )
    raw_match = RAW_ANSWER_RE.search(text)
    raw_answer = raw_match.group(1).strip() if raw_match else None
    answer_correct = raw_answer is not None and raw_answer.casefold() == example.final_answer.casefold()
    unique_tokens = sorted(set(tokens))
    token_pair = tuple(unique_tokens) if global_consistent else None
    return {
        "structural": structural,
        "both_states_observed": both_states_observed,
        "global_consistent": global_consistent,
        "nonliteral": nonliteral,
        "nonliteral_consistent": nonliteral and global_consistent,
        "decode_back_valid": decode_valid,
        "decode_back_self_consistent": decode_back_self_consistent,
        "answer_correct": answer_correct,
        "raw_answer": raw_answer,
        "tokens": tokens,
        "unique_token_count": len(unique_tokens),
        "token_pair": token_pair,
        "text": text,
    }
