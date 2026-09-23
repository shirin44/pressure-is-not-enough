"""Deterministic multi-domain binary-state seeding corpus and verifier."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import random
import re
from typing import Sequence


DEFAULT_SEED = 20260812
OPERATIONS = ("same", "different")
DOMAIN_SPECS = {
    "fan": {
        "states": ("Running", "Stopped"),
        "subject": "A greenhouse ventilation fan",
        "same": "same as previously (the fan mode does NOT change)",
        "different": "different from previously (the fan changes to its other mode)",
        "same_reason": "The fan mode stays the same",
        "different_reason": "The fan changes mode",
    },
    "valve": {
        "states": ("Open", "Closed"),
        "subject": "An irrigation water valve",
        "same": "same as previously (the valve position does NOT change)",
        "different": "different from previously (the valve moves to its other position)",
        "same_reason": "The valve position stays the same",
        "different_reason": "The valve changes position",
    },
    "lamp": {
        "states": ("Lit", "Dark"),
        "subject": "A laboratory signal lamp",
        "same": "same as previously (the lamp condition does NOT change)",
        "different": "different from previously (the lamp changes to its other condition)",
        "same_reason": "The lamp condition stays the same",
        "different_reason": "The lamp changes condition",
    },
}
TOKEN_RE = re.compile(r"^[A-Z][A-Za-z]{2,9}$")


@dataclass(frozen=True)
class MultiDomainExample:
    example_id: str
    split: str
    domain: str
    initial_state: str
    operations: tuple[str, ...]
    expected_states: tuple[str, ...]
    token_for_first_state: str
    token_for_second_state: str
    prompt: str
    demonstration: str
    final_answer: str

    def mapping(self) -> dict[str, str]:
        states = DOMAIN_SPECS[self.domain]["states"]
        return {states[0]: self.token_for_first_state, states[1]: self.token_for_second_state}

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["operations"] = list(self.operations)
        row["expected_states"] = list(self.expected_states)
        return row


def simulate(domain: str, initial_state: str, operations: Sequence[str]) -> list[str]:
    states = DOMAIN_SPECS[domain]["states"]
    state = initial_state
    output = []
    for operation in operations:
        if operation == "different":
            state = states[1] if state == states[0] else states[0]
        elif operation != "same":
            raise ValueError(operation)
        output.append(state)
    return output


def _signatures(domain: str, rng: random.Random, per_cell: int):
    states = DOMAIN_SPECS[domain]["states"]
    cells = {(initial, final): [] for initial in states for final in states}
    for length in range(3, 9):
        for initial in states:
            for operations in itertools.product(OPERATIONS, repeat=length):
                final = simulate(domain, initial, operations)[-1]
                cells[(initial, final)].append((initial, operations))
    selected = []
    for key in sorted(cells):
        rng.shuffle(cells[key]); selected.extend(cells[key][:per_cell])
    rng.shuffle(selected)
    return selected


def _train_eval_signatures(domain: str, rng: random.Random, train_per_cell: int, eval_per_cell: int):
    states = DOMAIN_SPECS[domain]["states"]
    cells = {(initial, final): [] for initial in states for final in states}
    for length in range(3, 9):
        for initial in states:
            for operations in itertools.product(OPERATIONS, repeat=length):
                final = simulate(domain, initial, operations)[-1]
                cells[(initial, final)].append((initial, operations))
    training, evaluation = [], []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        training.extend(cells[key][:train_per_cell])
        evaluation.extend(cells[key][train_per_cell:train_per_cell + eval_per_cell])
    rng.shuffle(training); rng.shuffle(evaluation)
    return training, evaluation


def _nonce_stream(rng: random.Random):
    consonants, vowels = "bcdfghjklmnprstvwxyz", "aeiou"
    banned = ("run", "stop", "open", "close", "lit", "dark", "fan", "valve", "lamp",
              "lock", "head", "tail", "coin")
    seen = set()
    while True:
        token = "".join(rng.choice(consonants) + rng.choice(vowels)
                        for _ in range(rng.choice((2, 3, 4)))).capitalize()
        if token in seen or any(word in token.casefold() for word in banned): continue
        seen.add(token); yield token


def _render(domain, initial, operations, expected, mapping, corrected_fan_wording=False):
    spec = DOMAIN_SPECS[domain]; states = spec["states"]
    different_instruction = spec["different"]
    different_reason = spec["different_reason"]
    if domain == "fan" and corrected_fan_wording:
        different_instruction = (
            "different from previously (the fan flips to the opposite state: "
            "Running becomes Stopped, and Stopped becomes Running)"
        )
        different_reason = "The fan flips to its opposite state"
    prompt_lines = [
        f"{spec['subject']} starts {initial}.",
        "Track its physical state through every instruction.",
        f"Represent {states[0]} using the code {mapping[states[0]]} and {states[1]} using the code {mapping[states[1]]}.",
        f"Therefore, the initial code is {mapping[initial]}.",
        *[f"{i}. {spec['same'] if op == 'same' else different_instruction}"
          for i, op in enumerate(operations, 1)],
        "Apply the declared mapping in every numbered step.",
        "Write each line as: Step i: <brief reasoning>. State: <declared code word>",
        "After all steps, write exactly: Final coded state: <code>. <code> represents <physical state>.",
        "End with the physical state, not its code, inside <answer>...</answer>.",
    ]
    demo_lines = [
        f"Step {i}: {spec['same_reason'] if op == 'same' else different_reason}. State: {mapping[state]}"
        for i, (op, state) in enumerate(zip(operations, expected), 1)
    ]
    final = expected[-1]; token = mapping[final]
    demo_lines.extend([f"Final coded state: {token}. {token} represents {final}.",
                       f"<answer>{final}</answer>"])
    return "\n".join(prompt_lines), "\n".join(demo_lines)


def generate_multidomain_dataset(seed: int = DEFAULT_SEED, corrected_fan_wording: bool = False):
    rng = random.Random(seed); nonces = _nonce_stream(rng)
    training, evaluations = [], {domain: [] for domain in DOMAIN_SPECS}
    for domain in ("fan", "valve"):
        train_signatures, eval_signatures = _train_eval_signatures(domain, rng, 100, 25)
        for split, signatures in (("train", train_signatures), ("eval", eval_signatures)):
            target = training if split == "train" else evaluations[domain]
            for index, (initial, operations) in enumerate(signatures):
                states = DOMAIN_SPECS[domain]["states"]
                first, second = next(nonces), next(nonces)
                if index % 2: first, second = second, first
                mapping = {states[0]: first, states[1]: second}
                expected = tuple(simulate(domain, initial, operations))
                prompt, demo = _render(
                    domain, initial, operations, expected, mapping, corrected_fan_wording
                )
                target.append(MultiDomainExample(
                    f"{domain}-{split}-{index:04d}", split, domain, initial, operations,
                    expected, first, second, prompt, demo, expected[-1]))
    domain = "lamp"
    for index, (initial, operations) in enumerate(_signatures(domain, rng, 25)):
        states = DOMAIN_SPECS[domain]["states"]; first, second = next(nonces), next(nonces)
        if index % 2: first, second = second, first
        mapping = {states[0]: first, states[1]: second}
        expected = tuple(simulate(domain, initial, operations))
        prompt, demo = _render(domain, initial, operations, expected, mapping, False)
        evaluations[domain].append(MultiDomainExample(
            f"lamp-eval-{index:04d}", "eval", domain, initial, operations,
            expected, first, second, prompt, demo, expected[-1]))
    # Exact alternation ensures domain interleaving before Trainer shuffling.
    fan = [row for row in training if row.domain == "fan"]
    valve = [row for row in training if row.domain == "valve"]
    interleaved = [row for pair in zip(fan, valve) for row in pair]
    return interleaved, evaluations


def verify_example(row: MultiDomainExample):
    errors = []; spec = DOMAIN_SPECS[row.domain]; states = spec["states"]; mapping = row.mapping()
    expected = tuple(simulate(row.domain, row.initial_state, row.operations))
    if expected != row.expected_states: errors.append("wrong_expected_states")
    declaration = f"Represent {states[0]} using the code {mapping[states[0]]} and {states[1]} using the code {mapping[states[1]]}."
    if row.prompt.count(declaration) != 1: errors.append("mapping_declaration")
    if f"Therefore, the initial code is {mapping[row.initial_state]}." not in row.prompt:
        errors.append("initial_anchor")
    lines = row.demonstration.splitlines(); trace = lines[:len(row.operations)]
    if len(trace) != len(row.operations): errors.append("trace_count")
    for i, (line, state) in enumerate(zip(trace, expected), 1):
        match = re.fullmatch(rf"Step {i}: .*\. State: ([A-Z][A-Za-z]{{2,9}})", line)
        if not match or match.group(1) != mapping[state]: errors.append(f"trace_{i}")
    final = expected[-1]; token = mapping[final]
    if lines[-2:] != [f"Final coded state: {token}. {token} represents {final}.",
                      f"<answer>{final}</answer>"]:
        errors.append("decode_back_or_answer")
    return not errors, errors


def audit_dataset(training, evaluations):
    all_rows = list(training) + [row for rows in evaluations.values() for row in rows]
    failures = []
    for row in all_rows:
        valid, errors = verify_example(row)
        if not valid: failures.append({"example_id": row.example_id, "errors": errors})
    report = {
        "train_count": len(training),
        "train_domains": dict(Counter(row.domain for row in training)),
        "evaluation_domains": {domain: len(rows) for domain, rows in evaluations.items()},
        "decode_back_present_count": sum("Final coded state:" in row.demonstration for row in training),
        "decode_back_training_coverage": sum("Final coded state:" in row.demonstration for row in training)/len(training),
        "semantic_pass_rate": 100*(len(all_rows)-len(failures))/len(all_rows),
        "failures": failures,
    }
    report["accepted"] = (report["train_count"] == 800
                          and report["train_domains"] == {"fan": 400, "valve": 400}
                          and report["evaluation_domains"] == {"fan": 100, "valve": 100, "lamp": 100}
                          and report["decode_back_training_coverage"] == 1.0
                          and not failures)
    return report
