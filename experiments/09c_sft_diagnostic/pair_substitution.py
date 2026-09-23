"""Stage 9c tier (c): score a completion against an EXPLICITLY INSTRUCTED substitution
pair (e.g. Yelt/Yark) never seen during training. reward_v3.py's _extract_answer/
normalize_state_token are reused for token extraction (they're already pair-agnostic --
just casefold whatever's in <answer>...</answer> or after "State:"), but correctness
here is judged against the INSTRUCTED mapping, not the literal Heads/Tails vocabulary,
so this needs its own scoring function rather than reward_v3's ground-truth comparison.
"""
from __future__ import annotations

import re
from typing import Any

LITERAL_WORDS = {"heads", "tails", "head", "tail", "h", "t"}


def build_substitution_instruction(heads_word: str, tails_word: str) -> str:
    """Appended as an extra sentence AFTER the standard prompt's instructions paragraph
    -- never touches the 'Starting state:' or numbered operation lines that
    taxonomy/reward prompt-parsing depend on."""
    return (f" In this exercise, use the word '{heads_word}' in place of Heads and the "
            f"word '{tails_word}' in place of Tails throughout your reasoning and in "
            f"the final <answer> tag -- do not use the words Heads or Tails at all.")


def build_prompt_with_substitution(base_prompt: str, heads_word: str, tails_word: str) -> str:
    return base_prompt + build_substitution_instruction(heads_word, tails_word)


def score_pair_substitution(
    completion: str, true_states: list[str], heads_word: str, tails_word: str,
) -> dict[str, Any]:
    """true_states: the length-5 physical Heads/Tails trace (independently computable
    from the prompt's starting state + operations, same as everywhere else in this
    project). Returns per-slot correctness, final-answer correctness, and whether the
    model stayed within the instructed pair (no lapse into literal Heads/Tails or any
    other word) -- three independent, separately-reportable signals, not collapsed into
    one pass/fail."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
    from reward_v3 import parse_state_slots, _extract_answer  # noqa: E402

    heads_norm = heads_word.strip().casefold()
    tails_norm = tails_word.strip().casefold()
    expected_tokens = [heads_norm if s == "Heads" else tails_norm for s in true_states]

    slots = parse_state_slots(completion)
    actual_tokens = [t for _i, t in slots]

    lapsed_into_literal = any(t in LITERAL_WORDS for t in actual_tokens)
    used_other_word = any(t not in (heads_norm, tails_norm) and t not in LITERAL_WORDS for t in actual_tokens)
    stayed_within_instructed_pair = not lapsed_into_literal and not used_other_word and bool(actual_tokens)

    correct_slot_count = len(actual_tokens) == len(expected_tokens) and [i for i, _t in slots] == list(range(1, len(expected_tokens) + 1))
    per_slot_correct = (actual_tokens == expected_tokens) if correct_slot_count else False

    answer, valid_answer = _extract_answer(completion)
    expected_final = expected_tokens[-1] if expected_tokens else None
    final_answer_correct = bool(valid_answer and answer == expected_final)

    return {
        "expected_tokens": expected_tokens,
        "actual_tokens": actual_tokens,
        "slot_count_correct": correct_slot_count,
        "per_slot_correct": per_slot_correct,
        "final_answer_correct": final_answer_correct,
        "stayed_within_instructed_pair": stayed_within_instructed_pair,
        "lapsed_into_literal_heads_tails": lapsed_into_literal,
        "fully_correct": per_slot_correct and final_answer_correct and stayed_within_instructed_pair,
    }
