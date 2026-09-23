"""Stage 9e, direct code-word answer design: token-position logging (task item 5).

Logs the EXACT token position of every Bek/Ner (or Jub/Kag, for the held-out pair)
occurrence in a bank entry's full chat-wrapped completion -- first-class metadata
built in now, not reconstructed retroactively, for the eventual SAE activation-
extraction work (Stage 10).

Requires Llama's real tokenizer (CPU-only -- tokenization does not need the model
loaded, no GPU required for this module). Checked directly (code_word_answer_bank.py's
own docstring): the code word is single-token ONLY in a leading-space context. Both
occurrence types in this bank's completions have a leading space by construction
('State: <token>' and '<answer> <token>'), so every occurrence logged here is a
single token position, not a multi-token span -- verified per-row, not assumed.
"""
from __future__ import annotations

from typing import Any


def build_full_chat_text(tokenizer, prompt: str, completion: str) -> str:
    """The exact text a training/generation pipeline would see: chat-templated
    prefix (ending right before the assistant's turn) + the completion appended as
    the assistant's content -- same pattern as chat_prefix_for() elsewhere in this
    project (sft_seeded_rl_decode_back.py, truncation_diagnostic.py)."""
    prefix = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    return prefix + completion


def log_code_word_positions(tokenizer, prompt: str, completion: str, code_words: tuple[str, str]) -> dict[str, Any]:
    """Tokenizes the full chat text and finds the token index of every occurrence of
    either code word in `code_words` (e.g. ('Bek', 'Ner')). Returns one entry per
    occurrence, tagged with its role (which step, or 'final_answer') and whether it
    was confirmed single-token at that position (should always be True by
    construction here -- checked, not assumed)."""
    full_text = build_full_chat_text(tokenizer, prompt, completion)
    token_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # Single-token ids for each code word, WITH the leading space that both
    # occurrence contexts in this bank use ('State: <token>' and '<answer> <token>').
    code_word_token_ids = {}
    for word in code_words:
        ids = tokenizer.encode(' ' + word, add_special_tokens=False)
        code_word_token_ids[word] = ids[0] if len(ids) == 1 else None  # None flags a real problem, checked below

    occurrences = []
    completion_lines = completion.strip().split('\n')
    n_steps = len(completion_lines) - 1  # last line is the answer

    for position, token_id in enumerate(token_ids):
        for word, expected_id in code_word_token_ids.items():
            if expected_id is not None and token_id == expected_id:
                occurrences.append({'token_position': position, 'code_word': word, 'token_id': token_id})

    # Tag occurrences with their role by matching order: this bank's completions have
    # exactly n_steps intermediate occurrences followed by exactly 1 final-answer
    # occurrence, always in that order (verified per-row by the caller against
    # n_expected_occurrences below, not assumed silently).
    for i, occ in enumerate(occurrences):
        occ['role'] = f'intermediate_step_{i + 1}' if i < n_steps else 'final_answer'

    return {
        'full_text': full_text,
        'n_tokens_total': len(token_ids),
        'code_word_token_ids': code_word_token_ids,
        'occurrences': occurrences,
        'n_occurrences_found': len(occurrences),
        'n_occurrences_expected': n_steps + 1,
        'all_occurrences_single_token_confirmed': all(v is not None for v in code_word_token_ids.values()),
        'occurrence_count_matches_expected': len(occurrences) == n_steps + 1,
    }
