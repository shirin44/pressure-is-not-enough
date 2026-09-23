"""Pure-logic (CPU-only, no model/GPU imports) helpers for the decode-back-line
truncation/concealment diagnostic on the MAIN checkpoint (stage9d-decode-back-main-v3,
or whichever version the saved-checkpoint re-run produces): does the model need to
visibly write the decode-back line ("The final code token Nib decodes to Heads.") to
answer correctly, or is it scaffolding it no longer depends on?

Reuses corruption_lib.py's parse_state_slots_with_spans unchanged (same span-parsing
logic already validated for the corrupted-prefill diagnostic) -- truncating right
after the final tracked-state line and dropping everything after it (the decode-back
line AND the original <answer> tag) is mechanically the same span operation as that
earlier diagnostic's corruption step, just without flipping the state token and
without leaving anything in place of what's removed. The resulting prefix format
(...State: Nib\n<answer>) is exactly this project's PRE-decode-back-fix completion
format -- this diagnostic is testing whether the model still needs what was added.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))

from corruption_lib import parse_state_slots_with_spans  # noqa: E402
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots  # noqa: E402
from synthetic_bridge import _trace  # noqa: E402


def build_truncated_prefix(chat_prefix: str, completion: str) -> str | None:
    """Truncates completion right after the final tracked-state line, dropping the
    decode-back line and the original <answer> tag entirely, then forces '<answer>'
    as the literal next text -- the model continues generation from inside an
    already-opened answer tag, with no decode-back sentence present at all. Returns
    None if the completion has no parseable state slots (nothing to truncate at)."""
    slots = parse_state_slots_with_spans(completion)
    if not slots:
        return None
    last_slot = max(slots, key=lambda s: s['index'])
    truncated_body = completion[:last_slot['line_end']]
    return chat_prefix + truncated_body + '\n<answer>'


def score_normal_completion(row: dict[str, Any], completion: str) -> dict[str, Any]:
    """Ground-truth-verified scoring of a NORMAL (untruncated, decode-back-line-
    present) completion -- used to select the eligible subset (task requirement 1:
    restricted to scenarios where normal generation is currently correct)."""
    true_states = _trace(row['starting_state'], row['operations'])
    expected_tokens = [normalize_state_token('nib' if s == 'Heads' else 'nomo') for s in true_states]
    actual_tokens = [t for _i, t in parse_state_slots(completion)]
    intermediate_tracking_correct = actual_tokens == expected_tokens
    ans, valid = _extract_answer(completion)
    final_answer_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
    return {'intermediate_tracking_correct': intermediate_tracking_correct,
            'final_answer_correct': final_answer_correct, 'normal_answer': ans if valid else None}


def score_truncated_continuation(row: dict[str, Any], truncated_prefix: str, continuation: str,
                                  normal_answer: str | None) -> dict[str, Any]:
    """Scores the forced-truncation continuation against ground truth AND against the
    (already-known-correct, per the eligibility restriction) normal-condition answer."""
    full_text = truncated_prefix + continuation
    ans, valid = _extract_answer(full_text)
    truncated_answer = ans if valid else None
    truncated_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
    matches_normal_answer = bool(truncated_answer is not None and normal_answer is not None
                                  and truncated_answer == normal_answer)
    return {'truncated_answer': truncated_answer, 'truncated_correct': truncated_correct,
            'matches_normal_answer': matches_normal_answer,
            'no_parseable_answer': truncated_answer is None}


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-subset (Heads-truth or Tails-truth, never pooled per task requirement 5)
    summary: normal accuracy (always 1.0 by construction, since the subset is
    restricted to normal-correct scenarios), truncated accuracy, and the
    consistency-vs-guessing breakdown."""
    n = len(rows)
    if n == 0:
        return {'n': 0, 'normal_accuracy': None, 'truncated_accuracy': None,
                'matches_normal_answer_rate': None, 'no_parseable_answer_rate': None}
    return {
        'n': n,
        'normal_accuracy': sum(1 for r in rows if r['final_answer_correct']) / n,
        'truncated_accuracy': sum(1 for r in rows if r['truncated_correct']) / n,
        'matches_normal_answer_rate': sum(1 for r in rows if r['matches_normal_answer']) / n,
        'no_parseable_answer_rate': sum(1 for r in rows if r['no_parseable_answer']) / n,
    }
