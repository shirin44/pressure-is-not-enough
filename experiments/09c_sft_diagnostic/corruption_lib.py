"""Pure-logic (CPU-only, no model/GPU imports) helpers for the corrupted-prefill causal
diagnostic on the rebalanced-bank SFT checkpoint (stage9c-sft-rebalanced-v1). Adapts
Stage 8's exact corrupted-state-prefill methodology
(experiments/08_load_bearing_and_exploration_audit/aws_runs/exp1_load_bearing_regime_v1/run_script.py,
parse_state_slots_with_spans / LITERAL_MAP / the condition-(c) corruption loop) from the
literal Heads/Tails domain to this stage's Nib/Nomo coded domain, and from Stage 8's
middle-step corruption target to THIS task's specified target: the LAST tracked-state
token (the step immediately preceding the <answer> tag).

Two eligible-sample groups are selected, not just the one literally named in the task:
  - group 'heads_to_tails': true final state Heads, model's own trace correctly tracked
    (ends in Nib) -- what the task's instruction 1 literally asks for. Note this
    direction is NOT decisive on its own: the model's already-known unconditional
    default is "Tails" regardless of input, so both the tracking-based-default-to-Tails
    hypothesis and the pure-structural-disconnect hypothesis predict the SAME outcome
    (Tails) here.
  - group 'tails_to_heads': true final state Tails, correctly tracked (ends in Nomo),
    corrupted the opposite direction. This is the decisive group: the two hypotheses
    predict DIFFERENT outcomes (Heads if the prefix is causally read; still-Tails if
    structurally disconnected). Included per the task's own instruction 2 wording
    ("even when corrupted to say Heads"), which anticipates this direction too.
"""
from __future__ import annotations

import re

# Same regexes as reward_v3.py / Stage 8's run_script.py -- domain-agnostic (they match
# any single capitalized alphabetic token after "State:", literal or coded).
_STATE_LINE_RE = re.compile(r"^\s*Step\s+(\d+)\s*:\s*.*?\bState:\s*(.*?)$")
_STRICT_STATE_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z]{0,14}$")

CODE_LITERAL_MAP = {'nib': 'nomo', 'nomo': 'nib'}
CODE_TO_LITERAL_ANSWER = {'nib': 'heads', 'nomo': 'tails'}


def _normalize_strict_state_span(span: str) -> str | None:
    candidate = span.strip()
    if candidate.endswith(('.', ',')):
        candidate = candidate[:-1].rstrip()
    if not _STRICT_STATE_TOKEN_RE.fullmatch(candidate):
        return None
    return candidate.casefold()


def parse_state_slots_with_spans(text: str) -> list[dict]:
    """Like reward_v3.parse_state_slots, but also returns each line's (start, end)
    character offsets in the ORIGINAL text -- needed to truncate/corrupt at a specific
    step's line without disturbing anything before it."""
    reasoning = re.split(r"<answer>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    slots = []
    pos = 0
    for line in reasoning.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.rstrip('\n')
        m = _STATE_LINE_RE.fullmatch(stripped)
        if m:
            token = _normalize_strict_state_span(m.group(2))
            if token is not None:
                line_end = line_start + len(stripped)
                slots.append({'index': int(m.group(1)), 'token': token,
                              'line_start': line_start, 'line_end': line_end})
    return slots


def extract_answer(text: str) -> str | None:
    matches = list(re.finditer(
        r"<answer>\s*((?:(?!</?answer>).)*?)\s*</answer>\s*$", text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    answer = matches[-1].group(1).strip().casefold()
    return answer or None


def select_eligible_samples(tier_b_samples: list[dict]) -> dict[str, list[dict]]:
    """Splits Tier B held-out samples into the two corruption-eligible groups. A sample
    is eligible for a group if its ground-truth-verified intermediate tracking is fully
    correct (intermediate_tracking_correct) AND its true final answer matches the
    group's starting side. Both groups additionally require the completion to actually
    contain a parseable final state-slot line to corrupt."""
    groups: dict[str, list[dict]] = {'heads_to_tails': [], 'tails_to_heads': []}
    for row in tier_b_samples:
        if not row.get('intermediate_tracking_correct'):
            continue
        slots = parse_state_slots_with_spans(row['completion'])
        if not slots:
            continue
        last_slot = max(slots, key=lambda s: s['index'])
        if last_slot['token'] not in CODE_LITERAL_MAP:
            continue
        true_answer = row['final_answer'].strip().casefold() if 'final_answer' in row else None
        if true_answer is None:
            # Tier B rows store expected_tokens (ground-truth code sequence); derive the
            # true literal answer from its last entry instead.
            true_answer = CODE_TO_LITERAL_ANSWER.get(row['expected_tokens'][-1])
        if true_answer == 'heads' and last_slot['token'] == 'nib':
            groups['heads_to_tails'].append({**row, '_target_slot': last_slot})
        elif true_answer == 'tails' and last_slot['token'] == 'nomo':
            groups['tails_to_heads'].append({**row, '_target_slot': last_slot})
    return groups


def build_corrupted_prefix(chat_prefix: str, completion: str, target_slot: dict) -> tuple[str, str]:
    """Constructs the corrupted generation prefix: the chat-templated prompt opening,
    plus the completion truncated right after the target (final) state line, with that
    line's code token flipped to its opposite. Everything after the target line --
    including any existing <answer> tag -- is dropped, matching Stage 8's methodology.
    Returns (corrupted_prefix_text, corruption_direction_literal)."""
    original_token = target_slot['token']
    flipped_token = CODE_LITERAL_MAP[original_token]
    original_line = completion[target_slot['line_start']:target_slot['line_end']]
    state_kw_match = re.search(r'\bState:\s*', original_line)
    if state_kw_match is None:
        raise ValueError(f'target slot line has no "State:" keyword: {original_line!r}')
    corrupted_line = original_line[:state_kw_match.end()] + flipped_token.capitalize()
    corrupted_prefix = chat_prefix + completion[:target_slot['line_start']] + corrupted_line
    corruption_direction = CODE_TO_LITERAL_ANSWER[flipped_token]
    return corrupted_prefix, corruption_direction


def classify_outcome(new_answer: str | None, original_answer: str, corruption_direction: str) -> str:
    """Classifies a corrupted-prefill continuation's resulting answer against the two
    competing hypotheses. original_answer is the answer the (uncorrupted) rollout
    actually produced (per this project's evidence, always 'tails' in Tier B)."""
    if new_answer is None:
        return 'no_parseable_answer'
    if new_answer == corruption_direction:
        return 'tracks_corrupted_prefix'
    if new_answer == original_answer:
        return 'stays_at_original_answer'
    return 'other'
