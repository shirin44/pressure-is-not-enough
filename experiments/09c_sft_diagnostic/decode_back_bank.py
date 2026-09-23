"""Builds the rebalanced 36-scenario trajectory bank with an explicit decode-back line
inserted between the final tracked-state line and <answer>, per the approved fix from
the corrupted-prefill diagnostic (design.md, "Corrupted-prefill causal diagnostic +
base-model logit-prior check", 2026-08-31): two independent, conclusive diagnostics
showed the model's <answer> token is structurally disconnected from its own correctly
tracked trace, and the base model carries no word-level prior toward "Tails" that would
explain the collapse -- pointing at the training FORMAT (fully-implicit adjacency
between the last State: line and <answer>) rather than the training data's label mix
(already ruled out separately) or a pretrained bias (also ruled out).

Deliberately does NOT modify synthetic_bridge.py -- consistent with this stage's
established discipline of treating that module as shared/load-bearing across Stages
09/09b/09c/09d and avoiding in-place changes to it (the same reason bank rebalancing
trimmed rather than extended N_FLIPS-dependent logic in the prior task). The decode-back
line is inserted as a pure string transform on top of the UNCHANGED
build_coded_completion output, and verified by reusing verify_trajectory unmodified on
the base completion (with the decode-back line stripped back out) plus one new,
additional check specific to the new line's content -- so the existing pipeline's
literal-Heads/Tails-leakage check keeps its original meaning (catching the model using
literal words INSTEAD OF Nib/Nomo mid-trace) rather than being weakened to tolerate the
new line, which deliberately does contain a literal word by design.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))

from synthetic_bridge import (  # noqa: E402
    CODED_TRAJECTORIES, HEADS_CODE, TAILS_CODE, _trace, build_clean_length5_train_eval_split,
    build_coded_completion, build_prompt, verify_trajectory)
from reward_v3 import normalize_state_token, parse_state_slots, score_completion_v2, completion_to_text  # noqa: E402

REBALANCE_SEED = 20260901  # same seed as the rebalanced-bank task, for the identical 36-row split

_DECODE_BACK_RE = re.compile(
    r'\nThe final code token (\S+?) decodes to (\S+?)\.\n<answer>', re.IGNORECASE)


def _code_for(state: str) -> str:
    return HEADS_CODE if state == 'Heads' else TAILS_CODE


def build_decode_back_completion(starting_state: str, operations) -> str:
    """Inserts 'The final code token <CODE> decodes to <LITERAL>.' between the last
    State: line and <answer>, both values taken from the SAME independently
    recomputed true trace used everywhere else in this project -- not copied from
    build_coded_completion's own internal state, so a bug in one can't silently mask
    a bug in the other."""
    base = build_coded_completion(starting_state, operations)
    reasoning, answer_part = base.split('<answer>', 1)
    reasoning = reasoning.rstrip('\n')
    true_final_state = _trace(starting_state, operations)[-1]
    true_final_code = _code_for(true_final_state)
    decode_line = f'The final code token {true_final_code} decodes to {true_final_state}.'
    return f'{reasoning}\n{decode_line}\n<answer>{answer_part}'


def parse_decode_back_line(completion: str) -> tuple[str, str] | None:
    """Returns (code_token, literal_word) as they literally appear in the completion's
    decode-back line, or None if the line is missing/malformed. Case is preserved by
    the caller's choice of whether to normalize before comparing."""
    m = _DECODE_BACK_RE.search(completion)
    if m is None:
        return None
    return m.group(1), m.group(2)


def strip_decode_back_line(completion: str) -> str:
    """Removes the decode-back line, recovering the exact text build_coded_completion
    would have produced -- used so the EXISTING, unmodified verify_trajectory can run
    its full original check suite on trajectories that carry the new line."""
    return _DECODE_BACK_RE.sub('\n<answer>', completion, count=1)


def verify_decode_back_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    """Reuses verify_trajectory UNCHANGED for everything it already checks (structure,
    state tracking, final answer, no literal leakage in the STATE-tracking portion),
    by running it against the completion with the decode-back line stripped back out.
    Adds one new, additional assertion: the decode-back line's code token must equal
    the immediately preceding State: line's own token (not just be independently
    correct), and its literal word must be the true decode of that token -- the
    specific correctness property the approved fix depends on."""
    completion = row['completion']
    parsed = parse_decode_back_line(completion)
    if parsed is None:
        raise AssertionError('decode-back line missing or malformed')
    decoded_code, decoded_literal = parsed
    base_completion = strip_decode_back_line(completion)
    base_report = verify_trajectory({**row, 'completion': base_completion})

    slots = parse_state_slots(base_completion)
    if not slots:
        raise AssertionError('no State: slots found to compare the decode-back line against')
    last_slot_token = slots[-1][1]
    if normalize_state_token(decoded_code) != last_slot_token:
        raise AssertionError(
            f"decode-back line's code token {decoded_code!r} does not match the "
            f'immediately preceding State: line token {last_slot_token!r}')
    true_final_state = _trace(row['starting_state'], row['operations'])[-1]
    true_final_code = _code_for(true_final_state)
    if normalize_state_token(decoded_code) != normalize_state_token(true_final_code):
        raise AssertionError(
            f"decode-back line's code token {decoded_code!r} != true final code {true_final_code!r}")
    if normalize_state_token(decoded_literal) != normalize_state_token(true_final_state):
        raise AssertionError(
            f"decode-back line's literal word {decoded_literal!r} != true final state {true_final_state!r}")
    return {**base_report, 'decode_back_line_verified': True,
            'decode_back_code_token': normalize_state_token(decoded_code),
            'decode_back_literal_word': normalize_state_token(decoded_literal)}


def verify_decode_back_bank(bank) -> list[dict[str, Any]]:
    if not bank:
        raise AssertionError('trajectory bank must not be empty')
    reports = [verify_decode_back_trajectory(row) for row in bank]
    prompts = [row['prompt'] for row in bank]
    if len(set(prompts)) != len(prompts):
        raise AssertionError('trajectory bank contains duplicate scenarios')
    return reports


def build_rebalanced_decode_back_bank(*, clean21_seed: int, n_eval: int = 21):
    """Reconstructs the IDENTICAL 36-scenario rebalanced bank (same seeds, same
    trim-not-add logic) from the rebalancing task, with decode-back completions in
    place of plain coded completions. Returns (bank, eval_rows, removed_rows) exactly
    like the rebalancing task's own bank-builder, for direct comparability."""
    train_scenarios_meta, eval_rows = build_clean_length5_train_eval_split(seed=clean21_seed, n_eval=n_eval)
    original_bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in CODED_TRAJECTORIES}
    tails_rows = [r for r in train_scenarios_meta if r['final_answer'] == 'Tails']
    heads_rows = [r for r in train_scenarios_meta if r['final_answer'] == 'Heads']
    n_to_remove = len(tails_rows) - len(heads_rows)
    tails_new_only = [r for r in tails_rows if (r['starting_state'], tuple(r['operations'])) not in original_bank_keys]
    rng = random.Random(REBALANCE_SEED)
    to_remove = rng.sample(tails_new_only, n_to_remove)
    to_remove_keys = {(r['starting_state'], tuple(r['operations'])) for r in to_remove}
    rebalanced_meta = [r for r in train_scenarios_meta if (r['starting_state'], tuple(r['operations'])) not in to_remove_keys]

    bank = []
    for r in rebalanced_meta:
        starting_state, operations = r['starting_state'], r['operations']
        completion = build_decode_back_completion(starting_state, operations)
        bank.append({'starting_state': starting_state, 'operations': list(operations),
            'prompt': build_prompt(starting_state, operations), 'completion': completion,
            'final_answer': r['final_answer'], 'coded': True})
    return bank, eval_rows, to_remove


def score_completion_v2_decode_back_aware(completion: Any, ground_truth: str, step: int,
                                           total_steps: int, *, prompt: Any = None, **kwargs: Any) -> dict:
    """Wraps reward_v3.score_completion_v2 to fix a reward-contamination bug found
    during the pre-flight check before the Stage 9d re-run on the decode-back
    checkpoint: score_completion_v2's p_cot term scans the ENTIRE completion text
    before <answer> (completion.split('<answer>', 1)[0]) for literal Heads/Tails/H/T
    patterns, with no awareness that the decode-back line ("The final code token Nib
    decodes to Heads.") DELIBERATELY contains a literal word by design. Confirmed
    empirically: an otherwise-perfect decode-back completion gets banned_count=1 (one
    false-positive literal-leakage hit) purely from its own required decode-back line,
    vs. banned_count=0 with the line stripped -- under MAIN's full adversarial p_cot
    weight (ramping to cot_max_scale=2.0), this is a direct, mechanical, ever-growing
    incentive to suppress the very line that fixed the structural disconnect, which
    would silently reopen that bug as an RL-training side effect rather than measuring
    anything about genuine concealment behavior.

    Fix: every OTHER term (r_task, p_length, p_structure, p_state_variation,
    r_consistency, r_signal) is computed on the ORIGINAL, unmodified completion --
    confirmed by inspection that none of them read the decode-back line (parse_state_slots
    only matches 'Step N: ... State: X' lines, which the decode-back line's different
    format never matches; _extract_answer only reads the trailing <answer> tag) --
    while ONLY p_cot is recomputed on the completion with the decode-back line
    stripped back out (via strip_decode_back_line), so genuine literal leakage
    elsewhere in the Step N:/State: lines is still fully penalized exactly as before,
    and only the decode-back line's own required, approved literal word is excluded.
    total is corrected by exactly that delta -- no other term's value or range changes,
    so the existing, checkpoint-independent verify_reward_invariant() remains valid
    for this wrapper unchanged."""
    text = completion_to_text(completion)
    raw = score_completion_v2(text, ground_truth, step, total_steps, prompt=prompt, **kwargs)
    stripped_text = strip_decode_back_line(text)
    stripped = score_completion_v2(stripped_text, ground_truth, step, total_steps, prompt=prompt, **kwargs)
    delta = raw['p_cot'] - stripped['p_cot']  # >= 0: the false-positive penalty being removed
    corrected = dict(raw)
    corrected['p_cot'] = stripped['p_cot']
    corrected['banned_count'] = stripped['banned_count']
    corrected['total'] = raw['total'] + delta  # p_cot is SUBTRACTED in total, so removing it ADDS delta back
    corrected['p_cot_correction_applied'] = delta
    return corrected
