"""Reusable, rigorous error-tracing analysis for any Stage 9d decode-back-seeded run's
evidence (2026-09-03, built after MAIN v3 diverged dramatically from v2 despite
identical seed/config/code -- see design.md). Applies the SAME root-cause tracing
methodology validated on v3 to any run's evidence, without assuming the mechanism
transfers: every wrong final answer is independently classified as a genuine
tracking error, a decode-back-line mistranslation, a decode-back structural
inconsistency (the line references a DIFFERENT code than the model's own last
tracked token -- a potential re-opening signal, not yet seen in this stage but not
assumed away either), or unparseable output.

Milestone samples in the RL evidence do NOT store starting_state/operations (unlike
the SFT evidence) -- but eval_rows is fully deterministic (same CLEAN21_SEED, same 21
scenarios, same order every time, since evaluate_and_track always iterates
zip(eval_rows, prompts, completions) in eval_rows' own fixed order), so ground truth
is recoverable by rebuilding eval_rows fresh and zipping it against each milestone's
samples by index position.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))

from synthetic_bridge import build_clean_length5_train_eval_split, _trace, HEADS_CODE, TAILS_CODE  # noqa: E402
from decode_back_bank import parse_decode_back_line  # noqa: E402
from reward_v3 import normalize_state_token  # noqa: E402

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'


def _eval_rows() -> list[dict[str, Any]]:
    import hashlib
    import json
    _, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
    sha = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
    if sha != EXPECTED_CLEAN21_SHA256:
        raise RuntimeError(f'eval_rows reconstruction mismatch: expected {EXPECTED_CLEAN21_SHA256}, got {sha}')
    return eval_rows


def classify_wrong_sample(sample: dict[str, Any], row: dict[str, Any]) -> str:
    """Classifies ONE wrong-final-answer sample against ground truth, independent of
    any assumption about which mechanism is responsible. Returns one of:
    'genuine_tracking_error', 'decode_back_mistranslation',
    'decode_back_structural_inconsistency', 'unparseable_or_other'."""
    true_states = _trace(row['starting_state'], row['operations'])
    expected_tokens = [normalize_state_token(HEADS_CODE if s == 'Heads' else TAILS_CODE) for s in true_states]
    actual_tokens = sample.get('nonliteral_tokens') or []
    if actual_tokens != expected_tokens:
        return 'genuine_tracking_error'
    parsed = parse_decode_back_line(sample['completion'])
    if parsed is None:
        return 'unparseable_or_other'
    decoded_code, _decoded_literal = parsed
    decoded_code_n = normalize_state_token(decoded_code)
    if not actual_tokens or decoded_code_n != actual_tokens[-1]:
        return 'decode_back_structural_inconsistency'
    return 'decode_back_mistranslation'


def analyze_run(evidence: dict[str, Any]) -> dict[str, Any]:
    """Full per-run analysis: decode_back_matches_own_trace_rate summary across all
    milestones, plus a rigorous classification of every wrong-final-answer sample."""
    rows = _eval_rows()
    milestones = evidence['result']['milestones']
    dbmr_values = [m['decode_back_matches_own_trace_rate'] for m in milestones]
    genuine_correct_values = [m['genuine_correct_among_structural_nonliteral'] for m in milestones]

    from collections import Counter
    taxonomy = Counter()
    total_wrong = 0
    for m in milestones:
        samples = m['samples']
        if len(samples) != len(rows):
            raise RuntimeError(f"milestone step {m['step']} has {len(samples)} samples, expected {len(rows)}")
        for sample, row in zip(samples, rows):
            if sample['final_answer_correct']:
                continue
            total_wrong += 1
            taxonomy[classify_wrong_sample(sample, row)] += 1

    return {
        'terminal_step': evidence['result']['terminal_step'],
        'hard_stop': evidence['result']['hard_stop'],
        'n_milestones': len(milestones),
        'decode_back_matches_own_trace_rate': {
            'min': min(dbmr_values), 'max': max(dbmr_values),
            'all_perfect': all(x == 1.0 for x in dbmr_values),
        },
        'genuine_correct_among_nonliteral': {
            'mean': sum(genuine_correct_values) / len(genuine_correct_values),
            'min': min(genuine_correct_values), 'max': max(genuine_correct_values),
            'final': genuine_correct_values[-1],
        },
        'total_wrong_samples': total_wrong,
        'error_taxonomy': dict(taxonomy),
        'error_taxonomy_fractions': ({k: v / total_wrong for k, v in taxonomy.items()} if total_wrong else {}),
    }
