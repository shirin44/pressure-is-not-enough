"""Stage 11, Part A: locate and load the evaluation set.

Sources (all already generated, hash-verified evidence -- NO new generation here):

1. PRE-RL (SFT-only checkpoint, `stage9e-llama-sft-code-word-v1`), trained code pair
   Bek/Ner: `experiments/09e_same_different_llama/aws_runs/stage9e-llama-sft-code-word-v1/
   stage9e_llama_sft_code_word.json`, key `tier_b_heldout_same_pair.samples` (21 held-out
   scenarios, `intermediate_tracking_correct=True, final_answer_correct=True` on all 21 --
   verified below, not assumed).

2. POST-RL (MAIN seed 43, `stage9e-llama-rl-main-v3`), same trained pair Bek/Ner:
   `experiments/09e_same_different_llama/aws_runs/stage9e-llama-rl-main-v3/
   stage9e_llama_rl_main.json`, key `result.milestones[-1].samples` (step 150, the final
   milestone; 21 held-out scenarios, `genuine_correct_rate=1.0`,
   `intermediate_tracking_accuracy=1.0`).

   This is NOT independently re-verified against the causal-flip evidence by assumption --
   it was cross-checked directly (same session) that
   `stage9e-llama-causal-flip-post-rl-seed43-v1`'s 21 `flip_at_step=2` intervention records
   have IDENTICAL (starting_state, operations) pairs, in the SAME order, as these 21
   milestone samples -- i.e. this file is confirmed to be the actual source the post-RL
   causal-flip test itself drew its "eligible" natural completions from, not a merely
   plausible substitute.

3. HELD-OUT DIFFERENT PAIR (Jub/Kag), PRE-RL ONLY -- no post-RL equivalent exists (checked:
   MAIN seed 43's own milestones carry only one label, `main_code_word`, tracking Bek/Ner;
   the RL stage never re-evaluated the Jub/Kag pair at any step -- see
   `experiments/09e_same_different_llama/llama_code_word_rl.py`. Reported as NOT AVAILABLE
   for post-RL, not silently substituted or skipped without comment.):
   same SFT evidence file, key `tier_c_heldout_different_pair.samples` (21 scenarios,
   `fully_correct_rate=0.524` -- only 11/21 fully correct; the SFT checkpoint was never
   trained on this pair, so the lower correctness rate is expected and reported as-is,
   not filtered away).

Each loaded trace is annotated with the ground-truth physical state sequence, computed
INDEPENDENTLY from (starting_state, operations) via `synthetic_bridge._trace` -- never
read from the completion text itself -- and the judge-facing trace text is exactly the
model's own `completion` field, verified clean of any literal Heads/Tails leakage by the
project's own existing scanner (`same_different_leakage_audit.check_completion_for_leakage`)
before being used anywhere in Part B/C.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE09E_DIR = REPO_ROOT / 'experiments' / '09e_same_different_llama'
STAGE09_DIR = REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
sys.path.insert(0, str(STAGE09E_DIR))
sys.path.insert(0, str(STAGE09_DIR))

from synthetic_bridge import _trace  # noqa: E402
from same_different_leakage_audit import check_completion_for_leakage  # noqa: E402
from code_word_answer_bank import STATE_FOR_CODE as TRAINED_STATE_FOR_CODE  # noqa: E402

SFT_EVIDENCE_PATH = STAGE09E_DIR / 'aws_runs' / 'stage9e-llama-sft-code-word-v1' / 'stage9e_llama_sft_code_word.json'
MAIN_SEED43_EVIDENCE_PATH = STAGE09E_DIR / 'aws_runs' / 'stage9e-llama-rl-main-v3' / 'stage9e_llama_rl_main.json'

# Jub/Kag: the held-out different pair, mapping stated in the SFT evidence's own config
# (['Jub', 'Kag'] == [heads_code, tails_code], matching the CODE_FOR convention).
HELDOUT_STATE_FOR_CODE = {'jub': 'Heads', 'kag': 'Tails'}


def _ground_truth_codes(starting_state: str, operations: list, state_for_code: dict) -> list[str]:
    """Ground-truth CODE sequence, computed independently from (starting_state,
    operations) via _trace -- never read back from the completion."""
    code_for_state = {v: k for k, v in state_for_code.items()}
    true_states = _trace(starting_state, operations)
    return [code_for_state[s].capitalize() for s in true_states]


def _build_record(raw: dict, group: str, state_for_code: dict) -> dict[str, Any]:
    completion = raw['completion']
    ground_truth_states = _trace(raw['starting_state'], raw['operations'])
    ground_truth_codes = _ground_truth_codes(raw['starting_state'], raw['operations'], state_for_code)
    leakage = check_completion_for_leakage(completion)
    return {
        'group': group,
        'starting_state': raw['starting_state'],  # kept for provenance/scoring only, NEVER shown to a judge
        'operations': list(raw['operations']),  # same -- provenance/scoring only
        'trace_text': completion,  # exactly what a judge sees (Bek/Ner or Jub/Kag, never Heads/Tails)
        'ground_truth_state_sequence': ground_truth_states,  # Heads/Tails, for scoring only
        'ground_truth_code_sequence': ground_truth_codes,  # Bek/Ner or Jub/Kag, for scoring only
        'final_answer_correct': raw.get('final_answer_correct'),
        'intermediate_tracking_correct': raw.get('intermediate_tracking_correct',
                                                   raw.get('score', {}).get('per_slot_correct')),
        'leakage_scan': leakage,
    }


def load_pre_rl_trained_pair() -> list[dict]:
    """21 pre-RL (SFT-only checkpoint) held-out completions, trained pair Bek/Ner."""
    data = json.loads(SFT_EVIDENCE_PATH.read_text())
    samples = data['tier_b_heldout_same_pair']['samples']
    eligible = [s for s in samples if s['intermediate_tracking_correct'] and s['final_answer_correct']]
    assert len(eligible) == 21, f'expected 21 eligible pre-RL trained-pair traces, got {len(eligible)}'
    return [_build_record(s, 'pre_rl_trained_pair', TRAINED_STATE_FOR_CODE) for s in eligible]


def load_post_rl_trained_pair() -> list[dict]:
    """21 post-RL (MAIN seed 43, step 150) held-out completions, trained pair Bek/Ner."""
    data = json.loads(MAIN_SEED43_EVIDENCE_PATH.read_text())
    final_milestone = data['result']['milestones'][-1]
    assert final_milestone['step'] == 150
    samples = final_milestone['samples']
    assert len(samples) == 21, f'expected 21 post-RL trained-pair traces, got {len(samples)}'
    return [_build_record(s, 'post_rl_trained_pair', TRAINED_STATE_FOR_CODE) for s in samples]


def load_pre_rl_heldout_pair() -> list[dict]:
    """21 pre-RL (SFT-only checkpoint) completions on the held-out DIFFERENT pair
    (Jub/Kag), never trained on. No post-RL equivalent exists (see module docstring)."""
    data = json.loads(SFT_EVIDENCE_PATH.read_text())
    samples = data['tier_c_heldout_different_pair']['samples']
    assert len(samples) == 21, f'expected 21 pre-RL held-out-pair traces, got {len(samples)}'
    records = []
    for s in samples:
        rec = _build_record(
            {**s, 'final_answer_correct': s['score']['final_answer_correct'],
             'intermediate_tracking_correct': s['score']['per_slot_correct']},
            'pre_rl_heldout_pair', HELDOUT_STATE_FOR_CODE,
        )
        records.append(rec)
    return records


def load_all() -> dict[str, list[dict]]:
    return {
        'pre_rl_trained_pair': load_pre_rl_trained_pair(),
        'post_rl_trained_pair': load_post_rl_trained_pair(),
        'pre_rl_heldout_pair': load_pre_rl_heldout_pair(),
    }


if __name__ == '__main__':
    groups = load_all()
    total = 0
    for name, records in groups.items():
        n_clean = sum(1 for r in records if r['leakage_scan']['clean'])
        n_correct = sum(1 for r in records if r['final_answer_correct'])
        print(f'{name}: n={len(records)}, leakage_clean={n_clean}/{len(records)}, '
              f'final_answer_correct={n_correct}/{len(records)}')
        total += len(records)
    print(f'\nTOTAL usable traces: {total}')
