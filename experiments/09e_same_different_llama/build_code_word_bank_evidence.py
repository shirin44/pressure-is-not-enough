"""Stage 9e, direct code-word answer design: builds and persists the full verified
trajectory bank, WITH token-position metadata for every entry (task item 5) and a
leakage-audit pass (task item 6). CPU-only (tokenizer only, no model, no GPU) -- run
locally, matching this task's own explicit "no SFT seeding" scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from code_word_answer_bank import (
    ALL_SCENARIOS, BANK_KEYS, HEADS_CODE, TAILS_CODE,
    build_code_word_completion, build_code_word_prompt, build_code_word_train_eval_split,
    verify_code_word_trajectory,
)
from code_word_position_logging import log_code_word_positions
from same_different_leakage_audit import audit_bank_for_leakage

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
CODE_WORD_SEED = 20260907
N_EVAL = 21

print('===== LOAD TOKENIZER (CPU-only, no model, no GPU) =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('===== VERIFY ALL 64 SCENARIOS IN THE FULL LENGTH-5 UNIVERSE =====')
all_verification = [verify_code_word_trajectory(s, list(ops)) for s, ops in ALL_SCENARIOS]
n_verified = sum(1 for v in all_verification if v['all_verified'])
print({'n_scenarios': len(ALL_SCENARIOS), 'n_verified': n_verified,
       'all_verified': n_verified == len(ALL_SCENARIOS)})
if n_verified != len(ALL_SCENARIOS):
    raise RuntimeError('Not every scenario verified -- stopping before persisting anything.')

print('===== BUILD STRATIFIED, BALANCED TRAIN/EVAL SPLIT =====')
train_rows, eval_rows = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=N_EVAL)
print({'train_n': len(train_rows), 'eval_n': len(eval_rows)})


def materialize_row(row):
    starting_state, operations = row['starting_state'], row['operations']
    prompt = build_code_word_prompt(starting_state, operations)
    completion = build_code_word_completion(starting_state, operations)
    verification = verify_code_word_trajectory(starting_state, operations)
    positions = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
    return {
        **row,
        'prompt': prompt,
        'completion': completion,
        'verification': {k: v for k, v in verification.items() if k != 'completion'},
        'token_positions': {k: v for k, v in positions.items() if k != 'full_text'},
    }


print('===== MATERIALIZE FULL BANK (prompt, completion, verification, token positions) =====')
train_materialized = [materialize_row(r) for r in train_rows]
eval_materialized = [materialize_row(r) for r in eval_rows]

print('===== LEAKAGE AUDIT (literal words + abbreviations, no synonym classifier -- Decision 3) =====')
all_completions = [r['completion'] for r in train_materialized] + [r['completion'] for r in eval_materialized]
leakage_result = audit_bank_for_leakage(all_completions)
print({'n_total': leakage_result['n_total'], 'n_clean': leakage_result['n_clean'],
       'all_clean': leakage_result['all_clean']})
if not leakage_result['all_clean']:
    raise RuntimeError(f"Leakage found: {leakage_result['failures']}")

print('===== POSITION-LOGGING SANITY CHECK ACROSS THE FULL BANK =====')
n_single_token_confirmed = sum(
    1 for r in train_materialized + eval_materialized
    if r['token_positions']['all_occurrences_single_token_confirmed'])
n_occurrence_count_matches = sum(
    1 for r in train_materialized + eval_materialized
    if r['token_positions']['occurrence_count_matches_expected'])
n_total = len(train_materialized) + len(eval_materialized)
print({'n_total': n_total, 'n_single_token_confirmed': n_single_token_confirmed,
       'n_occurrence_count_matches': n_occurrence_count_matches})
if n_single_token_confirmed != n_total or n_occurrence_count_matches != n_total:
    raise RuntimeError('Position-logging sanity check failed for at least one bank entry.')

print('===== SAVE EVIDENCE =====')
OUTPUT_DIR = Path(__file__).resolve().parent / 'aws_runs' / 'stage9e-code-word-bank-v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
report = {
    'config': {'model': MODEL_NAME, 'code_word_seed': CODE_WORD_SEED, 'n_eval': N_EVAL,
               'heads_code': HEADS_CODE, 'tails_code': TAILS_CODE,
               'bank_keys_n': len(BANK_KEYS)},
    'full_universe_verification': {'n_scenarios': len(ALL_SCENARIOS), 'n_verified': n_verified},
    'leakage_audit': {k: v for k, v in leakage_result.items()},
    'train_rows': train_materialized,
    'eval_rows': eval_materialized,
}
EVENT_LOG = OUTPUT_DIR / 'stage9e_code_word_bank.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
sha = hashlib.sha256(EVENT_LOG.read_bytes()).hexdigest()
print({'evidence_path': str(EVENT_LOG), 'sha256': sha})
print('STAGE 9E CODE-WORD BANK BUILD COMPLETE.')
