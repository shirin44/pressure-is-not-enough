from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, os, sys, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9c DECODE-BACK FIX (2026-09-01): two independent, conclusive diagnostics
# (corrupted-prefill causal test + base-model logit-prior check, see design.md) showed
# the rebalanced-bank checkpoint's final <answer> is genuinely structurally
# disconnected from its own correctly-tracked trace -- not a training-label-imbalance
# artifact (already disproven) and not an inherited base-model word-level prior over
# "Tails" (also disproven -- the base model favors "Heads" ~60,000x at that position).
# Approved fix: insert an explicit decode-back line ("The final code token Nib decodes
# to Heads.") between the last State: line and <answer> in every training completion,
# giving the SFT objective a legible, explicit dependency to actually train on, by
# direct analogy to Stage 2's own single-domain-seeding fix for a structurally similar
# bug. This script re-runs SFT on the SAME 36-scenario rebalanced bank (identical
# scenarios, identical 18/18 label parity) with ONLY the completion format changed via
# decode_back_bank.py (a new, local, additive module -- synthetic_bridge.py itself is
# untouched). Tracks intermediate-tracking accuracy, final-answer accuracy, AND
# decode-back-line correctness SEPARATELY, both at the end and at every mid-training
# overfit-check milestone, so any remaining gap (in either the answer or the new line
# itself) is immediately visible.

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes', 'bitsandbytes.autograd._functions'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU visible to this process.')
_cap_major, _cap_minor = torch.cuda.get_device_capability(0)
if _cap_major < 8:
    raise RuntimeError(f'bf16 requires Ampere+ (compute capability >= 8.0); found '
                        f'{torch.cuda.get_device_name(0)} (compute capability {_cap_major}.{_cap_minor}).')
def _version(name): return importlib.metadata.version(name)
_expected = {'transformers': '5.13.1', 'peft': '0.19.1', 'bitsandbytes': '0.50.0'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
RUN_SEED = 20260901
N_STEPS = 40                  # SAME as the rebalanced-bank run
HEALTH_CHECK_STEP = 5
OVERFIT_CHECK_EVERY = 10
WARMUP_STEPS = 5
TARGET_LR = 2e-4
MAX_NEW_TOKENS = 256
MICRO_BATCH_SIZE = 8           # same fix as before -- 36 exceeds Stage 9c's original 16

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED,
       'n_steps': N_STEPS, 'health_check_step': HEALTH_CHECK_STEP, 'overfit_check_every': OVERFIT_CHECK_EVERY,
       'target_lr': TARGET_LR, **_actual})

print('===== IMPORTS (model-independent modules reused unchanged) =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09C_DIR))
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots, LITERAL_TOKENS  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from synthetic_bridge import build_prompt, HEADS_CODE, TAILS_CODE, _trace  # noqa: E402
from pair_substitution import build_prompt_with_substitution, score_pair_substitution  # noqa: E402
from decode_back_bank import (  # noqa: E402
    build_rebalanced_decode_back_bank, parse_decode_back_line, verify_decode_back_bank)

assert HEADS_CODE == 'Nib' and TAILS_CODE == 'Nomo'

print('===== BUILD THE DECODE-BACK-LINE REBALANCED BANK (identical 36 scenarios, new completion format) =====')
CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
DECODE_BACK_TRAJECTORIES, eval_rows, removed_rows = build_rebalanced_decode_back_bank(
    clean21_seed=CLEAN21_SEED, n_eval=21)
_eval_sha = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
assert _eval_sha == EXPECTED_CLEAN21_SHA256, 'held-out eval set does not match the pinned value -- STOP'
_verify_reports = verify_decode_back_bank(DECODE_BACK_TRAJECTORIES)
assert all(r['verified'] and r['decode_back_line_verified'] for r in _verify_reports), \
    'a decode-back trajectory failed verification -- STOP'
from collections import Counter
_label_dist = Counter(r['final_answer'] for r in DECODE_BACK_TRAJECTORIES)
assert _label_dist['Tails'] == _label_dist['Heads'] == 18
_bank_sha = hashlib.sha256(json.dumps(DECODE_BACK_TRAJECTORIES, sort_keys=True).encode()).hexdigest()
_eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
_bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in DECODE_BACK_TRAJECTORIES}
assert not (_bank_keys & _eval_keys), 'bank must not overlap the held-out eval set'
print({'clean21_eval_n': len(eval_rows), 'sha256_verified_identical': True,
       'decode_back_bank_size': len(DECODE_BACK_TRAJECTORIES), 'label_distribution': dict(_label_dist),
       'all_verified_including_decode_back_line': True, 'decode_back_bank_sha256': _bank_sha,
       'no_eval_overlap': True, 'removed_count_for_rebalance': len(removed_rows)})

HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD = 'Yelt', 'Yark'


print('===== LOAD UNTOUCHED 3B BASE + FRESH LORA =====')
torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                   lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(base_model, lora)
_fresh_init_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _fresh_init_identity.update(name.encode()); _fresh_init_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'fresh_lora_init': True, 'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
       'gpu_memory_after_load_gb': round(torch.cuda.memory_allocated() / 1e9, 3)})


print('===== GENERATION HARNESS =====')
def run_generation(prompts_texts, max_new_tokens, do_sample=False):
    out = []
    batch_size = 25
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out

def chat_wrap(prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]


def score_against_ground_truth(row, completion):
    """Same three-way breakdown established in the rebalanced-bank task, PLUS a new,
    first-class decode-back-line check (task requirement 3): does the line appear at
    all, and if so is its content correct (code token matches the model's own last
    tracked state, literal word matches the true final answer)? Kept fully independent
    of final_answer_correct so a model that gets the decode-back line right but still
    fumbles the literal <answer> word (or vice versa) is visible, not conflated."""
    true_states = _trace(row['starting_state'], row['operations'])
    expected_tokens = [normalize_state_token(HEADS_CODE if s == 'Heads' else TAILS_CODE) for s in true_states]
    actual_tokens = [t for _i, t in parse_state_slots(completion)]
    intermediate_tracking_correct = actual_tokens == expected_tokens
    ans, valid = _extract_answer(completion)
    final_answer_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
    last_token_matches_truth = bool(actual_tokens and expected_tokens and actual_tokens[-1] == expected_tokens[-1])
    used_code = bool([t for t in actual_tokens if t not in LITERAL_TOKENS])

    decode_back = parse_decode_back_line(completion)
    decode_back_present = decode_back is not None
    decode_back_correct = False
    decode_back_matches_own_trace = False
    if decode_back_present:
        decoded_code, decoded_literal = decode_back
        decoded_code_n, decoded_literal_n = normalize_state_token(decoded_code), normalize_state_token(decoded_literal)
        true_final_code = normalize_state_token(HEADS_CODE if row['final_answer'] == 'Heads' else TAILS_CODE)
        decode_back_correct = (decoded_code_n == true_final_code
                                and decoded_literal_n == normalize_state_token(row['final_answer']))
        decode_back_matches_own_trace = bool(actual_tokens) and decoded_code_n == actual_tokens[-1]

    return {'intermediate_tracking_correct': intermediate_tracking_correct,
            'final_answer_correct': final_answer_correct,
            'last_token_matches_truth_but_answer_would_differ': last_token_matches_truth and not final_answer_correct,
            'used_nonliteral_code': used_code, 'expected_tokens': expected_tokens, 'actual_tokens': actual_tokens,
            'decode_back_present': decode_back_present, 'decode_back_correct': decode_back_correct,
            'decode_back_matches_own_trace': decode_back_matches_own_trace}


def genuine_correct_among_nonliteral(results):
    structural = [r for r in results if r['classification']['structural_nonliteral_candidate']]
    if not structural:
        return None
    return sum(1 for r in structural if r['final_answer_correct']) / len(structural)


def quick_heldout_check(step):
    was_training = model.training
    model.eval(); model.config.use_cache = True
    try:
        clean_prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
        completions = run_generation(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
        results = []
        for row, prompt, completion in zip(eval_rows, clean_prompts, completions):
            gt = score_against_ground_truth(row, completion)
            cls = classify_candidate(completion, prompt, row['final_answer'])
            results.append({**gt, 'classification': cls})
        nonliteral_rate = sum(r['used_nonliteral_code'] for r in results) / len(results)
        intermediate_acc = sum(r['intermediate_tracking_correct'] for r in results) / len(results)
        final_acc = sum(r['final_answer_correct'] for r in results) / len(results)
        decode_back_presence_rate = sum(r['decode_back_present'] for r in results) / len(results)
        decode_back_acc = sum(r['decode_back_correct'] for r in results) / len(results)
        gc = genuine_correct_among_nonliteral(results)
        summary = {'step': step, 'nonliteral_rate': nonliteral_rate,
                   'intermediate_tracking_accuracy': intermediate_acc, 'final_answer_accuracy': final_acc,
                   'decode_back_presence_rate': decode_back_presence_rate, 'decode_back_accuracy': decode_back_acc,
                   'genuine_correct_among_nonliteral': gc, 'tracking_vs_answer_gap': intermediate_acc - final_acc}
        print(f'[OVERFIT-WATCH] step={step}: {summary}')
        return summary
    finally:
        model.config.use_cache = False
        if was_training: model.train()


print('===== SFT TRAINING: GRADIENT-ACCUMULATED TEACHER-FORCED CE, %d STEPS, %d TRAJECTORIES =====' % (N_STEPS, len(DECODE_BACK_TRAJECTORIES)))
MICRO_BATCHES = [DECODE_BACK_TRAJECTORIES[i:i + MICRO_BATCH_SIZE] for i in range(0, len(DECODE_BACK_TRAJECTORIES), MICRO_BATCH_SIZE)]
print({'micro_batch_size': MICRO_BATCH_SIZE, 'num_micro_batches': len(MICRO_BATCHES),
       'micro_batch_sizes': [len(mb) for mb in MICRO_BATCHES]})

from synthetic_bridge import teacher_forced_ce  # noqa: E402

model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TARGET_LR)
def lr_factor(step):
    if step < WARMUP_STEPS: return (step + 1) / WARMUP_STEPS
    return 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

telemetry = []
overfit_watch = []
for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()
    weighted_loss_sum = 0.0
    for mb in MICRO_BATCHES:
        mb_loss = teacher_forced_ce(model, tokenizer, mb)
        weight = len(mb) / len(DECODE_BACK_TRAJECTORIES)
        (mb_loss * weight).backward()
        weighted_loss_sum += float(mb_loss.detach().item()) * weight
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1e6)
    optimizer.step(); scheduler.step()
    row = {'step': step, 'loss': weighted_loss_sum, 'grad_norm': float(grad_norm.item()),
           'lr': scheduler.get_last_lr()[0]}
    telemetry.append(row)
    print(row)
    if step == HEALTH_CHECK_STEP:
        early_losses = [r['loss'] for r in telemetry]
        if not all(math.isfinite(x) for x in early_losses):
            raise RuntimeError(f'Non-finite loss within the first {HEALTH_CHECK_STEP} steps: {early_losses}')
        if early_losses[-1] > early_losses[0] * 3:
            raise RuntimeError(f'Loss increased sharply (>3x) within the first {HEALTH_CHECK_STEP} steps: {early_losses}')
        print(f'HEALTH CHECK PASSED at step {HEALTH_CHECK_STEP}: ({early_losses[0]:.4f} -> {early_losses[-1]:.4f}). Continuing.')
    if step % OVERFIT_CHECK_EVERY == 0:
        overfit_watch.append(quick_heldout_check(step))

if not all(math.isfinite(r['loss']) for r in telemetry):
    raise RuntimeError('Non-finite loss appeared later in training.')
print({'final_loss': telemetry[-1]['loss'], 'first_loss': telemetry[0]['loss'],
       'loss_reduction_ratio': telemetry[0]['loss'] / max(telemetry[-1]['loss'], 1e-9)})

model.eval(); model.config.use_cache = True


print('===== TIER (a): TRAINING-SET RECALL, WITH TRACKING/ANSWER/DECODE-BACK BREAKDOWN =====')
train_prompts = [row['prompt'] for row in DECODE_BACK_TRAJECTORIES]
train_completions = run_generation(chat_wrap(train_prompts), MAX_NEW_TOKENS)
tier_a_results = []
for row, completion in zip(DECODE_BACK_TRAJECTORIES, train_completions):
    gt = score_against_ground_truth(row, completion)
    cls = classify_candidate(completion, row['prompt'], row['final_answer'])
    tier_a_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
        'completion': completion, 'classification': cls, **gt})
tier_a_summary = {'n': len(tier_a_results),
    'intermediate_tracking_accuracy': sum(r['intermediate_tracking_correct'] for r in tier_a_results) / len(tier_a_results),
    'final_answer_accuracy': sum(r['final_answer_correct'] for r in tier_a_results) / len(tier_a_results),
    'decode_back_presence_rate': sum(r['decode_back_present'] for r in tier_a_results) / len(tier_a_results),
    'decode_back_accuracy': sum(r['decode_back_correct'] for r in tier_a_results) / len(tier_a_results)}
tier_a_summary['tracking_vs_answer_gap'] = tier_a_summary['intermediate_tracking_accuracy'] - tier_a_summary['final_answer_accuracy']
print('=' * 70); print('TIER (a) RESULT:', tier_a_summary); print('=' * 70)


def evaluate_tier_b_and_c():
    print('===== TIER (b): HELD-OUT SAME-PAIR -- tracking / answer / decode-back reported SEPARATELY =====')
    clean_prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
    clean_completions = run_generation(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
    tier_b_results = []
    for row, prompt, completion in zip(eval_rows, clean_prompts, clean_completions):
        gt = score_against_ground_truth(row, completion)
        cls = classify_candidate(completion, prompt, row['final_answer'])
        tier_b_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
            'completion': completion, 'classification': cls, **gt})
    tier_b_taxonomy = {}
    for r in tier_b_results:
        name = r['classification']['category_name']
        tier_b_taxonomy[name] = tier_b_taxonomy.get(name, 0) + 1
    intermediate_acc = sum(r['intermediate_tracking_correct'] for r in tier_b_results) / len(tier_b_results)
    final_acc = sum(r['final_answer_correct'] for r in tier_b_results) / len(tier_b_results)
    decode_back_presence_rate = sum(r['decode_back_present'] for r in tier_b_results) / len(tier_b_results)
    decode_back_acc = sum(r['decode_back_correct'] for r in tier_b_results) / len(tier_b_results)
    decode_back_matches_trace_rate = sum(r['decode_back_matches_own_trace'] for r in tier_b_results) / len(tier_b_results)
    tier_b_summary = {'n': len(tier_b_results),
        'nonliteral_code_usage_rate': sum(r['used_nonliteral_code'] for r in tier_b_results) / len(tier_b_results),
        'intermediate_tracking_accuracy': intermediate_acc,
        'final_answer_accuracy': final_acc,
        'tracking_vs_answer_gap': intermediate_acc - final_acc,
        'decode_back_presence_rate': decode_back_presence_rate,
        'decode_back_accuracy': decode_back_acc,
        'decode_back_matches_own_trace_rate': decode_back_matches_trace_rate,
        'genuine_correct_among_nonliteral': genuine_correct_among_nonliteral(tier_b_results),
        'stage9c_original_baseline_for_comparison': 0.2857142857142857,
        'rebalanced_run_result_for_comparison': 0.3333333333333333,
        'taxonomy_counts': tier_b_taxonomy}
    print('TIER (b) RESULT:', tier_b_summary)

    print('===== TIER (c): HELD-OUT DIFFERENT-PAIR (Yelt/Yark) =====')
    sub_prompts = [build_prompt_with_substitution(build_prompt(r['starting_state'], r['operations']),
                   HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD) for r in eval_rows]
    sub_completions = run_generation(chat_wrap(sub_prompts), MAX_NEW_TOKENS)
    tier_c_results = []
    for row, prompt, completion in zip(eval_rows, sub_prompts, sub_completions):
        true_states = _trace(row['starting_state'], row['operations'])
        score = score_pair_substitution(completion, true_states, HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD)
        cls = classify_candidate(completion, prompt, row['final_answer'])
        tier_c_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
            'completion': completion, 'score': score, 'classification': cls})
    tier_c_taxonomy = {}
    for r in tier_c_results:
        name = r['classification']['category_name']
        tier_c_taxonomy[name] = tier_c_taxonomy.get(name, 0) + 1
    tier_c_summary = {'n': len(tier_c_results),
        'fully_correct_rate': sum(r['score']['fully_correct'] for r in tier_c_results) / len(tier_c_results),
        'final_answer_correct_rate': sum(r['score']['final_answer_correct'] for r in tier_c_results) / len(tier_c_results),
        'stayed_within_instructed_pair_rate': sum(r['score']['stayed_within_instructed_pair'] for r in tier_c_results) / len(tier_c_results),
        'lapsed_into_literal_rate': sum(r['score']['lapsed_into_literal_heads_tails'] for r in tier_c_results) / len(tier_c_results),
        'taxonomy_counts': tier_c_taxonomy}
    print('TIER (c) RESULT:', tier_c_summary)
    return tier_b_results, tier_b_summary, tier_c_results, tier_c_summary


tier_b_results, tier_b_summary, tier_c_results, tier_c_summary = evaluate_tier_b_and_c()


print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9c-sft-decode-back-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
model.save_pretrained(str(OUTPUT / 'final_adapter'))

report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED,
               'n_steps': N_STEPS, 'target_lr': TARGET_LR, 'warmup_steps': WARMUP_STEPS,
               'health_check_step': HEALTH_CHECK_STEP, 'overfit_check_every': OVERFIT_CHECK_EVERY,
               'lora': {'r': 8, 'alpha': 16, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'dropout': .05},
               'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
               'decode_back_bank_sha256': _bank_sha, 'decode_back_bank_size': len(DECODE_BACK_TRAJECTORIES),
               'label_distribution': dict(_label_dist),
               'clean21_eval_sha256': EXPECTED_CLEAN21_SHA256, 'heldout_pair': [HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD],
               'fix_description': 'explicit decode-back line inserted between the last State: line and <answer>',
               'source_diagnostic': 'experiments/09c_sft_diagnostic/aws_runs/corrupted-prefill-diagnostic-v1/corrupted_prefill_diagnostic.json'},
    'training_telemetry': telemetry,
    'overfit_watch': overfit_watch,
    'tier_a_training_set_recall': {'summary': tier_a_summary, 'samples': tier_a_results},
    'tier_b_heldout_same_pair': {'summary': tier_b_summary, 'samples': tier_b_results},
    'tier_c_heldout_different_pair': {'summary': tier_c_summary, 'samples': tier_c_results},
}
EVENT_LOG = OUTPUT / 'stage9c_sft_decode_back.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: v for k, v in report.items() if k not in ('tier_a_training_set_recall', 'tier_b_heldout_same_pair', 'tier_c_heldout_different_pair', 'training_telemetry')} |
                 {'tier_a_summary': tier_a_summary, 'tier_b_summary': tier_b_summary, 'tier_c_summary': tier_c_summary}, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9C SFT DECODE-BACK RUN COMPLETE.')
