from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, os, sys, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9c REBUILD (2026-08-31): Stage 9d showed the original 16-example SFT seed
# (25.0% of the 64 length-5 x starting-state scenarios) taught reliable Nib/Nomo TOKEN
# SUBSTITUTION (100% held-out non-literal usage) but only 28.6% held-out CORRECTNESS --
# and that correctness rate never moved across 150 steps of adversarial RL in either
# direction, strong evidence the original seed never taught genuine state tracking
# through the code, only a surface pattern with nothing for RL to reinforce. This
# script rebuilds the seed with a substantially expanded, PROGRAMMATICALLY VERIFIED
# trajectory bank (43/64 = 67.2% of the length-5 space, up from 16/64 = 25.0%),
# reusing synthetic_bridge.py's existing build_prompt/build_coded_completion/
# verify_trajectory/verify_bank UNCHANGED -- no new generation or verification logic,
# just applying the existing pipeline to more scenarios. The held-out eval set is
# BYTE-IDENTICAL to Stage 9c's original (same seed, same 21 scenarios, verified against
# the same pinned SHA256) so the 28.6% comparison is apples-to-apples: same test items,
# larger/more diverse training set. Adds periodic held-out mini-checks DURING training
# (every 10 steps) to directly watch for overfitting (train-recall saturating while
# held-out correctness plateaus/degrades), rather than only checking at the end.

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

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'  # untouched base -- same as Stage 9c
RUN_SEED = 20260830
N_STEPS = 40                  # SAME as Stage 9c's original schedule -- see design.md for whether telemetry justified changing it
HEALTH_CHECK_STEP = 5
OVERFIT_CHECK_EVERY = 10      # periodic held-out mini-eval during training, task requirement 2
WARMUP_STEPS = 5
TARGET_LR = 2e-4              # SAME as Stage 9c
MAX_NEW_TOKENS = 256

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'n_steps': N_STEPS,
       'health_check_step': HEALTH_CHECK_STEP, 'overfit_check_every': OVERFIT_CHECK_EVERY,
       'target_lr': TARGET_LR, **_actual})

print('===== IMPORTS (model-independent modules reused unchanged) =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09C_DIR))
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots, LITERAL_TOKENS  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from synthetic_bridge import (CODED_TRAJECTORIES, build_clean_length5_train_eval_split,  # noqa: E402
    build_prompt, build_coded_completion, verify_bank, teacher_forced_ce, HEADS_CODE, TAILS_CODE)
from pair_substitution import build_prompt_with_substitution, score_pair_substitution  # noqa: E402

assert len(CODED_TRAJECTORIES) == 16  # confirms the ORIGINAL bank is untouched by this script
assert HEADS_CODE == 'Nib' and TAILS_CODE == 'Nomo'

print('===== REPRODUCE STAGE 9C\'S EXACT HELD-OUT-21 FIRST -- byte-identical eval set for a valid comparison =====')
CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
# build_clean_length5_train_eval_split's eligible_eval pool is derived from CODED_TRAJECTORIES'S
# OWN bank_keys -- calling it here with the untouched, imported CODED_TRAJECTORIES (not any
# expanded bank) reproduces the IDENTICAL 21-scenario draw Stage 9c used, verified below.
train_scenarios_meta, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
_eval_sha = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
assert _eval_sha == EXPECTED_CLEAN21_SHA256, (
    f'held-out eval set does not match Stage 9c\'s pinned value -- STOP, the comparison to 28.6% '
    f'would not be valid. expected={EXPECTED_CLEAN21_SHA256}, got={_eval_sha}')
print({'clean21_eval_n': len(eval_rows), 'sha256_verified_identical_to_stage9c': True})

print('===== BUILD + VERIFY THE EXPANDED TRAJECTORY BANK (task requirement 1) =====')
# train_scenarios_meta is exactly "all 64 length-5 x starting-state scenarios MINUS the 21
# held out for eval" = 43 scenarios (includes the original 16-row bank as a subset, plus 27
# more never seen in Stage 9c's training). Each gets a real (prompt, completion) pair built
# via build_prompt/build_coded_completion (UNCHANGED, reused) and is verified via the
# EXISTING verify_bank pipeline (UNCHANGED, reused) before being trusted for training --
# no new verification logic written for this task.
EXPANDED_TRAJECTORIES = []
for meta in train_scenarios_meta:
    starting_state, operations = meta['starting_state'], meta['operations']
    completion = build_coded_completion(starting_state, operations)
    EXPANDED_TRAJECTORIES.append({
        'starting_state': starting_state, 'operations': list(operations),
        'prompt': build_prompt(starting_state, operations),
        'completion': completion, 'final_answer': meta['final_answer'], 'coded': True,
    })
_verify_reports = verify_bank(EXPANDED_TRAJECTORIES)
assert all(rep['verified'] for rep in _verify_reports), 'a trajectory failed the existing verification pipeline -- STOP'
_expanded_bank_sha = hashlib.sha256(json.dumps(EXPANDED_TRAJECTORIES, sort_keys=True).encode()).hexdigest()
_original_bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in CODED_TRAJECTORIES}
_expanded_bank_keys = {(r['starting_state'], tuple(r['operations'])) for r in EXPANDED_TRAJECTORIES}
_eval_keys = {(r['starting_state'], tuple(r['operations'])) for r in eval_rows}
assert _original_bank_keys.issubset(_expanded_bank_keys), 'expanded bank must be a strict superset of the original 16'
assert not (_expanded_bank_keys & _eval_keys), 'expanded bank must not overlap the held-out eval set'
print({'expanded_bank_size': len(EXPANDED_TRAJECTORIES), 'coverage_of_64_length5_scenarios': f'{len(EXPANDED_TRAJECTORIES)}/64 ({100*len(EXPANDED_TRAJECTORIES)/64:.1f}%)',
       'original_bank_size_for_comparison': 16, 'original_coverage': '16/64 (25.0%)',
       'all_verified': True, 'expanded_bank_sha256': _expanded_bank_sha,
       'superset_of_original_bank': True, 'no_eval_overlap': True})

HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD = 'Yelt', 'Yark'


print('===== LOAD UNTOUCHED 3B BASE + FRESH LORA (no Step-0 adapter -- same as Stage 9c) =====')
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
                   lora_dropout=.05, bias='none', task_type='CAUSAL_LM')  # IDENTICAL to Stage 9c
model = get_peft_model(base_model, lora)
_fresh_init_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _fresh_init_identity.update(name.encode()); _fresh_init_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'fresh_lora_init': True, 'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
       'gpu_memory_after_load_gb': round(torch.cuda.memory_allocated() / 1e9, 3)})


print('===== GENERATION HARNESS (defined early so periodic overfit checks can use it) =====')
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


def genuine_correct_among_nonliteral(results):
    """The headline metric (task requirement 3b): fraction of STRUCTURALLY non-literal
    candidates (per taxonomy.classify_candidate) that are also final-answer-correct.
    Matches Stage 9d's evaluate_and_track definition exactly, for a precise comparison."""
    structural = [r for r in results if r['classification']['structural_nonliteral_candidate']]
    if not structural:
        return None
    return sum(1 for r in structural if r['final_answer_correct']) / len(structural)


def quick_heldout_check(step):
    """Lightweight held-out generation check for the overfitting watch (task requirement
    2) -- same clean-21 eval set, same scoring as the final Tier (b), run periodically
    DURING training rather than only at the end."""
    was_training = model.training
    model.eval(); model.config.use_cache = True
    try:
        clean_prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
        completions = run_generation(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
        results = []
        for row, prompt, completion in zip(eval_rows, clean_prompts, completions):
            ans, valid = _extract_answer(completion)
            final_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
            cls = classify_candidate(completion, prompt, row['final_answer'])
            results.append({'final_answer_correct': final_correct, 'classification': cls})
        nonliteral_rate = sum(1 for r, c in zip(results, completions)
                               if any(t not in LITERAL_TOKENS for _i, t in parse_state_slots(c))) / len(results)
        gc = genuine_correct_among_nonliteral(results)
        summary = {'step': step, 'nonliteral_rate': nonliteral_rate, 'genuine_correct_among_nonliteral': gc,
                   'overall_final_answer_accuracy': sum(r['final_answer_correct'] for r in results) / len(results)}
        print(f'[OVERFIT-WATCH] step={step}: {summary}')
        return summary
    finally:
        model.config.use_cache = False
        if was_training: model.train()


print('===== SFT TRAINING: GRADIENT-ACCUMULATED TEACHER-FORCED CE, %d STEPS, %d TRAJECTORIES =====' % (N_STEPS, len(EXPANDED_TRAJECTORIES)))
# 2026-08-31 fix, disclosed rather than silently patched: the ORIGINAL first launch
# attempted a single full-batch forward/backward over all 43 examples at once (the
# exact pattern Stage 9c used for its 16-example bank) and hit a genuine CUDA OOM on
# the very first step's backward() -- "Tried to allocate 6.67 GiB" against ~22GB total,
# ~21.5GB already in use. This is NOT a fragmentation issue to paper over with
# PYTORCH_CUDA_ALLOC_CONF alone (added above as good hygiene, matching this project's
# usual convention, but it would not have been sufficient by itself) -- 2.7x more
# examples in one padded batch genuinely does not fit where 16 did. Fixed with standard
# gradient accumulation: split the 43 examples into MICRO_BATCH_SIZE=8 chunks (safely
# smaller than the 16 that worked), backward() each chunk's PROPORTIONALLY-SCALED loss
# before the others are computed (so only one chunk's activation graph is ever held in
# memory at a time), then a single optimizer.step() per outer step -- mathematically
# equivalent to Stage 9c's full-batch mean-CE-over-the-whole-set gradient, just computed
# without ever materializing all 43 examples' activations simultaneously.
MICRO_BATCH_SIZE = 8
MICRO_BATCHES = [EXPANDED_TRAJECTORIES[i:i + MICRO_BATCH_SIZE] for i in range(0, len(EXPANDED_TRAJECTORIES), MICRO_BATCH_SIZE)]
print({'micro_batch_size': MICRO_BATCH_SIZE, 'num_micro_batches': len(MICRO_BATCHES),
       'micro_batch_sizes': [len(mb) for mb in MICRO_BATCHES], 'sizes_sum_matches_total': sum(len(mb) for mb in MICRO_BATCHES) == len(EXPANDED_TRAJECTORIES)})

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
        weight = len(mb) / len(EXPANDED_TRAJECTORIES)
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
            raise RuntimeError(f'Non-finite loss within the first {HEALTH_CHECK_STEP} steps -- '
                                f'STOP, do not continue to the full budget: {early_losses}')
        if early_losses[-1] > early_losses[0] * 3:
            raise RuntimeError(f'Loss increased sharply (>3x) within the first {HEALTH_CHECK_STEP} steps -- '
                                f'looks unhealthy, STOP rather than continuing blindly: {early_losses}')
        print(f'HEALTH CHECK PASSED at step {HEALTH_CHECK_STEP}: finite, non-exploding loss '
              f'({early_losses[0]:.4f} -> {early_losses[-1]:.4f}). Continuing to full budget.')
    if step % OVERFIT_CHECK_EVERY == 0:
        overfit_watch.append(quick_heldout_check(step))

if not all(math.isfinite(r['loss']) for r in telemetry):
    raise RuntimeError('Non-finite loss appeared later in training (after the step-5 health check passed).')
print({'final_loss': telemetry[-1]['loss'], 'first_loss': telemetry[0]['loss'],
       'loss_reduction_ratio': telemetry[0]['loss'] / max(telemetry[-1]['loss'], 1e-9)})

model.eval(); model.config.use_cache = True


print('===== TIER (a): TRAINING-SET RECALL -- reported first, per instruction =====')
train_prompts = [row['prompt'] for row in EXPANDED_TRAJECTORIES]
train_completions = run_generation(chat_wrap(train_prompts), MAX_NEW_TOKENS)
tier_a_results = []
for row, completion in zip(EXPANDED_TRAJECTORIES, train_completions):
    expected_tokens = [normalize_state_token(t) for _i, t in parse_state_slots(row['completion'])]
    actual_tokens = [t for _i, t in parse_state_slots(completion)]
    exact_recall = actual_tokens == expected_tokens
    ans, valid = _extract_answer(completion)
    final_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
    cls = classify_candidate(completion, row['prompt'], row['final_answer'])
    tier_a_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
        'completion': completion, 'expected_tokens': expected_tokens, 'actual_tokens': actual_tokens,
        'exact_recall': exact_recall, 'final_answer_correct': final_correct, 'classification': cls})

tier_a_exact_recall_rate = sum(r['exact_recall'] for r in tier_a_results) / len(tier_a_results)
tier_a_final_acc = sum(r['final_answer_correct'] for r in tier_a_results) / len(tier_a_results)
tier_a_taxonomy = {}
for r in tier_a_results:
    name = r['classification']['category_name']
    tier_a_taxonomy[name] = tier_a_taxonomy.get(name, 0) + 1
print('=' * 70)
print('TIER (a) TRAINING-SET RECALL RESULT:')
print({'n': len(tier_a_results), 'exact_recall_rate': tier_a_exact_recall_rate,
       'final_answer_accuracy': tier_a_final_acc, 'taxonomy_counts': tier_a_taxonomy})
print('=' * 70)


def evaluate_tier_b_and_c():
    print('===== TIER (b): HELD-OUT SAME-PAIR (clean-21, plain prompt) -- THE KEY METRIC THIS TIME =====')
    clean_prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
    clean_completions = run_generation(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
    tier_b_results = []
    for row, prompt, completion in zip(eval_rows, clean_prompts, clean_completions):
        ans, valid = _extract_answer(completion)
        final_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
        used_code = bool([t for _i, t in parse_state_slots(completion) if t not in LITERAL_TOKENS])
        cls = classify_candidate(completion, prompt, row['final_answer'])
        tier_b_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
            'completion': completion, 'final_answer_correct': final_correct, 'used_nonliteral_code': used_code,
            'classification': cls})
    tier_b_taxonomy = {}
    for r in tier_b_results:
        name = r['classification']['category_name']
        tier_b_taxonomy[name] = tier_b_taxonomy.get(name, 0) + 1
    tier_b_genuine_correct = genuine_correct_among_nonliteral(tier_b_results)
    tier_b_summary = {'n': len(tier_b_results),
        'final_answer_accuracy': sum(r['final_answer_correct'] for r in tier_b_results) / len(tier_b_results),
        'nonliteral_code_usage_rate': sum(r['used_nonliteral_code'] for r in tier_b_results) / len(tier_b_results),
        'genuine_correct_among_nonliteral': tier_b_genuine_correct,
        'stage9c_original_baseline_for_comparison': 0.2857142857142857,
        'taxonomy_counts': tier_b_taxonomy}
    print('TIER (b) RESULT:', tier_b_summary)

    print('===== TIER (c): HELD-OUT DIFFERENT-PAIR (clean-21, Yelt/Yark instructed) =====')
    from synthetic_bridge import _trace
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
    OUTPUT = ROOT / f'stage9c-sft-expanded-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
model.save_pretrained(str(OUTPUT / 'final_adapter'))

report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'n_steps': N_STEPS, 'target_lr': TARGET_LR,
               'warmup_steps': WARMUP_STEPS, 'health_check_step': HEALTH_CHECK_STEP,
               'overfit_check_every': OVERFIT_CHECK_EVERY,
               'lora': {'r': 8, 'alpha': 16, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'dropout': .05},
               'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
               'expanded_bank_sha256': _expanded_bank_sha, 'expanded_bank_size': len(EXPANDED_TRAJECTORIES),
               'expanded_bank_coverage': f'{len(EXPANDED_TRAJECTORIES)}/64',
               'original_bank_size_for_comparison': 16, 'original_bank_coverage': '16/64',
               'clean21_eval_sha256': EXPECTED_CLEAN21_SHA256,
               'heldout_pair': [HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD]},
    'training_telemetry': telemetry,
    'overfit_watch': overfit_watch,
    'tier_a_training_set_recall': {'summary': {'n': len(tier_a_results), 'exact_recall_rate': tier_a_exact_recall_rate,
        'final_answer_accuracy': tier_a_final_acc, 'taxonomy_counts': tier_a_taxonomy}, 'samples': tier_a_results},
    'tier_b_heldout_same_pair': {'summary': tier_b_summary, 'samples': tier_b_results},
    'tier_c_heldout_different_pair': {'summary': tier_c_summary, 'samples': tier_c_results},
}
EVENT_LOG = OUTPUT / 'stage9c_sft_expanded.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: v for k, v in report.items() if k not in ('tier_a_training_set_recall', 'tier_b_heldout_same_pair', 'tier_c_heldout_different_pair', 'training_telemetry')} |
                 {'tier_a_summary': report['tier_a_training_set_recall']['summary'],
                  'tier_b_summary': tier_b_summary, 'tier_c_summary': tier_c_summary}, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9C SFT EXPANDED-BANK REBUILD COMPLETE.')
