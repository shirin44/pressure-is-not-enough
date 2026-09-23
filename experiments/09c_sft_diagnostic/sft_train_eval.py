from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, sys, warnings
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9c: plain SFT representational-capacity diagnostic. NO GRPO, NO reward
# function, NO rollout generation, NO injection mechanism -- direct teacher-forced
# cross-entropy on the 16 verified Nib/Nomo trajectories, reusing
# synthetic_bridge.teacher_forced_ce unchanged. Full design and rationale: design.md.
# Sanity-check discipline: a hard health assertion after step 5 (finite, non-exploding
# loss) before continuing to the full 40-step budget -- fails fast and cheaply rather
# than a separate discard-and-restart dry run, since SFT here has no scenario-draw
# randomness to control for the way the RL stages did (full-batch, fixed 16 examples).

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

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'  # untouched base -- NOT Step-0's milestone-12 (design.md section 1)
RUN_SEED = 20260830
N_STEPS = 40                 # full-batch "epochs" over the 16 examples (design.md section 2)
HEALTH_CHECK_STEP = 5        # hard health assertion here before continuing (lighter than a separate dry run)
WARMUP_STEPS = 5
TARGET_LR = 2e-4
MAX_NEW_TOKENS = 256

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'n_steps': N_STEPS,
       'health_check_step': HEALTH_CHECK_STEP, 'target_lr': TARGET_LR, **_actual})

print('===== IMPORTS (model-independent modules reused unchanged) =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09C_DIR))
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots, LITERAL_TOKENS  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from synthetic_bridge import (CODED_TRAJECTORIES, build_clean_length5_train_eval_split,  # noqa: E402
    build_prompt, teacher_forced_ce, HEADS_CODE, TAILS_CODE)
from pair_substitution import build_prompt_with_substitution, score_pair_substitution  # noqa: E402

assert len(CODED_TRAJECTORIES) == 16
assert all(row['coded'] for row in CODED_TRAJECTORIES)
assert HEADS_CODE == 'Nib' and TAILS_CODE == 'Nomo'
print({'coded_bank_size': len(CODED_TRAJECTORIES), 'heads_code': HEADS_CODE, 'tails_code': TAILS_CODE})

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
_, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
assert hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest() == EXPECTED_CLEAN21_SHA256
print({'clean21_eval_n': len(eval_rows), 'sha256_verified': True})

HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD = 'Yelt', 'Yark'  # token_pair_selection.md's reserved held-out pair


print('===== LOAD UNTOUCHED 3B BASE + FRESH LORA (no Step-0 adapter -- design.md section 1) =====')
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
                   lora_dropout=.05, bias='none', task_type='CAUSAL_LM')  # IDENTICAL to every Stage 9/9b run
model = get_peft_model(base_model, lora)
_fresh_init_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _fresh_init_identity.update(name.encode()); _fresh_init_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'fresh_lora_init': True, 'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
       'gpu_memory_after_load_gb': round(torch.cuda.memory_allocated() / 1e9, 3)})


print('===== SFT TRAINING: FULL-BATCH TEACHER-FORCED CE, EXACTLY %d STEPS =====' % N_STEPS)
model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TARGET_LR)
def lr_factor(step):
    if step < WARMUP_STEPS: return (step + 1) / WARMUP_STEPS
    return 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

telemetry = []
for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()
    loss = teacher_forced_ce(model, tokenizer, CODED_TRAJECTORIES)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1e6)
    optimizer.step(); scheduler.step()
    row = {'step': step, 'loss': float(loss.detach().item()), 'grad_norm': float(grad_norm.item()),
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

if not all(math.isfinite(r['loss']) for r in telemetry):
    raise RuntimeError('Non-finite loss appeared later in training (after the step-5 health check passed).')
print({'final_loss': telemetry[-1]['loss'], 'first_loss': telemetry[0]['loss'],
       'loss_reduction_ratio': telemetry[0]['loss'] / max(telemetry[-1]['loss'], 1e-9)})

model.eval(); model.config.use_cache = True


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


print('===== TIER (a): TRAINING-SET RECALL -- reported first, per instruction =====')
train_prompts = [row['prompt'] for row in CODED_TRAJECTORIES]
train_completions = run_generation(chat_wrap(train_prompts), MAX_NEW_TOKENS)
tier_a_results = []
for row, completion in zip(CODED_TRAJECTORIES, train_completions):
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
    print('===== TIER (b): HELD-OUT SAME-PAIR (clean-21, plain prompt) =====')
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
    tier_b_summary = {'n': len(tier_b_results), 'final_answer_accuracy': sum(r['final_answer_correct'] for r in tier_b_results) / len(tier_b_results),
        'nonliteral_code_usage_rate': sum(r['used_nonliteral_code'] for r in tier_b_results) / len(tier_b_results),
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


TIER_A_MINIMUM_RECALL_TO_PROCEED = 0.0  # per instruction: report (a) first; tiers b/c run regardless so the
                                          # full picture is captured in one evidence file, but (a)'s result
                                          # is what determines how (b)/(c) should be INTERPRETED, not whether
                                          # they're computed at all -- cheap enough (no extra GPU load) to
                                          # just run everything and let the write-up state the implication.
tier_b_results, tier_b_summary, tier_c_results, tier_c_summary = evaluate_tier_b_and_c()


print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9c-sft-diagnostic-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
model.save_pretrained(str(OUTPUT / 'final_adapter'))

report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'n_steps': N_STEPS, 'target_lr': TARGET_LR,
               'warmup_steps': WARMUP_STEPS, 'health_check_step': HEALTH_CHECK_STEP,
               'lora': {'r': 8, 'alpha': 16, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'dropout': .05},
               'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
               'coded_bank_sha256': hashlib.sha256(json.dumps(CODED_TRAJECTORIES, sort_keys=True).encode()).hexdigest(),
               'clean21_eval_sha256': EXPECTED_CLEAN21_SHA256,
               'heldout_pair': [HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD]},
    'training_telemetry': telemetry,
    'tier_a_training_set_recall': {'summary': {'n': len(tier_a_results), 'exact_recall_rate': tier_a_exact_recall_rate,
        'final_answer_accuracy': tier_a_final_acc, 'taxonomy_counts': tier_a_taxonomy}, 'samples': tier_a_results},
    'tier_b_heldout_same_pair': {'summary': tier_b_summary, 'samples': tier_b_results},
    'tier_c_heldout_different_pair': {'summary': tier_c_summary, 'samples': tier_c_results},
}
EVENT_LOG = OUTPUT / 'stage9c_sft_diagnostic.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({k: v for k, v in report.items() if k not in ('tier_a_training_set_recall', 'tier_b_heldout_same_pair', 'tier_c_heldout_different_pair')} |
                 {'tier_a_summary': report['tier_a_training_set_recall']['summary'],
                  'tier_b_summary': tier_b_summary, 'tier_c_summary': tier_c_summary}, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9C SFT DIAGNOSTIC COMPLETE.')
