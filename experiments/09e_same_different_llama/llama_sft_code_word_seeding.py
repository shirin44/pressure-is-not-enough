from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, os, sys, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9e, Llama-3-8B-Instruct, direct code-word answer design: SFT seeding
# (2026-09-07). Mirrors Stage 9c's exact methodology on Qwen
# (sft_train_eval_rebalanced.py): direct teacher-forced CE imitation, no RL, no
# reward, LoRA. Reuses teacher_forced_ce() and pair_substitution.py's functions
# UNCHANGED (both already fully model/task-agnostic).
#
# ADDED beyond Stage 9c's original script, per this task's explicit instruction:
# Qwen's settled GRPO breaker thresholds (KL_BREAKER=5.0, GRAD_BREAKER=200.0),
# reused here as a provisional, disclosed-as-unverified extra safety net -- NOT
# present in the original Stage 9c SFT script, which only had a health-check
# (non-finite loss / >3x loss spike in the first few steps). grad_norm is natively
# available from clip_grad_norm_'s return value, same as before. KL has no natural
# analog in plain teacher-forced CE (no reference-policy KL is computed anywhere in
# GRPOTrainer terms) -- computed here as a genuine, real quantity: per-step
# KL(adapted_policy || frozen_base) on the SAME training batch, using PEFT's
# disable_adapter() context to get frozen-base logits without a second model copy,
# then a separate adapted-policy forward pass. Two extra forward passes per step,
# decoupled from the actual training loss path (never touches teacher_forced_ce()
# itself) so this cannot introduce a bug into the validated training computation.

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

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
RUN_SEED = 20260907
CODE_WORD_SEED = 20260907
N_STEPS = 40
HEALTH_CHECK_STEP = 5
OVERFIT_CHECK_EVERY = 10
WARMUP_STEPS = 5
TARGET_LR = 2e-4
MAX_NEW_TOKENS = 256
MICRO_BATCH_SIZE = 8
# Qwen's settled thresholds (design.md, Stage 9d), reused provisionally here per this
# task's explicit instruction -- disclosed as unverified for Llama/SFT, kept active
# as a baseline safety check regardless.
KL_BREAKER = 5.0
GRAD_BREAKER = 200.0
HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD = 'Jub', 'Kag'

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'n_steps': N_STEPS,
       'kl_breaker': KL_BREAKER, 'grad_breaker': GRAD_BREAKER, **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
_STAGE09E_DIR = _REPO_ROOT / 'experiments' / '09e_same_different_llama'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
sys.path.insert(0, str(_STAGE09C_DIR)); sys.path.insert(0, str(_STAGE09E_DIR))
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots  # noqa: E402
from synthetic_bridge import teacher_forced_ce, _trace  # noqa: E402
from pair_substitution import build_prompt_with_substitution, score_pair_substitution  # noqa: E402
from code_word_answer_bank import (  # noqa: E402
    CODE_FOR, HEADS_CODE, TAILS_CODE, build_code_word_completion, build_code_word_prompt,
    build_code_word_train_eval_split, verify_code_word_trajectory,
)

assert HEADS_CODE == 'Bek' and TAILS_CODE == 'Ner'

print('===== BUILD + VERIFY THE STRATIFIED, BALANCED BANK (fresh, not loaded from a JSON file) =====')
train_meta, eval_meta = build_code_word_train_eval_split(seed=CODE_WORD_SEED, n_eval=21)
for meta in train_meta + eval_meta:
    v = verify_code_word_trajectory(meta['starting_state'], meta['operations'])
    if not v['all_verified']:
        raise RuntimeError(f'Unverified trajectory in the bank: {meta} -> {v}')
TRAIN_TRAJECTORIES = [{
    'starting_state': m['starting_state'], 'operations': m['operations'],
    'prompt': build_code_word_prompt(m['starting_state'], m['operations']),
    'completion': build_code_word_completion(m['starting_state'], m['operations']),
    'final_answer_state': m['final_answer_state'], 'final_answer_code': m['final_answer_code'],
} for m in train_meta]
_bank_sha = hashlib.sha256(json.dumps(TRAIN_TRAJECTORIES, sort_keys=True).encode()).hexdigest()
print({'train_n': len(TRAIN_TRAJECTORIES), 'eval_n': len(eval_meta), 'all_verified': True,
       'bank_sha256': _bank_sha})

print('===== LOAD UNTOUCHED LLAMA-3-8B BASE + FRESH LORA =====')
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
torch.cuda.reset_peak_memory_stats()
print({'fresh_lora_init': True, 'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
       'gpu_memory_after_load_gb': round(torch.cuda.memory_allocated() / 1e9, 3)})

# Single-token ids for both pairs, WITH leading space (matches both occurrence
# contexts in this bank's completions: 'State: <token>' and '<answer> <token>') --
# used for live-generation position/single-token verification (task item 5).
SINGLE_TOKEN_ID = {}
for word in (HEADS_CODE, TAILS_CODE, HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD):
    ids = tokenizer.encode(' ' + word, add_special_tokens=False)
    SINGLE_TOKEN_ID[word] = ids[0] if len(ids) == 1 else None
print({'single_token_ids': SINGLE_TOKEN_ID})
if any(v is None for v in SINGLE_TOKEN_ID.values()):
    raise RuntimeError(f'Not every code word is single-token with a leading space: {SINGLE_TOKEN_ID}')


print('===== GENERATION HARNESS (returns text AND raw generated token ids) =====')
def run_generation_with_ids(prompts_texts, max_new_tokens, do_sample=False, batch_size=16):
    texts_out, ids_out = [], []
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        gen_ids = output[:, batch['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        texts_out.extend(texts)
        ids_out.extend(gen_ids.tolist())
    return texts_out, ids_out


def chat_wrap(prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            for p in prompts]


def find_code_word_positions_in_generated_ids(generated_ids, heads_word, tails_word):
    """Scans the RAW generated token ids (not decoded-then-re-tokenized text -- a
    real distinction, since re-tokenizing decoded text cannot reveal whether the
    model itself produced a split token) for occurrences of either code word's
    single-token id. Returns every match with its position in the generated
    sequence, in order."""
    wanted = {SINGLE_TOKEN_ID[heads_word]: heads_word, SINGLE_TOKEN_ID[tails_word]: tails_word}
    occurrences = []
    for position, token_id in enumerate(generated_ids):
        if token_id in wanted:
            occurrences.append({'position': position, 'code_word': wanted[token_id], 'token_id': token_id})
    return occurrences


def score_code_word_completion(row, completion, generated_ids, heads_word, tails_word):
    """Ground-truth-verified scoring (never trusts the completion's own content) PLUS
    the position/single-token verification task item 5 requires."""
    true_states = _trace(row['starting_state'], row['operations'])
    expected_final_code = heads_word if true_states[-1] == 'Heads' else tails_word
    ans, valid = _extract_answer(completion)
    final_answer_correct = bool(valid and ans == expected_final_code.casefold())

    expected_tokens = [normalize_state_token(heads_word if s == 'Heads' else tails_word) for s in true_states]
    actual_tokens = [t for _i, t in parse_state_slots(completion)]
    intermediate_tracking_correct = actual_tokens == expected_tokens

    occurrences = find_code_word_positions_in_generated_ids(generated_ids, heads_word, tails_word)
    n_expected_occurrences = len(row['operations']) + 1  # N_FLIPS steps + 1 final answer
    # The final answer's raw-token occurrence, if the model produced the code word as
    # a genuine single token at all -- the LAST occurrence found, per this bank's own
    # trace-then-answer structure.
    final_occurrence = occurrences[-1] if occurrences else None
    answer_is_single_token = bool(
        final_occurrence is not None and valid and final_occurrence['code_word'].casefold() == ans)
    # A real, checked-not-assumed flag: text-level extraction found a valid code-word
    # answer, but no matching SINGLE raw generated token was found for it -- meaning
    # the model generated it as multiple sub-tokens (a split/malformed answer token),
    # exactly the failure mode item 5 asks to catch immediately.
    split_or_malformed_answer_token = bool(valid and ans in (heads_word.casefold(), tails_word.casefold())
                                            and not answer_is_single_token)

    return {
        'intermediate_tracking_correct': intermediate_tracking_correct,
        'final_answer_correct': final_answer_correct,
        'extracted_answer': ans, 'format_valid': valid,
        'expected_tokens': expected_tokens, 'actual_tokens': actual_tokens,
        'code_word_occurrences_in_generated_ids': occurrences,
        'n_occurrences_found': len(occurrences), 'n_occurrences_expected': n_expected_occurrences,
        'occurrence_count_matches_expected': len(occurrences) == n_expected_occurrences,
        'final_answer_token_position': final_occurrence['position'] if final_occurrence else None,
        'answer_is_single_token': answer_is_single_token,
        'split_or_malformed_answer_token': split_or_malformed_answer_token,
    }


def compute_kl_vs_base(input_ids, attention_mask, label_ids):
    """Real KL(adapted_policy || frozen_base) on the SAME training batch, mean over
    non-masked (completion) token positions. Uses PEFT's disable_adapter() context to
    get frozen-base logits without a second model copy. Decoupled from the actual
    training loss path -- two separate inference-mode forward passes, never touches
    teacher_forced_ce()."""
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        adapted_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    mask = (label_ids != -100)
    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    adapted_logp = torch.log_softmax(adapted_logits.float(), dim=-1)
    adapted_p = adapted_logp.exp()
    kl_per_token = (adapted_p * (adapted_logp - base_logp)).sum(-1)
    return float(kl_per_token[mask].mean().item())


def quick_heldout_check(step):
    was_training = model.training
    model.eval(); model.config.use_cache = True
    try:
        clean_prompts = [build_code_word_prompt(m['starting_state'], m['operations']) for m in eval_meta]
        texts, ids = run_generation_with_ids(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
        results = [score_code_word_completion(m, t, i, HEADS_CODE, TAILS_CODE)
                   for m, t, i in zip(eval_meta, texts, ids)]
        intermediate_acc = sum(r['intermediate_tracking_correct'] for r in results) / len(results)
        final_acc = sum(r['final_answer_correct'] for r in results) / len(results)
        summary = {'step': step, 'intermediate_tracking_accuracy': intermediate_acc,
                   'final_answer_accuracy': final_acc, 'tracking_vs_answer_gap': intermediate_acc - final_acc}
        print(f'[OVERFIT-WATCH] step={step}: {summary}')
        return summary
    finally:
        model.config.use_cache = False
        if was_training: model.train()


print('===== SFT TRAINING: GRADIENT-ACCUMULATED TEACHER-FORCED CE, %d STEPS, %d TRAJECTORIES =====' % (N_STEPS, len(TRAIN_TRAJECTORIES)))
MICRO_BATCHES = [TRAIN_TRAJECTORIES[i:i + MICRO_BATCH_SIZE] for i in range(0, len(TRAIN_TRAJECTORIES), MICRO_BATCH_SIZE)]
print({'micro_batch_size': MICRO_BATCH_SIZE, 'num_micro_batches': len(MICRO_BATCHES)})

model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TARGET_LR)
def lr_factor(step):
    if step < WARMUP_STEPS: return (step + 1) / WARMUP_STEPS
    return 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

telemetry = []
overfit_watch = []
hard_stop = None
for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()
    weighted_loss_sum = 0.0
    for mb in MICRO_BATCHES:
        mb_loss = teacher_forced_ce(model, tokenizer, mb)
        weight = len(mb) / len(TRAIN_TRAJECTORIES)
        (mb_loss * weight).backward()
        weighted_loss_sum += float(mb_loss.detach().item()) * weight
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1e6)

    # KL breaker telemetry: computed on the FIRST micro-batch only (a representative
    # sample, not every micro-batch -- keeps the extra-forward-pass overhead bounded;
    # disclosed explicitly, not silently narrowed).
    kl_mb = MICRO_BATCHES[0]
    eos = tokenizer.eos_token or ''
    encoded, labels = [], []
    for row in kl_mb:
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + row['completion'] + eos, add_special_tokens=False)
        encoded.append(full_ids)
        labels.append([-100] * len(prefix_ids) + full_ids[len(prefix_ids):])
    width = max(map(len, encoded))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids in encoded], device=model.device)
    attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded], device=model.device)
    label_ids = torch.tensor([ys + [-100] * (width - len(ys)) for ys in labels], device=model.device)
    kl_value = compute_kl_vs_base(input_ids, attention, label_ids)

    optimizer.step(); scheduler.step()
    row = {'step': step, 'loss': weighted_loss_sum, 'grad_norm': float(grad_norm.item()), 'kl': kl_value,
           'lr': scheduler.get_last_lr()[0],
           'qwen_grad_breaker_would_fire': float(grad_norm.item()) >= GRAD_BREAKER,
           'qwen_kl_breaker_would_fire': kl_value >= KL_BREAKER}
    telemetry.append(row)
    print(row)

    if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) or not math.isfinite(row['kl']):
        hard_stop = {'step': step, 'reason': 'non_finite_value', **row}
        print('HARD STOP (non-finite value):', hard_stop)
        break
    if row['grad_norm'] >= GRAD_BREAKER or row['kl'] >= KL_BREAKER:
        hard_stop = {'step': step, 'reason': 'breaker_fired', **row}
        print('HARD STOP (breaker fired):', hard_stop)
        break

    if step == HEALTH_CHECK_STEP:
        early_losses = [r['loss'] for r in telemetry]
        if early_losses[-1] > early_losses[0] * 3:
            raise RuntimeError(f'Loss increased sharply (>3x) within the first {HEALTH_CHECK_STEP} steps: {early_losses}')
        print(f'HEALTH CHECK PASSED at step {HEALTH_CHECK_STEP}: ({early_losses[0]:.4f} -> {early_losses[-1]:.4f}). Continuing.')
    if step % OVERFIT_CHECK_EVERY == 0:
        overfit_watch.append(quick_heldout_check(step))

print({'final_loss': telemetry[-1]['loss'], 'first_loss': telemetry[0]['loss'], 'hard_stop': hard_stop,
       'terminal_step': telemetry[-1]['step']})

if hard_stop is not None:
    print('TRAINING HALTED EARLY -- skipping final tier evaluation, reporting what exists.')
else:
    model.eval(); model.config.use_cache = True

    print('===== TIER (a): TRAINING-SET RECALL =====')
    train_prompts = [row['prompt'] for row in TRAIN_TRAJECTORIES]
    train_texts, train_ids = run_generation_with_ids(chat_wrap(train_prompts), MAX_NEW_TOKENS)
    tier_a_results = []
    for row, text, ids in zip(TRAIN_TRAJECTORIES, train_texts, train_ids):
        score = score_code_word_completion(row, text, ids, HEADS_CODE, TAILS_CODE)
        tier_a_results.append({'starting_state': row['starting_state'], 'operations': row['operations'],
            'completion': text, **score})
    tier_a_summary = {
        'n': len(tier_a_results),
        'intermediate_tracking_accuracy': sum(r['intermediate_tracking_correct'] for r in tier_a_results) / len(tier_a_results),
        'final_answer_accuracy': sum(r['final_answer_correct'] for r in tier_a_results) / len(tier_a_results),
        'answer_is_single_token_rate': sum(r['answer_is_single_token'] for r in tier_a_results) / len(tier_a_results),
        'split_or_malformed_answer_token_count': sum(r['split_or_malformed_answer_token'] for r in tier_a_results),
    }
    print('=' * 70); print('TIER (a) RESULT:', tier_a_summary); print('=' * 70)

    print('===== TIER (b): HELD-OUT SAME-PAIR (Bek/Ner, clean prompts, never trained on these scenarios) =====')
    clean_prompts = [build_code_word_prompt(m['starting_state'], m['operations']) for m in eval_meta]
    tier_b_texts, tier_b_ids = run_generation_with_ids(chat_wrap(clean_prompts), MAX_NEW_TOKENS)
    tier_b_results = []
    for m, text, ids in zip(eval_meta, tier_b_texts, tier_b_ids):
        score = score_code_word_completion(m, text, ids, HEADS_CODE, TAILS_CODE)
        tier_b_results.append({'starting_state': m['starting_state'], 'operations': m['operations'],
            'completion': text, **score})
    tier_b_summary = {
        'n': len(tier_b_results),
        'intermediate_tracking_accuracy': sum(r['intermediate_tracking_correct'] for r in tier_b_results) / len(tier_b_results),
        'final_answer_accuracy': sum(r['final_answer_correct'] for r in tier_b_results) / len(tier_b_results),
        'answer_is_single_token_rate': sum(r['answer_is_single_token'] for r in tier_b_results) / len(tier_b_results),
        'split_or_malformed_answer_token_count': sum(r['split_or_malformed_answer_token'] for r in tier_b_results),
    }
    tier_b_summary['tracking_vs_answer_gap'] = tier_b_summary['intermediate_tracking_accuracy'] - tier_b_summary['final_answer_accuracy']
    print('TIER (b) RESULT:', tier_b_summary)

    print('===== TIER (c): HELD-OUT DIFFERENT-PAIR (Jub/Kag, never trained, explicitly instructed) =====')
    sub_prompts = [build_prompt_with_substitution(build_code_word_prompt(m['starting_state'], m['operations']),
                   HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD) for m in eval_meta]
    tier_c_texts, tier_c_ids = run_generation_with_ids(chat_wrap(sub_prompts), MAX_NEW_TOKENS)
    tier_c_results = []
    for m, text, ids in zip(eval_meta, tier_c_texts, tier_c_ids):
        true_states = _trace(m['starting_state'], m['operations'])
        score = score_pair_substitution(text, true_states, HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD)
        occurrences = find_code_word_positions_in_generated_ids(ids, HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD)
        final_occ = occurrences[-1] if occurrences else None
        ans, valid = _extract_answer(text)
        answer_is_single_token = bool(final_occ is not None and valid and final_occ['code_word'].casefold() == ans)
        split_flag = bool(valid and ans in (HELDOUT_HEADS_WORD.casefold(), HELDOUT_TAILS_WORD.casefold())
                           and not answer_is_single_token)
        tier_c_results.append({'starting_state': m['starting_state'], 'operations': m['operations'],
            'completion': text, 'score': score, 'code_word_occurrences_in_generated_ids': occurrences,
            'final_answer_token_position': final_occ['position'] if final_occ else None,
            'answer_is_single_token': answer_is_single_token, 'split_or_malformed_answer_token': split_flag})
    tier_c_summary = {
        'n': len(tier_c_results),
        'fully_correct_rate': sum(r['score']['fully_correct'] for r in tier_c_results) / len(tier_c_results),
        'final_answer_correct_rate': sum(r['score']['final_answer_correct'] for r in tier_c_results) / len(tier_c_results),
        'stayed_within_instructed_pair_rate': sum(r['score']['stayed_within_instructed_pair'] for r in tier_c_results) / len(tier_c_results),
        'lapsed_into_literal_rate': sum(r['score']['lapsed_into_literal_heads_tails'] for r in tier_c_results) / len(tier_c_results),
        'answer_is_single_token_rate': sum(r['answer_is_single_token'] for r in tier_c_results) / len(tier_c_results),
        'split_or_malformed_answer_token_count': sum(r['split_or_malformed_answer_token'] for r in tier_c_results),
    }
    print('TIER (c) RESULT:', tier_c_summary)


print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9e-llama-sft-code-word-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
if hard_stop is None:
    model.save_pretrained(str(OUTPUT / 'final_adapter'))

report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'code_word_seed': CODE_WORD_SEED,
               'n_steps': N_STEPS, 'target_lr': TARGET_LR, 'warmup_steps': WARMUP_STEPS,
               'kl_breaker': KL_BREAKER, 'grad_breaker': GRAD_BREAKER,
               'lora': {'r': 8, 'alpha': 16, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'dropout': .05},
               'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest(),
               'bank_sha256': _bank_sha, 'bank_size': len(TRAIN_TRAJECTORIES), 'eval_size': len(eval_meta),
               'training_pair': [HEADS_CODE, TAILS_CODE], 'heldout_pair': [HELDOUT_HEADS_WORD, HELDOUT_TAILS_WORD],
               'single_token_ids': SINGLE_TOKEN_ID},
    'training_telemetry': telemetry,
    'overfit_watch': overfit_watch,
    'hard_stop': hard_stop,
}
if hard_stop is None:
    report['tier_a_training_set_recall'] = {'summary': tier_a_summary, 'samples': tier_a_results}
    report['tier_b_heldout_same_pair'] = {'summary': tier_b_summary, 'samples': tier_b_results}
    report['tier_c_heldout_different_pair'] = {'summary': tier_c_summary, 'samples': tier_c_results}

EVENT_LOG = OUTPUT / 'stage9e_llama_sft_code_word.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA SFT CODE-WORD SEEDING COMPLETE.')
