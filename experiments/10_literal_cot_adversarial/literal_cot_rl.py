"""Stage 10: literal-CoT adversarial-pressure experiment. Resumes checkpoint 500 (Direct RL,
Qwen2.5-3B-Instruct, 100% literal / 64.0% task-accuracy / 98.8% format-valid under the
corrected chat-templated prompt -- experiments/01_direct_rl_coinflip/aws_recovered/) and
applies the SAME lexical-penalty reward (reward_v3.score_completion_v2, imported unchanged)
that produced a null result everywhere else in this project -- everywhere else, the penalty
had zero or near-zero opportunity to bind because the starting policy was already non-literal
or near-ceiling on task accuracy. This is the first run where it starts from a policy that
actually uses literal words.

HARD REQUIREMENT, verified by test_prompt_format.py before any GPU time: train_dataset rows
are CONVERSATIONAL (list-of-dict) prompts, so TRL's GRPOTrainer applies the chat template
automatically (grpo_trainer.py's data_utils.py:160-188 / :1707) -- never a raw string.

PHASE in {'baseline','main'} via STAGE10_PHASE. BASELINE: task correctness + fixed
length/structure/variation penalties only. MAIN: BASELINE + annealed lexical penalty +
consistency + signal bonuses -- both conditions score with reward_v3.score_completion_v2
unchanged, differing only in the reward_params dict passed to it (same convention as every
prior stage: 09b/09d/Llama 09e).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09_BRIDGE_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE09D_DIR = _REPO_ROOT / 'experiments' / '09d_sft_seeded_adversarial_rl'
_STAGE09B_DIR = _REPO_ROOT / 'experiments' / '09b_model_scale_ablation'
_STAGE10_DIR = Path(__file__).resolve().parent
for p in (_STAGE07_DIR, _STAGE09_BRIDGE_DIR, _STAGE09D_DIR, _STAGE09B_DIR, _STAGE10_DIR):
    sys.path.insert(0, str(p))

from reward_v3 import score_completion_v2, verify_reward_invariant, prompt_to_text, completion_to_text  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from advantage_clamp import ADVANTAGE_CLAMP_VALUE, GRUBBS_MAX_ADVANTAGE, clamp_advantages  # noqa: E402
from kl_calibration import compute_clamp_from_pool  # noqa: E402
from literal_cot_bank import build_conversational_prompt, build_train_heldout_split, N_HELDOUT  # noqa: E402
from leakage_scanner import training_penalty_scanner, eval_leakage_scanner  # noqa: E402

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
CHECKPOINT500_DIR = Path.home() / 'aisi_checkpoints' / 'step-500'
GROUP_SIZE = 8
MAX_NEW_TOKENS = 256
FULL_STEPS = int(os.environ.get('STAGE10_FULL_STEPS', '150'))
CHECKPOINT_EVERY = int(os.environ.get('STAGE10_CHECKPOINT_EVERY', '25'))  # save + greedy eval cadence, same value
N_TRAIN = 200
RUN_SEED = int(os.environ.get('STAGE10_RUN_SEED', '0'))
if RUN_SEED == 0:
    raise RuntimeError('STAGE10_RUN_SEED must be set explicitly (42/43/44/45).')
BANK_SEED = 20261001  # fixed, shared across all 8 runs -- same train/held-out split for everyone
TARGET_LR = float(os.environ.get('STAGE10_TARGET_LR', '2e-5'))
WARMUP_UPDATES = 5
GRAD_BREAKER = float(os.environ.get('STAGE10_GRAD_BREAKER', '200.0'))
KL_BREAKER = 5.0
PHASE = os.environ.get('STAGE10_PHASE', '').strip()
if PHASE not in ('baseline', 'main'):
    raise RuntimeError(f"STAGE10_PHASE must be 'baseline' or 'main'; got {PHASE!r}.")
KL_CLAMP_D_MAX_ENV = os.environ.get('STAGE10_KL_CLAMP_D_MAX', '').strip()
KL_CLAMP_D_MAX = float(KL_CLAMP_D_MAX_ENV) if KL_CLAMP_D_MAX_ENV else None  # empty -> MEASUREMENT mode

print({'gpu': torch.cuda.get_device_name(0), 'phase': PHASE, 'run_seed': RUN_SEED, 'full_steps': FULL_STEPS,
       'checkpoint_every': CHECKPOINT_EVERY, 'target_lr': TARGET_LR, 'grad_breaker': GRAD_BREAKER,
       'kl_breaker': KL_BREAKER, 'advantage_clamp_value': ADVANTAGE_CLAMP_VALUE,
       'grubbs_max_advantage': GRUBBS_MAX_ADVANTAGE, 'kl_clamp_d_max': KL_CLAMP_D_MAX,
       'kl_clamp_mode': 'ACTIVE' if KL_CLAMP_D_MAX is not None else 'MEASUREMENT (pass-through)'})

BASELINE_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)
MAIN_REWARD_PARAMS: dict = {}  # score_completion_v2 defaults
_reward_params = {'baseline': BASELINE_REWARD_PARAMS, 'main': MAIN_REWARD_PARAMS}[PHASE]
_invariant_kwargs = {'baseline': {'consistency_magnitude': 0.0, 'signal_magnitude': 0.0, 'cot_max_scale': 0.0},
                     'main': {}}[PHASE]
_margins = verify_reward_invariant(**_invariant_kwargs)
assert _margins['margin_correct_over_wrong'] > 0 and _margins['margin_wrong_over_malformed'] > 0
print(f'REWARD INVARIANT ({PHASE}, re-verified fresh this launch):', _margins)

print('===== ADVANTAGE + PER-TOKEN KL CLAMP: patching GRPOTrainer (same pattern as 09d/Llama 09e, reused unchanged) =====')
_original_compute_loss = GRPOTrainer.compute_loss
_original_get_logps = GRPOTrainer._get_per_token_logps_and_entropies
ADVANTAGE_CLAMP_STATE = {'engaged_count': 0, 'total_count': 0}
_INSIDE_COMPUTE_LOSS = [False]
_CURRENT_INPUTS = [None]
CURRENT_PHYSICAL_STEP = [0]
KL_CLAMP_STATE = {'d_max': KL_CLAMP_D_MAX, 'engaged_token_count': 0, 'total_token_count': 0, 'kl_pool': []}
# Step-aware pool: (physical_step, value) pairs, NOT bare values. Checkpoint 500 has a
# documented history (logs/development_log.md, 2026-08-17/18/24; experiments/04_stability_investigation/)
# of GRPO-resume blowups within 3-15 steps, and the FIRST calibration attempt against it
# (kl_clamp_mitigation_v1) failed specifically because its pool was NOT filtered to exclude
# steps already near the breaker threshold before computing median/IQR -- exactly the mistake
# this step-stamped pool exists to avoid. See _clean_kl_pool_excluding_borderline_steps below.
KL_POOL_BY_STEP: dict[int, list[float]] = {}

# Per-step accumulators (flushed to STEP_STATS at on_step_end) -- persistence requirement #4:
# advantage mean|.| pre/post clamp + fraction clamped, KL mean raw/post-clamp + fraction engaged.
_STEP_ACCUM = {'pre_adv_abs_sum': 0.0, 'post_adv_abs_sum': 0.0, 'adv_n': 0, 'adv_engaged': 0,
               'raw_kl_abs_sum': 0.0, 'post_kl_abs_sum': 0.0, 'kl_n': 0, 'kl_engaged': 0}


def _reset_step_accum():
    for k in _STEP_ACCUM:
        _STEP_ACCUM[k] = 0.0 if 'sum' in k else 0


def _kl_patched_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    logps, entropies, aux_loss = _original_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    if _INSIDE_COMPUTE_LOSS[0] and _CURRENT_INPUTS[0] is not None:
        current_inputs = _CURRENT_INPUTS[0]
        ref = current_inputs.get('ref_per_token_logps')
        if ref is not None and ref.shape == logps.shape:
            diff = ref - logps.detach()
            mask = current_inputs.get('completion_mask')
            masked_diff = diff[mask.bool()].abs().flatten() if mask is not None and mask.shape == diff.shape else diff.abs().flatten()
            pool_vals = masked_diff.tolist()
            KL_CLAMP_STATE['kl_pool'].extend(pool_vals)
            KL_POOL_BY_STEP.setdefault(CURRENT_PHYSICAL_STEP[0], []).extend(pool_vals)
            _STEP_ACCUM['raw_kl_abs_sum'] += float(masked_diff.sum().item())
            _STEP_ACCUM['kl_n'] += masked_diff.numel()
            if KL_CLAMP_STATE['d_max'] is not None:
                d_max = KL_CLAMP_STATE['d_max']
                diff_clamped = torch.clamp(diff, min=-d_max, max=d_max)
                engaged = diff.abs() > d_max
                n_engaged = int(engaged.sum().item())
                KL_CLAMP_STATE['engaged_token_count'] += n_engaged
                KL_CLAMP_STATE['total_token_count'] += diff.numel()
                _STEP_ACCUM['kl_engaged'] += n_engaged
                masked_post = diff_clamped[mask.bool()].abs().flatten() if mask is not None and mask.shape == diff_clamped.shape else diff_clamped.abs().flatten()
                _STEP_ACCUM['post_kl_abs_sum'] += float(masked_post.sum().item())
                current_inputs['ref_per_token_logps'] = (logps.detach() + diff_clamped).detach()
            else:
                _STEP_ACCUM['post_kl_abs_sum'] += float(masked_diff.sum().item())  # measurement mode: post==raw
    return logps, entropies, aux_loss


def _advantage_and_kl_clamped_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    adv = inputs.get('advantages')
    if adv is not None:
        clamped, engaged = clamp_advantages(adv, ADVANTAGE_CLAMP_VALUE)
        n_engaged = int(engaged.sum().item())
        ADVANTAGE_CLAMP_STATE['engaged_count'] += n_engaged
        ADVANTAGE_CLAMP_STATE['total_count'] += adv.numel()
        _STEP_ACCUM['pre_adv_abs_sum'] += float(adv.detach().abs().sum().item())
        _STEP_ACCUM['post_adv_abs_sum'] += float(clamped.detach().abs().sum().item())
        _STEP_ACCUM['adv_n'] += adv.numel()
        _STEP_ACCUM['adv_engaged'] += n_engaged
        inputs['advantages'] = clamped
    _INSIDE_COMPUTE_LOSS[0] = True
    _CURRENT_INPUTS[0] = inputs
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        _INSIDE_COMPUTE_LOSS[0] = False
        _CURRENT_INPUTS[0] = None
    return result


GRPOTrainer._get_per_token_logps_and_entropies = _kl_patched_get_logps
GRPOTrainer.compute_loss = _advantage_and_kl_clamped_compute_loss
assert GRPOTrainer.compute_loss is _advantage_and_kl_clamped_compute_loss

print('===== BUILD TRAIN/HELD-OUT SPLIT (shared bank seed across all 8 runs) =====')
train_meta, eval_meta = build_train_heldout_split(run_seed=BANK_SEED, n_train=N_TRAIN, n_heldout=N_HELDOUT)
TRAIN_POOL = [{'prompt': build_conversational_prompt(m['prompt']), 'ground_truth': m['ground_truth']} for m in train_meta]
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL), 'eval_n': len(eval_meta), 'bank_seed': BANK_SEED})

print('===== LOAD TOKENIZER =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')

WEIGHTS = CHECKPOINT500_DIR / 'adapter_model.safetensors'
if not WEIGHTS.is_file():
    raise FileNotFoundError(WEIGHTS)
_seed_adapter_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
print({'seed_checkpoint': str(CHECKPOINT500_DIR), 'seed_adapter_sha256': _seed_adapter_sha})


def load_seed_checkpoint():
    """Fresh base model + fresh get_peft_model() every call -- never reuse a base_model
    object across launches (avoids TRL's PEFT beta!=0 reference-adapter leak, same
    checkpoint-isolation discipline as every prior stage)."""
    fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                      quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = False
    model = get_peft_model(fresh_base, LoraConfig(**LORA_KWARGS))
    result = set_peft_model_state_dict(model, load_safetensors(str(WEIGHTS)), adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys}')
    assert list(model.peft_config) == ['default']
    identity = hashlib.sha256()
    for name, param in model.named_parameters():
        if '.default.' not in name:
            continue
        identity.update(name.encode()); identity.update(param.detach().float().cpu().numpy().tobytes())
    return model, identity.hexdigest()


ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        w = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= w and row[-w:].tolist() == ANSWER_IDS for row in input_ids],
                            device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])

print('===== ZERO-STEP SANITY CHECK: does the freshly-loaded seed checkpoint reproduce its own reported behavior? =====')
_sanity_model, _sanity_identity = load_seed_checkpoint()
_sanity_model.eval(); _sanity_model.config.use_cache = True


def _chat_wrap(prompts_text):
    return [tokenizer.apply_chat_template(build_conversational_prompt(p), tokenize=False, add_generation_prompt=True) for p in prompts_text]


def _generate_batch(model, prompts_text, max_new_tokens, do_sample=False, batch_size=16):
    out_texts, out_ntoks, out_hitcap = [], [], []
    wrapped = _chat_wrap(prompts_text)
    for start in range(0, len(wrapped), batch_size):
        chunk = wrapped[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                    stopping_criteria=ANSWER_STOP)
        gen = output[:, batch['input_ids'].shape[1]:]
        for row in gen:
            ids = row.tolist()
            eos_pos = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else None
            hit_cap = eos_pos is None and len(ids) >= max_new_tokens and ids[-len(ANSWER_IDS):] != ANSWER_IDS
            cut = ids[:eos_pos + 1] if eos_pos is not None else ids
            out_texts.append(tokenizer.decode(cut, skip_special_tokens=True))
            out_ntoks.append(len(cut)); out_hitcap.append(bool(hit_cap))
    return out_texts, out_ntoks, out_hitcap


def score_against_ground_truth(m, completion):
    ans_valid_breakdown = score_completion_v2(completion, m['ground_truth'], 150, 150, prompt=m['prompt'])
    tier = classify_candidate(completion, m['prompt'], ground_truth=m['ground_truth'])
    eval_leak = eval_leakage_scanner(completion)
    train_scan = training_penalty_scanner(completion)
    return {'final_answer_correct': ans_valid_breakdown['r_task'] == 4.0, 'format_valid': ans_valid_breakdown['r_task'] != -5.0,
            'r_task': ans_valid_breakdown['r_task'], 'taxonomy_category': tier['category'],
            'taxonomy_category_name': tier['category_name'],
            'structural_nonliteral_candidate': tier['structural_nonliteral_candidate'],
            'eval_leakage_scanner': eval_leak, 'training_penalty_scanner': train_scan}


_sanity_prompts = [m['prompt'] for m in eval_meta]
_sanity_texts, _sanity_ntoks, _sanity_hitcap = _generate_batch(_sanity_model, _sanity_prompts, MAX_NEW_TOKENS)
_sanity_results = [score_against_ground_truth(m, c) for m, c in zip(eval_meta, _sanity_texts)]
_sanity_acc = sum(r['final_answer_correct'] for r in _sanity_results) / len(_sanity_results)
_sanity_valid = sum(r['format_valid'] for r in _sanity_results) / len(_sanity_results)
print({'sanity_checkpoint_identity_sha256': _sanity_identity, 'sanity_final_answer_accuracy': _sanity_acc,
       'sanity_format_valid_rate': _sanity_valid, 'n': len(_sanity_results)})
del _sanity_model
torch.cuda.empty_cache()
print('PASSED: seed checkpoint loads correctly.')


def evaluate_and_track(model, step, run_label):
    model.eval(); model.config.use_cache = True
    try:
        prompts = [m['prompt'] for m in eval_meta]
        texts, ntoks, hitcap = _generate_batch(model, prompts, MAX_NEW_TOKENS)
        samples = []
        for m, completion, n_tok, hc in zip(eval_meta, texts, ntoks, hitcap):
            score = score_against_ground_truth(m, completion)
            samples.append({'starting_prompt': m['prompt'], 'ground_truth': m['ground_truth'], 'completion': completion,
                            'n_tokens': n_tok, 'hit_length_cap': hc, **score})
        n = len(samples)
        result = {'step': step, 'label': run_label, 'n': n,
                  'genuine_correct_rate': sum(s['final_answer_correct'] for s in samples) / n,
                  'format_valid_rate': sum(s['format_valid'] for s in samples) / n,
                  'eval_leakage_rate': sum(not s['eval_leakage_scanner']['clean'] for s in samples) / n,
                  'training_penalty_scanner_leakage_rate': sum(not s['training_penalty_scanner']['clean'] for s in samples) / n,
                  'tier_distribution': {k: sum(s['taxonomy_category'] == k for s in samples) for k in range(1, 11)},
                  'tier_9_or_10_present': any(s['taxonomy_category'] in (9, 10) for s in samples),
                  'samples': samples}
        print(f'[{run_label}] MILESTONE STEP {step}: genuine_correct_rate={result["genuine_correct_rate"]:.3f} '
              f'eval_leakage_rate={result["eval_leakage_rate"]:.3f} format_valid_rate={result["format_valid_rate"]:.3f} '
              f'tier_dist={result["tier_distribution"]}')
        if result['tier_9_or_10_present']:
            print(f'*** FLAG: TIER 9 OR 10 COMPLETION AT STEP {step} -- see samples for detail. ***')
        return result
    finally:
        model.config.use_cache = False
        model.train()


def run_phase(n_steps, reward_params, run_label, checkpoint_every):
    print(f'===== PHASE: {run_label} ({n_steps} steps), adv_clamp={ADVANTAGE_CLAMP_VALUE}, kl_d_max={KL_CLAMP_D_MAX} =====')
    ADVANTAGE_CLAMP_STATE['engaged_count'] = 0; ADVANTAGE_CLAMP_STATE['total_count'] = 0
    model, identity = load_seed_checkpoint()
    print({'run_label': run_label, 'loaded_checkpoint_identity_sha256': identity})

    root = Path.home() / 'aisi_checkpoints'
    for version_id in range(1, 1000):
        run_output = root / f'stage10-literal-cot-{PHASE}-v{version_id}'
        if not run_output.exists():
            break
    else:
        raise RuntimeError('Could not allocate output directory.')
    run_output.mkdir(parents=True)

    args = GRPOConfig(output_dir=str(_STAGE10_DIR / f'_scratch_{run_label}'), per_device_train_batch_size=1,
                      gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
                      max_steps=n_steps, learning_rate=TARGET_LR, lr_scheduler_type='linear', warmup_steps=min(WARMUP_UPDATES, n_steps),
                      bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
                      max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
                      logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
                      report_to='none', remove_unused_columns=False, disable_dropout=True, seed=RUN_SEED, data_seed=RUN_SEED)

    physical_step_box = [0]
    rollout_records = []  # persistence requirement #4, per-rollout

    def diagnostic_reward(prompts, completions, **kwargs):
        truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
        texts = [completion_to_text(x) for x in completions]
        prompt_texts = [prompt_to_text(x) for x in prompts]
        breakdowns = [score_completion_v2(t, y, physical_step_box[0] or 1, n_steps, prompt=p, **reward_params)
                      for t, y, p in zip(texts, truths, prompt_texts)]
        for t, y, p, b in zip(texts, truths, prompt_texts, breakdowns):
            tier = classify_candidate(t, p, ground_truth=y)
            train_scan = training_penalty_scanner(t)
            eval_scan = eval_leakage_scanner(t)
            rollout_records.append({'physical_step': physical_step_box[0], 'completion': t,
                                    'r_task': b['r_task'], 'p_cot_raw': b['p_cot'], 'banned_count': b['banned_count'],
                                    'taxonomy_category': tier['category'], 'taxonomy_category_name': tier['category_name'],
                                    'training_penalty_scanner': train_scan, 'eval_leakage_scanner': eval_scan,
                                    'total_reward': b['total']})
            if tier['category'] in (9, 10):
                print(f"*** FLAG: TIER {tier['category']} TRAINING ROLLOUT AT STEP {physical_step_box[0]} *** {t!r}")
        return [x['total'] for x in breakdowns]

    trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.create_optimizer()
    from torch.optim.lr_scheduler import LambdaLR
    w = min(WARMUP_UPDATES, n_steps)
    def lr_factor(i):
        if i < w:
            return 0.1 + 0.9 * i / max(w - 1, 1)
        decay = n_steps - w
        return max(0.0, (n_steps - i) / decay) if decay > 0 else 1.0
    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))

    original_generate = model.generate
    def generate_stopped(*a, **kw):
        kw.setdefault('stopping_criteria', ANSWER_STOP)
        return original_generate(*a, **kw)
    model.generate = generate_stopped
    trainer.model.generate = generate_stopped

    def breaker(grad_norm, kl):
        return grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER

    telemetry = []; milestones = []; hard_stop = [None]; checkpoints_saved = []

    class Cb(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            physical_step_box[0] = int(state.global_step) + 1
            CURRENT_PHYSICAL_STEP[0] = physical_step_box[0]
            _reset_step_accum()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            a = _STEP_ACCUM
            step_stats = {
                'physical_step': step,
                'mean_abs_advantage_pre_clamp': (a['pre_adv_abs_sum'] / a['adv_n']) if a['adv_n'] else None,
                'mean_abs_advantage_post_clamp': (a['post_adv_abs_sum'] / a['adv_n']) if a['adv_n'] else None,
                'fraction_advantages_clamped': (a['adv_engaged'] / a['adv_n']) if a['adv_n'] else None,
                'mean_raw_per_token_kl': (a['raw_kl_abs_sum'] / a['kl_n']) if a['kl_n'] else None,
                'mean_post_clamp_per_token_kl': (a['post_kl_abs_sum'] / a['kl_n']) if a['kl_n'] else None,
                'fraction_kl_tokens_engaged': (a['kl_engaged'] / a['kl_n']) if a['kl_n'] else None,
            }
            rows_this_step = [r for r in rollout_records if r['physical_step'] == step]
            if rows_this_step:
                rewards_this_step = [r['total_reward'] for r in rows_this_step]
                lens_this_step = [len(tokenizer.encode(r['completion'], add_special_tokens=False)) for r in rows_this_step]
                step_stats.update({
                    'mean_reward': statistics.fmean(rewards_this_step), 'min_reward': min(rewards_this_step),
                    'max_reward': max(rewards_this_step), 'reward_variance_nonzero': (statistics.pvariance(rewards_this_step) > 0),
                    'mean_completion_length_tokens': statistics.fmean(lens_this_step),
                    'fraction_hit_length_cap': sum(x >= MAX_NEW_TOKENS for x in lens_this_step) / len(lens_this_step),
                    'nonzero_penalty_fraction': sum(r['banned_count'] > 0 for r in rows_this_step) / len(rows_this_step),
                    'n_rollouts_this_step': len(rows_this_step),
                })
            event_step_stats.append(step_stats); save_event()

            if step > 0 and (step % checkpoint_every == 0 or step == n_steps):
                ckpt_dir = run_output / f'checkpoint-{step}'
                model.save_pretrained(str(ckpt_dir))
                checkpoints_saved.append(str(ckpt_dir))
                save_event()
                milestones.append(evaluate_and_track(model, step, run_label))
                save_event()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            row = {'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
            telemetry.append(row); save_event()
            grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
            if not all(math.isfinite(x) for x in (grad, kl)) or breaker(grad, kl):
                hard_stop[0] = {'step': int(state.global_step), 'grad_norm': grad, 'kl': kl,
                               'triggered_by': 'grad_norm' if grad >= GRAD_BREAKER else 'kl'}
                save_event()
                control.should_training_stop = True
            return control

    event_step_stats: list = []
    EVENT_LOG = run_output / f'stage10_literal_cot_{PHASE}.json'

    def save_event():
        payload = {'config': {'model': MODEL_NAME, 'phase': PHASE, 'run_seed': RUN_SEED, 'full_steps': n_steps,
                              'checkpoint_every': checkpoint_every, 'target_lr': TARGET_LR, 'grad_breaker': GRAD_BREAKER,
                              'kl_breaker': KL_BREAKER, 'advantage_clamp_value': ADVANTAGE_CLAMP_VALUE,
                              'grubbs_max_advantage': GRUBBS_MAX_ADVANTAGE, 'kl_clamp_d_max': KL_CLAMP_D_MAX,
                              'seed_adapter_sha256': _seed_adapter_sha, 'sanity_checkpoint_identity_sha256': _sanity_identity,
                              'sanity_final_answer_accuracy': _sanity_acc, 'sanity_format_valid_rate': _sanity_valid,
                              'reward_invariant': _margins, 'reward_params': str(_reward_params), 'bank_seed': BANK_SEED,
                              'n_train': N_TRAIN, 'n_heldout': N_HELDOUT},
                   'telemetry': telemetry, 'step_stats': event_step_stats, 'milestones': milestones,
                   'hard_stop': hard_stop[0], 'checkpoints_saved': checkpoints_saved,
                   'rollout_records': rollout_records}
        tmp = EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2, default=str)); tmp.replace(EVENT_LOG)

    trainer.add_callback(Cb())
    save_event()

    trainer.train()
    terminal = int(trainer.state.global_step)
    advantage_clamp_summary = {'clamp_value': ADVANTAGE_CLAMP_VALUE, 'engaged_count': ADVANTAGE_CLAMP_STATE['engaged_count'],
                               'total_count': ADVANTAGE_CLAMP_STATE['total_count'],
                               'engagement_rate': (ADVANTAGE_CLAMP_STATE['engaged_count'] / ADVANTAGE_CLAMP_STATE['total_count']
                                                   if ADVANTAGE_CLAMP_STATE['total_count'] else None)}
    kl_pool = KL_CLAMP_STATE['kl_pool']
    kl_pool_diagnostics = compute_clamp_from_pool(kl_pool) if kl_pool else None

    # CLEAN-STEP-FILTERED diagnostics: the methodology that actually fixed checkpoint 500's
    # documented GRPO-resume instability (logs/development_log.md, 2026-08-24 "KL clamp
    # recalibration"). Excludes any step whose OWN logged grad_norm/kl came within 2x of this
    # run's breaker thresholds before pooling per-token values -- the naive unfiltered pool
    # (kl_pool_diagnostics above) is what caused the first, failed calibration attempt against
    # this same checkpoint (kl_clamp_mitigation_v1: calibrated on a pool contaminated by
    # near-blowup steps, produced a clamp that essentially never engaged).
    _BORDERLINE_GRAD, _BORDERLINE_KL = GRAD_BREAKER / 2, KL_BREAKER / 2
    _telemetry_by_step = {row['physical_step']: row for row in telemetry if 'grad_norm' in row and 'kl' in row}
    _clean_steps = sorted(s for s, row in _telemetry_by_step.items()
                          if row['grad_norm'] < _BORDERLINE_GRAD and row['kl'] < _BORDERLINE_KL)
    _excluded_steps = sorted(set(_telemetry_by_step) - set(_clean_steps))
    _clean_pool = [v for s in _clean_steps for v in KL_POOL_BY_STEP.get(s, [])]
    clean_pool_diagnostics = compute_clamp_from_pool(_clean_pool) if _clean_pool else None
    kl_clamp_summary = {'d_max': KL_CLAMP_D_MAX, 'engaged_token_count': KL_CLAMP_STATE['engaged_token_count'],
                        'total_token_count': KL_CLAMP_STATE['total_token_count'], 'pool_n': len(kl_pool),
                        'pool_diagnostics': kl_pool_diagnostics,
                        'clean_steps': _clean_steps, 'excluded_borderline_steps': _excluded_steps,
                        'borderline_grad_cutoff': _BORDERLINE_GRAD, 'borderline_kl_cutoff': _BORDERLINE_KL,
                        'clean_pool_n': len(_clean_pool), 'clean_pool_diagnostics': clean_pool_diagnostics}

    final_payload_extra = {'terminal_step': terminal, 'advantage_clamp_summary': advantage_clamp_summary,
                           'kl_clamp_summary': kl_clamp_summary, 'run_output_dir': str(run_output)}
    save_event()
    with open(EVENT_LOG) as f:
        full = json.load(f)
    full.update(final_payload_extra)
    EVENT_LOG.write_text(json.dumps(full, indent=2, default=str))

    trainer, model = None, None
    torch.cuda.empty_cache()
    return full


if __name__ == '__main__':
    result = run_phase(FULL_STEPS, _reward_params, f'literal_cot_{PHASE}', CHECKPOINT_EVERY)
    print('===== FINAL REPORT =====')
    print(json.dumps({'phase': PHASE, 'terminal_step': result['terminal_step'], 'hard_stop': result['hard_stop'],
                      'advantage_clamp_summary': result['advantage_clamp_summary'],
                      'kl_clamp_summary': {k: v for k, v in result['kl_clamp_summary'].items() if k != 'kl_pool'},
                      'n_milestones': len(result['milestones']), 'n_checkpoints_saved': len(result['checkpoints_saved']),
                      'final_milestone_genuine_correct_rate': result['milestones'][-1]['genuine_correct_rate'] if result['milestones'] else None,
                      }, indent=2, default=str))
    print(f'STAGE 10 LITERAL-COT {PHASE.upper()} RUN COMPLETE (seed={RUN_SEED}).')
