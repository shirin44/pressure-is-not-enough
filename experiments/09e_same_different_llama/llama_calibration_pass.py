from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, sys, warnings
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# Stage 9e, Llama-3-8B-Instruct pass, Part A, Step 3 (2026-09-07): a SHORT, PURELY
# DIAGNOSTIC calibration pass -- measures this model's real gradient-norm/KL telemetry
# scale under actual GRPO optimizer steps, and real peak GPU memory during those steps
# (gradients + optimizer state + generation KV-cache, on top of the 8,666 MiB static
# 8-bit load footprint already measured in llama_model_validation.py). Explicitly NOT
# Part B/C work: no Same/Different task, no trained checkpoint produced, no held-out
# evaluation -- informational only, same category as the memory-footprint check.
#
# Reuses the EXISTING, already-verified Coin Flip literal task and reward pipeline
# (synthetic_bridge.py + reward_v3.py's score_completion_v2) as the placeholder task,
# per the task's own explicit instruction ("fine to use the existing Coin Flip literal
# task ... the goal is measuring THIS MODEL's telemetry characteristics, not testing
# the new task"). Uses a FRESH LoRA adapter on the untouched base model -- no SFT-seed
# checkpoint exists for Llama, and none is needed for a telemetry-only measurement.
#
# No breaker enforcement (deliberately): the whole point is observing whether Qwen's
# settled thresholds (KL_BREAKER=5.0, GRAD_BREAKER=200.0) would even be appropriate --
# enforcing them here would truncate exactly the data needed to answer that question.
# Telemetry is logged every step regardless; breaker-would-have-fired is computed
# afterward from the full, unclipped record.

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'bitsandbytes.autograd._functions'):
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
_expected = {'transformers': '5.13.1', 'trl': '1.9.2', 'peft': '0.19.1', 'bitsandbytes': '0.50.0'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
RUN_SEED = 20260907
GROUP_SIZE = 8; MAX_NEW_TOKENS = 256
CALIBRATION_STEPS = 12
WARMUP_UPDATES = 5; TARGET_LR = 2e-5
# Qwen's SETTLED thresholds, reused only as a comparison reference -- NOT enforced as
# a hard stop in this run (see module docstring).
QWEN_KL_BREAKER = 5.0
QWEN_GRAD_BREAKER = 200.0

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'calibration_steps': CALIBRATION_STEPS, **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
from reward_v3 import (score_completion_v2, verify_reward_invariant,  # noqa: E402
    completion_to_text, prompt_to_text)
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt  # noqa: E402

MAIN_REWARD_PARAMS = {}  # score_completion_v2's own defaults (cot_max_scale=2.0, etc.)
_main_margins = verify_reward_invariant()
assert _main_margins['margin_correct_over_wrong'] > 0 and _main_margins['margin_wrong_over_malformed'] > 0
print('REWARD INVARIANT (checkpoint-independent by construction, reused unchanged from Qwen):', _main_margins)

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
train_rows, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
assert hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest() == EXPECTED_CLEAN21_SHA256
TRAIN_POOL = [{'prompt': build_prompt(r['starting_state'], r['operations']), 'ground_truth': r['final_answer']} for r in train_rows]
import random
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL), 'clean21_sha256_verified': True})

print('===== LOAD LLAMA-3-8B BASE + FRESH LORA (no SFT-seed checkpoint -- diagnostic only) =====')
torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token  # confirmed necessary: Llama's pad_token is None by default (Part A Step 2 finding)
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')

torch.cuda.reset_peak_memory_stats()
mem_before_mib = torch.cuda.memory_allocated() / (1024 ** 2)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
model = get_peft_model(base_model, LoraConfig(**LORA_KWARGS))
identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    identity.update(name.encode()); identity.update(param.detach().float().cpu().numpy().tobytes())
print({'model_loaded': True, 'lora_identity_sha256': identity.hexdigest()})

torch.cuda.synchronize()
mem_after_load_mib = torch.cuda.memory_allocated() / (1024 ** 2)
print({'allocated_before_load_mib': mem_before_mib, 'allocated_after_lora_load_mib': mem_after_load_mib})


def chat_wrap(prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]


args = GRPOConfig(output_dir=str(Path(__file__).resolve().parent / '_scratch_calibration'),
    per_device_train_batch_size=1, gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
    max_steps=CALIBRATION_STEPS, learning_rate=TARGET_LR, lr_scheduler_type='linear',
    warmup_steps=min(WARMUP_UPDATES, CALIBRATION_STEPS),
    bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
    logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
    report_to='none', remove_unused_columns=False, disable_dropout=True, seed=RUN_SEED, data_seed=RUN_SEED)

physical_step_box = [0]


def diagnostic_reward(prompts, completions, **kwargs):
    truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
    texts = [completion_to_text(x) for x in completions]
    prompt_texts = [prompt_to_text(x) for x in prompts]
    breakdowns = [score_completion_v2(t, y, physical_step_box[0] or 1, CALIBRATION_STEPS, prompt=p, **MAIN_REWARD_PARAMS)
                  for t, y, p in zip(texts, truths, prompt_texts)]
    return [x['total'] for x in breakdowns]


print('===== BUILD TRAINER =====')
trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
trainer.create_optimizer()
from torch.optim.lr_scheduler import LambdaLR
w = min(WARMUP_UPDATES, CALIBRATION_STEPS)


def lr_factor(i):
    if i < w: return 0.1 + 0.9 * i / max(w - 1, 1)
    decay = CALIBRATION_STEPS - w
    return max(0.0, (CALIBRATION_STEPS - i) / decay) if decay > 0 else 1.0


trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))

telemetry = []


class Cb(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        physical_step_box[0] = int(state.global_step) + 1
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        row = {'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
        telemetry.append(row)
        grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
        print(f"[calibration] step {row['physical_step']}: grad_norm={grad:.3f} kl={kl:.4f} "
              f"(qwen_would_breaker={grad >= QWEN_GRAD_BREAKER or kl >= QWEN_KL_BREAKER})")
        return control


trainer.add_callback(Cb())

print('===== TRAIN (diagnostic only, no breaker enforcement, no checkpoint saved) =====')
trainer.train()
terminal = int(trainer.state.global_step)

torch.cuda.synchronize()
peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)
total_gpu_mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
print({'terminal_step': terminal, 'peak_allocated_during_training_mib': peak_allocated_mib,
       'peak_reserved_during_training_mib': peak_reserved_mib, 'total_gpu_mib': total_gpu_mib,
       'headroom_remaining_mib': total_gpu_mib - peak_reserved_mib})

del trainer, model
torch.cuda.empty_cache()

print('===== ANALYSIS: llama telemetry vs qwen settled thresholds =====')
grad_norms = [r['grad_norm'] for r in telemetry if 'grad_norm' in r]
kls = [r['kl'] for r in telemetry if 'kl' in r]
any_breaker_would_fire = any(
    (r.get('grad_norm', 0) >= QWEN_GRAD_BREAKER or r.get('kl', 0) >= QWEN_KL_BREAKER) for r in telemetry)
analysis = {
    'n_telemetry_rows': len(telemetry),
    'grad_norm_min': min(grad_norms) if grad_norms else None, 'grad_norm_max': max(grad_norms) if grad_norms else None,
    'kl_min': min(kls) if kls else None, 'kl_max': max(kls) if kls else None,
    'qwen_grad_breaker_would_have_fired': any(r.get('grad_norm', 0) >= QWEN_GRAD_BREAKER for r in telemetry),
    'qwen_kl_breaker_would_have_fired': any(r.get('kl', 0) >= QWEN_KL_BREAKER for r in telemetry),
    'any_qwen_breaker_would_have_fired': any_breaker_would_fire,
}
print(analysis)

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9e-llama-calibration-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'calibration_steps': CALIBRATION_STEPS,
               'group_size': GROUP_SIZE, 'target_lr': TARGET_LR,
               'qwen_kl_breaker_reference': QWEN_KL_BREAKER, 'qwen_grad_breaker_reference': QWEN_GRAD_BREAKER,
               'lora_identity_sha256': identity.hexdigest(), 'reward_invariant': _main_margins},
    'memory': {'total_gpu_mib': total_gpu_mib,
               'allocated_after_lora_load_mib': mem_after_load_mib,
               'peak_allocated_during_training_mib': peak_allocated_mib,
               'peak_reserved_during_training_mib': peak_reserved_mib,
               'headroom_remaining_mib': total_gpu_mib - peak_reserved_mib},
    'terminal_step': terminal,
    'telemetry': telemetry,
    'analysis': analysis,
}
EVENT_LOG = OUTPUT / 'stage9e_llama_calibration.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA CALIBRATION PASS COMPLETE.')
