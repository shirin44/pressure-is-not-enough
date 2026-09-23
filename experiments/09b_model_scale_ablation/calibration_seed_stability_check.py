from __future__ import annotations

import gc, hashlib, importlib.metadata, json, logging, math, os, random, statistics, sys, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# Stage 9b: seed-stability check for the nonzero-only KL-clamp pooling fix (design.md
# section 4 addendum). Loads the 7B base ONCE, then runs the 6-step calibration procedure
# 4 times with different seeds (fresh LoRA each time, discarded after), reporting D_MAX
# under BOTH the old (all-values) and new (nonzero-only) pooling methods per seed, so the
# fix's actual effect is visible, not just asserted. This is a standalone verification
# script, not part of the training pipeline -- injection_dryrun_7b.py is updated
# separately once this confirms the fix is stable.

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

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
GROUP_SIZE = 8; MAX_NEW_TOKENS = 256
CALIBRATION_STEPS = 6
WARMUP_UPDATES = 6; TARGET_LR = 1e-6
GRAD_BREAKER = 50.0; KL_BREAKER = 5.0
SEEDS_TO_CHECK = [20260829, 30000001, 30000002, 30000003]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09B_DIR = _REPO_ROOT / 'experiments' / '09b_model_scale_ablation'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09B_DIR))
from reward_v3 import verify_reward_invariant  # noqa: E402
from synthetic_bridge import CODED_TRAJECTORIES, build_clean_length5_train_eval_split  # noqa: E402
from step14_reward_gate import score_completion_gated, verify_step14_reward_invariant, bank_gate_resample_reasons  # noqa: E402,F401
from kl_calibration import compute_clamp_from_pool, solve_kl_clamp_bound  # noqa: E402

print('===== SANITY CHECKS (unchanged, model-independent) =====')
_ = verify_step14_reward_invariant()
_ = verify_reward_invariant()
BRIDGE_REWARD_PARAMS = {}

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
train_rows, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
assert hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest() == EXPECTED_CLEAN21_SHA256

def build_cot_prompt(starting_state, operations):
    lines = [f'Starting state: {starting_state}', 'Instructions:']
    clarified = {'same': 'same as previous (the state does NOT change)', 'different': 'different from previous (the state flips)'}
    for i, op in enumerate(operations, start=1): lines.append(f'{i}. {clarified[op]}')
    lines.append("Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        "tags for intermediate states. After all steps, give exactly one final state "
        "inside <answer>...</answer> tags.")
    return '\n'.join(lines)

TRAIN_POOL = [{'prompt': build_cot_prompt(r['starting_state'], r['operations']),
               'ground_truth': r['final_answer'], 'n_flips': 5} for r in train_rows]
random.Random(20260828).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL)})

print('===== LOAD 7B TOKENIZER/QUANT CONFIG (base model loaded FRESH per trial, see below) =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
# NOTE (2026-08-30, disclosed defensively): base_model used to be loaded ONCE here and
# reused (via get_peft_model) across all 4 seed trials below. That is exactly the
# reused-base_model pattern that leaked a stray "ref" adapter forward across phases in
# Stage 9d (TRL's GRPOTrainer, beta!=0, PEFT: model.add_adapter("ref", ...) registers
# directly onto the shared base model -- trl/trainer/grpo_trainer.py, confirmed by
# reading the installed source). This script uses beta=.04 and never asserted adapter
# composition, so trials 2-4 (every call after the first GRPOTrainer init in the same
# process) could plausibly have run under an unintended ['default', 'ref'] composition
# rather than a clean ['default'] -- see design.md's 2026-08-30 addendum for the full
# accounting of what this script's numeric outputs were and weren't used for downstream.
# Fixed defensively: base_model is now loaded fresh inside run_one_calibration_trial().

# Minimal per-token logps interception -- same mechanism as injection_dryrun_7b.py's patch,
# but only what's needed to pool per-token KL (no full instrumentation logging needed here).
import trl.trainer.grpo_trainer as _grpo_mod
_original_get_logps = GRPOTrainer._get_per_token_logps_and_entropies
_original_compute_loss = GRPOTrainer.compute_loss
_STATE = {'inside_compute_loss': False, 'physical_step': 0, 'pool_all': [], 'pool_by_step': {}}

def _hash_rows(id_tensor):
    return [hashlib.sha256(str(row).encode()).hexdigest() for row in id_tensor.detach().cpu().tolist()]

def _patched_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    logps, entropies, aux_loss = _original_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    is_policy = _STATE['inside_compute_loss']
    _STATE.setdefault('last_calls', []).append({'tag': 'policy' if is_policy else 'reference',
        'physical_step': _STATE['physical_step'], 'row_hashes': _hash_rows(input_ids[:, -logits_to_keep:]),
        'logps': logps.detach().float().cpu().tolist()})
    return logps, entropies, aux_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    _STATE.setdefault('last_cl_calls', []).append({'physical_step': _STATE['physical_step'],
        'completion_ids_hash': _hash_rows(inputs['completion_ids']),
        'completion_mask': inputs['completion_mask'].detach().cpu().tolist()})
    _STATE['inside_compute_loss'] = True
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        _STATE['inside_compute_loss'] = False
    return result

GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_logps
GRPOTrainer.compute_loss = _patched_compute_loss

def run_one_calibration_trial(seed, calibration_steps):
    _STATE['last_calls'] = []; _STATE['last_cl_calls'] = []; _STATE['physical_step'] = 0
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Fresh, independent base-model load per trial -- see the module-level note above for why
    # (never share a base_model object across get_peft_model() calls when beta!=0).
    fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = False
    fresh_base = prepare_model_for_kbit_training(fresh_base, use_gradient_checkpointing=True)
    model = get_peft_model(fresh_base, LoraConfig(**LORA_KWARGS))
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    from torch.optim.lr_scheduler import LambdaLR
    args = GRPOConfig(output_dir=str(_STAGE09B_DIR / '_calib_scratch'), per_device_train_batch_size=1,
        gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
        max_steps=calibration_steps, learning_rate=TARGET_LR, lr_scheduler_type='linear',
        warmup_steps=min(WARMUP_UPDATES, calibration_steps), bf16=True, num_generations=GROUP_SIZE,
        generation_batch_size=GROUP_SIZE, num_iterations=1, max_completion_length=MAX_NEW_TOKENS,
        temperature=.8, top_p=.95, beta=.04, entropy_coef=.05, logging_strategy='steps', logging_steps=1,
        disable_tqdm=True, save_strategy='no', report_to='none', remove_unused_columns=False,
        disable_dropout=True, seed=seed, data_seed=seed)

    def diagnostic_reward(prompts, completions, **kwargs):
        truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
        physical_step = _STATE['physical_step'] or 1
        from reward_v3 import completion_to_text, prompt_to_text
        texts = [completion_to_text(x) for x in completions]
        prompt_texts = [prompt_to_text(x) for x in prompts]
        breakdowns = [score_completion_gated(t, y, physical_step, calibration_steps, prompt=p, **BRIDGE_REWARD_PARAMS)
                      for t, y, p in zip(texts, truths, prompt_texts)]
        return [x['total'] for x in breakdowns]

    trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.create_optimizer()
    def lr_factor(i):
        w = min(WARMUP_UPDATES, calibration_steps)
        if i < w: return 0.1 + 0.9 * i / max(w - 1, 1)
        decay = calibration_steps - w
        return max(0.0, (calibration_steps - i) / decay) if decay > 0 else 1.0
    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))

    telemetry = []
    class Cb(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            _STATE['physical_step'] = int(state.global_step) + 1
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            telemetry.append({'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}})
            return control
    trainer.add_callback(Cb())
    trainer.train()

    _BORDERLINE_GRAD, _BORDERLINE_KL = GRAD_BREAKER / 2, KL_BREAKER / 2
    by_step = {r['physical_step']: r for r in telemetry if 'grad_norm' in r and 'kl' in r}
    clean_steps = {s for s, r in by_step.items() if r['grad_norm'] < _BORDERLINE_GRAD and r['kl'] < _BORDERLINE_KL}

    pool = []
    for cl in _STATE['last_cl_calls']:
        step = cl['physical_step']
        if step not in clean_steps: continue
        row_hash = cl['completion_ids_hash'][0]
        policy_call = next((c for c in _STATE['last_calls'] if c['tag'] == 'policy' and c['physical_step'] == step and c['row_hashes'] == [row_hash]), None)
        ref_call = next((c for c in _STATE['last_calls'] if c['tag'] == 'reference' and c['physical_step'] == step and row_hash in c['row_hashes']), None)
        if policy_call is None or ref_call is None: continue
        ref_idx = ref_call['row_hashes'].index(row_hash)
        policy_logps = policy_call['logps'][0]; ref_logps = ref_call['logps'][ref_idx]; mask = cl['completion_mask'][0]
        if len(ref_logps) != len(policy_logps) or len(mask) != len(policy_logps): continue
        for rp, pp, m in zip(ref_logps, policy_logps, mask):
            if not m: continue
            diff = rp - pp
            pool.append(math.exp(diff) - diff - 1)

    result_all = compute_clamp_from_pool(pool, nonzero_only=False)
    result_nonzero = compute_clamp_from_pool(pool, nonzero_only=True)
    d_max_all = solve_kl_clamp_bound(result_all['clamp_value']) if not result_all['degenerate'] else None
    d_max_nonzero = solve_kl_clamp_bound(result_nonzero['clamp_value']) if not result_nonzero['degenerate'] else None

    del trainer, model
    gc.collect(); torch.cuda.empty_cache()

    return {'seed': seed, 'clean_steps': sorted(clean_steps), 'pool_size': len(pool),
            'zero_count': sum(1 for v in pool if v == 0.0), 'nonzero_count': sum(1 for v in pool if v > 0.0),
            'all_values_method': {'clamp_value': result_all['clamp_value'], 'degenerate': result_all['degenerate'], 'd_max': d_max_all},
            'nonzero_only_method': {'clamp_value': result_nonzero['clamp_value'], 'degenerate': result_nonzero['degenerate'], 'd_max': d_max_nonzero,
                                     'n_nonzero': result_nonzero['n']}}


print(f'===== SEED-STABILITY CHECK: {len(SEEDS_TO_CHECK)} calibration trials, {CALIBRATION_STEPS} steps each =====')
results = []
for seed in SEEDS_TO_CHECK:
    print(f'--- trial seed={seed} ---')
    r = run_one_calibration_trial(seed, CALIBRATION_STEPS)
    results.append(r)
    print(json.dumps(r, indent=2))

print('\n===== SUMMARY =====')
for r in results:
    print(f"seed={r['seed']:>10}  pool_size={r['pool_size']:>4}  zero={r['zero_count']:>4}  nonzero={r['nonzero_count']:>4}  "
          f"D_MAX(all_values)={r['all_values_method']['d_max']}  D_MAX(nonzero_only)={r['nonzero_only_method']['d_max']}")

nonzero_d_maxes = [r['nonzero_only_method']['d_max'] for r in results if r['nonzero_only_method']['d_max'] is not None]
all_d_maxes = [r['all_values_method']['d_max'] for r in results if r['all_values_method']['d_max'] is not None]
print(f"\nnonzero_only D_MAX values: {nonzero_d_maxes}")
print(f"all_values D_MAX values: {all_d_maxes}")
if nonzero_d_maxes:
    print(f"nonzero_only: min={min(nonzero_d_maxes):.6f} max={max(nonzero_d_maxes):.6f} "
          f"ratio(max/min)={max(nonzero_d_maxes)/min(nonzero_d_maxes):.2f} n_degenerate={len(SEEDS_TO_CHECK)-len(nonzero_d_maxes)}")
else:
    print("nonzero_only: ALL TRIALS DEGENERATE -- fix did not work.")
print('\nSEED-STABILITY CHECK COMPLETE.')
