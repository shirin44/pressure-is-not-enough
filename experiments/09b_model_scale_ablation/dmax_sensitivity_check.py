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

# Stage 9b: D_MAX sensitivity check. The KL-clamp calibration fix (nonzero-only pooling)
# reliably avoids degenerate collapse-to-zero, but the actual measured D_MAX value swings
# ~14.5x across runs with the SAME seed (0.050781 to 0.736196), attributed to ordinary
# GPU/CUDA nondeterminism that manual_seed doesn't fully control. Before trusting per-run
# calibration for the real 50-step launch, this checks whether that swing actually matters
# for training dynamics: run the identical 8-step Phase-1-equivalent procedure (same seed,
# same fresh LoRA, same 2-of-8 injection, same reward gate) twice, with D_MAX FIXED to the
# two already-observed endpoints of the swing (no re-calibration), and compare telemetry.
# design.md documents the full context and verdict.

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
RUN_SEED = 20260828  # SAME seed for both trials -- isolates D_MAX as the only varied factor
GROUP_SIZE = 8; MAX_NEW_TOKENS = 256; N_STEPS = 8
WARMUP_UPDATES = 12; TARGET_LR = 1e-6
GRAD_BREAKER = 50.0; KL_BREAKER = 5.0
INJECTIONS_PER_GROUP = 2
BANK_MODE = 'coded'
# Fixed entropy clamp (measured value from the already-completed, clean dry run), held
# constant across both trials so D_MAX is the ONLY varied quantity -- not re-measured here.
ENTROPY_CLAMP_VALUE = 1.2189
# The two already-observed endpoints of the D_MAX swing: low = seed-stability check's first
# trial (calibration_seed_stability_check.py, seed=20260829), high = the actual completed
# 8-step dry run's own calibration (injection_dryrun_7b.py, same nominal seed). Not
# re-calibrated here -- used exactly as already measured.
D_MAX_TRIALS = [('low', 0.050781), ('high', 0.736196)]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09B_DIR = _REPO_ROOT / 'experiments' / '09b_model_scale_ablation'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09B_DIR))
from reward_v3 import verify_reward_invariant  # noqa: E402
from synthetic_bridge import CODED_TRAJECTORIES, build_clean_length5_train_eval_split  # noqa: E402
from step14_reward_gate import (score_completion_gated, verify_step14_reward_invariant,  # noqa: E402
    bank_gate_resample_reasons, is_bank_scenario, BANK_SCENARIO_KEYS)
from step14b_injection import verify_grpo_ratio_is_unconditionally_one  # noqa: E402
from injection_scaling import inject_into_generation_output_n  # noqa: E402

print('===== SANITY CHECKS (unchanged, model-independent) =====')
_step14_invariant_report = verify_step14_reward_invariant()
_bridge_margins = verify_reward_invariant()
BRIDGE_REWARD_PARAMS = {}
ACTIVE_TRAJECTORIES = CODED_TRAJECTORIES
assert BANK_SCENARIO_KEYS == {(r['starting_state'], tuple(r['operations'])) for r in ACTIVE_TRAJECTORIES}

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

TRAIN_POOL = [{'prompt': build_cot_prompt(r['starting_state'], r['operations']), 'ground_truth': r['final_answer'], 'n_flips': 5} for r in train_rows]
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL)})

print('===== LOAD 7B TOKENIZER/QUANT CONFIG (base model loaded FRESH per D_MAX trial, see below) =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
# NOTE (2026-08-30, disclosed defensively): base_model used to be loaded ONCE here and
# reused (via get_peft_model) across both D_MAX trials below. That is exactly the
# reused-base_model pattern that leaked a stray "ref" adapter forward across phases in
# Stage 9d (TRL's GRPOTrainer, beta!=0, PEFT: model.add_adapter("ref", ...) registers
# directly onto the shared base model -- trl/trainer/grpo_trainer.py, confirmed by
# reading the installed source). This script uses beta=.04 and never asserted adapter
# composition, so the "high" D_MAX trial (the second get_peft_model() call in the
# process) could plausibly have run under an unintended ['default', 'ref'] composition
# rather than a clean ['default'] -- see design.md's 2026-08-30 addendum for the full
# accounting of what this script's numeric outputs were and weren't used for downstream.
# Fixed defensively: base_model is now loaded fresh inside run_one_dmax_trial().

from transformers import StoppingCriteria, StoppingCriteriaList
ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= width and row[-width:].tolist() == ANSWER_IDS for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])

print('===== KL/ENTROPY PATCH INFRASTRUCTURE (installed once, D_MAX fixed per trial, never calibrated here) =====')
import trl.trainer.grpo_trainer as _grpo_mod
_original_entropy_from_logits = _grpo_mod.entropy_from_logits
def _clamped_entropy_from_logits(logits, chunk_size=128):
    return torch.clamp(_original_entropy_from_logits(logits, chunk_size=chunk_size), max=ENTROPY_CLAMP_VALUE)
_grpo_mod.entropy_from_logits = _clamped_entropy_from_logits

_original_get_logps = GRPOTrainer._get_per_token_logps_and_entropies
_original_compute_loss = GRPOTrainer.compute_loss
INSTRUMENTATION = {'inside_compute_loss': False, 'physical_step': 0, 'logps_calls': [], 'compute_loss_calls': []}
KL_CLAMP_STATE = {'enabled': True, 'd_max': None, 'current_inputs': None, 'engaged_token_count': 0, 'total_token_count': 0}

def _hash_rows(id_tensor):
    return [hashlib.sha256(str(row).encode()).hexdigest() for row in id_tensor.detach().cpu().tolist()]

def _patched_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    logps, entropies, aux_loss = _original_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    is_policy = INSTRUMENTATION['inside_compute_loss']
    if is_policy and KL_CLAMP_STATE['enabled'] and KL_CLAMP_STATE['current_inputs'] is not None:
        current_inputs = KL_CLAMP_STATE['current_inputs']
        ref = current_inputs.get('ref_per_token_logps')
        if ref is not None and ref.shape == logps.shape:
            diff = ref - logps.detach()
            d_max = KL_CLAMP_STATE['d_max']
            diff_clamped = torch.clamp(diff, min=-d_max, max=d_max)
            engaged = (diff.abs() > d_max)
            KL_CLAMP_STATE['engaged_token_count'] += int(engaged.sum().item())
            KL_CLAMP_STATE['total_token_count'] += diff.numel()
            current_inputs['ref_per_token_logps'] = (logps.detach() + diff_clamped).detach()
    return logps, entropies, aux_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    INSTRUMENTATION['inside_compute_loss'] = True
    KL_CLAMP_STATE['current_inputs'] = inputs
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        INSTRUMENTATION['inside_compute_loss'] = False
        KL_CLAMP_STATE['current_inputs'] = None
    return result

GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_logps
GRPOTrainer.compute_loss = _patched_compute_loss


def run_one_dmax_trial(label, d_max):
    KL_CLAMP_STATE['d_max'] = d_max
    KL_CLAMP_STATE['engaged_token_count'] = 0
    KL_CLAMP_STATE['total_token_count'] = 0
    random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
    # Fresh, independent base-model load per trial -- see the module-level note above for why
    # (never share a base_model object across get_peft_model() calls when beta!=0).
    fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = False
    fresh_base = prepare_model_for_kbit_training(fresh_base, use_gradient_checkpointing=True)
    model = get_peft_model(fresh_base, LoraConfig(**LORA_KWARGS))
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    from torch.optim.lr_scheduler import LambdaLR
    args = GRPOConfig(output_dir=str(_STAGE09B_DIR / '_dmax_scratch'), per_device_train_batch_size=1,
        gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
        max_steps=N_STEPS, learning_rate=TARGET_LR, lr_scheduler_type='linear', warmup_steps=WARMUP_UPDATES,
        bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
        max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
        logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
        report_to='none', remove_unused_columns=False, disable_dropout=True, seed=RUN_SEED, data_seed=RUN_SEED)

    trainer_ref = [None]
    scenario_log = []
    def diagnostic_reward(prompts, completions, **kwargs):
        truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
        physical_step = INSTRUMENTATION['physical_step'] or 1
        from reward_v3 import completion_to_text, prompt_to_text
        texts = [completion_to_text(x) for x in completions]
        prompt_texts = [prompt_to_text(x) for x in prompts]
        scenario_log.append({'physical_step': physical_step, 'prompts': prompt_texts,
                              'is_bank': [is_bank_scenario(p) for p in prompt_texts]})
        breakdowns = [score_completion_gated(t, y, physical_step, N_STEPS, prompt=p, **BRIDGE_REWARD_PARAMS)
                      for t, y, p in zip(texts, truths, prompt_texts)]
        return [x['total'] for x in breakdowns]

    trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer_ref[0] = trainer
    trainer.create_optimizer()
    def lr_factor(i):
        if i < WARMUP_UPDATES: return 0.1 + 0.9 * i / max(WARMUP_UPDATES - 1, 1)
        decay = N_STEPS - WARMUP_UPDATES
        return max(0.0, (N_STEPS - i) / decay) if decay > 0 else 1.0
    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))
    assert verify_grpo_ratio_is_unconditionally_one(
        gradient_accumulation_steps=int(trainer.args.gradient_accumulation_steps),
        steps_per_generation=int(trainer.args.steps_per_generation), num_iterations=int(trainer.args.num_iterations))

    original_generate = model.generate
    def generate_stopped(*a, **kw):
        kw.setdefault('stopping_criteria', ANSWER_STOP)
        return original_generate(*a, **kw)
    model.generate = generate_stopped
    trainer.model.generate = generate_stopped

    injection_log = []
    _original_generate_method = type(trainer)._generate
    def _injection_aware_generate(self, prompts):
        result = _original_generate_method(self, prompts)
        prompt_ids, completion_ids, tool_mask, completions, logprobs, extra_fields, images, tool_images = result
        new_completion_ids, new_completions, injected = inject_into_generation_output_n(prompts, completion_ids, completions, tokenizer, INJECTIONS_PER_GROUP)
        if injected:
            injection_log.append({'physical_step': INSTRUMENTATION['physical_step'], 'injected_indices': list(injected.keys())})
        return prompt_ids, new_completion_ids, tool_mask, new_completions, logprobs, extra_fields, images, tool_images
    type(trainer)._generate = _injection_aware_generate

    def breaker(grad_norm, kl): return grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER
    telemetry = []
    hard_stop = [None]
    class Cb(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            INSTRUMENTATION['physical_step'] = int(state.global_step) + 1
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            row = {'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
            telemetry.append(row)
            grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
            if not all(math.isfinite(x) for x in (grad, kl)) or breaker(grad, kl):
                hard_stop[0] = {'step': int(state.global_step), 'grad_norm': grad, 'kl': kl}
                control.should_training_stop = True
            return control
    trainer.add_callback(Cb())

    print(f'--- D_MAX trial "{label}" = {d_max} ---')
    trainer.train()
    terminal = int(trainer.state.global_step)

    grads = [r['grad_norm'] for r in telemetry if 'grad_norm' in r and math.isfinite(r['grad_norm'])]
    kls = [r['kl'] for r in telemetry if 'kl' in r and math.isfinite(r['kl'])]
    result = {
        'label': label, 'd_max': d_max, 'terminal_step': terminal, 'hard_stop': hard_stop[0],
        'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
                                      if KL_CLAMP_STATE['total_token_count'] else None),
        'engaged_token_count': KL_CLAMP_STATE['engaged_token_count'], 'total_token_count': KL_CLAMP_STATE['total_token_count'],
        'grad_norm_max': max(grads) if grads else None, 'grad_norm_mean': statistics.fmean(grads) if grads else None,
        'kl_max': max(kls) if kls else None, 'kl_mean': statistics.fmean(kls) if kls else None,
        'injection_events': len(injection_log), 'injection_log': injection_log,
        'scenario_log': [{'physical_step': s['physical_step'], 'is_bank': s['is_bank'][0] if s['is_bank'] else None,
                           'prompt_hash': hashlib.sha256(s['prompts'][0].encode()).hexdigest()[:16]} for s in scenario_log],
        'telemetry': telemetry,
    }
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    return result


print(f'===== D_MAX SENSITIVITY CHECK: {len(D_MAX_TRIALS)} trials, {N_STEPS} steps each, ENTROPY_CLAMP_VALUE={ENTROPY_CLAMP_VALUE} fixed =====')
all_results = []
for label, d_max in D_MAX_TRIALS:
    r = run_one_dmax_trial(label, d_max)
    all_results.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != 'telemetry'}, indent=2, default=str))

print('\n===== SCENARIO-DRAW MATCH CHECK (same seed -> should be identical prompt sequence) =====')
low_scenarios = all_results[0]['scenario_log']
high_scenarios = all_results[1]['scenario_log']
scenario_match = [a['prompt_hash'] == b['prompt_hash'] for a, b in zip(low_scenarios, high_scenarios)]
print({'n_steps_compared': len(scenario_match), 'all_match': all(scenario_match),
       'mismatches': [i for i, m in enumerate(scenario_match) if not m]})

print('\n===== SUMMARY =====')
for r in all_results:
    print(f"{r['label']:>5} (D_MAX={r['d_max']}): terminal_step={r['terminal_step']} hard_stop={r['hard_stop']} "
          f"clamp_engagement_rate={r['kl_clamp_engagement_rate']} "
          f"grad_norm(max/mean)={r['grad_norm_max']}/{r['grad_norm_mean']} "
          f"kl(max/mean)={r['kl_max']}/{r['kl_mean']} injection_events={r['injection_events']}")

with open(_STAGE09B_DIR / 'dmax_sensitivity_check_result.json', 'w') as f:
    json.dump({'results': all_results, 'scenario_match': {'all_match': all(scenario_match), 'mismatches': [i for i, m in enumerate(scenario_match) if not m]}}, f, indent=2, default=str)
print('\nSaved full result to dmax_sensitivity_check_result.json')
print('D_MAX SENSITIVITY CHECK COMPLETE.')
