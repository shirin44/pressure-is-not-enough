from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, re, statistics, sys, warnings
from collections import Counter
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

# Stage 9d: SFT-seeded adversarial RL. Loads Stage 9c's SFT checkpoint (100% held-out
# non-literal usage, 28.6% correct, no RL involved in producing it) and applies real
# RL pressure on top -- a baseline correctness-only control FIRST (does ordinary RL
# alone erode the seeded behavior?), then the full Stage-1-equivalent adversarial
# reward (p_CoT reinstated + r_task + consistency/signal bonuses) as an 8-step dry
# run. No bank/gate/injection machinery -- deliberately returns to Stage 1/Step 0's
# uniform-reward design, since injection's premise (the model never produces
# non-literal tokens on its own) doesn't hold for this checkpoint. Full design and the
# "Stage-1-equivalent" interpretation: design.md.

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

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
RUN_SEED = 20260830
GROUP_SIZE = 8; MAX_DYNAMIC_ATTEMPTS = 3; MAX_NEW_TOKENS = 256
BASELINE_STEPS = 16
DRYRUN_STEPS = 8
WARMUP_UPDATES = 5; TARGET_LR = 1e-6
GRAD_BREAKER = 50.0; KL_BREAKER = 5.0
STAGE9C_ADAPTER_DIR = Path.home() / 'aisi_checkpoints' / 'stage9c-sft-diagnostic-v1' / 'final_adapter'

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED,
       'baseline_steps': BASELINE_STEPS, 'dryrun_steps': DRYRUN_STEPS, **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09D_DIR = _REPO_ROOT / 'experiments' / '09d_sft_seeded_adversarial_rl'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR)); sys.path.insert(0, str(_STAGE09D_DIR))
from reward_v3 import (score_completion_v2, verify_reward_invariant, _extract_answer,  # noqa: E402
    normalize_state_token, parse_state_slots, LITERAL_TOKENS, completion_to_text, prompt_to_text)
from taxonomy import classify_candidate  # noqa: E402
from soft_stops import SoftStopTracker  # noqa: E402
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt, CODED_TRAJECTORIES  # noqa: E402
from advantage_clamp import ADVANTAGE_CLAMP_VALUE, GRUBBS_MAX_ADVANTAGE, clamp_advantages  # noqa: E402
import os  # noqa: E402

print('===== ADVANTAGE CLAMP: patching GRPOTrainer.compute_loss (design.md, advantage-clamp addendum) =====')
_original_compute_loss = GRPOTrainer.compute_loss
ADVANTAGE_CLAMP_STATE = {'engaged_count': 0, 'total_count': 0}
# 2026-08-30 signal-preservation + per-config-split investigation: CURRENT_CLAMP_VALUE
# is a mutable box (not a bare global) so run_phase() can set a DIFFERENT clamp value
# per phase -- read fresh on every compute_loss call, not frozen at patch-install time
# -- enabling a baseline-specific vs main-specific clamp test without editing this file
# between launches. Defaults to the shared ADVANTAGE_CLAMP_VALUE when no per-phase
# override is given. PER_ROW_CAPTURES records every row's (physical_step, run_label,
# pre_clamp_advantage, post_clamp_advantage) -- both values already computed in memory
# at this exact point, just not previously persisted -- for the signal-preservation
# (CHECK 1) and per-config structural-diagnosis (CHECK 2) analyses; since raw/pre-clamp
# advantage is clamp-VALUE-invariant (clamping happens strictly after), this single
# capture serves both checks from one run regardless of which clamp value(s) are active.
CURRENT_CLAMP_VALUE = [ADVANTAGE_CLAMP_VALUE]
CURRENT_RUN_LABEL = ['unset']
PER_ROW_CAPTURES = []

def _advantage_clamped_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    adv = inputs.get('advantages')
    if adv is not None:
        clamp_value = CURRENT_CLAMP_VALUE[0]
        clamped, engaged = clamp_advantages(adv, clamp_value)
        ADVANTAGE_CLAMP_STATE['engaged_count'] += int(engaged.sum().item())
        ADVANTAGE_CLAMP_STATE['total_count'] += adv.numel()
        pre_vals = adv.detach().float().cpu().flatten().tolist()
        post_vals = clamped.detach().float().cpu().flatten().tolist()
        for pre, post in zip(pre_vals, post_vals):
            PER_ROW_CAPTURES.append({'run_label': CURRENT_RUN_LABEL[0], 'clamp_value': clamp_value,
                                      'pre_clamp_advantage': pre, 'post_clamp_advantage': post})
        inputs['advantages'] = clamped
    return _original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

GRPOTrainer.compute_loss = _advantage_clamped_compute_loss
assert GRPOTrainer.compute_loss is _advantage_clamped_compute_loss
print({'advantage_clamp_value': ADVANTAGE_CLAMP_VALUE, 'grubbs_max_advantage': GRUBBS_MAX_ADVANTAGE,
       'clamp_below_theoretical_ceiling': ADVANTAGE_CLAMP_VALUE < GRUBBS_MAX_ADVANTAGE})

STEP0_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)
MAIN_REWARD_PARAMS = {}  # empty -> score_completion_v2's own defaults (cot_max_scale=2.0, etc.)

_baseline_margins = verify_reward_invariant(
    consistency_magnitude=STEP0_REWARD_PARAMS['consistency_magnitude'],
    signal_magnitude=STEP0_REWARD_PARAMS['signal_magnitude'],
    cot_max_scale=STEP0_REWARD_PARAMS['cot_max_scale'])
_main_margins = verify_reward_invariant()
assert _baseline_margins['margin_correct_over_wrong'] > 0 and _baseline_margins['margin_wrong_over_malformed'] > 0
assert _main_margins['margin_correct_over_wrong'] > 0 and _main_margins['margin_wrong_over_malformed'] > 0
print('REWARD INVARIANT (checkpoint-independent by construction -- design.md section 2):',
      {'baseline': _baseline_margins, 'main_adversarial': _main_margins})

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
train_rows, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
assert hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest() == EXPECTED_CLEAN21_SHA256
TRAIN_POOL = [{'prompt': build_prompt(r['starting_state'], r['operations']), 'ground_truth': r['final_answer']} for r in train_rows]
import random
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL), 'eval_n': len(eval_rows), 'clean21_sha256_verified': True})

CODE_WORDS = ('nib', 'nomo')


print('===== LOAD 3B BASE (once, shared across both runs) =====')
torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')

STAGE9C_WEIGHTS = STAGE9C_ADAPTER_DIR / 'adapter_model.safetensors'
if not STAGE9C_WEIGHTS.is_file(): raise FileNotFoundError(STAGE9C_WEIGHTS)
_stage9c_bytes_sha = hashlib.sha256(STAGE9C_WEIGHTS.read_bytes()).hexdigest()
print({'stage9c_adapter_path': str(STAGE9C_WEIGHTS), 'stage9c_adapter_sha256': _stage9c_bytes_sha})

def load_stage9c_checkpoint():
    """A genuinely FRESH, independent base-model load plus a fresh PeftModel wrapper,
    loaded with Stage 9c's saved LoRA weights -- single adapter, no composition.
    Deliberately does NOT share a base_model object across calls: TRL's GRPOTrainer
    (beta!=0, PEFT) calls model.add_adapter("ref", ...) directly onto the wrapped
    base model to hold a frozen reference-policy snapshot (trl/trainer/grpo_trainer.py
    -- confirmed by reading the installed source, not assumed) -- a real, by-design
    mechanism, but one that leaves a 'ref' adapter registered on the shared base model
    across trainer instances if the same base_model object is reused for a second
    get_peft_model() call, as this script's first version did. Caught by the
    'unexpected adapter composition' assertion below on Phase 2's checkpoint load
    (['default', 'ref'] instead of ['default']) -- fixed by never reusing base_model
    across calls, rather than by relying on an unverified assumption that PEFT's
    active-adapter isolation would have made the leaked 'ref' adapter harmless. Costs
    a few extra seconds per call (re-loading ~3GB of 8-bit weights) in exchange for
    zero ambiguity about checkpoint purity -- worth it for a diagnostic run whose
    checkpoint identity is exactly what's under scrutiny."""
    from peft import set_peft_model_state_dict
    fresh_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = False
    fresh_base = prepare_model_for_kbit_training(fresh_base, use_gradient_checkpointing=True)
    model = get_peft_model(fresh_base, LoraConfig(**LORA_KWARGS))
    result = set_peft_model_state_dict(model, load_safetensors(str(STAGE9C_WEIGHTS)), adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise RuntimeError(f'Unexpected Stage 9c adapter keys: {result.unexpected_keys}')
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    identity = hashlib.sha256()
    for name, param in model.named_parameters():
        if '.default.' not in name: continue
        identity.update(name.encode()); identity.update(param.detach().float().cpu().numpy().tobytes())
    return model, identity.hexdigest()

ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= width and row[-width:].tolist() == ANSWER_IDS for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])


print('===== ZERO-STEP SANITY CHECK: does the freshly-loaded Stage 9c checkpoint reproduce its own reported ~100% held-out non-literal rate? =====')
_sanity_model, _sanity_identity = load_stage9c_checkpoint()
_sanity_model.eval(); _sanity_model.config.use_cache = True
def _generate_batch(model, prompts_texts, max_new_tokens, do_sample=False):
    out = []
    for start in range(0, len(prompts_texts), 25):
        chunk = prompts_texts[start:start + 25]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        out.extend(tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True))
    return out
def _chat_wrap(prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]

_sanity_prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
_sanity_completions = _generate_batch(_sanity_model, _chat_wrap(_sanity_prompts), MAX_NEW_TOKENS)
_sanity_nonliteral = sum(1 for c in _sanity_completions if any(t not in LITERAL_TOKENS for _i, t in parse_state_slots(c)))
_sanity_rate = _sanity_nonliteral / len(_sanity_completions)
print({'sanity_checkpoint_identity_sha256': _sanity_identity, 'sanity_nonliteral_rate': _sanity_rate,
       'n': len(_sanity_completions),
       'matches_stage9c_report': abs(_sanity_rate - 1.0) < 0.15})
if _sanity_rate < 0.7:
    raise RuntimeError(f'Checkpoint loaded but sanity non-literal rate ({_sanity_rate:.3f}) is far below '
                        f'Stage 9c\'s own reported ~100% -- STOP, do not proceed, checkpoint identity/loading '
                        f'is suspect. Reported, not silently ignored.')
del _sanity_model
torch.cuda.empty_cache()
print('PASSED: checkpoint loads correctly and reproduces its own reported behavior before any Stage 9d training.')


def build_trainer(model, n_steps, reward_params, run_label):
    args = GRPOConfig(output_dir=str(_STAGE09D_DIR / f'_scratch_{run_label}'), per_device_train_batch_size=1,
        gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
        max_steps=n_steps, learning_rate=TARGET_LR, lr_scheduler_type='linear', warmup_steps=min(WARMUP_UPDATES, n_steps),
        bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
        max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
        logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
        report_to='none', remove_unused_columns=False, disable_dropout=True, seed=RUN_SEED, data_seed=RUN_SEED)

    physical_step_box = [0]
    def diagnostic_reward(prompts, completions, **kwargs):
        truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
        texts = [completion_to_text(x) for x in completions]
        prompt_texts = [prompt_to_text(x) for x in prompts]
        breakdowns = [score_completion_v2(t, y, physical_step_box[0] or 1, n_steps, prompt=p, **reward_params)
                      for t, y, p in zip(texts, truths, prompt_texts)]
        return [x['total'] for x in breakdowns]

    trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    trainer.create_optimizer()
    from torch.optim.lr_scheduler import LambdaLR
    w = min(WARMUP_UPDATES, n_steps)
    def lr_factor(i):
        if i < w: return 0.1 + 0.9 * i / max(w - 1, 1)
        decay = n_steps - w
        return max(0.0, (n_steps - i) / decay) if decay > 0 else 1.0
    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))

    original_generate = model.generate
    def generate_stopped(*a, **kw):
        kw.setdefault('stopping_criteria', ANSWER_STOP)
        return original_generate(*a, **kw)
    model.generate = generate_stopped
    trainer.model.generate = generate_stopped
    return trainer, physical_step_box


def evaluate_and_track(model, step, label):
    model.eval(); model.config.use_cache = True
    try:
        prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
        completions = _generate_batch(model, _chat_wrap(prompts), MAX_NEW_TOKENS)
        samples = []
        for row, prompt, completion in zip(eval_rows, prompts, completions):
            ans, valid = _extract_answer(completion)
            final_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
            cls = classify_candidate(completion, prompt, row['final_answer'])
            tokens = [t for _i, t in parse_state_slots(completion)]
            nonliteral_tokens = [t for t in tokens if t not in LITERAL_TOKENS]
            samples.append({'completion': completion, 'final_answer_correct': final_correct,
                'classification': cls, 'nonliteral_tokens': nonliteral_tokens})
        n = len(samples)
        nonliteral_rate = sum(1 for s in samples if s['nonliteral_tokens']) / n
        structural_nonliteral = [s for s in samples if s['classification']['structural_nonliteral_candidate']]
        genuine_correct_among_nonliteral = (sum(1 for s in structural_nonliteral if s['final_answer_correct']) / len(structural_nonliteral)
                                             if structural_nonliteral else None)
        taxonomy_counts = dict(Counter(s['classification']['category_name'] for s in samples))
        token_histogram = Counter(t for s in samples for t in s['nonliteral_tokens'])
        other_token_count = sum(v for k, v in token_histogram.items() if k not in CODE_WORDS)
        result = {'step': step, 'label': label, 'n': n,
            'nonliteral_rate': nonliteral_rate,
            'structural_nonliteral_candidate_rate': len(structural_nonliteral) / n,
            'genuine_correct_among_structural_nonliteral': genuine_correct_among_nonliteral,
            'overall_final_answer_accuracy': sum(1 for s in samples if s['final_answer_correct']) / n,
            'taxonomy_counts': taxonomy_counts,
            'token_identity_histogram': dict(token_histogram),
            'drifted_from_nib_nomo_count': other_token_count,
            'samples': samples}
        print(f'[{label}] MILESTONE STEP {step}: nonliteral_rate={nonliteral_rate:.3f} '
              f'structural_nonliteral_rate={result["structural_nonliteral_candidate_rate"]:.3f} '
              f'genuine_correct_among_nonliteral={genuine_correct_among_nonliteral} '
              f'overall_acc={result["overall_final_answer_accuracy"]:.3f} '
              f'taxonomy={taxonomy_counts} token_drift_count={other_token_count}')
        return result
    finally:
        model.config.use_cache = False; model.train()


def run_phase(n_steps, reward_params, run_label, milestone_every, clamp_value=None):
    clamp_value = ADVANTAGE_CLAMP_VALUE if clamp_value is None else clamp_value
    print(f'===== PHASE: {run_label} ({n_steps} steps), clamp_value={clamp_value} =====')
    CURRENT_CLAMP_VALUE[0] = clamp_value
    CURRENT_RUN_LABEL[0] = run_label
    ADVANTAGE_CLAMP_STATE['engaged_count'] = 0
    ADVANTAGE_CLAMP_STATE['total_count'] = 0
    phase_capture_start = len(PER_ROW_CAPTURES)
    model, identity = load_stage9c_checkpoint()
    print({'run_label': run_label, 'loaded_checkpoint_identity_sha256': identity})
    trainer, physical_step_box = build_trainer(model, n_steps, reward_params, run_label)

    def breaker(grad_norm, kl): return grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER
    telemetry = []; milestones = []; hard_stop = [None]; soft_stop = [None]
    SOFT_STOP = SoftStopTracker()

    class Cb(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            physical_step_box[0] = int(state.global_step) + 1
            return control
        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step > 0 and (step % milestone_every == 0 or step == n_steps):
                # model is run_phase's own local, closed over correctly at runtime by
                # this nested callback method; pyflakes doesn't track closures through
                # a class body reliably, hence the false-positive F821 here.
                milestones.append(evaluate_and_track(model, step, run_label))  # noqa: F821
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

    trainer.train()
    terminal = int(trainer.state.global_step)
    advantage_clamp_summary = {
        'clamp_value': clamp_value, 'engaged_count': ADVANTAGE_CLAMP_STATE['engaged_count'],
        'total_count': ADVANTAGE_CLAMP_STATE['total_count'],
        'engagement_rate': (ADVANTAGE_CLAMP_STATE['engaged_count'] / ADVANTAGE_CLAMP_STATE['total_count']
                             if ADVANTAGE_CLAMP_STATE['total_count'] else None),
    }
    phase_per_row_advantages = list(PER_ROW_CAPTURES[phase_capture_start:])
    del trainer, model
    torch.cuda.empty_cache()
    return {'run_label': run_label, 'n_steps_requested': n_steps, 'terminal_step': terminal,
            'hard_stop': hard_stop[0], 'soft_stop': soft_stop[0], 'telemetry': telemetry, 'milestones': milestones,
            'checkpoint_identity_sha256': identity, 'advantage_clamp_summary': advantage_clamp_summary,
            'per_row_advantages': phase_per_row_advantages}


# Per-phase clamp override (2026-08-30 signal-preservation + per-config-split
# investigation): STAGE9D_ADVANTAGE_CLAMP_BASELINE / STAGE9D_ADVANTAGE_CLAMP_MAIN, each
# falling back to the shared ADVANTAGE_CLAMP_VALUE (itself STAGE9D_ADVANTAGE_CLAMP_VALUE
# or the 0.4 adopted default) when unset -- lets one launch test a per-config split
# without editing this file.
BASELINE_CLAMP_VALUE = float(os.environ.get('STAGE9D_ADVANTAGE_CLAMP_BASELINE', str(ADVANTAGE_CLAMP_VALUE)))
MAIN_CLAMP_VALUE = float(os.environ.get('STAGE9D_ADVANTAGE_CLAMP_MAIN', str(ADVANTAGE_CLAMP_VALUE)))
print({'baseline_clamp_value': BASELINE_CLAMP_VALUE, 'main_clamp_value': MAIN_CLAMP_VALUE})

print('===== PHASE 1: BASELINE CONTROL (correctness-only, no p_cot/consistency/signal) =====')
baseline_result = run_phase(BASELINE_STEPS, STEP0_REWARD_PARAMS, 'baseline_correctness_only', milestone_every=4,
                             clamp_value=BASELINE_CLAMP_VALUE)

print('===== PHASE 2: MAIN ADVERSARIAL DRY RUN (full Stage-1-equivalent reward, p_cot reinstated) =====')
main_result = run_phase(DRYRUN_STEPS, MAIN_REWARD_PARAMS, 'main_adversarial_dryrun', milestone_every=4,
                         clamp_value=MAIN_CLAMP_VALUE)


print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9d-sft-seeded-adversarial-rl-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'stage9c_adapter_sha256': _stage9c_bytes_sha,
               'sanity_check_identity_sha256': _sanity_identity, 'sanity_nonliteral_rate': _sanity_rate,
               'reward_invariant': {'baseline': _baseline_margins, 'main_adversarial': _main_margins},
               'baseline_reward_params': STEP0_REWARD_PARAMS, 'main_reward_params': 'score_completion_v2 defaults (cot_max_scale=2.0, consistency_magnitude=0.15, signal_magnitude=0.15)'},
    'baseline_result': baseline_result,
    'main_dryrun_result': main_result,
}
EVENT_LOG = OUTPUT / 'stage9d_sft_seeded_adversarial_rl.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({
    'baseline_terminal_step': baseline_result['terminal_step'], 'baseline_hard_stop': baseline_result['hard_stop'],
    'baseline_advantage_clamp_summary': baseline_result['advantage_clamp_summary'],
    'baseline_milestones_summary': [{'step': m['step'], 'nonliteral_rate': m['nonliteral_rate'],
        'structural_nonliteral_rate': m['structural_nonliteral_candidate_rate'],
        'genuine_correct_among_nonliteral': m['genuine_correct_among_structural_nonliteral'],
        'overall_acc': m['overall_final_answer_accuracy']} for m in baseline_result['milestones']],
    'main_terminal_step': main_result['terminal_step'], 'main_hard_stop': main_result['hard_stop'],
    'main_advantage_clamp_summary': main_result['advantage_clamp_summary'],
    'main_milestones_summary': [{'step': m['step'], 'nonliteral_rate': m['nonliteral_rate'],
        'structural_nonliteral_rate': m['structural_nonliteral_candidate_rate'],
        'genuine_correct_among_nonliteral': m['genuine_correct_among_structural_nonliteral'],
        'overall_acc': m['overall_final_answer_accuracy']} for m in main_result['milestones']],
}, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9D BASELINE + MAIN DRY RUN COMPLETE. No longer main run launched.')
