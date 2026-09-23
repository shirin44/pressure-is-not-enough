from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, os, sys, warnings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# Stage 9d follow-up #3: entropy (single-completion confidence) and rollout-group
# homogeneity/advantage-extremity were both directly measured and disconfirmed as the
# cause of the step-3 grad_norm breaker trip. This inspects the actual quantity
# directly: per-token GRADIENT contribution at the spike step, comparing Nib/Nomo
# positions against all other positions -- the same interception methodology this
# project used for the per-token KL clamp (Stage 4 precedent: an aggregate/mean metric
# can hide an extreme per-token outlier). Uses retain_grad() on the per-token policy
# logprob tensor to capture d(loss)/d(logp_i) at every completion position -- a
# precise, position-level gradient-contribution measure, not a loss-magnitude proxy.
# design.md documents the full result.

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
_expected = {'transformers': '5.13.1', 'trl': '1.9.2', 'peft': '0.19.1', 'bitsandbytes': '0.50.0'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
RUN_SEED = 20260830
GROUP_SIZE = 8; MAX_NEW_TOKENS = 256
N_STEPS = 8  # generous margin: prior runs hit the breaker at step 3, but this project has
             # repeatedly demonstrated real run-to-run GPU/CUDA nondeterminism even at a fixed
             # seed (D_MAX calibration swings, malformation-pattern differences) -- 4 steps
             # turned out insufficient on this run's first attempt (no trip by step 3); 8
             # matches this project's standard dry-run budget and gives real margin
WARMUP_UPDATES = 5; TARGET_LR = 1e-6
GRAD_BREAKER = 50.0; KL_BREAKER = 5.0
STAGE9C_ADAPTER_DIR = Path.home() / 'aisi_checkpoints' / 'stage9c-sft-diagnostic-v1' / 'final_adapter'
CODE_WORDS = ('nib', 'nomo')

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'n_steps': N_STEPS, **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
from reward_v3 import score_completion_v2, completion_to_text, prompt_to_text  # noqa: E402
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt  # noqa: E402

STEP0_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)

CLEAN21_SEED = 20260831
train_rows, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
TRAIN_POOL = [{'prompt': build_prompt(r['starting_state'], r['operations']), 'ground_truth': r['final_answer']} for r in train_rows]
import random
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL)})


print('===== LOAD 3B BASE + STAGE 9C CHECKPOINT (fresh, independent load) =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
STAGE9C_WEIGHTS = STAGE9C_ADAPTER_DIR / 'adapter_model.safetensors'
if not STAGE9C_WEIGHTS.is_file(): raise FileNotFoundError(STAGE9C_WEIGHTS)
_stage9c_bytes_sha = hashlib.sha256(STAGE9C_WEIGHTS.read_bytes()).hexdigest()

torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, LoraConfig(**LORA_KWARGS))
result = set_peft_model_state_dict(model, load_safetensors(str(STAGE9C_WEIGHTS)), adapter_name='default')
if getattr(result, 'unexpected_keys', None):
    raise RuntimeError(f'Unexpected Stage 9c adapter keys: {result.unexpected_keys}')
assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
print({'stage9c_adapter_sha256': _stage9c_bytes_sha, 'adapter_composition': list(model.peft_config)})

ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= width and row[-width:].tolist() == ANSWER_IDS for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])


print('===== PATCH: retain_grad() on the per-token POLICY logprob tensor, per micro-batch =====')
if not hasattr(GRPOTrainer, '_get_per_token_logps_and_entropies') or not hasattr(GRPOTrainer, 'compute_loss'):
    raise ImportError('Expected GRPOTrainer methods not found; do not proceed unverified.')
_original_get_logps = GRPOTrainer._get_per_token_logps_and_entropies
_original_compute_loss = GRPOTrainer.compute_loss

INSTRUMENTATION = {'inside_compute_loss': False, 'physical_step': 0}
# Per micro-batch call: {'physical_step', 'call_index', 'logps' (grad-retaining tensor),
# 'completion_ids' (list[int], the ROW this call scored), 'completion_mask', 'advantage',
# 'reward', 'completion_text', 'prompt_text', 'ground_truth'}.
GRAD_CAPTURES = []
# Per physical step: {'texts', 'prompts', 'truths', 'rewards'} -- captured once per step
# by diagnostic_reward (called once per GROUP, before the 8 per-row compute_loss calls),
# in the SAME row order the 8 subsequent micro-batches process (per_device_train_batch_size=1,
# standard gradient-accumulation order) -- linked to each compute_loss call below via a
# running row-index counter, not re-fetched by any fuzzy matching.
REWARD_CAPTURES = {}

def _hash_id_rows(id_tensor):
    rows = id_tensor.detach().cpu().tolist()
    return [hashlib.sha256(str(row).encode()).hexdigest() for row in rows]

def _patched_get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    logps, entropies, aux_loss = _original_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    if INSTRUMENTATION['inside_compute_loss']:
        logps.retain_grad()  # non-leaf tensor -- must explicitly retain to read .grad after backward
        GRAD_CAPTURES.append({
            'physical_step': INSTRUMENTATION['physical_step'],
            'logps_tensor': logps,  # kept as a live reference; .grad populated after this micro-batch's backward()
            'completion_ids': input_ids[:, -logits_to_keep:].detach().cpu().tolist(),
        })
    return logps, entropies, aux_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    INSTRUMENTATION['inside_compute_loss'] = True
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
    finally:
        INSTRUMENTATION['inside_compute_loss'] = False
    # Attach the completion_mask for the row(s) just scored to the most recent capture(s)
    # from THIS compute_loss call (there's exactly one, since per_device_train_batch_size=1).
    if GRAD_CAPTURES:
        cap = GRAD_CAPTURES[-1]
        cap['completion_mask'] = inputs['completion_mask'].detach().cpu().tolist()
        # advantages: already computed by TRL before compute_loss is called, directly available
        # on inputs -- exactly the value identified in the prior (read-only) task as already
        # in memory at this point, just never persisted before now.
        adv = inputs.get('advantages')
        cap['advantage'] = float(adv.detach().float().cpu().flatten()[0].item()) if adv is not None else None
        # reward/completion/prompt/ground_truth: captured once per physical step by
        # diagnostic_reward (called once per group, before these per-row compute_loss calls),
        # linked here by row order.
        step = INSTRUMENTATION['physical_step']
        row_index = sum(1 for c in GRAD_CAPTURES if c['physical_step'] == step) - 1
        step_rewards = REWARD_CAPTURES.get(step)
        if step_rewards is not None and 0 <= row_index < len(step_rewards['rewards']):
            cap['reward'] = step_rewards['rewards'][row_index]
            cap['completion_text'] = step_rewards['texts'][row_index]
            cap['prompt_text'] = step_rewards['prompts'][row_index]
            cap['ground_truth'] = step_rewards['truths'][row_index]
        else:
            cap['reward'] = None
    return result

GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_per_token_logps_and_entropies
GRPOTrainer.compute_loss = _patched_compute_loss
assert GRPOTrainer._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
assert GRPOTrainer.compute_loss is _patched_compute_loss
print('Patches installed: per-token policy logps now retain_grad()-enabled, one capture per micro-batch.')


args = GRPOConfig(output_dir=str(_REPO_ROOT / 'experiments' / '09d_sft_seeded_adversarial_rl' / '_grad_scratch'),
    per_device_train_batch_size=1, gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
    max_steps=N_STEPS, learning_rate=TARGET_LR, lr_scheduler_type='linear', warmup_steps=min(WARMUP_UPDATES, N_STEPS),
    bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
    logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
    report_to='none', remove_unused_columns=False, disable_dropout=True, seed=RUN_SEED, data_seed=RUN_SEED)

physical_step_box = [0]
def diagnostic_reward(prompts, completions, **kwargs):
    truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
    texts = [completion_to_text(x) for x in completions]
    prompt_texts = [prompt_to_text(x) for x in prompts]
    breakdowns = [score_completion_v2(t, y, physical_step_box[0] or 1, N_STEPS, prompt=p, **STEP0_REWARD_PARAMS)
                  for t, y, p in zip(texts, truths, prompt_texts)]
    rewards = [x['total'] for x in breakdowns]
    # Persisted here (once per group, in row order) so _patched_compute_loss can attach the
    # matching reward/text/prompt/ground_truth to each of the 8 subsequent per-row captures --
    # this is the exact gap the prior (read-only) task identified: these values were already
    # computed here before, just never saved anywhere.
    REWARD_CAPTURES[physical_step_box[0]] = {'texts': texts, 'prompts': prompt_texts,
                                              'truths': list(truths), 'rewards': rewards}
    return rewards

trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args, train_dataset=train_dataset, processing_class=tokenizer)
trainer.create_optimizer()
from torch.optim.lr_scheduler import LambdaLR
w = min(WARMUP_UPDATES, N_STEPS)
def lr_factor(i):
    if i < w: return 0.1 + 0.9 * i / max(w - 1, 1)
    decay = N_STEPS - w
    return max(0.0, (N_STEPS - i) / decay) if decay > 0 else 1.0
trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))

original_generate = model.generate
def generate_stopped(*a, **kw):
    kw.setdefault('stopping_criteria', ANSWER_STOP)
    return original_generate(*a, **kw)
model.generate = generate_stopped
trainer.model.generate = generate_stopped


def find_code_word_token_positions(ids):
    """Span-based word location, NOT per-token exact/substring matching. A prior task
    in this same project (Step 14b's completion-level qualitative analysis) already
    discovered and fixed exactly this pitfall: decoding one token at a time and
    substring-matching against the decoded piece MISSES words split across a token
    boundary (e.g. "Tails" as ">T"+"ails"; the same risk applies to "Nib"/"Nomo" here).
    This script initially repeated that exact mistake -- caught when it produced
    token_share_code=0.0 for a checkpoint independently established to use Nib/Nomo on
    ~100% of completions, which is not plausible and correctly flagged the detection
    as broken rather than the model as having stopped using the code. Fixed here using
    the same reconstructed-text-plus-character-offsets approach that worked before."""
    text_so_far = ''
    offsets = []
    for tid in ids:
        piece = tokenizer.decode([tid])
        start = len(text_so_far)
        text_so_far += piece
        offsets.append((start, len(text_so_far)))
    text_lower = text_so_far.lower()
    code_positions = set()
    for word in CODE_WORDS:
        start = 0
        while True:
            idx = text_lower.find(word, start)
            if idx == -1:
                break
            end = idx + len(word)
            before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
            after_ok = end == len(text_lower) or not text_lower[end].isalpha()
            if before_ok and after_ok:
                for i, (s, e) in enumerate(offsets):
                    if s < end and e > idx:
                        code_positions.add(i)
            start = idx + 1
    return code_positions


def analyze_step_gradients(step, captures):
    """For every micro-batch captured during this physical step: locate Nib/Nomo token
    POSITIONS via find_code_word_token_positions (span-based, not per-token substring
    matching), and split total |grad| by position category."""
    total_grad_all = 0.0; total_grad_code = 0.0
    total_tokens_all = 0; total_tokens_code = 0
    per_row_details = []
    for cap in captures:
        if cap['physical_step'] != step: continue
        logps = cap['logps_tensor']
        if logps.grad is None:
            per_row_details.append({'warning': 'no .grad captured for this row -- backward may not have reached it'})
            continue
        grad = logps.grad.detach().float().cpu()[0]  # (seq_len,) -- batch dim is 1 per micro-batch
        mask = cap.get('completion_mask')
        ids = cap['completion_ids'][0]
        n = min(len(ids), grad.shape[0])
        all_code_positions = find_code_word_token_positions(ids[:n])
        code_positions = [i for i in sorted(all_code_positions)
                           if not (mask is not None and i < len(mask[0]) and not mask[0][i])]
        row_grad_abs = grad[:n].abs()
        row_total = float(row_grad_abs.sum().item())
        row_code = float(row_grad_abs[code_positions].sum().item()) if code_positions else 0.0
        total_grad_all += row_total; total_grad_code += row_code
        total_tokens_all += n; total_tokens_code += len(code_positions)
        per_row_details.append({
            'n_tokens': n, 'n_code_tokens': len(code_positions), 'code_positions': code_positions,
            'total_abs_grad': row_total, 'code_abs_grad': row_code,
            'max_abs_grad_position': int(row_grad_abs.argmax().item()), 'max_abs_grad_value': float(row_grad_abs.max().item()),
            'max_grad_is_code_token': int(row_grad_abs.argmax().item()) in code_positions,
            'reward': cap.get('reward'), 'advantage': cap.get('advantage'),
            'completion_text': cap.get('completion_text'), 'prompt_text': cap.get('prompt_text'),
            'ground_truth': cap.get('ground_truth'),
        })
    token_share_code = total_tokens_code / total_tokens_all if total_tokens_all else None
    grad_share_code = total_grad_code / total_grad_all if total_grad_all else None
    return {'step': step, 'total_tokens_all': total_tokens_all, 'total_tokens_code': total_tokens_code,
            'token_share_code': token_share_code, 'total_grad_all': total_grad_all, 'total_grad_code': total_grad_code,
            'grad_share_code': grad_share_code,
            'disproportionality_ratio': (grad_share_code / token_share_code) if (token_share_code and grad_share_code is not None) else None,
            'per_row_details': per_row_details}


telemetry = []; hard_stop = [None]; step_analyses = []
def breaker(grad_norm, kl): return grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER

class Cb(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        physical_step_box[0] = int(state.global_step) + 1
        INSTRUMENTATION['physical_step'] = int(state.global_step) + 1
        return control
    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        analysis = analyze_step_gradients(step, GRAD_CAPTURES)
        step_analyses.append(analysis)
        print(f'STEP {step} GRADIENT ANALYSIS: token_share_code={analysis["token_share_code"]} '
              f'grad_share_code={analysis["grad_share_code"]} disproportionality_ratio={analysis["disproportionality_ratio"]}')
        # Free this step's retained-grad tensors now that the needed numbers are extracted
        # to plain Python/CPU data above -- retain_grad() keeps backward-graph buffers
        # alive per micro-batch; leaving all steps' captures live for the whole run (the
        # first version of this script did) exhausts GPU memory by step 4. Caught by a
        # real CUDA OOM, not anticipated -- fixed by releasing each step's captures right
        # after analysis, not accumulating them for the run's duration.
        remaining = []
        for cap in GRAD_CAPTURES:
            if cap['physical_step'] == step:
                cap['logps_tensor'] = None
            else:
                remaining.append(cap)
        GRAD_CAPTURES[:] = remaining
        torch.cuda.empty_cache()
        return control
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        row = {'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
        telemetry.append(row)
        grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
        print(f'  telemetry: step={row["physical_step"]} grad_norm={grad} kl={kl}')
        if not all(math.isfinite(x) for x in (grad, kl)) or breaker(grad, kl):
            hard_stop[0] = {'step': int(state.global_step), 'grad_norm': grad, 'kl': kl}
            control.should_training_stop = True
        return control
trainer.add_callback(Cb())

print(f'===== MINIMAL REPRO: UP TO {N_STEPS} STEPS, STOP AT FIRST BREAKER =====')
trainer.train()
terminal = int(trainer.state.global_step)
print({'terminal_step': terminal, 'hard_stop': hard_stop[0]})


print('===== FINAL REPORT =====')
OUTPUT_DIR = Path.home() / 'aisi_checkpoints' / 'stage9d-per-token-gradient-diagnostic'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Strip live tensor references before serializing.
serializable_analyses = []
for a in step_analyses:
    a2 = dict(a)
    a2['per_row_details'] = [{k: v for k, v in d.items()} for d in a['per_row_details']]
    serializable_analyses.append(a2)
report = {'stage9c_adapter_sha256': _stage9c_bytes_sha, 'terminal_step': terminal, 'hard_stop': hard_stop[0],
          'telemetry': telemetry, 'step_analyses': serializable_analyses}
EVENT_LOG = OUTPUT_DIR / 'per_token_gradient_diagnostic_result.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({'terminal_step': terminal, 'hard_stop': hard_stop[0],
                   'step_summaries': [{'step': a['step'], 'token_share_code': a['token_share_code'],
                                        'grad_share_code': a['grad_share_code'],
                                        'disproportionality_ratio': a['disproportionality_ratio']} for a in step_analyses]},
                  indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('PER-TOKEN GRADIENT DIAGNOSTIC COMPLETE.')
