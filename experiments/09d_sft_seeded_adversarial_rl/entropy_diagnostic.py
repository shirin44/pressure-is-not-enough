from __future__ import annotations

import hashlib, importlib.metadata, json, logging, sys, warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9d follow-up: quantitative diagnosis of WHY resuming GRPO from Stage 9c's
# SFT checkpoint produced grad_norm breaker trips at step 3 regardless of reward
# config. Measures each checkpoint's own output-distribution entropy and chosen-
# token logprob on its OWN greedily-generated completions (apples-to-apples: each
# policy's confidence in the tokens IT would actually choose, not one model's
# confidence scored against another's text) -- confirms or refutes the "SFT
# converged to an unusually peaked policy" hypothesis with real numbers before
# picking any LR/warmup revision. design.md documents the full result and the
# schedule decision it justifies.

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

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
MAX_NEW_TOKENS = 256
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
sys.path.insert(0, str(_STAGE09_DIR))
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt  # noqa: E402

CHECKPOINTS = {
    'step0_milestone12': Path.home() / 'aisi_checkpoints' / 'exp3-step0-task-foundation-full-v1' / 'milestone-12' / 'adapter_model.safetensors',
    'stage9c_sft': Path.home() / 'aisi_checkpoints' / 'stage9c-sft-diagnostic-v1' / 'final_adapter' / 'adapter_model.safetensors',
}
for label, path in CHECKPOINTS.items():
    if not path.is_file(): raise FileNotFoundError(f'{label}: {path}')

CLEAN21_SEED = 20260831
_, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
SAMPLE_PROMPTS = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows[:8]]  # first 8, deterministic

print({'gpu': torch.cuda.get_device_name(0), 'n_sample_prompts': len(SAMPLE_PROMPTS),
       'checkpoints': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in CHECKPOINTS.items()}})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')


def load_checkpoint_fresh(weights_path):
    """Independent base-model load per checkpoint -- same isolation fix applied in
    sft_seeded_rl.py, for the same reason (no shared state across loads)."""
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    base.config.use_cache = True
    model = get_peft_model(base, LoraConfig(**LORA_KWARGS))
    result = set_peft_model_state_dict(model, load_safetensors(str(weights_path)), adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise RuntimeError(f'Unexpected adapter keys loading {weights_path}: {result.unexpected_keys}')
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    model.eval()
    return model


def measure_entropy_and_logprob(model, prompts):
    """For each prompt: generate greedily (deterministic, reproducible), then
    teacher-force a fresh forward pass over (prompt+completion) to extract the
    REAL per-position output distribution (not just the sampled logprob) --
    entropy = -sum(p*log(p)) over the full vocab at each completion position, and
    the logprob the model assigned to the token it actually chose there. Measures
    each checkpoint's confidence in ITS OWN natural output, not one model scored
    against another's text."""
    per_prompt_results = []
    for prompt in prompts:
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors='pt').to(next(model.parameters()).device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        completion_ids = generated[0, inputs['input_ids'].shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        full_ids = generated  # prompt + completion, exactly what was generated
        with torch.inference_mode():
            logits = model(input_ids=full_ids).logits  # (1, seq_len, vocab)
        prompt_len = inputs['input_ids'].shape[1]
        completion_len = completion_ids.shape[0]
        if completion_len == 0:
            continue
        # logits[i] predicts token i+1; completion tokens occupy positions
        # [prompt_len, prompt_len+completion_len) in full_ids, so the logits that
        # PREDICT them are at [prompt_len-1, prompt_len+completion_len-1).
        pred_logits = logits[0, prompt_len - 1: prompt_len + completion_len - 1, :].float()
        probs = F.softmax(pred_logits, dim=-1)
        log_probs = F.log_softmax(pred_logits, dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # (completion_len,)
        chosen_logprob_per_token = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
        per_prompt_results.append({
            'completion_text': completion_text,
            'completion_len': int(completion_len),
            'mean_entropy': float(entropy_per_token.mean().item()),
            'max_entropy': float(entropy_per_token.max().item()),
            'min_entropy': float(entropy_per_token.min().item()),
            'mean_chosen_logprob': float(chosen_logprob_per_token.mean().item()),
            'median_chosen_logprob': float(chosen_logprob_per_token.median().item()),
            'frac_tokens_logprob_above_neg0p01': float((chosen_logprob_per_token > -0.01).float().mean().item()),
        })
    return per_prompt_results


results = {}
for label, weights_path in CHECKPOINTS.items():
    print(f'===== {label} =====')
    model = load_checkpoint_fresh(weights_path)
    per_prompt = measure_entropy_and_logprob(model, SAMPLE_PROMPTS)
    all_mean_entropies = [r['mean_entropy'] for r in per_prompt]
    all_mean_logprobs = [r['mean_chosen_logprob'] for r in per_prompt]
    summary = {
        'n_prompts': len(per_prompt),
        'grand_mean_entropy': sum(all_mean_entropies) / len(all_mean_entropies),
        'grand_mean_chosen_logprob': sum(all_mean_logprobs) / len(all_mean_logprobs),
        'min_mean_entropy_across_prompts': min(all_mean_entropies),
        'max_mean_entropy_across_prompts': max(all_mean_entropies),
        'mean_frac_high_confidence_tokens': sum(r['frac_tokens_logprob_above_neg0p01'] for r in per_prompt) / len(per_prompt),
    }
    print(json.dumps(summary, indent=2))
    results[label] = {'summary': summary, 'per_prompt': per_prompt}
    del model
    torch.cuda.empty_cache()

print('===== COMPARISON =====')
s0 = results['step0_milestone12']['summary']
s9c = results['stage9c_sft']['summary']
entropy_ratio = s0['grand_mean_entropy'] / s9c['grand_mean_entropy'] if s9c['grand_mean_entropy'] > 0 else float('inf')
comparison = {
    'step0_grand_mean_entropy': s0['grand_mean_entropy'],
    'stage9c_grand_mean_entropy': s9c['grand_mean_entropy'],
    'entropy_ratio_step0_over_stage9c': entropy_ratio,
    'step0_grand_mean_chosen_logprob': s0['grand_mean_chosen_logprob'],
    'stage9c_grand_mean_chosen_logprob': s9c['grand_mean_chosen_logprob'],
    'step0_mean_frac_high_confidence_tokens': s0['mean_frac_high_confidence_tokens'],
    'stage9c_mean_frac_high_confidence_tokens': s9c['mean_frac_high_confidence_tokens'],
}
print(json.dumps(comparison, indent=2))

OUTPUT_DIR = Path.home() / 'aisi_checkpoints' / 'stage9d-entropy-diagnostic'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'entropy_diagnostic_result.json').write_text(json.dumps({'results': results, 'comparison': comparison}, indent=2, default=str))
print('\nEvidence:', OUTPUT_DIR / 'entropy_diagnostic_result.json')
print('ENTROPY DIAGNOSTIC COMPLETE.')
