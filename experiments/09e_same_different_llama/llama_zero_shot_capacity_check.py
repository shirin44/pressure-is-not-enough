from __future__ import annotations

import json
import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU visible to this process.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from synthetic_bridge import _SCENARIOS  # noqa: E402
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from same_different_logic import same_different_answer  # noqa: E402
from same_different_prompt import build_same_different_prompt  # noqa: E402

# Stage 9e, Decision 2 (2026-09-07): zero-shot capacity check. Same/Different requires
# tracking a relation to the ORIGIN state throughout the sequence -- a different
# cognitive load than simple forward state-tracking. Confirm the base model can do
# this raw computation at all, using PLAIN Heads/Tails (no code substitution), before
# spending an SFT pass that would conflate "learn the task" with "learn the code
# substitution." If accuracy is near chance, this blocks proceeding -- the task
# itself would need redesigning, independent of anything about code substitution.
#
# Uses the existing, already-verified 16-scenario _SCENARIOS bank (exactly 8
# Same / 8 Different by construction under Decision 4's origin-anchoring, confirmed
# separately) as the zero-shot batch -- a reasonable "small batch," already balanced,
# already diverse (all-same/all-different/alternating/clustered/mixed patterns, both
# starting states). No training, no LoRA -- the untouched Instruct model, zero-shot.

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
MAX_NEW_TOKENS = 256

print('===== LOAD TOKENIZER =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print('===== LOAD MODEL (8-bit, untouched, no LoRA, zero-shot) =====')
quant = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
model.eval()
device = next(model.parameters()).device
print({'model_loaded': True, 'device': str(device)})


def chat_wrap(prompts):
    return [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]


def generate_batch(prompts_texts, max_new_tokens, batch_size=16):
    out = []
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        out.extend(tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True))
    return out


print('===== BUILD ZERO-SHOT BATCH (16 scenarios, plain Heads/Tails, no code substitution) =====')
prompts = [build_same_different_prompt(s, ops) for s, ops in _SCENARIOS]
truths = [same_different_answer(s, ops) for s, ops in _SCENARIOS]
print({'n_scenarios': len(prompts), 'n_same': truths.count('Same'), 'n_different': truths.count('Different')})

print('===== GENERATE (zero-shot, greedy, no LoRA) =====')
completions = generate_batch(chat_wrap(prompts), MAX_NEW_TOKENS)

print('===== SCORE =====')
results = []
for (starting_state, operations), truth, completion in zip(_SCENARIOS, truths, completions):
    ans, valid = _extract_answer(completion)
    correct = bool(valid and ans == truth.casefold())
    slots = parse_state_slots(completion)
    intermediate_tokens = [t for _i, t in slots]
    results.append({
        'starting_state': starting_state, 'operations': list(operations), 'truth': truth,
        'completion': completion, 'extracted_answer': ans, 'format_valid': valid, 'correct': correct,
        'intermediate_tokens': intermediate_tokens,
    })
    print(f"  truth={truth:9s} extracted={str(ans):9s} valid={valid} correct={correct}")

n_correct = sum(1 for r in results if r['correct'])
n_valid = sum(1 for r in results if r['format_valid'])
accuracy = n_correct / len(results)
same_results = [r for r in results if r['truth'] == 'Same']
diff_results = [r for r in results if r['truth'] == 'Different']
same_acc = sum(1 for r in same_results if r['correct']) / len(same_results) if same_results else None
diff_acc = sum(1 for r in diff_results if r['correct']) / len(diff_results) if diff_results else None

analysis = {
    'n': len(results), 'n_correct': n_correct, 'accuracy': accuracy,
    'n_format_valid': n_valid, 'format_valid_rate': n_valid / len(results),
    'same_truth_accuracy': same_acc, 'different_truth_accuracy': diff_acc,
    'near_chance_flag': accuracy < 0.65,  # meaningfully above the 0.5 chance floor is the bar
}
print('===== ANALYSIS =====')
print(analysis)

model = None
torch.cuda.empty_cache()

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9e-llama-zero-shot-capacity-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'max_new_tokens': MAX_NEW_TOKENS, 'n_scenarios': len(prompts)},
    'results': results,
    'analysis': analysis,
}
EVENT_LOG = OUTPUT / 'stage9e_llama_zero_shot_capacity.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA ZERO-SHOT CAPACITY CHECK COMPLETE.')
