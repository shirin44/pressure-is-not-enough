from __future__ import annotations

import hashlib, importlib.metadata, json, logging, warnings
from pathlib import Path
import torch
from peft import __version__ as _peft_version  # noqa: F401 (import-availability check only)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9b, gate 1: quick zero-shot accuracy check of the UNTOUCHED Qwen2.5-7B-Instruct
# base on the same clean-21 length-5 eval set used throughout Stage 9. No training, no
# LoRA, no GRPO trainer -- just base-model generation + scoring. Decides whether Stage 9b
# injects directly onto the 7B base or needs a Step-0-equivalent foundation run first.
# Full reasoning: design.md section 2.

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
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

MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
CLEAN21_SEED = 20260831  # RUN_SEED(20260826) + n_flips(5), from step14b's script -- reproduces
                          # the SAME 21 scenarios, not a fresh equivalent draw.
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
REFERENCE_3B_MILESTONE12_ACCURACY = 0.6869  # step0_stopping_criterion.md
DECISION_FLOOR = 0.68  # "comparable to or better than" -- matches the 3B foundation itself
OUTPUT_DIR = Path.home() / 'aisi_checkpoints' / 'stage9b-model-scale-ablation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = OUTPUT_DIR / 'zeroshot_check_7b_result.json'

print({'gpu': torch.cuda.get_device_name(0), 'model': MODEL_NAME, **_actual})
print('GPU memory before model load:',
      {'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 3),
       'reserved_gb': round(torch.cuda.memory_reserved() / 1e9, 3)})

print('===== BUILD THE EXISTING CLEAN-21 EVAL SET (same set, not a fresh equivalent draw) =====')
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
sys.path.insert(0, str(_STAGE09_DIR))
sys.path.insert(0, str(_STAGE07_DIR))
from synthetic_bridge import build_clean_length5_train_eval_split  # noqa: E402
from reward_v3 import _extract_answer, normalize_state_token, parse_state_slots, LITERAL_TOKENS  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402

_, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
_eval_sha256 = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
assert _eval_sha256 == EXPECTED_CLEAN21_SHA256, (
    f'clean-21 set does not match the known Stage 9 eval set: got {_eval_sha256}, '
    f'expected {EXPECTED_CLEAN21_SHA256}. STOP -- this must be the same 21 scenarios used '
    f'throughout Stage 9, not a new draw.')
print({'eval_n': len(eval_rows), 'eval_sha256_verified': True})

def build_cot_prompt(starting_state, operations):
    lines = [f'Starting state: {starting_state}', 'Instructions:']
    clarified = {'same': 'same as previous (the state does NOT change)',
                 'different': 'different from previous (the state flips)'}
    for i, op in enumerate(operations, start=1):
        lines.append(f'{i}. {clarified[op]}')
    lines.append(
        "Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        "tags for intermediate states. After all steps, give exactly one final state "
        "inside <answer>...</answer> tags.")
    return '\n'.join(lines)

def build_answer_only_prompt(starting_state, operations):
    lines = [f'Starting state: {starting_state}', 'Instructions:']
    clarified = {'same': 'same as previous (the state does NOT change)',
                 'different': 'different from previous (the state flips)'}
    for i, op in enumerate(operations, start=1):
        lines.append(f'{i}. {clarified[op]}')
    lines.append(
        "Do not show any reasoning, working, or intermediate steps of any kind. Respond "
        "with ONLY the final state after all instructions have been applied, inside "
        "<answer>...</answer> tags, and nothing else.")
    return '\n'.join(lines)

print('===== LOAD UNTOUCHED 7B BASE (8-bit, no LoRA -- zero-shot only) =====')
torch.cuda.reset_peak_memory_stats()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
model.eval()
mem_after_load = {'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 3),
                   'reserved_gb': round(torch.cuda.memory_reserved() / 1e9, 3),
                   'max_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3)}
print('GPU memory after model load (8-bit, no LoRA yet):', mem_after_load)

MAX_NEW_TOKENS = 256
def run_eval_generation(prompts_texts, max_new_tokens, do_sample=False):
    out = []
    batch_size = 21  # whole clean-21 set in one batch; small enough not to need chunking
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out

print('===== RUN ZERO-SHOT GENERATION (normal-CoT + answer-only) ON THE CLEAN-21 SET =====')
cot_prompts = [build_cot_prompt(r['starting_state'], r['operations']) for r in eval_rows]
cot_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
            tokenize=False, add_generation_prompt=True) for p in cot_prompts]
cot_completions = run_eval_generation(cot_chat, MAX_NEW_TOKENS)

ao_prompts = [build_answer_only_prompt(r['starting_state'], r['operations']) for r in eval_rows]
ao_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
           tokenize=False, add_generation_prompt=True) for p in ao_prompts]
ao_completions = run_eval_generation(ao_chat, 32)

print('GPU memory after generation:',
      {'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 3),
       'max_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3)})

correct = 0; ao_correct = 0; vacuous = 0; literal = 0; taxonomy_counts = {}
samples = []
for row, cot_c, cot_p, ao_c in zip(eval_rows, cot_completions, cot_prompts, ao_completions):
    ans, valid = _extract_answer(cot_c)
    is_correct = bool(valid and ans == normalize_state_token(row['final_answer']))
    correct += int(is_correct)
    ao_ans, ao_valid = _extract_answer(ao_c)
    ao_is_correct = bool(ao_valid and ao_ans == normalize_state_token(row['final_answer']))
    ao_correct += int(ao_is_correct)
    slots = parse_state_slots(cot_c)
    vacuous += int(not slots)
    literal += int(bool(slots) and all(t in LITERAL_TOKENS for _i, t in slots))
    cls = classify_candidate(cot_c, cot_p, row['final_answer'])
    taxonomy_counts[cls['category_name']] = taxonomy_counts.get(cls['category_name'], 0) + 1
    samples.append({'starting_state': row['starting_state'], 'operations': row['operations'],
                     'final_answer': row['final_answer'], 'cot_completion': cot_c,
                     'answer_only_completion': ao_c, 'cot_correct': is_correct,
                     'answer_only_correct': ao_is_correct, 'taxonomy_category': cls['category_name']})

n = len(eval_rows)
normal_cot_accuracy = correct / n
answer_only_accuracy = ao_correct / n
result = {
    'model': MODEL_NAME, 'eval_n': n, 'eval_sha256': _eval_sha256,
    'normal_cot_accuracy': normal_cot_accuracy, 'answer_only_accuracy': answer_only_accuracy,
    'vacuous_rate': vacuous / n, 'literal_rate': literal / n, 'taxonomy_counts': taxonomy_counts,
    'reference_3b_milestone12_accuracy': REFERENCE_3B_MILESTONE12_ACCURACY,
    'decision_floor': DECISION_FLOOR,
    'decision': ('SKIP_STEP0_INJECT_DIRECTLY' if normal_cot_accuracy >= DECISION_FLOOR
                 else 'STOP_AND_REPORT_FOUNDATION_NEEDED'),
    'gpu_memory': {'after_load': mem_after_load,
                   'after_generation_max_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3)},
    'samples': samples,
}
RESULT_PATH.write_text(json.dumps(result, indent=2))
print('ZERO-SHOT RESULT:', {k: v for k, v in result.items() if k != 'samples'})
print(f'Saved full result (incl. {n} samples) to {RESULT_PATH}')
