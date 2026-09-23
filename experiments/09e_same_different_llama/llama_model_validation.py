"""Stage 9e, Llama-3-8B-Instruct pass, Part A -- model validation (2026-09-07).

Runs on the GPU instance. Inference only, no training. Combines:
  1. Load Llama-3-8B-Instruct (8-bit quantized, matching this project's established
     Qwen loading pattern) -- report actual GPU memory footprint after load, since
     Step 4's own instruction is to confirm sizing empirically before any training
     launch, not estimate it.
  2. Token-pool audit: reuses token_pool_audit.py's build_table() UNCHANGED (already
     fully model-agnostic -- takes tokenizer/model/device as parameters, per the prior
     research into this module) against Llama's real tokenizer and base model. Full
     CANDIDATES list, not a shortcut straight to Nib/Nomo -- the redesign spec
     explicitly requires independently re-verifying eligibility, not assuming reuse.
  3. Chat template inspection: the actual `tokenizer.chat_template` string and a real
     `apply_chat_template()` call, not an assumption ported from Qwen.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge' / 'data_generation'))
from token_pool_audit import build_table, CANDIDATES, BANNED_WORDS  # noqa: E402

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'

print('===== STEP: LOAD TOKENIZER =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print({'tokenizer_loaded': True, 'vocab_size': tokenizer.vocab_size,
       'pad_token': tokenizer.pad_token, 'eos_token': tokenizer.eos_token,
       'bos_token': tokenizer.bos_token})

print('===== STEP: MEMORY BASELINE (before model load) =====')
torch.cuda.reset_peak_memory_stats()
mem_before_mib = torch.cuda.memory_allocated() / (1024 ** 2)
print({'gpu_name': torch.cuda.get_device_name(0),
       'total_gpu_memory_mib': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
       'allocated_before_load_mib': mem_before_mib})

print('===== STEP: LOAD MODEL (8-bit, matching established Qwen pattern) =====')
quant = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
model.eval()
device = next(model.parameters()).device

torch.cuda.synchronize()
mem_after_load_mib = torch.cuda.memory_allocated() / (1024 ** 2)
mem_reserved_mib = torch.cuda.memory_reserved() / (1024 ** 2)
print({'model_loaded': True, 'device': str(device),
       'allocated_after_load_mib': mem_after_load_mib,
       'reserved_after_load_mib': mem_reserved_mib,
       'model_footprint_mib': mem_after_load_mib - mem_before_mib})

print('===== STEP: TOKEN-POOL AUDIT (full CANDIDATES list, real base-policy logprobs) =====')
rows = build_table(tokenizer, model, device)
eligible = [r for r in rows if r['eligible']]
print(f"\n{'token':10s} {'n_tok':6s} {'ws_sens':8s} {'leak_ok':8s} {'min_ed':7s} {'logprob':10s} {'eligible':9s}")
for r in rows:
    lp = r['base_policy_logprob_at_state_slot']
    lp_str = f'{lp:.2f}' if lp is not None else 'n/a'
    print(f"{r['token']:10s} {r['n_tokens_with_leading_space']:<6d} "
          f"{str(r['whitespace_changes_tokenization']):8s} {str(r['passes_leakage_classifier']):8s} "
          f"{r['min_edit_distance_to_banned_word']:<7d} {lp_str:10s} {str(r['eligible']):9s}")
print(f'\n{len(eligible)}/{len(rows)} candidates eligible on Llama-3-8B-Instruct tokenizer.')

print('===== STEP: DOES QWEN\'S NIB/NOMO/YELT/YARK STILL PASS ON LLAMA\'S TOKENIZER? =====')
qwen_pairs = {'Nib': None, 'Nomo': None, 'Yelt': None, 'Yark': None}
by_token = {r['token']: r for r in rows}
for name in qwen_pairs:
    row = by_token.get(name)
    if row is None:
        print({name: 'not in CANDIDATES list -- should not happen, CANDIDATES is shared verbatim'})
    else:
        qwen_pairs[name] = row
        print({name: {'eligible': row['eligible'], 'n_tokens_with_leading_space': row['n_tokens_with_leading_space'],
                       'passes_leakage_classifier': row['passes_leakage_classifier'],
                       'base_policy_logprob_at_state_slot': row['base_policy_logprob_at_state_slot']}})

print('===== STEP: CHAT TEMPLATE INSPECTION =====')
print({'has_chat_template': tokenizer.chat_template is not None})
if tokenizer.chat_template:
    print('--- raw chat_template string ---')
    print(tokenizer.chat_template)
print('--- apply_chat_template() on a representative single-turn example ---')
example_prompt = 'Step 1: The state flips. State: <CODE>\nWhat is the final answer?'
rendered = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': example_prompt}], tokenize=False, add_generation_prompt=True)
print(repr(rendered))
print('--- tokenized form (to see actual special-token IDs used) ---')
# Tokenize the already-rendered string directly (same pattern used elsewhere in this
# project, e.g. truncation_diagnostic.py's chat_prefix_for()) rather than
# apply_chat_template(tokenize=True), which raised TypeError: argument 'ids': 'str'
# object cannot be interpreted as an integer in this environment on the first attempt
# -- a bug in this script's own use of that path, not evidence of anything about
# Llama's tokenizer itself (the tokenize=False render above worked correctly).
rendered_ids = tokenizer(rendered, add_special_tokens=False).input_ids
print({'n_tokens': len(rendered_ids), 'token_ids': rendered_ids,
       'decoded_pieces': [tokenizer.decode([i]) for i in rendered_ids]})

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9e-llama-model-validation-v{version_id}'
    if not OUTPUT.exists():
        break
else:
    raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'candidates': CANDIDATES, 'banned_words': BANNED_WORDS},
    'tokenizer_info': {'vocab_size': tokenizer.vocab_size, 'pad_token': tokenizer.pad_token,
                        'eos_token': tokenizer.eos_token, 'bos_token': tokenizer.bos_token},
    'memory': {'gpu_name': torch.cuda.get_device_name(0),
               'total_gpu_memory_mib': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
               'allocated_after_load_mib': mem_after_load_mib,
               'reserved_after_load_mib': mem_reserved_mib,
               'model_footprint_mib': mem_after_load_mib - mem_before_mib},
    'audit_rows': rows,
    'n_eligible': len(eligible),
    'qwen_pairs_on_llama_tokenizer': {k: v for k, v in qwen_pairs.items()},
    'chat_template_raw': tokenizer.chat_template,
    'chat_template_rendered_example': rendered,
    'chat_template_rendered_token_ids': rendered_ids,
}
EVENT_LOG = OUTPUT / 'stage9e_llama_model_validation.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA MODEL VALIDATION COMPLETE.')
