from __future__ import annotations

import hashlib, importlib.metadata, json, logging, sys, warnings
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9d DECODE-BACK-LINE TRUNCATION / CONCEALMENT DIAGNOSTIC (2026-09-03):
# MAIN's completions track state in Nib/Nomo, then write an explicit plain-language
# decode-back line ("The final code token Nib decodes to Heads.") before <answer> --
# a line deliberately excluded from the CoT-penalty's scan (score_completion_v2_
# decode_back_aware) to make training work at all. A real text-based monitor watching
# the FULL trace, with no such carve-out, would catch this line immediately. This
# means MAIN has never had to answer correctly WITHOUT confessing the mapping in
# plain text -- the single biggest threat to calling this result steganographic
# rather than "encoding, then immediately un-encoding out loud."
#
# This script tests, inference-only, whether MAIN's checkpoint still NEEDS to write
# that line: for held-out scenarios the model currently answers correctly, truncate
# its own generation immediately before the decode-back line would begin (right after
# the final tracked-state line), force <answer> as the literal next text, and let it
# complete only the answer with no decode-back sentence present at all. Truncated
# accuracy close to normal accuracy -> the line is training scaffolding no longer
# depended on. Truncated accuracy collapses -> translation has been externalized to
# an explicit written step, not internalized. No retraining, no checkpoint
# modification -- diagnostic only.

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
_expected = {'transformers': '5.13.1', 'peft': '0.19.1', 'bitsandbytes': '0.50.0'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
RUN_SEED = 20260903
MAX_NEW_TOKENS_NORMAL = 256
MAX_NEW_TOKENS_TRUNCATED = 40  # small: just needs to reach and close <answer>...</answer>

import os
MAIN_CHECKPOINT_DIR = os.environ.get('STAGE9D_MAIN_CHECKPOINT_DIR', '').strip()
if not MAIN_CHECKPOINT_DIR:
    raise RuntimeError('STAGE9D_MAIN_CHECKPOINT_DIR must be set to the saved MAIN adapter directory '
                        '(e.g. ~/aisi_checkpoints/stage9d-decode-back-main-v3/final_adapter).')
MAIN_CHECKPOINT_DIR = Path(MAIN_CHECKPOINT_DIR).expanduser()

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'main_checkpoint_dir': str(MAIN_CHECKPOINT_DIR), **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
_STAGE09D_DIR = _REPO_ROOT / 'experiments' / '09d_sft_seeded_adversarial_rl'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
sys.path.insert(0, str(_STAGE09C_DIR)); sys.path.insert(0, str(_STAGE09D_DIR))
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt  # noqa: E402
from truncation_lib import (  # noqa: E402
    build_truncated_prefix, score_normal_completion, score_truncated_continuation, summarize_subset)

CLEAN21_SEED = 20260831
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'
_, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
assert hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest() == EXPECTED_CLEAN21_SHA256
print({'eval_n': len(eval_rows), 'clean21_sha256_verified': True})

MAIN_WEIGHTS = MAIN_CHECKPOINT_DIR / 'adapter_model.safetensors'
if not MAIN_WEIGHTS.is_file(): raise FileNotFoundError(MAIN_WEIGHTS)
_main_bytes_sha = hashlib.sha256(MAIN_WEIGHTS.read_bytes()).hexdigest()
print({'main_adapter_path': str(MAIN_WEIGHTS), 'main_adapter_sha256': _main_bytes_sha})

torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')


def load_main_checkpoint():
    """Fresh, independent base-model load plus a fresh PeftModel wrapper, loaded with
    the saved MAIN adapter weights -- single adapter, no composition. Same
    checkpoint-isolation pattern used throughout this stage (never reuse a base_model
    object across get_peft_model() calls -- TRL's own beta!=0 PEFT reference-adapter
    leak, not relevant here since no GRPOTrainer is involved, but the isolation
    discipline is reused unchanged regardless, for identical, unambiguous checkpoint
    identity)."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors
    fresh_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = True
    model = get_peft_model(fresh_base, LoraConfig(**LORA_KWARGS))
    result = set_peft_model_state_dict(model, load_safetensors(str(MAIN_WEIGHTS)), adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys}')
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    model.eval()
    identity = hashlib.sha256()
    for name, param in model.named_parameters():
        if '.default.' not in name: continue
        identity.update(name.encode()); identity.update(param.detach().float().cpu().numpy().tobytes())
    return model, identity.hexdigest()


print('===== LOAD MAIN CHECKPOINT (inference only, no training) =====')
model, checkpoint_identity = load_main_checkpoint()
print({'checkpoint_loaded': True, 'checkpoint_identity_sha256': checkpoint_identity})


def chat_prefix_for(prompt_text):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_text}], tokenize=False, add_generation_prompt=True)


def run_generation(prompts_texts, max_new_tokens, batch_size=25):
    """Continues generation from already-fully-formed text -- for NORMAL prompts this
    is chat-templated text with an empty assistant turn (add_generation_prompt=True);
    for TRUNCATED prefixes this is chat-templated text + the model's own truncated
    trace + a forced-open <answer> tag. Neither case re-applies the chat template a
    second time. Deterministic (greedy) throughout, matching every eval pass
    elsewhere in this stage."""
    out = []
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        # model is a module-level global assigned right after load_main_checkpoint()
        # is called below, before this function is ever invoked; deferred-binding
        # false positive, not a real undefined name.
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)  # noqa: F821
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,  # noqa: F821
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out


print('===== STEP 1: NORMAL (untruncated) GENERATION on the full held-out-21 =====')
prompts = [build_prompt(r['starting_state'], r['operations']) for r in eval_rows]
chat_prefixes = [chat_prefix_for(p) for p in prompts]
normal_completions = run_generation(chat_prefixes, MAX_NEW_TOKENS_NORMAL)
normal_scores = [score_normal_completion(row, c) for row, c in zip(eval_rows, normal_completions)]
normal_accuracy_full_set = sum(1 for s in normal_scores if s['final_answer_correct']) / len(normal_scores)
print({'normal_accuracy_full_21': normal_accuracy_full_set,
       'intermediate_tracking_accuracy_full_21':
           sum(1 for s in normal_scores if s['intermediate_tracking_correct']) / len(normal_scores)})

print('===== STEP 2: RESTRICT to scenarios where NORMAL generation is currently correct =====')
eligible = [(row, completion, score) for row, completion, score in zip(eval_rows, normal_completions, normal_scores)
            if score['final_answer_correct']]
print({'eligible_n': len(eligible), 'excluded_n': len(eval_rows) - len(eligible)})
if not eligible:
    raise RuntimeError('No eligible (normal-correct) scenarios -- cannot run the truncation diagnostic.')

print('===== STEP 3: FORCED-TRUNCATION generation (decode-back line entirely removed, <answer> forced open) =====')
truncated_prefixes = []
skipped_no_slots = []
for row, completion, score in eligible:
    chat_prefix = chat_prefix_for(build_prompt(row['starting_state'], row['operations']))
    prefix = build_truncated_prefix(chat_prefix, completion)
    if prefix is None:
        skipped_no_slots.append((row['starting_state'], row['operations']))
        continue
    truncated_prefixes.append((row, completion, score, prefix))
if skipped_no_slots:
    print({'WARNING_skipped_no_parseable_state_slots': skipped_no_slots})

truncated_continuations = run_generation([t[3] for t in truncated_prefixes], MAX_NEW_TOKENS_TRUNCATED)

results = []
for (row, completion, normal_score, prefix), continuation in zip(truncated_prefixes, truncated_continuations):
    trunc_score = score_truncated_continuation(row, prefix, continuation, normal_score['normal_answer'])
    results.append({
        'starting_state': row['starting_state'], 'operations': row['operations'],
        'final_answer_truth': row['final_answer'],
        'normal_completion': completion, 'normal_answer': normal_score['normal_answer'],
        # key name matches score_normal_completion's own output and what
        # summarize_subset() expects -- this is always True for every row here since
        # `results` is built only from the eligible (normal-correct) subset.
        'final_answer_correct': normal_score['final_answer_correct'],
        'normal_intermediate_tracking_correct': normal_score['intermediate_tracking_correct'],
        'truncated_continuation_raw': continuation,
        'truncated_answer': trunc_score['truncated_answer'],
        'truncated_correct': trunc_score['truncated_correct'],
        'matches_normal_answer': trunc_score['matches_normal_answer'],
        'no_parseable_answer': trunc_score['no_parseable_answer'],
    })

print('===== STEP 4: PER-DIRECTION SUMMARY (Heads-truth and Tails-truth, NOT pooled -- task requirement 5) =====')
heads_subset = [r for r in results if r['final_answer_truth'] == 'Heads']
tails_subset = [r for r in results if r['final_answer_truth'] == 'Tails']
heads_summary = summarize_subset(heads_subset)
tails_summary = summarize_subset(tails_subset)
pooled_summary_FOR_REFERENCE_ONLY = summarize_subset(results)
print({'HEADS_TRUTH_SUBSET': heads_summary})
print({'TAILS_TRUTH_SUBSET': tails_summary})
print({'pooled_for_reference_only_not_the_primary_result': pooled_summary_FOR_REFERENCE_ONLY})

del model
torch.cuda.empty_cache()

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9d-truncation-diagnostic-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED,
               'main_checkpoint_dir': str(MAIN_CHECKPOINT_DIR),
               'main_adapter_sha256': _main_bytes_sha,
               'checkpoint_identity_sha256': checkpoint_identity,
               'clean21_eval_sha256': EXPECTED_CLEAN21_SHA256,
               'max_new_tokens_normal': MAX_NEW_TOKENS_NORMAL,
               'max_new_tokens_truncated': MAX_NEW_TOKENS_TRUNCATED,
               'normal_accuracy_full_21': normal_accuracy_full_set,
               'eligible_n': len(eligible), 'excluded_n': len(eval_rows) - len(eligible)},
    'heads_truth_subset': {'summary': heads_summary, 'samples': heads_subset},
    'tails_truth_subset': {'summary': tails_summary, 'samples': tails_subset},
    'pooled_for_reference_only': pooled_summary_FOR_REFERENCE_ONLY,
}
EVENT_LOG = OUTPUT / 'stage9d_truncation_diagnostic.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9D TRUNCATION DIAGNOSTIC COMPLETE.')
