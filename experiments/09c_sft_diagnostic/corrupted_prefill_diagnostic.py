from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, sys, warnings
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9c CORRUPTED-PREFILL DIAGNOSTIC (2026-08-31): the rebalanced-bank SFT checkpoint
# (stage9c-sft-rebalanced-v1) achieves 95.2% ground-truth-verified intermediate state
# tracking but a complete (100%) collapse to answering "Tails" on held-out completions,
# mathematically identical to an unconditional always-Tails policy (33.3% = 7/21 true
# Tails). This script directly tests whether the final <answer> is causally conditioned
# on the model's own tracked trace via a corrupted-prefill intervention: take the
# checkpoint's own correctly-tracked held-out completions (already generated and
# ground-truth-verified in stage9c_sft_rebalanced.json -- reused here, not regenerated,
# since do_sample=False makes them deterministic and already-verified), flip ONLY the
# final tracked-state token, truncate right after it (dropping any existing <answer>
# tag), and continue generation from the corrupted prefix. Two groups are tested:
# heads_to_tails (the task's literal ask -- NOT decisive alone, since the model's known
# default is already "Tails") and tails_to_heads (the decisive reverse direction: if the
# prefix is causally read the answer should flip to "Heads"; if structurally
# disconnected it stays "Tails" regardless). See corruption_lib.py's module docstring
# for the full reasoning.
#
# Independently, this script also probes the FRESH BASE (pre-SFT) model's raw
# next-token logits/probabilities for "Heads" vs "Tails" at the exact <answer> position,
# on ground-truth-correct Nib/Nomo-coded reasoning prefixes (both a Nib-ending and a
# Nomo-ending version of several scenarios), to check for a base-model prior toward one
# answer word independent of any reasoning content or fine-tuning.
#
# Diagnostic only, per instruction -- no fix is attempted here.

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
RUN_SEED = 20260901
MAX_NEW_TOKENS_CONTINUATION = 40  # small: just needs to reach and close <answer>...</answer>
EXPECTED_EVIDENCE_SHA256 = '3d5ab4404464b04280aa18a7bcb1e6e91a0edc9d548b97edb30d0c2fa8eb603c'

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, **_actual})

print('===== IMPORTS (model-independent modules reused unchanged) =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE09C_DIR))
from synthetic_bridge import build_prompt, build_coded_completion, _trace  # noqa: E402
from corruption_lib import (  # noqa: E402
    CODE_LITERAL_MAP, CODE_TO_LITERAL_ANSWER, build_corrupted_prefix, classify_outcome,
    extract_answer, select_eligible_samples)

EVIDENCE_PATH = _STAGE09C_DIR / 'aws_runs' / 'stage9c-sft-rebalanced-v1' / 'stage9c_sft_rebalanced.json'
_evidence_bytes = EVIDENCE_PATH.read_bytes()
_evidence_sha = hashlib.sha256(_evidence_bytes).hexdigest()
if _evidence_sha != EXPECTED_EVIDENCE_SHA256:
    raise RuntimeError(f'Evidence file hash mismatch: expected {EXPECTED_EVIDENCE_SHA256}, got {_evidence_sha}')
EVIDENCE = json.loads(_evidence_bytes)
TIER_B = EVIDENCE['tier_b_heldout_same_pair']['samples']
print({'evidence_sha256_verified': True, 'tier_b_n': len(TIER_B)})

ELIGIBLE = select_eligible_samples(TIER_B)
print({'eligible_heads_to_tails': len(ELIGIBLE['heads_to_tails']), 'eligible_tails_to_heads': len(ELIGIBLE['tails_to_heads'])})

ADAPTER_DIR = Path.home() / 'aisi_checkpoints' / 'stage9c-sft-rebalanced-v1' / 'final_adapter'
ADAPTER_WEIGHTS = ADAPTER_DIR / 'adapter_model.safetensors'
if not ADAPTER_WEIGHTS.is_file(): raise FileNotFoundError(ADAPTER_WEIGHTS)
_adapter_sha = hashlib.sha256(ADAPTER_WEIGHTS.read_bytes()).hexdigest()
print({'adapter_path': str(ADAPTER_WEIGHTS), 'adapter_sha256': _adapter_sha})

torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')


def chat_prefix_for(prompt_text):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_text}], tokenize=False, add_generation_prompt=True)


def run_generation(prompts_texts, max_new_tokens, do_sample=False, batch_size=25):
    """Continues generation from already-fully-formed text (chat-templated prompt +
    optional partial/corrupted assistant-turn content) -- does NOT re-apply the chat
    template, matching Stage 8's generate_continuation_batch convention exactly."""
    out = []
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        # model is a module-level global assigned further down (loaded per diagnostic
        # section); resolved correctly at call time, flagged by pyflakes as a false
        # positive on this deferred-binding pattern.
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)  # noqa: F821
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,  # noqa: F821
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out


print('===== DIAGNOSTIC 1: CORRUPTED-PREFILL CAUSAL TEST (stage9c-sft-rebalanced-v1 checkpoint) =====')
torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
base_for_adapter = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_for_adapter.config.use_cache = True
base_for_adapter = prepare_model_for_kbit_training(base_for_adapter, use_gradient_checkpointing=False)
model = get_peft_model(base_for_adapter, LoraConfig(**LORA_KWARGS))
_load_result = set_peft_model_state_dict(model, load_safetensors(str(ADAPTER_WEIGHTS)), adapter_name='default')
if getattr(_load_result, 'unexpected_keys', None):
    raise RuntimeError(f'Unexpected adapter keys: {_load_result.unexpected_keys}')
assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
model.eval()
_adapter_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _adapter_identity.update(name.encode()); _adapter_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'checkpoint_loaded': True, 'loaded_adapter_state_sha256': _adapter_identity.hexdigest()})

corruption_results = {}
for group_name, rows in ELIGIBLE.items():
    prefixes, directions, originals = [], [], []
    for row in rows:
        chat_prefix = chat_prefix_for(build_prompt(row['starting_state'], row['operations']))
        corrupted_prefix, direction = build_corrupted_prefix(chat_prefix, row['completion'], row['_target_slot'])
        original_answer = extract_answer(row['completion'])
        prefixes.append(corrupted_prefix); directions.append(direction); originals.append(original_answer)
    continuations = run_generation(prefixes, MAX_NEW_TOKENS_CONTINUATION, do_sample=False)
    samples = []
    outcome_counts = {}
    for row, prefix, cont, direction, original_answer in zip(rows, prefixes, continuations, directions, originals):
        new_answer = extract_answer(prefix + cont)
        outcome = classify_outcome(new_answer, original_answer, direction)
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        samples.append({'starting_state': row['starting_state'], 'operations': row['operations'],
            'original_answer': original_answer, 'corruption_direction': direction,
            'continuation_raw': cont, 'new_answer': new_answer, 'outcome': outcome})
    corruption_results[group_name] = {'n': len(rows), 'outcome_counts': outcome_counts, 'samples': samples}
    print(f'GROUP {group_name}:', {'n': len(rows), 'outcome_counts': outcome_counts})

del model, base_for_adapter
torch.cuda.empty_cache()


print('===== DIAGNOSTIC 2: FRESH BASE MODEL RAW LOGIT PRIOR CHECK (no LoRA, no fine-tuning) =====')
torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True,
    use_safetensors=True, trust_remote_code=False)
base_model.eval()
base_model.config.use_cache = True
for p in base_model.parameters(): p.requires_grad_(False)
print({'base_model_loaded_no_lora': True})


def teacher_forced_word_logprob(prefix_text, word):
    """Sum of log P(token_i | prefix + tokens_{<i}) for each token attributable to
    `word`, computed via a single teacher-forced forward pass on the JOINTLY tokenized
    prefix+word text. Uses character-offset mapping (not a token-count-boundary
    assumption) to find where the word begins in the joint tokenization, since BPE can
    re-merge across the prefix/word boundary and make `tokenize(prefix)` a different
    token sequence than the first len(tokenize(prefix)) tokens of
    `tokenize(prefix+word)` -- robust to that instead of asserting it can't happen."""
    n_prefix_chars = len(prefix_text)
    encoded = tokenizer(prefix_text + word, return_tensors='pt', return_offsets_mapping=True)
    full_ids = encoded.input_ids[0]
    offsets = encoded.offset_mapping[0].tolist()
    word_start_idx = next((i for i, (_s, e) in enumerate(offsets) if e > n_prefix_chars), None)
    if word_start_idx is None or word_start_idx == 0:
        raise RuntimeError(f'Could not locate word {word!r} start in the joint tokenization.')
    # base_model is a module-level global assigned further down (loaded fresh for the
    # base-model logit-prior check); same deferred-binding false positive as above.
    input_ids = full_ids.unsqueeze(0).to(next(base_model.parameters()).device)  # noqa: F821
    with torch.inference_mode():
        logits = base_model(input_ids=input_ids).logits[0]  # noqa: F821
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    total_logprob = 0.0
    first_token_logit = None
    for i in range(word_start_idx, full_ids.shape[0]):
        pos = i - 1  # logits at position p predict token p+1
        tok_id = full_ids[i].item()
        total_logprob += log_probs[pos, tok_id].item()
        if i == word_start_idx:
            first_token_logit = logits[pos, tok_id].item()
    return {'total_logprob': total_logprob, 'total_prob': math.exp(total_logprob),
            'first_token_logit': first_token_logit, 'n_word_tokens': int(full_ids.shape[0] - word_start_idx)}


# Representative scenarios: draw from both eligible groups so probes cover a Nib-ending
# and a Nomo-ending version of real, in-distribution reasoning content. Ground-truth
# CORRECT completions (via build_coded_completion, not model-generated) are used so the
# probe isolates the base model's own prior at this position, independent of anything
# the fine-tuned model would have generated.
_probe_rows = (ELIGIBLE['heads_to_tails'][:3] + ELIGIBLE['tails_to_heads'][:3])
logit_probe_results = []
for row in _probe_rows:
    correct_completion = build_coded_completion(row['starting_state'], row['operations'])
    reasoning_only = correct_completion.split('<answer>')[0]
    chat_prefix = chat_prefix_for(build_prompt(row['starting_state'], row['operations']))
    prefix_ending_at_answer = chat_prefix + reasoning_only + '<answer>'
    true_final_state = _trace(row['starting_state'], row['operations'])[-1]
    heads_score = teacher_forced_word_logprob(prefix_ending_at_answer, 'Heads')
    tails_score = teacher_forced_word_logprob(prefix_ending_at_answer, 'Tails')
    logit_probe_results.append({
        'starting_state': row['starting_state'], 'operations': row['operations'],
        'true_final_state': true_final_state,
        'last_coded_token_in_prefix': row['_target_slot']['token'],
        'heads_score': heads_score, 'tails_score': tails_score,
        'prior_favors': 'Heads' if heads_score['total_logprob'] > tails_score['total_logprob'] else 'Tails',
        'logprob_gap_tails_minus_heads': tails_score['total_logprob'] - heads_score['total_logprob']})
    print('PROBE:', logit_probe_results[-1]['starting_state'], logit_probe_results[-1]['operations'],
          '-> prior_favors=', logit_probe_results[-1]['prior_favors'],
          'gap(tails-heads)=', round(logit_probe_results[-1]['logprob_gap_tails_minus_heads'], 4))

del base_model
torch.cuda.empty_cache()


print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'corrupted-prefill-diagnostic-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)

report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED,
               'rebalanced_checkpoint_adapter_sha256': _adapter_sha,
               'source_evidence_sha256': _evidence_sha,
               'source_evidence_path': str(EVIDENCE_PATH.relative_to(_REPO_ROOT)),
               'max_new_tokens_continuation': MAX_NEW_TOKENS_CONTINUATION},
    'diagnostic_1_corrupted_prefill': corruption_results,
    'diagnostic_2_base_model_logit_prior': {
        'n_probes': len(logit_probe_results),
        'n_probes_favoring_tails': sum(1 for r in logit_probe_results if r['prior_favors'] == 'Tails'),
        'n_probes_favoring_heads': sum(1 for r in logit_probe_results if r['prior_favors'] == 'Heads'),
        'mean_logprob_gap_tails_minus_heads': sum(r['logprob_gap_tails_minus_heads'] for r in logit_probe_results) / len(logit_probe_results),
        'samples': logit_probe_results},
}
(OUTPUT / 'corrupted_prefill_diagnostic.json').write_text(json.dumps(report, indent=2))
print({'saved_to': str(OUTPUT / 'corrupted_prefill_diagnostic.json')})
print('DONE.')
