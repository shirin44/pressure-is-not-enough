from __future__ import annotations

import hashlib, importlib.metadata, json, logging, os, sys, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU visible to this process.')
_cap_major, _cap_minor = torch.cuda.get_device_capability(0)
if _cap_major < 8:
    raise RuntimeError(f'bf16 requires Ampere+ (compute capability >= 8.0); found '
                        f'{torch.cuda.get_device_name(0)} (compute capability {_cap_major}.{_cap_minor}).')
def _version(name): return importlib.metadata.version(name)
_expected = {'transformers': '5.13.1', 'peft': '0.19.1', 'bitsandbytes': '0.50.0'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
RUN_SEED = 20260907
MAX_NEW_TOKENS = 150  # generous for 2-3 remaining steps + answer at length 5
FLIP_AT_STEPS = (2, 3)

# Stage 9e, direct code-word answer design: causal-flip test (2026-09-07). Inference
# only, no training, no checkpoint modification. Loads the SFT-only checkpoint from
# llama_sft_code_word_seeding.py (stage9e-llama-sft-code-word-v1/final_adapter).
#
# PRIMARY TEST (final-token flip) is addressed ANALYTICALLY, not run here: in this
# design the code word IS the answer with nothing generated after it, so flipping the
# final token trivially flips the recorded answer by construction -- there is nothing
# left to observe post-hoc. Stated explicitly in the evidence/report; the SECONDARY
# test (mid-sequence flip propagation) is the real causal evidence for this design,
# per the task's own anticipation of this outcome.

print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'flip_at_steps': FLIP_AT_STEPS, **_actual})

print('===== IMPORTS =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09C_DIR = _REPO_ROOT / 'experiments' / '09c_sft_diagnostic'
_STAGE09E_DIR = _REPO_ROOT / 'experiments' / '09e_same_different_llama'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
sys.path.insert(0, str(_STAGE09C_DIR)); sys.path.insert(0, str(_STAGE09E_DIR))
from reward_v3 import _extract_answer  # noqa: E402
from synthetic_bridge import _trace  # noqa: E402
from code_word_answer_bank import HEADS_CODE, TAILS_CODE, build_code_word_prompt  # noqa: E402
from code_word_causal_flip import build_mid_sequence_corrupted_prefix, classify_continuation  # noqa: E402

assert HEADS_CODE == 'Bek' and TAILS_CODE == 'Ner'

CHECKPOINT_DIR = Path(os.environ.get(
    'STAGE9E_CAUSAL_CHECKPOINT_DIR',
    str(Path.home() / 'aisi_checkpoints' / 'stage9e-llama-sft-code-word-v1' / 'final_adapter'),
)).expanduser()
WEIGHTS = CHECKPOINT_DIR / 'adapter_model.safetensors'
if not WEIGHTS.is_file(): raise FileNotFoundError(WEIGHTS)
_adapter_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
print({'checkpoint_dir': str(CHECKPOINT_DIR), 'adapter_sha256': _adapter_sha})

print('===== LOAD TOKENIZER =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)


def load_sft_checkpoint():
    """Fresh, independent base-model load + fresh PeftModel wrapper, loaded with the
    saved SFT adapter weights -- same checkpoint-isolation pattern used throughout
    this project (never reuse a base_model object across get_peft_model() calls)."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors
    fresh_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
    fresh_base.config.use_cache = True
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                       lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(fresh_base, lora)
    result = set_peft_model_state_dict(model, load_safetensors(str(WEIGHTS)), adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise RuntimeError(f'Unexpected adapter keys: {result.unexpected_keys}')
    assert list(model.peft_config) == ['default'], f'Unexpected adapter composition: {list(model.peft_config)}'
    model.eval()
    identity = hashlib.sha256()
    for name, param in model.named_parameters():
        if '.default.' not in name: continue
        identity.update(name.encode()); identity.update(param.detach().float().cpu().numpy().tobytes())
    return model, identity.hexdigest()


print('===== LOAD SFT CHECKPOINT (inference only, no training) =====')
model, checkpoint_identity = load_sft_checkpoint()
print({'checkpoint_loaded': True, 'checkpoint_identity_sha256': checkpoint_identity})


def chat_prefix_for(prompt_text):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_text}], tokenize=False, add_generation_prompt=True)


def run_generation(prompts_texts, max_new_tokens, batch_size=16):
    out = []
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out


print('===== LOAD TIER (b) HELD-OUT RESULTS FROM THE SFT EVIDENCE (all 21, all fully correct) =====')
# Read from the ORIGINAL save location (~/aisi_checkpoints/...), not a repo-relative
# aws_runs/ path -- that directory only exists locally (where evidence gets retrieved
# to); on the GPU instance itself, the SFT run's own output directory is still
# present and is the authoritative source, unchanged since that run completed.
SFT_EVIDENCE_PATH = Path.home() / 'aisi_checkpoints' / 'stage9e-llama-sft-code-word-v1' / 'stage9e_llama_sft_code_word.json'
if not SFT_EVIDENCE_PATH.is_file(): raise FileNotFoundError(SFT_EVIDENCE_PATH)
sft_evidence = json.loads(SFT_EVIDENCE_PATH.read_text())
tier_b_samples = sft_evidence['tier_b_heldout_same_pair']['samples']
eligible = [r for r in tier_b_samples if r['intermediate_tracking_correct'] and r['final_answer_correct']]
print({'tier_b_n': len(tier_b_samples), 'eligible_n': len(eligible)})
if len(eligible) < 20:
    raise RuntimeError(f'Only {len(eligible)} eligible scenarios -- below the requested 20-30 sample target.')

print('===== ZERO-STEP SANITY CHECK: does the checkpoint reproduce a sample of its own tier (b) completions? =====')
sanity_sample = eligible[:5]
sanity_prompts = [build_code_word_prompt(r['starting_state'], r['operations']) for r in sanity_sample]
sanity_completions = run_generation([chat_prefix_for(p) for p in sanity_prompts], 150)
sanity_matches = sum(1 for r, c in zip(sanity_sample, sanity_completions) if c.strip() == r['completion'].strip())
print({'sanity_n': len(sanity_sample), 'sanity_matches': sanity_matches})
if sanity_matches < len(sanity_sample) * 0.8:
    raise RuntimeError('Checkpoint does not reproduce its own reported tier (b) completions -- STOP, checkpoint identity suspect.')
print('PASSED: checkpoint reproduces its own reported behavior before any causal-flip work.')

print('===== SECONDARY TEST: MID-SEQUENCE FLIP PROPAGATION =====')
all_intervention_results = []
for flip_at_step in FLIP_AT_STEPS:
    print(f'--- flip_at_step={flip_at_step} ---')
    batch_meta = []
    batch_prefixes = []
    for r in eligible:
        prompt = build_code_word_prompt(r['starting_state'], r['operations'])
        prefix = chat_prefix_for(prompt)
        corruption = build_mid_sequence_corrupted_prefix(prefix, r['completion'], flip_at_step)
        if not corruption['ok']:
            print({'skipped': True, 'reason': corruption['reason'], 'starting_state': r['starting_state']})
            continue
        batch_meta.append({'starting_state': r['starting_state'], 'operations': r['operations'],
                            'flip_at_step': flip_at_step, 'original_token': corruption['original_token'],
                            'flipped_token': corruption['flipped_token']})
        batch_prefixes.append(corruption['corrupted_prefix'])

    continuations = run_generation(batch_prefixes, MAX_NEW_TOKENS)
    for meta, prefix, continuation in zip(batch_meta, batch_prefixes, continuations):
        # Reconstruct the FULL corrupted completion (prefix's completion portion + the
        # new continuation) for parsing -- the prefix already contains the corrupted
        # prefix through flip_at_step; the continuation is what the model generated
        # after that, which together form the corrupted completion for parse_state_slots_with_spans.
        corrupted_prompt = build_code_word_prompt(meta['starting_state'], meta['operations'])
        prefix_wo_chat = chat_prefix_for(corrupted_prompt)
        corrupted_completion_so_far = prefix[len(prefix_wo_chat):]
        full_corrupted_completion = corrupted_completion_so_far + continuation
        classification = classify_continuation(meta['starting_state'], meta['operations'],
                                                 meta['flip_at_step'], full_corrupted_completion)
        all_intervention_results.append({**meta, 'continuation': continuation,
                                          'full_corrupted_completion': full_corrupted_completion, **classification})
        print(f"  flip@{meta['flip_at_step']} {meta['original_token']}->{meta['flipped_token']}: "
              f"{classification['final_answer_classification']} "
              f"(propagation_matches_cf={classification['propagation_matches_counterfactual']})")

print('===== ANALYSIS: per intervention step, per direction, NOT pooled =====')


def summarize(results):
    n = len(results)
    if n == 0:
        return {'n': 0}
    n_genuine = sum(1 for r in results if r['final_answer_classification'] == 'tracks_flip_genuine_causal_use')
    n_disconnect = sum(1 for r in results if r['final_answer_classification'] == 'ignores_flip_stays_at_original')
    n_other = n - n_genuine - n_disconnect
    n_propagation_matches_cf = sum(1 for r in results if r['propagation_matches_counterfactual'])
    return {
        'n': n,
        'final_answer_tracks_flip_rate': n_genuine / n,
        'final_answer_ignores_flip_rate': n_disconnect / n,
        'final_answer_other_rate': n_other / n,
        'intermediate_propagation_matches_counterfactual_rate': n_propagation_matches_cf / n,
    }


analysis = {}
for flip_at_step in FLIP_AT_STEPS:
    step_results = [r for r in all_intervention_results if r['flip_at_step'] == flip_at_step]
    bek_to_ner = [r for r in step_results if r['original_token'] == 'bek']  # flipped Bek(Heads) -> Ner(Tails)
    ner_to_bek = [r for r in step_results if r['original_token'] == 'ner']  # flipped Ner(Tails) -> Bek(Heads)
    analysis[f'step_{flip_at_step}'] = {
        'all': summarize(step_results),
        'bek_to_ner_direction': summarize(bek_to_ner),
        'ner_to_bek_direction': summarize(ner_to_bek),
    }
print(json.dumps(analysis, indent=2))

model = None
torch.cuda.empty_cache()

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
OUTPUT_PREFIX = os.environ.get('STAGE9E_CAUSAL_OUTPUT_PREFIX', 'stage9e-llama-causal-flip')
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'{OUTPUT_PREFIX}-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'flip_at_steps': list(FLIP_AT_STEPS),
               'max_new_tokens': MAX_NEW_TOKENS, 'checkpoint_dir': str(CHECKPOINT_DIR),
               'adapter_sha256': _adapter_sha, 'checkpoint_identity_sha256': checkpoint_identity,
               'sanity_n': len(sanity_sample), 'sanity_matches': sanity_matches,
               'eligible_n': len(eligible),
               'primary_test_note': ('Final-token flip addressed analytically, not run: in this '
                   'design the code word IS the answer with nothing generated after it, so '
                   'flipping the final token trivially flips the recorded answer by construction '
                   '-- there is nothing left to observe post-hoc. The secondary (mid-sequence '
                   'propagation) test below is the real causal evidence for this design.')},
    'intervention_results': all_intervention_results,
    'analysis': analysis,
}
EVENT_LOG = OUTPUT / 'stage9e_llama_causal_flip.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA CAUSAL FLIP TEST COMPLETE.')
