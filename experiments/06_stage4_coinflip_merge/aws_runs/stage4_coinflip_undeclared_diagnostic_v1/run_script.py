"""Stage 4: merge checkpoint 500 (direct-RL Coin Flip) with the Dataset-B-bootstrapped
checkpoint 130 lineage, then run the undeclared-invention diagnostic on Coin Flip itself.

Base choice (decided, not asked): checkpoint 500 is the ONLY artifact in this project
actually RL-trained on Coin Flip (01_direct_rl_coinflip's own README: "merge that
capability into this Coin Flip model" -- checkpoint 500 is explicitly the thing to merge
INTO). checkpoint 130 (and its Dataset-B-refined descendant, checkpoint 150 under
dataset_b_sft_v1) was trained on fan/valve/lamp only and has never seen Coin Flip prompts.
So this is not an either/or -- it's checkpoint 500 as the base, with the seeded-invention
LoRA composed on top, per Stage 4's own original plan and 04_stability_investigation's
conclusion that checkpoint 500 is "now a viable Stage 4 merge target."

Merge mechanics: both adapters share an identical LoRA config (r=8, alpha=16, dropout=0.05,
target_modules q/k/v/o_proj) over the same base (Qwen/Qwen2.5-3B-Instruct), confirmed by
diffing adapter_config.json for both. Sequential composition, not weighted averaging:
step-500's LoRA is merged permanently into full bf16 weights (this IS the Coin-Flip-capable
model), then checkpoint-150's LoRA is attached on top as a live adapter for generation
(not merged again -- keeps it swappable/inspectable, e.g. for a future step-500-only
ablation without re-deriving the merged base).
"""
from __future__ import annotations

import gc, hashlib, importlib.metadata, json, logging, os, random, sys, warnings
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error(); _hf_logging.disable_progress_bar()
except Exception:
    pass

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU visible to this process.')
_cap_major, _cap_minor = torch.cuda.get_device_capability(0)
if _cap_major < 8:
    raise RuntimeError(f'bf16 requires Ampere+; found compute capability {_cap_major}.{_cap_minor}.')


def _version(name):
    return importlib.metadata.version(name)


_expected = {'transformers': '5.13.1', 'peft': '0.19.1'}
_actual = {k: _version(k) for k in _expected}
if _actual != _expected:
    raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

sys.path.insert(0, str(Path.home()))
from coinflip_module_wip import (  # noqa: E402  (local, uncommitted-on-purpose module -- see script docstring)
    generate_coinflip_example,
    audit_global_state_consistency,
    count_flips,
)

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
STEP500_ADAPTER = Path.home() / 'aisi_checkpoints' / 'step-500'
CHECKPOINT150_ADAPTER = Path.home() / 'aisi_checkpoints' / 'dataset_b_sft_v1' / 'trainer-output' / 'checkpoint-150'
MERGED_BASE_DIR = Path.home() / 'aisi_checkpoints' / 'stage4_coinflip_rl_merged_base'
OUTPUT_DIR = Path.home() / 'aisi_checkpoints' / 'stage4_coinflip_undeclared_diagnostic_v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_EXAMPLES = 6000
PROMPT_SEED = 20260825
N_FLIPS_RANGE = (2, 6)  # matches scripts/run_grpo_dryrun_colab.py's generate_dataset(40, (2, 6))
BATCH_SIZE = 64
MAX_NEW_TOKENS = 300
CHECKPOINT_EVERY = 200

print({'gpu': torch.cuda.get_device_name(0), 'n_examples': N_EXAMPLES,
       'prompt_seed': PROMPT_SEED, 'batch_size': BATCH_SIZE, **_actual})


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===== STEP 1: MERGE checkpoint 500 into a full-precision Coin-Flip-RL base =====
if not (MERGED_BASE_DIR / 'model.safetensors.index.json').is_file() and not any(
    MERGED_BASE_DIR.glob('model*.safetensors')
):
    print('===== MERGING CHECKPOINT 500 INTO BASE (one-time) =====')
    if not (STEP500_ADAPTER / 'adapter_config.json').is_file():
        raise RuntimeError(f'Invalid checkpoint: {STEP500_ADAPTER}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map={'': 0},
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    )
    step500_sha = sha256_of(STEP500_ADAPTER / 'adapter_model.safetensors')
    rl_model = PeftModel.from_pretrained(base, STEP500_ADAPTER, is_trainable=False)
    merged = rl_model.merge_and_unload()
    MERGED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_BASE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_BASE_DIR)
    manifest = {
        'source_adapter': str(STEP500_ADAPTER), 'source_adapter_sha256': step500_sha,
        'base_model': MODEL_NAME, 'merge_dtype': 'bfloat16',
    }
    atomic_json(MERGED_BASE_DIR / 'merge_manifest.json', manifest)
    print(f'MERGED BASE SAVED: {MERGED_BASE_DIR}  sha256(step-500 adapter)={step500_sha}')
    del base, rl_model, merged
    gc.collect(); torch.cuda.empty_cache()
else:
    print(f'===== MERGED BASE ALREADY EXISTS, REUSING: {MERGED_BASE_DIR} =====')

# ===== STEP 2: LOAD merged base + attach checkpoint-150 (Dataset-B-bootstrapped seeded capability) =====
print('===== LOAD MERGED BASE + ATTACH CHECKPOINT 150 =====')
if not (CHECKPOINT150_ADAPTER / 'adapter_config.json').is_file():
    raise RuntimeError(f'Invalid checkpoint: {CHECKPOINT150_ADAPTER}')
tokenizer = AutoTokenizer.from_pretrained(MERGED_BASE_DIR, trust_remote_code=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
merged_base = AutoModelForCausalLM.from_pretrained(
    MERGED_BASE_DIR, dtype=torch.bfloat16, device_map={'': 0},
    low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
)
checkpoint150_sha = sha256_of(CHECKPOINT150_ADAPTER / 'adapter_model.safetensors')
model = PeftModel.from_pretrained(merged_base, CHECKPOINT150_ADAPTER, is_trainable=False)
model.eval(); model.config.use_cache = True
for p in model.parameters():
    p.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
print({'checkpoint150_adapter_sha256': checkpoint150_sha, 'trainable_parameters': 0,
       'mode': 'eval, greedy decoding, Stage-4 merge (step-500 base + checkpoint-150 adapter)'})

# ===== STEP 3: BUILD 6000 UNDECLARED COIN FLIP PROMPTS =====
# NOTE: prompt text depends only on (starting_state, operations), not on the seed used to
# draw them -- for n_flips in [2,6] there are only 2*(2^2+..+2^6) = 248 distinct possible
# prompts. A large N therefore means many repeats of the same 248 underlying scenarios,
# which is why generation below uses SAMPLED (not greedy) decoding: under greedy decoding
# a repeated prompt yields an identical completion every time and adds zero information at
# this scale, whereas sampling (temperature 0.8, top_p 0.95 -- the exact rollout distribution
# checkpoint 500 was actually RL-trained under) makes every repeat an independent draw.
print(f'===== BUILDING {N_EXAMPLES} UNDECLARED COIN FLIP PROMPTS (seed {PROMPT_SEED}) =====')
rng = random.Random(PROMPT_SEED)
pool = []
seen_prompts = set()
for idx in range(N_EXAMPLES):
    n_flips = rng.randint(*N_FLIPS_RANGE)
    example_seed = rng.randint(0, 10**9)
    prompt, final_answer = generate_coinflip_example(n_flips, example_seed)
    seen_prompts.add(prompt)
    pool.append({
        'example_id': f'coinflip-undeclared-{idx:05d}',
        'n_flips': n_flips,
        'example_seed': example_seed,
        'prompt': prompt,
        'final_answer': final_answer,
    })
assert len(pool) == N_EXAMPLES
by_direction = {'Heads': sum(1 for r in pool if r['final_answer'] == 'Heads'),
                'Tails': sum(1 for r in pool if r['final_answer'] == 'Tails')}
print({'total': len(pool), 'distinct_prompt_texts': len(seen_prompts),
       'by_final_answer_direction': by_direction})

# ===== STEP 4: GENERATE (checkpointed, resumable) =====
print(f'===== SAMPLED GENERATION: {len(pool)} PROMPTS, temperature=0.8, top_p=0.95 '
      f'(matches checkpoint 500\'s own GRPO rollout distribution) =====')
GEN_SEED = PROMPT_SEED
torch.manual_seed(GEN_SEED)
SYSTEM = 'You solve state-tracking tasks accurately and follow the requested output format.'
PROGRESS = OUTPUT_DIR / 'raw_completions.json'

saved = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() and PROGRESS.stat().st_size else {'rows': []}
completed = {row['example_id'] for row in saved['rows']}
print(f'Resuming: {len(completed)}/{len(pool)} already completed.')

for start in range(0, len(pool), BATCH_SIZE):
    chunk = [row for row in pool[start:start + BATCH_SIZE] if row['example_id'] not in completed]
    if not chunk:
        continue
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['prompt']}],
        tokenize=False, add_generation_prompt=True) for row in chunk]
    batch = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                 temperature=0.8, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    for row, text in zip(chunk, texts):
        saved['rows'].append({**row, 'completion': text})
        completed.add(row['example_id'])
    if len(completed) % CHECKPOINT_EVERY < BATCH_SIZE or len(completed) == len(pool):
        atomic_json(PROGRESS, saved)
        print(f'SAVED: {len(completed)}/{len(pool)}')

atomic_json(PROGRESS, saved)
assert len(saved['rows']) == len(pool) == len(completed)
print(f'\nDONE: {len(saved["rows"])} raw completions saved to {PROGRESS}')
print('GENERATION COMPLETE.')
