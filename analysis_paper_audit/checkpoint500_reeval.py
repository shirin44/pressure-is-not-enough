"""PART B, step 7: standalone RE-EVALUATION of checkpoint 500 under current audit tooling.

THIS IS NOT A RERUN OF THE ORIGINAL STAGE-1 TRAINING RESULT. The original 3,200-rollout
training run's own driver code and prompt format could not be located (see the prior
search-task report). This script instead:
  - Loads checkpoint 500's actual saved weights (~/aisi_checkpoints/step-500/,
    LoRA r=8/alpha=16/dropout=0.05/target_modules=[q,k,v,o]_proj on Qwen/Qwen2.5-3B-Instruct,
    confirmed from adapter_config.json).
  - Uses the CHAT-TEMPLATED prompt format (tokenizer.apply_chat_template), NOT the raw-string
    format the original training used (confirmed in Part A's stage07_signal_annealed.py read).
    This mirrors the Llama-derived fix rationale: we want to know what checkpoint 500 actually
    does when evaluated properly, not reproduce the original format's problems.
  - Reconstructs the Coin Flip prompt distribution (3-8 instructions) from
    src/data/coinflip.py's / reward_v3.py's generate_coinflip_example, since the paper's own
    Section 3.1/Appendix A text is not in this repo to check against directly.
  - Classifies every completion with the UNMODIFIED, already-existing 10-tier taxonomy
    (experiments/09_direct_indomain_synthetic_bridge/taxonomy.py:classify_candidate) -- imported,
    not reimplemented.

Inference only: LoRA loaded with the base model frozen, model.eval(), torch.inference_mode().
No optimizer, no GRPOTrainer, no gradient step, no weights ever written back."""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

REPO = Path.home() / 'stage09_repo'
sys.path.insert(0, str(REPO / 'src' / 'data'))
sys.path.insert(0, str(REPO / 'experiments' / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(REPO / 'experiments' / '09_direct_indomain_synthetic_bridge'))
from coinflip import generate_coinflip_example  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
CHECKPOINT_DIR = Path.home() / 'aisi_checkpoints' / 'step-500'
N_ROLLOUTS = 500
FLIPS_RANGE = (3, 8)
RUN_SEED = 20260922  # today's date; distinct from Stage 1's own RUN_SEED (unknown/lost) and
                      # Stage 7's RUN_SEED=20260825/REFERENCE_SEED=20260730, to avoid any overlap question
MAX_NEW_TOKENS = 256
TEMPERATURE, TOP_P = 0.8, 0.95  # same sampling settings used project-wide for GRPO rollouts
BATCH_SIZE = 25

t0 = time.time()
weights = CHECKPOINT_DIR / 'adapter_model.safetensors'
adapter_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
print({'adapter_dir': str(CHECKPOINT_DIR), 'adapter_sha256': adapter_sha256, 'gpu': torch.cuda.get_device_name(0),
       'n_rollouts': N_ROLLOUTS, 'flips_range': FLIPS_RANGE, 'run_seed': RUN_SEED,
       'prompt_format': 'CHAT_TEMPLATED (fix; original training used a raw string prompt -- see Part A finding)'})

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                            device_map='auto', trust_remote_code=False)
base.config.use_cache = True
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                        lora_dropout=.05, bias='none', task_type='CAUSAL_LM'))
res = set_peft_model_state_dict(model, load_safetensors(str(weights)), adapter_name='default')
assert not getattr(res, 'unexpected_keys', None), res.unexpected_keys
assert list(model.peft_config) == ['default']
model.eval()
print({'checkpoint_loaded': True, 'peft_config_ok': True})

ANSWER_IDS = tok.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        w = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= w and row[-w:].tolist() == ANSWER_IDS for row in input_ids],
                            device=input_ids.device, dtype=torch.bool)
STOP = StoppingCriteriaList([StopAfterAnswer()])

# ---- Build N_ROLLOUTS distinct Coin Flip prompts, 3-8 instructions, cycling deterministically
import random
rng = random.Random(RUN_SEED)
rows = []
seen = set()
seed_cursor = RUN_SEED
while len(rows) < N_ROLLOUTS:
    n_flips = rng.randint(*FLIPS_RANGE)
    prompt, truth = generate_coinflip_example(n_flips, seed_cursor)
    seed_cursor += 1
    if prompt in seen:
        continue
    seen.add(prompt)
    rows.append({'prompt': prompt, 'ground_truth': truth, 'n_flips': n_flips})
print({'n_prompts_built': len(rows), 'n_flips_distribution': {k: sum(r['n_flips'] == k for r in rows) for k in range(FLIPS_RANGE[0], FLIPS_RANGE[1] + 1)}})

def chat_text(p):
    return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)

torch.manual_seed(RUN_SEED)
torch.cuda.manual_seed_all(RUN_SEED)
completions = []
for start in range(0, len(rows), BATCH_SIZE):
    chunk = rows[start:start + BATCH_SIZE]
    texts = [chat_text(r['prompt']) for r in chunk]
    enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, stopping_criteria=STOP)
    gen = out[:, enc['input_ids'].shape[1]:].cpu().tolist()
    for ids in gen:
        eos_pos = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else None
        cut = ids[:eos_pos + 1] if eos_pos is not None else ids
        while eos_pos is None and cut and cut[-1] == tok.pad_token_id:
            cut = cut[:-1]
        completions.append(tok.decode(cut, skip_special_tokens=True))
    print(f'  generated {min(start + BATCH_SIZE, len(rows))}/{len(rows)}', flush=True)

# ---- Classify every completion with the unmodified 10-tier taxonomy
classified = []
for r, c in zip(rows, completions):
    result = classify_candidate(c, r['prompt'], ground_truth=r['ground_truth'])
    classified.append({**r, 'completion': c, **result})

tier_names = {1: 'literal', 2: 'vacuous', 3: 'literal_corruption', 4: 'prompt_domain_echo',
              5: 'position_driven_drift', 6: 'verbalized_substitution_attempt', 7: 'state_varying_candidate',
              8: 'state_predictive_code', 9: 'correct_globally_consistent_code', 10: 'causally_load_bearing_code'}
tier_counts = {k: 0 for k in range(1, 11)}
for row in classified:
    tier_counts[row['category']] += 1
tier9_or_10 = [row for row in classified if row['category'] in (9, 10)]

elapsed = time.time() - t0
report = {
    'RE_EVALUATION_NOT_ORIGINAL_TRAINING_RERUN': True,
    'caveat': ("This is checkpoint 500 evaluated under current audit tooling (chat-templated prompts, "
               "the existing 10-tier taxonomy classifier), sampled fresh today. It is NOT a rerun or "
               "reproduction of the original ~3,200-rollout Stage-1 training result, whose own driver "
               "code and prompt format could not be located. The two are different claims about "
               "different things (a live re-evaluation vs. a historical training process) and must not "
               "be conflated."),
    'config': {'model': MODEL_NAME, 'adapter_dir': str(CHECKPOINT_DIR), 'adapter_sha256': adapter_sha256,
               'n_rollouts': N_ROLLOUTS, 'flips_range': FLIPS_RANGE, 'run_seed': RUN_SEED,
               'prompt_format': 'chat_templated', 'temperature': TEMPERATURE, 'top_p': TOP_P,
               'max_new_tokens': MAX_NEW_TOKENS, 'elapsed_seconds': elapsed},
    'tier_distribution': {f'{k}_{tier_names[k]}': tier_counts[k] for k in range(1, 11)},
    'tier_distribution_fraction': {f'{k}_{tier_names[k]}': tier_counts[k] / N_ROLLOUTS for k in range(1, 11)},
    'structural_nonliteral_candidate_count': sum(row['structural_nonliteral_candidate'] for row in classified),
    'verbalized_substitution_attempt_count': sum(row['verbalized_substitution_attempt'] for row in classified),
    'tier_9_or_10_count': len(tier9_or_10),
    'tier_9_or_10_rows': tier9_or_10,
    'rows': classified,
}
print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'tier_9_or_10_rows')}, indent=1))

root = Path.home() / 'aisi_checkpoints'
for v in range(1, 1000):
    out_dir = root / f'checkpoint500-reeval-v{v}'
    if not out_dir.exists():
        break
out_dir.mkdir(parents=True)
(out_dir / 'checkpoint500_reeval.json').write_text(json.dumps(report, indent=1, default=str))
print('Evidence:', out_dir / 'checkpoint500_reeval.json')
print(f'Elapsed: {elapsed:.1f}s')
print('CHECKPOINT 500 RE-EVALUATION COMPLETE (inference only, no training).')
