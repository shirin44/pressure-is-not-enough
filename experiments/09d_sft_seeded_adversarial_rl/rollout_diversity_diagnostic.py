from __future__ import annotations

import difflib, hashlib, itertools, json, logging, statistics, sys, warnings
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stage 9d follow-up #2: the greedy-entropy diagnostic disconfirmed "SFT checkpoint is
# overconfident" (entropy essentially identical to Step-0's milestone-12). This tests
# the leading remaining hypothesis directly: does Stage 9c's checkpoint produce
# unusually HOMOGENEOUS sampled rollout groups (temperature=0.8, matching GRPO's real
# config) -- which would make advantage normalization (reward - mean)/(std + 1e-4)
# numerically extreme when std is near zero, largely independent of prompt content?
# Same base-model-isolation fix as the prior two diagnostics (fresh base model per
# checkpoint load). design.md documents the full result.

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
GROUP_SIZE = 8            # matches GRPOConfig's num_generations/GROUP_SIZE everywhere in this project
TEMPERATURE = 0.8; TOP_P = 0.95   # matches GRPOConfig exactly (sft_seeded_rl.py / every other stage)
RUN_SEED = 20260830

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
sys.path.insert(0, str(_STAGE09_DIR)); sys.path.insert(0, str(_STAGE07_DIR))
from synthetic_bridge import build_clean_length5_train_eval_split, build_prompt  # noqa: E402
from reward_v3 import score_completion_v2, completion_to_text, prompt_to_text  # noqa: E402

CHECKPOINTS = {
    'step0_milestone12': Path.home() / 'aisi_checkpoints' / 'exp3-step0-task-foundation-full-v1' / 'milestone-12' / 'adapter_model.safetensors',
    'stage9c_sft': Path.home() / 'aisi_checkpoints' / 'stage9c-sft-diagnostic-v1' / 'final_adapter' / 'adapter_model.safetensors',
}
for label, path in CHECKPOINTS.items():
    if not path.is_file(): raise FileNotFoundError(f'{label}: {path}')

STEP0_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)
MAIN_REWARD_PARAMS = {}  # score_completion_v2 defaults

CLEAN21_SEED = 20260831
_, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=21)
SAMPLE_ROWS = eval_rows[:8]  # SAME 8 prompts as entropy_diagnostic.py, for direct comparability
SAMPLE_PROMPTS = [build_prompt(r['starting_state'], r['operations']) for r in SAMPLE_ROWS]

print({'gpu': torch.cuda.get_device_name(0), 'n_sample_prompts': len(SAMPLE_PROMPTS), 'group_size': GROUP_SIZE,
       'temperature': TEMPERATURE, 'top_p': TOP_P,
       'checkpoints': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in CHECKPOINTS.items()}})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')


def load_checkpoint_fresh(weights_path):
    """Independent base-model load per checkpoint (same isolation fix as
    sft_seeded_rl.py / entropy_diagnostic.py -- test_checkpoint_isolation.py covers
    this script's pattern too)."""
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


def generate_rollout_group(model, prompt, group_size, seed_offset):
    """GROUP_SIZE sampled completions for ONE prompt, temperature/top_p matching
    GRPOConfig exactly -- batched generation (one call, group_size copies of the same
    prompt), exactly how TRL's own generation batches a rollout group."""
    torch.manual_seed(RUN_SEED + seed_offset); torch.cuda.manual_seed_all(RUN_SEED + seed_offset)
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat] * group_size, return_tensors='pt').to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=TEMPERATURE, top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    completions = tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return completions


def pairwise_diversity(completions):
    pairs = list(itertools.combinations(range(len(completions)), 2))
    similarities = [difflib.SequenceMatcher(None, completions[i], completions[j]).ratio() for i, j in pairs]
    exact_identical_pairs = sum(1 for i, j in pairs if completions[i] == completions[j])
    return {
        'n_pairs': len(pairs),
        'mean_pairwise_similarity': statistics.fmean(similarities),
        'max_pairwise_similarity': max(similarities),
        'min_pairwise_similarity': min(similarities),
        'exact_identical_pairs': exact_identical_pairs,
        'exact_identical_pair_fraction': exact_identical_pairs / len(pairs),
        'n_distinct_completions': len(set(completions)),
    }


def score_group(completions, prompt, ground_truth, reward_params):
    breakdowns = [score_completion_v2(completion_to_text(c), ground_truth, 1, 50, prompt=prompt_to_text(prompt), **reward_params)
                  for c in completions]
    rewards = [b['total'] for b in breakdowns]
    mean_r = statistics.fmean(rewards)
    std_r = statistics.pstdev(rewards)
    advantages = [(r - mean_r) / (std_r + 1e-4) for r in rewards]
    return {'rewards': rewards, 'reward_mean': mean_r, 'reward_std': std_r,
            'advantages': advantages, 'max_abs_advantage': max(abs(a) for a in advantages),
            'near_zero_variance': std_r < 1e-3}


results = {}
for label, weights_path in CHECKPOINTS.items():
    print(f'===== {label} =====')
    model = load_checkpoint_fresh(weights_path)
    groups = []
    for idx, (row, prompt) in enumerate(zip(SAMPLE_ROWS, SAMPLE_PROMPTS)):
        completions = generate_rollout_group(model, prompt, GROUP_SIZE, seed_offset=idx)
        diversity = pairwise_diversity(completions)
        baseline_scoring = score_group(completions, prompt, row['final_answer'], STEP0_REWARD_PARAMS)
        main_scoring = score_group(completions, prompt, row['final_answer'], MAIN_REWARD_PARAMS)
        group_result = {'prompt_index': idx, 'starting_state': row['starting_state'], 'operations': row['operations'],
            'completions': completions, 'diversity': diversity,
            'baseline_scoring': baseline_scoring, 'main_scoring': main_scoring}
        groups.append(group_result)
        print(f'  prompt {idx}: mean_pairwise_sim={diversity["mean_pairwise_similarity"]:.4f} '
              f'exact_identical_pairs={diversity["exact_identical_pairs"]}/{diversity["n_pairs"]} '
              f'n_distinct={diversity["n_distinct_completions"]}/{GROUP_SIZE} '
              f'baseline_reward_std={baseline_scoring["reward_std"]:.4f} '
              f'baseline_max_abs_adv={baseline_scoring["max_abs_advantage"]:.4f} '
              f'main_reward_std={main_scoring["reward_std"]:.4f} '
              f'main_max_abs_adv={main_scoring["max_abs_advantage"]:.4f} '
              f'near_zero_var(baseline/main)={baseline_scoring["near_zero_variance"]}/{main_scoring["near_zero_variance"]}')
    mean_sim = statistics.fmean(g['diversity']['mean_pairwise_similarity'] for g in groups)
    mean_baseline_std = statistics.fmean(g['baseline_scoring']['reward_std'] for g in groups)
    mean_main_std = statistics.fmean(g['main_scoring']['reward_std'] for g in groups)
    mean_baseline_max_adv = statistics.fmean(g['baseline_scoring']['max_abs_advantage'] for g in groups)
    mean_main_max_adv = statistics.fmean(g['main_scoring']['max_abs_advantage'] for g in groups)
    n_near_zero_baseline = sum(1 for g in groups if g['baseline_scoring']['near_zero_variance'])
    n_near_zero_main = sum(1 for g in groups if g['main_scoring']['near_zero_variance'])
    summary = {'n_groups': len(groups), 'grand_mean_pairwise_similarity': mean_sim,
        'grand_mean_baseline_reward_std': mean_baseline_std, 'grand_mean_main_reward_std': mean_main_std,
        'grand_mean_baseline_max_abs_advantage': mean_baseline_max_adv,
        'grand_mean_main_max_abs_advantage': mean_main_max_adv,
        'n_groups_near_zero_variance_baseline': n_near_zero_baseline,
        'n_groups_near_zero_variance_main': n_near_zero_main}
    print(f'SUMMARY [{label}]:', json.dumps(summary, indent=2))
    results[label] = {'summary': summary, 'groups': groups}
    del model
    torch.cuda.empty_cache()

print('===== COMPARISON =====')
s0 = results['step0_milestone12']['summary']
s9c = results['stage9c_sft']['summary']
comparison = {
    'step0_mean_pairwise_similarity': s0['grand_mean_pairwise_similarity'],
    'stage9c_mean_pairwise_similarity': s9c['grand_mean_pairwise_similarity'],
    'similarity_ratio_stage9c_over_step0': s9c['grand_mean_pairwise_similarity'] / s0['grand_mean_pairwise_similarity'],
    'step0_mean_baseline_reward_std': s0['grand_mean_baseline_reward_std'],
    'stage9c_mean_baseline_reward_std': s9c['grand_mean_baseline_reward_std'],
    'step0_mean_main_reward_std': s0['grand_mean_main_reward_std'],
    'stage9c_mean_main_reward_std': s9c['grand_mean_main_reward_std'],
    'step0_mean_baseline_max_abs_advantage': s0['grand_mean_baseline_max_abs_advantage'],
    'stage9c_mean_baseline_max_abs_advantage': s9c['grand_mean_baseline_max_abs_advantage'],
    'step0_mean_main_max_abs_advantage': s0['grand_mean_main_max_abs_advantage'],
    'stage9c_mean_main_max_abs_advantage': s9c['grand_mean_main_max_abs_advantage'],
    'step0_n_near_zero_variance_groups_baseline_of_8': s0['n_groups_near_zero_variance_baseline'],
    'stage9c_n_near_zero_variance_groups_baseline_of_8': s9c['n_groups_near_zero_variance_baseline'],
    'step0_n_near_zero_variance_groups_main_of_8': s0['n_groups_near_zero_variance_main'],
    'stage9c_n_near_zero_variance_groups_main_of_8': s9c['n_groups_near_zero_variance_main'],
}
print(json.dumps(comparison, indent=2))

OUTPUT_DIR = Path.home() / 'aisi_checkpoints' / 'stage9d-rollout-diversity-diagnostic'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_LOG = OUTPUT_DIR / 'rollout_diversity_diagnostic_result.json'
EVENT_LOG.write_text(json.dumps({'results': results, 'comparison': comparison}, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('ROLLOUT DIVERSITY DIAGNOSTIC COMPLETE.')
