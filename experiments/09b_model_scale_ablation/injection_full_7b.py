from __future__ import annotations

import hashlib, importlib.metadata, json, logging, math, os, random, re, statistics, sys, warnings
from collections import Counter, defaultdict
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

# Stage 9b: the FULL 50-step run of the 7B model-scale ablation (Qwen2.5-7B-Instruct,
# no Step-0 foundation -- zero-shot gate cleared, 0.7619 >= 0.68 -- 2-of-8 guaranteed
# rollout injection on the 16 gated bank scenarios, same reward gate/taxonomy/breakers
# as the 3B's Step 14b design). D_MAX is a FIXED constant (0.505, the 3B's own
# calibrated value, explicitly NOT measured on 7B -- see the comment above the KL-clamp
# self-test below and design.md for the full disclosed reasoning; per-run calibration
# was removed after dmax_sensitivity_check.py showed it produces training-relevant
# ~200x KL-magnitude swings across process boundaries). Diff-verified against
# injection_dryrun_7b.py before launch: differs ONLY in N_STEPS (8->50), the 'dry_run'
# flag (True->False), and output-path/run-identifier naming -- see design.md's launch
# entry for the actual diff output. This is the go/no-go-cleared full run, launched
# after: the zero-shot foundation check, three successive dry runs (calibration-based,
# then fixed-D_MAX), the D_MAX sensitivity check, and a read-only mechanism-behavior
# sanity check on the final dry run's evidence (all documented in design.md).

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'bitsandbytes.autograd._functions'):
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
RUN_SEED = 20260828
GROUP_SIZE = 8; MAX_DYNAMIC_ATTEMPTS = 3; MAX_NEW_TOKENS = 256
N_STEPS = 50               # full-run budget (design.md; exploratory, not 150 -- see the original task brief)
MILESTONE_EVERY = 4
BANK_MODE = 'coded'
INJECTIONS_PER_GROUP = 2  # design requirement 3: 2-of-8 first
WARMUP_UPDATES = 12; TARGET_LR = 1e-6
GRAD_BREAKER = 50.0; KL_BREAKER = 5.0  # unchanged from the 3B run; sanity-checked against real 7B telemetry, see below
# Per-run calibration (Phase 0) is REMOVED from the automatic launch path as of 2026-08-29 --
# D_MAX and ENTROPY_CLAMP_VALUE are now FIXED CONSTANTS, set just before the KL-clamp self-test
# below. See the comment there for the full, disclosed reasoning (dmax_sensitivity_check.py).
INSTRUMENTATION_RUN_DIR = Path.home() / 'aisi_checkpoints' / 'stage9b-model-scale-ablation'
INSTRUMENTATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
print({'gpu': torch.cuda.get_device_name(0), 'run_seed': RUN_SEED, 'n_steps': N_STEPS,
       'injections_per_group': INJECTIONS_PER_GROUP, 'calibration_phase_removed': True, **_actual})

print('===== IMPORTS FROM STAGE 9 / STAGE 9B (model-independent modules reused unchanged) =====')
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
_STAGE09B_DIR = _REPO_ROOT / 'experiments' / '09b_model_scale_ablation'
sys.path.insert(0, str(_STAGE09_DIR))
sys.path.insert(0, str(_STAGE07_DIR))
sys.path.insert(0, str(_STAGE09B_DIR))
from reward_v3 import verify_reward_invariant, parse_state_slots, LITERAL_TOKENS, _extract_answer, normalize_state_token  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from soft_stops import SoftStopTracker  # noqa: E402
from synthetic_bridge import CODED_TRAJECTORIES, build_clean_length5_train_eval_split  # noqa: E402
from step14_reward_gate import (  # noqa: E402
    BANK_SCENARIO_KEYS, bank_gate_resample_reasons, bank_gate_scale, is_bank_scenario,
    score_completion_gated, verify_step14_reward_invariant,
    DEFAULT_RAMP_STEPS, DEFAULT_INITIAL_SCALE, DEFAULT_FINAL_SCALE,
)
from step14b_injection import injected_trajectory_for_prompt, verify_grpo_ratio_is_unconditionally_one  # noqa: E402
from injection_scaling import inject_into_generation_output_n  # noqa: E402
# kl_calibration.py's compute_clamp_from_pool/solve_kl_clamp_bound/floor_d_max are no longer
# imported here -- Phase 0 (which used them) is removed from the automatic launch path as of
# 2026-08-29 (dmax_sensitivity_check.py finding). The module itself, and
# calibration_seed_stability_check.py, remain in this directory for reference/future use, per
# instruction -- they are simply not exercised by this script anymore.

ACTIVE_TRAJECTORIES = CODED_TRAJECTORIES
assert len(ACTIVE_TRAJECTORIES) == 16
assert all(row['coded'] for row in ACTIVE_TRAJECTORIES)
assert BANK_SCENARIO_KEYS == {(row['starting_state'], tuple(row['operations'])) for row in ACTIVE_TRAJECTORIES}
_active_bank_sha256 = hashlib.sha256(json.dumps([
    {'starting_state': row['starting_state'], 'operations': row['operations'],
     'completion': row['completion']} for row in ACTIVE_TRAJECTORIES
], sort_keys=True).encode()).hexdigest()
print('REWARD-GATE BANK STARTUP AUDIT:', {
    'active_bank_size': len(ACTIVE_TRAJECTORIES), 'active_bank_sha256': _active_bank_sha256,
    'ramp_steps': DEFAULT_RAMP_STEPS, 'initial_scale': DEFAULT_INITIAL_SCALE, 'final_scale': DEFAULT_FINAL_SCALE})

_step14_invariant_report = verify_step14_reward_invariant()
assert _step14_invariant_report['margin_code_over_gated_at_full_anneal'] > 0
assert _step14_invariant_report['margin_gated_over_malformed'] > 0
assert _step14_invariant_report['vacuous_exceeds_gated_literal_at_full_anneal'] is False
assert _step14_invariant_report['malformed_ties_gated_at_full_anneal'] is False
print('STEP 14 REWARD INVARIANT (fresh re-derivation, model-independent, reused unchanged):', _step14_invariant_report)

BRIDGE_REWARD_PARAMS = {}
_bridge_margins = verify_reward_invariant()
assert _bridge_margins['margin_correct_over_wrong'] > 0
assert _bridge_margins['margin_wrong_over_malformed'] > 0
print('BRIDGE REWARD INVARIANT:', _bridge_margins)


print('===== BUILD PRIMARY LENGTH-5 TRAIN/EVAL POOLS (same clean-21, reused unchanged) =====')
EVAL_LENGTHS = [5]
N_EVAL_PER_LENGTH = 21
CLEAN21_SEED = 20260831  # same hardcoded seed as Stage 9's own scripts -- identical 21 scenarios
EXPECTED_CLEAN21_SHA256 = '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'

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

train_rows, eval_rows = build_clean_length5_train_eval_split(seed=CLEAN21_SEED, n_eval=N_EVAL_PER_LENGTH)
_eval_sha256 = hashlib.sha256(json.dumps(eval_rows, sort_keys=True).encode()).hexdigest()
assert _eval_sha256 == EXPECTED_CLEAN21_SHA256
EVAL_POOL_BY_LENGTH = {5: eval_rows}
# NOTE: train_rows above come from THIS SAME call (build_clean_length5_train_eval_split),
# not an independently-seeded split -- that function partitions all 64 length-5 scenarios
# into eval_scenarios and train_scenarios = all_scenarios - eval_keys, so disjointness from
# eval AND from the bank-16 is guaranteed by construction, not by chance across two
# uncoordinated RNG draws. (An earlier version of this script called a second, independently-
# seeded split for the train pool and hit exactly the train/eval overlap this project's own
# precedent -- the length-4 bug referenced in Stage 9's build_train_eval_split docstring --
# warns about; caught immediately by the assertion below before any GPU time was spent
# training on it, fixed here rather than silently.)

TRAIN_POOL = []
for row in train_rows:
    TRAIN_POOL.append({'prompt': build_cot_prompt(row['starting_state'], row['operations']),
                        'ground_truth': row['final_answer'], 'n_flips': 5})
random.Random(RUN_SEED).shuffle(TRAIN_POOL)
assert {r['prompt'] for r in TRAIN_POOL}.isdisjoint(
    {build_cot_prompt(row['starting_state'], row['operations']) for row in EVAL_POOL_BY_LENGTH[5]})
_bank_overlap = {(r['starting_state'], tuple(r['operations'])) for r in EVAL_POOL_BY_LENGTH[5]} & \
                {(r['starting_state'], tuple(r['operations'])) for r in ACTIVE_TRAJECTORIES}
assert not _bank_overlap
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL), 'eval_n': len(EVAL_POOL_BY_LENGTH[5]), 'eval_sha256_verified': True,
       'train_eval_disjoint': True, 'train_bank_disjoint': True})


print('===== LOAD UNTOUCHED 7B BASE -- NO STEP-0 ADAPTER (gate 1 result: SKIP_STEP0_INJECT_DIRECTLY) =====')
random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
LORA_KWARGS = dict(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                    lora_dropout=.05, bias='none', task_type='CAUSAL_LM')  # IDENTICAL to the 3B setup (design req. 5)
print('GPU memory after 8-bit base load (before any LoRA):',
      {'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 3),
       'max_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3)})

ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= width and row[-width:].tolist() == ANSWER_IDS
                             for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])


print('===== KL/ENTROPY PATCH INFRASTRUCTURE (installed once, clamp toggled per phase) =====')
import trl.trainer.grpo_trainer as _grpo_mod
if not hasattr(_grpo_mod, 'entropy_from_logits'):
    raise ImportError('trl.trainer.grpo_trainer.entropy_from_logits not found; do not proceed unverified.')
_original_entropy_from_logits = _grpo_mod.entropy_from_logits
ENTROPY_CLAMP_STATE = {'value': None}  # set after Phase 0 measures a healthy range
def _clamped_entropy_from_logits(logits, chunk_size=128):
    raw = _original_entropy_from_logits(logits, chunk_size=chunk_size)
    if ENTROPY_CLAMP_STATE['value'] is None:
        return raw
    return torch.clamp(raw, max=ENTROPY_CLAMP_STATE['value'])
_grpo_mod.entropy_from_logits = _clamped_entropy_from_logits
assert _grpo_mod.entropy_from_logits is _clamped_entropy_from_logits

if not hasattr(GRPOTrainer, '_get_per_token_logps_and_entropies') or not hasattr(GRPOTrainer, 'compute_loss'):
    raise ImportError('Expected GRPOTrainer methods not found at their known names; do not proceed unverified.')
_original_get_logps = GRPOTrainer._get_per_token_logps_and_entropies
_original_compute_loss = GRPOTrainer.compute_loss
INSTRUMENTATION = {'inside_compute_loss': False, 'physical_step': 0, 'logps_calls': [], 'compute_loss_calls': []}
KL_CLAMP_STATE = {'enabled': False, 'd_max': None, 'current_inputs': None,
                   'engaged_token_count': 0, 'total_token_count': 0}

def _hash_id_rows(id_tensor):
    rows = id_tensor.detach().cpu().tolist()
    return [hashlib.sha256(str(row).encode()).hexdigest() for row in rows]

def _patched_get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
    logps, entropies, aux_loss = _original_get_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    is_policy_call = INSTRUMENTATION['inside_compute_loss']
    INSTRUMENTATION['logps_calls'].append({
        'call_index': len(INSTRUMENTATION['logps_calls']), 'physical_step': INSTRUMENTATION['physical_step'],
        'tag': 'policy' if is_policy_call else 'reference',
        'row_hashes': _hash_id_rows(input_ids[:, -logits_to_keep:]),
        'logps': logps.detach().float().cpu().tolist()})
    if is_policy_call and KL_CLAMP_STATE['enabled'] and KL_CLAMP_STATE['current_inputs'] is not None:
        current_inputs = KL_CLAMP_STATE['current_inputs']
        ref = current_inputs.get('ref_per_token_logps')
        if ref is not None and ref.shape == logps.shape:
            diff = ref - logps.detach()
            d_max = KL_CLAMP_STATE['d_max']
            diff_clamped = torch.clamp(diff, min=-d_max, max=d_max)
            engaged = (diff.abs() > d_max)
            KL_CLAMP_STATE['engaged_token_count'] += int(engaged.sum().item())
            KL_CLAMP_STATE['total_token_count'] += diff.numel()
            current_inputs['ref_per_token_logps'] = (logps.detach() + diff_clamped).detach()
    return logps, entropies, aux_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    record = {'call_index': len(INSTRUMENTATION['compute_loss_calls']), 'physical_step': INSTRUMENTATION['physical_step'],
        'completion_ids_hash': _hash_id_rows(inputs['completion_ids']),
        'completion_mask': inputs['completion_mask'].detach().cpu().tolist(),
        'advantages': inputs['advantages'].detach().float().cpu().tolist(),
        'old_per_token_logps_present': inputs.get('old_per_token_logps') is not None,
        'ref_per_token_logps_present': inputs.get('ref_per_token_logps') is not None}
    INSTRUMENTATION['inside_compute_loss'] = True
    KL_CLAMP_STATE['current_inputs'] = inputs
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
    finally:
        INSTRUMENTATION['inside_compute_loss'] = False
        KL_CLAMP_STATE['current_inputs'] = None
    loss_value = result[0] if return_outputs else result
    record['loss'] = float(loss_value.detach().item())
    INSTRUMENTATION['compute_loss_calls'].append(record)
    return result

GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_per_token_logps_and_entropies
GRPOTrainer.compute_loss = _patched_compute_loss
assert GRPOTrainer._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
assert GRPOTrainer.compute_loss is _patched_compute_loss
print('Patches installed. KL clamp disabled (pass-through) and entropy clamp unset for Phase 0 measurement.')


def build_fresh_trainer(*, max_steps, run_seed, lr_curve_len, warmup_updates, target_lr):
    """Wrap the SAME already-loaded 8-bit base_model with a FRESH LoraConfig each time --
    discards any adapter weights from a prior phase without re-downloading/re-quantizing
    the 7GB base weights. base_model's own parameters are frozen and untouched by LoRA
    training (PEFT never mutates base weights unless merge_and_unload() is called, which
    is never called here), so this is a genuine fresh start each time, not a continuation."""
    model = get_peft_model(base_model, LoraConfig(**LORA_KWARGS))
    from torch.optim.lr_scheduler import LambdaLR
    args = GRPOConfig(output_dir=str(INSTRUMENTATION_RUN_DIR / '_trainer_scratch'),
        per_device_train_batch_size=1, gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
        max_steps=max_steps, learning_rate=target_lr, lr_scheduler_type='linear', warmup_steps=warmup_updates,
        bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
        max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
        logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
        report_to='none', remove_unused_columns=False, disable_dropout=True, seed=run_seed, data_seed=run_seed)

    candidate_calls = []
    def diagnostic_reward(prompts, completions, **kwargs):
        truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
        # Latent off-by-one, found here and disclosed rather than silently patched: the
        # already-committed step14b_bridge_reward_gate_injection_dryrun.py's diagnostic_reward
        # reads trainer.state.global_step directly (0-indexed; stays 0 for all GROUP_SIZE
        # micro-batches of the FIRST optimizer step, only becoming 1 after it completes),
        # then passes that straight into score_completion_gated/bank_gate_scale, which require
        # step>=1. That script never crashed on it only because its train pool's first-drawn
        # scenario never happened by chance to be a gated bank scenario across this project's
        # prior launches (~37% chance per launch of hitting it) -- not because the logic is
        # actually correct. INSTRUMENTATION['physical_step'] (maintained by the on_step_begin
        # callback below as int(state.global_step)+1, i.e. already 1-indexed) is the value
        # every OTHER patch in this file correctly uses for exactly this reason; using it here
        # too, instead of re-deriving an off-by-one value from trainer.state.global_step directly.
        physical_step = INSTRUMENTATION['physical_step'] or 1
        from reward_v3 import completion_to_text, prompt_to_text
        texts = [completion_to_text(x) for x in completions]
        prompt_texts = [prompt_to_text(x) for x in prompts]
        breakdowns = [score_completion_gated(t, y, physical_step, max_steps, prompt=p, **BRIDGE_REWARD_PARAMS)
                      for t, y, p in zip(texts, truths, prompt_texts)]
        call = {'texts': texts, 'truths': list(truths), 'prompts': prompt_texts, 'breakdowns': breakdowns,
                'rewards': [x['total'] for x in breakdowns], 'physical_step': physical_step}
        candidate_calls.append(call)
        return call['rewards']

    trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args,
        train_dataset=train_dataset, processing_class=tokenizer)
    trainer_ref[0] = trainer
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    trainer.create_optimizer()
    def lr_factor(update_index):
        if update_index < warmup_updates: return 0.1 + 0.9 * update_index / (warmup_updates - 1)
        decay = lr_curve_len - warmup_updates
        return max(0.0, (lr_curve_len - update_index) / decay) if decay > 0 else 1.0
    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lambda s: lr_factor(s))
    assert int(trainer.args.steps_per_generation) == GROUP_SIZE
    assert trainer.ref_model is None
    assert type(trainer)._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
    assert type(trainer).compute_loss is _patched_compute_loss
    assert verify_grpo_ratio_is_unconditionally_one(
        gradient_accumulation_steps=int(trainer.args.gradient_accumulation_steps),
        steps_per_generation=int(trainer.args.steps_per_generation),
        num_iterations=int(trainer.args.num_iterations)), 'GRPO ratio=1 invariant does not hold for this config'
    original_generate = model.generate
    def generate_stopped(*a, **kw):
        kw.setdefault('stopping_criteria', ANSWER_STOP)
        return original_generate(*a, **kw)
    model.generate = generate_stopped
    trainer.model.generate = generate_stopped
    return trainer, model, candidate_calls

trainer_ref = [None]


print('===== D_MAX AND ENTROPY CLAMP: FIXED CONSTANTS, NOT PER-RUN CALIBRATION =====')
# Per-run calibration (Phase 0) is REMOVED from the automatic launch path as of 2026-08-29,
# per the dmax_sensitivity_check.py finding: two 8-step trials, identical seed/model/
# injection/scenario draws (scenario match verified: 8/8 identical prompt hashes), differing
# ONLY in D_MAX (0.050781 "low" vs. 0.736196 "high", the two already-observed endpoints of a
# ~14.5x same-seed calibration swing traced to cross-process GPU/CUDA nondeterminism, not a
# seeding bug) -- produced near-identical clamp ENGAGEMENT rates (21.0% vs 21.8%) but a
# ~200x difference in the aggregate KL metric (mean 0.00028 vs 0.061), because the clamp
# ceiling itself (CLAMP_VALUE = exp(D_MAX)-D_MAX-1) differs by ~270x between the two
# endpoints and ~21% of tokens hit it either way. This directly scales the beta*KL
# regularization term in GRPO's loss (beta=0.04) -- a mechanistically real, non-trivial
# effect on training dynamics, not measured-but-inconsequential noise. Verdict: per-run
# calibration is NOT safe to use for this ablation; a single fixed D_MAX is used instead.
#
# FIXED_D_MAX = 0.505: this is the 3B model's OWN calibrated D_MAX
# (step14b_bridge_reward_gate_injection_dryrun.py, D_MAX=0.504718, from a mature 500-step
# instrumentation run -- see step14_design.md). STATED EXPLICITLY, not glossed over: this
# value was NEVER MEASURED on the 7B model. It is borrowed from the 3B model specifically
# because per-run measurement on the 7B was found unreliable across process boundaries (the
# sensitivity-check finding above) -- reliably measuring it on 7B would require repeated,
# expensive independent-PROCESS sampling (in-process seed variation alone understated the
# true swing: the 4-seed check within one process gave 0.0508-0.0584, but an independent
# process with the "same" seed gave 0.736196). This is a DELIBERATE, DISCLOSED EXCEPTION to
# this project's "always measure, never guess" practice -- every other constant in this
# ablation (the zero-shot foundation check, the breaker-threshold sanity check against real
# 7B telemetry, the entropy clamp below) is measured on the 7B model itself; D_MAX alone is
# not, for the specific, disclosed reason above.
FIXED_D_MAX = 0.505
D_MAX = FIXED_D_MAX
CLAMP_VALUE = math.exp(D_MAX) - D_MAX - 1

# ENTROPY_CLAMP_VALUE = 1.2189: UNLIKE D_MAX, this IS a 7B-measured value -- it was
# calibrated on this same 7B model during the earlier (now-removed-from-the-automatic-path)
# Phase 0 of the run that completed cleanly on 2026-08-29 (injection_dryrun_7b.py,
# 'measured_7b_calibration'). It is being frozen and reused here rather than re-measured
# per launch, for the same practical reason as D_MAX (Phase 0 no longer runs automatically),
# but it does NOT carry D_MAX's cross-model-borrowing caveat. It DOES carry a narrower,
# still-disclosed caveat: its own cross-process stability was never independently checked
# the way D_MAX's was (no entropy-clamp analogue of dmax_sensitivity_check.py was run) --
# flagged here rather than silently assumed stable by analogy.
ENTROPY_CLAMP_VALUE = 1.2189

# BREAKER SANITY (design requirement 6, now satisfied by REAL Phase-1-equivalent telemetry
# rather than calibration-only telemetry -- arguably stronger evidence): both the completed
# 8-step dry run (2026-08-29) and both dmax_sensitivity_check.py trials showed 7B grad_norm
# topping out around 34-40 (comfortably under the unchanged GRAD_BREAKER=50.0) and kl topping
# out around 0.12 at worst (comfortably under KL_BREAKER=5.0) under REAL training, not just
# calibration steps. No flag needed; breakers are not close to normal operating range.
print({'FIXED_D_MAX': FIXED_D_MAX, 'D_MAX_source': '3B model\'s own calibrated value, NOT measured on 7B (see comment above)',
       'CLAMP_VALUE': CLAMP_VALUE, 'ENTROPY_CLAMP_VALUE': ENTROPY_CLAMP_VALUE,
       'ENTROPY_CLAMP_VALUE_source': 'measured on 7B (2026-08-29 dry run Phase 0), frozen and reused',
       'calibration_phase_removed_from_automatic_launch_path': True})


print('===== KL CLAMP SELF-TEST (same standalone check as the 3B run, against this D_MAX) =====')
def _reference_k3_formula(per_token_logps, inputs):
    ref = inputs['ref_per_token_logps']; diff = ref - per_token_logps
    return torch.exp(diff) - diff - 1
def _apply_clamp_standalone(per_token_logps, inputs, d_max):
    ref = inputs['ref_per_token_logps']; diff = ref - per_token_logps.detach()
    diff_clamped = torch.clamp(diff, min=-d_max, max=d_max)
    inputs['ref_per_token_logps'] = (per_token_logps.detach() + diff_clamped).detach()
    return diff, diff_clamped

policy1 = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
ref1 = torch.tensor([-1.0, -1.0, -1.0 + D_MAX + 9.5])
inputs1 = {'ref_per_token_logps': ref1.clone()}
_apply_clamp_standalone(policy1, inputs1, D_MAX)
kl1 = _reference_k3_formula(policy1, inputs1)
assert math.isclose(kl1[2].item(), CLAMP_VALUE, rel_tol=1e-4)
assert math.isclose(kl1[0].item(), 0.0, abs_tol=1e-6)
policy2 = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True); ref2 = torch.tensor([-1.2, -1.8, -0.6])
inputs2 = {'ref_per_token_logps': ref2.clone()}
diff2, diff2_clamped = _apply_clamp_standalone(policy2, inputs2, D_MAX)
assert torch.allclose(diff2, diff2_clamped)
print('PASSED: KL clamp self-test (pathological token bounded, healthy tokens unaffected).')

ENTROPY_CLAMP_STATE['value'] = ENTROPY_CLAMP_VALUE
KL_CLAMP_STATE['enabled'] = True
KL_CLAMP_STATE['d_max'] = D_MAX
KL_CLAMP_STATE['engaged_token_count'] = 0
KL_CLAMP_STATE['total_token_count'] = 0
print(f'KL CLAMP ENABLED FOR PHASE 1: CLAMP_VALUE={CLAMP_VALUE:.4f}, D_MAX={D_MAX:.6f}, '
      f'ENTROPY_CLAMP_VALUE={ENTROPY_CLAMP_VALUE:.4f}')


print('===== PHASE 1: FULL 50-STEP INJECTION RUN (fresh LoRA init, fixed D_MAX=0.505) =====')
LOGICAL_OFFSET = 0  # no Step-0 foundation lineage -- untouched 7B base, offset is 0 by construction
random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
diagnostic_trainer, model, candidate_calls = build_fresh_trainer(
    max_steps=N_STEPS, run_seed=RUN_SEED, lr_curve_len=N_STEPS, warmup_updates=WARMUP_UPDATES, target_lr=TARGET_LR)
trainer_ref[0] = diagnostic_trainer
INSTRUMENTATION['physical_step'] = 0
_fresh_init_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _fresh_init_identity.update(name.encode()); _fresh_init_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'fresh_lora_init': True, 'active_adapters': list(model.peft_config),
       'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest()})

print('===== INSTALL STEP 14B-STYLE ROLLOUT-GROUP INJECTION AT N=%d-OF-%d =====' % (INJECTIONS_PER_GROUP, GROUP_SIZE))
INJECTION_LOG = []
_original_generate_method = type(diagnostic_trainer)._generate
def _injection_aware_generate(self, prompts):
    result = _original_generate_method(self, prompts)
    prompt_ids, completion_ids, tool_mask, completions, logprobs, extra_fields, images, tool_images = result
    new_completion_ids, new_completions, injected = inject_into_generation_output_n(
        prompts, completion_ids, completions, tokenizer, INJECTIONS_PER_GROUP)
    if injected:
        INJECTION_LOG.append({'physical_step': INSTRUMENTATION['physical_step'],
            'injected_indices': list(injected.keys()),
            'scenarios': [{'starting_state': r['starting_state'], 'operations': r['operations']}
                          for r in injected.values()]})
    return prompt_ids, new_completion_ids, tool_mask, new_completions, logprobs, extra_fields, images, tool_images
type(diagnostic_trainer)._generate = _injection_aware_generate
assert type(diagnostic_trainer)._generate is _injection_aware_generate
print({'injection_installed': True, 'injections_per_bank_scenario_group': INJECTIONS_PER_GROUP})


print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9b-7b-injection-full-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG = OUTPUT / 'stage9b_7b_injection_full.json'
INSTRUMENTATION_LOG = OUTPUT / 'per_token_instrumentation.json'

event = {'config': {'model': MODEL_NAME, 'run_seed': RUN_SEED, 'n_steps': N_STEPS, 'dry_run': False,
         'milestone_every': MILESTONE_EVERY, 'entropy_coef': .05, 'entropy_clamp_value': ENTROPY_CLAMP_VALUE,
         'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_d_max': D_MAX, 'warmup_updates': WARMUP_UPDATES,
         'target_lr': TARGET_LR, 'max_new_tokens': MAX_NEW_TOKENS, 'stop': '</answer>',
         'hard_step_ceiling': N_STEPS, 'grad_breaker': GRAD_BREAKER, 'kl_breaker': KL_BREAKER,
         'breaker_sanity_check': 'satisfied by real Phase-1-equivalent telemetry from the 2026-08-29 '
                                  'dry run and dmax_sensitivity_check.py (grad_norm max ~34-40 vs '
                                  'breaker 50.0, kl max ~0.12 vs breaker 5.0) -- see design.md',
         'foundation_checkpoint': None, 'skip_step0_reason': 'zeroshot_check_7b.py: 0.7619 >= 0.68 floor '
                                  '(single-process measurement, not independently reproduced across '
                                  'processes -- see design.md cross-process-reproducibility caveat)',
         'active_adapters': list(model.peft_config),
         'mechanism': 'stage9b_model_scale_ablation_reward_gate_plus_rollout_injection',
         'gate_ramp_steps': DEFAULT_RAMP_STEPS, 'gate_initial_scale': DEFAULT_INITIAL_SCALE,
         'gate_final_scale': DEFAULT_FINAL_SCALE, 'injections_per_bank_scenario_group': INJECTIONS_PER_GROUP,
         'grpo_ratio_is_one_invariant_verified': True,
         'd_max_fixed_not_calibrated': {'value': FIXED_D_MAX, 'source': '3B model\'s own calibrated D_MAX, '
                                         'NOT measured on 7B -- see design.md and the comment above the '
                                         'KL clamp self-test for the full disclosed reasoning'},
         'soft_stop_config': {'absolute_word_floor': SoftStopTracker().absolute_word_floor,
                               'floor_consecutive_steps': SoftStopTracker().floor_consecutive_steps,
                               'trend_min_observations': SoftStopTracker().trend_min_observations,
                               'severe_accuracy_window': SoftStopTracker().severe_accuracy_window,
                               'severe_accuracy_threshold': SoftStopTracker().severe_accuracy_threshold},
         'loaded_adapter_state_sha256': _fresh_init_identity.hexdigest()},
         'groups': [], 'telemetry': [], 'adapter_updates': [], 'milestones': [], 'training_started': False,
         'purpose': 'stage9b_model_scale_ablation_7b_2of8_injection_full',
         'auxiliary_bank_mode': BANK_MODE, 'auxiliary_bank_sha256': _active_bank_sha256,
         'step14_reward_invariant_report': _step14_invariant_report, 'injection_log': INJECTION_LOG}

def save_event():
    tmp = EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(event, indent=2)); tmp.replace(EVENT_LOG)
    if not EVENT_LOG.is_file() or not EVENT_LOG.stat().st_size: raise RuntimeError('Evidence save failed.')
def save_instrumentation():
    tmp = INSTRUMENTATION_LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(INSTRUMENTATION, indent=2)); tmp.replace(INSTRUMENTATION_LOG)
def adapter_state_evidence():
    digest = hashlib.sha256(); squared = 0.0; count = 0
    for name, param in model.named_parameters():
        if '.default.' not in name: continue
        value = param.detach().float().cpu().contiguous()
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
        squared += float(value.square().sum()); count += value.numel()
    return {'sha256': digest.hexdigest(), 'l2_norm': math.sqrt(squared), 'parameter_count': count}

save_event(); save_instrumentation()
base_generate = diagnostic_trainer._generate_and_score_completions
def dynamic_generate(inputs):
    for attempt in range(1, MAX_DYNAMIC_ATTEMPTS + 1):
        result = base_generate(inputs); call = candidate_calls[-1]
        tensor_evidence = {}
        for key, value in result.items():
            if torch.is_tensor(value) and value.numel() <= 200000:
                cpu = value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
                tensor_evidence[key] = {'shape': list(cpu.shape), 'dtype': str(cpu.dtype), 'values': cpu.tolist()}
        row_hashes = _hash_id_rows(result['completion_ids']) if 'completion_ids' in result else None
        correct = sum(x['r_task'] == 4.0 for x in call['breakdowns'])
        structural = sum(x['p_structure'] == 0.0 for x in call['breakdowns'])
        reasons = bank_gate_resample_reasons(call['rewards'], correct_count=correct, structural_count=structural, group_size=GROUP_SIZE)
        accepted = not reasons or attempt == MAX_DYNAMIC_ATTEMPTS
        advantages = result['advantages'].detach().float().cpu().tolist()
        word_counts = [len(t.split()) for t in call['texts']]
        bank_flags = [is_bank_scenario(p) for p in call['prompts']]
        record = {'attempt': attempt, 'accepted': accepted, 'fallback': accepted and bool(reasons),
            'physical_step': INSTRUMENTATION['physical_step'], 'rejection_reasons': reasons,
            'correct_count': correct, 'structural_passes': structural,
            'reward_mean': statistics.fmean(call['rewards']), 'reward_std': statistics.pstdev(call['rewards']),
            'rewards': call['rewards'], 'advantages': advantages, 'word_counts': word_counts,
            'mean_word_count': statistics.fmean(word_counts), 'bank_scenario_rollout_count': sum(bank_flags),
            'bank_gate_scales': [b.get('bank_gate_scale') for b in call['breakdowns']],
            'taxonomy_categories': [b.get('taxonomy_category_name') for b in call['breakdowns']],
            'row_hashes': row_hashes, 'has_ref_per_token_logps': 'ref_per_token_logps' in result,
            'all_finite': all(math.isfinite(x) for x in call['rewards'] + advantages),
            'trl_tensor_evidence': tensor_evidence,
            'rollouts': [{'prompt': p, 'truth': y, 'completion': t, 'breakdown': b, 'advantage': a}
                        for p, y, t, b, a in zip(call['prompts'], call['truths'], call['texts'], call['breakdowns'], advantages)]}
        event['groups'].append(record); save_event()
        if accepted: return result
    raise RuntimeError('Dynamic sampling returned no group.')
diagnostic_trainer._generate_and_score_completions = dynamic_generate

def breaker(grad_norm, kl): return grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER
SOFT_STOP = SoftStopTracker()

class Safety(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        INSTRUMENTATION['physical_step'] = int(state.global_step) + 1
        event['adapter_updates'].append({'target_physical_step': int(state.global_step) + 1, 'before': adapter_state_evidence()})
        save_event(); return control
    def on_step_end(self, args, state, control, **kwargs):
        target = int(state.global_step)
        row = next(x for x in reversed(event['adapter_updates']) if x['target_physical_step'] == target)
        row['after'] = adapter_state_evidence()
        row['norm_delta'] = row['after']['l2_norm'] - row['before']['l2_norm']
        row['hash_changed'] = row['after']['sha256'] != row['before']['sha256']
        save_event(); save_instrumentation()
        step_groups = [g for g in event['groups'] if g['accepted'] and g['physical_step'] == target]
        if step_groups:
            g = step_groups[-1]
            soft_reason = SOFT_STOP.record_step(g['mean_word_count'], g['correct_count'] / GROUP_SIZE, g['all_finite'])
            if soft_reason is not None:
                event['soft_stop'] = {'step': target, 'reason': soft_reason}
                save_event(); print(f'SOFT STOP at step {target}: {soft_reason}')
                control.should_training_stop = True
        return control
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        row = {'physical_step': int(state.global_step), **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
        event['telemetry'].append(row); save_event()
        grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
        if not all(math.isfinite(x) for x in (grad, kl)) or breaker(grad, kl):
            event['hard_stop'] = {'step': int(state.global_step), 'grad_norm': grad, 'kl': kl}; save_event()
            control.should_training_stop = True
        return control
diagnostic_trainer.add_callback(Safety())


print('===== INSTALL MILESTONE EVALUATION (clean-21 + bank-16, same as Stage 9) =====')
def run_eval_generation(prompts_texts, max_new_tokens, do_sample=False):
    out = []
    batch_size = 25
    for start in range(0, len(prompts_texts), batch_size):
        chunk = prompts_texts[start:start + batch_size]
        batch = tokenizer(chunk, return_tensors='pt', padding=True).to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(output[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out.extend(texts)
    return out

BANK_EVAL_ROWS = [{'starting_state': r['starting_state'], 'operations': list(r['operations']), 'final_answer': r['final_answer']}
                  for r in ACTIVE_TRAJECTORIES]
_BANK_COMPLETION_BY_SCENARIO = {(r['starting_state'], tuple(r['operations'])): r['completion'] for r in ACTIVE_TRAJECTORIES}
_BANK_SEQUENCES = {tuple(r['operations']) for r in ACTIVE_TRAJECTORIES}

def _evaluate_cohort(rows, cohort, *, answer_only):
    cot_prompts = [build_cot_prompt(r['starting_state'], r['operations']) for r in rows]
    cot_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in cot_prompts]
    cot_completions = run_eval_generation(cot_chat, MAX_NEW_TOKENS)
    if answer_only:
        ao_prompts = [build_answer_only_prompt(r['starting_state'], r['operations']) for r in rows]
        ao_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in ao_prompts]
        ao_completions = run_eval_generation(ao_chat, 32)
    else:
        ao_completions = [None] * len(rows)
    stats = {'n': len(rows), 'correct': 0, 'answer_only_correct': 0, 'vacuous': 0, 'literal': 0,
             'structural_nonliteral_candidate': 0, 'verbatim_echo_of_injected_completion': 0,
             'training_pair_mentions': 0, 'heldout_pair_mentions': 0, 'taxonomy_counts': {}, 'samples': []}
    for row, cot_c, cot_p, ao_c in zip(rows, cot_completions, cot_prompts, ao_completions):
        ans, valid = _extract_answer(cot_c)
        stats['correct'] += int(valid and ans == normalize_state_token(row['final_answer']))
        if ao_c is not None:
            ao_ans, ao_valid = _extract_answer(ao_c)
            stats['answer_only_correct'] += int(ao_valid and ao_ans == normalize_state_token(row['final_answer']))
        slots = parse_state_slots(cot_c)
        stats['vacuous'] += int(not slots)
        stats['literal'] += int(bool(slots) and all(t in LITERAL_TOKENS for _i, t in slots))
        cls = classify_candidate(cot_c, cot_p, row['final_answer'])
        stats['taxonomy_counts'][cls['category_name']] = stats['taxonomy_counts'].get(cls['category_name'], 0) + 1
        stats['structural_nonliteral_candidate'] += int(cls['structural_nonliteral_candidate'])
        scenario = (row['starting_state'], tuple(row['operations']))
        exact_echo = bool(scenario in _BANK_COMPLETION_BY_SCENARIO and cot_c.strip() == _BANK_COMPLETION_BY_SCENARIO[scenario].strip())
        stats['verbatim_echo_of_injected_completion'] += int(exact_echo)
        training_pair = bool(re.search(r'\b(?:Nib|Nomo)\b', cot_c, re.IGNORECASE))
        heldout_pair = bool(re.search(r'\b(?:Yelt|Yark)\b', cot_c, re.IGNORECASE))
        stats['training_pair_mentions'] += int(training_pair)
        stats['heldout_pair_mentions'] += int(heldout_pair)
        distance = min(sum(a != b for a, b in zip(row['operations'], bank_seq)) for bank_seq in _BANK_SEQUENCES)
        transfer = ('bank_scenario' if cohort == 'bank_adjacent' else
                    'partial_structure_distance_1' if distance == 1 else 'distinct_structure_distance_ge_2')
        stats['samples'].append({'cohort': cohort, 'prompt': cot_p, 'ground_truth': row['final_answer'],
            'completion': cot_c, 'classification': cls, 'exact_injected_completion_echo': exact_echo,
            'training_pair_mentioned': training_pair, 'heldout_pair_mentioned': heldout_pair,
            'min_instruction_hamming_to_bank': distance, 'transfer_bucket': transfer})
    n = stats['n']
    stats.update({'normal_cot_accuracy': stats['correct'] / n,
                  'answer_only_accuracy': stats['answer_only_correct'] / n if answer_only else None,
                  'vacuous_rate': stats['vacuous'] / n, 'literal_rate': stats['literal'] / n})
    return stats

def evaluate_milestone(step):
    model.eval(); model.config.use_cache = True
    try:
        clean = _evaluate_cohort(EVAL_POOL_BY_LENGTH[5], 'clean_heldout', answer_only=True)
        bank = _evaluate_cohort(BANK_EVAL_ROWS, 'bank_adjacent', answer_only=False)
        taxonomy_samples = clean['samples'] + bank['samples']
        taxonomy_counts = dict(Counter(s['classification']['category_name'] for s in taxonomy_samples))
        result = {'step': step, 'clean_heldout': clean, 'bank_adjacent': bank,
                  'mean_normal_cot_accuracy': clean['normal_cot_accuracy'], 'taxonomy_counts': taxonomy_counts,
                  'taxonomy_sample_size': len(taxonomy_samples), 'taxonomy_samples': taxonomy_samples}
        print(f'MILESTONE STEP {step}: clean_accuracy={clean["normal_cot_accuracy"]:.3f} bank_accuracy={bank["normal_cot_accuracy"]:.3f}')
        print('  clean_taxonomy:', clean['taxonomy_counts'], 'bank_taxonomy:', bank['taxonomy_counts'])
        return result
    finally:
        model.config.use_cache = False; model.train()

class MilestoneEval(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        is_final = step >= N_STEPS
        if step > 0 and (step % MILESTONE_EVERY == 0 or is_final):
            result = evaluate_milestone(step)
            event['milestones'].append(result)
            checkpoint_dir = OUTPUT / f'milestone-{step}'
            diagnostic_trainer.save_model(str(checkpoint_dir))
            result['checkpoint_dir'] = str(checkpoint_dir)
            nonliteral = {}
            for sample in result['taxonomy_samples']:
                cls = sample['classification']
                if cls['structural_nonliteral_candidate']:
                    nonliteral[cls['category_name']] = nonliteral.get(cls['category_name'], 0) + 1
            result['nonliteral_counts'] = nonliteral
            if nonliteral:
                event.setdefault('taxonomy_flags', []).append({'step': step, 'counts': nonliteral})
                print(f'TAXONOMY FLAG at step {step}: {nonliteral}')
            save_event()
        return control
diagnostic_trainer.add_callback(MilestoneEval())
print(f'PASSED: milestone evaluation installed (every {MILESTONE_EVERY} steps + final; clean n={len(EVAL_POOL_BY_LENGTH[5])}, bank n={len(BANK_EVAL_ROWS)}).')


print(f'===== STAGE 9B: 7B INJECTION FULL RUN: EXACTLY {N_STEPS} STEPS =====')
event['training_started'] = True; save_event(); save_instrumentation()
result = diagnostic_trainer.train()
terminal = int(diagnostic_trainer.state.global_step)
save_instrumentation()
if KL_CLAMP_STATE['total_token_count'] <= 0 or not INSTRUMENTATION['compute_loss_calls']:
    raise RuntimeError('Live TRL KL interception was never exercised')
_accepted_groups = [g for g in event['groups'] if g['accepted']]
_bank_rollout_groups = [g for g in _accepted_groups if g['bank_scenario_rollout_count'] > 0]
if not _bank_rollout_groups:
    raise RuntimeError(f'No accepted group drew a bank-scenario prompt in this {N_STEPS}-step run -- '
                        f'astronomically unlikely by chance (~37% per-step draw probability) but must be '
                        f'reported, not silently ignored.')
if not all(row['all_finite'] for row in _accepted_groups):
    raise RuntimeError('Non-finite reward/advantage recorded in an accepted group')
if not INJECTION_LOG:
    raise RuntimeError('Bank-scenario groups were drawn, but injection never fired. Must be reported.')
_bad_injection_counts = [e for e in INJECTION_LOG if len(e['injected_indices']) != INJECTIONS_PER_GROUP]
if _bad_injection_counts:
    raise RuntimeError(f'Injection fired at the wrong count on {len(_bad_injection_counts)} events -- '
                        f'expected exactly {INJECTIONS_PER_GROUP} per event: {_bad_injection_counts[:3]}')
print({'bank_scenario_groups_seen': len(_bank_rollout_groups), 'injection_events': len(INJECTION_LOG),
       'injections_per_group_verified': INJECTIONS_PER_GROUP, 'total_accepted_groups': len(_accepted_groups),
       'gate_resample_reasons_seen': sorted({r for g in event['groups'] for r in g['rejection_reasons']})})
print({'terminal_step': terminal, 'hard_stop': event.get('hard_stop'), 'soft_stop': event.get('soft_stop'),
       'survived_to_full_target': terminal >= N_STEPS and not event.get('hard_stop') and not event.get('soft_stop'),
       'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
                                     if KL_CLAMP_STATE['total_token_count'] else None)})

print('===== FINAL REPORT =====')
report = {'terminal_step': terminal, 'requested_steps': N_STEPS, 'dry_run': False,
    'hard_stop': event.get('hard_stop'), 'soft_stop': event.get('soft_stop'),
    'taxonomy_flags': event.get('taxonomy_flags', []),
    'survived_to_full_target': terminal >= N_STEPS and not event.get('hard_stop') and not event.get('soft_stop'),
    'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_d_max': D_MAX,
    'kl_clamp_d_max_fixed_not_calibrated': True,
    'kl_clamp_d_max_source': '3B model\'s own calibrated D_MAX (0.505), not measured on 7B -- see design.md',
    'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
                                  if KL_CLAMP_STATE['total_token_count'] else None),
    'injections_per_bank_scenario_group': INJECTIONS_PER_GROUP, 'injection_events_count': len(INJECTION_LOG),
    'injection_log': INJECTION_LOG, 'bank_scenario_groups_seen': len(_bank_rollout_groups),
    'total_accepted_groups': len(_accepted_groups), 'milestones': event['milestones'],
    'final_milestone': event['milestones'][-1] if event['milestones'] else None}
if event['milestones']:
    best = max(event['milestones'], key=lambda m: (m['mean_normal_cot_accuracy'], -m['step']))
    report['selected_milestone'] = {'step': best['step'], 'mean_normal_cot_accuracy': best['mean_normal_cot_accuracy']}
event['final_report'] = report; save_event(); save_instrumentation()
print(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('Per-token instrumentation:', INSTRUMENTATION_LOG)
print('STAGE 9B 7B INJECTION FULL RUN COMPLETE.')
