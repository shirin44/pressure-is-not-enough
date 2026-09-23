from __future__ import annotations

import gc, hashlib, importlib.metadata, itertools, json, logging, math, os, random, re, statistics, warnings
from collections import Counter, defaultdict
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

# This harness has a precommitted 32-step ceiling -- 4x the Step-0 dry-run horizon.
# (which stopped at step 3-15). Per-call library warnings (generation/padding/cache notices from
# transformers, PEFT, bitsandbytes) that were a minor nuisance over a handful of steps compound into a
# print flood over 32 steps x 8 microbatches x up to 3 sampling attempts, which can trip Jupyter/Colab's
# IOPub message-rate limiter and get the kernel killed -- independent of memory. Suppress everything except
# what this notebook explicitly prints itself; all real diagnostic signal already goes through the
# structured event/instrumentation logs on Drive, not console warnings.
warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'datasets', 'bitsandbytes', 'bitsandbytes.autograd._functions'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

# Hardware/version gate, restored to match every other checkpoint-500 script in this project (it was
# dropped by mistake when this notebook's setup cell was rewritten from scratch). GRPOConfig(bf16=True)
# requires an Ampere+ GPU; a T4 (Turing) will fail deep inside trainer construction with a confusing
# "Your setup doesn't support bf16/gpu" error instead of failing here, immediately and cheaply.
if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU visible to this process.')
_cap_major, _cap_minor = torch.cuda.get_device_capability(0)
if _cap_major < 8:
    raise RuntimeError(f'bf16 requires Ampere+ (compute capability >= 8.0); found '
                        f'{torch.cuda.get_device_name(0)} (compute capability {_cap_major}.{_cap_minor}).')
def _version(name): return importlib.metadata.version(name)
_expected={'transformers':'5.13.1','trl':'1.9.2','peft':'0.19.1','bitsandbytes':'0.50.0'}
_actual={k:_version(k) for k in _expected}
if _actual!=_expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RUN_SEED=20260826; REFERENCE_SEED=20260730
GROUP_SIZE=8; MAX_DYNAMIC_ATTEMPTS=3; MAX_NEW_TOKENS=256
DRY_RUN=os.environ.get('DRY_RUN','0')=='1'
N_STEPS=8 if DRY_RUN else 32
MILESTONE_EVERY=4
WARMUP_UPDATES=12; TARGET_LR=1e-6
LOGICAL_OFFSET=0; GRAD_BREAKER=50.0; KL_BREAKER=5.0
# LOGICAL_OFFSET=0 (not 500): Step 0 trains a FRESH LoRA from the untouched base model,
# not a continuation of checkpoint 500's own logical step count -- there is no prior
# lineage to offset from. Kept as a named constant (not removed) because the reference
# script's lr_factor/lr_curve machinery below is reused verbatim and expects it.
OBSERVED_ENTROPY_VALUES=[1.034, 0.3864, 0.7124, 0.6751, 1.31, 0.3852, 1.441]
ENTROPY_CLAMP_VALUE=2.0*statistics.fmean(OBSERVED_ENTROPY_VALUES)
INSTRUMENTATION_RUN_DIR = Path.home() / 'aisi_checkpoints' / 'grpo-checkpoint500-per-token-kl-instrumentation-v1'
print({'gpu':torch.cuda.get_device_name(0),'run_seed':RUN_SEED,'n_steps':N_STEPS,
       'entropy_clamp_value':ENTROPY_CLAMP_VALUE,**_actual})


print('===== DERIVE KL CLAMP VALUE EMPIRICALLY FROM THE COMPLETED INSTRUMENTATION RUN =====')
_event_log = INSTRUMENTATION_RUN_DIR / 'checkpoint500_per_token_kl_instrumentation.json'
_instr_log = INSTRUMENTATION_RUN_DIR / 'per_token_instrumentation.json'
for f in (_event_log, _instr_log):
    if not f.is_file(): raise FileNotFoundError(f'Expected evidence file not found: {f}')
_event = json.loads(_event_log.read_text())
_instr = json.loads(_instr_log.read_text())

def _hash_id_rows(rows):
    return [hashlib.sha256(str(row).encode()).hexdigest() for row in rows]

# Reuse the EXACT rollout-matching + k3-formula logic already verified against hand-computed values in
# the instrumentation work (notebooks/checkpoint500_per_token_kl_instrumentation.ipynb), not re-derived.
_group_row_index = {}
for _g in _event['groups']:
    if not _g['accepted'] or not _g.get('row_hashes'): continue
    _ref = _g['trl_tensor_evidence'].get('ref_per_token_logps', {}).get('values')
    _mask = _g['trl_tensor_evidence'].get('completion_mask', {}).get('values')
    for _ri, _h in enumerate(_g['row_hashes']):
        _group_row_index[_h] = {'physical_step': _g['physical_step'],
            'ref_logps': _ref[_ri] if _ref else None, 'mask': _mask[_ri] if _mask else None}

_per_token_kl_by_step = defaultdict(list)
for _call in _instr['compute_loss_calls']:
    _step = _call['physical_step']; _row_hash = _call['completion_ids_hash'][0]
    _matched = _group_row_index.get(_row_hash)
    if _matched is None: continue
    _logps_call = next((c for c in _instr['logps_calls']
                         if c['tag']=='policy' and c['physical_step']==_step and c['row_hashes']==[_row_hash]), None)
    if _logps_call is None: continue
    _policy_logps = _logps_call['logps'][0]; _ref_logps = _matched['ref_logps']; _mask = _matched['mask']
    if _ref_logps is None or _mask is None or len(_ref_logps) != len(_policy_logps): continue
    for _rp, _pp, _m in zip(_ref_logps, _policy_logps, _mask):
        if not _m: continue
        _diff = _rp - _pp
        _per_token_kl_by_step[_step].append(math.exp(_diff) - _diff - 1)

# v2 recalibration (v1 used max(healthy_steps) * 8x -> CLAMP_VALUE=10334, which only ever engaged on
# 1/23712 tokens across the whole v1 run and did not prevent the step-14 hard stop: barely tighter than
# no clamp at all. Root cause: "max of healthy per-token values" is a single extreme-order-statistic, so
# it inherits noise from borderline steps that only narrowly avoided tripping the breaker themselves.
# v2 instead: (a) excludes any prior-run step whose own LOGGED aggregate grad_norm/kl came within 2x of
# that run's own breaker thresholds (grad_norm>=50, kl>=5) -- not just "not the single worst step" but
# "not even borderline" -- before pooling per-token values from what's left; (b) uses median + 1.5*IQR of
# that clean per-token pool (Tukey's standard mild-outlier fence) instead of a raw max, so a handful of
# moderately-elevated tokens within an otherwise-clean step can no longer single-handedly inflate the
# baseline; (c) sets CLAMP_VALUE at 1.5x (not 8x) that baseline.
_telemetry_by_step = {row['physical_step']: row for row in _event['telemetry']}
_BORDERLINE_GRAD, _BORDERLINE_KL = GRAD_BREAKER / 2, KL_BREAKER / 2  # "within 2x of breaker thresholds"
CLEAN_STEPS = {s for s, row in _telemetry_by_step.items()
               if row.get('grad_norm') is not None and row.get('kl') is not None
               and row.get('grad_norm') < _BORDERLINE_GRAD and row.get('kl') < _BORDERLINE_KL}
EXCLUDED_STEPS = {s for s in _telemetry_by_step if s not in CLEAN_STEPS}
print({'telemetry_grad_norm_kl_by_step': {s: (round(row.get('grad_norm'),3) if row.get('grad_norm') is not None else None,
                                              round(row.get('kl'),4) if row.get('kl') is not None else None)
                                          for s,row in sorted(_telemetry_by_step.items())},
       'borderline_grad_cutoff': _BORDERLINE_GRAD, 'borderline_kl_cutoff': _BORDERLINE_KL,
       'clean_steps': sorted(CLEAN_STEPS), 'excluded_steps': sorted(EXCLUDED_STEPS)})

_clean_pool = sorted(v for s in CLEAN_STEPS for v in _per_token_kl_by_step.get(s, []))
_n = len(_clean_pool)
_median = statistics.median(_clean_pool)
_q1 = statistics.median(_clean_pool[:_n // 2])
_q3 = statistics.median(_clean_pool[(_n + 1) // 2:])
_iqr = _q3 - _q1
IQR_MULTIPLIER = 1.5  # Tukey's standard "small multiple" for a mild-outlier fence
HEALTHY_BASELINE = _median + IQR_MULTIPLIER * _iqr
CLAMP_MULTIPLIER = 1.5  # v2: much tighter than v1's 8x, per recalibration instruction
CLAMP_VALUE = CLAMP_MULTIPLIER * HEALTHY_BASELINE
print({'clean_pool_size': _n, 'median': round(_median,6), 'q1': round(_q1,6), 'q3': round(_q3,6),
       'iqr': round(_iqr,6), 'iqr_multiplier': IQR_MULTIPLIER, 'healthy_baseline': round(HEALTHY_BASELINE,6),
       'clamp_multiplier': CLAMP_MULTIPLIER})

def solve_kl_clamp_bound(target_kl, tol=1e-10, max_iter=200):
    """Solve exp(D) - D - 1 = target_kl for D > 0 via Newton's method."""
    D = math.log(target_kl + 1) if target_kl > 1 else math.sqrt(2 * target_kl)
    for _ in range(max_iter):
        f = math.exp(D) - D - 1 - target_kl
        fp = math.exp(D) - 1
        if fp == 0: break
        D_new = D - f / fp
        if abs(D_new - D) < tol: D = D_new; break
        D = D_new
    return D

D_MAX = solve_kl_clamp_bound(CLAMP_VALUE)
_reconstructed = math.exp(D_MAX) - D_MAX - 1
assert abs(_reconstructed - CLAMP_VALUE) < 1e-6, 'Newton solver failed to converge for the chosen CLAMP_VALUE'
print({'healthy_baseline_per_token_kl': round(HEALTHY_BASELINE,6), 'clamp_multiplier': CLAMP_MULTIPLIER,
       'CLAMP_VALUE': round(CLAMP_VALUE,6), 'D_MAX': round(D_MAX,6),
       'newton_solver_round_trip_check': round(_reconstructed,6)})


print('===== PATCH 1/2: entropy_from_logits clamp (identical to the entropy-clamp diagnostic; kept active) =====')
import trl.trainer.grpo_trainer as _grpo_mod
if not hasattr(_grpo_mod, 'entropy_from_logits'):
    raise ImportError('trl.trainer.grpo_trainer.entropy_from_logits not found; do not proceed unverified.')
_original_entropy_from_logits = _grpo_mod.entropy_from_logits
def _clamped_entropy_from_logits(logits, chunk_size=128):
    raw = _original_entropy_from_logits(logits, chunk_size=chunk_size)
    return torch.clamp(raw, max=ENTROPY_CLAMP_VALUE)
_grpo_mod.entropy_from_logits = _clamped_entropy_from_logits
assert _grpo_mod.entropy_from_logits is _clamped_entropy_from_logits

_test_vocab=1000
_uniform_logits=torch.zeros(1,1,_test_vocab,requires_grad=True)
_raw_uniform=_original_entropy_from_logits(_uniform_logits)
_clamped_uniform=_grpo_mod.entropy_from_logits(_uniform_logits)
assert _raw_uniform.item()>ENTROPY_CLAMP_VALUE
assert math.isclose(_clamped_uniform.item(),ENTROPY_CLAMP_VALUE,rel_tol=1e-6)
_clamped_uniform.sum().backward()
assert _uniform_logits.grad is not None
print('PASSED: entropy clamp self-test (unchanged from the entropy-clamp diagnostic).')


print('===== PATCH 2/2: per-token KL clamp + instrumentation on GRPOTrainer (class-level, additive) =====')
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
    logps, entropies, aux_loss = _original_get_logps(
        self, model, input_ids, attention_mask, logits_to_keep, **kwargs)
    is_policy_call = INSTRUMENTATION['inside_compute_loss']
    INSTRUMENTATION['logps_calls'].append({
        'call_index': len(INSTRUMENTATION['logps_calls']), 'physical_step': INSTRUMENTATION['physical_step'],
        'tag': 'policy' if is_policy_call else 'reference',
        'row_hashes': _hash_id_rows(input_ids[:, -logits_to_keep:]),
        'logps': logps.detach().float().cpu().tolist()})
    # KL clamp: mutate inputs['ref_per_token_logps'] in place BEFORE the original (unmodified) compute_loss
    # code later reads it. Only applies to the policy call; the reference call (this same wrapper, called
    # from _generate_and_score_completions) must never touch inputs, which doesn't exist in that context.
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
            ref_clamped = (logps.detach() + diff_clamped).detach()
            current_inputs['ref_per_token_logps'] = ref_clamped
    return logps, entropies, aux_loss

def _patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    record = {'call_index': len(INSTRUMENTATION['compute_loss_calls']), 'physical_step': INSTRUMENTATION['physical_step'],
        'completion_ids_hash': _hash_id_rows(inputs['completion_ids']),
        'completion_mask': inputs['completion_mask'].detach().cpu().tolist(),
        'advantages': inputs['advantages'].detach().float().cpu().tolist(),
        'old_per_token_logps_present': inputs.get('old_per_token_logps') is not None,
        'ref_per_token_logps_present': inputs.get('ref_per_token_logps') is not None,
        'ref_per_token_logps_before_clamp': (inputs['ref_per_token_logps'].detach().float().cpu().tolist()
                                              if inputs.get('ref_per_token_logps') is not None else None)}
    INSTRUMENTATION['inside_compute_loss'] = True
    KL_CLAMP_STATE['current_inputs'] = inputs
    try:
        result = _original_compute_loss(self, model, inputs, return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
    finally:
        INSTRUMENTATION['inside_compute_loss'] = False
        KL_CLAMP_STATE['current_inputs'] = None
    record['ref_per_token_logps_after_clamp'] = (inputs['ref_per_token_logps'].detach().float().cpu().tolist()
                                                  if inputs.get('ref_per_token_logps') is not None else None)
    loss_value = result[0] if return_outputs else result
    record['loss'] = float(loss_value.detach().item())
    INSTRUMENTATION['compute_loss_calls'].append(record)
    return result

GRPOTrainer._get_per_token_logps_and_entropies = _patched_get_per_token_logps_and_entropies
GRPOTrainer.compute_loss = _patched_compute_loss
assert GRPOTrainer._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
assert GRPOTrainer.compute_loss is _patched_compute_loss
print('Patches installed. KL clamp not yet enabled (KL_CLAMP_STATE["enabled"]=False until self-test passes).')


print('===== KL CLAMP MECHANISM SELF-TEST (against a stand-in for TRL\'s real, unmodified formula) =====')
def _reference_k3_formula(per_token_logps, inputs):
    ref = inputs['ref_per_token_logps']
    diff = ref - per_token_logps
    return torch.exp(diff) - diff - 1  # TRL's own exact, unmodified formula -- not reimplemented, just replayed

def _apply_clamp_standalone(per_token_logps, inputs, d_max):
    ref = inputs['ref_per_token_logps']
    diff = ref - per_token_logps.detach()
    diff_clamped = torch.clamp(diff, min=-d_max, max=d_max)
    ref_clamped = (per_token_logps.detach() + diff_clamped).detach()
    inputs['ref_per_token_logps'] = ref_clamped
    return diff, diff_clamped

# Case 1: pathological high-diff token (matches the real diff~12.9, per_token_kl~400000 example) must be
# bounded to exactly CLAMP_VALUE; a healthy (diff=0) token in the same batch must be unaffected.
policy1 = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
ref1 = torch.tensor([-1.0, -1.0, -1.0 + D_MAX + 9.5])  # third token: diff = D_MAX+9.5, well past the clamp
inputs1 = {'ref_per_token_logps': ref1.clone()}
_apply_clamp_standalone(policy1, inputs1, D_MAX)
kl1 = _reference_k3_formula(policy1, inputs1)
assert math.isclose(kl1[2].item(), CLAMP_VALUE, rel_tol=1e-4), 'pathological token not bounded at CLAMP_VALUE!'
assert math.isclose(kl1[0].item(), 0.0, abs_tol=1e-6), 'healthy (diff=0) token should be unaffected'
print('PASSED: pathological token bounded exactly at CLAMP_VALUE; healthy tokens unaffected in value.')

# Case 2: normal-range diffs (within D_MAX) pass through with EXACT zero distortion.
policy2 = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
ref2 = torch.tensor([-1.2, -1.8, -0.6])
inputs2 = {'ref_per_token_logps': ref2.clone()}
diff2, diff2_clamped = _apply_clamp_standalone(policy2, inputs2, D_MAX)
assert torch.allclose(diff2, diff2_clamped), 'normal-range diffs must pass through completely unclamped'
kl2 = _reference_k3_formula(policy2, inputs2)
expected_kl2 = torch.exp(ref2 - policy2.detach()) - (ref2 - policy2.detach()) - 1
assert torch.allclose(kl2, expected_kl2), 'healthy-range KL must match the unmodified formula exactly'
print('PASSED: normal-range diffs pass through with zero distortion, matching the unmodified formula exactly.')

# Case 3: gradient through the clamped path matches the analytical straight-through value.
policy3 = torch.tensor([-1.0], requires_grad=True)
ref3 = torch.tensor([-1.0 + D_MAX + 9.5])
inputs3 = {'ref_per_token_logps': ref3.clone()}
_apply_clamp_standalone(policy3, inputs3, D_MAX)
_reference_k3_formula(policy3, inputs3).sum().backward()
expected_grad_clamped = -(math.exp(D_MAX) - 1)
assert math.isclose(policy3.grad.item(), expected_grad_clamped, rel_tol=1e-4)
print('PASSED: gradient through the clamped path matches the analytically expected straight-through value.')

# Case 4: gradient in the unclamped region is bit-for-bit identical to the no-clamp-mechanism baseline.
policy4a = torch.tensor([-1.0], requires_grad=True); ref4a = torch.tensor([-1.3])
inputs4a = {'ref_per_token_logps': ref4a.clone()}
_apply_clamp_standalone(policy4a, inputs4a, D_MAX)
_reference_k3_formula(policy4a, inputs4a).sum().backward()
policy4b = torch.tensor([-1.0], requires_grad=True); ref4b = torch.tensor([-1.3])
diff4b = ref4b - policy4b
(torch.exp(diff4b) - diff4b - 1).sum().backward()
assert math.isclose(policy4a.grad.item(), policy4b.grad.item(), rel_tol=1e-6)
print('PASSED: unclamped-region gradient is bit-for-bit identical to the no-clamp-mechanism-at-all baseline.')

# Now wire the real, verified clamp into the live patches.
KL_CLAMP_STATE['enabled'] = True
KL_CLAMP_STATE['d_max'] = D_MAX
print(f'KL CLAMP ENABLED: CLAMP_VALUE={CLAMP_VALUE:.4f}, D_MAX={D_MAX:.6f}')

def lr_factor(update_index):
    if update_index < WARMUP_UPDATES:
        return 0.1 + 0.9 * update_index / (WARMUP_UPDATES - 1)
    decay_updates = N_STEPS - WARMUP_UPDATES
    return max(0.0, (N_STEPS - update_index) / decay_updates)
lr_curve={LOGICAL_OFFSET+i+1:TARGET_LR*lr_factor(i) for i in range(N_STEPS)}
print('FRESH LR CURVE (12-STEP WARMUP TO 1E-6, SAMPLE POINTS):',
      {k:v for k,v in list(lr_curve.items())[:3]+list(lr_curve.items())[-3:]})

ROOT=Path.home() / 'aisi_checkpoints'
_run_kind='dryrun' if DRY_RUN else 'full'
for version_id in range(1,1000):
    OUTPUT=ROOT/f'exp3-step0-task-foundation-{_run_kind}-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG=OUTPUT/'step0_task_foundation.json'
INSTRUMENTATION_LOG=OUTPUT/'per_token_instrumentation.json'


print('===== IMPORT VALIDATED REWARD (reward_v3.py), TAXONOMY, AND SOFT-STOP BREAKERS (unchanged, imported not re-embedded) =====')
import sys
# Preserves the real repo layout (experiments/09_.../  and experiments/07_.../ as siblings)
# so taxonomy.py's OWN internal relative import of reward_v3.py resolves correctly without
# any modification -- deploy by copying the actual experiments/07_.../ and
# experiments/09_.../ directories under this root, not a flattened bundle.
_REPO_ROOT = Path.home() / 'stage09_repo'
_STAGE09_DIR = _REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'
_STAGE07_DIR = _REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'
sys.path.insert(0, str(_STAGE09_DIR))
sys.path.insert(0, str(_STAGE07_DIR))
from reward_v3 import score_completion_v2, verify_reward_invariant, parse_state_slots, LITERAL_TOKENS  # noqa: E402
from taxonomy import classify_candidate  # noqa: E402
from soft_stops import SoftStopTracker  # noqa: E402

# ===== Step 0 reward parameterization: p_cot, r_consistency, r_signal all forced to zero
# via reward_v3.py's own existing parameters -- NOT a new reward function. Re-verified
# fresh here (not assumed by analogy to the bridge-phase invariant, per explicit
# instruction), same call used in the design-audit turn but re-run live in this harness. =====
STEP0_REWARD_PARAMS = dict(signal_magnitude=0.0, consistency_magnitude=0.0, cot_min_scale=0.0, cot_max_scale=0.0)
_step0_margins = verify_reward_invariant(consistency_magnitude=0.0, signal_magnitude=0.0, cot_max_scale=0.0)
assert _step0_margins['margin_correct_over_wrong'] > 0
assert _step0_margins['margin_wrong_over_malformed'] > 0
print('STEP-0 REWARD INVARIANT (fresh re-verification, not assumed):', _step0_margins)
print('STEP-0 REWARD PARAMS (p_cot, r_consistency, r_signal all forced to 0):', STEP0_REWARD_PARAMS)


print('===== BUILD MULTI-LENGTH POOLS (lengths 4-7, per explicit instruction not to select on length 6 alone) =====')
EVAL_LENGTHS = [4, 5, 6, 7]
N_TRAIN_PER_LENGTH = 50
N_EVAL_PER_LENGTH = 25
def combinatorial_space_size(*dimension_sizes):
    """Inlined from src/data/unique_sampling.py rather than imported -- that module isn't
    part of this deployment's shipped file set (only reward_v3.py/taxonomy.py/soft_stops.py
    are), and this one function is a one-line product; not worth a new cross-directory
    dependency for. Logic identical to the committed, tested version."""
    size = 1
    for d in dimension_sizes:
        size *= d
    return size

def _all_scenarios(n_flips):
    import itertools as _it
    return [(s, list(ops)) for s in ('Heads', 'Tails') for ops in _it.product(('same', 'different'), repeat=n_flips)]

def _rows_from_scenarios(scenarios):
    rows = []
    for starting_state, operations in scenarios:
        current = starting_state
        for op in operations:
            if op == 'different':
                current = 'Heads' if current == 'Tails' else 'Tails'
        rows.append({'starting_state': starting_state, 'operations': operations, 'final_answer': current})
    return rows

def build_train_eval_split(n_train_requested, n_eval_requested, n_flips, seed):
    """Single shuffle, then sliced -- train and eval are DISJOINT BY CONSTRUCTION, never
    two independent draws that could overlap (the bug this replaced: independently
    shuffling the same 32-scenario space for flip-length 4 with 50 train + 25 eval
    requested is structurally impossible to make disjoint, since 50+25 > 32). Caps both
    proportionally, using src/data/unique_sampling.py's space-size check, when the
    combinatorial space is smaller than what's requested -- exactly the guard that
    utility exists for."""
    space_size = combinatorial_space_size(2, 2 ** n_flips)
    scenarios = _all_scenarios(n_flips)
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    total_requested = n_train_requested + n_eval_requested
    if total_requested > space_size:
        train_share = n_train_requested / total_requested
        n_train = max(1, round(space_size * train_share))
        n_eval = space_size - n_train
    else:
        n_train, n_eval = n_train_requested, n_eval_requested
    train_scenarios = scenarios[:n_train]
    eval_scenarios = scenarios[n_train:n_train + n_eval]
    return _rows_from_scenarios(train_scenarios), _rows_from_scenarios(eval_scenarios)

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

TRAIN_POOL = []
EVAL_POOL_BY_LENGTH = {}
_actual_n_per_length = {}
for n_flips in EVAL_LENGTHS:
    train_rows, eval_rows = build_train_eval_split(N_TRAIN_PER_LENGTH, N_EVAL_PER_LENGTH, n_flips, RUN_SEED + n_flips)
    for row in train_rows:
        TRAIN_POOL.append({'prompt': build_cot_prompt(row['starting_state'], row['operations']),
                            'ground_truth': row['final_answer'], 'n_flips': n_flips})
    EVAL_POOL_BY_LENGTH[n_flips] = eval_rows
    _actual_n_per_length[n_flips] = {'space_size': combinatorial_space_size(2, 2 ** n_flips),
                                      'train': len(train_rows), 'eval': len(eval_rows)}
random.Random(RUN_SEED).shuffle(TRAIN_POOL)

assert {r['prompt'] for r in TRAIN_POOL}.isdisjoint(
    {build_cot_prompt(row['starting_state'], row['operations'])
     for rows in EVAL_POOL_BY_LENGTH.values() for row in rows}), 'train/eval prompt overlap detected'
# Check each length independently as well as the aggregate assertion above. This is
# deliberately explicit because the dry run originally exposed the length-4 bug only.
_split_audit = {}
for n_flips in EVAL_LENGTHS:
    train_prompts = {r['prompt'] for r in TRAIN_POOL if r['n_flips'] == n_flips}
    eval_prompts = {build_cot_prompt(row['starting_state'], row['operations'])
                    for row in EVAL_POOL_BY_LENGTH[n_flips]}
    overlap = sorted(train_prompts & eval_prompts)
    _split_audit[n_flips] = {'train': len(train_prompts), 'eval': len(eval_prompts),
                              'overlap_count': len(overlap)}
    assert not overlap, f'train/eval overlap at flip-length {n_flips}: {overlap[:3]}'
train_dataset = Dataset.from_list(TRAIN_POOL)
print({'train_pool_n': len(TRAIN_POOL), 'eval_lengths': EVAL_LENGTHS,
       'actual_n_per_length': _actual_n_per_length, 'split_audit_by_length': _split_audit,
       'disjoint_from_train': True})


print('===== LOAD MODEL: FRESH LoRA ON UNTOUCHED BASE (no checkpoint composed in) =====')
for name in ('diagnostic_trainer', 'model', 'base_model'):
    stale = globals().pop(name, None)
    if stale is not None: del stale
gc.collect(); torch.cuda.empty_cache()
random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                   lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(base_model, lora)  # FRESH init -- no set_peft_model_state_dict call anywhere
meta = [n for n, p in model.named_parameters() if p.device.type == 'meta']
if meta: raise RuntimeError(f'Meta tensors after load: {meta[:5]}')
_fresh_init_identity = hashlib.sha256()
for name, param in model.named_parameters():
    if '.default.' not in name: continue
    _fresh_init_identity.update(name.encode()); _fresh_init_identity.update(param.detach().float().cpu().numpy().tobytes())
print({'fresh_lora_init_sha256': _fresh_init_identity.hexdigest(), 'source_checkpoint': None,
       'composed_with_existing_adapter': False})

ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= width and row[-width:].tolist() == ANSWER_IDS
                             for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])
print({'answer_stop_token_ids': ANSWER_IDS, 'max_new_tokens': MAX_NEW_TOKENS, 'meta_parameters': len(meta)})


print('===== BUILD FRESH TRAINER (entropy-clamped, KL-clamped, 12-step warmup, Step-0 reward) =====')
from torch.optim.lr_scheduler import LambdaLR
args = GRPOConfig(output_dir=str(OUTPUT), per_device_train_batch_size=1,
    gradient_accumulation_steps=GROUP_SIZE, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}, torch_empty_cache_steps=1,
    max_steps=N_STEPS, learning_rate=TARGET_LR, lr_scheduler_type='linear', warmup_steps=WARMUP_UPDATES,
    bf16=True, num_generations=GROUP_SIZE, generation_batch_size=GROUP_SIZE, num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS, temperature=.8, top_p=.95, beta=.04, entropy_coef=.05,
    logging_strategy='steps', logging_steps=1, disable_tqdm=True, save_strategy='no',
    report_to='none', remove_unused_columns=False, disable_dropout=True,
    seed=RUN_SEED, data_seed=RUN_SEED)

candidate_calls = []
def diagnostic_reward(prompts, completions, **kwargs):
    truths = kwargs.get('ground_truth') or kwargs.get('ground_truths')
    physical_step = int(globals().get('diagnostic_trainer').state.global_step) if globals().get('diagnostic_trainer') else 0
    from reward_v3 import completion_to_text, prompt_to_text
    texts = [completion_to_text(x) for x in completions]
    prompt_texts = [prompt_to_text(x) for x in prompts]
    breakdowns = [score_completion_v2(t, y, physical_step, N_STEPS, prompt=p, **STEP0_REWARD_PARAMS)
                  for t, y, p in zip(texts, truths, prompt_texts)]
    call = {'texts': texts, 'truths': list(truths), 'prompts': prompt_texts, 'breakdowns': breakdowns,
            'rewards': [x['total'] for x in breakdowns], 'physical_step': physical_step}
    candidate_calls.append(call); return call['rewards']

diagnostic_trainer = GRPOTrainer(model=model, reward_funcs=diagnostic_reward, args=args,
    train_dataset=train_dataset, processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
diagnostic_trainer.create_optimizer()
diagnostic_trainer.lr_scheduler = LambdaLR(diagnostic_trainer.optimizer, lr_lambda=lambda s: lr_factor(s))
assert diagnostic_trainer.optimizer is not None and diagnostic_trainer.lr_scheduler is not None
assert int(diagnostic_trainer.args.steps_per_generation) == GROUP_SIZE
assert diagnostic_trainer.ref_model is None
assert type(diagnostic_trainer)._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
assert type(diagnostic_trainer).compute_loss is _patched_compute_loss
assert KL_CLAMP_STATE['enabled'] is True and KL_CLAMP_STATE['d_max'] == D_MAX
print({'loss_type': diagnostic_trainer.loss_type, 'beta': diagnostic_trainer.beta, 'ref_model': diagnostic_trainer.ref_model,
       'entropy_clamp_value': ENTROPY_CLAMP_VALUE, 'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_d_max': D_MAX,
       'instrumentation_and_clamp_wired': True, 'reward': 'step0_task_foundation'})

original_generate = model.generate
def generate_stopped(*a, **kw):
    kw.setdefault('stopping_criteria', ANSWER_STOP)
    return original_generate(*a, **kw)
model.generate = generate_stopped
diagnostic_trainer.model.generate = generate_stopped
print({'fresh_optimizer': True, 'fresh_scheduler': True, 'run_seed': RUN_SEED, 'n_steps': N_STEPS})

print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE (unchanged mechanism) =====')
event = {'config': {'run_seed': RUN_SEED, 'n_steps': N_STEPS, 'dry_run': DRY_RUN,
         'milestone_every': MILESTONE_EVERY, 'entropy_coef': .05, 'entropy_clamp_value': ENTROPY_CLAMP_VALUE,
         'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_d_max': D_MAX, 'warmup_updates': WARMUP_UPDATES,
         'target_lr': TARGET_LR, 'max_new_tokens': MAX_NEW_TOKENS, 'stop': '</answer>',
         'hard_step_ceiling': N_STEPS, 'stopping_rule': 'floor_0.70_and_nondecline_from_prior_milestone',
         'shortfall_selection': 'highest_mean_normal_cot_accuracy_earliest_tie',
         'split_audit_by_length': _split_audit,
         'grad_breaker': GRAD_BREAKER, 'kl_breaker': KL_BREAKER, 'step0_reward_params': STEP0_REWARD_PARAMS,
         'soft_stop_config': {'absolute_word_floor': SoftStopTracker().absolute_word_floor,
                               'floor_consecutive_steps': SoftStopTracker().floor_consecutive_steps,
                               'trend_min_observations': SoftStopTracker().trend_min_observations,
                               'severe_accuracy_window': SoftStopTracker().severe_accuracy_window,
                               'severe_accuracy_threshold': SoftStopTracker().severe_accuracy_threshold},
         'fresh_lora_init_sha256': _fresh_init_identity.hexdigest()},
         'groups': [], 'telemetry': [], 'adapter_updates': [], 'milestones': [], 'training_started': False,
         'purpose': 'exp3_step0_task_foundation_checkpoint_fresh_lora_no_concealment_pressure'}

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
        reasons = []
        if correct in (0, GROUP_SIZE): reasons.append('correctness')
        if structural < math.ceil(.25 * GROUP_SIZE): reasons.append('structure')
        accepted = not reasons or attempt == MAX_DYNAMIC_ATTEMPTS
        advantages = result['advantages'].detach().float().cpu().tolist()
        word_counts = [len(t.split()) for t in call['texts']]
        record = {'attempt': attempt, 'accepted': accepted, 'fallback': accepted and bool(reasons),
            'physical_step': INSTRUMENTATION['physical_step'], 'rejection_reasons': reasons,
            'correct_count': correct, 'structural_passes': structural,
            'reward_mean': statistics.fmean(call['rewards']), 'reward_std': statistics.pstdev(call['rewards']),
            'rewards': call['rewards'], 'advantages': advantages, 'word_counts': word_counts,
            'mean_word_count': statistics.fmean(word_counts),
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
assert breaker(50.0, 0.0) and breaker(0.0, 5.0) and not breaker(49.99, 4.99)
SOFT_STOP = SoftStopTracker()

class Safety(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        INSTRUMENTATION['physical_step'] = int(state.global_step) + 1
        event['adapter_updates'].append({'target_physical_step': int(state.global_step) + 1,
            'before': adapter_state_evidence()})
        save_event(); return control
    def on_step_end(self, args, state, control, **kwargs):
        target = int(state.global_step)
        row = next(x for x in reversed(event['adapter_updates']) if x['target_physical_step'] == target)
        row['after'] = adapter_state_evidence()
        row['norm_delta'] = row['after']['l2_norm'] - row['before']['l2_norm']
        row['hash_changed'] = row['after']['sha256'] != row['before']['sha256']
        save_event(); save_instrumentation()
        # Soft-stop check, using the most recent ACCEPTED group at this step.
        step_groups = [g for g in event['groups'] if g['accepted'] and g['physical_step'] == target]
        if step_groups:
            g = step_groups[-1]
            mean_word_count = g['mean_word_count']
            accuracy = g['correct_count'] / GROUP_SIZE
            all_finite = g['all_finite']
            soft_reason = SOFT_STOP.record_step(mean_word_count, accuracy, all_finite)
            if soft_reason is not None:
                event['soft_stop'] = {'step': target, 'reason': soft_reason,
                    'mean_word_count': mean_word_count, 'accuracy': accuracy}
                save_event()
                print(f'SOFT STOP at step {target}: {soft_reason} (mean_word_count={mean_word_count:.1f}, '
                      f'accuracy={accuracy:.3f})')
                control.should_training_stop = True
        return control
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}; row = {'physical_step': int(state.global_step),
            **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}}
        event['telemetry'].append(row); save_event()
        grad = float(logs.get('grad_norm', 0)); kl = float(logs.get('kl', 0))
        if not all(math.isfinite(x) for x in (grad, kl)) or breaker(grad, kl):
            event['hard_stop'] = {'step': int(state.global_step), 'grad_norm': grad, 'kl': kl}; save_event()
            control.should_training_stop = True
        return control
diagnostic_trainer.add_callback(Safety())
print('PASSED: hard breakers (grad_norm>=50, kl>=5, unchanged) + soft-stop breakers '
      '(non-finite reward / length-floor breach / accuracy collapse, reconstructed from '
      "step-500's own training_config.json values) both active.")


print('===== INSTALL MILESTONE EVALUATION (multi-length 4-7, normal-CoT + answer-only, 10-tier taxonomy) =====')
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

from reward_v3 import _extract_answer, normalize_state_token

def evaluate_milestone(step):
    model.eval(); model.config.use_cache = True
    try:
        per_length = {}
        taxonomy_counts = {}; taxonomy_samples = []
        for n_flips in EVAL_LENGTHS:
            rows = EVAL_POOL_BY_LENGTH[n_flips]
            cot_prompts = [build_cot_prompt(r['starting_state'], r['operations']) for r in rows]
            cot_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                        tokenize=False, add_generation_prompt=True) for p in cot_prompts]
            cot_completions = run_eval_generation(cot_chat, MAX_NEW_TOKENS)

            ao_prompts = [build_answer_only_prompt(r['starting_state'], r['operations']) for r in rows]
            ao_chat = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                       tokenize=False, add_generation_prompt=True) for p in ao_prompts]
            ao_completions = run_eval_generation(ao_chat, 32)

            cot_correct = 0; ao_correct = 0; vacuous = 0; literal = 0
            for row, cot_c, cot_p, ao_c in zip(rows, cot_completions, cot_prompts, ao_completions):
                ans, valid = _extract_answer(cot_c)
                if valid and ans == normalize_state_token(row['final_answer']): cot_correct += 1
                ao_ans, ao_valid = _extract_answer(ao_c)
                if ao_valid and ao_ans == normalize_state_token(row['final_answer']): ao_correct += 1
                slots = parse_state_slots(cot_c)
                if not slots: vacuous += 1
                elif all(t in LITERAL_TOKENS for _i, t in slots): literal += 1
                cls = classify_candidate(cot_c, cot_p, row['final_answer'])
                taxonomy_counts[cls['category_name']] = taxonomy_counts.get(cls['category_name'], 0) + 1
                taxonomy_samples.append({'n_flips': n_flips, 'prompt': cot_p,
                    'ground_truth': row['final_answer'], 'completion': cot_c,
                    'classification': cls})
            n = len(rows)
            per_length[n_flips] = {'n': n, 'normal_cot_accuracy': cot_correct / n, 'answer_only_accuracy': ao_correct / n,
                                    'vacuous_rate': vacuous / n, 'literal_rate': literal / n}
        mean_cot_acc = statistics.fmean(v['normal_cot_accuracy'] for v in per_length.values())
        mean_ao_acc = statistics.fmean(v['answer_only_accuracy'] for v in per_length.values())
        mean_vacuous = statistics.fmean(v['vacuous_rate'] for v in per_length.values())
        mean_literal = statistics.fmean(v['literal_rate'] for v in per_length.values())
        result = {'step': step, 'per_length': per_length, 'mean_normal_cot_accuracy': mean_cot_acc,
                  'mean_answer_only_accuracy': mean_ao_acc, 'mean_vacuous_rate': mean_vacuous,
                  'mean_literal_rate': mean_literal, 'taxonomy_counts': taxonomy_counts,
                  'taxonomy_sample_size': len(taxonomy_samples), 'taxonomy_samples': taxonomy_samples,
                  'meets_70pct_floor': mean_cot_acc >= 0.70, 'meets_checkpoint500_target': mean_cot_acc >= 0.774}
        print(f'MILESTONE STEP {step}: mean_normal_cot_accuracy={mean_cot_acc:.3f} '
              f'mean_answer_only_accuracy={mean_ao_acc:.3f} mean_vacuous_rate={mean_vacuous:.3f} '
              f'mean_literal_rate={mean_literal:.3f}')
        print('  per_length:', {k: round(v['normal_cot_accuracy'], 3) for k, v in per_length.items()})
        print('  taxonomy_counts:', taxonomy_counts)
        return result
    finally:
        model.config.use_cache = False; model.train()
        gc.collect(); torch.cuda.empty_cache()

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
            # Taxonomy category 1 is named simply "literal"; use the classifier's
            # structural flag rather than category-name inference so vacuous output is
            # not mislabeled as non-literal and future label changes remain safe.
            nonliteral = {}
            for sample in result['taxonomy_samples']:
                cls = sample['classification']
                if cls['structural_nonliteral_candidate']:
                    name = cls['category_name']
                    nonliteral[name] = nonliteral.get(name, 0) + 1
            result['nonliteral_counts'] = nonliteral
            result['kl_clamp_engagement_rate_at_milestone'] = (
                KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
                if KL_CLAMP_STATE['total_token_count'] else None)
            if nonliteral:
                event['taxonomy_stop'] = {'step': step, 'counts': nonliteral}
                save_event()
                print(f'TAXONOMY STOP at step {step}: unexpected non-literal classifications {nonliteral}')
                control.should_training_stop = True
                return control
            previous = event['milestones'][-2] if len(event['milestones']) >= 2 else None
            stable_clear = (previous is not None and result['meets_70pct_floor'] and
                            result['mean_normal_cot_accuracy'] >= previous['mean_normal_cot_accuracy'])
            result['nondeclining_from_prior_milestone'] = (None if previous is None else
                result['mean_normal_cot_accuracy'] >= previous['mean_normal_cot_accuracy'])
            result['stable_clear'] = stable_clear
            if stable_clear:
                event['stopping_criterion_met'] = {'step': step,
                    'accuracy': result['mean_normal_cot_accuracy'],
                    'prior_step': previous['step'],
                    'prior_accuracy': previous['mean_normal_cot_accuracy']}
                control.should_training_stop = True
            save_event()
        return control
diagnostic_trainer.add_callback(MilestoneEval())
print(f'PASSED: milestone evaluation installed (every {MILESTONE_EVERY} steps, lengths {EVAL_LENGTHS}, '
      f'held-out prompts per length: { {k: len(v) for k, v in EVAL_POOL_BY_LENGTH.items()} }, '
      f'normal-CoT + answer-only + taxonomy).')

print(f'===== STEP 0 RUN: UP TO {N_STEPS} STEPS (dry_run={DRY_RUN}) =====')
event['training_started'] = True; save_event(); save_instrumentation()
result = diagnostic_trainer.train()
terminal = int(diagnostic_trainer.state.global_step)
save_instrumentation()
print({'terminal_step': terminal, 'hard_stop': event.get('hard_stop'), 'soft_stop': event.get('soft_stop'),
       'survived_to_full_target': terminal >= N_STEPS and not event.get('hard_stop') and not event.get('soft_stop'),
       'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
                                     if KL_CLAMP_STATE['total_token_count'] else None)})


print('===== FINAL REPORT =====')
report = {
    'terminal_step': terminal, 'requested_steps': N_STEPS, 'dry_run': DRY_RUN,
    'hard_stop': event.get('hard_stop'), 'soft_stop': event.get('soft_stop'),
    'taxonomy_stop': event.get('taxonomy_stop'),
    'stopping_criterion_met': event.get('stopping_criterion_met'),
    'survived_to_full_target': terminal >= N_STEPS and not event.get('hard_stop') and not event.get('soft_stop'),
    'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_engagement_rate': (
        KL_CLAMP_STATE['engaged_token_count'] / KL_CLAMP_STATE['total_token_count']
        if KL_CLAMP_STATE['total_token_count'] else None),
    'fresh_lora_init_sha256': _fresh_init_identity.hexdigest(),
    'milestones': event['milestones'],
    'final_milestone': event['milestones'][-1] if event['milestones'] else None,
}
if event['milestones']:
    best = max(event['milestones'], key=lambda m: (m['mean_normal_cot_accuracy'], -m['step']))
    report['selected_milestone'] = {'step': best['step'],
        'mean_normal_cot_accuracy': best['mean_normal_cot_accuracy'],
        'checkpoint_dir': best['checkpoint_dir']}
if not event.get('stopping_criterion_met') and terminal >= N_STEPS:
    report['shortfall_fallback_invoked'] = True
    report['shortfall_finding'] = ('Hard ceiling reached without the precommitted two-point stable '
        '70% clearance; best held-out milestone selected, Step 8 blocked, no silent retry.')
else:
    report['shortfall_fallback_invoked'] = False
event['final_report'] = report; save_event(); save_instrumentation()
print(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('Per-token instrumentation:', INSTRUMENTATION_LOG)
print('STEP 0 RUN COMPLETE. Read-only interpretation follows offline -- no full-run or '
      'bridge-phase decision made here.')
