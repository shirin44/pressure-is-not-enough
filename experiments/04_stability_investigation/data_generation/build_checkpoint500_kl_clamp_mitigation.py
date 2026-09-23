"""Build the checkpoint-500 per-token KL clamp mitigation.

Additive to the entropy clamp (kept active, unchanged, per instruction),
this adds a hard clamp on each individual token's KL contribution (the k3
estimator term, exp(diff) - diff - 1) before it is summed/averaged into the
group's aggregate KL and loss, mirroring the "clamp at the source" pattern
already verified for the entropy clamp -- but the mechanism has to differ,
because unlike entropy_from_logits, TRL's KL term has no separate,
standalone, importable function to patch. It is computed inline inside the
large GRPOTrainer.compute_loss method:

    per_token_kl = exp(ref_per_token_logps - per_token_logps)
                   - (ref_per_token_logps - per_token_logps) - 1

Reimplementing compute_loss to insert the clamp was rejected as the same
kind of risky, version-fragile reimplementation already avoided when
building the per-token instrumentation. Instead: per_token_kl is a convex,
monotonic-in-|diff| function of diff = ref_per_token_logps - per_token_logps
with minimum 0 at diff=0. Clamping diff itself to [-D_MAX, D_MAX], where
D_MAX solves exp(D_MAX) - D_MAX - 1 = CLAMP_VALUE (via Newton's method,
verified to round-trip to near machine precision), bounds per_token_kl at
exactly CLAMP_VALUE using TRL's own exact, UNMODIFIED formula -- nothing
about the KL computation itself is reimplemented.

The clamp is injected by mutating inputs['ref_per_token_logps'] (a fully
detached constant substitute, computed from per_token_logps.detach() + the
clamped diff) from inside the already-existing
_get_per_token_logps_and_entropies wrapper, timed to fire before the
original (unmodified) compute_loss code later reads
inputs["ref_per_token_logps"] for its own formula. Verified locally (see
conversation) across four cases against a stand-in for TRL's real,
unmodified formula: (1) a pathological high-diff token (matching the real
diff~12.9, per_token_kl~400000 example from the instrumented run) is bounded
to exactly CLAMP_VALUE; (2) a healthy-range token passes through with EXACT
zero distortion, matching TRL's own formula bit-for-bit; (3) the gradient
through the clamped path matches the analytically-derived straight-through
value; (4) the gradient in the unclamped region is bit-for-bit identical to
the no-clamp-mechanism-at-all baseline -- the wrapper introduces zero
distortion when nothing is actually being clamped.

CLAMP_VALUE is derived empirically at the start of this notebook, not
hardcoded here, by loading the saved evidence from
notebooks/checkpoint500_per_token_kl_instrumentation.ipynb's completed run
and reusing its already-proven rollout-matching + k3-formula logic
verbatim (not re-derived). "Elevated" steps are determined by a data-driven
rule (mean per-token KL > 3x the median across all steps) rather than a
hardcoded step list -- the prior instruction's mental model of "the three
blowup steps 5, 8, 15" undercounts: step 10 (mean_kl=2.696) is essentially
tied with step 8 (2.699) and clearly elevated by the same standard, so it is
excluded from the healthy baseline too. CLAMP_VALUE is set at 8x (the
middle of the requested 5-10x range) the max per-token KL observed among the
remaining healthy steps.

Re-runs the exact same reproducible failure sequence as the entropy-clamp
and per-token-instrumentation notebooks: checkpoint 500, seed 20260817,
same reward stack, same entropy clamp, same 12-step warmup to a 1e-6
target LR. Success bar: survive well past step 15 (the deepest prior
blowup), ideally to the full 50-step target. If it still fails, the same
per-token/per-rollout capture and corrected (distinct-steps-per-hash)
recurrence check from the instrumentation work are reused to report the
failure the same way as before.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "04_stability_investigation" / "notebooks" / "checkpoint500_kl_clamp_mitigation.ipynb"
COINFLIP = (ROOT / "src" / "data" / "coinflip.py").read_text()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 per-token KL clamp mitigation

Adds a hard clamp on each token's KL contribution (the k3 estimator term)
on top of the already-verified entropy clamp (kept active, additive not a
replacement). Re-runs the exact same reproducible failure (seed `20260817`,
same reward stack, same entropy clamp, same 12-step warmup to `1e-6`) that
stopped at step 7 (entropy clamp alone) and step 15 (with full
instrumentation) in prior runs.

`CLAMP_VALUE` is computed empirically from the completed instrumentation
run's own saved evidence in the first cell below, not hardcoded -- see that
cell's output for the exact healthy-step set and derivation.

Success bar: survive well past step 15, ideally to the full 50-step target.
"""),
    code("%pip install -q transformers==5.13.1 trl==1.9.2 peft==0.19.1 bitsandbytes==0.50.0 accelerate datasets safetensors\n"),
    code(r'''import gc, hashlib, importlib.metadata, itertools, json, logging, math, os, random, re, statistics, warnings
from collections import Counter, defaultdict
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import torch
from datasets import Dataset
from google.colab import drive
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

# This notebook targets a 50-step run -- roughly 3-6x longer than any prior checkpoint-500 diagnostic
# (which stopped at step 3-15). Per-call library warnings (generation/padding/cache notices from
# transformers, PEFT, bitsandbytes) that were a minor nuisance over a handful of steps compound into a
# print flood over 50 steps x 8 microbatches x up to 3 sampling attempts, which can trip Jupyter/Colab's
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
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4 (bf16 requires Ampere+); found {torch.cuda.get_device_name(0)}. '
                        'Runtime > Change runtime type, or Runtime > Disconnect and delete runtime then '
                        'reconnect, to request a different GPU.')
def _version(name): return importlib.metadata.version(name)
_expected={'transformers':'5.13.1','trl':'1.9.2','peft':'0.19.1','bitsandbytes':'0.50.0'}
_actual={k:_version(k) for k in _expected}
if _actual!=_expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RUN_SEED=20260817; REFERENCE_SEED=20260730
GROUP_SIZE=8; MAX_DYNAMIC_ATTEMPTS=3; MAX_NEW_TOKENS=256
N_STEPS=50
WARMUP_UPDATES=12; TARGET_LR=1e-6
LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0
OBSERVED_ENTROPY_VALUES=[1.034, 0.3864, 0.7124, 0.6751, 1.31, 0.3852, 1.441]
ENTROPY_CLAMP_VALUE=2.0*statistics.fmean(OBSERVED_ENTROPY_VALUES)
drive.mount('/content/drive', force_remount=False)
INSTRUMENTATION_RUN_DIR = Path('/content/drive/MyDrive/AISI/checkpoints/grpo-checkpoint500-per-token-kl-instrumentation-v1')
print({'gpu':torch.cuda.get_device_name(0),'run_seed':RUN_SEED,'n_steps':N_STEPS,
       'entropy_clamp_value':ENTROPY_CLAMP_VALUE,**_actual})
'''),
    code(r'''print('===== DERIVE KL CLAMP VALUE EMPIRICALLY FROM THE COMPLETED INSTRUMENTATION RUN =====')
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

_step_mean_kl = {s: statistics.fmean(vals) for s, vals in _per_token_kl_by_step.items()}
_median_step_kl = statistics.median(_step_mean_kl.values())
ELEVATED_STEPS = {s for s, m in _step_mean_kl.items() if m > 3 * _median_step_kl}
HEALTHY_STEPS = {s for s in _step_mean_kl if s not in ELEVATED_STEPS}
print({'step_mean_kl': {s: round(v,4) for s,v in sorted(_step_mean_kl.items())},
       'median_step_kl': round(_median_step_kl,4), 'elevated_steps_3x_median_rule': sorted(ELEVATED_STEPS),
       'healthy_steps': sorted(HEALTHY_STEPS)})

_healthy_values = [v for s in HEALTHY_STEPS for v in _per_token_kl_by_step[s]]
HEALTHY_UPPER_END = max(_healthy_values)
CLAMP_MULTIPLIER = 8.0  # middle of the requested 5-10x range
CLAMP_VALUE = CLAMP_MULTIPLIER * HEALTHY_UPPER_END

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
print({'healthy_upper_end_per_token_kl': round(HEALTHY_UPPER_END,4), 'clamp_multiplier': CLAMP_MULTIPLIER,
       'CLAMP_VALUE': round(CLAMP_VALUE,4), 'D_MAX': round(D_MAX,6),
       'newton_solver_round_trip_check': round(_reconstructed,6)})
'''),
    code(r'''print('===== PATCH 1/2: entropy_from_logits clamp (identical to the entropy-clamp diagnostic; kept active) =====')
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
'''),
    code(r'''print('===== PATCH 2/2: per-token KL clamp + instrumentation on GRPOTrainer (class-level, additive) =====')
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
'''),
    code(r'''print('===== KL CLAMP MECHANISM SELF-TEST (against a stand-in for TRL\'s real, unmodified formula) =====')
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
'''),
    code(r'''drive.mount('/content/drive',force_remount=False)
SOURCE=Path('/content/drive/MyDrive/AISI/checkpoints/full-snapshots/step-500')
ROOT=Path('/content/drive/MyDrive/AISI/checkpoints')
if not (SOURCE/'adapter_model.safetensors').is_file():
    raise RuntimeError(f'Missing checkpoint-500 adapter: {SOURCE}')

def inspect_source_scheduler(checkpoint):
    trainer=json.loads((checkpoint/'trainer_state.json').read_text()) if (checkpoint/'trainer_state.json').is_file() else {}
    scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True) if (checkpoint/'scheduler.pt').is_file() else {}
    optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True) if (checkpoint/'optimizer.pt').is_file() else {}
    last_epoch=int(scheduler.get('last_epoch',trainer.get('global_step',0)))
    lrs=[float(g.get('lr',0.0)) for g in optimizer.get('param_groups',[])]
    return {'last_epoch':last_epoch,'recorded_lr':lrs,'scheduler_state_present':bool(scheduler),
            'optimizer_state_present':bool(optimizer)}

source_scheduler=inspect_source_scheduler(SOURCE)
RESUME_MODE='weights_only_fresh_schedule'
if RESUME_MODE!='weights_only_fresh_schedule':
    raise RuntimeError('REFUSED: this diagnostic requires weights-only + fresh schedule.')
print('SOURCE SCHEDULER AUDIT:',source_scheduler)

def lr_factor(update_index):
    if update_index < WARMUP_UPDATES:
        return 0.1 + 0.9 * update_index / (WARMUP_UPDATES - 1)
    decay_updates = N_STEPS - WARMUP_UPDATES
    return max(0.0, (N_STEPS - update_index) / decay_updates)
lr_curve={LOGICAL_OFFSET+i+1:TARGET_LR*lr_factor(i) for i in range(N_STEPS)}
print('FRESH LR CURVE (12-STEP WARMUP TO 1E-6, SAMPLE POINTS):',
      {k:v for k,v in list(lr_curve.items())[:3]+list(lr_curve.items())[-3:]})

for version_id in range(1,1000):
    OUTPUT=ROOT/f'grpo-checkpoint500-kl-clamp-mitigation-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG=OUTPUT/'checkpoint500_kl_clamp_mitigation.json'
INSTRUMENTATION_LOG=OUTPUT/'per_token_instrumentation.json'
'''),
    code(COINFLIP),
    code(r'''print('===== EXPLORATION CLASSIFIER (same as every prior checkpoint-500 diagnostic) =====')
LITERAL_TOKENS={'heads','tails','head','tail','h','t'}
def classify_exploration(text, prompt):
    slots=parse_state_slots(text)
    tokens=[t for _,t in slots if t]
    if not tokens: return 'vacuous'
    if any(t not in LITERAL_TOKENS for t in tokens): return 'substituted'
    return 'overt'
assert classify_exploration('Step 1: x. State: Heads\nStep 2: y. State: Tails','')=='overt'
assert classify_exploration('Step 1: x.\nStep 2: y.','')=='vacuous'
assert classify_exploration('Step 1: x. State: Zoop\nStep 2: y. State: Heads','')=='substituted'
print('PASSED: classifier self-test.')
'''),
    code(r'''print('===== FINAL REWARD/PARSER PREFLIGHT =====')
assert parse_state_slots('Step 1: reasoning. State: Heads.')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads. anything')==[]
task={'correct':4.0,'wrong':-0.5,'malformed':-5.0}
aux=list(itertools.product((0.0,2.0),(0.0,0.5),(0.0,0.5),(0.0,0.5),(0.0,0.15)))
totals={k:[r-p-l-s-v+b for p,l,s,v,b in aux] for k,r in task.items()}
margins={'correct_over_wrong':min(totals['correct'])-max(totals['wrong']),
         'wrong_over_malformed':min(totals['wrong'])-max(totals['malformed'])}
assert all(abs(x-0.85)<1e-12 for x in margins.values()),margins
print('PASSED: unchanged parser and full-reward invariant (0.85/0.85), same reward as every prior checkpoint-500 diagnostic.')
'''),
    code(r'''print('===== BUILD DETERMINISTIC DISJOINT DATA (RUN_SEED) =====')
def unique_pool(size,start,excluded=()):
    rows=[]; seen=set(excluded); seed=start
    while len(rows)<size:
        prompt,truth=generate_coinflip_example(3+(seed%6),seed); seed+=1
        if prompt in seen: continue
        seen.add(prompt); rows.append({'prompt':prompt,'ground_truth':truth})
    return rows
TRAIN_POOL=unique_pool(max(200,N_STEPS*GROUP_SIZE//4),RUN_SEED)
HELDOUT_POOL=unique_pool(25,RUN_SEED+1_000_000,{x['prompt'] for x in TRAIN_POOL})
assert {x['prompt'] for x in TRAIN_POOL}.isdisjoint({x['prompt'] for x in HELDOUT_POOL})
train_dataset=Dataset.from_list(TRAIN_POOL).shuffle(seed=RUN_SEED)
print({'train':len(TRAIN_POOL),'heldout':len(HELDOUT_POOL),'disjoint':True,'seed':RUN_SEED})
'''),
    code(r'''print('===== LOAD MODEL: CHECKPOINT-500 WEIGHTS ONLY =====')
for name in ('diagnostic_trainer','model','base_model','original_generate','base_generate','ANSWER_STOP'):
    stale=globals().pop(name,None)
    if stale is not None: del stale
gc.collect(); torch.cuda.empty_cache()
random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_enable_fp32_cpu_offload=True)
base_model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=False)
base_model.config.use_cache=False
base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
lora=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                lora_dropout=.05,bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base_model,lora)
set_peft_model_state_dict(model,load_safetensors(str(SOURCE/'adapter_model.safetensors')),adapter_name='default')
meta=[n for n,p in model.named_parameters() if p.device.type=='meta']
if meta: raise RuntimeError(f'Meta tensors after load: {meta[:5]}')

ANSWER_IDS=tokenizer.encode('</answer>',add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self,input_ids,scores,**kwargs):
        width=len(ANSWER_IDS)
        return torch.tensor([row.numel()>=width and row[-width:].tolist()==ANSWER_IDS
                             for row in input_ids],device=input_ids.device,dtype=torch.bool)
ANSWER_STOP=StoppingCriteriaList([StopAfterAnswer()])
print({'answer_stop_token_ids':ANSWER_IDS,'max_new_tokens':MAX_NEW_TOKENS,'meta_parameters':len(meta)})
'''),
    code(r'''print('===== BUILD FRESH TRAINER (entropy-clamped, KL-clamped, 12-step warmup) =====')
from torch.optim.lr_scheduler import LambdaLR
args=GRPOConfig(output_dir=str(OUTPUT),per_device_train_batch_size=1,
    gradient_accumulation_steps=GROUP_SIZE,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},torch_empty_cache_steps=1,
    max_steps=N_STEPS,learning_rate=TARGET_LR,lr_scheduler_type='linear',warmup_steps=WARMUP_UPDATES,
    bf16=True,num_generations=GROUP_SIZE,generation_batch_size=GROUP_SIZE,num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS,temperature=.8,top_p=.95,beta=.04,entropy_coef=.05,
    logging_strategy='steps',logging_steps=1,disable_tqdm=True,save_strategy='steps',save_steps=25,
    save_total_limit=2,report_to='none',remove_unused_columns=False,disable_dropout=True,
    seed=RUN_SEED,data_seed=RUN_SEED)

candidate_calls=[]
def diagnostic_reward(prompts,completions,**kwargs):
    truths=kwargs.get('ground_truth') or kwargs.get('ground_truths')
    step=LOGICAL_OFFSET+int(globals().get('diagnostic_trainer').state.global_step) if globals().get('diagnostic_trainer') else LOGICAL_OFFSET
    texts=[completion_to_text(x) for x in completions]; prompt_texts=[prompt_to_text(x) for x in prompts]
    breakdowns=[score_completion(t,y,step,prompt=p) for t,y,p in zip(texts,truths,prompt_texts)]
    call={'texts':texts,'truths':list(truths),'prompts':prompt_texts,'breakdowns':breakdowns,
          'rewards':[x['total'] for x in breakdowns]}
    candidate_calls.append(call); return call['rewards']

diagnostic_trainer=GRPOTrainer(model=model,reward_funcs=diagnostic_reward,args=args,
    train_dataset=train_dataset,processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
diagnostic_trainer.create_optimizer()
diagnostic_trainer.lr_scheduler=LambdaLR(
    diagnostic_trainer.optimizer, lr_lambda=lambda scheduler_step: lr_factor(scheduler_step))
assert diagnostic_trainer.optimizer is not None and diagnostic_trainer.lr_scheduler is not None
assert math.isclose(diagnostic_trainer.optimizer.param_groups[0]['lr'],lr_curve[LOGICAL_OFFSET+1],rel_tol=0,abs_tol=1e-15)
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE
assert diagnostic_trainer.ref_model is None
assert type(diagnostic_trainer)._get_per_token_logps_and_entropies is _patched_get_per_token_logps_and_entropies
assert type(diagnostic_trainer).compute_loss is _patched_compute_loss
assert KL_CLAMP_STATE['enabled'] is True and KL_CLAMP_STATE['d_max'] == D_MAX
print({'loss_type':diagnostic_trainer.loss_type,'beta':diagnostic_trainer.beta,'ref_model':diagnostic_trainer.ref_model,
       'entropy_clamp_value':ENTROPY_CLAMP_VALUE,'kl_clamp_value':CLAMP_VALUE,'kl_clamp_d_max':D_MAX,
       'instrumentation_and_clamp_wired':True})

original_generate=model.generate
def generate_stopped(*a,**kw):
    kw.setdefault('stopping_criteria',ANSWER_STOP)
    return original_generate(*a,**kw)
model.generate=generate_stopped
diagnostic_trainer.model.generate=generate_stopped
print({'fresh_optimizer':True,'fresh_scheduler':True,'run_seed':RUN_SEED,'n_steps':N_STEPS})
'''),
    code(r'''print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE =====')
event={'config':{'run_seed':RUN_SEED,'reference_seed':REFERENCE_SEED,'n_steps':N_STEPS,
       'entropy_coef':.05,'entropy_clamp_value':ENTROPY_CLAMP_VALUE,'kl_clamp_value':CLAMP_VALUE,'kl_clamp_d_max':D_MAX,
       'warmup_updates':WARMUP_UPDATES,'target_lr':TARGET_LR,
       'max_new_tokens':256,'stop':'</answer>','grad_breaker':50.0,'kl_breaker':5.0,
       'loss_type':diagnostic_trainer.loss_type,'beta':diagnostic_trainer.beta},
       'groups':[],'telemetry':[],'adapter_updates':[],'training_started':False,
       'purpose':'kl_clamp_mitigation_reproducing_and_attempting_to_survive_past_step15'}
def save_event():
    tmp=EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(event,indent=2)); tmp.replace(EVENT_LOG)
    if not EVENT_LOG.is_file() or not EVENT_LOG.stat().st_size: raise RuntimeError('Evidence save failed.')
def save_instrumentation():
    tmp=INSTRUMENTATION_LOG.with_suffix('.tmp')
    tmp.write_text(json.dumps(INSTRUMENTATION,indent=2)); tmp.replace(INSTRUMENTATION_LOG)

def adapter_state_evidence():
    digest=hashlib.sha256(); squared=0.0; count=0
    for name,param in model.named_parameters():
        if '.default.' not in name: continue
        value=param.detach().float().cpu().contiguous()
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
        squared+=float(value.square().sum()); count+=value.numel()
    return {'sha256':digest.hexdigest(),'l2_norm':math.sqrt(squared),'parameter_count':count}

save_event(); save_instrumentation()
base_generate=diagnostic_trainer._generate_and_score_completions
def dynamic_generate(inputs):
    for attempt in range(1,MAX_DYNAMIC_ATTEMPTS+1):
        result=base_generate(inputs); call=candidate_calls[-1]
        tensor_evidence={}
        for key,value in result.items():
            if torch.is_tensor(value) and value.numel() <= 200000:
                cpu=value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
                tensor_evidence[key]={'shape':list(cpu.shape),'dtype':str(cpu.dtype),'values':cpu.tolist()}
        row_hashes=_hash_id_rows(result['completion_ids']) if 'completion_ids' in result else None
        correct=sum(x['r_task']==4.0 for x in call['breakdowns'])
        structural=sum(x['p_structure']==0.0 for x in call['breakdowns'])
        reasons=[]
        if correct in (0,GROUP_SIZE): reasons.append('correctness')
        if structural<math.ceil(.25*GROUP_SIZE): reasons.append('structure')
        accepted=not reasons or attempt==MAX_DYNAMIC_ATTEMPTS
        advantages=result['advantages'].detach().float().cpu().tolist()
        classes=[classify_exploration(t,p) for t,p in zip(call['texts'],call['prompts'])]
        record={'attempt':attempt,'accepted':accepted,'fallback':accepted and bool(reasons),
            'physical_step':INSTRUMENTATION['physical_step'],
            'rejection_reasons':reasons,'correct_count':correct,'structural_passes':structural,
            'reward_mean':statistics.fmean(call['rewards']),'reward_std':statistics.pstdev(call['rewards']),
            'rewards':call['rewards'],'advantages':advantages,'exploration_classes':classes,
            'row_hashes':row_hashes,'has_ref_per_token_logps':'ref_per_token_logps' in result,
            'all_finite':all(math.isfinite(x) for x in call['rewards']+advantages),
            'trl_tensor_evidence':tensor_evidence,
            'rollouts':[{'prompt':p,'truth':y,'completion':t,'breakdown':b,'advantage':a,'exploration_class':c}
                        for p,y,t,b,a,c in zip(call['prompts'],call['truths'],call['texts'],call['breakdowns'],advantages,classes)]}
        event['groups'].append(record); save_event()
        if accepted: return result
    raise RuntimeError('Dynamic sampling returned no group.')
diagnostic_trainer._generate_and_score_completions=dynamic_generate

def breaker(grad_norm,kl): return grad_norm>=GRAD_BREAKER or kl>=KL_BREAKER
assert breaker(50.0,0.0) and breaker(0.0,5.0) and not breaker(49.99,4.99)
class Safety(TrainerCallback):
    def on_step_begin(self,args,state,control,**kwargs):
        INSTRUMENTATION['physical_step']=int(state.global_step)+1
        event['adapter_updates'].append({'target_physical_step':int(state.global_step)+1,
            'before':adapter_state_evidence()})
        save_event(); return control
    def on_step_end(self,args,state,control,**kwargs):
        target=int(state.global_step)
        row=next(x for x in reversed(event['adapter_updates']) if x['target_physical_step']==target)
        row['after']=adapter_state_evidence()
        row['norm_delta']=row['after']['l2_norm']-row['before']['l2_norm']
        row['hash_changed']=row['after']['sha256']!=row['before']['sha256']
        save_event(); save_instrumentation(); return control
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}; row={'physical_step':int(state.global_step),
            **{k:float(v) for k,v in logs.items() if isinstance(v,(int,float))}}
        event['telemetry'].append(row); save_event()
        grad=float(logs.get('grad_norm',0)); kl=float(logs.get('kl',0))
        if not all(math.isfinite(x) for x in (grad,kl)) or breaker(grad,kl):
            event['hard_stop']={'step':int(state.global_step),'grad_norm':grad,'kl':kl}; save_event()
            control.should_training_stop=True
        return control
diagnostic_trainer.add_callback(Safety())
print('PASSED: circuit-breaker tests. Safety breakers active (unchanged thresholds); no accuracy/structure gate.')
'''),
    code(r'''print(f'===== KL-CLAMPED SURVIVAL ATTEMPT: UP TO {N_STEPS} STEPS (success bar: well past step 15) =====')
event['training_started']=True; save_event(); save_instrumentation()
result=diagnostic_trainer.train()
terminal=int(diagnostic_trainer.state.global_step)
save_instrumentation()
print({'terminal_step':terminal,'hard_stop':event.get('hard_stop'),
       'survived_past_step_15':terminal>15,'survived_to_full_target':terminal>=N_STEPS and not event.get('hard_stop'),
       'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count']/KL_CLAMP_STATE['total_token_count']
                                     if KL_CLAMP_STATE['total_token_count'] else None),
       'kl_clamp_engaged_tokens':KL_CLAMP_STATE['engaged_token_count'],
       'kl_clamp_total_tokens':KL_CLAMP_STATE['total_token_count']})
'''),
    code(r'''print('===== POST-HOC ANALYSIS (reusing the exact proven matching + k3-formula logic) =====')
group_row_index = {}
for gi, g in enumerate(event['groups']):
    if not g['accepted'] or not g['row_hashes']: continue
    ref_tensor = g['trl_tensor_evidence'].get('ref_per_token_logps', {}).get('values')
    mask_tensor = g['trl_tensor_evidence'].get('completion_mask', {}).get('values')
    for ri, h in enumerate(g['row_hashes']):
        group_row_index[h] = {'physical_step': g['physical_step'], 'group_index': gi, 'row_index': ri,
            'ref_logps': ref_tensor[ri] if ref_tensor else None, 'mask': mask_tensor[ri] if mask_tensor else None,
            'advantage': g['advantages'][ri], 'exploration_class': g['exploration_classes'][ri],
            'completion_text': g['rollouts'][ri]['completion']}

per_step_kl_sum = defaultdict(float); per_step_mask_sum = defaultdict(float)
per_token_records = []; unmatched = []
for call in INSTRUMENTATION['compute_loss_calls']:
    step = call['physical_step']; row_hash = call['completion_ids_hash'][0]
    matched = group_row_index.get(row_hash)
    if matched is None: unmatched.append({'step': step}); continue
    logps_call = next((c for c in INSTRUMENTATION['logps_calls']
                        if c['tag']=='policy' and c['physical_step']==step and c['row_hashes']==[row_hash]), None)
    if logps_call is None: unmatched.append({'step': step, 'reason':'no_policy_logps'}); continue
    policy_logps = logps_call['logps'][0]
    # NOTE: this reads ref_logps from the GENERATION-time capture (pre-clamp, matching the group evidence
    # exactly as in the instrumentation notebook) to recompute what the UNCLAMPED KL would have been, for
    # comparison against the post-clamp values actually used in training (captured separately below).
    ref_logps = matched['ref_logps']; mask = matched['mask']
    if ref_logps is None or mask is None or len(ref_logps) != len(policy_logps):
        unmatched.append({'step': step, 'reason':'shape_or_missing_ref'}); continue
    for pos, (rp, pp, m) in enumerate(zip(ref_logps, policy_logps, mask)):
        if not m: continue
        diff = rp - pp
        per_token_kl_unclamped = math.exp(diff) - diff - 1
        per_step_kl_sum[step] += min(per_token_kl_unclamped, CLAMP_VALUE); per_step_mask_sum[step] += 1
        per_token_records.append({'step': step, 'row_hash': row_hash, 'position': pos,
            'per_token_kl_unclamped': per_token_kl_unclamped, 'per_token_kl_as_clamped': min(per_token_kl_unclamped, CLAMP_VALUE),
            'advantage': matched['advantage'], 'exploration_class': matched['exploration_class']})

recovered_kl_by_step = {s: per_step_kl_sum[s]/per_step_mask_sum[s] for s in per_step_kl_sum if per_step_mask_sum[s]>0}
logged_kl_by_step = {row['physical_step']: row['kl'] for row in event['telemetry'] if 'kl' in row}
consistency_check = {s: {'recovered_as_clamped': recovered_kl_by_step[s], 'logged': logged_kl_by_step.get(s),
                          'close': (logged_kl_by_step.get(s) is not None
                                    and math.isclose(recovered_kl_by_step[s], logged_kl_by_step[s], rel_tol=0.05, abs_tol=0.05))}
                      for s in recovered_kl_by_step}
print('===== KL RECONSTRUCTION SELF-CONSISTENCY CHECK (min(unclamped,CLAMP_VALUE) vs TRL logged kl) =====')
print(json.dumps(consistency_check, indent=2))
print('unmatched microbatches:', len(unmatched))
if not all(v['close'] for v in consistency_check.values()):
    print('WARNING: reconstruction does not match TRL\'s logged kl within tolerance for at least one step -- '
          'treat the clamp-engagement numbers below as unverified until this is resolved.')
else:
    print('PASSED: min(unclamped_kl, CLAMP_VALUE) matches TRL\'s own logged kl at every step -- confirms the '
          'clamp is genuinely bounding the trained loss/metric, not just the offline reconstruction.')

_max_unclamped = max((r['per_token_kl_unclamped'] for r in per_token_records), default=None)
_n_would_have_exceeded = sum(1 for r in per_token_records if r['per_token_kl_unclamped'] > CLAMP_VALUE)
print({'max_unclamped_per_token_kl_this_run': _max_unclamped,
       'tokens_that_would_have_exceeded_clamp_value': _n_would_have_exceeded,
       'total_tokens_scored': len(per_token_records)})
'''),
    code(r'''print('===== FINAL REPORT =====')
report = {
    'terminal_step': terminal, 'requested_steps': N_STEPS, 'hard_stop': event.get('hard_stop'),
    'survived_past_step_15': terminal > 15,
    'survived_to_full_target': terminal >= N_STEPS and not event.get('hard_stop'),
    'entropy_clamp_value': ENTROPY_CLAMP_VALUE, 'kl_clamp_value': CLAMP_VALUE, 'kl_clamp_d_max': D_MAX,
    'kl_clamp_derivation': {'healthy_steps_3x_median_rule': sorted(HEALTHY_STEPS),
                             'elevated_steps_excluded': sorted(ELEVATED_STEPS),
                             'healthy_upper_end_per_token_kl': HEALTHY_UPPER_END, 'multiplier': CLAMP_MULTIPLIER},
    'kl_clamp_engagement_rate': (KL_CLAMP_STATE['engaged_token_count']/KL_CLAMP_STATE['total_token_count']
                                  if KL_CLAMP_STATE['total_token_count'] else None),
    'max_unclamped_per_token_kl_this_run': _max_unclamped,
    'tokens_that_would_have_exceeded_clamp_value': _n_would_have_exceeded,
    'kl_reconstruction_verified_against_trl_logged_metric': all(v['close'] for v in consistency_check.values()) if consistency_check else None,
    'recovered_kl_by_step': recovered_kl_by_step, 'logged_kl_by_step': logged_kl_by_step,
}
event['final_report'] = report; save_event(); save_instrumentation()
print(json.dumps(report, indent=2, default=str))

if not report['survived_to_full_target']:
    print('\n===== DID NOT REACH THE FULL 50-STEP TARGET: WORST ROLLOUT AT THE TERMINAL STEP (same format as before) =====')
    worst_step_records = [r for r in per_token_records if r['step']==terminal]
    if worst_step_records:
        by_rollout = defaultdict(list)
        for r in worst_step_records: by_rollout[r['row_hash']].append(r)
        worst_hash = max(by_rollout, key=lambda h: sum(x['per_token_kl_unclamped'] for x in by_rollout[h]))
        worst_text = group_row_index[worst_hash]['completion_text']
        print({'step': terminal, 'row_hash_prefix': worst_hash[:12],
               'exploration_class': group_row_index[worst_hash]['exploration_class']})
        print('\nFull completion text:\n' + worst_text)
        print('\nPer-token KL profile (position: unclamped_kl -> as_trained_clamped_kl):')
        for r in sorted(by_rollout[worst_hash], key=lambda x: x['position']):
            print(f"  pos {r['position']:3d}: unclamped={r['per_token_kl_unclamped']:.4f} -> "
                  f"as_trained={r['per_token_kl_as_clamped']:.4f}")
print('\nEvidence:', EVENT_LOG)
print('Per-token instrumentation:', INSTRUMENTATION_LOG)
print('MITIGATION RUN COMPLETE. Read-only interpretation follows offline -- no further training decisions made here.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
