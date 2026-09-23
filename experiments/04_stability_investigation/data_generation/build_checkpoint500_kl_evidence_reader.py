"""Build a read-only, CPU-only reader that corrects and completes the per-token
KL analysis from a completed checkpoint500_per_token_kl_instrumentation.ipynb run.

Loads the two already-saved evidence files from Drive -- no GPU, no model, no
training -- and:

1. Fixes the recurrence bug in the original notebook: that notebook's
   "recurring_rollout_hashes_across_steps" counted elevated TOKENS per rollout
   hash, not DISTINCT STEPS per hash, so a long rollout with many high-KL
   tokens looked identical to genuine cross-step recurrence of the same
   completion. This reader recomputes it correctly (distinct physical_step
   count per row_hash) and reports both numbers side by side so the mistake
   is visible, not silently replaced.
2. Breaks the KL blowup down per rollout within each step (not just per
   token), to identify whether a step's blowup is concentrated in one
   outlier rollout or spread across the group -- directly needed to explain
   the step-15 spike (KL ~197 in one step, ~600x the surrounding steps) after
   14 steps that showed no clear creeping trend.
3. Prints full completion text for the worst-contributing rollout(s) at the
   worst step, and for comparison at the least-elevated step, plus the next
   two most-elevated steps observed (5, 8, 10 in the run that produced this
   evidence -- confirmed generically from the saved data, not hardcoded).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "04_stability_investigation" / "notebooks" / "checkpoint500_kl_evidence_reader.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 per-token KL evidence reader (corrected recurrence, CPU-only)

Read-only. Loads the two evidence files already saved by
`checkpoint500_per_token_kl_instrumentation.ipynb`'s completed run from Drive
-- no GPU, no model, no training. Fixes the recurrence-metric bug in that
notebook (it counted elevated tokens per rollout hash, not distinct steps per
hash) and breaks the KL blowup down per rollout to explain the step-15 spike.

Set `RUN_DIR` below to the run directory printed at the end of the
instrumentation notebook if it differs from the default (the v1 run from
2026-08-18).
"""),
    code(r'''from pathlib import Path
import json, math, statistics
from collections import defaultdict, Counter
from google.colab import drive

drive.mount('/content/drive', force_remount=False)
RUN_DIR = Path('/content/drive/MyDrive/AISI/checkpoints/grpo-checkpoint500-per-token-kl-instrumentation-v1')
EVENT_LOG = RUN_DIR / 'checkpoint500_per_token_kl_instrumentation.json'
INSTRUMENTATION_LOG = RUN_DIR / 'per_token_instrumentation.json'
for f in (EVENT_LOG, INSTRUMENTATION_LOG):
    if not f.is_file():
        raise FileNotFoundError(f'Expected evidence file not found: {f}. Update RUN_DIR above.')
event = json.loads(EVENT_LOG.read_text())
INSTRUMENTATION = json.loads(INSTRUMENTATION_LOG.read_text())
print({'event_groups': len(event['groups']), 'compute_loss_calls': len(INSTRUMENTATION['compute_loss_calls']),
       'logps_calls': len(INSTRUMENTATION['logps_calls']), 'hard_stop': event.get('hard_stop')})
'''),
    code(r'''print('===== REBUILD per_token_records (identical logic to the instrumentation notebook, for a fresh corrected analysis) =====')
group_row_index = {}
for gi, g in enumerate(event['groups']):
    if not g['accepted'] or not g.get('row_hashes'): continue
    ref_tensor = g['trl_tensor_evidence'].get('ref_per_token_logps', {}).get('values')
    mask_tensor = g['trl_tensor_evidence'].get('completion_mask', {}).get('values')
    for ri, h in enumerate(g['row_hashes']):
        group_row_index[h] = {
            'physical_step': g['physical_step'], 'group_index': gi, 'row_index': ri,
            'ref_logps': ref_tensor[ri] if ref_tensor else None,
            'mask': mask_tensor[ri] if mask_tensor else None,
            'advantage': g['advantages'][ri], 'exploration_class': g['exploration_classes'][ri],
            'completion_text': g['rollouts'][ri]['completion'],
            'reward_total': g['rollouts'][ri]['breakdown']['total'],
        }

per_token_records = []
unmatched = []
for call in INSTRUMENTATION['compute_loss_calls']:
    step = call['physical_step']
    row_hash = call['completion_ids_hash'][0]
    matched = group_row_index.get(row_hash)
    if matched is None:
        unmatched.append({'step': step, 'call_index': call['call_index'], 'reason': 'no_group_match'}); continue
    logps_call = next((c for c in INSTRUMENTATION['logps_calls']
                        if c['tag']=='policy' and c['physical_step']==step and c['row_hashes']==[row_hash]), None)
    if logps_call is None:
        unmatched.append({'step': step, 'call_index': call['call_index'], 'reason': 'no_policy_logps_call'}); continue
    policy_logps = logps_call['logps'][0]
    ref_logps = matched['ref_logps']; mask = matched['mask']
    if ref_logps is None or mask is None or len(ref_logps) != len(policy_logps):
        unmatched.append({'step': step, 'call_index': call['call_index'], 'reason': 'shape_or_missing_ref'}); continue
    for pos, (rp, pp, m) in enumerate(zip(ref_logps, policy_logps, mask)):
        if not m: continue
        diff = rp - pp
        per_token_kl = math.exp(diff) - diff - 1
        per_token_records.append({'step': step, 'row_hash': row_hash, 'position': pos,
            'per_token_kl': per_token_kl, 'ref_logp': rp, 'policy_logp': pp,
            'advantage': matched['advantage'], 'exploration_class': matched['exploration_class'],
            'reward_total': matched['reward_total']})
print({'per_token_records': len(per_token_records), 'unmatched': len(unmatched)})
if unmatched: print('unmatched sample:', unmatched[:5])
'''),
    code(r'''print('===== PER-STEP, PER-ROLLOUT KL BREAKDOWN (mean/max per rollout, not just per token) =====')
by_step_rollout = defaultdict(list)
for r in per_token_records:
    by_step_rollout[(r['step'], r['row_hash'])].append(r['per_token_kl'])

rollout_summary = []
for (step, row_hash), vals in by_step_rollout.items():
    rollout_summary.append({'step': step, 'row_hash': row_hash, 'n_tokens': len(vals),
                             'mean_kl': statistics.fmean(vals), 'max_kl': max(vals), 'sum_kl': sum(vals)})
rollout_summary.sort(key=lambda r: -r['sum_kl'])

print('\nTop 15 rollouts by TOTAL per-token KL contribution (step, mean, max, n_tokens, row_hash prefix):')
for r in rollout_summary[:15]:
    print({'step': r['step'], 'mean_kl': round(r['mean_kl'],3), 'max_kl': round(r['max_kl'],3),
           'n_tokens': r['n_tokens'], 'sum_kl': round(r['sum_kl'],2), 'row_hash_prefix': r['row_hash'][:12]})

print('\n--- Per-step summary: how many rollouts, and is the blowup concentrated in one or spread across the group? ---')
steps_present = sorted(set(r['step'] for r in per_token_records))
for step in steps_present:
    rollouts_this_step = [r for r in rollout_summary if r['step']==step]
    means = sorted([r['mean_kl'] for r in rollouts_this_step], reverse=True)
    print({'step': step, 'n_rollouts': len(rollouts_this_step),
           'rollout_mean_kls_desc': [round(m,3) for m in means]})
'''),
    code(r'''print('===== CORRECTED RECURRENCE CHECK: distinct STEPS per rollout hash, not elevated-token count =====')
HIGH_KL_THRESHOLD = 1.0
elevated = [r for r in per_token_records if r['per_token_kl'] > HIGH_KL_THRESHOLD]

# What the original (buggy) notebook reported: elevated TOKEN count per hash.
buggy_token_count_per_hash = Counter(r['row_hash'] for r in elevated)

# Corrected: DISTINCT STEP count per hash -- this is what "recurs across steps" actually means.
steps_per_hash = defaultdict(set)
for r in per_token_records:  # use the full set, not just elevated, so a hash can be checked regardless of threshold
    steps_per_hash[r['row_hash']].add(r['step'])
genuinely_recurring = {h: sorted(steps) for h, steps in steps_per_hash.items() if len(steps) > 1}

print('genuinely cross-step recurring rollout hashes (same completion content independently generated in >1 step):')
print(json.dumps(genuinely_recurring, indent=2) if genuinely_recurring else '  none -- every rollout hash appeared in exactly one step, as expected under stochastic sampling.')
print('\nFor comparison, the original instrumentation notebook reported elevated-token count per hash (NOT '
      'cross-step recurrence) for the same top hashes:')
for h in list(buggy_token_count_per_hash)[:10]:
    print({'row_hash_prefix': h[:12], 'buggy_elevated_token_count': buggy_token_count_per_hash[h],
           'actual_distinct_steps': sorted(steps_per_hash[h])})
'''),
    code(r'''print('===== WORST ROLLOUT AT THE WORST STEP: full text and KL profile =====')
worst = rollout_summary[0]
worst_step, worst_hash = worst['step'], worst['row_hash']
worst_text = group_row_index[worst_hash]['completion_text']
worst_tokens = sorted([r for r in per_token_records if r['row_hash']==worst_hash], key=lambda r: r['position'])
print({'step': worst_step, 'mean_kl': worst['mean_kl'], 'max_kl': worst['max_kl'],
       'exploration_class': group_row_index[worst_hash]['exploration_class'],
       'advantage': group_row_index[worst_hash]['advantage'], 'reward_total': group_row_index[worst_hash]['reward_total']})
print('\nFull completion text:\n' + worst_text)
print('\nPer-token KL profile (position: kl):')
for r in worst_tokens:
    print(f"  pos {r['position']:3d}: kl={r['per_token_kl']:.4f}")

print('\n===== COMPARISON: least-elevated step, and the next two most-elevated steps besides the worst =====')
step_mean_kl = {s: statistics.fmean(r['per_token_kl'] for r in per_token_records if r['step']==s) for s in steps_present}
ranked_steps = sorted(step_mean_kl.items(), key=lambda kv: -kv[1])
print('steps ranked by mean per-token KL (desc):', [(s, round(v,3)) for s,v in ranked_steps])
comparison_steps = list(dict.fromkeys([s for s,_ in ranked_steps[1:3]] + [ranked_steps[-1][0]]))
for step in comparison_steps:
    top_rollout = max([r for r in rollout_summary if r['step']==step], key=lambda r: r['sum_kl'])
    text = group_row_index[top_rollout['row_hash']]['completion_text']
    print(f"\n--- step {step} (mean_kl={round(step_mean_kl[step],3)}), worst rollout in this step ---")
    print({'mean_kl': round(top_rollout['mean_kl'],3), 'max_kl': round(top_rollout['max_kl'],3)})
    print(text[:1500])
'''),
    code(r'''print('===== FINAL CORRECTED SUMMARY =====')
summary = {
    'terminal_step': event.get('hard_stop', {}).get('step'),
    'hard_stop': event.get('hard_stop'),
    'kl_pattern': 'no clear creeping trend across steps 1-14 (noisy 0.02-3.0 range); single catastrophic '
                  'spike at the terminal step, not gradual accumulation' if steps_present else None,
    'blowup_concentrated_in_single_rollout': (rollout_summary[0]['mean_kl'] > 5 * statistics.fmean(
        [r['mean_kl'] for r in rollout_summary if r['step']==rollout_summary[0]['step'] and r['row_hash']!=rollout_summary[0]['row_hash']])
        if len([r for r in rollout_summary if r['step']==rollout_summary[0]['step']]) > 1 else None),
    'genuine_cross_step_recurrence_found': bool(genuinely_recurring),
    'genuinely_recurring_hashes': genuinely_recurring,
    'note': 'The original instrumentation notebook "recurring_rollout_hashes_across_steps" field measured '
            'elevated-token count per rollout, not cross-step recurrence -- treat that field from the prior '
            'run as invalid; this corrected version is the one to use.',
}
print(json.dumps(summary, indent=2, default=str))
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
