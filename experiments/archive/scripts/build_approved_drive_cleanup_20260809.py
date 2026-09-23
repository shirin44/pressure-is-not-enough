"""Build the explicitly approved Drive checkpoint-pruning notebook."""
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").is_dir())
sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook

TARGET=ROOT/'experiments'/'archive'/'notebooks'/'approved_drive_checkpoint_cleanup_20260809.ipynb'

def cell(kind,source):
    c={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
    if kind=='code': c.update({'execution_count':None,'outputs':[]})
    return c

cells=[
cell('markdown','''# Approved Drive checkpoint cleanup — 2026-08-09

Destructive but narrowly scoped. Verifies the logical-step 600/750/900 recovery checkpoints and protected snapshot/audit trees before deleting only the approved redundant checkpoints and superseded directories.
'''),
cell('code',r'''from pathlib import Path
import json,shutil
from google.colab import drive
drive.mount('/content/drive',force_remount=False)
ROOT=Path('/content/drive/MyDrive/AISI')
CHECKPOINTS=ROOT/'checkpoints'
if not CHECKPOINTS.is_dir(): raise RuntimeError(f'Missing checkpoint root: {CHECKPOINTS}')
print('CLEANUP ROOT:',ROOT)
'''),
cell('code',r'''def tree_bytes(path):
    return sum(x.stat().st_size for x in path.rglob('*') if x.is_file()) if path.exists() else 0

def valid_recovery(path):
    if not path.is_dir(): return False
    files=[x for x in path.rglob('*') if x.is_file()]
    names={x.name for x in files}
    return ('trainer_state.json' in names and 'adapter_model.safetensors' in names
            and 'optimizer.pt' in names and 'scheduler.pt' in names
            and files and all(x.stat().st_size>0 for x in files))

KEEP={
 'logical_600':CHECKPOINTS/'grpo-step500-structure-consistency-continuation-v3'/'checkpoint-100',
 'logical_750':CHECKPOINTS/'grpo-step600-structure-consistency-continuation-v5'/'checkpoint-150',
 'logical_900':CHECKPOINTS/'grpo-step750-pcot2-instrumented-real-v2'/'checkpoint-150',
}
for label,path in KEEP.items():
    if not valid_recovery(path): raise RuntimeError(f'REFUSE DELETE: invalid retained {label}: {path}')

PROTECTED=[CHECKPOINTS/'full-snapshots'/'step-500',ROOT/'audits']
protected_before={str(path):tree_bytes(path) for path in PROTECTED}
if not (CHECKPOINTS/'full-snapshots'/'step-500'/'adapter_model.safetensors').is_file():
    raise RuntimeError('REFUSE DELETE: protected checkpoint-500 snapshot is missing.')
print('PREFLIGHT PASSED:')
print({label:{'path':str(path),'MiB':round(tree_bytes(path)/2**20,2)} for label,path in KEEP.items()})
print('Protected trees:',protected_before)
'''),
cell('code',r'''# Resolve only exact approved targets. Never use recursive globs outside these phase directories.
targets=[]

def add_nested_except(directory,keep_names):
    if not directory.is_dir(): return
    for child in directory.iterdir():
        if child.is_dir() and child.name.startswith('checkpoint-') and child.name not in keep_names:
            targets.append(child)

add_nested_except(CHECKPOINTS/'grpo-step500-structure-consistency-continuation-v3',{'checkpoint-100'})
add_nested_except(CHECKPOINTS/'grpo-step600-structure-consistency-continuation-v5',{'checkpoint-150'})
add_nested_except(CHECKPOINTS/'grpo-step750-pcot2-instrumented-real-v2',{'checkpoint-150'})
add_nested_except(CHECKPOINTS/'grpo-step900-novelty-seed-v1',set())
# Checkpoint 500 is the only full snapshot needed for the current recovery.
# Logical 600/750/900 are retained above as verified full trainer checkpoints.
full_snapshot_root=CHECKPOINTS/'full-snapshots'
if full_snapshot_root.is_dir():
    for child in full_snapshot_root.iterdir():
        if child.is_dir() and child.name.startswith('step-') and child.name!='step-500':
            targets.append(child)

whole_directories=[
 'grpo-step500-structure-consistency-continuation-v2',
 'grpo-step500-structure-consistency-continuation-v4',
 'grpo-step600-structure-consistency-continuation-v1',
 'grpo-step600-structure-consistency-continuation-v2',
 'grpo-step600-structure-consistency-continuation-v3',
 'grpo-step600-structure-consistency-continuation-v4',
 'grpo-step750-pcot2-instrumented-real-v1',
 'grpo-step750-pcot2-replay754-v1',
 'grpo-checkpoint500-strict-grammar-probe-v2',
 'grpo-checkpoint500-strict-grammar-probe-v3',
 'grpo-checkpoint500-strict-grammar-probe-v4',
 'grpo-checkpoint500-strict-grammar-probe-v5',
 'grpo-checkpoint500-strict-grammar-probe384-v2',
 # Superseded pre-500 runs and completed diagnostics. Their small audit JSON
 # evidence is preserved separately under AISI/audits.
 'grpo-coinflip-100-step',
 'grpo-continuation-95-to500-v1',
 'grpo-entropy005-continuation-to95-v15',
 'grpo-entropy005-continuation-to95-v16',
 'grpo-entropy005-continuation-to95-v17-quiet',
 'grpo-entropy-sanity-coef005-v13',
 'grpo-entropy-sanity-coef005-v14',
 'grpo-entropy-sanity-fresh-schedule-v12',
 'grpo-entropy-sanity-step70-73-v11',
 'grpo-step70-100-continuation-v10-full-log',
 'grpo-step50-70-tight-length-v9-l4-microbatch-full-log',
 'grpo-step50-70-shared-trend-clean-v6-full-log',
 'grpo-step50-70-trailing-window-v4-full-log',
 'grpo-step50-70-shared-trend-v5-full-log',
 'grpo-step50-70-tight-length-v7-full-log',
 'grpo-step50-70-tight-length-v8-memory-safe-full-log',
 'grpo-step50-60-fresh-kernel-v3-full-log',
 'grpo-step50-60-fixed-v2-full-log',
 'grpo-step50-60-full-log',
 'grpo-step500-structure-consistency-dryrun-v1',
 'grpo-step500-structure-consistency-dryrun-v2',
 'grpo-step500-structure-consistency-dryrun-v3',
 'grpo-step500-structure-consistency-dryrun-v4',
 'grpo-step750-pcot2-dryrun-v1',
 'grpo-step750-pcot2-dryrun25-v1',
 'grpo-step750-pcot2-instrumented20-v1',
 'grpo-step750-pcot2-replay754-v2',
]
targets.extend(CHECKPOINTS/name for name in whole_directories if (CHECKPOINTS/name).exists())

# Deduplicate and enforce containment before showing the final destructive scope.
targets=sorted(set(x.resolve() for x in targets),key=str)
checkpoint_root=CHECKPOINTS.resolve()
for path in targets:
    if path==checkpoint_root or checkpoint_root not in path.parents:
        raise RuntimeError(f'REFUSE DELETE: target outside checkpoint root: {path}')
    if path in {x.resolve() for x in KEEP.values()}:
        raise RuntimeError(f'REFUSE DELETE: retained checkpoint targeted: {path}')

planned_bytes=sum(tree_bytes(path) for path in targets)
print(f'APPROVED TARGETS: {len(targets)}; represented size={planned_bytes/2**30:.3f} GiB')
for path in targets: print(f'{tree_bytes(path)/2**20:9.2f} MiB | {path.relative_to(ROOT)}')
'''),
cell('code',r'''print('===== EXECUTING APPROVED DELETION =====')
deleted=[]
for path in targets:
    size=tree_bytes(path)
    shutil.rmtree(path)
    if path.exists(): raise RuntimeError(f'Deletion verification failed: {path}')
    deleted.append({'path':str(path.relative_to(ROOT)),'bytes':size})

# Verify every protected/recovery asset after deletion.
for label,path in KEEP.items():
    if not valid_recovery(path): raise RuntimeError(f'POST-CLEANUP RECOVERY FAILURE: {label}: {path}')
protected_after={str(path):tree_bytes(path) for path in PROTECTED}
if protected_after!=protected_before:
    raise RuntimeError(f'Protected tree changed: before={protected_before}, after={protected_after}')

report={'deleted_targets':len(deleted),'recovered_bytes':sum(x['bytes'] for x in deleted),
        'recovered_GiB':sum(x['bytes'] for x in deleted)/2**30,
        'retained_recovery_points':{k:str(v) for k,v in KEEP.items()},
        'protected_unchanged':True,'deleted':deleted}
REPORT=ROOT/'drive_cleanup_20260809_report.json'
REPORT.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='deleted'},indent=2))
print('REPORT:',REPORT)
print('Cleanup complete; approved deletions cannot be recovered from this notebook.')
''')]

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
safe_write_notebook(nb, TARGET); print(TARGET)
