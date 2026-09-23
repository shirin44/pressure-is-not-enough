"""Build a read-only Colab inventory for AISI Drive artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").is_dir())
sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Google Drive checkpoint inventory — read only\n\n",
                "This notebook only measures and classifies files under "
                "`MyDrive/AISI`. It never deletes or moves anything. It explicitly "
                "protects checkpoint 500, the active strict-grammar probe, and small audit evidence.\n",
            ],
        },
        code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import json
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/AISI')
if not ROOT.is_dir():
    raise RuntimeError(f'AISI Drive folder not found: {ROOT}')
print('READ-ONLY INVENTORY ROOT:', ROOT)
"""),
        code("""def recursive_size(path):
    total = 0
    files = 0
    errors = []
    try:
        for item in path.rglob('*'):
            try:
                if item.is_file():
                    total += item.stat().st_size
                    files += 1
            except OSError as exc:
                errors.append(f'{item}: {exc}')
    except OSError as exc:
        errors.append(f'{path}: {exc}')
    return total, files, errors

def human_size(size):
    value = float(size)
    for unit in ('B','KiB','MiB','GiB','TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024

def classify(path):
    relative = str(path.relative_to(ROOT))
    lower = relative.lower()
    if 'full-snapshots/step-500' in lower:
        return 'PROTECT_CHECKPOINT_500'
    if 'checkpoint500-strict-grammar-probe' in lower:
        return 'PROTECT_ACTIVE_GRAMMAR_PROBE'
    if 'development' in lower or 'dev-log' in lower:
        return 'PROTECT_DEVELOPMENT_LOG'
    if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.md','.txt'}:
        return 'KEEP_SMALL_EVIDENCE'
    if 'structure-consistency-continuation' in lower or 'pcot2' in lower or 'novelty' in lower:
        return 'REVIEW_COMPLETED_TRAINING_PHASE'
    if 'structure-consistency-dryrun' in lower:
        return 'CANDIDATE_COMPLETED_DRYRUN'
    if 'entropy-sanity' in lower or 'diagnostic' in lower:
        return 'CANDIDATE_COMPLETED_DIAGNOSTIC'
    if 'checkpoint-' in path.name.lower():
        return 'REVIEW_INTERMEDIATE_CHECKPOINT'
    if 'full-snapshots' in lower:
        return 'REVIEW_FULL_SNAPSHOT'
    if 'audit' in lower:
        return 'KEEP_SMALL_AUDIT_EVIDENCE'
    return 'REVIEW_OTHER'

targets=[]
for parent in (ROOT/'checkpoints', ROOT/'audits'):
    if not parent.is_dir():
        continue
    for child in sorted(parent.iterdir(), key=lambda x:x.name):
        size, files, errors = recursive_size(child)
        targets.append({
            'path': str(child),
            'relative_path': str(child.relative_to(ROOT)),
            'kind': 'directory' if child.is_dir() else 'file',
            'bytes': size if child.is_dir() else child.stat().st_size,
            'size': human_size(size if child.is_dir() else child.stat().st_size),
            'file_count': files if child.is_dir() else 1,
            'classification': classify(child),
            'errors': errors,
        })

targets.sort(key=lambda row: row['bytes'], reverse=True)
total=sum(row['bytes'] for row in targets)
print('===== AISI DRIVE INVENTORY (LARGEST FIRST) =====')
print('Total represented:', human_size(total))
for row in targets:
    print(f"{row['size']:>12} | {row['classification']:<39} | {row['relative_path']}")
    if row['errors']:
        print('  SIZE ERRORS:', row['errors'][:3])
"""),
        code("""# Show nested checkpoint directories separately; these are commonly the largest files.
nested=[]
checkpoint_root=ROOT/'checkpoints'
if checkpoint_root.is_dir():
    for path in checkpoint_root.rglob('checkpoint-*'):
        if not path.is_dir():
            continue
        size, files, errors=recursive_size(path)
        nested.append({
            'path':str(path),'relative_path':str(path.relative_to(ROOT)),
            'bytes':size,'size':human_size(size),'file_count':files,
            'classification':classify(path),'errors':errors,
        })
nested.sort(key=lambda row:row['bytes'],reverse=True)
print('===== NESTED CHECKPOINT DIRECTORIES =====')
for row in nested:
    print(f"{row['size']:>12} | {row['classification']:<39} | {row['relative_path']}")
"""),
        code("""report={
    'root':str(ROOT),
    'total_top_level_bytes':total,
    'top_level':targets,
    'nested_checkpoints':nested,
    'deletion_performed':False,
}
output=Path('/content/drive/MyDrive/AISI/drive_inventory.json')
temporary=output.with_suffix('.tmp')
temporary.write_text(json.dumps(report,indent=2))
temporary.replace(output)
print('Saved inventory:',output)
print('NO FILES WERE DELETED OR MOVED.')
"""),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = ROOT / "experiments" / "archive" / "notebooks" / "drive_checkpoint_inventory.ipynb"
safe_write_notebook(nb, target)
print(target)
