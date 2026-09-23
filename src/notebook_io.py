"""Shared safe-write helper for every build_*.py notebook builder.

Added 2026-08-22 after an incident where re-running build scripts to verify
a path fix silently overwrote 6 notebooks that had real, already-executed
Colab output with blank unexecuted templates (recovered by reconstruction
from the conversation record; see logs/development_log.md and the notice
at the top of each affected notebook). Every build script must route its
final write through safe_write_notebook() instead of calling
target.write_text(...) directly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _has_executed_output(notebook: dict) -> bool:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            return True
    return False


def safe_write_notebook(notebook: dict, target: Path, *, force: bool | None = None) -> Path:
    """Write notebook JSON to target, refusing to silently clobber executed output.

    If target already exists and contains any code cell with non-empty
    outputs or a non-None execution_count, the existing file is renamed to a
    timestamped backup before the fresh template is written -- never
    silently overwritten. Pass force=True (or run with --force on the
    command line) to skip the backup step and overwrite directly; this is
    never the default.
    """
    target = Path(target)
    if force is None:
        force = "--force" in sys.argv

    if target.is_file():
        try:
            existing = json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        if existing is not None and _has_executed_output(existing):
            if force:
                print(f"WARNING: {target} has executed output cells; --force set, overwriting directly.")
            else:
                backup = target.with_name(f"{target.stem}.backup-{int(time.time())}{target.suffix}")
                target.rename(backup)
                print(f"EXISTING NOTEBOOK HAD EXECUTED OUTPUT -- backed up to {backup} "
                      f"before writing a fresh template. Pass --force to skip this backup instead.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1))
    return target
