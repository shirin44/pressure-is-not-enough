#!/usr/bin/env python3
"""Build a lightweight, read-only inspector for the completed step-100 gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
OUTPUT = ROOT / "notebooks" / "multidomain_stage3_step100_report.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True),
    }


def main() -> None:
    notebook = {
        "cells": [
            markdown(
                "# Multi-domain Stage 3 — Step 100 completed-run report\n\n"
                "Read-only: this notebook does not load a model, change a checkpoint, or train.\n"
            ),
            code(
                "from pathlib import Path\n"
                "import json\n"
                "from google.colab import drive\n\n"
                "drive.mount('/content/drive', force_remount=False)\n"
                "REPORT = Path('/content/drive/MyDrive/AISI/checkpoints/multidomain-stage3-v1/step100_fan_fix_report.json')\n"
                "CHECKPOINT = Path('/content/drive/MyDrive/AISI/checkpoints/multidomain-stage3-v1/trainer-output/checkpoint-100')\n"
                "if not REPORT.is_file():\n"
                "    raise RuntimeError(f'Missing report: {REPORT}')\n"
                "report = json.loads(REPORT.read_text())\n"
                "if not report.get('step100_complete'):\n"
                "    raise RuntimeError('The report exists but is not marked step100_complete.')\n"
                "required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt')\n"
                "checkpoint_ok = CHECKPOINT.is_dir() and all((CHECKPOINT / name).is_file() and (CHECKPOINT / name).stat().st_size > 0 for name in required)\n"
                "adapter_ok = CHECKPOINT.is_dir() and any(p.name.startswith('adapter_model') and p.stat().st_size > 0 for p in CHECKPOINT.iterdir())\n"
                "print('COMPLETED RUN VERIFIED:', {'step100_complete': True, 'checkpoint': str(CHECKPOINT), 'checkpoint_valid': checkpoint_ok and adapter_ok})\n"
            ),
            code(
                "print('===== STEP 100 PER-DOMAIN AND POOLED METRICS =====')\n"
                "print(json.dumps(report.get('metrics', {}), indent=2, sort_keys=True))\n\n"
                "print('===== FAN WORDING-FIX RESPONSE =====')\n"
                "print(json.dumps(report.get('fan_response_trajectory', {}), indent=2, sort_keys=True))\n\n"
                "print('===== VALVE/LAMP NON-REGRESSION =====')\n"
                "print(json.dumps(report.get('valve_lamp_nonregression', {}), indent=2, sort_keys=True))\n\n"
                "print('===== POOLED 95% ACCEPTANCE GATE =====')\n"
                "print(json.dumps(report.get('pooled_acceptance_gate', {}), indent=2, sort_keys=True))\n"
            ),
            code(
                "metrics = report['metrics']\n"
                "summary = {\n"
                "    'checkpoint': report.get('checkpoint'),\n"
                "    'stop_reason': report.get('stop_reason'),\n"
                "    'fan_tracking': metrics['per_domain']['fan']['transition_tracking_rate'],\n"
                "    'valve_tracking': metrics['per_domain']['valve']['transition_tracking_rate'],\n"
                "    'lamp_tracking': metrics['per_domain']['lamp']['transition_tracking_rate'],\n"
                "    'pooled_accuracy': metrics['pooled']['physical_final_answer_accuracy'],\n"
                "    'full_95_percent_gate_passed': report['pooled_acceptance_gate']['full_pooled_gate_passed'],\n"
                "    'stage4_authorized': report.get('stage4_authorized', False),\n"
                "}\n"
                "print('===== TL;DR =====')\n"
                "print(json.dumps(summary, indent=2, sort_keys=True))\n"
                "print('READ-ONLY INSPECTION COMPLETE. No training was run.')\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
