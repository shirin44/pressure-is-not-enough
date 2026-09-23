from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/stage35_declared_mapping_control.ipynb"
OUTPUT = ROOT / "notebooks/stage35_base_declared_mapping_control.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one builder substitution for: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError("Build the checkpoint-130 declared-mapping control first.")
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Stage 3.5 control — untrained base model with supplied mapping\n\n"
        "This is the exact declared-mapping greenhouse-fan control, evaluated on "
        "the untrained Qwen2.5-3B-Instruct base with no adapter loaded. It is "
        "read-only, resumable, and contains no training or Stage 4 composition.\n"
    ).splitlines(True)

    config = "".join(notebook["cells"][2]["source"])
    config = replace_once(config, "from peft import PeftModel\n", "")
    config = replace_once(
        config,
        "CHECKPOINT=Path('/content/drive/MyDrive/AISI/checkpoints/latch-seed-stage3-decode-back-sft-v3/trainer-output/checkpoint-130')\n",
        "",
    )
    config = replace_once(
        config,
        "OUTPUT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/stage35-declared-mapping-control-v1')",
        "OUTPUT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/stage35-base-declared-mapping-control-v1')",
    )
    start = config.index("def valid_adapter(path):")
    end = config.index("random.seed(SEED)")
    config = config[:start] + config[end:]
    config = replace_once(
        config,
        "print({'gpu':torch.cuda.get_device_name(0),'checkpoint':str(CHECKPOINT),\n       'training':False,'stage4_blocked':True})",
        "print({'gpu':torch.cuda.get_device_name(0),'model':MODEL_NAME,\n       'adapter_loaded':False,'training':False,'stage4_blocked':True})",
    )
    notebook["cells"][2]["source"] = config.splitlines(True)

    model_cell = "".join(notebook["cells"][5]["source"])
    model_cell = replace_once(
        model_cell,
        "base=AutoModelForCausalLM.from_pretrained(",
        "model=AutoModelForCausalLM.from_pretrained(",
    )
    old_tail = """model=PeftModel.from_pretrained(base,CHECKPOINT,is_trainable=False)
model.eval(); model.config.use_cache=True
assert not any(p.requires_grad for p in model.parameters())
adapter_files=sorted(p for p in CHECKPOINT.iterdir() if p.name.startswith('adapter_model'))
adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter_files}
print({'adapter_sha256':adapter_sha256,'trainable_parameters':0})
"""
    new_tail = """model.requires_grad_(False); model.eval(); model.config.use_cache=True
assert not any(p.requires_grad for p in model.parameters())
print({'model':MODEL_NAME,'adapter_loaded':False,'trainable_parameters':0,
       'confirmed':'untrained instruction-tuned base model only'})
"""
    model_cell = replace_once(model_cell, old_tail, new_tail)
    notebook["cells"][5]["source"] = model_cell.splitlines(True)

    report_cell = "".join(notebook["cells"][7]["source"])
    report_cell = replace_once(
        report_cell,
        "report={'stage':'3.5_declared_mapping_control','source_checkpoint':str(CHECKPOINT),\n        'mode':'zero_shot_no_training','audit':audit,'adapter_sha256':adapter_sha256,",
        "report={'stage':'3.5_base_declared_mapping_control','source_model':MODEL_NAME,\n        'adapter_loaded':False,'mode':'zero_shot_no_training','audit':audit,",
    )
    report_cell = replace_once(
        report_cell,
        "print('===== DECLARED-MAPPING CONTROL REPORT =====')",
        "print('===== BASE-MODEL DECLARED-MAPPING CONTROL REPORT =====')",
    )
    report_cell = replace_once(
        report_cell,
        "'next_action':'STOP_FOR_REVIEW'}",
        "'checkpoint130_reference':{'declared_mapping_adherence_rate':.05,\n"
        "                                  'final_answer_accuracy':.01,\n"
        "                                  'structural_rate':.88},\n"
        "        'interpretation_thresholds':{'active_damage_if_base_adherence_at_least':.30,\n"
        "                                     'similar_difficulty_band':[.05,.15]},\n"
        "        'next_action':'STOP_FOR_REVIEW'}",
    )
    notebook["cells"][7]["source"] = report_cell.splitlines(True)

    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
