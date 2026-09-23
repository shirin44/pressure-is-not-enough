"""Step 8: re-audit token pairs under the selected Step-0 milestone-12 policy.

This is inference-only. It loads the untouched Qwen base plus the actual milestone-12
LoRA, scores the same 30 candidates in the same representative state-slot context as
Step 4, and preserves the static tokenizer/edit-distance/leakage checks unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from token_pool_audit import CANDIDATES, build_table

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SELECTED_PAIRS = {
    "training": ("Nib", "Nomo"),
    "held_out": ("Yelt", "Yark"),
}
ORIGINAL_BASE_LOGPROBS = {
    "Nib": -6.6816816329956055,
    "Nomo": -6.6816816329956055,
    "Yelt": -7.2441816329956055,
    "Yark": -7.2441816329956055,
}
ADAPTER = Path(os.environ.get(
    "STEP0_ADAPTER",
    Path.home() / "aisi_checkpoints" / "exp3-step0-task-foundation-full-v1" / "milestone-12",
)).expanduser()
OUTPUT = Path(os.environ.get(
    "STEP8_OUTPUT",
    Path.home() / "aisi_checkpoints" / "exp3-step8-policy-token-pair-reaudit-v1.json",
)).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_pairs(rows: list[dict]) -> dict:
    by_token = {row["token"]: row for row in rows}
    result = {}
    for role, (left, right) in SELECTED_PAIRS.items():
        a, b = by_token[left], by_token[right]
        lp_a = a["step0_policy_logprob_at_state_slot"]
        lp_b = b["step0_policy_logprob_at_state_slot"]
        result[role] = {
            "tokens": [left, right],
            "original_base_logprobs": [ORIGINAL_BASE_LOGPROBS[left], ORIGINAL_BASE_LOGPROBS[right]],
            "original_base_abs_gap": abs(ORIGINAL_BASE_LOGPROBS[left] - ORIGINAL_BASE_LOGPROBS[right]),
            "step0_policy_logprobs": [lp_a, lp_b],
            "step0_policy_abs_gap": abs(lp_a - lp_b),
            "both_static_eligible": a["eligible"] and b["eligible"],
            "both_pass_leakage_classifier": (
                a["passes_leakage_classifier"] and b["passes_leakage_classifier"]
            ),
            "min_edit_distances": [
                a["min_edit_distance_to_banned_word"],
                b["min_edit_distance_to_banned_word"],
            ],
        }
    return result


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Step-0 policy re-audit")
    adapter_weights = ADAPTER / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False)
    model.eval()

    rows = build_table(tokenizer, model=model, device=torch.device("cuda:0"))
    for row in rows:
        row["step0_policy_logprob_at_state_slot"] = row.pop("base_policy_logprob_at_state_slot")
    result = {
        "purpose": "stage09_step8_selected_pair_reaudit_under_actual_step0_milestone12_policy",
        "model_name": MODEL_NAME,
        "adapter_path": str(ADAPTER),
        "adapter_sha256": sha256_file(adapter_weights),
        "context": "Step 1: The state flips. State:",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("transformers", "peft", "torch")},
        "candidate_count": len(CANDIDATES),
        "pairs": summarize_pairs(rows),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(OUTPUT)
    print(json.dumps({"adapter_sha256": result["adapter_sha256"], "pairs": result["pairs"]}, indent=2))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
