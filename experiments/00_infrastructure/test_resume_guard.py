from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.resume_guard import (
    assess_scheduler_state,
    enforce_resume_policy,
    linear_lr_used_for_update,
)


def main() -> None:
    healthy = assess_scheduler_state(
        {"last_epoch": 20, "total_steps": 100, "_last_lr": [8e-6], "base_lrs": [1e-5]},
        {"param_groups": [{"lr": 8e-6}]},
    )
    assert not healthy["exhausted"]
    enforce_resume_policy(healthy, resume_mode="literal")

    exhausted = assess_scheduler_state(
        {"last_epoch": 100, "total_steps": 100, "_last_lr": [0.0], "base_lrs": [1e-5]},
        {"param_groups": [{"lr": 0.0}]},
    )
    assert exhausted["exhausted"]
    try:
        enforce_resume_policy(exhausted, resume_mode="literal")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Exhausted literal resume was not refused")
    enforce_resume_policy(exhausted, resume_mode="weights_only_fresh_schedule")
    enforce_resume_policy(exhausted, resume_mode="literal", explicit_override=True)

    nearly_exhausted = assess_scheduler_state(
        {"last_epoch": 99, "_last_lr": [1e-7], "base_lrs": [1e-5]},
        {"param_groups": [{"lr": 1e-7}]},
    )
    assert nearly_exhausted["near_zero_lr"] and nearly_exhausted["exhausted"]

    values = [linear_lr_used_for_update(1e-5, 150, step) for step in (1, 75, 150)]
    assert values[0] == 1e-5
    assert values[0] > values[1] > values[2] > 0
    print("PASSED: permanent exhausted-scheduler resume guard.")
    print({"update_1": values[0], "update_75": values[1], "update_150": values[2]})


if __name__ == "__main__":
    main()
