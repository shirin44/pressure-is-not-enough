from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def assess_scheduler_state(
    scheduler_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    *,
    near_zero: float = 1e-12,
    near_zero_fraction: float = 0.01,
) -> dict[str, Any]:
    """Return a structured exhaustion assessment for checkpoint state.

    A scheduler is exhausted when its recorded learning rate is effectively
    zero or its recorded epoch has reached a known finite schedule horizon.
    Optimizer group learning rates are included when available because they
    are the values a literal resume would actually use.
    """
    scheduler_lrs = [float(x) for x in scheduler_state.get("_last_lr", [])]
    base_lrs = [float(x) for x in scheduler_state.get("base_lrs", [])]
    optimizer_lrs = []
    if optimizer_state is not None:
        optimizer_lrs = [
            float(group["lr"])
            for group in optimizer_state.get("param_groups", [])
            if group.get("lr") is not None
        ]
    effective_lrs = optimizer_lrs or scheduler_lrs
    last_epoch = scheduler_state.get("last_epoch")
    horizon = scheduler_state.get("total_steps")
    horizon_exhausted = (
        isinstance(last_epoch, (int, float))
        and isinstance(horizon, (int, float))
        and float(last_epoch) >= float(horizon)
    )
    relative_threshold = (
        max(abs(x) for x in base_lrs) * near_zero_fraction if base_lrs else 0.0
    )
    effective_threshold = max(near_zero, relative_threshold)
    near_zero_lr = (
        bool(effective_lrs)
        and max(abs(x) for x in effective_lrs) <= effective_threshold
    )
    nonfinite_lr = any(not math.isfinite(x) for x in effective_lrs)
    exhausted = horizon_exhausted or near_zero_lr or nonfinite_lr
    return {
        "exhausted": exhausted,
        "near_zero_threshold": near_zero,
        "near_zero_fraction": near_zero_fraction,
        "effective_near_zero_threshold": effective_threshold,
        "base_lrs": base_lrs,
        "scheduler_lrs": scheduler_lrs,
        "optimizer_lrs": optimizer_lrs,
        "effective_lrs": effective_lrs,
        "last_epoch": last_epoch,
        "schedule_horizon": horizon,
        "horizon_exhausted": horizon_exhausted,
        "near_zero_lr": near_zero_lr,
        "nonfinite_lr": nonfinite_lr,
    }


def enforce_resume_policy(
    report: Mapping[str, Any],
    *,
    resume_mode: str,
    explicit_override: bool = False,
) -> None:
    """Refuse an implicit literal resume from an exhausted scheduler."""
    allowed_modes = {"literal", "weights_only_fresh_schedule"}
    if resume_mode not in allowed_modes:
        raise ValueError(f"Unknown resume mode: {resume_mode!r}")
    if report.get("exhausted") and resume_mode == "literal" and not explicit_override:
        raise RuntimeError(
            "REFUSED: checkpoint scheduler is exhausted or has a near-zero/nonfinite "
            "learning rate. Choose weights_only_fresh_schedule or set an explicit "
            "override after reviewing the scheduler report."
        )


def inspect_checkpoint_scheduler(
    checkpoint: str | Path,
    *,
    near_zero: float = 1e-12,
    near_zero_fraction: float = 0.01,
) -> dict[str, Any]:
    """Load only optimizer/scheduler state files and assess resume safety."""
    import torch

    checkpoint = Path(checkpoint)
    scheduler_path = checkpoint / "scheduler.pt"
    optimizer_path = checkpoint / "optimizer.pt"
    if not scheduler_path.is_file() or not optimizer_path.is_file():
        raise FileNotFoundError(
            f"Expected scheduler.pt and optimizer.pt in checkpoint: {checkpoint}"
        )
    scheduler_state = torch.load(scheduler_path, map_location="cpu", weights_only=True)
    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    report = assess_scheduler_state(
        scheduler_state, optimizer_state, near_zero=near_zero,
        near_zero_fraction=near_zero_fraction
    )
    return {
        **report,
        "checkpoint": str(checkpoint),
        "scheduler_file": str(scheduler_path),
        "optimizer_file": str(optimizer_path),
    }


def linear_lr_used_for_update(
    base_lr: float, total_steps: int, update_number: int
) -> float:
    """Expected LR used by a zero-warmup linear schedule for a 1-based update."""
    if base_lr <= 0 or total_steps <= 0:
        raise ValueError("base_lr and total_steps must be positive")
    if not 1 <= update_number <= total_steps:
        raise ValueError("update_number must be in [1, total_steps]")
    return base_lr * (total_steps - update_number + 1) / total_steps
