"""Soft-stop circuit breakers for Step-0 training.

VALUES are reused exactly from step-500/training_config.json's "safety" section (the
project's original Stage 01 direct-RL training config):

    {"absolute_word_floor": 14, "floor_consecutive_steps": 2, "trend_min_observations": 10,
     "severe_accuracy_window": 10, "severe_accuracy_threshold": 0.1}

IMPORTANT, flagged explicitly: the ENFORCEMENT CODE that originally used these values could
not be located anywhere in this project's accessible history (searched the EC2 instance's
full home directory and aisi_checkpoints tree; only reward_function.py -- the reward
computation itself, an even simpler 3-term version than reward_v3.py -- was saved alongside
step-500, not a training-loop callback). What follows is a good-faith reconstruction of the
enforcement logic from the parameter names and values, NOT a verified byte-identical reuse
like the KL clamp (which was diffed line-for-line against the proven kl_clamp_mitigation_v2
run). Reviewed here, not assumed correct by analogy.

These are SEPARATE from, and layered on top of, the existing hard breakers (grad_norm>=50,
kl>=5 -- unchanged, reused verbatim from Stage 04/07/08) and the KL clamp (also unchanged).
"Soft" here means they require accumulated evidence across multiple steps before firing,
rather than an immediate single-step threshold breach.
"""
from __future__ import annotations

import statistics

# Reused exactly from step-500/training_config.json's "safety" section.
ABSOLUTE_WORD_FLOOR = 14
FLOOR_CONSECUTIVE_STEPS = 2
TREND_MIN_OBSERVATIONS = 10
SEVERE_ACCURACY_WINDOW = 10
SEVERE_ACCURACY_THRESHOLD = 0.1


class SoftStopTracker:
    """Call record_step() once per training step with that step's aggregate stats.
    Returns a stop reason string the first time a soft-stop condition fires, else None."""

    def __init__(
        self,
        absolute_word_floor: int = ABSOLUTE_WORD_FLOOR,
        floor_consecutive_steps: int = FLOOR_CONSECUTIVE_STEPS,
        trend_min_observations: int = TREND_MIN_OBSERVATIONS,
        severe_accuracy_window: int = SEVERE_ACCURACY_WINDOW,
        severe_accuracy_threshold: float = SEVERE_ACCURACY_THRESHOLD,
    ):
        self.absolute_word_floor = absolute_word_floor
        self.floor_consecutive_steps = floor_consecutive_steps
        self.trend_min_observations = trend_min_observations
        self.severe_accuracy_window = severe_accuracy_window
        self.severe_accuracy_threshold = severe_accuracy_threshold

        self.step_mean_word_counts: list[float] = []
        self.step_accuracies: list[float] = []
        self._consecutive_floor_breaches = 0

    def record_step(self, mean_word_count: float, accuracy: float, all_rewards_finite: bool) -> str | None:
        if not all_rewards_finite:
            return "non_finite_reward"

        self.step_mean_word_counts.append(mean_word_count)
        self.step_accuracies.append(accuracy)

        # Length-floor breach: mean completion word count below the floor for
        # `floor_consecutive_steps` steps IN A ROW.
        if mean_word_count < self.absolute_word_floor:
            self._consecutive_floor_breaches += 1
        else:
            self._consecutive_floor_breaches = 0
        if self._consecutive_floor_breaches >= self.floor_consecutive_steps:
            return "length_floor_breach"

        # Accuracy collapse: only evaluated once there's enough history to trust the trend
        # (trend_min_observations), then checked over a trailing window
        # (severe_accuracy_window) against a hard floor (severe_accuracy_threshold).
        if len(self.step_accuracies) >= self.trend_min_observations:
            window = self.step_accuracies[-self.severe_accuracy_window:]
            if statistics.fmean(window) < self.severe_accuracy_threshold:
                return "accuracy_collapse"

        return None
