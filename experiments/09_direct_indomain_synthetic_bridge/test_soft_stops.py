from soft_stops import SoftStopTracker


def test_non_finite_reward_stops_immediately():
    t = SoftStopTracker()
    assert t.record_step(mean_word_count=50, accuracy=0.8, all_rewards_finite=False) == "non_finite_reward"


def test_no_stop_under_normal_conditions():
    t = SoftStopTracker()
    for _ in range(20):
        assert t.record_step(mean_word_count=50, accuracy=0.8, all_rewards_finite=True) is None


def test_length_floor_breach_requires_consecutive_steps():
    t = SoftStopTracker(absolute_word_floor=14, floor_consecutive_steps=2)
    # One breach alone must not stop.
    assert t.record_step(mean_word_count=10, accuracy=0.8, all_rewards_finite=True) is None
    # A healthy step in between resets the counter.
    assert t.record_step(mean_word_count=50, accuracy=0.8, all_rewards_finite=True) is None
    assert t.record_step(mean_word_count=10, accuracy=0.8, all_rewards_finite=True) is None
    # Two in a row must stop.
    assert t.record_step(mean_word_count=10, accuracy=0.8, all_rewards_finite=True) == "length_floor_breach"


def test_length_floor_exactly_at_floor_does_not_count_as_breach():
    t = SoftStopTracker(absolute_word_floor=14, floor_consecutive_steps=2)
    # mean_word_count == floor is NOT below it -- must not count as a breach.
    assert t.record_step(mean_word_count=14, accuracy=0.8, all_rewards_finite=True) is None
    assert t.record_step(mean_word_count=14, accuracy=0.8, all_rewards_finite=True) is None


def test_accuracy_collapse_not_evaluated_before_trend_min_observations():
    t = SoftStopTracker(trend_min_observations=10, severe_accuracy_window=10, severe_accuracy_threshold=0.1)
    for _ in range(9):
        assert t.record_step(mean_word_count=50, accuracy=0.0, all_rewards_finite=True) is None
    # 10th observation: now enough history to evaluate the trend.
    assert t.record_step(mean_word_count=50, accuracy=0.0, all_rewards_finite=True) == "accuracy_collapse"


def test_accuracy_collapse_uses_trailing_window_not_full_history():
    t = SoftStopTracker(trend_min_observations=10, severe_accuracy_window=10, severe_accuracy_threshold=0.1)
    # First 10 steps: perfect accuracy (establishes history, no collapse).
    for _ in range(10):
        assert t.record_step(mean_word_count=50, accuracy=1.0, all_rewards_finite=True) is None
    # Next 10 steps: zero accuracy -- the trailing window is now all zeros, must collapse.
    results = [t.record_step(mean_word_count=50, accuracy=0.0, all_rewards_finite=True) for _ in range(10)]
    assert "accuracy_collapse" in results


def test_severe_accuracy_threshold_is_a_mean_not_single_step():
    t = SoftStopTracker(trend_min_observations=10, severe_accuracy_window=10, severe_accuracy_threshold=0.1)
    # Alternate high/low so the windowed MEAN stays above threshold despite some zero steps.
    for i in range(20):
        acc = 1.0 if i % 2 == 0 else 0.0  # mean = 0.5, comfortably above 0.1
        result = t.record_step(mean_word_count=50, accuracy=acc, all_rewards_finite=True)
    assert result is None


def test_values_match_step500_training_config():
    t = SoftStopTracker()
    assert t.absolute_word_floor == 14
    assert t.floor_consecutive_steps == 2
    assert t.trend_min_observations == 10
    assert t.severe_accuracy_window == 10
    assert t.severe_accuracy_threshold == 0.1
