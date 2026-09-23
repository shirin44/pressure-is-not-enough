"""Unit tests for src/data/unique_sampling.py -- the standing guard against the
draw-random-and-dedupe infinite-loop bug (hit twice: the Coin Flip bootstrap-generation
prompt pool, and Experiment 1's flip-length sweep)."""
import itertools

from src.data.unique_sampling import (
    cap_unique_sample_size,
    check_unique_sample_feasible,
    combinatorial_space_size,
    sample_unique,
)


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


def test_combinatorial_space_size_multiplies_dimensions():
    assert combinatorial_space_size(2, 4) == 8
    assert combinatorial_space_size(2, 2 ** 2) == 8  # Coin Flip at n_flips=2
    assert combinatorial_space_size(2, 2 ** 6) == 128  # Coin Flip at n_flips=6


def test_combinatorial_space_size_requires_at_least_one_dimension():
    assert _raises(ValueError, combinatorial_space_size)


def test_combinatorial_space_size_rejects_negative_dimension():
    assert _raises(ValueError, combinatorial_space_size, 2, -1)


def test_check_unique_sample_feasible_raises_when_infeasible():
    # Reproduces the exact Experiment 1 bug: 40 requested, only 8 available.
    assert _raises(ValueError, check_unique_sample_feasible, 40, combinatorial_space_size(2, 2 ** 2))


def test_check_unique_sample_feasible_passes_when_feasible():
    check_unique_sample_feasible(40, combinatorial_space_size(2, 2 ** 6))  # should not raise


def test_check_unique_sample_feasible_boundary_exact_equal_is_feasible():
    check_unique_sample_feasible(8, 8)  # exactly the full space -- not an error


def test_check_unique_sample_feasible_includes_context_in_message():
    try:
        check_unique_sample_feasible(100, 10, context="flip-length 2")
    except ValueError as e:
        assert "flip-length 2" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_cap_unique_sample_size_caps_when_too_large():
    assert cap_unique_sample_size(40, 8) == 8


def test_cap_unique_sample_size_passes_through_when_feasible():
    assert cap_unique_sample_size(40, 128) == 40


def test_sample_unique_returns_exact_count_when_feasible():
    scenarios = list(range(100))
    result = sample_unique(scenarios, 40, seed=1)
    assert len(result) == 40
    assert len(set(result)) == 40  # all unique
    assert set(result).issubset(set(scenarios))


def test_sample_unique_caps_without_hanging_when_infeasible():
    # This is the exact scenario that used to hang forever: request more than exists.
    scenarios = list(range(8))
    result = sample_unique(scenarios, 40, seed=1)
    assert len(result) == 8  # capped, not padded or repeated
    assert set(result) == set(scenarios)


def test_sample_unique_deterministic_for_fixed_seed():
    scenarios = list(range(50))
    a = sample_unique(scenarios, 20, seed=42)
    b = sample_unique(scenarios, 20, seed=42)
    assert a == b


def test_sample_unique_matches_experiment1_coinflip_scenario_space():
    # End-to-end reproduction of the actual bug: Coin Flip (starting_state, operations)
    # space at flip-length 2 has only 8 distinct scenarios.
    scenarios = [(s, tuple(ops)) for s in ('Heads', 'Tails')
                 for ops in itertools.product(('same', 'different'), repeat=2)]
    assert len(scenarios) == 8
    check_unique_sample_feasible(8, len(scenarios))  # exactly fits, fine
    assert _raises(ValueError, check_unique_sample_feasible, 40, len(scenarios))  # the actual bug's numbers
    result = sample_unique(scenarios, 40, seed=1)
    assert len(result) == 8
    assert len(set(result)) == 8
