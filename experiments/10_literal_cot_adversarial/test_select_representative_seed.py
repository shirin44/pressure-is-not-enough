"""Locks in the pre-declared selection rule from select_representative_seed.py so it cannot
drift after real results exist."""
from select_representative_seed import select_representative_main_seed, welch_t_test


def test_four_distinct_scores_picks_lower_of_two_middle():
    r = select_representative_main_seed({42: 0.60, 43: 0.90, 44: 0.75, 45: 0.95})
    assert r['selected_seed'] == 44 and r['selected_score'] == 0.75 and not r['tie_broken']


def test_exact_middle_tie_uses_lower_seed_number():
    r = select_representative_main_seed({42: 0.60, 45: 0.90, 44: 0.90, 43: 0.95})
    assert r['selected_seed'] == 44 and r['tie_broken'] is True


def test_llama_like_case_seed43_and_seed45_tie_at_the_TOP_does_not_affect_this_rule():
    # Reproduces the exact numbers that caused the Llama 09e ambiguity (43 and 45 tied for
    # BEST, not median) -- confirms this rule never even looks at the top tie, since it
    # selects by median, not max.
    r = select_representative_main_seed({42: 0.905, 43: 1.0, 44: 0.810, 45: 1.0})
    assert r['selected_seed'] == 42  # lower of the two middle scores: 0.810 (44) vs 0.905 (42) -> 42 is lower-middle
    assert r['tie_broken'] is False


def test_rejects_anything_other_than_exactly_four_seeds():
    import pytest
    try:
        select_representative_main_seed({1: 0.1, 2: 0.2, 3: 0.3})
        assert False, 'should have raised'
    except ValueError:
        pass


def test_welch_t_test_matches_known_llama_baseline_vs_main_numbers():
    # Cross-check against the already-verified Stage 9e Llama result (not this experiment's
    # own data) -- confirms this reused implementation reproduces a known-correct answer.
    main = [0.9047619047619048, 1.0, 0.8095238095238095, 1.0]
    baseline = [0.8571428571428571, 0.7619047619047619, 0.5238095238095238, 0.9523809523809523]
    r = welch_t_test(main, baseline)
    assert abs(r['t_statistic'] - 1.5078271124027665) < 1e-9
    assert abs(r['degrees_of_freedom'] - 4.390823203932072) < 1e-9


if __name__ == '__main__':
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t(); print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append(t.__name__); print(f'FAILED: {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
