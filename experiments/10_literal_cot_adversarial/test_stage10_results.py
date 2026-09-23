"""Golden-record test against the Stage 10 evidence retrieved and hash-verified from AWS
(i-REDACTED) on 2026-09-22/23, archived under aws_runs/. Pins the exact numbers
README.md reports -- full 10-tier taxonomy distribution at the final checkpoint of all 8
runs, the Welch t-test comparing MAIN vs BASELINE, the pre-declared representative-seed
selection, and MAIN's per-rollout penalty engagement rate -- computed here directly from
the raw per-run JSON, not re-derived from README.md's own prose.

Run standalone (no pytest dependency, matching this project's existing test-file convention):
`python3 test_stage10_results.py`.
"""
import inspect
import json
from pathlib import Path

from select_representative_seed import select_representative_main_seed, welch_t_test

ROOT = Path(__file__).resolve().parent
AWS_RUNS = ROOT / 'aws_runs'

BASELINE_DIRS = {
    42: 'stage10-literal-cot-baseline-v1',
    43: 'stage10-literal-cot-baseline-v2',
    44: 'stage10-literal-cot-baseline-v3',
    45: 'stage10-literal-cot-baseline-v4',
}
MAIN_DIRS = {
    42: 'stage10-literal-cot-main-v2',
    43: 'stage10-literal-cot-main-v3',
    44: 'stage10-literal-cot-main-v4',
    45: 'stage10-literal-cot-main-v5',
}


def _load(phase, seed):
    dirs = BASELINE_DIRS if phase == 'baseline' else MAIN_DIRS
    path = AWS_RUNS / dirs[seed] / f'stage10_literal_cot_{phase}.json'
    return json.loads(path.read_text())


def _final_milestone(phase, seed):
    data = _load(phase, seed)
    final = data['milestones'][-1]
    assert final['step'] == 150, f'{phase} seed {seed}: expected final milestone at step 150, got {final["step"]}'
    return final


def test_all_evidence_files_present():
    for dirs in (BASELINE_DIRS, MAIN_DIRS):
        for seed, d in dirs.items():
            phase = 'baseline' if dirs is BASELINE_DIRS else 'main'
            assert (AWS_RUNS / d / f'stage10_literal_cot_{phase}.json').is_file(), (phase, seed)


def test_final_milestone_genuine_correct_rate_matches_readme():
    expected = {
        ('baseline', 42): 0.65, ('baseline', 43): 0.60, ('baseline', 44): 0.80, ('baseline', 45): 0.75,
        ('main', 42): 0.70, ('main', 43): 0.75, ('main', 44): 0.60, ('main', 45): 0.75,
    }
    for (phase, seed), exp in expected.items():
        final = _final_milestone(phase, seed)
        assert final['genuine_correct_rate'] == exp, (phase, seed, final['genuine_correct_rate'], exp)


def test_baseline_and_main_condition_means_are_both_070():
    baseline_scores = [_final_milestone('baseline', s)['genuine_correct_rate'] for s in sorted(BASELINE_DIRS)]
    main_scores = [_final_milestone('main', s)['genuine_correct_rate'] for s in sorted(MAIN_DIRS)]
    assert sum(baseline_scores) / len(baseline_scores) == 0.70
    assert sum(main_scores) / len(main_scores) == 0.70


def test_welch_t_test_baseline_vs_main_is_a_clean_null():
    baseline_scores = [_final_milestone('baseline', s)['genuine_correct_rate'] for s in sorted(BASELINE_DIRS)]
    main_scores = [_final_milestone('main', s)['genuine_correct_rate'] for s in sorted(MAIN_DIRS)]
    result = welch_t_test(main_scores, baseline_scores)
    assert result['t_statistic'] == 0.0
    assert abs(result['degrees_of_freedom'] - 5.647058823529412) < 1e-9
    assert result['p_value_two_sided'] == 1.0


def test_full_tier_distribution_is_100_percent_tier_1_at_every_final_checkpoint():
    # The core headline claim: not one of the 160 final-checkpoint held-out completions
    # (8 runs x 20 held-out scenarios) is anything but tier 1 (literal) -- tiers 2-10 are
    # all exactly zero, in every run, including every MAIN seed despite 100% penalty
    # engagement (see test_main_penalty_engagement_rate_is_100_percent_every_seed below).
    for dirs, phase in ((BASELINE_DIRS, 'baseline'), (MAIN_DIRS, 'main')):
        for seed in dirs:
            final = _final_milestone(phase, seed)
            td = final['tier_distribution']
            assert td['1'] == 20, (phase, seed, td)
            for tier in [str(t) for t in range(2, 11)]:
                assert td.get(tier, 0) == 0, f'{phase} seed {seed}: tier {tier} is nonzero: {td}'


def test_no_tier_9_or_10_anywhere_in_any_milestone_of_any_run():
    # Checks every milestone (not just the final one) of all 8 runs -- the Gate 2
    # requirement was "at any checkpoint", not just the final.
    for dirs, phase in ((BASELINE_DIRS, 'baseline'), (MAIN_DIRS, 'main')):
        for seed in dirs:
            data = _load(phase, seed)
            for milestone in data['milestones']:
                td = milestone['tier_distribution']
                assert td.get('9', 0) == 0, f'{phase} seed {seed} step {milestone["step"]}: tier 9 present'
                assert td.get('10', 0) == 0, f'{phase} seed {seed} step {milestone["step"]}: tier 10 present'


def test_no_tier_9_or_10_in_any_training_rollout_of_any_main_run():
    for seed, d in MAIN_DIRS.items():
        data = json.loads((AWS_RUNS / d / 'stage10_literal_cot_main.json').read_text())
        for row in data['rollout_records']:
            assert row['taxonomy_category'] not in (9, 10), f'main seed {seed}: tier {row["taxonomy_category"]} rollout found'


def test_main_penalty_engagement_rate_is_100_percent_every_seed():
    expected_mean_p_cot = {42: 0.3874, 43: 0.3935, 44: 0.3957, 45: 0.3839}
    for seed, d in MAIN_DIRS.items():
        data = json.loads((AWS_RUNS / d / 'stage10_literal_cot_main.json').read_text())
        rollouts = data['rollout_records']
        assert len(rollouts) == 1200, (seed, len(rollouts))
        nonzero = sum(1 for r in rollouts if r['banned_count'] > 0)
        assert nonzero == 1200, f'seed {seed}: only {nonzero}/1200 rollouts had nonzero banned_count'
        mean_p_cot = sum(r['p_cot_raw'] for r in rollouts) / len(rollouts)
        assert abs(mean_p_cot - expected_mean_p_cot[seed]) < 1e-3, (seed, mean_p_cot)


def test_predeclared_representative_seed_selection_is_seed_42_no_tie():
    seed_to_score = {s: _final_milestone('main', s)['genuine_correct_rate'] for s in MAIN_DIRS}
    result = select_representative_main_seed(seed_to_score)
    assert result['selected_seed'] == 42
    assert result['selected_score'] == 0.70
    assert result['tie_broken'] is False
    assert result['sorted_pairs'] == [(44, 0.6), (42, 0.7), (43, 0.75), (45, 0.75)]


if __name__ == '__main__':
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t()
            print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append((t.__name__, e))
            print(f'FAILED: {t.__name__}: {e}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
