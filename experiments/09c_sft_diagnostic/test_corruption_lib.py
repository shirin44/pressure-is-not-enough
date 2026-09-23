"""CPU-only tests for corruption_lib.py, run against the ALREADY-RETRIEVED rebalanced-bank
evidence (aws_runs/stage9c-sft-rebalanced-v1/stage9c_sft_rebalanced.json) before spending any
GPU time on the corrupted-prefill diagnostic, per this project's established discipline."""
import json
from pathlib import Path

from corruption_lib import (
    CODE_LITERAL_MAP, CODE_TO_LITERAL_ANSWER, build_corrupted_prefix, classify_outcome,
    extract_answer, parse_state_slots_with_spans, select_eligible_samples,
)

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9c-sft-rebalanced-v1'
                        / 'stage9c_sft_rebalanced.json').read_text())
TIER_B = EVIDENCE['tier_b_heldout_same_pair']['samples']


def test_tier_b_has_21_samples_with_20_correctly_tracked():
    assert len(TIER_B) == 21
    n_correct = sum(1 for r in TIER_B if r['intermediate_tracking_correct'])
    assert n_correct == 20  # 95.2% of 21


def test_parse_state_slots_with_spans_recovers_all_five_steps_in_order():
    sample = next(r for r in TIER_B if r['intermediate_tracking_correct'])
    slots = parse_state_slots_with_spans(sample['completion'])
    assert [s['index'] for s in slots] == [1, 2, 3, 4, 5]
    assert all(s['token'] in ('nib', 'nomo') for s in slots)


def test_parse_state_slots_spans_exactly_bracket_the_state_line_text():
    sample = next(r for r in TIER_B if r['intermediate_tracking_correct'])
    slots = parse_state_slots_with_spans(sample['completion'])
    for s in slots:
        line_text = sample['completion'][s['line_start']:s['line_end']]
        assert line_text.startswith(f"Step {s['index']}:")
        assert 'State:' in line_text
        assert line_text.rstrip().endswith(s['token'].capitalize())


def test_extract_answer_matches_the_known_complete_tails_collapse():
    # Every stored Tier B completion's own (uncorrupted) answer is "Tails" -- the
    # documented finding this whole diagnostic exists to explain.
    for r in TIER_B:
        assert extract_answer(r['completion']) == 'tails'


def test_select_eligible_samples_splits_by_true_final_state_and_correct_tracking():
    groups = select_eligible_samples(TIER_B)
    assert set(groups.keys()) == {'heads_to_tails', 'tails_to_heads'}
    assert len(groups['heads_to_tails']) + len(groups['tails_to_heads']) == 20
    for r in groups['heads_to_tails']:
        assert r['_target_slot']['token'] == 'nib'
        assert r['intermediate_tracking_correct']
    for r in groups['tails_to_heads']:
        assert r['_target_slot']['token'] == 'nomo'
        assert r['intermediate_tracking_correct']


def test_select_eligible_samples_group_sizes_roughly_match_the_14_7_true_split():
    # Eval set is 14 true-Heads / 7 true-Tails; with 20/21 correctly tracked, expect group
    # sizes close to that ratio (allowing for wherever the single mistracked row falls).
    groups = select_eligible_samples(TIER_B)
    assert 12 <= len(groups['heads_to_tails']) <= 14
    assert 5 <= len(groups['tails_to_heads']) <= 7


def test_build_corrupted_prefix_flips_only_the_last_line_and_drops_everything_after():
    groups = select_eligible_samples(TIER_B)
    sample = groups['heads_to_tails'][0]
    slot = sample['_target_slot']
    prefix, direction = build_corrupted_prefix('<CHAT_PREFIX>', sample['completion'], slot)
    assert direction == 'tails'
    assert prefix.startswith('<CHAT_PREFIX>')
    assert '<answer>' not in prefix  # original answer tag must be dropped, not just the word
    assert prefix.rstrip().endswith('State: Nomo')
    # every earlier step line must be byte-identical to the original completion, and the
    # target line's reasoning text (everything before "State:") must be unchanged too
    body = prefix[len('<CHAT_PREFIX>'):]
    assert body.startswith(sample['completion'][:slot['line_start']])
    original_line = sample['completion'][slot['line_start']:slot['line_end']]
    reasoning_part = original_line.split('State:')[0]
    assert body.endswith(reasoning_part + 'State: Nomo')


def test_build_corrupted_prefix_opposite_direction_flips_nomo_to_nib():
    groups = select_eligible_samples(TIER_B)
    sample = groups['tails_to_heads'][0]
    prefix, direction = build_corrupted_prefix('<CHAT_PREFIX>', sample['completion'], sample['_target_slot'])
    assert direction == 'heads'
    assert prefix.rstrip().endswith('State: Nib')


def test_classify_outcome_all_three_paths():
    assert classify_outcome('tails', 'tails', 'tails') == 'tracks_corrupted_prefix'
    assert classify_outcome('heads', 'tails', 'heads') == 'tracks_corrupted_prefix'
    assert classify_outcome('tails', 'tails', 'heads') == 'stays_at_original_answer'
    assert classify_outcome(None, 'tails', 'heads') == 'no_parseable_answer'
    assert classify_outcome('nib', 'tails', 'heads') == 'other'


def test_code_maps_are_consistent_inverses():
    assert CODE_LITERAL_MAP == {'nib': 'nomo', 'nomo': 'nib'}
    for code, flipped in CODE_LITERAL_MAP.items():
        assert CODE_LITERAL_MAP[flipped] == code
    assert CODE_TO_LITERAL_ANSWER == {'nib': 'heads', 'nomo': 'tails'}


if __name__ == '__main__':
    import inspect
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
