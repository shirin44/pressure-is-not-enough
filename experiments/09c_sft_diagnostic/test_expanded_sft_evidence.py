"""Golden-record regression tests against the expanded-bank SFT run's evidence
(aws_runs/stage9c-sft-expanded-v1/, 2026-08-31). Captures BOTH halves of the finding:
the reported headline metric (genuine_correct_among_nonliteral) looks only modestly
improved over Stage 9c's 28.6% baseline, but tracing it down reveals the underlying
state-tracking skill is fully fixed (100%, verified against ground truth) -- the
modest reported number is an artifact of a separate, narrow bug (majority-class
collapse to "Tails" at the final-answer-translation step), not evidence that tracking
itself remains weak. Both halves are asserted so neither can silently disappear from
the record."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from reward_v3 import parse_state_slots, normalize_state_token
from synthetic_bridge import _trace

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9c-sft-expanded-v1'
                        / 'stage9c_sft_expanded.json').read_text())
STAGE9C_BASELINE = 0.2857142857142857


def test_expanded_bank_config_matches_the_build():
    cfg = EVIDENCE['config']
    assert cfg['expanded_bank_size'] == 43
    assert cfg['expanded_bank_coverage'] == '43/64'
    assert cfg['original_bank_size_for_comparison'] == 16
    assert cfg['clean21_eval_sha256'] == '947260ebc7bba7584b39839c8b4d248a2aa1612a9e049f46128e0ed3901e245e'


def test_training_completed_without_oom_this_time():
    telemetry = EVIDENCE['training_telemetry']
    assert len(telemetry) == 40
    assert all(__import__('math').isfinite(r['loss']) for r in telemetry)


def test_reported_headline_metric_is_only_modestly_above_baseline():
    # The literal, as-defined metric the task asked for -- correctly computed, but (per
    # the next test) misleading in isolation.
    tier_b = EVIDENCE['tier_b_heldout_same_pair']['summary']
    assert tier_b['nonliteral_code_usage_rate'] == 1.0
    gc = tier_b['genuine_correct_among_nonliteral']
    assert abs(gc - 1 / 3) < 1e-9
    assert gc > STAGE9C_BASELINE  # a real, if modest, increase over 28.6%
    assert gc < STAGE9C_BASELINE * 2  # NOT the substantial jump the task was hoping for, at face value


def test_held_out_completions_collapse_to_always_answering_tails():
    # The mechanism behind the modest headline number: the model answers "Tails" on
    # EVERY held-out completion, regardless of what it actually tracked.
    tier_b_samples = EVIDENCE['tier_b_heldout_same_pair']['samples']
    assert len(tier_b_samples) == 21
    answers = Counter()
    for s in tier_b_samples:
        m = re.search(r'<answer>(.*?)</answer>', s['completion'])
        answers[m.group(1) if m else 'MISSING'] += 1
    assert answers == Counter({'Tails': 21}), (
        f'expected a complete collapse to "Tails" on every held-out completion; got {answers}')


def test_held_out_intermediate_state_tracking_is_actually_100_percent_correct():
    # The finding that reframes the whole result: verified against GROUND TRUTH (not
    # just self-consistency), every held-out completion's intermediate Nib/Nomo
    # sequence exactly matches the true expected trace -- genuine, fully-generalized
    # state tracking, a complete fix of Stage 9c's original weakness. The modest 33.3%
    # headline number is NOT evidence tracking is still weak -- it's masked by a
    # separate, narrow answer-translation bug.
    tier_b_samples = EVIDENCE['tier_b_heldout_same_pair']['samples']
    correct_tracking = 0
    last_token_matches_truth = 0
    for r in tier_b_samples:
        true_states = _trace(r['starting_state'], r['operations'])
        expected_tokens = [normalize_state_token('Nib' if s == 'Heads' else 'Nomo') for s in true_states]
        actual_tokens = [t for _i, t in parse_state_slots(r['completion'])]
        if actual_tokens == expected_tokens:
            correct_tracking += 1
        if actual_tokens and actual_tokens[-1] == expected_tokens[-1]:
            last_token_matches_truth += 1
    assert correct_tracking == 21, (
        f'expected 21/21 held-out completions to exactly match the true expected trace; got {correct_tracking}/21')
    assert last_token_matches_truth == 21
    # If the model had correctly translated its own (perfectly correct) tracked state
    # into the answer instead of always saying "Tails", held-out correctness would be
    # 100%, not 33.3% -- the gap is entirely attributable to the translation bug.


def test_training_set_shows_the_same_collapse_not_just_held_out():
    tier_a_samples = EVIDENCE['tier_a_training_set_recall']['samples']
    assert len(tier_a_samples) == 43
    assert EVIDENCE['tier_a_training_set_recall']['summary']['exact_recall_rate'] == 1.0
    answers = Counter()
    for s in tier_a_samples:
        m = re.search(r'<answer>(.*?)</answer>', s['completion'])
        answers[m.group(1) if m else 'MISSING'] += 1
    assert answers == Counter({'Tails': 43}), (
        'expected the SAME complete collapse on the training set too -- not a held-out-specific artifact')


def test_stage9c_original_training_set_did_not_show_this_collapse():
    # Comparison point: the original, perfectly-balanced (8/8) 16-example bank did NOT
    # produce this collapse -- final_answer_accuracy=1.0 on its own training set.
    # Supports the hypothesis that the expanded bank's mild 58/42 Tails/Heads imbalance
    # (not present in the original bank) is what tipped the model into the majority-
    # class shortcut specifically at the answer-translation step.
    original = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9c-sft-diagnostic-v1'
                            / 'stage9c_sft_diagnostic.json').read_text())
    assert original['tier_a_training_set_recall']['summary']['final_answer_accuracy'] == 1.0


if __name__ == '__main__':
    import inspect
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
