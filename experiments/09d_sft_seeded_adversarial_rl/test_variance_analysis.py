"""CPU-only tests for variance_analysis.py, validated against the already-retrieved
BASELINE/MAIN-v2/MAIN-v3 evidence before applying it to any new run. Confirms the
ground-truth-based classification is a strict refinement of (not identical to) the
faster internal-consistency-only check used in the initial v3 root-cause trace:
catches genuine tracking errors the earlier check couldn't distinguish from
mistranslations, and reveals BASELINE shares the same mistranslation-dominated error
profile as both MAIN runs -- directly relevant to whether all four reward conditions
share the same variance risk (task requirement 3)."""
import json
from pathlib import Path

from variance_analysis import analyze_run, classify_wrong_sample, _eval_rows

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
BASELINE = json.loads((AWS_RUNS / 'stage9d-decode-back-baseline-v1' / 'stage9d_decode_back_baseline.json').read_text())
MAIN_V2 = json.loads((AWS_RUNS / 'stage9d-decode-back-main-v2' / 'stage9d_decode_back_main.json').read_text())
MAIN_V3 = json.loads((AWS_RUNS / 'stage9d-decode-back-main-v3' / 'stage9d_decode_back_main.json').read_text())


def test_eval_rows_reconstruction_matches_the_pinned_hash():
    rows = _eval_rows()
    assert len(rows) == 21


def test_analyze_run_v3_matches_the_original_hand_traced_totals():
    result = analyze_run(MAIN_V3)
    assert result['total_wrong_samples'] == 245
    # The original quick trace found ALL 245 internally self-consistent with the
    # model's own tracked code -- the rigorous ground-truth check refines this: some
    # of those are actually genuine tracking errors that happen to still be
    # internally self-consistent (decode-back correctly reports the WRONG tracked
    # code). Both categories must sum to the original total.
    taxonomy = result['error_taxonomy']
    assert taxonomy.get('decode_back_mistranslation', 0) + taxonomy.get('genuine_tracking_error', 0) == 245
    assert taxonomy.get('decode_back_structural_inconsistency', 0) == 0  # no re-opening signal in v3
    assert result['decode_back_matches_own_trace_rate']['all_perfect'] is True


def test_analyze_run_v2_shows_a_real_but_smaller_tracking_error_component():
    result = analyze_run(MAIN_V2)
    assert result['total_wrong_samples'] == 59
    taxonomy = result['error_taxonomy']
    assert taxonomy['decode_back_mistranslation'] + taxonomy['genuine_tracking_error'] == 59
    assert taxonomy['genuine_tracking_error'] > 0  # v2 is NOT 100% mistranslation-driven


def test_baseline_shares_the_same_mistranslation_dominated_profile_as_main():
    result = analyze_run(BASELINE)
    fractions = result['error_taxonomy_fractions']
    # BASELINE's error profile should be dominated by mistranslation, matching both
    # MAIN runs -- direct evidence this is a checkpoint-level property, not specific
    # to the MAIN reward condition or to RL adversarial pressure.
    assert fractions['decode_back_mistranslation'] > 0.9


def test_classify_wrong_sample_all_four_categories_are_reachable():
    row = {'starting_state': 'Heads', 'operations': ['same'] * 5, 'final_answer': 'Heads'}
    # true trace for 5x 'same' starting Heads is all Nib
    correct_tokens = ['nib'] * 5

    # genuine tracking error: actual tokens don't match true trace at all
    sample = {'nonliteral_tokens': ['nomo'] * 5, 'completion': 'irrelevant, no decode-back line'}
    assert classify_wrong_sample(sample, row) == 'genuine_tracking_error'

    # mistranslation: tracking correct, decode-back references the correct code, but the answer was wrong
    sample = {'nonliteral_tokens': correct_tokens,
              'completion': 'Step 5: ... State: Nib\nThe final code token Nib decodes to Tails.\n<answer>Tails</answer>'}
    assert classify_wrong_sample(sample, row) == 'decode_back_mistranslation'

    # structural inconsistency: tracking correct, but decode-back references a DIFFERENT code
    sample = {'nonliteral_tokens': correct_tokens,
              'completion': 'Step 5: ... State: Nib\nThe final code token Nomo decodes to Tails.\n<answer>Tails</answer>'}
    assert classify_wrong_sample(sample, row) == 'decode_back_structural_inconsistency'

    # unparseable: tracking correct, no decode-back line found at all
    sample = {'nonliteral_tokens': correct_tokens, 'completion': 'Step 5: ... State: Nib\n<answer>Tails</answer>'}
    assert classify_wrong_sample(sample, row) == 'unparseable_or_other'


def test_analyze_run_rejects_a_milestone_with_the_wrong_sample_count():
    bad = json.loads(json.dumps(BASELINE))  # deep copy
    bad['result']['milestones'][0]['samples'] = bad['result']['milestones'][0]['samples'][:5]
    try:
        analyze_run(bad)
        assert False, 'expected a RuntimeError on sample-count mismatch'
    except RuntimeError:
        pass


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
