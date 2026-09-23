"""Golden-record regression test against the real Llama-3-8B-Instruct zero-shot
Same/Different capacity check (aws_runs/stage9e-llama-zero-shot-capacity-v1/,
2026-09-07), Stage 9e Decision 2. Plain Heads/Tails, no code substitution, no
training -- the untouched Instruct model, zero-shot, on the existing 16-scenario
balanced bank.

Decisive, nuanced finding: overall accuracy (0.5625) is near chance, formally
triggering the task's own stop condition ("If it's low (e.g. near chance), STOP and
report"). But the failure is NOT a raw-computation deficit -- 15/16 completions
(93.75%) show PERFECT intermediate state tracking (the model's own emitted trace
exactly matches the true trace at every step), yet 6 of those 15 still land on the
wrong final Same/Different answer. The model can execute the mechanical forward
tracking almost flawlessly zero-shot; the failure is narrowly at the final
relational-comparison step (comparing the last tracked state against the DECLARED
starting state stated several lines earlier), where it defaults to 'Same' in 14/16
cases regardless of ground truth. This is a materially different diagnosis than "the
relation is too complex for the model to compute," and is reported precisely, not
just as a single failing accuracy number."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from synthetic_bridge import _trace  # noqa: E402

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-llama-zero-shot-capacity-v1' / 'stage9e_llama_zero_shot_capacity.json').read_text())


def test_ran_sixteen_balanced_scenarios():
    results = EVIDENCE['results']
    assert len(results) == 16
    truths = [r['truth'] for r in results]
    assert truths.count('Same') == 8
    assert truths.count('Different') == 8


def test_overall_accuracy_matches_measured_value_and_is_near_chance():
    a = EVIDENCE['analysis']
    assert abs(a['accuracy'] - 0.5625) < 1e-9
    assert a['near_chance_flag'] is True


def test_accuracy_is_lopsided_by_truth_label_not_uniformly_near_chance():
    a = EVIDENCE['analysis']
    assert a['same_truth_accuracy'] == 1.0
    assert a['different_truth_accuracy'] == 0.125


def test_almost_all_completions_show_perfect_intermediate_tracking():
    results = EVIDENCE['results']
    n_tracking_correct = 0
    for r in results:
        expected = [s.lower() for s in _trace(r['starting_state'], r['operations'])]
        if r['intermediate_tokens'] == expected:
            n_tracking_correct += 1
    assert n_tracking_correct == 15


def test_most_wrong_answers_come_from_perfect_tracking_with_a_comparison_failure():
    results = EVIDENCE['results']
    n_perfect_tracking_wrong_answer = 0
    for r in results:
        expected = [s.lower() for s in _trace(r['starting_state'], r['operations'])]
        if r['intermediate_tokens'] == expected and not r['correct']:
            n_perfect_tracking_wrong_answer += 1
    assert n_perfect_tracking_wrong_answer == 6


def test_model_defaults_to_same_regardless_of_truth():
    results = EVIDENCE['results']
    n_extracted_same = sum(1 for r in results if r['extracted_answer'] == 'same')
    assert n_extracted_same == 15  # 15/16 (93.75%) -- extracted 'Same' almost every time


def test_all_completions_were_format_valid_not_a_formatting_failure():
    a = EVIDENCE['analysis']
    assert a['format_valid_rate'] == 1.0


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
