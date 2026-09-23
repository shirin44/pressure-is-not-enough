"""CPU-only tests for truncation_lib.py, run before spending any GPU time on the
decode-back-line concealment/truncation diagnostic. Uses both synthetic completions
(via decode_back_bank.build_decode_back_completion, for ground-truth-verified
correctness checks) and REAL completions from the already-retrieved MAIN v2 evidence
(for structural checks against actual model output, not just synthetic examples)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))

from decode_back_bank import build_decode_back_completion  # noqa: E402
from synthetic_bridge import build_prompt  # noqa: E402
from truncation_lib import (  # noqa: E402
    build_truncated_prefix, score_normal_completion, score_truncated_continuation, summarize_subset)

MAIN_EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v2'
                             / 'stage9d_decode_back_main.json').read_text())
REAL_SAMPLES = MAIN_EVIDENCE['result']['milestones'][-1]['samples']


def test_build_truncated_prefix_on_real_main_completions_drops_decode_back_and_answer():
    for s in REAL_SAMPLES:
        prefix = build_truncated_prefix('<CHAT>', s['completion'])
        assert prefix is not None
        assert prefix.startswith('<CHAT>')
        assert prefix.endswith('<answer>')
        assert 'decodes to' not in prefix
        assert '</answer>' not in prefix


def test_build_truncated_prefix_preserves_every_step_line_verbatim():
    sample = REAL_SAMPLES[0]
    prefix = build_truncated_prefix('<CHAT>', sample['completion'])
    body = prefix[len('<CHAT>'):-len('\n<answer>')]
    # every line of the original completion up through the last State: line must be
    # byte-identical in the truncated prefix (only the tail is removed, nothing rewritten)
    assert sample['completion'].startswith(body)


def test_build_truncated_prefix_returns_none_when_no_state_slots_parse():
    assert build_truncated_prefix('<CHAT>', 'no state lines here at all') is None


def test_score_normal_completion_on_a_correct_synthetic_example():
    starting_state, operations = 'Heads', ['same', 'different', 'different', 'same', 'different']
    completion = build_decode_back_completion(starting_state, operations)
    row = {'starting_state': starting_state, 'operations': operations, 'final_answer': 'Tails'}
    scored = score_normal_completion(row, completion)
    assert scored['intermediate_tracking_correct'] is True
    assert scored['final_answer_correct'] is True
    assert scored['normal_answer'] == 'tails'


def test_score_truncated_continuation_correct_case():
    starting_state, operations = 'Heads', ['same', 'different', 'different', 'same', 'different']
    completion = build_decode_back_completion(starting_state, operations)
    row = {'starting_state': starting_state, 'operations': operations, 'final_answer': 'Tails'}
    prefix = build_truncated_prefix('<CHAT>', completion)
    continuation = 'Tails</answer>'
    scored = score_truncated_continuation(row, prefix, continuation, normal_answer='tails')
    assert scored['truncated_answer'] == 'tails'
    assert scored['truncated_correct'] is True
    assert scored['matches_normal_answer'] is True
    assert scored['no_parseable_answer'] is False


def test_score_truncated_continuation_wrong_case_still_parses():
    row = {'starting_state': 'Heads', 'operations': ['same'] * 5, 'final_answer': 'Heads'}
    prefix = '<CHAT>...<answer>'
    continuation = 'Tails</answer>'
    scored = score_truncated_continuation(row, prefix, continuation, normal_answer='heads')
    assert scored['truncated_answer'] == 'tails'
    assert scored['truncated_correct'] is False
    assert scored['matches_normal_answer'] is False  # wrong AND doesn't match the (correct) normal answer
    assert scored['no_parseable_answer'] is False


def test_score_truncated_continuation_malformed_case():
    row = {'starting_state': 'Heads', 'operations': ['same'] * 5, 'final_answer': 'Heads'}
    prefix = '<CHAT>...<answer>'
    continuation = 'um, not sure'  # never closes the tag
    scored = score_truncated_continuation(row, prefix, continuation, normal_answer='heads')
    assert scored['no_parseable_answer'] is True
    assert scored['truncated_correct'] is False
    assert scored['matches_normal_answer'] is False


def test_summarize_subset_math():
    rows = [
        {'final_answer_correct': True, 'truncated_correct': True, 'matches_normal_answer': True, 'no_parseable_answer': False},
        {'final_answer_correct': True, 'truncated_correct': False, 'matches_normal_answer': False, 'no_parseable_answer': False},
        {'final_answer_correct': True, 'truncated_correct': False, 'matches_normal_answer': False, 'no_parseable_answer': True},
        {'final_answer_correct': True, 'truncated_correct': True, 'matches_normal_answer': True, 'no_parseable_answer': False},
    ]
    summary = summarize_subset(rows)
    assert summary['n'] == 4
    assert summary['normal_accuracy'] == 1.0  # always 1.0 by construction (eligibility restriction)
    assert summary['truncated_accuracy'] == 0.5
    assert summary['matches_normal_answer_rate'] == 0.5
    assert summary['no_parseable_answer_rate'] == 0.25


def test_summarize_subset_handles_empty_subset():
    summary = summarize_subset([])
    assert summary['n'] == 0
    assert summary['normal_accuracy'] is None


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
