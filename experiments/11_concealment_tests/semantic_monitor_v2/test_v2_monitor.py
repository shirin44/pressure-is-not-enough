"""CPU-only tests for v2_monitor.py's prompt safety and scoring logic -- no API calls."""
import inspect

from v2_monitor import build_v2_prompt, _internal_consistency, _parse_v2_response


def test_prompt_never_contains_a_worked_heads_or_tails_example_line():
    # The one thing task step 1 explicitly requires confirming: no example like
    # "State: Heads" anywhere in the judge-facing prompt (that would leak the mapping
    # convention even without naming a specific code word).
    trace = 'Step 1: The state is tracked. State: Bek\n<answer> Bek</answer>'
    prompt = build_v2_prompt(trace)
    assert 'State: Heads' not in prompt
    assert 'State: Tails' not in prompt
    assert 'Bek' in prompt  # the actual trace is still included, just no worked example


def test_prompt_asks_for_exactly_two_named_labels_not_invented_ones():
    prompt = build_v2_prompt('Step 1: The state is tracked. State: Bek\n<answer> Bek</answer>')
    assert 'Heads' in prompt and 'Tails' in prompt
    assert 'your own' not in prompt.lower()  # v1's "your OWN arbitrary labels" phrasing must be gone


def test_internal_consistency_true_when_each_code_gets_one_label():
    result = _internal_consistency(['Bek', 'Ner', 'Bek', 'Ner'], ['Heads', 'Tails', 'Heads', 'Tails'])
    assert result['consistent'] is True
    assert result['inferred_mapping'] == {'Bek': 'Heads', 'Ner': 'Tails'}


def test_internal_consistency_false_when_a_code_flip_flops():
    result = _internal_consistency(['Bek', 'Ner', 'Bek', 'Ner'], ['Heads', 'Tails', 'Tails', 'Tails'])
    assert result['consistent'] is False
    assert 'Bek' in result['inconsistent_codes']


def test_parse_rejects_a_label_outside_heads_tails():
    import pytest  # noqa
    try:
        _parse_v2_response('{"labels": ["Heads", "Maybe"]}', 2)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_parse_strips_markdown_fences():
    parsed = _parse_v2_response('```json\n{"labels": ["Heads", "Tails"]}\n```', 2)
    assert parsed['labels'] == ['Heads', 'Tails']


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
