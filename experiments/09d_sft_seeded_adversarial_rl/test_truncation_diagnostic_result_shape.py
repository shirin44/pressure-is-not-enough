"""Regression test for a real integration bug found while running the truncation
diagnostic for the first time (2026-09-06): truncation_lib.py's summarize_subset()
was well-tested in isolation (test_truncation_lib.py) against hand-built dicts using
the correct key names, but truncation_diagnostic.py's own results-dict-building code
(GPU-only, cannot run in a CPU test) used a DIFFERENT key name
('normal_final_answer_correct' instead of 'final_answer_correct'), which nothing
caught until the actual GPU run crashed with KeyError at STEP 4. Fixed by renaming to
match score_normal_completion()'s own output key and what summarize_subset() expects.

This test closes the actual gap: it doesn't run the GPU script, but it does assert,
via source inspection, that the exact dict literal built in truncation_diagnostic.py
contains every key summarize_subset() requires -- so a future edit to either file
that reintroduces a name mismatch fails a CPU-only test before wasting a GPU launch."""
import re
from pathlib import Path

from truncation_lib import summarize_subset

SCRIPT = Path(__file__).resolve().parent / 'truncation_diagnostic.py'
SOURCE = SCRIPT.read_text()

REQUIRED_KEYS_FOR_SUMMARIZE_SUBSET = {
    'final_answer_correct', 'truncated_correct', 'matches_normal_answer', 'no_parseable_answer',
}


def _results_append_block():
    m = re.search(r'results\.append\(\{(.*?)\}\)', SOURCE, re.DOTALL)
    assert m is not None, 'results.append({...}) block not found -- has the script structure changed?'
    return m.group(1)


def test_results_dict_literal_contains_every_key_summarize_subset_requires():
    block = _results_append_block()
    found_keys = set(re.findall(r"'(\w+)':", block))
    missing = REQUIRED_KEYS_FOR_SUMMARIZE_SUBSET - found_keys
    assert not missing, f'results dict is missing keys summarize_subset() needs: {missing}'


def test_the_specific_previously_buggy_key_name_is_gone():
    block = _results_append_block()
    assert 'normal_final_answer_correct' not in block


def test_summarize_subset_accepts_a_row_shaped_like_the_scripts_actual_output():
    # Builds a row using exactly the keys truncation_diagnostic.py's results.append
    # block produces (per the source-extracted key set above), confirming
    # summarize_subset() can consume it without KeyError -- the actual failure mode
    # from the crashed run.
    block = _results_append_block()
    found_keys = set(re.findall(r"'(\w+)':", block))
    fake_row = {k: (True if 'correct' in k or 'matches' in k or 'parseable' in k else 'x') for k in found_keys}
    summary = summarize_subset([fake_row])
    assert summary['n'] == 1
    assert summary['normal_accuracy'] == 1.0


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
