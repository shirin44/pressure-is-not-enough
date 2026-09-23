"""Tests for token-position logging (task item 5), against Llama's real tokenizer
(local access, no GPU required for tokenization -- only the tokenizer is loaded, not
the model). meta-llama/Meta-Llama-3-8B-Instruct is a license-gated HF model: these
tests need an authenticated token with accepted terms (env HF_TOKEN, or a prior
`huggingface-cli login`) to download the tokenizer. Skipped, not failed, when no such
token is available -- e.g. in a CI job without the HF_TOKEN secret configured."""
import os

import pytest
from transformers import AutoTokenizer

from code_word_answer_bank import (
    ALL_SCENARIOS, HEADS_CODE, TAILS_CODE, build_code_word_completion, build_code_word_prompt,
)
from code_word_position_logging import log_code_word_positions

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
_tokenizer = None

pytestmark = pytest.mark.skipif(
    not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')),
    reason='needs an authenticated HF token with accepted Llama-3 license terms '
           '(HF_TOKEN/HUGGING_FACE_HUB_TOKEN not set)',
)


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def test_all_code_word_tokens_are_confirmed_single_token():
    tokenizer = _get_tokenizer()
    starting_state, operations = 'Heads', ['different', 'same', 'different', 'same', 'different']
    prompt = build_code_word_prompt(starting_state, operations)
    completion = build_code_word_completion(starting_state, operations)
    result = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
    assert result['all_occurrences_single_token_confirmed'] is True


def test_finds_exactly_six_occurrences_for_a_length_five_sequence():
    tokenizer = _get_tokenizer()
    starting_state, operations = 'Tails', ['same', 'same', 'different', 'different', 'same']
    prompt = build_code_word_prompt(starting_state, operations)
    completion = build_code_word_completion(starting_state, operations)
    result = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
    assert result['n_occurrences_found'] == 6  # 5 intermediate steps + 1 final answer
    assert result['occurrence_count_matches_expected'] is True


def test_occurrences_are_correctly_role_tagged_in_order():
    tokenizer = _get_tokenizer()
    starting_state, operations = 'Heads', ['different', 'different', 'different', 'different', 'different']
    prompt = build_code_word_prompt(starting_state, operations)
    completion = build_code_word_completion(starting_state, operations)
    result = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
    roles = [occ['role'] for occ in result['occurrences']]
    assert roles == ['intermediate_step_1', 'intermediate_step_2', 'intermediate_step_3',
                      'intermediate_step_4', 'intermediate_step_5', 'final_answer']


def test_token_positions_are_strictly_increasing():
    tokenizer = _get_tokenizer()
    starting_state, operations = 'Tails', ['different', 'same', 'same', 'different', 'different']
    prompt = build_code_word_prompt(starting_state, operations)
    completion = build_code_word_completion(starting_state, operations)
    result = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
    positions = [occ['token_position'] for occ in result['occurrences']]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)  # no duplicate positions


def test_final_answer_occurrence_code_word_matches_completion_final_answer():
    tokenizer = _get_tokenizer()
    for starting_state, operations in list(ALL_SCENARIOS)[:5]:
        prompt = build_code_word_prompt(starting_state, list(operations))
        completion = build_code_word_completion(starting_state, list(operations))
        result = log_code_word_positions(tokenizer, prompt, completion, (HEADS_CODE, TAILS_CODE))
        final_occ = result['occurrences'][-1]
        assert final_occ['role'] == 'final_answer'
        assert f"<answer> {final_occ['code_word']}</answer>" in completion


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
