"""Regression test for find_code_word_token_positions (per_token_gradient_diagnostic.py):
the first version of that script decoded one token at a time and substring-matched
against the decoded piece, which silently found ZERO Nib/Nomo positions (Qwen's
tokenizer splits "Nib" as " N"+"ib" and "Nomo" as " N"+"omo" -- neither sub-piece
contains the full word). Caught because it produced token_share_code=0.0 for a
checkpoint independently established to use Nib/Nomo on ~100% of completions, which
is not plausible. This is the exact same pitfall Step 14b's completion-level
qualitative analysis (experiments/09_direct_indomain_synthetic_bridge) already found
and fixed once in this project -- repeated here by mistake, now guarded against by
this test using the REAL Qwen2.5-3B-Instruct tokenizer (not a mock), since the
mock-vs-real tokenization behavior is exactly what caused the original miss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer

_tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')


def find_code_word_token_positions(ids, tokenizer=_tok, CODE_WORDS=('nib', 'nomo')):
    text_so_far = ''
    offsets = []
    for tid in ids:
        piece = tokenizer.decode([tid])
        start = len(text_so_far)
        text_so_far += piece
        offsets.append((start, len(text_so_far)))
    text_lower = text_so_far.lower()
    code_positions = set()
    for word in CODE_WORDS:
        start = 0
        while True:
            idx = text_lower.find(word, start)
            if idx == -1:
                break
            end = idx + len(word)
            before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
            after_ok = end == len(text_lower) or not text_lower[end].isalpha()
            if before_ok and after_ok:
                for i, (s, e) in enumerate(offsets):
                    if s < end and e > idx:
                        code_positions.add(i)
            start = idx + 1
    return code_positions


def test_nib_and_nomo_split_across_two_tokens_are_both_found():
    text = 'Step 1: The state is tracked. State: Nib\nStep 2: The code remains unchanged. State: Nomo\n<answer>Nomo</answer>'
    ids = _tok.encode(text, add_special_tokens=False)
    positions = find_code_word_token_positions(ids)
    # 3 occurrences (Nib, Nomo, Nomo-in-answer-tag), 2 tokens each = 6 positions.
    assert len(positions) == 6, f'expected 6 code-word token positions, got {len(positions)}: {sorted(positions)}'


def test_naive_per_token_substring_check_would_have_found_nothing():
    # Reproduces the ORIGINAL bug directly, as a permanent record of why the fix exists.
    text = 'State: Nib\nState: Nomo'
    ids = _tok.encode(text, add_special_tokens=False)
    naive_hits = [i for i, tid in enumerate(ids) if any(w in _tok.decode([tid]).lower() for w in ('nib', 'nomo'))]
    assert naive_hits == [], f'expected the naive per-token check to find nothing (that was the bug); found {naive_hits}'
    fixed_hits = find_code_word_token_positions(ids)
    assert len(fixed_hits) > 0, 'the fixed span-based check must find the code words the naive check misses'


def test_no_false_positives_on_literal_heads_tails_text():
    text = 'Step 1: The state remains unchanged. State: Heads\n<answer>Heads</answer>'
    ids = _tok.encode(text, add_special_tokens=False)
    positions = find_code_word_token_positions(ids)
    assert positions == set(), f'literal Heads/Tails text must not match Nib/Nomo; found {positions}'


def test_no_false_positive_on_substring_like_word():
    # "Nibble" contains "nib" as a substring but is NOT the word "Nib" -- must not match
    # (word-boundary check).
    text = 'This is a nibble of text, not the code word.'
    ids = _tok.encode(text, add_special_tokens=False)
    positions = find_code_word_token_positions(ids)
    assert positions == set(), f'"nibble" must not be matched as "Nib"; found {positions}'


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
