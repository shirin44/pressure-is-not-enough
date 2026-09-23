"""Stage 10, task step 3 (the one hard requirement): CPU-only, tokenizer-only automated
verification that the RL dataset built by this experiment's runner produces a CHAT-TEMPLATED
prompt when tokenised the way TRL's GRPOTrainer actually tokenises a train_dataset row --
NOT the raw-string signature every earlier stage in this project (Stage 1, Stage 7, 09b, 09d,
Llama 09e's RL script) accidentally used. No model is loaded; this only exercises the
tokenizer, matching the read-only "print the first prompt's token ids locally" check the task
required, turned into an assertion instead of a manual read.

Reference signatures (established in the prior Qwen prompt-format audit this session,
`analysis_paper_audit/q_qwen_tokenization_probe.py`, Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct
share one tokenizer):
  RAW  (a plain string tokenised directly, i.e. TRL's non-conversational branch):
       first ids [24617, 1584, 25, 70507, 198, ...] = "Starting", " state", ":", ...
  CHAT (apply_chat_template, i.e. TRL's conversational branch):
       first ids [151644, 8948, 198, 2610, 525, 1207, 16948, 11, ...] = "<|im_start|>", "system", "\\n", ...
"""
from __future__ import annotations

from transformers import AutoTokenizer

from literal_cot_bank import build_conversational_prompt, unique_pool

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
CHAT_SIGNATURE_PREFIX = [151644, 8948, 198]  # <|im_start|>, system, \n
RAW_SIGNATURE_PREFIX = [24617, 1584, 25]  # Starting, " state", :


def _tok():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _simulate_trl_conversational_tokenisation(tok, conversational_prompt):
    """Reproduces TRL 1.9.2's own branch for a conversational (list-of-dict) prompt
    (grpo_trainer.py's data_utils.py:160-188 confirms list-of-dict -> is_conversational=True;
    grpo_trainer.py:1780's else-branch, `self.processing_class.apply_chat_template(...,
    add_generation_prompt=True, tokenize=True)`, is what actually runs for such a row).
    Kept as a thin, clearly-cited wrapper, not a reimplementation of TRL's internals."""
    # transformers 5.13.1's apply_chat_template(tokenize=True) returns a BatchEncoding;
    # ['input_ids'] is the standard accessor for the flat token-id list (confirmed directly,
    # not assumed -- .ids on the object returns a list of internal Encoding objects instead).
    return tok.apply_chat_template(conversational_prompt, add_generation_prompt=True, tokenize=True)['input_ids']


def _simulate_trl_raw_string_tokenisation(tok, prompt_text):
    """The MISTAKE path, kept here only so the two signatures can be told apart in the same
    test file: what TRL would have done if the prompt had been a plain string
    (grpo_trainer.py:1780's non-conversational branch, `processing_class(text=prompts)`)."""
    return tok(prompt_text)['input_ids']


def test_build_conversational_prompt_is_list_of_dict_not_a_string():
    row = unique_pool(1, 20261001)[0]
    conv = build_conversational_prompt(row['prompt'])
    assert isinstance(conv, list)
    assert conv == [{'role': 'user', 'content': row['prompt']}]
    assert not isinstance(conv, str)


def test_conversational_prompt_tokenises_to_the_chat_template_signature():
    tok = _tok()
    row = unique_pool(1, 20261001)[0]
    conv = build_conversational_prompt(row['prompt'])
    ids = _simulate_trl_conversational_tokenisation(tok, conv)
    assert ids[:3] == CHAT_SIGNATURE_PREFIX, (
        f'Expected the chat-template signature {CHAT_SIGNATURE_PREFIX}, got {ids[:8]}. '
        'This means the prompt is NOT being chat-templated -- the exact bug found in every '
        'prior Qwen/Llama RL stage this session.')
    decoded_first3 = [tok.decode([i]) for i in ids[:3]]
    assert decoded_first3 == ['<|im_start|>', 'system', '\n']


def test_conversational_prompt_does_NOT_tokenise_to_the_raw_string_signature():
    tok = _tok()
    row = unique_pool(1, 20261001)[0]
    conv = build_conversational_prompt(row['prompt'])
    ids = _simulate_trl_conversational_tokenisation(tok, conv)
    assert ids[:3] != RAW_SIGNATURE_PREFIX


def test_raw_string_signature_is_reproduced_only_by_the_mistake_path_never_by_this_experiment():
    # Confirms the two signatures really are distinguishable with THIS tokenizer/prompt text
    # (i.e. the test above is a meaningful check, not a vacuous one).
    tok = _tok()
    row = unique_pool(1, 20261001)[0]
    raw_ids = _simulate_trl_raw_string_tokenisation(tok, row['prompt'])
    assert raw_ids[:3] == RAW_SIGNATURE_PREFIX


def test_all_train_and_heldout_prompts_in_a_real_split_are_conversational():
    from literal_cot_bank import build_train_heldout_split
    train_rows, heldout_rows = build_train_heldout_split(run_seed=20261001, n_train=43, n_heldout=20)
    tok = _tok()
    for row in train_rows[:5] + heldout_rows[:5]:
        conv = build_conversational_prompt(row['prompt'])
        ids = _simulate_trl_conversational_tokenisation(tok, conv)
        assert ids[:3] == CHAT_SIGNATURE_PREFIX


if __name__ == '__main__':
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t(); print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append(t.__name__); print(f'FAILED: {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
