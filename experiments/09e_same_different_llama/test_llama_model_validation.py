"""Golden-record regression test against the real Llama-3-8B-Instruct model
validation evidence (aws_runs/stage9e-llama-model-validation-v1/, 2026-09-07),
Stage 9e Part A, Steps 0-2. Inference only, no training.

Confirms, against REAL measured evidence (not assumption or reuse from Qwen):
- HF gated-model authentication worked end-to-end on the GPU instance.
- 8-bit-quantized model footprint (~8.5GB) fits comfortably in a single A10G
  (22.6GB total on g5.xlarge) -- no instance upgrade needed for Part A/inference
  work; this does NOT by itself confirm sizing for full GRPO training with
  activations/gradients/generation KV-cache, which is a separate, larger question.
- Qwen's Nib/Nomo (training pair) and Yelt/Yark (held-out pair) independently
  re-audited and re-verified eligible on Llama's own tokenizer and own base-policy
  log-probabilities -- not reused by assumption. Both pairs show an exact
  within-pair logprob match, same as they did on Qwen.
- Llama-3's chat template is confirmed structurally DIFFERENT from Qwen's
  (<|start_header_id|>/<|eot_id|> vs Qwen's <|im_start|>/<|im_end|>), and its
  tokenizer has pad_token=None by default (Qwen's does not) -- both flagged
  explicitly per the task's own requirement not to assume parity."""
import json
from pathlib import Path

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9e-llama-model-validation-v1' / 'stage9e_llama_model_validation.json').read_text())

QWEN_PAIRS_ELIGIBLE = ('Nib', 'Nomo', 'Yelt', 'Yark')


def test_ran_against_the_correct_model():
    assert EVIDENCE['config']['model'] == 'meta-llama/Meta-Llama-3-8B-Instruct'


def test_model_footprint_fits_comfortably_in_a_single_a10g():
    mem = EVIDENCE['memory']
    assert mem['gpu_name'] == 'NVIDIA A10G'
    assert mem['total_gpu_memory_mib'] > 22000
    # 8-bit weights measured directly, not estimated
    assert 8000 < mem['model_footprint_mib'] < 9500
    # comfortable headroom remains for LoRA + activations + generation KV-cache,
    # though this test does not itself validate a full training run fits
    headroom = mem['total_gpu_memory_mib'] - mem['model_footprint_mib']
    assert headroom > 12000


def test_thirteen_of_thirty_candidates_eligible():
    assert EVIDENCE['n_eligible'] == 13
    assert len(EVIDENCE['audit_rows']) == 30


def test_qwens_four_pairs_independently_reverified_eligible_on_llama_tokenizer():
    qwen_pairs = EVIDENCE['qwen_pairs_on_llama_tokenizer']
    for name in QWEN_PAIRS_ELIGIBLE:
        assert qwen_pairs[name]['eligible'] is True, name
        assert qwen_pairs[name]['passes_leakage_classifier'] is True, name
        assert qwen_pairs[name]['n_tokens_with_leading_space'] == 2, name


def test_training_pair_nib_nomo_shows_exact_within_pair_logprob_match():
    qwen_pairs = EVIDENCE['qwen_pairs_on_llama_tokenizer']
    nib_lp = qwen_pairs['Nib']['base_policy_logprob_at_state_slot']
    nomo_lp = qwen_pairs['Nomo']['base_policy_logprob_at_state_slot']
    assert abs(nib_lp - nomo_lp) < 1e-6


def test_heldout_pair_yelt_yark_shows_exact_within_pair_logprob_match():
    qwen_pairs = EVIDENCE['qwen_pairs_on_llama_tokenizer']
    yelt_lp = qwen_pairs['Yelt']['base_policy_logprob_at_state_slot']
    yark_lp = qwen_pairs['Yark']['base_policy_logprob_at_state_slot']
    assert abs(yelt_lp - yark_lp) < 1e-6


def test_chat_template_is_structurally_different_from_qwens():
    template = EVIDENCE['chat_template_raw']
    assert '<|start_header_id|>' in template
    assert '<|eot_id|>' in template
    # Qwen's own template markers should NOT appear -- confirms this really is
    # Llama's own template, not an accidental fallback to a cached Qwen one
    assert '<|im_start|>' not in template
    assert '<|im_end|>' not in template


def test_pad_token_is_none_by_default_unlike_qwen():
    # A real, flagged divergence from Qwen -- the existing project scripts already
    # defend against this (if tokenizer.pad_token is None: pad_token = eos_token),
    # but this confirms the fallback is actually exercised for Llama, not dead code.
    assert EVIDENCE['tokenizer_info']['pad_token'] is None
    assert EVIDENCE['tokenizer_info']['eos_token'] == '<|eot_id|>'


def test_rendered_chat_template_produces_the_expected_special_token_structure():
    ids = EVIDENCE['chat_template_rendered_token_ids']
    assert len(ids) > 0
    rendered = EVIDENCE['chat_template_rendered_example']
    assert rendered.startswith('<|begin_of_text|><|start_header_id|>user<|end_header_id|>')
    assert rendered.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n')


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
