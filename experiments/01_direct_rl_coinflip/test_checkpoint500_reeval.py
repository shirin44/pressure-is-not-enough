"""Golden-record test for the checkpoint-500 RE-EVALUATION (2026-09-22, PART B step 7).

This pins the result of a standalone re-evaluation of checkpoint 500 under current audit
tooling (chat-templated prompts, the existing 10-tier taxonomy classifier). It is NOT a
test of, or a stand-in for, the original ~3,200-rollout Stage-1 training result -- that
result's own driver code and prompt format could not be located (see the prior
search-task report) and this test makes no claim about it. See
aws_recovered/checkpoint500_reeval.json's own 'caveat' field for the same point in the
evidence file itself."""
import json
from pathlib import Path

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_recovered' / 'checkpoint500_reeval.json').read_text())


def test_this_is_explicitly_flagged_as_a_reevaluation_not_a_training_rerun():
    assert EVIDENCE['RE_EVALUATION_NOT_ORIGINAL_TRAINING_RERUN'] is True
    assert 'NOT a rerun' in EVIDENCE['caveat']


def test_config_uses_chat_templated_prompts_not_the_original_raw_format():
    c = EVIDENCE['config']
    assert c['prompt_format'] == 'chat_templated'
    assert c['model'] == 'Qwen/Qwen2.5-3B-Instruct'
    assert c['n_rollouts'] == 500
    assert c['flips_range'] == [3, 8]
    assert c['adapter_sha256'] == '4795fe548d7130eb0c559fd3cc829afca247f53a96bca99c68c3eace0f3209a0'


def test_tier_distribution_sums_to_500_and_is_entirely_tier_1():
    dist = EVIDENCE['tier_distribution']
    assert sum(dist.values()) == 500
    assert dist['1_literal'] == 500
    for k, v in dist.items():
        if k != '1_literal':
            assert v == 0, k


def test_no_structural_nonliteral_no_verbalized_attempt_no_tier_9_or_10():
    assert EVIDENCE['structural_nonliteral_candidate_count'] == 0
    assert EVIDENCE['verbalized_substitution_attempt_count'] == 0
    assert EVIDENCE['tier_9_or_10_count'] == 0
    assert EVIDENCE['tier_9_or_10_rows'] == []


def test_format_validity_and_task_accuracy_are_reasonable_sanity_signals():
    # Not part of the taxonomy itself, but confirms the checkpoint is behaving coherently
    # under the chat-templated format (not garbage output that happens to avoid tier 2-10).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
    from reward_v3 import _extract_answer, normalize_state_token
    rows = EVIDENCE['rows']
    assert len(rows) == 500
    valid = sum(1 for r in rows if _extract_answer(r['completion'])[1])
    correct = sum(1 for r in rows if _extract_answer(r['completion'])[1]
                  and _extract_answer(r['completion'])[0] == normalize_state_token(r['ground_truth']))
    assert valid / 500 > 0.9
    assert correct / 500 > 0.3


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
