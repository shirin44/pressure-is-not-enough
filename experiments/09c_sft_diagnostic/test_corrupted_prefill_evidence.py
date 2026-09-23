"""Golden-record regression tests against the corrupted-prefill causal diagnostic's
evidence (aws_runs/corrupted-prefill-diagnostic-v1/, 2026-08-31). Captures the two
findings this task exists to establish: (1) the rebalanced checkpoint's final <answer>
does NOT causally track its own corrupted trace in either direction on real held-out
data -- confirming genuine structural disconnect, not a narrower "defaults to Tails
under uncertainty" bug; (2) the fresh BASE (pre-SFT) model shows no raw-logit prior
favoring "Tails" at the answer position -- if anything the opposite -- ruling out an
inherited base-model word-preference bias as the explanation."""
import json
from pathlib import Path

EVIDENCE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'corrupted-prefill-diagnostic-v1'
                        / 'corrupted_prefill_diagnostic.json').read_text())


def test_config_used_the_correct_checkpoint_and_source_evidence():
    cfg = EVIDENCE['config']
    assert cfg['rebalanced_checkpoint_adapter_sha256'] == 'f19fe37eae12cef50dbeef012f5e50c9748b57e66d1bb101888ec80f0fafe673'
    assert cfg['source_evidence_sha256'] == '3d5ab4404464b04280aa18a7bcb1e6e91a0edc9d548b97edb30d0c2fa8eb603c'


def test_eligible_group_sizes_match_the_20_correctly_tracked_split():
    d1 = EVIDENCE['diagnostic_1_corrupted_prefill']
    assert d1['heads_to_tails']['n'] == 14
    assert d1['tails_to_heads']['n'] == 6
    assert d1['heads_to_tails']['n'] + d1['tails_to_heads']['n'] == 20


def test_heads_to_tails_direction_is_the_non_decisive_one_and_matches_expectation():
    # Not decisive alone (both hypotheses predict "Tails" here) -- recorded for
    # completeness, not as evidence either way.
    d1 = EVIDENCE['diagnostic_1_corrupted_prefill']
    assert d1['heads_to_tails']['outcome_counts'] == {'tracks_corrupted_prefix': 14}


def test_tails_to_heads_direction_is_decisive_and_confirms_structural_disconnect():
    # THE decisive result: every corrupted-to-Nib (Heads) prefix still produced "Tails"
    # -- the answer does not track the corrupted prefix at all in the direction where
    # the two competing hypotheses actually disagree.
    d1 = EVIDENCE['diagnostic_1_corrupted_prefill']
    assert d1['tails_to_heads']['outcome_counts'] == {'stays_at_original_answer': 6}
    for sample in d1['tails_to_heads']['samples']:
        assert sample['corruption_direction'] == 'heads'
        assert sample['new_answer'] == 'tails'
        assert sample['outcome'] == 'stays_at_original_answer'


def test_base_model_shows_no_tails_favoring_prior_ruling_out_that_hypothesis():
    d2 = EVIDENCE['diagnostic_2_base_model_logit_prior']
    assert d2['n_probes'] == 6
    assert d2['n_probes_favoring_tails'] == 0
    assert d2['n_probes_favoring_heads'] == 6
    # Substantial gap (not a marginal, noise-level preference either way).
    assert d2['mean_logprob_gap_tails_minus_heads'] < -5.0


def test_base_model_prior_direction_is_consistent_across_both_true_answer_sides():
    # The base-model prior favors "Heads" regardless of whether the scenario's true
    # answer is Heads or Tails, and regardless of whether the preceding trace ends in
    # Nib or Nomo -- a position-specific effect, not one driven by trace content.
    d2 = EVIDENCE['diagnostic_2_base_model_logit_prior']
    assert all(s['prior_favors'] == 'Heads' for s in d2['samples'])
    tokens_seen = {s['last_coded_token_in_prefix'] for s in d2['samples']}
    assert tokens_seen == {'nib', 'nomo'}


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
