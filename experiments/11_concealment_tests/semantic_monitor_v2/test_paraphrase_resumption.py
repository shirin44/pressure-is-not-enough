"""Golden-record test pinning the paraphrase-resumption GPU result
(aws_runs/stage11-paraphrase-resumption-v1/stage11_paraphrase_resumption.json,
retrieved and hash-verified 2026-09-23 from AWS i-REDACTED). Run standalone:
`python3 test_paraphrase_resumption.py`."""
import inspect
import json
from pathlib import Path

RESULTS = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage11-paraphrase-resumption-v1' /
                       'stage11_paraphrase_resumption.json').read_text())


def test_checkpoint_identity_matches_the_original_causal_flip_test():
    # Confirms this run loaded the EXACT SAME checkpoint (not merely the same-named
    # adapter directory) as experiments/09e_same_different_llama's
    # stage9e-llama-causal-flip-post-rl-seed43-v1 run.
    assert RESULTS['config']['checkpoint_identity_sha256'] == \
        '9e93ec5072bead6408dfeedaeae3689c83531f98b3e970d5bf827a8a7ded9dc2'


def test_sanity_check_passed_before_resumption_test_ran():
    assert RESULTS['config']['sanity_matches'] == RESULTS['config']['sanity_n'] == 5


def test_42_of_42_results_present_both_cut_points():
    assert RESULTS['summary']['n_total'] == 42
    cut2 = [r for r in RESULTS['results'] if r['cut_step'] == 2]
    cut3 = [r for r in RESULTS['results'] if r['cut_step'] == 3]
    assert len(cut2) == 21
    assert len(cut3) == 21


def test_all_42_track_correctly_through_paraphrased_prefix():
    assert RESULTS['summary']['fraction_tracks_correctly'] == 1.0
    assert RESULTS['summary']['classification_counts'] == {'tracks_correctly_genuine_causal_use': 42}
    for r in RESULTS['results']:
        assert r['classification'] == 'tracks_correctly_genuine_causal_use'
        assert r['final_matches_true'] is True
        assert r['propagation_matches_true'] is True
        assert r['format_valid'] is True


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
