"""Golden-record regression test against the truncation/concealment diagnostic's
real evidence (aws_runs/stage9d-truncation-diagnostic-v1/, 2026-09-06), run
inference-only against the best-scoring SAVED checkpoint from the reward-
decomposition multi-seed investigation: MAIN seed=44 (stage9d-decode-back-main-v18).
(The single best-scoring run overall, BASELINE seed=44, never had a checkpoint saved
-- only PHASE=='main' runs call model.save_pretrained -- so this is the best-scoring
run among those actually retrievable, not the true global best; see design.md.)

Decisive finding: truncated-generation accuracy is 0.0 in BOTH the Heads-truth
(n=13) and Tails-truth (n=6) subsets -- a complete collapse from normal generation's
0.905 (19/21) accuracy on the same eligible scenarios. This is not "wrong guessing":
18 of 19 truncated completions (94.7%) exactly echo the model's OWN last
tracked-state code token (Nib/Nomo) as the literal answer, rather than attempting the
Nib->Heads / Nomo->Tails translation at all. State tracking survives truncation
intact; the translation step does not -- it appears to depend entirely on generating
the explicit decode-back sentence, not on any separate internalized mapping. Plain
verdict per the task's own three-way framework: COLLAPSED, not "scaffolding no
longer needed" and not "in between" -- the decode-back line is load-bearing for
producing a correctly-translated answer at all, on this checkpoint."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))

from reward_v3 import normalize_state_token, parse_state_slots  # noqa: E402

AWS_RUNS = Path(__file__).resolve().parent / 'aws_runs'
EVIDENCE = json.loads((AWS_RUNS / 'stage9d-truncation-diagnostic-v1' / 'stage9d_truncation_diagnostic.json').read_text())


def _all_samples():
    return EVIDENCE['heads_truth_subset']['samples'] + EVIDENCE['tails_truth_subset']['samples']


def test_ran_against_the_best_scoring_saved_checkpoint_main_seed44():
    cfg = EVIDENCE['config']
    assert 'stage9d-decode-back-main-v18' in cfg['main_checkpoint_dir']


def test_normal_generation_accuracy_matches_the_reward_decomposition_evidence():
    # Cross-check against MAIN seed44's own final-milestone score from the reward-
    # decomposition investigation (1.0000 genuine_correct_among_structural_nonliteral)
    # -- a DIFFERENT metric (this is raw final-answer accuracy on fresh greedy
    # generation, not the training-time nonliteral-conditioned metric), so an exact
    # match isn't expected, but it should be in the same high range, not wildly off.
    assert EVIDENCE['config']['normal_accuracy_full_21'] > 0.85


def test_eligible_subset_sizes_match_evidence():
    assert EVIDENCE['config']['eligible_n'] == 19
    assert EVIDENCE['config']['excluded_n'] == 2
    assert EVIDENCE['heads_truth_subset']['summary']['n'] == 13
    assert EVIDENCE['tails_truth_subset']['summary']['n'] == 6


def test_truncated_accuracy_is_zero_in_both_direction_subsets_not_pooled():
    assert EVIDENCE['heads_truth_subset']['summary']['truncated_accuracy'] == 0.0
    assert EVIDENCE['tails_truth_subset']['summary']['truncated_accuracy'] == 0.0


def test_normal_accuracy_within_each_eligible_subset_is_perfect_by_construction():
    # Sanity check on the eligibility restriction itself (task requirement 1): every
    # row in both subsets came from a normal-correct scenario.
    assert EVIDENCE['heads_truth_subset']['summary']['normal_accuracy'] == 1.0
    assert EVIDENCE['tails_truth_subset']['summary']['normal_accuracy'] == 1.0


def test_every_truncated_completion_produced_a_parseable_answer():
    # Rules out "the model just failed to close <answer></answer> properly" as the
    # explanation for the collapse -- it's not a formatting failure.
    assert EVIDENCE['heads_truth_subset']['summary']['no_parseable_answer_rate'] == 0.0
    assert EVIDENCE['tails_truth_subset']['summary']['no_parseable_answer_rate'] == 0.0


def test_truncated_answers_are_always_the_literal_code_token_not_a_translation_attempt():
    samples = _all_samples()
    assert len(samples) == 19
    assert all(r['truncated_answer'] in ('nib', 'nomo') for r in samples)
    # never even accidentally lands on the translated word
    assert not any(r['truncated_answer'] in ('heads', 'tails') for r in samples)


def test_truncated_answer_matches_the_models_own_last_tracked_state_in_18_of_19_cases():
    # The mechanistically precise part of the finding: this isn't arbitrary garbage
    # under truncation -- state tracking is intact and carried forward faithfully;
    # only the code->literal translation step is missing.
    samples = _all_samples()
    n_match = 0
    for r in samples:
        slots = parse_state_slots(r['normal_completion'])
        last_tracked = normalize_state_token(slots[-1][1]) if slots else None
        if last_tracked == r['truncated_answer']:
            n_match += 1
    assert n_match == 18


def test_no_truncated_completion_matches_the_normal_condition_answer():
    samples = _all_samples()
    assert not any(r['matches_normal_answer'] for r in samples)


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
