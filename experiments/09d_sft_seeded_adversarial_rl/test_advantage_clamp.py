"""CPU unit tests for advantage_clamp.py -- mirrors test_kl_calibration.py's style
(self-test properties for a clamp: healthy values pass through untouched, pathological
values get bounded not zeroed, gradient through the clamped path behaves sensibly)."""
import math

import torch

from advantage_clamp import GROUP_SIZE, GRUBBS_MAX_ADVANTAGE, ADVANTAGE_CLAMP_VALUE, \
    clamp_advantages, compute_diagnostics_from_pool


def test_grubbs_bound_matches_group_size_8():
    assert GROUP_SIZE == 8
    assert math.isclose(GRUBBS_MAX_ADVANTAGE, 7 / math.sqrt(8), rel_tol=1e-12)


def test_grubbs_bound_is_the_true_tight_maximum_for_n8():
    # Direct empirical confirmation of the theoretical claim: construct the extremal
    # configuration (n-1 identical values, one outlier) for n=8 and verify the resulting
    # studentized residual equals the Grubbs bound, and that NO other configuration of
    # 8 values can exceed it (checked via random search, not just the analytic case).
    import random
    x = torch.tensor([0.0] * 7 + [1.0])
    z = (x - x.mean()).abs() / x.std(unbiased=True)
    assert math.isclose(z.max().item(), GRUBBS_MAX_ADVANTAGE, rel_tol=1e-6)

    rng = random.Random(20260830)
    worst = 0.0
    for _ in range(20000):
        vals = torch.tensor([rng.uniform(-10, 10) for _ in range(8)])
        if vals.std(unbiased=True) < 1e-9:
            continue
        z = (vals - vals.mean()).abs() / vals.std(unbiased=True)
        worst = max(worst, z.max().item())
    assert worst <= GRUBBS_MAX_ADVANTAGE + 1e-6, f'found a configuration exceeding the claimed bound: {worst}'


def test_advantage_clamp_value_is_below_the_theoretical_ceiling():
    # The provisional clamp must be a real, engaging bound -- strictly below the
    # mathematical maximum, not (accidentally) at or above it like the naive
    # Stage-4-formula transplant this module's docstring found and rejected.
    assert 0 < ADVANTAGE_CLAMP_VALUE < GRUBBS_MAX_ADVANTAGE


def test_adopted_default_is_0_4_after_the_candidate_search():
    # 1.2 (the original provisional value) was superseded after the candidate search
    # (design.md, 2026-08-30): 0.7 was a mixed result (not a clean pass), 0.4 produced
    # a clean pass and was adopted as the new default. This test guards against a
    # future edit silently reverting the default without updating this record.
    assert ADVANTAGE_CLAMP_VALUE == 0.4, (
        f'expected the adopted default 0.4 (no STAGE9D_ADVANTAGE_CLAMP_VALUE override '
        f'set); got {ADVANTAGE_CLAMP_VALUE} -- if this is an intentional change, update '
        f'this test and design.md together')


def test_healthy_values_pass_through_with_zero_distortion():
    advantages = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])  # all within [-1.2, 1.2]
    clamped, engaged = clamp_advantages(advantages, clamp_value=1.2)
    assert torch.equal(clamped, advantages), 'healthy values must be bit-for-bit unchanged'
    assert not engaged.any()


def test_pathological_values_are_capped_not_zeroed():
    advantages = torch.tensor([-3.0, -1.2001, 1.2001, 5.0])
    clamped, engaged = clamp_advantages(advantages, clamp_value=1.2)
    assert torch.equal(engaged, torch.tensor([True, True, True, True]))
    assert torch.allclose(clamped, torch.tensor([-1.2, -1.2, 1.2, 1.2]))
    # Explicitly NOT zero -- the whole point of a clamp vs. a mask-to-zero.
    assert (clamped != 0).all()


def test_clamp_preserves_sign_at_the_boundary():
    advantages = torch.tensor([-10.0, 10.0])
    clamped, engaged = clamp_advantages(advantages, clamp_value=1.2)
    assert clamped[0].item() < 0 and clamped[1].item() > 0
    assert engaged.all()


def test_mixed_batch_only_engages_on_the_outliers():
    advantages = torch.tensor([0.1, 0.8865, 1.2952, -1.4387, 0.3924])
    clamped, engaged = clamp_advantages(advantages, clamp_value=1.2)
    assert torch.equal(engaged, torch.tensor([False, False, True, True, False]))
    assert torch.allclose(clamped, torch.tensor([0.1, 0.8865, 1.2, -1.2, 0.3924]))


def test_gradient_sanity_clamped_row_still_contributes_a_bounded_nonzero_gradient():
    # Advantage itself carries no gradient (it's a fixed per-row loss weight, not a
    # differentiable function of the policy in GRPO's formulation) -- what must behave
    # sensibly is the DOWNSTREAM gradient w.r.t. logprob when weighted by a clamped
    # advantage: nonzero, correctly signed, and bounded by the clamp value, not zeroed
    # out the way a naive "drop pathological rows" implementation would behave.
    raw_advantages = torch.tensor([-10.0, 0.5, 10.0])
    clamped, _ = clamp_advantages(raw_advantages, clamp_value=1.2)
    logprob = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss = -(clamped * logprob).sum()
    loss.backward()
    # d(loss)/d(logprob_i) = -clamped_i
    assert torch.allclose(logprob.grad, -clamped)
    assert (logprob.grad != 0).all(), 'clamped rows must still produce a nonzero gradient, not be silently dropped'
    assert logprob.grad.abs().max().item() <= 1.2 + 1e-6, 'gradient contribution must be bounded by the clamp value'


def test_unclamped_gradient_is_unbounded_by_contrast():
    # Sanity check on the test methodology itself: WITHOUT the clamp, the same setup
    # produces an unbounded gradient contribution -- confirms the clamp is doing
    # something, not just reproducing what would have happened anyway.
    raw_advantages = torch.tensor([-10.0, 0.5, 10.0])
    logprob = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    loss = -(raw_advantages * logprob).sum()
    loss.backward()
    assert logprob.grad.abs().max().item() == 10.0


def test_compute_diagnostics_from_pool_matches_manual_calculation():
    pool = [0.1148, 0.1476, 0.1476, 0.2459, 0.2761, 0.3924, 0.7702,
            0.8865, 0.8865, 0.8865, 1.2952, 1.2952, 1.3224, 1.3280, 1.3280, 1.4387]
    diag = compute_diagnostics_from_pool(pool)
    assert diag['n'] == 16
    assert math.isclose(diag['median'], 0.8865, abs_tol=1e-4)
    assert math.isclose(diag['iqr'], 1.0478, abs_tol=1e-3)


def test_naive_stage4_formula_transplant_would_not_engage_on_this_pool():
    # Permanent record of the finding that motivated the provisional-constant design:
    # median + 1.5*IQR, *1.5 (Stage 4/9b's exact KL-clamp formula) produces a clamp
    # ABOVE the observed max on this pool -- i.e. it would never engage. This is why
    # ADVANTAGE_CLAMP_VALUE is a disclosed provisional choice, not a mechanical
    # transplant of the KL-clamp calibration recipe.
    pool = [0.1148, 0.1476, 0.1476, 0.2459, 0.2761, 0.3924, 0.7702,
            0.8865, 0.8865, 0.8865, 1.2952, 1.2952, 1.3224, 1.3280, 1.3280, 1.4387]
    diag = compute_diagnostics_from_pool(pool)
    naive_clamp = 1.5 * (diag['median'] + 1.5 * diag['iqr'])
    assert naive_clamp > max(pool), 'expected the naive transplant to sit above the observed max (the finding this module documents)'


def test_sft_seeded_rl_wires_per_row_capture_and_per_phase_clamp_override():
    # Static, source-level guard (2026-08-30 signal-preservation + per-config-split
    # investigation): confirms PER_ROW_CAPTURES is actually appended to inside the
    # compute_loss patch (not just defined and unused), and that run_phase() accepts
    # and applies a per-phase clamp_value override, with baseline/main each passing
    # their own (possibly distinct) override at the top-level call sites.
    from pathlib import Path
    script = (Path(__file__).resolve().parent / 'sft_seeded_rl.py').read_text()
    assert 'PER_ROW_CAPTURES.append(' in script
    assert 'CURRENT_CLAMP_VALUE[0] = clamp_value' in script
    assert "'per_row_advantages': phase_per_row_advantages" in script
    assert 'clamp_value=BASELINE_CLAMP_VALUE' in script
    assert 'clamp_value=MAIN_CLAMP_VALUE' in script


def test_sft_seeded_rl_wires_the_advantage_clamp_into_compute_loss():
    # Static, source-level guard (same style as test_checkpoint_isolation.py) that the
    # clamp is actually installed as a GRPOTrainer.compute_loss patch in the script that
    # runs the 8-step repro, not just defined and left unused.
    from pathlib import Path
    script = (Path(__file__).resolve().parent / 'sft_seeded_rl.py').read_text()
    assert 'from advantage_clamp import' in script
    assert 'clamp_advantages(' in script
    assert 'GRPOTrainer.compute_loss = _advantage_clamped_compute_loss' in script
    assert "ADVANTAGE_CLAMP_STATE['engaged_count'] = 0" in script, (
        'per-phase reset missing -- engagement counts would leak between the baseline '
        'and main phases')


def test_clamped_rerun_evidence_shows_the_clamp_engaged_and_delayed_but_did_not_prevent_the_breaker():
    # Golden-record regression test against the ACTUAL retrieved evidence from the
    # clamped 8-step repro (aws_runs/stage9d-advantage-clamp-v1/, 2026-08-30). Written
    # to reflect what genuinely happened, not what the task originally hoped for: the
    # clamp measurably engaged and the baseline phase survived to and past its first
    # milestone (further than any prior unclamped attempt), but the grad_norm breaker
    # still fired in BOTH phases eventually. This test intentionally does NOT assert
    # "hard_stop is None" -- that would misrepresent the result. See design.md's
    # advantage-clamp addendum for the full, disclosed write-up.
    import json
    from pathlib import Path
    evidence_path = (Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-advantage-clamp-v1'
                      / 'stage9d_sft_seeded_adversarial_rl.json')
    d = json.loads(evidence_path.read_text())
    b, m = d['baseline_result'], d['main_dryrun_result']

    # The clamp genuinely engaged in both phases (not a no-op).
    assert b['advantage_clamp_summary']['engaged_count'] > 0
    assert m['advantage_clamp_summary']['engaged_count'] > 0

    # Baseline progress: reached step 6 (vs. 2-3 in every prior unclamped attempt at
    # resuming RL on this checkpoint) and cleared its first milestone (step 4) with the
    # seeded behavior fully intact.
    assert b['terminal_step'] >= 6
    assert any(ms['step'] == 4 and ms['nonliteral_rate'] == 1.0 for ms in b['milestones'])

    # Honest negative: the breaker still fired in both phases. Recorded explicitly so a
    # future change to advantage_clamp.py that silently regresses this further (e.g.
    # breaks even earlier) is caught, without pretending this run fully succeeded.
    assert b['hard_stop'] is not None and b['hard_stop']['grad_norm'] >= 50.0
    assert m['hard_stop'] is not None and m['hard_stop']['grad_norm'] >= 50.0
    assert m['milestones'] == [], 'main adversarial phase broke before reaching any milestone'


def test_candidate_0_7_evidence_shows_a_mixed_not_clean_result():
    # Golden-record regression test against the actual retrieved evidence from the
    # clamp=0.7 candidate run (aws_runs/stage9d-advantage-clamp-0.7/, 2026-08-30):
    # baseline regressed (broke at step 3, worse than 1.2's step 6, no milestone) while
    # main improved (reached step 5 and cleared its first-ever milestone before
    # breaking). Recorded so this specific, informative mixed result isn't lost.
    import json
    from pathlib import Path
    evidence_path = (Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-advantage-clamp-0.7'
                      / 'stage9d_sft_seeded_adversarial_rl.json')
    d = json.loads(evidence_path.read_text())
    b, m = d['baseline_result'], d['main_dryrun_result']
    assert b['advantage_clamp_summary']['clamp_value'] == 0.7
    assert b['hard_stop'] is not None and b['terminal_step'] == 3
    assert b['milestones'] == []
    assert m['hard_stop'] is not None and m['terminal_step'] == 5
    assert any(ms['step'] == 4 for ms in m['milestones']), 'main should have cleared its step-4 milestone before breaking at step 5'


def test_candidate_0_4_evidence_shows_a_clean_pass_with_high_engagement_disclosed():
    # Golden-record regression test against the actual retrieved evidence from the
    # clamp=0.4 candidate run (aws_runs/stage9d-advantage-clamp-0.4/, 2026-08-30): the
    # clean pass this whole candidate search was looking for -- but ALSO asserts the
    # disclosed caveat (78%/70% engagement, i.e. the clamp is bounding the majority of
    # rows, not a genuine tail) so that caveat can't silently disappear from the record.
    import json
    from pathlib import Path
    evidence_path = (Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-advantage-clamp-0.4'
                      / 'stage9d_sft_seeded_adversarial_rl.json')
    d = json.loads(evidence_path.read_text())
    b, m = d['baseline_result'], d['main_dryrun_result']

    # Clean pass: hard_stop is None, both phases ran to their full requested length.
    assert b['hard_stop'] is None and b['terminal_step'] == b['n_steps_requested'] == 16
    assert m['hard_stop'] is None and m['terminal_step'] == m['n_steps_requested'] == 8

    # Both phases cleared every milestone.
    assert [ms['step'] for ms in b['milestones']] == [4, 8, 12, 16]
    assert [ms['step'] for ms in m['milestones']] == [4, 8]

    # KL stayed in its normal healthy range throughout (KL_BREAKER=5.0).
    assert all(t.get('kl', 0) < 1.0 for t in b['telemetry'] if 'kl' in t)
    assert all(t.get('kl', 0) < 1.0 for t in m['telemetry'] if 'kl' in t)

    # Disclosed caveat: this is NOT a genuine-tail clamp at 0.4 -- it engages on the
    # clear majority of rows in both phases. Asserted explicitly so it can't be quietly
    # dropped from the record while still calling this a "clean pass."
    assert b['advantage_clamp_summary']['engagement_rate'] > 0.7
    assert m['advantage_clamp_summary']['engagement_rate'] > 0.6


def test_signal_preservation_evidence_confirms_second_clean_pass_and_quantifies_dampening():
    # Golden-record regression test against the instrumented signal-preservation run
    # (aws_runs/stage9d-signal-preservation-0.4/, 2026-08-30, CHECK 1/CHECK 2 task):
    # a SECOND independent clean pass at clamp=0.4 (robustness confirmation), plus the
    # quantitative signal-dampening numbers that drove the CHECK 1 verdict.
    import json
    from pathlib import Path
    evidence_path = (Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-signal-preservation-0.4'
                      / 'stage9d_sft_seeded_adversarial_rl.json')
    d = json.loads(evidence_path.read_text())
    b, m = d['baseline_result'], d['main_dryrun_result']

    # Second independent clean pass.
    assert b['hard_stop'] is None and b['terminal_step'] == 16
    assert m['hard_stop'] is None and m['terminal_step'] == 8

    # Per-row advantage data present for both phases.
    assert len(b['per_row_advantages']) == 128
    assert len(m['per_row_advantages']) == 64

    # Quantified dampening: at 0.4, roughly half the total raw advantage signal is
    # removed by clamping (56% in this run) -- asserted with headroom so this test
    # doesn't require bit-exact reproduction of a nondeterministic GPU run, but does
    # catch a gross regression in either direction.
    def total_signal_removed_fraction(rows):
        pre = sum(abs(r['pre_clamp_advantage']) for r in rows)
        post = sum(abs(r['post_clamp_advantage']) for r in rows)
        return (pre - post) / pre
    assert 0.4 < total_signal_removed_fraction(b['per_row_advantages']) < 0.7
    assert 0.4 < total_signal_removed_fraction(m['per_row_advantages']) < 0.7


def test_baseline_and_main_raw_advantage_distributions_are_not_meaningfully_different():
    # Golden-record test for the CHECK 2 finding that closed the per-config-split
    # search: baseline's and main's PRE-clamp (clamp-invariant) advantage
    # distributions are nearly statistically indistinguishable in this run (means
    # 0.815 vs 0.826, medians 0.811 vs 0.812) -- refuting the hypothesis that main's
    # CoT-penalty term produces a systematically different/less-extreme advantage
    # distribution than baseline's correctness-only reward. This is the data-grounded
    # reason a per-config clamp split was not pursued further.
    import json
    import statistics as st
    from pathlib import Path
    evidence_path = (Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-signal-preservation-0.4'
                      / 'stage9d_sft_seeded_adversarial_rl.json')
    d = json.loads(evidence_path.read_text())
    b, m = d['baseline_result'], d['main_dryrun_result']
    bpre = [abs(r['pre_clamp_advantage']) for r in b['per_row_advantages']]
    mpre = [abs(r['pre_clamp_advantage']) for r in m['per_row_advantages']]
    assert abs(st.fmean(bpre) - st.fmean(mpre)) < 0.1, 'baseline/main mean |advantage| diverged more than expected'
    assert abs(st.median(bpre) - st.median(mpre)) < 0.1, 'baseline/main median |advantage| diverged more than expected'


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
