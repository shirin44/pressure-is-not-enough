"""Stage 9d: advantage clamp -- structurally analogous to Stage 4 / Stage 9b's per-token
KL clamp, but at the row/advantage level rather than the per-token/KL level.

Background (design.md, reward/gradient-linkage task): |advantage| correlates with
per-row gradient magnitude at r=0.9998 (essentially GRPO's own loss formula --
loss ~ -advantage * logprob -- showing up directly in measured data), while the
originally-observed code-token-density/gradient correlation (r=0.65) is fully explained
as a confound once advantage is controlled for. Resuming RL on Stage 9c's SFT-seeded
checkpoint hits a grad_norm breaker within 2-3 steps; this is a generic,
content-independent instability driven by advantage magnitude, unrelated to the
Nib/Nomo behavior under test.

Calibration, characterized from the 16-row linked dataset
(aws_runs/stage9d-reward-gradient-linkage/per_token_gradient_diagnostic_result_v2.json)
-- disclosed in full because the direct Stage-4-formula transplant (median + 1.5*IQR,
then *1.5 for the clamp) does NOT work here:

  sorted |advantage| (N=16): 0.1148, 0.1476, 0.1476, 0.2459, 0.2761, 0.3924, 0.7702,
  0.8865, 0.8865, 0.8865, 1.2952, 1.2952, 1.3224, 1.3280, 1.3280, 1.4387
  median=0.8865  q1=0.2610  q3=1.3088  iqr=1.0478

  Stage-4-style clamp (median + 1.5*IQR, *1.5) = 3.6873 -- ABOVE the observed max
  (1.4387) and close to GRPO's own theoretical ceiling (below). Mechanically applying
  Stage 4's exact formula here would produce a clamp that NEVER ENGAGES -- disclosed as
  a genuine finding, not silently worked around: this is not a case of "the same
  calibration recipe, different numbers," it's a case where the recipe's own
  assumptions (a heavy-tailed, empirically-unbounded quantity with a large pool of
  "typical" values and a genuine, separable tail of true outliers) do not hold for
  |advantage|, which is bounded by construction.

  THEORETICAL HARD BOUND (Grubbs' maximum-studentized-residual bound, distribution-free,
  exact, sample-size-independent -- not fit to this or any sample): for ANY n real
  numbers, the largest |x_i - sample_mean| / sample_std cannot exceed (n-1)/sqrt(n).
  TRL's advantage formula divides by (group_std + 1e-4) >= group_std, which can only
  SHRINK the ratio relative to dividing by group_std alone, so this bound applies to
  |advantage| itself, unconditionally, for GROUP_SIZE=8 (confirmed: this stage's
  GRPOConfig never sets scale_rewards, so TRL's default "group" mode applies --
  per-group std with GROUP_SIZE=8 generations, matching the bound below exactly):

      GRUBBS_MAX_ADVANTAGE = (GROUP_SIZE - 1) / sqrt(GROUP_SIZE) = 7/sqrt(8) = 2.4749...

  This means |advantage| can mathematically never exceed ~2.4749 for this config, no
  matter what the actual reward values are -- confirming the empirical 16-row sample
  (max 1.4387) is comfortably inside the possible range, not touching a hard boundary.

  Disclosed tension, not glossed over: the two group-steps in this 16-row sample ARE
  the steps that tripped the grad_norm breaker (52-65, vs. GRAD_BREAKER=50.0), yet no
  single row's |advantage| was close to the theoretical ceiling. This means the
  instability is not about rare PATHOLOGICAL individual-row outliers in an absolute
  sense (unlike Stage 4's per-token KL, which was genuinely unbounded) -- it is about
  the upper portion of an inherently compressed, group-normalized distribution still
  being large enough, and correlated enough with per-row gradient (r=0.9998), that
  several such rows accumulating within one 8-row optimizer step pushes the aggregate
  grad_norm over the breaker. A per-row magnitude clamp is still the right first lever
  to test (same "compress the upper end without touching the bulk" mechanism as the KL
  clamp), but this asymmetry between "no absolute outliers" and "still triggers the
  breaker" is flagged explicitly rather than assumed away.

  N=16 (2 independent 8-row groups) is too small for a trustworthy empirical
  percentile/IQR-style tail calibration -- with only 2 groups, there is no way to
  separate genuine population spread from between-group sampling noise. A calibration
  that could support real percentile/IQR-style tail estimation would need on the order
  of 10+ independent groups (80+ rows) minimum; obtaining that many groups requires
  running substantially more (expensive) GPU training steps, which was not done here.

  PROVISIONAL CHOICE (disclosed, not mechanically derived): ADVANTAGE_CLAMP_VALUE = 1.2,
  justified by TWO independent anchors that agree closely:
    (a) empirical: sits between the observed "lower cluster" (<=0.8865) and "upper
        cluster" (>=1.2952) of the 16-row sample -- engages on 6/16 (38%) of the
        observed rows, including the largest rows from BOTH breaker-triggering steps,
        while leaving the bottom 10/16 rows (everything <=0.8865) completely untouched.
    (b) theoretical: ~48% of the Grubbs hard ceiling (2.4749) -- "cap advantage
        magnitude at roughly half its mathematically-possible maximum for this group
        size" is an interpretable, round rule independent of small-N curve-fitting.
  38% engagement is a larger fraction than Stage 4's own calibrated engagement rate
  (~21%), consistent with |advantage| being an inherently more compressed quantity than
  per-token KL divergence -- reported as a real difference, not hidden by picking a
  looser clamp to superficially match Stage 4's engagement rate.

  Validation path (per instruction, since N=16 is too small to trust in isolation):
  this is a PROVISIONAL estimate, validated directly by whether the actual re-run (with
  the clamp active) avoids the grad_norm breaker -- not by further small-sample
  statistics. If a future run needs a firmer empirical calibration, the 10+ group
  minimum above is the concrete target.
"""
from __future__ import annotations

import math
import os
import statistics

GROUP_SIZE = 8
GRUBBS_MAX_ADVANTAGE = (GROUP_SIZE - 1) / math.sqrt(GROUP_SIZE)  # ~2.4749, theoretical hard ceiling

# 2026-08-30 addendum: ADVANTAGE_CLAMP_VALUE=1.2 measurably helped (baseline reached
# step 6 and its first milestone) but did not fully resolve the breaker (still fired in
# both phases -- design.md, advantage-clamp addendum). Searching for a tighter,
# minimally-sufficient value before considering any change to the breaker's own
# semantics (explicitly out of scope without separate sign-off). Candidates, same
# reasoning style as the original 1.2 choice (fraction of the Grubbs ceiling + fraction
# of the 16-row diagnostic sample that would be clamped), computed directly rather than
# picked as round numbers:
#
#   clamp=1.2   (48.5% of ceiling): engages  6/16 (37.5%) -- the value already tested,
#               insufficient alone.
#   clamp=0.7   (28.3% of ceiling): engages 10/16 (62.5%) -- sits centrally in the wide
#               empirical gap between the sample's "low cluster" (<=0.3924, 6 rows) and
#               "high cluster" (>=0.7702, 10 rows); the natural next step down.
#   clamp=0.4   (16.2% of ceiling): engages the SAME 10/16 (62.5%) rows as 0.7 on this
#               static sample (both thresholds fall in the same empty gap, so they catch
#               identical rows) -- but clamps them down to a smaller final magnitude,
#               i.e. a strictly MORE aggressive intervention despite an identical
#               engagement RATE. Flagged explicitly: engagement rate alone doesn't fully
#               capture how aggressive a clamp is: the boundary VALUE among engaged rows
#               matters too.
#   clamp=0.25  (10.1% of ceiling): engages 12/16 (75.0%) -- now clamping the clear
#               MAJORITY of rows, which would functionally resemble a blanket downscale
#               of advantage rather than a genuine-tail clamp (the KL/entropy
#               coefficient's job, not this mechanism's) -- the outer edge of the tested
#               range; not gone below this without direction, per instruction.
#
# RESULT (design.md has the full re-run evidence): tested in order 0.7 -> 0.4, stopped
# at 0.4 (a clean pass) without needing 0.25.
#   0.7: MIXED, not a clean pass -- baseline regressed (hard_stop at step 3, worse than
#        1.2's step 6, no milestone reached) while main improved (reached step 5 and
#        cleared its first-ever milestone before breaking). Engagement 52-54%.
#   0.4: CLEAN PASS -- both phases ran to completion with hard_stop=None (baseline
#        16/16 steps, all 4 milestones; main 8/8 steps, both milestones), KL stayed in
#        its normal healthy range throughout (0.0-0.36, nowhere near KL_BREAKER=5.0).
#        DISCLOSED CAVEAT, not glossed over: engagement rate at 0.4 is 78.1% (baseline)
#        and 70.3% (main) -- the clear MAJORITY of all rows, not a genuine tail. This is
#        no longer a surgical outlier-clamp in the Stage-4-KL-clamp sense; at this
#        value the mechanism functions closer to a blanket downscale of advantage
#        magnitude across nearly the whole distribution -- closer to what the KL
#        clamp / entropy coefficient's job is than a targeted intervention. Adopted as
#        the new default because it is the value that actually produces a clean pass,
#        not because the high engagement rate is unremarkable -- it is remarked upon
#        explicitly here and in design.md.
# ADOPTED DEFAULT: 0.4. Still overridable via STAGE9D_ADVANTAGE_CLAMP_VALUE so a
# candidate can be launched from the SAME hash-checked file without editing it (the
# chosen value is printed in the run's own startup config for cross-verification).
ADVANTAGE_CLAMP_VALUE = float(os.environ.get('STAGE9D_ADVANTAGE_CLAMP_VALUE', '0.4'))


def compute_diagnostics_from_pool(abs_advantage_pool):
    """Median/IQR diagnostics over a pool of |advantage| values -- same shape as
    kl_calibration.compute_clamp_from_pool, kept for future recalibration with a larger
    pool, NOT used to derive ADVANTAGE_CLAMP_VALUE above (that Stage-4-style formula was
    checked against this pool and found to produce a non-engaging clamp -- see module
    docstring). Returns n, median, q1, q3, iqr."""
    pool = sorted(abs(v) for v in abs_advantage_pool)
    n = len(pool)
    if n == 0:
        return {'n': 0, 'median': None, 'q1': None, 'q3': None, 'iqr': None}
    median = statistics.median(pool)
    q1 = statistics.median(pool[:n // 2])
    q3 = statistics.median(pool[(n + 1) // 2:])
    return {'n': n, 'median': median, 'q1': q1, 'q3': q3, 'iqr': q3 - q1}


def clamp_advantages(advantages, clamp_value=ADVANTAGE_CLAMP_VALUE):
    """Symmetric hard clamp on advantage magnitude, sign preserved -- additive/surgical,
    same design philosophy as the per-token KL clamp (dmax_sensitivity_check.py /
    kl_calibration.py): values already within [-clamp_value, clamp_value] pass through
    with EXACTLY zero distortion (torch.clamp is the identity function in that region);
    values outside are hard-capped at the boundary, NOT zeroed -- an extreme-advantage
    row still contributes a bounded, correctly-signed gradient rather than being
    silently dropped from training (advantage is a fixed per-example weight on the
    logprob gradient in GRPO's loss, not itself a differentiable quantity here, so this
    clamp only affects which VALUE is used to weight the loss, not any gradient path
    through the advantage tensor itself).

    Returns (clamped_advantages, engaged_mask) where engaged_mask is a bool tensor,
    True where |advantages| > clamp_value (i.e. where clamping actually changed the
    value)."""
    engaged = advantages.abs() > clamp_value
    clamped = advantages.clamp(min=-clamp_value, max=clamp_value)
    return clamped, engaged
