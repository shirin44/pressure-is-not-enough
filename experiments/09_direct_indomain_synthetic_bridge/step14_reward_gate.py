"""Step 14: hard reward-gate for the 16 already-demonstrated (coded) bank scenarios.

Replaces Steps 10-13b's soft auxiliary-CE side channel with a structurally different
mechanism: on the 16 CODED_TRAJECTORIES scenarios only, a well-formed literal OR
vacuous completion's reward is pushed toward a genuine hard zero, annealed in rather
than applied on day one; a correct, fully consistent Nib/Nomo completion (category 9,
"correct_globally_consistent_code") keeps its full, unmodified reward_v3 score. Every
OTHER scenario -- the clean 21-eval set, the rest of the on-policy training pool, and
every non-bank length -- goes through the exact same, byte-for-byte unmodified
score_completion_v2 call as Step 0/the bridge phase. This is a genuinely new point in
reward space (a correct, well-formed answer scoring exactly 0 has never been produced
by this project's reward function before), so the invariant is re-derived fresh here
rather than assumed by analogy -- see verify_step14_reward_invariant below and
step14_reward_ordering_table.md for the concrete category table.

Scope note (explicit, per design requirement): this tests whether the model can be
forced to ADOPT a code it has already been shown via Steps 10-13b's demonstration
trajectories, not whether it can invent one from scratch (that is Stage 1's task,
untouched here).

Post-dry-run fixes (2026-08-27, after the first 8-step dry run and a full cumulative-
risk assessment): two of the three reward-invariant gaps disclosed at launch were
assessed as real-but-tolerable and then fixed anyway (cheap, no downside, removes a
confound from interpreting eventual results):

- GAP 1 FIXED: the gate trigger now covers `category in (LITERAL_CATEGORY,
  VACUOUS_CATEGORY)`, not literal alone. A vacuous completion on a bank scenario now
  ramps toward zero exactly like a literal one -- closing the 2.5-vs-0
  vacuous-outranks-gated-literal inversion, which was the same structural failure
  signature documented in Stage 1 (`experiments/01_direct_rl_coinflip/README.md`:
  "the model always used literal Heads/Tails, or went vacuous... to dodge the
  penalty"). Because literal and vacuous now share the identical `bank_gate_scale`
  factor at every step, this closes the gap throughout the whole ramp, not just at
  full anneal (their relative ordering was already scale-invariant; only their
  UNGATED base values' ordering mattered, and literal's ceiling, 4.15, already
  exceeded vacuous's ceiling, 3.0 -- see verify_step14_reward_invariant).
- GAP 3 FIXED: the gate trigger additionally requires `base['r_task'] != -5.0`, i.e.
  the completion must be well-formed (a valid, closed `<answer>` tag), not just
  taxonomy-literal-or-vacuous. A malformed completion (missing/invalid answer tag)
  now ALWAYS keeps its natural, ungated -5.0-based penalty, regardless of what state
  words it happens to contain -- closing the malformed-ties-gated-literal-at-0
  inversion that occurred live in the first dry run.
- GAP 2 NOT FIXED, re-confirmed negligible under the combined trigger (not assumed
  unaffected -- rechecked): the ramp-window margin question is governed by literal's
  ungated ceiling (4.15), which already exceeds vacuous's ungated ceiling (3.0) at
  every step since both are now scaled by the identical factor; widening the gate to
  include vacuous does not introduce a new or worse version of this gap. See
  verify_step14_reward_invariant's docstring for the full re-derivation.

See step14_design.md for the full cumulative-risk assessment that led to these
decisions, and step14_reward_ordering_table.md for the fixed concrete category table.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from reward_v3 import score_completion_v2, completion_to_text, prompt_to_text  # noqa: E402

from taxonomy import classify_candidate  # noqa: E402
from synthetic_bridge import CODED_TRAJECTORIES  # noqa: E402

LITERAL_CATEGORY = 1
VACUOUS_CATEGORY = 2
CORRECT_CODE_CATEGORY = 9
GATED_CATEGORIES = (LITERAL_CATEGORY, VACUOUS_CATEGORY)

BANK_SCENARIO_KEYS = frozenset(
    (row["starting_state"], tuple(row["operations"])) for row in CODED_TRAJECTORIES
)

_START_RE = re.compile(r"(?mi)^\s*Starting state:\s*(Heads|Tails)\s*$")
_OP_RE = re.compile(r"(?mi)^\s*\d+\.\s*(same\s+as|different\s+from)\s+previous\b")


def scenario_key_from_prompt(prompt: Any) -> tuple[str, tuple[str, ...]] | None:
    """Parse (starting_state, operations) out of a prompt, matching the exact
    convention build_cot_prompt/build_prompt use. Returns None if it doesn't parse
    (caller treats that as 'not a bank scenario', never as an error -- this must
    degrade gracefully, matching reward_v3._physical_states_by_index's own contract)."""
    text = prompt_to_text(prompt)
    start_match = _START_RE.search(text)
    ops = _OP_RE.findall(text)
    if not start_match or not ops:
        return None
    operations = tuple("same" if o.lower().startswith("same") else "different" for o in ops)
    return (start_match.group(1), operations)


def is_bank_scenario(prompt: Any) -> bool:
    return scenario_key_from_prompt(prompt) in BANK_SCENARIO_KEYS


# ===== Anneal schedule =====
#
# Ramp schedule: linear from initial_scale at step 1 to final_scale (=1.0, a genuine
# hard zero) at step ramp_steps, then HOLD at final_scale for the remaining steps
# through total_steps. Concrete numbers for the 150-step run: ramp_steps=30 (20% of
# the run), initial_scale=0.25.
#
# Reasoning: the 16 bank scenarios have NEVER been on-policy training targets before
# (Steps 10-13b excluded them from TRAIN_POOL by construction and only ever
# teacher-forced them off-policy). At step 1 the model has effectively a ~0% prior
# rate of emitting Nib/Nomo on them, so every rollout in an early group is category-1
# literal. If literal reward were already fully zeroed at step 1, EVERY rollout in
# that group would score identically 0 regardless of whether the literal answer was
# right or wrong, collapsing GRPO's within-group reward variance to exactly zero on
# the very first bank-scenario step it ever sees -- exactly the failure mode design
# requirement 5 asks to guard against. Scaling by (1 - initial_scale) = 0.75 instead
# of zero preserves the *ranking* between literal completions (correct-literal still
# scores above incorrect-literal, just uniformly compressed), which keeps typical
# per-group reward variance nonzero as long as the group's literal completions aren't
# all equally right or wrong (see verify_step14_reward_invariant and
# test_step14_reward_gate.py's zero-reward-variance stress test for why this matters).
# Ramping to the full 1.0 by step 30 rather than holding a permanent partial penalty
# honors requirement 2's explicit "reward = 0, regardless of correctness" as the
# steady state, not a permanently soft one.

DEFAULT_RAMP_STEPS = 30
DEFAULT_INITIAL_SCALE = 0.25
DEFAULT_FINAL_SCALE = 1.0


def bank_gate_scale(
    step: int,
    *,
    ramp_steps: int = DEFAULT_RAMP_STEPS,
    initial_scale: float = DEFAULT_INITIAL_SCALE,
    final_scale: float = DEFAULT_FINAL_SCALE,
) -> float:
    """Piecewise-linear penalty scale for literal completions on bank scenarios:
    initial_scale at step 1, linearly to final_scale at step == ramp_steps, then
    held at final_scale for step > ramp_steps. final_scale=1.0 means a completion's
    reward is multiplied by (1 - scale) == 0 -- a genuine hard zero, not an
    asymptotic approach to one."""
    if step < 1:
        raise ValueError("step must be >= 1")
    if ramp_steps < 1:
        raise ValueError("ramp_steps must be >= 1")
    if not 0.0 <= initial_scale <= final_scale <= 1.0:
        raise ValueError("require 0 <= initial_scale <= final_scale <= 1")
    if step >= ramp_steps:
        return final_scale
    if ramp_steps == 1:
        return final_scale
    progress = (step - 1) / (ramp_steps - 1)
    return initial_scale + (final_scale - initial_scale) * progress


def score_completion_gated(
    completion: Any,
    ground_truth: str,
    step: int,
    total_steps: int,
    *,
    prompt: Any,
    ramp_steps: int = DEFAULT_RAMP_STEPS,
    initial_scale: float = DEFAULT_INITIAL_SCALE,
    final_scale: float = DEFAULT_FINAL_SCALE,
    **bridge_reward_params: Any,
) -> dict[str, float | int | str | bool | None]:
    """score_completion_v2's full breakdown, with `total` replaced by the gated value
    on bank scenarios. Every other scenario passes through with `total` byte-identical
    to score_completion_v2's own output -- see
    test_step14_reward_gate.py::test_non_bank_scenarios_are_provably_unaffected.

    Gated when ALL of: the scenario is one of the 16 bank scenarios, the completion is
    taxonomy-literal or taxonomy-vacuous (GATED_CATEGORIES), AND the completion is
    well-formed (`r_task != -5.0`, i.e. it has a valid closed <answer> tag). The
    well-formedness condition is what excludes malformed completions from the gate --
    without it, a malformed-but-literal (or malformed-but-vacuous) completion would
    ALSO be swept in and collapse to the same 0 as a well-formed one at full anneal,
    erasing the -5.0 malformed penalty (this occurred live in the first 8-step dry
    run; see the module docstring's "GAP 3 FIXED" note)."""
    base = score_completion_v2(
        completion, ground_truth, step, total_steps, prompt=prompt, **bridge_reward_params
    )
    bank = is_bank_scenario(prompt)
    classification = classify_candidate(
        completion_to_text(completion), prompt, ground_truth=ground_truth
    )
    gate_scale = None
    gated_total = base["total"]
    if bank and classification["category"] in GATED_CATEGORIES and base["r_task"] != -5.0:
        gate_scale = bank_gate_scale(
            step, ramp_steps=ramp_steps, initial_scale=initial_scale, final_scale=final_scale
        )
        gated_total = base["total"] * (1.0 - gate_scale)
    return {
        **base,
        "total": gated_total,
        "ungated_total": base["total"],
        "is_bank_scenario": bank,
        "taxonomy_category": classification["category"],
        "taxonomy_category_name": classification["category_name"],
        "bank_gate_scale": gate_scale,
    }


# ===== Reward invariant, re-derived fresh for this (post-fix) configuration =====
#
# Both GAP 1 (vacuous outranks gated-literal) and GAP 3 (malformed-literal ties
# gated-literal) are FIXED as of 2026-08-27 -- see the module docstring's "Post-dry-
# run fixes" note for the risk assessment that led to fixing them. This function is
# re-derived from scratch for the combined trigger (not patched from the pre-fix
# version), per the explicit instruction to re-verify both fixes together rather than
# assume either is independently safe. GAP 2 (ramp-window margin) was assessed
# negligible and left as-is; it is rechecked below under the new trigger, not assumed
# unaffected.

def verify_step14_reward_invariant(
    *,
    ramp_steps: int = DEFAULT_RAMP_STEPS,
    initial_scale: float = DEFAULT_INITIAL_SCALE,
    final_scale: float = DEFAULT_FINAL_SCALE,
    consistency_magnitude: float = 0.15,
    signal_magnitude: float = 0.15,
    cot_max_scale: float = 2.0,
) -> dict[str, Any]:
    """Re-derive, fresh, the properties this design promises, using the ACTUAL
    per-category term constraints from reward_v3.py (not generic 'correct/wrong/
    malformed' extremes): r_consistency is exactly 0 for ANY literal or ANY vacuous
    completion (consistency_bonus_v2 excludes literal tokens, and vacuous has no
    parsed slots to be consistent over); vacuous additionally forces r_signal to 0
    (no slots to correlate) and p_structure/p_state_variation to 0.5 (structure_
    penalty and state_variation_penalty both require a full num_flips-length valid
    trace, impossible with zero slots); a genuine category-9 completion forces
    r_consistency to 0 (>=2 distinct tokens can never satisfy consistency_bonus_v2's
    single-token requirement) and r_signal to exactly signal_magnitude (ARI==1.0 by
    category-9's own definition).

    1. UNCONDITIONAL at full anneal: gated = base_total * (1 - scale), and
       final_scale == 1.0 forces EVERY gated category (literal or vacuous, well-
       formed) to exactly 0 regardless of its own base_total's sign or magnitude --
       so literal and vacuous are also EQUAL to each other at full anneal (gap 1's
       fix), and category-9's strictly-positive worst-case floor always exceeds
       both. Checked, not assumed.
    2. NOT unconditionally guaranteed during the transient ramp (scale < 1.0) against
       an adversarial category-9 completion: p_cot scans raw completion TEXT for
       banned literal words independent of which taxonomy category the parsed
       State-slots land in, so a contrived category-9 completion with literal-word-
       heavy surrounding prose can still hit p_cot's penalty while a best-case gated
       completion suppressed only by the weak initial gate scale is not guaranteed
       to fall below it. Computed for BOTH literal's and vacuous's own ceilings
       (rechecked, not assumed identical) and reported, not silently asserted past,
       since it CAN be negative.
    3. A fully-annealed gated completion (reward exactly 0) always outranks the
       worst MALFORMED completion of any category -- unconditional, checked
       exhaustively. Malformed completions are now categorically excluded from the
       gate (gap 3's fix), so this no longer needs a "restricted to non-literal
       malformed" carve-out the way the pre-fix version did.
    4. Confirms vacuous no longer exceeds gated-literal at full anneal (gap 1) and
       that malformed no longer ties gated-anything at full anneal (gap 3) --
       reported as explicit before/after facts, not just inferred from the margins.
    """
    # Category 9 (correct_globally_consistent_code): r_task=4.0 required; r_consistency
    # forced 0.0; r_signal forced to signal_magnitude. Format penalties and p_cot free.
    code_totals = [
        4.0 - p_cot - p_length - p_structure - p_state_variation + 0.0 + signal_magnitude
        for p_cot in (0.0, cot_max_scale)
        for p_length in (0.0, 0.5)
        for p_structure in (0.0, 0.5)
        for p_state_variation in (0.0, 0.5)
    ]
    min_correct_code = min(code_totals)

    # Literal (category 1, well-formed: r_task in {4.0, -0.5}): r_consistency forced
    # 0.0. p_cot, the three format penalties, and r_signal (content-agnostic) free.
    literal_totals = [
        r_task - p_cot - p_length - p_structure - p_state_variation + 0.0 + r_signal
        for r_task in (4.0, -0.5)
        for p_cot in (0.0, cot_max_scale)
        for p_length in (0.0, 0.5)
        for p_structure in (0.0, 0.5)
        for p_state_variation in (0.0, 0.5)
        for r_signal in (0.0, signal_magnitude)
    ]
    max_literal_base = max(literal_totals)

    # Vacuous (category 2, well-formed: r_task in {4.0, -0.5}): r_consistency AND
    # r_signal forced 0.0 (no parsed slots at all); p_structure AND p_state_variation
    # forced 0.5 (both require a full valid trace, impossible with zero slots). Only
    # p_cot (raw-text scan, independent of slot count) and p_length remain free.
    vacuous_totals = [
        r_task - p_cot - p_length - 0.5 - 0.5 + 0.0 + 0.0
        for r_task in (4.0, -0.5)
        for p_cot in (0.0, cot_max_scale)
        for p_length in (0.0, 0.5)
    ]
    max_vacuous_base = max(vacuous_totals)

    # (1) Unconditional at full anneal: computed genuinely from each ceiling scaled by
    # (1 - final_scale), not hardcoded to 0, so this stays a real check even if a
    # caller passes a final_scale other than the 1.0 default.
    gated_literal_at_full_anneal = max_literal_base * (1.0 - final_scale)
    gated_vacuous_at_full_anneal = max_vacuous_base * (1.0 - final_scale)
    gated_at_full_anneal = max(gated_literal_at_full_anneal, gated_vacuous_at_full_anneal)
    margin_code_over_gated_at_full_anneal = min_correct_code - gated_at_full_anneal
    assert margin_code_over_gated_at_full_anneal > 0, (
        f"invariant broken at full anneal: margin = {margin_code_over_gated_at_full_anneal}"
    )
    vacuous_exceeds_gated_literal_at_full_anneal = gated_vacuous_at_full_anneal > gated_literal_at_full_anneal

    # (2) NOT asserted -- reported, for both gated categories independently (gap 2
    # recheck: do not assume vacuous behaves identically to literal here).
    weakest_gate_scale = min(initial_scale, final_scale)
    max_gated_literal_during_ramp = max_literal_base * (1.0 - weakest_gate_scale)
    max_gated_vacuous_during_ramp = max_vacuous_base * (1.0 - weakest_gate_scale)
    margin_code_over_gated_literal_during_ramp = min_correct_code - max_gated_literal_during_ramp
    margin_code_over_gated_vacuous_during_ramp = min_correct_code - max_gated_vacuous_during_ramp
    # The binding (harder-to-satisfy) constraint is whichever gated ceiling is higher.
    margin_code_over_gated_during_ramp = min(
        margin_code_over_gated_literal_during_ramp, margin_code_over_gated_vacuous_during_ramp
    )

    # (3) Malformed (ANY category -- literal-malformed and vacuous-malformed are both
    # excluded from the gate now, so the worst-case bound is no longer restricted to
    # "non-literal only"). r_consistency and r_signal both remain free (a malformed
    # completion's state-slot content is independent of whether it closed its tag).
    malformed_totals = [
        -5.0 + r_consistency + r_signal
        for r_consistency in (0.0, consistency_magnitude)
        for r_signal in (0.0, signal_magnitude)
    ]
    max_malformed = max(malformed_totals)
    margin_gated_over_malformed = gated_at_full_anneal - max_malformed
    assert margin_gated_over_malformed > 0, (
        f"invariant broken: gated/malformed margin = {margin_gated_over_malformed}"
    )
    malformed_ties_gated_at_full_anneal = (max_malformed == gated_at_full_anneal)  # always False now

    return {
        "margin_code_over_gated_at_full_anneal": margin_code_over_gated_at_full_anneal,
        "margin_code_over_gated_literal_during_ramp": margin_code_over_gated_literal_during_ramp,
        "margin_code_over_gated_vacuous_during_ramp": margin_code_over_gated_vacuous_during_ramp,
        "margin_code_over_gated_during_ramp": margin_code_over_gated_during_ramp,
        "code_over_gated_guaranteed_during_ramp": margin_code_over_gated_during_ramp > 0,
        "margin_gated_over_malformed": margin_gated_over_malformed,
        "malformed_ties_gated_at_full_anneal": malformed_ties_gated_at_full_anneal,
        "gated_at_full_anneal": gated_at_full_anneal,
        "vacuous_exceeds_gated_literal_at_full_anneal": vacuous_exceeds_gated_literal_at_full_anneal,
        "max_literal_base": max_literal_base,
        "max_vacuous_base": max_vacuous_base,
    }


# ===== Dynamic-sampling safety net, extended =====
#
# The existing mechanism (bridge_full_10pct.py's dynamic_generate) resamples a group
# when task-correctness is unanimous (all right or all wrong) or when fewer than 25%
# of the group is structurally valid. Neither check looks at TOTAL REWARD. Under this
# gate, once gate_scale reaches 1.0, every literal completion in a group scores
# EXACTLY 0 regardless of correctness -- so a group that is entirely literal (the
# expected early/likely state on a scenario the model has never used the code on)
# can have correctness split 3-5 (neither 0 nor GROUP_SIZE, so the existing check does
# NOT fire) while every reward is still identically 0. That is a real, confirmed gap:
# GRPO's advantage computation degenerates (uniform reward -> zero/undefined
# advantage) exactly in the scenario design requirement 5 asks to check for, and the
# existing check does not catch it. Fixed here by adding an explicit total-reward
# check alongside the existing two, kept as a pure function so it's testable without
# a GPU (see test_step14_reward_gate.py's all-zero-reward stress test).

def bank_gate_resample_reasons(
    rewards: Sequence[float],
    *,
    correct_count: int,
    structural_count: int,
    group_size: int,
) -> list[str]:
    """Pure decision function extracted from bridge_full_10pct.py's dynamic_generate,
    extended with a reward-degeneracy check the correctness/structure checks miss.
    Returns the list of resample reasons (empty == accept as-is)."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rewards) != group_size:
        raise ValueError("rewards must have exactly group_size entries")
    reasons: list[str] = []
    if correct_count in (0, group_size):
        reasons.append("correctness")
    if structural_count < (-(-group_size // 4)):  # ceil(group_size / 4), matching math.ceil(.25*GROUP_SIZE)
        reasons.append("structure")
    if all(r == 0.0 for r in rewards):
        reasons.append("all_zero_reward")
    elif statistics.pstdev(rewards) == 0.0:
        reasons.append("reward_degenerate")
    return reasons


def simulate_dynamic_sampling(
    attempts_rewards: Sequence[Sequence[float]],
    *,
    correct_counts: Sequence[int],
    structural_counts: Sequence[int],
    group_size: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Drives bank_gate_resample_reasons across up to max_attempts, matching
    dynamic_generate's own accept/fallback logic exactly (accept if no reasons, or
    forced-accept-with-fallback-flag on the final attempt) -- never raises, never
    silently returns an unflagged degenerate batch."""
    if not (len(attempts_rewards) == len(correct_counts) == len(structural_counts)):
        raise ValueError("attempts_rewards, correct_counts, and structural_counts must align")
    if len(attempts_rewards) < max_attempts:
        raise ValueError("must supply at least max_attempts candidate attempts to simulate")
    for attempt in range(1, max_attempts + 1):
        idx = attempt - 1
        reasons = bank_gate_resample_reasons(
            attempts_rewards[idx],
            correct_count=correct_counts[idx],
            structural_count=structural_counts[idx],
            group_size=group_size,
        )
        accepted = not reasons or attempt == max_attempts
        if accepted:
            return {
                "attempt": attempt,
                "accepted": True,
                "fallback": bool(reasons),
                "reasons": reasons,
                "rewards": list(attempts_rewards[idx]),
            }
    raise RuntimeError("Dynamic sampling returned no group.")  # unreachable given the loop above
