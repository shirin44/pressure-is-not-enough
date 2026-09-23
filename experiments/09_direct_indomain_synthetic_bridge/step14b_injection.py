"""Step 14b: inject a verified correct Nib/Nomo trajectory directly into the rollout
comparison group on the 16 gated bank scenarios, so GRPO always has at least one
reliable, high-scoring comparison point -- rather than depending on the model to
spontaneously produce a non-literal token it has never once produced unprompted
(0/3,850+ prior tracked rollouts, 0/2,200 more in Step 14's own full run).

Root cause this addresses (see step14b_design.md for the full writeup): from full
anneal (step 30) onward, Step 14's own run showed ~90%+ of accepted bank-scenario
training groups either hit the dynamic-sampling all-zero-reward fallback or landed
at exact zero anyway (73/127 group-attempts, 28/54 accepted groups forced to
fallback, 21/54 more at exact zero) -- meaning GRPO had no usable relative-advantage
signal to learn from on the large majority of bank-scenario training rounds, no
matter how many times a group was resampled, because resampling draws from the SAME
policy that has never produced anything but literal/vacuous on these scenarios.

Deliberate reversal of Steps 10-13b's design principle (stated explicitly, per
design requirement 2): Step 10's design audit avoided injecting into the GRPO
rollout group specifically because a synthetic high-scoring member distorts the
group-relative advantage baseline for the genuinely-sampled members, confounding
"the model discovered this" with "the model is imitating an injected example" --
its own words were "mechanically valid but confounds discovery with reward-weighted
regression, working against the experiment's own purpose." That concern was about
SCIENTIFIC ATTRIBUTION for a spontaneous-discovery experiment, not about GRPO's
math being unsound (Step 10 said "mechanically valid" explicitly). Step 14b is not
a discovery experiment -- the whole Step 14 line of work already reframed the
question as "can the model be forced to ADOPT a code it has already been shown,"
not "will it invent one." Given that reframing, a zero-signal group is strictly
worse than a group with one deliberately-placed comparison point: the soft
auxiliary loss (Steps 10-13b) was the alternative that preserved discovery-style
attribution, and it produced no effect across two full 150-step runs either. The
distortion concern is real and unavoidable (see below), but it is now secondary to
having any learnable signal at all.

GRPO on/off-policy re-derivation (design requirement 4, re-derived fresh against
this project's ACTUAL trl==1.9.1/1.9.2 source, not by analogy to Step 10's
different injection point -- see step14b_design.md for the full trace through
trl/trainer/grpo_trainer.py):

1. This project's exact GRPOConfig (gradient_accumulation_steps=GROUP_SIZE=8,
   steps_per_generation=GROUP_SIZE=8, num_iterations=1) satisfies
   `gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0`,
   which is precisely the condition under which trl's `_generate_and_score_
   completions` sets `old_per_token_logps = None` (verified by reading trl's
   source directly, not inferred). At `compute_loss` time, trl's own fallback is
   `old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None
   else ...` -- i.e. old == current EXACTLY, by construction, for EVERY row,
   REGARDLESS of whether that row was genuinely sampled or substituted. The
   importance-sampling ratio is exactly 1 for every row already, with or without
   injection -- there was never a real on/off-policy distinction being leveraged
   by PPO/GRPO's clipping machinery in this project's configuration to begin with
   (num_iterations=1 makes it effectively single-step REINFORCE-with-baseline, not
   true multi-epoch proximal optimization). Injection does not introduce a NEW
   on/off-policy correction problem because there was no on-policy assumption in
   the ratio term to violate.
2. Reference-model KL logps (`ref_per_token_logps`, used for the beta*KL penalty)
   are computed via a forward pass over WHATEVER `completion_ids` end up in the
   batch, after generation, via the same per-token-logps computation this
   project's own KL-clamp patch already intercepts. An injected row's ref-logps
   are therefore computed identically to any generated row's, through the SAME
   already-verified clamp mechanism -- no special-casing needed, and the clamp's
   existing bound (D_MAX, ~0.5) already protects against exactly the failure mode
   an injected, very-low-policy-probability token sequence could otherwise cause
   (a KL spike from tokens the policy currently assigns near-zero probability to).
3. FLAGGED, NOT SILENTLY RESOLVED -- a real residual risk: the injected row's
   *policy-gradient* contribution (not the KL term) scales with the advantage
   times the gradient of the log-probability the CURRENT policy assigns to the
   injected tokens. Early in training (and plausibly throughout, given the model
   has never produced these tokens), that probability is likely very low, so the
   gradient magnitude from this one row could be unusually large relative to a
   typical on-policy row. This is not eliminated by anything in this design -- it
   is mitigated only by the pre-existing grad_norm hard breaker (50.0), which will
   halt the run rather than let an unstable update proceed silently, exactly as it
   already does for any other cause of a large gradient. Watch grad_norm
   specifically on steps where injection fires, during the dry run and any later
   full run.
4. A confirmed, deliberate side effect on the dynamic-sampling safety net
   (design requirement 5): because every gated-scenario group now always contains
   at least one non-zero, non-uniform reward (the injected row), the existing
   `all_zero_reward`/`reward_degenerate`/unanimous-`correctness` resample checks
   will rarely if ever fire for bank scenarios anymore. The safety net is kept
   fully active, unchanged, as the explicitly-requested backstop for the case
   where injection itself fails to apply (e.g. a bug in scenario matching) --
   not because it is expected to matter in the common case anymore.
"""
from __future__ import annotations

from typing import Any, Sequence

from step14_reward_gate import BANK_SCENARIO_KEYS, scenario_key_from_prompt
from synthetic_bridge import CODED_TRAJECTORIES

# Deterministic, documented choice (design requirement 3): replace exactly the
# FIRST rollout index within each bank-scenario group -- the minimum needed to
# guarantee a non-degenerate group. A larger fraction would proportionally shrink
# the number of genuinely-sampled on-policy attempts observed per group without
# addressing the failure mode any further (one non-degenerate member is already
# sufficient to give GRPO's group-relative advantage computation something to
# differentiate). INJECTIONS_PER_GROUP=1 is asserted, not just documented, in
# test_step14b_injection.py.
INJECTIONS_PER_GROUP = 1

_TRAJECTORY_BY_SCENARIO: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {
    (row["starting_state"], tuple(row["operations"])): row for row in CODED_TRAJECTORIES
}
assert len(_TRAJECTORY_BY_SCENARIO) == 16 == len(CODED_TRAJECTORIES) == len(BANK_SCENARIO_KEYS)
assert set(_TRAJECTORY_BY_SCENARIO) == BANK_SCENARIO_KEYS


def injected_trajectory_for_prompt(prompt: Any) -> dict[str, Any] | None:
    """The matching verified CODED_TRAJECTORIES row for this prompt's exact scenario,
    or None if the prompt is not one of the 16 gated bank scenarios. Matching is by
    the same (starting_state, operations) key scenario_key_from_prompt already uses
    for gating -- the injected trajectory for a given prompt is always the ONE
    trajectory verified (by synthetic_bridge.verify_trajectory at import time) to
    solve that exact scenario, never a different scenario's trajectory."""
    key = scenario_key_from_prompt(prompt)
    if key is None or key not in BANK_SCENARIO_KEYS:
        return None
    return _TRAJECTORY_BY_SCENARIO[key]


def build_injected_completion_ids(tokenizer: Any, completion_text: str) -> list[int]:
    """Tokenize a trajectory's completion text exactly as synthetic_bridge.
    teacher_forced_ce already does for its own teacher-forced targets in Steps
    10-13b: the raw completion text plus the tokenizer's eos token, with no added
    special tokens -- so an injected row's token boundary conventions match what
    this project has already used and verified elsewhere, not a new convention."""
    eos = tokenizer.eos_token or ""
    return tokenizer.encode(completion_text + eos, add_special_tokens=False)


def select_injection_indices(prompts: Sequence[Any]) -> dict[int, dict[str, Any]]:
    """For each DISTINCT bank scenario appearing among `prompts`, pick exactly one
    index (the first occurrence) to receive its matching injected trajectory.
    Returns {index: trajectory_row}; empty if no bank scenario is present.

    Deliberately does not assume every entry in `prompts` shares one scenario --
    this project's actual generation_batch_size=GROUP_SIZE, one-prompt-per-step
    setup has always produced uniform groups in every run observed so far, but
    this function stays correct even if that ever changes (e.g. exactly one
    injected slot per distinct bank scenario in a mixed batch, not one total)."""
    seen_scenarios: set[tuple[str, tuple[str, ...]]] = set()
    result: dict[int, dict[str, Any]] = {}
    for idx, prompt in enumerate(prompts):
        key = scenario_key_from_prompt(prompt)
        if key is None or key not in BANK_SCENARIO_KEYS:
            continue
        if key in seen_scenarios:
            continue
        seen_scenarios.add(key)
        result[idx] = _TRAJECTORY_BY_SCENARIO[key]
    return result


def inject_into_generation_output(
    prompts: Sequence[Any],
    completion_ids: Sequence[Sequence[int]],
    completions: Sequence[str],
    tokenizer: Any,
) -> tuple[list[list[int]], list[str], dict[int, dict[str, Any]]]:
    """Given the RAW, pre-padding output of GRPOTrainer._generate (plain Python
    lists of token ids and decoded strings, one entry per rollout in the group --
    see step14b_design.md for exactly why this is the correct, minimal
    intervention point in trl's pipeline), substitute exactly one row per distinct
    bank scenario with its verified correct trajectory. Returns fresh
    (completion_ids, completions, injected_info) -- never mutates the inputs in
    place; injected_info maps index -> the trajectory row actually injected there,
    for evidence/logging in the training script.

    Everything downstream in trl's own, unmodified pipeline (padding, reference-
    model logps via this project's existing KL-clamp patch, reward computation via
    score_completion_gated, group-relative advantage computation) then operates
    naturally and consistently on the substituted batch -- no other patch in this
    project needs to change for injection to take effect correctly."""
    completion_ids = [list(ids) for ids in completion_ids]
    completions = list(completions)
    to_inject = select_injection_indices(prompts)
    for idx, row in to_inject.items():
        completion_ids[idx] = build_injected_completion_ids(tokenizer, row["completion"])
        completions[idx] = row["completion"]
    return completion_ids, completions, to_inject


# ===== GRPO configuration invariant: verify (not assume) the ratio=1 property =====

def verify_grpo_ratio_is_unconditionally_one(
    *, gradient_accumulation_steps: int, steps_per_generation: int, num_iterations: int,
) -> bool:
    """Re-derives, from trl's own condition (trl/trainer/grpo_trainer.py's
    `_generate_and_score_completions`), whether `old_per_token_logps` is set to
    None for this exact configuration -- which is what makes the importance-
    sampling ratio exactly 1 for every row (see this module's docstring, point 1).
    trl's condition for old_per_token_logps to be COMPUTED (i.e. NOT None) is:
    `gradient_accumulation_steps % (steps_per_generation * num_iterations) != 0`
    (or vLLM importance-sampling correction, which this project does not use).
    Returns True iff the ratio-is-1 property holds -- i.e. iff that condition is
    FALSE, matching this project's actual launch configuration. This is a
    programmatic safeguard: if GROUP_SIZE, num_iterations, or the generation/
    accumulation-step wiring is ever changed later without re-checking this, this
    function (and its test) will catch it rather than silently reintroducing a
    real off-policy ratio correction requirement injection was never designed to
    handle."""
    generate_every = steps_per_generation * num_iterations
    return gradient_accumulation_steps % generate_every == 0
