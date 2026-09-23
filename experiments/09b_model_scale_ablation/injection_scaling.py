"""Stage 9b: generalize Step 14b's 1-of-8 rollout injection to N-of-8 (design
requirement 3: try 2-of-8 first, 3-of-8 as a same-pass fallback if appetite remains).

step14b_injection.py's own select_injection_indices()/inject_into_generation_output()
are structurally hardcoded to exactly one injected index per DISTINCT bank scenario
seen in a group (`if key in seen_scenarios: continue` -- a dedup, not a count parameter).
Every training group in this project has always been uniform (generation_batch_size=
GROUP_SIZE, one prompt per physical step -- documented explicitly in that module's own
select_injection_indices() docstring), so in practice that function injects exactly 1
of the group's GROUP_SIZE=8 rollouts. INJECTIONS_PER_GROUP=1 in that module is
documentation/an assertion target for its own tests, not a parameter the selection
logic actually reads.

Rather than edit the shared step14b_injection.py (used by Stage 9's own evidence-
preserving Step 14b scripts), this module reuses its verified building blocks
(injected_trajectory_for_prompt, build_injected_completion_ids) and adds a genuinely
N-parameterized selection/injection pair for Stage 9b's own scripts. With n=1 this is
provably byte-identical in behavior to step14b_injection.py's own functions on a
uniform (all-same-scenario) group -- verified in test_injection_scaling.py, not just
asserted here.
"""
from __future__ import annotations

from typing import Any, Sequence

from step14b_injection import build_injected_completion_ids, injected_trajectory_for_prompt


def select_injection_indices_n(prompts: Sequence[Any], n: int) -> dict[int, dict[str, Any]]:
    """First n indices among `prompts` whose scenario is a gated bank scenario,
    each mapped to that scenario's verified-correct trajectory row. Unlike step14b's
    single-injection selector, this does NOT dedup by distinct scenario -- within a
    uniform group (the only kind this project's generation config ever produces),
    all n selected indices share the same scenario/trajectory by construction, which
    is exactly the intended "N guaranteed-correct copies in this comparison group"
    behavior. If fewer than n indices in `prompts` are bank scenarios (only possible
    in a non-uniform group), returns however many bank-scenario indices exist -- never
    pads with a different scenario's trajectory.
    """
    if n < 0:
        raise ValueError(f'n must be >= 0, got {n}')
    result: dict[int, dict[str, Any]] = {}
    for idx, prompt in enumerate(prompts):
        if len(result) >= n:
            break
        traj = injected_trajectory_for_prompt(prompt)
        if traj is None:
            continue
        result[idx] = traj
    return result


def inject_into_generation_output_n(
    prompts: Sequence[Any],
    completion_ids: Sequence[Sequence[int]],
    completions: Sequence[str],
    tokenizer: Any,
    n: int,
) -> tuple[list[list[int]], list[str], dict[int, dict[str, Any]]]:
    """N-of-group generalization of step14b_injection.inject_into_generation_output.
    Same non-mutating, fresh-list-returning contract; same injection mechanics
    (build_injected_completion_ids) reused unchanged."""
    completion_ids = [list(ids) for ids in completion_ids]
    completions = list(completions)
    to_inject = select_injection_indices_n(prompts, n)
    for idx, row in to_inject.items():
        completion_ids[idx] = build_injected_completion_ids(tokenizer, row['completion'])
        completions[idx] = row['completion']
    return completion_ids, completions, to_inject
