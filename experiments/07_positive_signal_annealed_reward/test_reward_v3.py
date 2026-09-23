"""Unit tests for reward_v3.py: r_signal (content-agnostic positive-signaling reward),
annealed_cot_scale, and the re-derived full-reward invariant. Runs before any model
involvement, per this stage's staged-validation requirement."""
import math

from reward_v3 import (
    _adjusted_rand_index,
    annealed_cot_scale,
    consistency_bonus_v2,
    score_completion_v2,
    signal_bonus,
    signal_correlation,
    state_variation_penalty,
    verify_reward_invariant,
)


def _prompt(starting_state: str, operations: list[str]) -> str:
    lines = [f"Starting state: {starting_state}", "Instructions:"]
    clarified = {
        "same": "same as previous (the state does NOT change)",
        "different": "different from previous (the state flips)",
    }
    for i, op in enumerate(operations, start=1):
        lines.append(f"{i}. {clarified[op]}")
    return "\n".join(lines)


def _completion(tokens: dict[int, str], answer: str) -> str:
    lines = [f"Step {i}: reasoning. State: {tok}" for i, tok in sorted(tokens.items())]
    lines.append(f"<answer>{answer}</answer>")
    return "\n".join(lines)


def _true_states(starting_state: str, operations: list[str]) -> dict[int, str]:
    state = starting_state.casefold()
    out = {}
    for i, op in enumerate(operations, start=1):
        if op == "different":
            state = "tails" if state == "heads" else "heads"
        out[i] = state
    return out


# ===== ARI / signal_correlation: core correctness =====

def test_ari_perfect_two_cluster_mapping_is_one():
    assert math.isclose(_adjusted_rand_index(["a", "a", "b", "b"], ["x", "x", "y", "y"]), 1.0)


def test_ari_all_same_token_is_exactly_zero():
    # Both true states present, but every token identical -> chance level, not a bug.
    assert math.isclose(_adjusted_rand_index(["a", "a", "b", "b"], ["x", "x", "x", "x"]), 0.0, abs_tol=1e-12)


def test_ari_all_distinct_tokens_is_exactly_zero():
    assert math.isclose(_adjusted_rand_index(["a", "a", "b", "b"], ["w", "x", "y", "z"]), 0.0, abs_tol=1e-12)


def test_signal_correlation_perfect_mapping():
    ops = ["different", "same", "different", "same"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {i: ("Zorp" if s == "heads" else "Blim") for i, s in true_states.items()}
    completion = _completion(tokens, "Heads")
    assert math.isclose(signal_correlation(completion, prompt), 1.0)


def test_signal_correlation_single_token_is_zero():
    ops = ["different", "same", "different", "same"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {i: "Zorp" for i in true_states}
    completion = _completion(tokens, "Heads")
    assert math.isclose(signal_correlation(completion, prompt), 0.0, abs_tol=1e-12)


def test_signal_correlation_only_one_physical_state_observed_is_zero():
    # All "same as previous" -> only one true state ever occurs; no correlation is measurable.
    ops = ["same", "same", "same"]
    prompt = _prompt("Heads", ops)
    completion = _completion({1: "Zorp", 2: "Zorp", 3: "Zorp"}, "Heads")
    assert signal_correlation(completion, prompt) == 0.0


def test_signal_correlation_insufficient_slots_is_zero():
    prompt = _prompt("Heads", ["different"])
    assert signal_correlation("<answer>Tails</answer>", prompt) == 0.0  # zero valid slots
    assert signal_correlation("Step 1: x. State: Zorp\n<answer>Tails</answer>", prompt) == 0.0  # one slot


# ===== Content-agnostic requirement: arbitrary interchangeable tokens score identically =====

def test_content_agnostic_relabeling_gives_identical_score():
    ops = ["different", "same", "different", "different", "same"]
    prompt = _prompt("Tails", ops)
    true_states = _true_states("Tails", ops)

    mapping_a = {"heads": "Zorp", "tails": "Blim"}
    mapping_b = {"heads": "Foo", "tails": "Bar"}
    mapping_c = {"heads": "Xyzzy", "tails": "Plugh"}

    scores = []
    for mapping in (mapping_a, mapping_b, mapping_c):
        tokens = {i: mapping[s] for i, s in true_states.items()}
        completion = _completion(tokens, true_states[len(ops)].capitalize())
        scores.append(signal_correlation(completion, prompt))

    assert all(math.isclose(s, scores[0]) for s in scores), scores
    assert math.isclose(scores[0], 1.0)


def test_content_agnostic_partial_correlation_relabeling_also_matches():
    # A deliberately imperfect (partially-correlated) mapping: same PARTITION structure
    # under two different token vocabularies must give the same ARI either way.
    ops = ["different", "same", "different", "different", "same", "different"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    indices = sorted(true_states)

    # Flip the token for exactly one index, to make it imperfect but not degenerate --
    # applied identically (by position) under two different token vocabularies.
    def build(vocab):
        tokens = {i: vocab[true_states[i]] for i in indices}
        tokens[indices[2]] = vocab["tails"] if true_states[indices[2]] == "heads" else vocab["heads"]
        return tokens

    vocab_1 = {"heads": "Alpha", "tails": "Omega"}
    vocab_2 = {"heads": "Nib", "tails": "Vex"}
    c1 = _completion(build(vocab_1), "Heads")
    c2 = _completion(build(vocab_2), "Heads")
    ari1 = signal_correlation(c1, prompt)
    ari2 = signal_correlation(c2, prompt)
    assert math.isclose(ari1, ari2), (ari1, ari2)
    assert 0.0 < ari1 < 1.0, "expected a genuinely partial (imperfect) correlation for this test to be meaningful"


# ===== The key differentiator: partial credit where the all-or-nothing terms give zero =====

def test_partial_incomplete_trace_gets_signal_credit_but_zero_from_all_or_nothing_terms():
    # 5-flip prompt, but only steps 1 and 3 have valid slots (2, 4, 5 missing) -- an
    # incomplete trace. The two present slots correctly correlate with true state.
    ops = ["different", "same", "different", "same", "different"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {1: ("Zorp" if true_states[1] == "heads" else "Blim"),
              3: ("Zorp" if true_states[3] == "heads" else "Blim")}
    completion = _completion(tokens, "Tails")

    r_signal = signal_bonus(completion, prompt, magnitude=0.15)
    r_consistency = consistency_bonus_v2(completion, num_flips=5, magnitude=0.15)
    p_state_variation = state_variation_penalty(completion, prompt, num_flips=5)

    assert r_signal > 0.0, "r_signal should give partial credit for the 2 correctly-correlated slots"
    assert r_consistency == 0.0, "r_consistency requires full coverage -- must be zero here"
    assert p_state_variation == 0.5, "p_state_variation requires full coverage -- must be at its penalty here"


def test_signal_bonus_is_bounded_by_magnitude():
    ops = ["different"] * 6
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {i: ("Zorp" if s == "heads" else "Blim") for i, s in true_states.items()}
    completion = _completion(tokens, true_states[6].capitalize())
    for magnitude in (0.0, 0.05, 0.15, 0.5, 1.0):
        bonus = signal_bonus(completion, prompt, magnitude=magnitude)
        assert 0.0 <= bonus <= magnitude + 1e-12


def test_signal_bonus_never_negative_even_for_anticorrelation():
    # Deliberately anti-correlated: alternate tokens on a pattern that mismatches true state
    # transitions, driving ARI negative -- must still clip to a non-negative reward.
    ops = ["different", "different", "same", "different", "same", "different"]
    prompt = _prompt("Heads", ops)
    tokens = {1: "A", 2: "B", 3: "A", 4: "B", 5: "A", 6: "B"}
    completion = _completion(tokens, "Heads")
    ari = signal_correlation(completion, prompt)
    bonus = signal_bonus(completion, prompt, magnitude=0.15)
    assert bonus >= 0.0
    if ari < 0:
        assert bonus == 0.0


# ===== annealed_cot_scale =====

def test_annealed_cot_scale_flat_during_warmup():
    for step in range(5):
        assert annealed_cot_scale(step, total_steps=150, warmup_steps=5, min_scale=0.2, max_scale=2.0) == 0.2


def test_annealed_cot_scale_reaches_max_at_total_steps():
    assert math.isclose(
        annealed_cot_scale(150, total_steps=150, warmup_steps=5, min_scale=0.2, max_scale=2.0), 2.0
    )


def test_annealed_cot_scale_monotonic_nondecreasing():
    values = [annealed_cot_scale(s, total_steps=150, warmup_steps=5) for s in range(0, 151, 3)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_annealed_cot_scale_linear_at_midpoint():
    total, warmup, lo, hi = 100, 0, 0.2, 2.0
    mid = annealed_cot_scale(50, total_steps=total, warmup_steps=warmup, min_scale=lo, max_scale=hi)
    assert math.isclose(mid, lo + (hi - lo) * 0.5)


def test_annealed_cot_scale_holds_at_max_past_total_steps():
    assert annealed_cot_scale(500, total_steps=150, warmup_steps=5, min_scale=0.2, max_scale=2.0) == 2.0


def test_annealed_cot_scale_no_hard_jump_at_old_tier_boundaries():
    # The old design jumped discretely at steps 5, 10, 30. Confirm the new schedule changes
    # smoothly (small input delta -> small output delta) across each of those points.
    for boundary in (5, 10, 30):
        before = annealed_cot_scale(boundary - 1, total_steps=150, warmup_steps=5)
        after = annealed_cot_scale(boundary + 1, total_steps=150, warmup_steps=5)
        assert abs(after - before) < 0.05, f"unexpectedly large jump across old tier boundary {boundary}"


# ===== Full reward invariant =====

def test_reward_invariant_holds_at_defaults():
    margins = verify_reward_invariant()
    assert margins["margin_correct_over_wrong"] > 0
    assert margins["margin_wrong_over_malformed"] > 0
    assert math.isclose(margins["margin_correct_over_wrong"], 0.70, abs_tol=1e-9)
    assert math.isclose(margins["margin_wrong_over_malformed"], 0.70, abs_tol=1e-9)


def test_reward_invariant_breaks_if_signal_magnitude_too_large():
    # Sanity-check the derivation itself: if signal_magnitude alone exceeds the base 0.85
    # margin (i.e. consistency_magnitude + signal_magnitude >= 0.85 relative to the bare 4.5
    # r_task gap minus the 2.0+0.5+0.5+0.5 penalty ceiling), the invariant must correctly fail.
    try:
        verify_reward_invariant(consistency_magnitude=0.15, signal_magnitude=10.0)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected verify_reward_invariant to raise for an oversized signal_magnitude")


def test_score_completion_v2_includes_r_signal_and_uses_annealed_scale():
    ops = ["different", "same", "different"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {i: ("Zorp" if s == "heads" else "Blim") for i, s in true_states.items()}
    completion = _completion(tokens, true_states[3].capitalize())

    early = score_completion_v2(completion, true_states[3].capitalize(), step=0, total_steps=150, prompt=prompt)
    late = score_completion_v2(completion, true_states[3].capitalize(), step=150, total_steps=150, prompt=prompt)

    assert early["r_signal"] > 0.0
    assert math.isclose(early["r_signal"], late["r_signal"]), "r_signal must not depend on the anneal schedule"
    assert early["cot_scale"] < late["cot_scale"], "p_CoT weight must ramp up over training"
    assert math.isclose(early["cot_scale"], 0.2)
    assert math.isclose(late["cot_scale"], 2.0)


def test_score_completion_v2_malformed_still_ranks_lowest_with_max_bonuses():
    # A malformed (no <answer> tag) completion that nonetheless has perfect consistency AND
    # perfect signal correlation must still score below every valid wrong completion, and
    # every valid wrong completion must still score below every correct one.
    ops = ["different", "same", "different", "same"]
    prompt = _prompt("Heads", ops)
    true_states = _true_states("Heads", ops)
    tokens = {i: ("Zorp" if s == "heads" else "Blim") for i, s in true_states.items()}

    malformed = "\n".join(f"Step {i}: reasoning. State: {tok}" for i, tok in sorted(tokens.items()))
    wrong_answer = "Heads" if true_states[4] == "tails" else "Tails"
    correct = _completion(tokens, true_states[4].capitalize())
    wrong = _completion(tokens, wrong_answer)

    s_malformed = score_completion_v2(malformed, true_states[4].capitalize(), step=100, total_steps=150, prompt=prompt)
    s_wrong = score_completion_v2(wrong, true_states[4].capitalize(), step=100, total_steps=150, prompt=prompt)
    s_correct = score_completion_v2(correct, true_states[4].capitalize(), step=100, total_steps=150, prompt=prompt)

    assert s_malformed["r_signal"] > 0.0 and s_wrong["r_signal"] > 0.0 and s_correct["r_signal"] > 0.0
    assert s_correct["total"] > s_wrong["total"] > s_malformed["total"]
