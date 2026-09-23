from src.data.fan_generalization import FanExample, score_completion


EXAMPLE = FanExample(
    example_id="synthetic",
    initial_state="Running",
    operations=("different", "same", "different"),
    expected_states=("Stopped", "Stopped", "Running"),
    final_answer="Running",
    prompt="synthetic",
)


def completion(tokens, decode_token="Zorp", decode_state="Running"):
    lines = [f"Step {i}: State: {token}" for i, token in enumerate(tokens, 1)]
    lines += [
        f"Final coded state: {decode_token}. {decode_token} represents {decode_state}.",
        "<answer>Running</answer>",
    ]
    return "\n".join(lines)


def test_consistent_undeclared_two_token_trace_passes():
    score = score_completion(EXAMPLE, completion(("Kavi", "Kavi", "Zorp")))
    assert score["global_consistent"]
    assert score["decode_back_self_consistent"]
    assert score["nonliteral_consistent"]
    assert score["token_pair"] == ("kavi", "zorp")


def test_same_physical_state_cannot_change_token():
    score = score_completion(EXAMPLE, completion(("Kavi", "Mero", "Zorp")))
    assert not score["global_consistent"]
    assert not score["decode_back_self_consistent"]


def test_three_tokens_fail_even_if_final_decode_matches():
    score = score_completion(EXAMPLE, completion(("Kavi", "Mero", "Zorp")))
    assert len(set(score["tokens"])) == 3
    assert not score["global_consistent"]


def test_wrong_decode_token_fails_self_consistency():
    score = score_completion(
        EXAMPLE, completion(("Kavi", "Kavi", "Zorp"), decode_token="Kavi")
    )
    assert score["global_consistent"]
    assert not score["decode_back_self_consistent"]


def test_wrong_decode_state_fails_self_consistency():
    score = score_completion(
        EXAMPLE, completion(("Kavi", "Kavi", "Zorp"), decode_state="Stopped")
    )
    assert score["global_consistent"]
    assert not score["decode_back_self_consistent"]
