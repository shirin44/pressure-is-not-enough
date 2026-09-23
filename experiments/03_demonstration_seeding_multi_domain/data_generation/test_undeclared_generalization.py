from src.data.undeclared_generalization import UndeclaredExample, score_undeclared_completion


ROW = UndeclaredExample(
    "test", "lamp", "Lit", ("different", "same", "different"),
    ("Dark", "Dark", "Lit"), "Lit", "prompt",
)


def output(tokens, decode_token="Zorp", decode_state="Lit"):
    lines = [f"Step {i}: tracking. State: {token}" for i, token in enumerate(tokens, 1)]
    lines += [f"Final coded state: {decode_token}. {decode_token} represents {decode_state}.",
              "<answer>Lit</answer>"]
    return "\n".join(lines)


def test_genuinely_consistent_undeclared_trace():
    score = score_undeclared_completion(ROW, output(("Kavi", "Kavi", "Zorp")))
    assert score["global_consistent"] and score["nonliteral_consistent"]
    assert score["decode_back_self_consistent"] and score["answer_correct"]
    assert score["token_pair"] == ("kavi", "zorp")


def test_inconsistent_same_state_reuse_fails():
    score = score_undeclared_completion(ROW, output(("Kavi", "Mero", "Zorp")))
    assert not score["global_consistent"]
    assert not score["decode_back_self_consistent"]


def test_three_token_trace_fails():
    score = score_undeclared_completion(ROW, output(("Kavi", "Mero", "Zorp")))
    assert score["unique_token_count"] == 3
    assert not score["global_consistent"]


def test_wrong_decode_back_token_fails():
    score = score_undeclared_completion(ROW, output(("Kavi", "Kavi", "Zorp"), "Kavi"))
    assert score["global_consistent"] and not score["decode_back_self_consistent"]


def test_wrong_decode_back_state_fails():
    score = score_undeclared_completion(ROW, output(("Kavi", "Kavi", "Zorp"), decode_state="Dark"))
    assert score["global_consistent"] and not score["decode_back_self_consistent"]
