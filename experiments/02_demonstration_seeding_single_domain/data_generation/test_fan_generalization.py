from src.data.fan_generalization import (
    generate_declared_mapping_control,
    generate_fan_evaluation,
    score_completion,
    score_declared_completion,
    verify_corpus,
    verify_declared_control,
)


def test_corpus_and_scoring() -> None:
    rows = generate_fan_evaluation()
    audit = verify_corpus(rows)
    assert audit["accepted"] and audit["semantic_pass_rate"] == 100
    assert audit["initial_states"] == {"Running": 50, "Stopped": 50}
    assert audit["final_answers"] == {"Running": 50, "Stopped": 50}
    row = rows[0]
    mapping = {"Running": "Zorp", "Stopped": "Vex"}
    lines = [
        f"Step {index}: State: {mapping[state]}"
        for index, state in enumerate(row.expected_states, 1)
    ]
    text = "\n".join([*lines, f"<answer>{row.final_answer}</answer>"])
    score = score_completion(row, text)
    assert score["structural"]
    assert score["nonliteral_consistent"]
    assert score["answer_correct"]
    assert not score["answer_is_code_word"]
    assert score["token_pair"] == ("vex", "zorp")
    literal_text = text.replace("Zorp", "Running").replace("Vex", "Stopped")
    assert not score_completion(row, literal_text)["nonliteral"]
    leak_text = text.replace("Step 1:", "Step 1: latch")
    assert score_completion(row, leak_text)["latch_coin_leakage"]
    code_answer = text.replace(
        f"<answer>{row.final_answer}</answer>", "<answer>Zorp</answer>"
    )
    code_score = score_completion(row, code_answer)
    assert code_score["answer_is_code_word"]
    assert code_score["answer_failure_type"] == "code_word_instead_of_physical_state"


def test_declared_mapping_control() -> None:
    rows = generate_declared_mapping_control()
    audit = verify_declared_control(rows)
    assert audit["accepted"] and audit["semantic_pass_rate"] == 100
    assert audit["unique_token_pairs"] == 100
    row = rows[0]
    mapping = {"Running": row.token_for_running, "Stopped": row.token_for_stopped}
    text = "\n".join([
        *[f"Step {index}: State: {mapping[state]}"
          for index, state in enumerate(row.base.expected_states, 1)],
        f"<answer>{row.base.final_answer}</answer>",
    ])
    score = score_declared_completion(row, text)
    assert score["declared_mapping_adherence"]
    assert score["nonliteral_consistent"] and score["answer_correct"]
    declared_code_answer = text.replace(
        f"<answer>{row.base.final_answer}</answer>",
        f"<answer>{row.token_for_running}</answer>",
    )
    declared_code_score = score_declared_completion(row, declared_code_answer)
    assert declared_code_score["answer_is_code_word"]
