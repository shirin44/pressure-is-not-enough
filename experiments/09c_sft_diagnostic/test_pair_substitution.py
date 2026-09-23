import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_substitution import build_prompt_with_substitution, build_substitution_instruction, score_pair_substitution

TRUE_STATES_ALL_HEADS = ["Heads"] * 5
TRUE_STATES_MIXED = ["Heads", "Tails", "Tails", "Heads", "Heads"]


def _completion(tokens, final):
    lines = [f"Step {i+1}: whatever reasoning. State: {t}" for i, t in enumerate(tokens)]
    lines.append(f"<answer>{final}</answer>")
    return "\n".join(lines)


def test_fully_correct_substitution():
    comp = _completion(["Yelt", "Yark", "Yark", "Yelt", "Yelt"], "Yelt")
    r = score_pair_substitution(comp, TRUE_STATES_MIXED, "Yelt", "Yark")
    assert r["per_slot_correct"] is True
    assert r["final_answer_correct"] is True
    assert r["stayed_within_instructed_pair"] is True
    assert r["fully_correct"] is True
    assert r["lapsed_into_literal_heads_tails"] is False


def test_wrong_slot_values():
    comp = _completion(["Yelt", "Yelt", "Yark", "Yelt", "Yelt"], "Yelt")  # slot 2 wrong
    r = score_pair_substitution(comp, TRUE_STATES_MIXED, "Yelt", "Yark")
    assert r["per_slot_correct"] is False
    assert r["fully_correct"] is False


def test_wrong_final_answer():
    comp = _completion(["Yelt", "Yark", "Yark", "Yelt", "Yelt"], "Yark")  # final wrong
    r = score_pair_substitution(comp, TRUE_STATES_MIXED, "Yelt", "Yark")
    assert r["per_slot_correct"] is True
    assert r["final_answer_correct"] is False
    assert r["fully_correct"] is False


def test_lapse_into_literal_heads_tails_detected():
    comp = _completion(["Yelt", "Tails", "Yark", "Yelt", "Yelt"], "Yelt")
    r = score_pair_substitution(comp, TRUE_STATES_MIXED, "Yelt", "Yark")
    assert r["lapsed_into_literal_heads_tails"] is True
    assert r["stayed_within_instructed_pair"] is False
    assert r["fully_correct"] is False


def test_lapse_into_memorized_training_pair_detected_as_other_word():
    # Nib/Nomo is a DIFFERENT word than the instructed Yelt/Yark -- must be flagged too,
    # not silently treated as "within the instructed pair" just because it's not literal.
    comp = _completion(["Yelt", "Nomo", "Yark", "Yelt", "Yelt"], "Yelt")
    r = score_pair_substitution(comp, TRUE_STATES_MIXED, "Yelt", "Yark")
    assert r["stayed_within_instructed_pair"] is False
    assert r["fully_correct"] is False


def test_vacuous_completion_no_slots():
    comp = "I am not sure how to answer this.\n<answer>Yelt</answer>"
    r = score_pair_substitution(comp, TRUE_STATES_ALL_HEADS, "Yelt", "Yark")
    assert r["actual_tokens"] == []
    assert r["per_slot_correct"] is False
    assert r["stayed_within_instructed_pair"] is False  # empty actual_tokens -> not "stayed within"


def test_all_heads_scenario_expected_tokens():
    r = score_pair_substitution(_completion(["Yelt"]*5, "Yelt"), TRUE_STATES_ALL_HEADS, "Yelt", "Yark")
    assert r["expected_tokens"] == ["yelt"]*5
    assert r["fully_correct"] is True


def test_build_substitution_instruction_mentions_both_words():
    instr = build_substitution_instruction("Yelt", "Yark")
    assert "Yelt" in instr and "Yark" in instr
    assert "Heads" in instr and "Tails" in instr  # explains what they replace


def test_build_prompt_with_substitution_appends_not_replaces():
    base = "Starting state: Heads\nInstructions:\n1. same as previous\nReason through it."
    full = build_prompt_with_substitution(base, "Yelt", "Yark")
    assert full.startswith(base)
    assert "Yelt" in full and "Yark" in full


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
