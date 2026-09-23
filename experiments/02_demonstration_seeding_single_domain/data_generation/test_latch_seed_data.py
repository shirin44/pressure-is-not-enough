from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.latch_seed import (
    audit_latch_seed_dataset,
    generate_latch_seed_dataset,
    simulate,
    verify_example,
)


def main() -> None:
    train, heldout = generate_latch_seed_dataset()
    report = audit_latch_seed_dataset(train, heldout)
    assert report["accepted"] is True
    assert report["sizes"] == {"train": 800, "heldout": 200, "total": 1000}
    assert report["unique_prompt_percentage"] == 100.0
    assert report["unique_token_pair_percentage"] == 100.0
    assert report["max_individual_token_frequency"] == 1
    assert report["mapping_orientation"] == {
        "locked_is_alphabetically_first": 500,
        "locked_is_alphabetically_second": 500,
    }
    assert report["banned_substring_violation_count"] == 0
    assert report["semantic_verification_pass_rate"] == 100.0
    assert report["prompt_mapping_verification_pass_rate"] == 100.0
    assert report["decode_back_verification_pass_rate"] == 100.0
    assert report["mapping_anchor_verification_pass_rate"] == 100.0
    assert report["explicit_transition_verification_pass_rate"] == 100.0
    assert report["train_initial_states"] == {"Locked": 400, "Unlocked": 400}
    assert report["heldout_initial_states"] == {"Locked": 100, "Unlocked": 100}
    assert report["train_final_answers"] == {"Locked": 400, "Unlocked": 400}
    assert report["heldout_final_answers"] == {"Locked": 100, "Unlocked": 100}

    # The verifier must fail loudly when any part of the worked trace is
    # corrupted, independently of the generator that created it.
    valid, errors = verify_example(train[0])
    assert valid and not errors
    corrupted_token = replace(
        train[0], demonstration=train[0].demonstration.replace("State: ", "State: Wrong", 1)
    )
    assert verify_example(corrupted_token)[0] is False
    corrupted_previous = replace(
        train[0], demonstration=train[0].demonstration.replace(
            "Previous code: ", "Previous code: Wrong", 1
        )
    )
    previous_valid, previous_errors = verify_example(corrupted_previous)
    assert not previous_valid
    assert any(error.startswith("wrong_previous_code_token_") for error in previous_errors)
    corrupted_prompt_mapping = replace(
        train[0],
        prompt=train[0].prompt.replace(
            f"Represent Locked using the code {train[0].token_for_locked}",
            "Represent Locked using the code Wrongtoken",
            1,
        ),
    )
    assert verify_example(corrupted_prompt_mapping)[0] is False
    expected_initial_token = (
        train[0].token_for_locked
        if train[0].initial_state == "Locked"
        else train[0].token_for_unlocked
    )
    corrupted_anchor = replace(
        train[0],
        prompt=train[0].prompt.replace(
            f"Therefore, the initial code is {expected_initial_token}.",
            "Therefore, the initial code is Wrongtoken.",
            1,
        ),
    )
    anchor_valid, anchor_errors = verify_example(corrupted_anchor)
    assert not anchor_valid and "wrong_mapping_anchor_token" in anchor_errors
    missing_anchor = replace(
        train[0],
        prompt=train[0].prompt.replace(
            f"Therefore, the initial code is {expected_initial_token}.\n", "", 1
        ),
    )
    missing_valid, missing_errors = verify_example(missing_anchor)
    assert not missing_valid and "missing_or_duplicate_mapping_anchor" in missing_errors
    final_token = (
        train[0].token_for_locked
        if train[0].final_answer == "Locked"
        else train[0].token_for_unlocked
    )
    corrupted_decode = replace(
        train[0],
        demonstration=train[0].demonstration.replace(
            f"Final coded state: {final_token}.", "Final coded state: Wrongtoken.", 1
        ),
    )
    assert verify_example(corrupted_decode)[0] is False
    wrong_answer = replace(
        train[0], demonstration=train[0].demonstration.replace(
            f"<answer>{train[0].final_answer}</answer>",
            f"<answer>{'Unlocked' if train[0].final_answer == 'Locked' else 'Locked'}</answer>",
        )
    )
    assert verify_example(wrong_answer)[0] is False
    assert simulate("Locked", ("same", "different", "same")) == [
        "Locked",
        "Unlocked",
        "Unlocked",
    ]
    print("PASSED: latch seeding generator and independent semantic verifier")


if __name__ == "__main__":
    main()
