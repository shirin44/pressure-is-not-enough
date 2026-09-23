from src.data.multidomain_seed import audit_dataset, generate_multidomain_dataset, verify_example


def test_multidomain_corpus() -> None:
    training, evaluations = generate_multidomain_dataset()
    audit = audit_dataset(training, evaluations)
    assert audit["accepted"]
    assert audit["semantic_pass_rate"] == 100
    assert audit["decode_back_training_coverage"] == 1.0
    assert all(verify_example(row)[0] for row in training)
    assert all(training[i].domain != training[i + 1].domain for i in range(len(training) - 1))
    train_signatures={(row.domain,row.initial_state,row.operations) for row in training}
    eval_signatures={(row.domain,row.initial_state,row.operations)
                     for rows in evaluations.values() for row in rows}
    assert not train_signatures & eval_signatures


def test_corrected_fan_wording_is_strictly_localized() -> None:
    old_training, old_evaluations = generate_multidomain_dataset()
    new_training, new_evaluations = generate_multidomain_dataset(corrected_fan_wording=True)
    assert [row.example_id for row in old_training] == [row.example_id for row in new_training]
    changed = 0
    for old, new in zip(old_training, new_training):
        assert (old.domain, old.initial_state, old.operations, old.expected_states,
                old.token_for_first_state, old.token_for_second_state) == (
                    new.domain, new.initial_state, new.operations, new.expected_states,
                    new.token_for_first_state, new.token_for_second_state)
        if old.domain == "fan":
            if "different" in old.operations:
                assert "fan flips to the opposite state" in new.prompt
                assert "The fan flips to its opposite state" in new.demonstration
                assert old.prompt != new.prompt
            else:
                assert old == new
            changed += old.prompt != new.prompt
        else:
            assert old == new
    assert changed == sum(row.domain == "fan" and "different" in row.operations
                          for row in old_training)
    assert old_evaluations["valve"] == new_evaluations["valve"]
    assert old_evaluations["lamp"] == new_evaluations["lamp"]
    assert audit_dataset(new_training, new_evaluations)["accepted"]
