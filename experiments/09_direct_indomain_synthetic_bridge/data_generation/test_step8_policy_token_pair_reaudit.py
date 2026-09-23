from step8_policy_token_pair_reaudit import summarize_pairs


def _row(token, lp, eligible=True, leak=True, edit=3):
    return {
        "token": token,
        "step0_policy_logprob_at_state_slot": lp,
        "eligible": eligible,
        "passes_leakage_classifier": leak,
        "min_edit_distance_to_banned_word": edit,
    }


def test_summarize_selected_pair_gaps_and_static_gates():
    rows = [
        _row("Nib", -6.0), _row("Nomo", -6.25),
        _row("Yelt", -7.0), _row("Yark", -7.0),
    ]
    result = summarize_pairs(rows)
    assert result["training"]["step0_policy_abs_gap"] == 0.25
    assert result["training"]["original_base_abs_gap"] == 0.0
    assert result["training"]["both_static_eligible"] is True
    assert result["held_out"]["step0_policy_abs_gap"] == 0.0
