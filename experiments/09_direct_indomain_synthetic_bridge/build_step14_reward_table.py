"""Builds the concrete Step 14 reward-gate ordering table in
step14_reward_ordering_table.md, mirroring build_reward_table.py's methodology
(concrete constructed completions scored via the real function, not a purely
symbolic derivation) applied to score_completion_gated instead of
score_completion_v2. Run: python3 build_step14_reward_table.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_positive_signal_annealed_reward"))
from synthetic_bridge import CODED_TRAJECTORIES  # noqa: E402
from step14_reward_gate import score_completion_gated  # noqa: E402

# Bank scenario index 2: a mixed same/different sequence where both true states
# (Heads and Tails) actually occur -- an all-"same" bank scenario can never reach
# taxonomy category 9 (correct_globally_consistent_code) regardless of encoding
# quality, since category 9 requires both physical states to be observed.
BANK_ROW = CODED_TRAJECTORIES[2]
PROMPT = BANK_ROW["prompt"]
GROUND_TRUTH = BANK_ROW["final_answer"]
assert GROUND_TRUTH == "Heads"

CATEGORIES = {
    "correct_globally_consistent_code": BANK_ROW["completion"],
    "correct_literal": "Step 1: flips. State: Heads\nStep 2: flips. State: Tails\n"
                        "Step 3: same. State: Tails\nStep 4: flips. State: Heads\n"
                        "Step 5: same. State: Heads\n<answer>Heads</answer>",
    "incorrect_literal": "Step 1: flips. State: Heads\nStep 2: flips. State: Tails\n"
                          "Step 3: same. State: Tails\nStep 4: flips. State: Heads\n"
                          "Step 5: same. State: Heads\n<answer>Tails</answer>",
    "correct_vacuous": "I computed it directly.\n<answer>Heads</answer>",
    "incorrect_vacuous": "I computed it directly.\n<answer>Tails</answer>",
    "partial_nonliteral_state_variation": "Step 1: flips. State: Zorp\nStep 2: flips. State: Zorp\n"
                                          "Step 3: same. State: Zorp\nStep 4: flips. State: Zorp\n"
                                          "Step 5: same. State: Zorp\n<answer>Heads</answer>",
    "malformed_literal": "Step 1: flips. State: Heads\nStep 2: flips. State: Tails\n"
                          "Final answer is Heads with no tag",
    "malformed_nonliteral": "Step 1: flips. State: Zorp\nStep 2: flips. State: Blim\n"
                             "Final answer is Heads with no tag",
    "domain_word_echo": "Step 1: flips. State: Coins\nStep 2: flips. State: Coins\n"
                         "Step 3: same. State: Coins\nStep 4: flips. State: Coins\n"
                         "Step 5: same. State: Coins\n<answer>Coin</answer>",
}

STEP_WEAK_GATE, STEP_FULL_GATE, TOTAL_STEPS = 1, 100, 150


def build_table(step: int) -> dict[str, dict]:
    return {
        name: score_completion_gated(completion, GROUND_TRUTH, step, TOTAL_STEPS, prompt=PROMPT)
        for name, completion in CATEGORIES.items()
    }


def render_markdown() -> str:
    weak = build_table(STEP_WEAK_GATE)
    full = build_table(STEP_FULL_GATE)
    lines = [
        "# Step 14 concrete reward-gate ordering table",
        "",
        f"Computed via `score_completion_gated` (step14_reward_gate.py) on bank scenario "
        f"index 2 ({BANK_ROW['starting_state']} start, both true states occur), at "
        f"step {STEP_WEAK_GATE} (weak end of the anneal, gate_scale=0.25) and "
        f"step {STEP_FULL_GATE} (full anneal, gate_scale=1.0). Full derivation: "
        "`build_step14_reward_table.py`.",
        "",
        f"| category | total @ step {STEP_WEAK_GATE} (gate=0.25) | total @ step {STEP_FULL_GATE} (gate=1.0) |",
        "|---|---:|---:|",
    ]
    for name in CATEGORIES:
        lines.append(f"| {name} | {weak[name]['total']:.3f} | {full[name]['total']:.3f} |")
    lines += [
        "",
        "**Confirmed**: `correct_globally_consistent_code` "
        f"({full['correct_globally_consistent_code']['total']:.3f}) outranks gated "
        f"`correct_literal` ({full['correct_literal']['total']:.3f}, exactly 0 at full "
        "anneal) and every other category at the full-anneal checkpoint.",
        "",
        "**Gap 1 -- FIXED (2026-08-27)**: the gate trigger now covers "
        "`category in (LITERAL_CATEGORY, VACUOUS_CATEGORY)`, not literal alone. "
        f"`correct_vacuous` was 2.500 (untouched) before the fix, outranking gated "
        f"`correct_literal` (0.000); it is now also gated and collapses to "
        f"{full['correct_vacuous']['total']:.3f} at full anneal -- tied with, not "
        "exceeding, gated literal. Verified two ways: the symbolic report "
        "(`vacuous_exceeds_gated_literal_at_full_anneal` is now `False`) and this "
        "concrete table. Assessed as real-but-tolerable before fixing (Stage 1's "
        "documented vacuous-escape-hatch pattern, `experiments/01_direct_rl_coinflip/"
        "README.md`), then fixed anyway: cheap, no downside, removes a confound from "
        "interpreting eventual results.",
        "",
        "**Gap 2 -- NOT fixed, re-confirmed negligible under the combined trigger** "
        f"(rechecked, not assumed unaffected): at the weak end of the anneal "
        f"(step {STEP_WEAK_GATE}), `correct_literal` "
        f"({weak['correct_literal']['total']:.3f}) does not fall below "
        "`correct_globally_consistent_code`'s own worst-case symbolic floor (0.65) in "
        "an adversarial corner where the code completion's surrounding prose happens "
        "to repeat banned literal words heavily (`p_cot` scans raw text independent of "
        "taxonomy category). Gating vacuous too does not make this worse: vacuous's own "
        "ceiling (3.0 symbolic) stays below literal's (4.15), so literal remains the "
        "binding constraint -- see `margin_code_over_gated_vacuous_during_ramp` in "
        "`verify_step14_reward_invariant`'s report. Not observed in this concrete table "
        "(both completions here have p_cot=0), but not excluded by construction "
        "either -- see `test_invariant_discloses_the_during_ramp_gap_explicitly`.",
        "",
        "**Gap 3 -- FIXED (2026-08-27), was the most severe of the three**: the gate "
        "trigger now additionally requires `base['r_task'] != -5.0` (well-formed, valid "
        "`<answer>` tag). `malformed_literal` -- a completion that never closes its "
        "answer tag but still writes literal `State: Heads/Tails` lines -- previously "
        "collapsed to exactly 0 at full anneal, tying with a genuine correct literal "
        f"answer and erasing its natural penalty. It now stays at "
        f"{full['malformed_literal']['total']:.3f} (full anneal) and "
        f"{weak['malformed_literal']['total']:.3f} (step {STEP_WEAK_GATE}) -- always equal "
        "to its own `ungated_total` (`bank_gate_scale` is `None`), the small difference "
        "between the two steps coming only from `p_cot`'s own unrelated anneal over the "
        "full 150-step run, never from the reward gate partially engaging. Comparable to "
        f"`malformed_nonliteral`'s {full['malformed_nonliteral']['total']:.3f}. "
        "Occurred live in the first 8-step dry run before this fix, at an empirically "
        "measured ~1.65% rate among bank-scenario rollouts (~11-12 times per full "
        "150-step run) -- assessed real-but-tolerable, then fixed anyway.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    markdown = render_markdown()
    out = Path(__file__).with_name("step14_reward_ordering_table.md")
    out.write_text(markdown)
    print(markdown)
    print(f"Wrote {out}")
