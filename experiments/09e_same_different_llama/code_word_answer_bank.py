"""Stage 9e, Llama-3-8B-Instruct, revised design: direct code-word answer.

The Same/Different task is ABANDONED (see design.md) -- Part A's zero-shot capacity
check found the failure isolated specifically to the final relational-comparison
step (15/16 completions showed perfect intermediate tracking, but the model
defaulted to "Same" regardless of truth), a task-design problem unrelated to the
actual research question this project exists to answer.

Revised design: return to the ORIGINAL Coin Flip mechanics (track Heads/Tails through
a same/different instruction sequence, exactly as `synthetic_bridge.py` already does)
but change the FINAL ANSWER to be the code word itself (Bek or Ner) -- no translation
step, no decode-back line, no comparison. Correctness is checked externally, against
the known Bek=Heads / Ner=Tails mapping, never requiring the model to state the
literal word anywhere in a well-formed completion.

Reuses `_trace()` (ground truth) and the full 64-scenario length-5 universe
unchanged. Does NOT reuse `HEADS_CODE`/`TAILS_CODE` ('Nib'/'Nomo', Qwen's pair) --
this bank is for Llama's own audited training pair (Bek/Ner, see design.md Decision 1
of the prior Same/Different task, carried over unchanged).
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from synthetic_bridge import N_FLIPS, _trace  # noqa: E402

# Llama's audited, validated training pair (Part A Decision 1) -- arbitrary mapping
# direction, picked once and applied consistently everywhere in this module.
HEADS_CODE = 'Bek'
TAILS_CODE = 'Ner'
CODE_FOR = {'Heads': HEADS_CODE, 'Tails': TAILS_CODE}
STATE_FOR_CODE = {HEADS_CODE: 'Heads', TAILS_CODE: 'Tails'}

ALL_SCENARIOS = [
    (starting_state, tuple(operations))
    for starting_state in ('Heads', 'Tails')
    for operations in itertools.product(('same', 'different'), repeat=N_FLIPS)
]


def build_code_word_prompt(starting_state: str, operations: list[str]) -> str:
    """Same rigid-template wording as the original task (and the abandoned
    Same/Different variant) -- the ONLY change from the original Heads/Tails task is
    the final instruction: answer with the CODE WORD directly, not the literal state."""
    lines = [f'Starting state: {starting_state}', 'Instructions:']
    wording = {
        'same': 'same as previous (the state does NOT change)',
        'different': 'different from previous (the state flips)',
    }
    for i, operation in enumerate(operations, 1):
        lines.append(f'{i}. {wording[operation]}')
    lines.append(
        "Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        f"tags for intermediate states. After all steps, give the final tracked state "
        f"inside <answer>...</answer> tags."
    )
    return '\n'.join(lines)


def build_code_word_completion(starting_state: str, operations: list[str]) -> str:
    """Tracks in code (Bek/Ner) at every step, exactly like build_coded_completion()
    in synthetic_bridge.py -- but the final line is JUST the code word, no decode-back
    sentence, no literal word anywhere. `<answer> Bek</answer>` or `<answer> Ner</answer>`,
    never `<answer>Heads</answer>` or `<answer>Tails</answer>`.

    IMPORTANT, checked directly (not assumed) before finalizing this format: without a
    leading space, Llama's BPE tokenizer splits the code word inside the answer tag
    into TWO tokens (e.g. '<answer>Bek</answer>' tokenizes '>Bek' as '>B'+'ek' -- the
    code word itself is NOT single-token in that context, even though Part A's audit
    confirmed it single-token for the 'State: <token>' context, which always has a
    leading space). This would have smeared the final-answer occurrence across two
    activation positions -- exactly the confound Decision 1's single-token requirement
    exists to avoid, just relocated to the answer tag instead of the state-tracking
    lines. Fixed by adding a leading space directly inside the tag ('<answer> Bek'):
    confirmed this tokenizes the code word identically (same token ID) to its
    with-leading-space intermediate-step occurrences. _extract_answer()'s existing
    regex (`<answer>\\s*(...)\\s*</answer>`) already tolerates this whitespace, so no
    change to the scoring pipeline is needed."""
    states = _trace(starting_state, operations)
    lines = []
    for step, (operation, state) in enumerate(zip(operations, states), 1):
        token = CODE_FOR[state]
        if step == 1:
            lines.append(f'Step 1: The state is tracked. State: {token}')
        else:
            action = 'remains unchanged' if operation == 'same' else 'changes'
            lines.append(f'Step {step}: The code {action}. State: {token}')
    final_code = CODE_FOR[states[-1]]
    lines.append(f'<answer> {final_code}</answer>')
    return '\n'.join(lines)


def verify_code_word_trajectory(starting_state: str, operations: list[str]) -> dict[str, Any]:
    """Independently recomputes ground truth from starting_state/operations alone
    (never trusts the completion's own content) and checks: (1) every intermediate
    code-word matches the true state via CODE_FOR, (2) the final answer code-word
    matches the true final state, (3) no literal Heads/Tails word appears anywhere."""
    completion = build_code_word_completion(starting_state, operations)
    true_states = _trace(starting_state, operations)
    expected_codes = [CODE_FOR[s] for s in true_states]

    lines = completion.strip().split('\n')
    step_lines = lines[:-1]
    answer_line = lines[-1]

    actual_codes = []
    for line in step_lines:
        token = line.rsplit('State:', 1)[-1].strip()
        actual_codes.append(token)

    intermediate_correct = actual_codes == expected_codes
    expected_final_code = CODE_FOR[true_states[-1]]
    final_answer_correct = answer_line == f'<answer> {expected_final_code}</answer>'
    no_literal_leak = 'heads' not in completion.lower() and 'tails' not in completion.lower()

    return {
        'starting_state': starting_state, 'operations': operations, 'completion': completion,
        'intermediate_correct': intermediate_correct, 'final_answer_correct': final_answer_correct,
        'no_literal_leak': no_literal_leak,
        'all_verified': intermediate_correct and final_answer_correct and no_literal_leak,
    }


def _row(scenario: tuple[str, tuple[str, ...]]) -> dict[str, Any]:
    starting_state, operations = scenario
    final_answer_state = _trace(starting_state, list(operations))[-1]
    return {
        'starting_state': starting_state,
        'operations': list(operations),
        'final_answer_state': final_answer_state,
        'final_answer_code': CODE_FOR[final_answer_state],
    }


# Original Stage 9/9c hand-picked 16-scenario bank keys (same instruction PATTERNS,
# reused for pattern diversity -- all-same/all-different/alternating/clustered/mixed,
# both starting states independently), redefined here rather than imported from
# synthetic_bridge.CODED_TRAJECTORIES since that bank is keyed to Nib/Nomo completions,
# not this bank's Bek/Ner ones -- only the (starting_state, operations) KEYS are reused.
_SEQUENCES_WITH_ORIGINAL_START = (
    ('Heads', ('same', 'same', 'same', 'same', 'same')),
    ('Tails', ('different', 'different', 'different', 'different', 'different')),
    ('Heads', ('same', 'different', 'same', 'different', 'same')),
    ('Tails', ('different', 'same', 'different', 'same', 'different')),
    ('Heads', ('different', 'different', 'same', 'same', 'different')),
    ('Tails', ('same', 'different', 'different', 'same', 'same')),
    ('Heads', ('different', 'same', 'same', 'different', 'different')),
    ('Tails', ('same', 'same', 'different', 'different', 'same')),
)
BANK_KEYS = _SEQUENCES_WITH_ORIGINAL_START + tuple(
    ('Tails' if s == 'Heads' else 'Heads', ops) for s, ops in _SEQUENCES_WITH_ORIGINAL_START
)


def build_code_word_train_eval_split(*, seed: int, n_eval: int = 21) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified by final_answer_state (Heads/Tails) -- reusing the original,
    non-stratified seed/split gives a skewed bank (measured directly: train 25
    Tails/18 Heads, eval 14 Heads/7 Tails under CLEAN21_SEED), the same class of
    problem already found and fixed for the Same/Different bank. Guarantees
    near-exact balance by construction instead."""
    eligible_eval = [sc for sc in ALL_SCENARIOS if sc not in BANK_KEYS]
    heads_eligible = [sc for sc in eligible_eval if _trace(sc[0], list(sc[1]))[-1] == 'Heads']
    tails_eligible = [sc for sc in eligible_eval if _trace(sc[0], list(sc[1]))[-1] == 'Tails']

    n_heads_eval = n_eval // 2
    n_tails_eval = n_eval - n_heads_eval
    if n_heads_eval > len(heads_eligible) or n_tails_eval > len(tails_eligible):
        raise ValueError(f'n_eval={n_eval} exceeds available balanced pool '
                          f'(heads_eligible={len(heads_eligible)}, tails_eligible={len(tails_eligible)})')

    rng = random.Random(seed)
    heads_shuffled = list(heads_eligible); rng.shuffle(heads_shuffled)
    tails_shuffled = list(tails_eligible); rng.shuffle(tails_shuffled)
    eval_scenarios = heads_shuffled[:n_heads_eval] + tails_shuffled[:n_tails_eval]
    eval_keys = set(eval_scenarios)

    train_scenarios = [sc for sc in ALL_SCENARIOS if sc not in eval_keys]
    train_rows = [_row(sc) for sc in train_scenarios]
    eval_rows = [_row(sc) for sc in eval_scenarios]
    assert not (eval_keys & set(BANK_KEYS))
    assert not ({(r['starting_state'], tuple(r['operations'])) for r in train_rows} & eval_keys)
    return train_rows, eval_rows
