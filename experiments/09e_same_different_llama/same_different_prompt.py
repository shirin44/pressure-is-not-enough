"""Stage 9e Same/Different task: prompt construction, shared by the zero-shot
capacity check (Decision 2, plain Heads/Tails, no code substitution) and, later,
the coded trajectory bank (Decision 3's rigid template, applied with code words).

Mirrors synthetic_bridge.py's build_prompt() structure exactly (same rigid-template
instructions, same 'Step N: ... State: <token>' line format) but with the FINAL
QUESTION replaced: instead of asking for the literal final state, asks whether the
final tracked state is the SAME as or DIFFERENT from the initial (declared
starting_state) -- Decision 4's origin anchoring, stated explicitly in the prompt
itself so there's no ambiguity for the model or the grader about which state
"initial" refers to.
"""
from __future__ import annotations

from typing import Sequence


def build_same_different_prompt(starting_state: str, operations: Sequence[str], state_token_hint: str | None = None) -> str:
    """state_token_hint: None for plain Heads/Tails (Decision 2's zero-shot check);
    a code word description is NOT built here -- the coded bank (Decision 3 onward)
    reuses this same wording but the completion-building step substitutes the code
    words in, exactly as synthetic_bridge.py's build_coded_completion() does for the
    original task."""
    lines = [f"Starting state: {starting_state}", "Instructions:"]
    wording = {
        "same": "same as previous (the state does NOT change)",
        "different": "different from previous (the state flips)",
    }
    for i, operation in enumerate(operations, 1):
        lines.append(f"{i}. {wording[operation]}")
    lines.append(
        "Reason through every flip in order. Put Step and State on the SAME line for "
        "every instruction. Follow this complete example line exactly: "
        "'Step 1: The state remains unchanged. State: Heads'. Replace the number, "
        "reasoning, and state token as appropriate, but never put State on a new line. "
        "The state token must be one capitalized alphabetic word. Do not use answer "
        "tags for intermediate states. "
        f"After all steps, answer this question: is the final tracked state the SAME "
        f"as the DECLARED STARTING STATE above (\"Starting state: {starting_state}\"), "
        "or DIFFERENT from it? Give exactly one word, 'Same' or 'Different', inside "
        "<answer>...</answer> tags."
    )
    return "\n".join(lines)
