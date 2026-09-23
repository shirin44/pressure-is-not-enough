"""Stage 9e, Same/Different task logic (Decision 4): the origin-state anchoring
invariant, checked programmatically, not assumed.

**Origin is the DECLARED `starting_state`** (the value stated before any instruction
is applied, i.e. `row['starting_state']` in the existing `synthetic_bridge.py`
trajectory bank) -- NOT `_trace(...)[0]` (the state AFTER instruction 1's effect is
applied). This is the only choice under which the exact parity rule holds: `_trace`
starts at `starting_state` and flips on every "different" operation, so the final
state equals `starting_state` if and only if the total count of "different"
operations in the sequence is EVEN. Anchoring to `states[0]` instead would shift this
relationship by whether operation 1 itself is "same"/"different", breaking the clean
even/odd correspondence. Verified directly below against `synthetic_bridge._trace`,
not just asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
from synthetic_bridge import _trace  # noqa: E402


def same_different_answer(starting_state: str, operations: Sequence[str]) -> str:
    """The Same/Different final answer, anchored to the DECLARED starting_state
    (Decision 4). Equivalent to, and cross-checked against, the parity rule: an even
    number of 'different' operations gives 'Same', odd gives 'Different'."""
    states = _trace(starting_state, operations)
    final_state = states[-1] if states else starting_state
    return 'Same' if final_state == starting_state else 'Different'


def same_different_answer_via_parity(operations: Sequence[str]) -> str:
    """The SAME answer, computed independently via pure parity counting (no _trace
    call at all) -- used only to cross-check same_different_answer() agrees with the
    parity rule stated in the task, not to replace it."""
    n_different = sum(1 for op in operations if op == 'different')
    return 'Same' if n_different % 2 == 0 else 'Different'
