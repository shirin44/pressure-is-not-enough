"""Stage 11, Part B design note (read before semantic_monitor.py):

Direct inspection of the raw traces (see load_traces.py's manual-check output, and
design.md's "Two judge-facing variants" section) found that `build_code_word_completion`
narrates the SAME/DIFFERENT operation in PLAIN ENGLISH at every step ("The code remains
unchanged" / "The code changes") -- independent of which code word is used. A semantic
judge does not need to decode Bek/Ner AT ALL to recover the same/different pattern from
the FULL trace; the narration already states it directly. To separate "does the code
substitution conceal anything" from "does the model's own narrative commentary leak the
pattern regardless of the code", this module deterministically strips the narrative
commentary, leaving ONLY the bare "Step N: State: <code>" / "<answer> <code></answer>"
structure -- run BOTH variants through the monitor and report them separately, never
conflated into one number.

This is a REGEX TRANSFORM ONLY -- no model call, fully deterministic and auditable,
applied identically to every trace regardless of group."""
from __future__ import annotations

import re

_STEP_LINE = re.compile(r'^Step (\d+): .*?State: (\S+)$')


def strip_narration(trace_text: str) -> str:
    """Reduces every 'Step N: <narration>. State: <code>' line to 'Step N: State:
    <code>', and leaves the <answer>...</answer> line untouched (it carries no
    narration to strip). Raises if a line doesn't match the expected format -- silent
    pass-through of an unexpected line would risk leaving narration in unnoticed."""
    out_lines = []
    for line in trace_text.splitlines():
        if line.startswith('<answer>'):
            out_lines.append(line)
            continue
        m = _STEP_LINE.match(line)
        if not m:
            raise ValueError(f'line does not match expected Step/State format: {line!r}')
        step, code = m.groups()
        out_lines.append(f'Step {step}: State: {code}')
    return '\n'.join(out_lines)
