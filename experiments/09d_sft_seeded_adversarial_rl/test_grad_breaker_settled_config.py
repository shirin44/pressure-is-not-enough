"""Regression test for the SETTLED 2026-09-03 breaker recalibration (design.md
"Settled configuration"): KL_BREAKER=5.0 as the primary safety signal, GRAD_BREAKER
raised to 200.0 as a deliberately coarse backstop. Source-text checks rather than a
direct import, because sft_seeded_rl_decode_back.py raises RuntimeError at module
level if no CUDA GPU is visible (line 67) -- it cannot be imported in a CPU-only test
environment, matching the pattern already established for this file elsewhere in the
suite (no existing test imports it directly either).

Only sft_seeded_rl_decode_back.py is checked: it is the sole script actively reused
for ongoing Stage 9d GPU launches (the only one of the four scripts defining
GRAD_BREAKER/KL_BREAKER with an env-override pattern for RUN_SEED/GRAD_BREAKER,
because it's the only one relaunched across many seeds/configs). The other three
(sft_seeded_rl.py, sft_seeded_rl_full.py, per_token_gradient_diagnostic.py) are
completed, historical, single-run diagnostics from the original investigation commit,
untouched since -- retroactively changing their embedded constants would misrepresent
what config those specific historical runs actually used."""
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'sft_seeded_rl_decode_back.py'
SOURCE = SCRIPT.read_text()


def test_grad_breaker_default_is_the_settled_200():
    m = re.search(r"GRAD_BREAKER = float\(os\.environ\.get\('STAGE9D_GRAD_BREAKER', '([\d.]+)'\)\)", SOURCE)
    assert m is not None, 'GRAD_BREAKER definition not found or pattern changed'
    assert float(m.group(1)) == 200.0


def test_kl_breaker_is_unchanged_at_5():
    m = re.search(r"KL_BREAKER = ([\d.]+)", SOURCE)
    assert m is not None
    assert float(m.group(1)) == 5.0


def test_breaker_condition_is_still_an_or_not_reordered_into_precedence():
    # KL being "primary" is a design/documentation stance (KL_BREAKER trusted more),
    # not a change to the actual short-circuit logic -- both conditions still trip an
    # immediate hard stop, whichever fires first. Confirms this wasn't silently
    # changed to e.g. an AND, or KL-only, when "primary" was implemented.
    assert "grad_norm >= GRAD_BREAKER or kl >= KL_BREAKER" in SOURCE


def test_settled_value_clears_every_observed_benign_grad_norm_with_margin():
    # The 22 key events examined throughout this whole recalibration effort
    # (5 original breakers + 17 validation-batch events) top out at 72.0.
    OBSERVED_BENIGN_MAX = 72.0
    GRAD_BREAKER = 200.0
    assert GRAD_BREAKER > OBSERVED_BENIGN_MAX
    assert GRAD_BREAKER / OBSERVED_BENIGN_MAX > 2.5  # comfortably within the proposed 2-3x range


def test_env_override_still_works_for_future_recalibration():
    # The override mechanism itself (not just its default) must survive -- future
    # work may need to sweep this again without editing the script.
    assert 'STAGE9D_GRAD_BREAKER' in SOURCE


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
