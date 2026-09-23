"""Regression tests for the 2026-08-29 fixed-D_MAX decision (design.md): per-run calibration
(Phase 0) is removed from injection_dryrun_7b.py's automatic launch path, replaced by a
hardcoded D_MAX=0.505 borrowed from the 3B model's own calibration. kl_calibration.py's own
functions are still tested by test_kl_calibration.py (the module remains valid and in the
repo for reference/future use, per instruction) -- these tests instead confirm the LAUNCH
SCRIPT itself no longer runs a calibration procedure and correctly applies the fixed value.
injection_dryrun_7b.py is a top-to-bottom script that executes immediately on import (model
load, GPU-only), so these are static source-level checks, not an import-based test -- the
same constraint every other Stage 9b static test already works under
(test_injection_scaling.py, test_kl_calibration.py)."""
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent / 'injection_dryrun_7b.py'
SOURCE = SCRIPT_PATH.read_text()


def test_fixed_d_max_constant_is_present_and_correct():
    assert 'FIXED_D_MAX = 0.505' in SOURCE
    assert 'D_MAX = FIXED_D_MAX' in SOURCE


def test_calibration_phase_no_longer_present():
    # These were the calibration procedure's own identifying calls/names in the pre-fix
    # script -- their absence confirms Phase 0 is not silently still running under a
    # different name.
    removed_markers = [
        'CALIBRATION_STEPS', 'CALIBRATION_SEED', 'calib_trainer', 'calib_model',
        'CLEAN_CALIB_STEPS', 'PHASE 0: KL-CLAMP + BREAKER CALIBRATION',
    ]
    for marker in removed_markers:
        assert marker not in SOURCE, f'found calibration-era marker still present: {marker!r}'


def test_calibration_module_not_imported_by_launch_script():
    # kl_calibration.py's functions are no longer imported here -- Phase 0 (which used
    # them) is gone. The module itself is untouched and still tested by
    # test_kl_calibration.py, per instruction ("code can remain in the repo for
    # reference/future use").
    assert 'from kl_calibration import' not in SOURCE


def test_d_max_source_is_disclosed_as_not_measured_on_7b():
    # The exact disclosure language required by the task -- this value was NOT measured on
    # the 7B model, unlike every other constant in this ablation.
    assert 'NEVER MEASURED on the 7B model' in SOURCE
    assert "3B model's OWN calibrated D_MAX" in SOURCE or "3B model\\'s OWN calibrated D_MAX" in SOURCE


def test_entropy_clamp_value_is_frozen_measured_constant():
    assert 'ENTROPY_CLAMP_VALUE = 1.2189' in SOURCE
    # Distinguished explicitly from D_MAX's cross-model borrowing.
    assert 'UNLIKE D_MAX, this IS a 7B-measured value' in SOURCE


def test_clamp_value_computed_from_fixed_d_max_not_a_pool():
    # CLAMP_VALUE must be derived directly from FIXED_D_MAX via the same exp(D)-D-1 formula,
    # not from compute_clamp_from_pool (which would imply calibration is still running).
    assert 'CLAMP_VALUE = math.exp(D_MAX) - D_MAX - 1' in SOURCE
    assert 'compute_clamp_from_pool(' not in SOURCE


def test_kl_clamp_self_test_still_present():
    # The self-test that verifies the clamp MECHANISM itself (pathological token bounded,
    # healthy tokens unaffected) must still run against the fixed D_MAX -- removing
    # calibration should not have removed this correctness check.
    assert 'KL CLAMP SELF-TEST' in SOURCE
    assert "assert math.isclose(kl1[2].item(), CLAMP_VALUE, rel_tol=1e-4)" in SOURCE


def test_breaker_sanity_references_real_telemetry_not_calibration_telemetry():
    assert 'satisfied by real Phase-1-equivalent telemetry' in SOURCE


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
