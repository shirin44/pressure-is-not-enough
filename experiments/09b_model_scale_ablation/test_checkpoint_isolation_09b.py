"""Regression test for the base-model-isolation fix (design.md, 2026-08-30 addendum):
injection_dryrun_7b.py's build_fresh_trainer(), calibration_seed_stability_check.py's
run_one_calibration_trial(), and dmax_sensitivity_check.py's run_one_dmax_trial() must
never share a base-model object across calls, since TRL's GRPOTrainer (beta!=0, PEFT)
registers a frozen "ref" adapter directly onto whatever base model it wraps
(trl/trainer/grpo_trainer.py, confirmed by reading the installed source) -- residue that
persists across get_peft_model() calls sharing the same base model object. This is the
exact pattern that was caught (via an ['default', 'ref'] adapter-composition assertion)
in Stage 9d's own scripts and, on inspection, found to also be present -- though never
previously checked -- in these three Stage 9b scripts (design.md's 2026-08-30 addendum).

This can't be verified end-to-end without a GPU (it requires an actual PEFT+TRL model),
so this is a static, source-level regression guard: confirms the CODE STRUCTURE never
reintroduces a shared base-model object across trial/phase loads, which is what the
empirical in-script assertions (`assert list(model.peft_config) == ['default']`, present
in all three scripts now) then verify at runtime, every run."""
from pathlib import Path

DRYRUN_SCRIPT = (Path(__file__).resolve().parent / 'injection_dryrun_7b.py').read_text()
CALIBRATION_SCRIPT = (Path(__file__).resolve().parent / 'calibration_seed_stability_check.py').read_text()
DMAX_SCRIPT = (Path(__file__).resolve().parent / 'dmax_sensitivity_check.py').read_text()


def _function_body(script_text, def_marker):
    fn_start = script_text.index(def_marker)
    fn_end = script_text.find('\ndef ', fn_start + 1)
    if fn_end == -1:
        fn_end = len(script_text)
    return script_text[fn_start:fn_end]


def _no_module_level_shared_base_model(script_text, script_name):
    for line in script_text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith('base_model = AutoModelForCausalLM.from_pretrained('), (
            f'{script_name}: found a module-level shared base_model load -- this is the '
            'exact pattern that leaked a stray "ref" adapter across phases/trials; base '
            'model loading must happen fresh inside the per-call function only')


def test_injection_dryrun_7b_has_no_module_level_shared_base_model():
    _no_module_level_shared_base_model(DRYRUN_SCRIPT, 'injection_dryrun_7b.py')


def test_injection_dryrun_7b_loads_a_fresh_base_model_inside_build_fresh_trainer():
    fn_body = _function_body(DRYRUN_SCRIPT, 'def build_fresh_trainer(')
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert 'fresh_base' in fn_body


def test_injection_dryrun_7b_asserts_adapter_composition_after_load():
    fn_body = _function_body(DRYRUN_SCRIPT, 'def build_fresh_trainer(')
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_calibration_seed_stability_check_has_no_module_level_shared_base_model():
    _no_module_level_shared_base_model(CALIBRATION_SCRIPT, 'calibration_seed_stability_check.py')


def test_calibration_seed_stability_check_loads_a_fresh_base_model_inside_the_trial_function():
    fn_body = _function_body(CALIBRATION_SCRIPT, 'def run_one_calibration_trial(')
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert 'fresh_base' in fn_body
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_dmax_sensitivity_check_has_no_module_level_shared_base_model():
    _no_module_level_shared_base_model(DMAX_SCRIPT, 'dmax_sensitivity_check.py')


def test_dmax_sensitivity_check_loads_a_fresh_base_model_inside_the_trial_function():
    fn_body = _function_body(DMAX_SCRIPT, 'def run_one_dmax_trial(')
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert 'fresh_base' in fn_body
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_injection_full_7b_still_single_call_not_incorrectly_flagged():
    # injection_full_7b.py (the actual reported 50-step run) was found NOT vulnerable --
    # a single get_peft_model call per process, no reuse -- so it was deliberately left
    # untouched by this fix. This test guards against someone "fixing" it unnecessarily
    # in a way that would mask a future real regression (e.g. a second call being added
    # without updating this test), by confirming build_fresh_trainer is still called
    # exactly once in the module body.
    script = (Path(__file__).resolve().parent / 'injection_full_7b.py').read_text()
    assert script.count('build_fresh_trainer(') == 2  # 1 def + 1 call site


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
