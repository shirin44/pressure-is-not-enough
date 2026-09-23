"""Regression test for the base-model-isolation fix (design.md progress log,
2026-08-29): sft_seeded_rl.py's load_stage9c_checkpoint() and entropy_diagnostic.py's
load_checkpoint_fresh() must never share a base-model object across calls, since TRL's
GRPOTrainer (beta!=0, PEFT) registers a frozen "ref" adapter directly onto whatever
base model it wraps (trl/trainer/grpo_trainer.py, confirmed by reading the installed
source) -- residue that persists across get_peft_model() calls sharing the same base
model object. This can't be verified end-to-end without a GPU (it requires an actual
PEFT+TRL model), so this is a static, source-level regression guard: confirms the
CODE STRUCTURE never reintroduces a shared base-model object across phase loads,
which is what the empirical in-script assertions (`assert list(model.peft_config) ==
['default']`, present in both scripts) then verify at runtime, every run."""
from pathlib import Path

SFT_SCRIPT = (Path(__file__).resolve().parent / 'sft_seeded_rl.py').read_text()
ENTROPY_SCRIPT = (Path(__file__).resolve().parent / 'entropy_diagnostic.py').read_text()
DIVERSITY_SCRIPT = (Path(__file__).resolve().parent / 'rollout_diversity_diagnostic.py').read_text()


def test_sft_seeded_rl_loads_a_fresh_base_model_inside_the_checkpoint_loader():
    # The base-model load call must live INSIDE load_stage9c_checkpoint (so it runs
    # fresh on every call), not at module scope shared across calls.
    fn_start = SFT_SCRIPT.index('def load_stage9c_checkpoint()')
    fn_end = SFT_SCRIPT.index('\ndef ', fn_start + 1)
    fn_body = SFT_SCRIPT[fn_start:fn_end]
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert 'fresh_base' in fn_body


def test_sft_seeded_rl_has_no_module_level_shared_base_model():
    # No bare "base_model = AutoModelForCausalLM.from_pretrained(" assignment outside
    # the loader function -- that's exactly the pattern that caused the original leak.
    lines = SFT_SCRIPT.splitlines()
    for line in lines:
        stripped = line.strip()
        assert not stripped.startswith('base_model = AutoModelForCausalLM.from_pretrained('), (
            'found a module-level shared base_model load -- this is the exact pattern '
            'that leaked a stray "ref" adapter across phases; base model loading must '
            'happen fresh inside load_stage9c_checkpoint() only')


def test_sft_seeded_rl_asserts_adapter_composition_after_load():
    fn_start = SFT_SCRIPT.index('def load_stage9c_checkpoint()')
    fn_end = SFT_SCRIPT.index('\ndef ', fn_start + 1)
    fn_body = SFT_SCRIPT[fn_start:fn_end]
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_entropy_diagnostic_loads_a_fresh_base_model_inside_the_checkpoint_loader():
    fn_start = ENTROPY_SCRIPT.index('def load_checkpoint_fresh(')
    fn_end = ENTROPY_SCRIPT.index('\ndef ', fn_start + 1)
    fn_body = ENTROPY_SCRIPT[fn_start:fn_end]
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_entropy_diagnostic_calls_the_loader_once_per_checkpoint_not_shared():
    # The comparison loop must call load_checkpoint_fresh() once per checkpoint label,
    # inside the loop over CHECKPOINTS -- not once outside and reused.
    loop_start = ENTROPY_SCRIPT.index('for label, weights_path in CHECKPOINTS.items():')
    loop_region = ENTROPY_SCRIPT[loop_start:loop_start + 400]
    assert 'load_checkpoint_fresh(weights_path)' in loop_region


def test_diversity_diagnostic_loads_a_fresh_base_model_inside_the_checkpoint_loader():
    fn_start = DIVERSITY_SCRIPT.index('def load_checkpoint_fresh(')
    fn_end = DIVERSITY_SCRIPT.index('\ndef ', fn_start + 1)
    fn_body = DIVERSITY_SCRIPT[fn_start:fn_end]
    assert 'AutoModelForCausalLM.from_pretrained(' in fn_body
    assert "assert list(model.peft_config) == ['default']" in fn_body


def test_diversity_diagnostic_calls_the_loader_once_per_checkpoint_not_shared():
    loop_start = DIVERSITY_SCRIPT.index('for label, weights_path in CHECKPOINTS.items():')
    loop_region = DIVERSITY_SCRIPT[loop_start:loop_start + 400]
    assert 'load_checkpoint_fresh(weights_path)' in loop_region


def test_gradient_diagnostic_asserts_adapter_composition_after_load():
    script = (Path(__file__).resolve().parent / 'per_token_gradient_diagnostic.py').read_text()
    assert "assert list(model.peft_config) == ['default']" in script


def test_gradient_diagnostic_retains_grad_only_inside_compute_loss():
    script = (Path(__file__).resolve().parent / 'per_token_gradient_diagnostic.py').read_text()
    fn_start = script.index('def _patched_get_per_token_logps_and_entropies(')
    fn_end = script.index('\ndef ', fn_start + 1)
    fn_body = script[fn_start:fn_end]
    assert 'logps.retain_grad()' in fn_body
    assert "if INSTRUMENTATION['inside_compute_loss']:" in fn_body


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
