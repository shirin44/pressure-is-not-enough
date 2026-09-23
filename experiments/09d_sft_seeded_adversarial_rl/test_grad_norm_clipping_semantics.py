"""CPU-only regression test / permanent record for a finding that changed the plan for
this task (design.md, gradient-clipping addendum, 2026-08-30): HuggingFace's
Trainer._clip_grad_norm() / accelerate.Accelerator.clip_grad_norm_() -- which is what
GRPOTrainer inherits unmodified (confirmed: GRPOTrainer never overrides grad_norm
computation or logging; its own log() only merges in reward/KL metrics and calls
super().log() unchanged) -- ALWAYS returns the PRE-CLIP gradient norm, regardless of
max_grad_norm's value. This is standard, documented torch.nn.utils.clip_grad_norm_
behavior (it rescales the .grad tensors in place, but its return value is always the
norm BEFORE that rescaling), not a quirk of this project's code.

Consequence: Stage 9d's own grad_norm breaker (Cb.on_log, GRAD_BREAKER=50.0) reads
logs['grad_norm'], which transformers.Trainer._maybe_log_save_evaluate() populates
directly from that same pre-clip value (transformers/trainer.py line ~2087:
logs["grad_norm"] = grad_norm.item() ..., where grad_norm traces straight back to
_clip_grad_norm's return). Setting GRPOConfig's max_grad_norm to ANY value -- 1.0
(transformers' own default, already active in every run in this stage since it was
never explicitly overridden), 10, 15, or 50 -- cannot change what gets logged as
grad_norm, and therefore cannot change whether the breaker fires. Ordinary
optimizer-step gradient-norm clipping, as implemented by this trainer stack, is
mathematically the wrong tool for this specific symptom (a breaker keyed on the
pre-clip norm) -- not a matter of picking the right clip value, but a structural
mismatch between what the tool controls and what the breaker measures.

This test proves the invariance claim directly against real torch.nn.utils
functions (not a mock), and is meant to stand as the permanent citation for the
"gradient clipping cannot fix this breaker" conclusion -- so nobody re-attempts the
same experiment later without first re-deriving why it can't work."""
import torch


def test_clip_grad_norm_return_value_is_invariant_to_max_norm():
    # The core claim: for a FIXED gradient, clip_grad_norm_'s return value (what
    # ends up in the trainer's logs['grad_norm']) does not depend on max_norm at all.
    fixed_grad = torch.tensor([10.0, 0.0, 0.0, 0.0])  # norm = 10.0
    returned_norms = []
    for max_norm in (0.1, 1.0, 10.0, 15.0, 50.0, float('inf')):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = fixed_grad.clone()
        returned = torch.nn.utils.clip_grad_norm_([p], max_norm)
        returned_norms.append(returned.item())
    assert all(abs(v - 10.0) < 1e-6 for v in returned_norms), (
        f'expected the returned (pre-clip) norm to be 10.0 regardless of max_norm; got {returned_norms}')


def test_clip_grad_norm_does_rescale_the_actual_gradient_tensors():
    # Contrast case: max_norm DOES change the ACTUAL .grad values used by the
    # optimizer step -- just not the value returned/logged. Confirms the test isn't
    # trivially true because clipping does nothing at all.
    for max_norm, expected_post_clip_norm in [(1.0, 1.0), (10.0, 10.0), (50.0, 10.0), (float('inf'), 10.0)]:
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([10.0, 0.0, 0.0, 0.0])
        torch.nn.utils.clip_grad_norm_([p], max_norm)
        assert abs(p.grad.norm().item() - expected_post_clip_norm) < 1e-6


def test_grpo_trainer_does_not_override_grad_norm_computation_or_logging():
    # Static source-level guard: confirms the installed TRL's GRPOTrainer has no
    # grad_norm-related override that would invalidate the reasoning above. If a future
    # TRL upgrade adds one, this test should fail and force re-verification before
    # trusting the "clipping can't fix this breaker" conclusion again.
    import trl.trainer.grpo_trainer as grpo_mod
    import inspect
    source = inspect.getsource(grpo_mod.GRPOTrainer)
    assert 'grad_norm' not in source, (
        'GRPOTrainer now references grad_norm somewhere -- the installed TRL version '
        'has changed since this finding was derived; re-verify the pre-clip/post-clip '
        'logging chain before trusting the earlier conclusion')


def test_default_max_grad_norm_is_already_active_and_tighter_than_the_breaker():
    from trl import GRPOConfig
    # use_cpu=True: this module is CPU-only by design (see its own docstring) -- on
    # some transformers/accelerate versions and runners, constructing GRPOConfig with
    # its bf16 default runs a hardware capability check that raises without a GPU,
    # even though nothing about that check affects max_grad_norm, the only thing this
    # test verifies. Confirmed locally: max_grad_norm==1.0 identically with or without
    # use_cpu=True.
    default_args = GRPOConfig(output_dir='/tmp/_unused_grad_norm_test', use_cpu=True)
    assert default_args.max_grad_norm == 1.0, (
        f'expected the inherited transformers default of 1.0; got {default_args.max_grad_norm} -- '
        'if this changed, the "clipping already active at 1.0" finding needs re-checking')
    GRAD_BREAKER = 50.0
    assert default_args.max_grad_norm < GRAD_BREAKER, (
        'the ALREADY-ACTIVE default clip value is already far tighter than the breaker threshold')


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
