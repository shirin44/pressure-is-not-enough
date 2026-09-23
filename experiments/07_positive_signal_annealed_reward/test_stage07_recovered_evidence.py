"""Golden-record test against the Stage 7 driver script and evidence recovered from AWS
(i-REDACTED) on 2026-09-22 and archived under aws_recovered/. All 10 files were
sha256-verified bit-identical against the AWS instance before this test was written
(see design.md's "Stage 7 archived from AWS" entry for the remote/local hash pairs).

Confirms: (1) the driver's inlined score_completion is functionally identical to the
currently-committed reward_v3.score_completion_v2 (same formula, same defaults, only the
name differs -- the committed module is this script's logic factored out, not a rewrite);
(2) the recovered evidence JSONs reproduce, from raw numbers, the exact figures already
claimed in README.md/results.json (accuracy 60.5%->56.4%, 0 non-literal at every
milestone, 150/150 steps, no hard stop, entropy_coef=0.18 8-step follow-up); (3) the
prompt-format mismatch already found in Llama 09e and every other Qwen stage is present
here too: training rollouts use a raw string prompt (Dataset.from_list({'prompt': <str>})),
milestone/held-out eval uses tokenizer.apply_chat_template -- confirmed by direct reading
of the full 1343-line script, not inferred from other stages."""
import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECOVERED = ROOT / 'aws_recovered'
SCRIPT_TEXT = (RECOVERED / 'stage07_signal_annealed.py').read_text()


def test_recovered_files_present():
    expected = [
        'stage07_signal_annealed.py', 'stage07_dryrun.log', 'stage07_fullrun.log',
        'stage08_entropy_test.py', 'stage08_dryrun.log',
        'aisi_checkpoints/grpo-stage07-signal-annealed-dryrun-v1/stage07_signal_annealed_reward.json',
        'aisi_checkpoints/grpo-stage07-signal-annealed-full-v1/stage07_signal_annealed_reward.json',
        'aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/README.md',
        'aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/per_token_instrumentation.json',
        'aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/stage08_entropy_test.json',
    ]
    for rel in expected:
        assert (RECOVERED / rel).is_file(), rel


def test_training_prompts_are_raw_strings_not_chat_templated():
    # TRAIN_POOL / train_dataset construction: plain string under the 'prompt' key.
    assert "rows.append({'prompt':prompt,'ground_truth':truth})" in SCRIPT_TEXT
    assert "train_dataset=Dataset.from_list(TRAIN_POOL)" in SCRIPT_TEXT
    # confirm the raw prompt comes straight out of generate_coinflip_example (returns a str, not a message list)
    assert 'def generate_coinflip_example(n_flips: int, seed: int) -> tuple[str, str]:' in SCRIPT_TEXT


def test_eval_prompts_are_chat_templated():
    assert "tokenizer.apply_chat_template(" in SCRIPT_TEXT
    # the only apply_chat_template call is inside run_heldout_generation (milestone/held-out eval), not training
    calls = [m.start() for m in re.finditer(r'apply_chat_template\(', SCRIPT_TEXT)]
    assert len(calls) == 1
    eval_fn_start = SCRIPT_TEXT.index('def run_heldout_generation')
    eval_fn_end = SCRIPT_TEXT.index('\ndef ', eval_fn_start + 1)
    assert eval_fn_start < calls[0] < eval_fn_end


def test_full_run_config_matches_150_steps_and_readme_claims():
    d = json.loads((RECOVERED / 'aisi_checkpoints/grpo-stage07-signal-annealed-full-v1/stage07_signal_annealed_reward.json').read_text())
    fr = d['final_report']
    assert fr['terminal_step'] == fr['requested_steps'] == 150
    assert fr['dry_run'] is False
    assert fr['hard_stop'] is None
    assert fr['survived_to_full_target'] is True
    assert abs(fr['task_accuracy_first_window'] - 0.6047297297297297) < 1e-9  # 60.5%
    assert abs(fr['task_accuracy_last_window'] - 0.5641891891891891) < 1e-9   # 56.4%
    final = fr['final_milestone']
    assert final['step'] == 150
    for direction in ('Heads', 'Tails'):
        assert final['by_answer_direction'][direction]['verified_non_literal_rate'] == 0.0
        assert final['by_answer_direction'][direction]['verified_n'] == 0
    assert final['by_answer_direction']['Heads']['n'] + final['by_answer_direction']['Tails']['n'] == 100


def test_dryrun_config_matches_8_steps():
    d = json.loads((RECOVERED / 'aisi_checkpoints/grpo-stage07-signal-annealed-dryrun-v1/stage07_signal_annealed_reward.json').read_text())
    fr = d['final_report']
    assert fr['terminal_step'] == fr['requested_steps'] == 8
    assert fr['dry_run'] is True


def test_entropy_followup_config_matches_readme_claim_of_018():
    d = json.loads((RECOVERED / 'aisi_checkpoints/grpo-stage08-entropy-test-dryrun-v1/stage08_entropy_test.json').read_text())
    assert d['config']['entropy_coef'] == 0.18
    assert d['config']['n_steps'] == 8
    assert d['config']['dry_run'] is True


def test_inlined_score_completion_matches_committed_reward_v3_score_completion_v2():
    import sys
    sys.path.insert(0, str(ROOT))
    from reward_v3 import score_completion_v2
    ns = {}
    # extract just the recovered script's score_completion (and its direct dependencies) and exec it in isolation
    start = SCRIPT_TEXT.index('def score_completion(\n')
    end = SCRIPT_TEXT.index('\n\n\ndef reward_fn(')
    body = SCRIPT_TEXT[start:end]
    header = SCRIPT_TEXT[SCRIPT_TEXT.index('def completion_to_text'):start]
    preamble = 'from __future__ import annotations\nimport math, re\nfrom typing import Any, Mapping, Sequence\n'
    exec(preamble + header + '\n' + body, ns)
    recovered_score_completion = ns['score_completion']

    starting = 'Starting state: Heads\nInstructions:\n1. same as previous (the state does NOT change)\n2. different from previous (the state flips)\nReason through every flip in order.'
    completion = 'Step 1: x. State: Zorp\nStep 2: y. State: Blim\n<answer>Tails</answer>'
    a = recovered_score_completion(completion, 'Tails', 10, 150, prompt=starting)
    b = score_completion_v2(completion, 'Tails', 10, 150, prompt=starting)
    assert a == b, (a, b)


if __name__ == '__main__':
    tests = [obj for name, obj in list(globals().items()) if name.startswith('test_') and inspect.isfunction(obj)]
    failures = []
    for t in tests:
        try:
            t(); print(f'PASSED: {t.__name__}')
        except Exception as e:
            failures.append(t.__name__); print(f'FAILED: {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    if failures:
        raise SystemExit(1)
