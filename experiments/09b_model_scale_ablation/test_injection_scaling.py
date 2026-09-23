import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'experiments' / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(_REPO_ROOT / 'experiments' / '07_positive_signal_annealed_reward'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step14b_injection import inject_into_generation_output, select_injection_indices
from injection_scaling import inject_into_generation_output_n, select_injection_indices_n
from synthetic_bridge import CODED_TRAJECTORIES


class _FakeTokenizer:
    eos_token = '<eos>'

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [len(w) for w in text.split()]


BANK_ROW = CODED_TRAJECTORIES[0]


def _build_cot_prompt(starting_state, operations):
    # Same convention step14_reward_gate.scenario_key_from_prompt parses (build_cot_prompt
    # in the Step 14b scripts): "Starting state: X" then numbered "same as previous"/
    # "different from previous" instruction lines.
    lines = [f'Starting state: {starting_state}', 'Instructions:']
    clarified = {'same': 'same as previous (the state does NOT change)',
                 'different': 'different from previous (the state flips)'}
    for i, op in enumerate(operations, start=1):
        lines.append(f'{i}. {clarified[op]}')
    return '\n'.join(lines)


def _uniform_prompts(n=8):
    prompt = _build_cot_prompt(BANK_ROW['starting_state'], BANK_ROW['operations'])
    return [prompt] * n


def test_n1_matches_step14b_selection_exactly_on_uniform_group():
    prompts = _uniform_prompts(8)
    original = select_injection_indices(prompts)
    scaled = select_injection_indices_n(prompts, 1)
    assert original == scaled, (original, scaled)


def test_n1_matches_step14b_injection_output_exactly():
    prompts = _uniform_prompts(8)
    completion_ids = [[1, 2, 3]] * 8
    completions = ['placeholder'] * 8
    tok = _FakeTokenizer()
    orig_ids, orig_completions, orig_injected = inject_into_generation_output(
        prompts, completion_ids, completions, tok)
    new_ids, new_completions, new_injected = inject_into_generation_output_n(
        prompts, completion_ids, completions, tok, 1)
    assert orig_ids == new_ids
    assert orig_completions == new_completions
    assert set(orig_injected) == set(new_injected)


def test_n2_injects_exactly_two_indices_same_trajectory():
    prompts = _uniform_prompts(8)
    injected = select_injection_indices_n(prompts, 2)
    assert len(injected) == 2
    assert set(injected) == {0, 1}
    assert injected[0] is injected[1]  # same trajectory row object, both slots


def test_n3_injects_exactly_three_indices():
    prompts = _uniform_prompts(8)
    injected = select_injection_indices_n(prompts, 3)
    assert len(injected) == 3
    assert set(injected) == {0, 1, 2}


def test_n_larger_than_group_size_caps_at_available_bank_indices():
    prompts = _uniform_prompts(8)
    injected = select_injection_indices_n(prompts, 100)
    assert len(injected) == 8  # only 8 rollouts in the group, all bank scenario


def test_non_bank_group_injects_nothing_regardless_of_n():
    prompts = ['Starting state: Heads\n1. not a real bank scenario prompt'] * 8
    for n in (1, 2, 3):
        assert select_injection_indices_n(prompts, n) == {}


def test_n2_injection_output_replaces_correct_slots_with_verified_completion():
    prompts = _uniform_prompts(8)
    completion_ids = [[9, 9, 9]] * 8
    completions = ['UNTOUCHED'] * 8
    tok = _FakeTokenizer()
    new_ids, new_completions, injected = inject_into_generation_output_n(
        prompts, completion_ids, completions, tok, 2)
    assert new_completions[0] == BANK_ROW['completion']
    assert new_completions[1] == BANK_ROW['completion']
    assert new_completions[2] == 'UNTOUCHED'
    assert new_completions[3:] == ['UNTOUCHED'] * 5
    assert completions == ['UNTOUCHED'] * 8  # input not mutated
    assert completion_ids == [[9, 9, 9]] * 8  # input not mutated


def test_zero_n_injects_nothing():
    prompts = _uniform_prompts(8)
    assert select_injection_indices_n(prompts, 0) == {}


def test_negative_n_raises():
    prompts = _uniform_prompts(8)
    try:
        select_injection_indices_n(prompts, -1)
        assert False, 'expected ValueError'
    except ValueError:
        pass


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
