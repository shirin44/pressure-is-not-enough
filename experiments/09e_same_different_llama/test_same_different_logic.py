"""CPU-only verification of the Same/Different parity logic (Decision 4), run BEFORE
any trajectory is built, per the task's explicit instruction to verify programmatically
first. Cross-checks two independent computations (real state trace vs. pure parity
counting) against every scenario in the existing, already-verified Stage 9 scenario
space, and pins down the origin-anchoring choice with a worked example."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '07_positive_signal_annealed_reward'))
from synthetic_bridge import _SCENARIOS, _trace  # noqa: E402

from same_different_logic import same_different_answer, same_different_answer_via_parity


def test_trace_and_parity_agree_on_every_existing_scenario():
    assert len(_SCENARIOS) == 16
    for starting_state, operations in _SCENARIOS:
        via_trace = same_different_answer(starting_state, operations)
        via_parity = same_different_answer_via_parity(operations)
        assert via_trace == via_parity, (starting_state, operations)


def test_even_number_of_different_gives_same():
    assert same_different_answer_via_parity(['same', 'same', 'same']) == 'Same'
    assert same_different_answer_via_parity(['different', 'different']) == 'Same'
    assert same_different_answer_via_parity(['different', 'same', 'different', 'same']) == 'Same'
    assert same_different_answer_via_parity([]) == 'Same'


def test_odd_number_of_different_gives_different():
    assert same_different_answer_via_parity(['different']) == 'Different'
    assert same_different_answer_via_parity(['different', 'different', 'different']) == 'Different'
    assert same_different_answer_via_parity(['same', 'different', 'same']) == 'Different'


def test_origin_is_declared_starting_state_not_state_after_instruction_one():
    # Worked example pinning the Decision 4 choice down concretely: starting_state
    # is anchor, NOT states[0] (the state after instruction 1's own effect).
    starting_state = 'Heads'
    operations = ['different', 'same', 'same']  # 1 "different" total -> odd -> Different
    states = _trace(starting_state, operations)
    assert states[0] == 'Tails'  # instruction 1 already flipped it
    assert states[-1] == 'Tails'
    answer = same_different_answer(starting_state, operations)
    assert answer == 'Different'  # relative to Heads (starting_state), not to states[0]=Tails
    # if origin were states[0] instead, this scenario would trivially always be
    # "Same" for every remaining same/same suffix -- confirms the two anchors are
    # NOT interchangeable and the choice matters


def test_neither_answer_string_overlaps_any_banned_word():
    # 'Same'/'Different' are the answer vocabulary now, not banned words to avoid --
    # sanity check they don't collide with anything in the existing literal-token set.
    from reward_v3 import normalize_state_token
    assert normalize_state_token('Same') not in {'heads', 'tails'}
    assert normalize_state_token('Different') not in {'heads', 'tails'}


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
