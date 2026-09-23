"""Golden-record regression tests against the Stage 9d decode-back-seeded MAIN re-run
(aws_runs/stage9d-decode-back-main-v3/, 2026-09-03): launched solely to obtain a
saved checkpoint for a follow-up diagnostic (this script never saved one before),
using the identical seed/config/code as the original stage9d-decode-back-main-v2 run.
Captures the unplanned but important finding: v3's mean genuine_correct_among_
nonliteral (0.611) is significantly BELOW even BASELINE (0.698, t=-5.06) and far
below the original MAIN v2 result (0.906, t=-10.41), despite identical everything --
demonstrating MAIN's headline result is not reliably reproducible run-to-run. Also
captures the root cause, traced directly and completely: 100% of v3's wrong answers
are decode-back-line mistranslations of the model's own correctly-tracked state (the
same residual bias already characterized in the 09c decode-back SFT task), not a
reopening of the structural disconnect or a genuine tracking regression --
decode_back_matches_own_trace_rate stays perfect throughout."""
import json
import statistics
from pathlib import Path

BASELINE = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-baseline-v1'
                        / 'stage9d_decode_back_baseline.json').read_text())
MAIN_V2 = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v2'
                       / 'stage9d_decode_back_main.json').read_text())
MAIN_V3 = json.loads((Path(__file__).resolve().parent / 'aws_runs' / 'stage9d-decode-back-main-v3'
                       / 'stage9d_decode_back_main.json').read_text())


def _matched(*runs):
    ms_list = [{r['step']: r['genuine_correct_among_structural_nonliteral'] for r in d['result']['milestones']} for d in runs]
    steps = sorted(set.intersection(*[set(m) for m in ms_list]))
    return steps, [[m[s] for s in steps] for m in ms_list]


def test_ran_the_full_150_steps_with_no_breaker_despite_the_weak_outcome():
    result = MAIN_V3['result']
    assert result['terminal_step'] == 150
    assert result['hard_stop'] is None


def test_checkpoint_was_saved_this_time():
    assert MAIN_V3['result']['saved_checkpoint_dir'] is not None
    assert MAIN_V3['result']['saved_checkpoint_dir'].endswith('final_adapter')


def test_v3_is_significantly_below_baseline_not_just_below_v2():
    steps, (v3_vals, b_vals) = _matched(MAIN_V3, BASELINE)
    assert len(steps) == 30
    diffs = [v - b for v, b in zip(v3_vals, b_vals)]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    assert t_stat < -3.0
    assert statistics.mean(v3_vals) < statistics.mean(b_vals)


def test_v3_is_dramatically_below_the_original_v2_result():
    steps, (v3_vals, v2_vals) = _matched(MAIN_V3, MAIN_V2)
    diffs = [v3 - v2 for v3, v2 in zip(v3_vals, v2_vals)]
    n = len(diffs)
    t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (n ** 0.5))
    assert t_stat < -8.0
    assert statistics.mean(v2_vals) - statistics.mean(v3_vals) > 0.25


def test_decode_back_fidelity_remains_perfect_despite_the_weak_outcome():
    dbmr = [r['decode_back_matches_own_trace_rate'] for r in MAIN_V3['result']['milestones']]
    assert len(dbmr) == 30
    assert all(x == 1.0 for x in dbmr)


def test_every_wrong_answer_traces_to_decode_back_mistranslation_not_tracking_failure():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09c_sft_diagnostic'))
    from decode_back_bank import parse_decode_back_line

    total_wrong = 0
    mistranslation = 0
    for m in MAIN_V3['result']['milestones']:
        for s in m['samples']:
            if s['final_answer_correct']:
                continue
            total_wrong += 1
            parsed = parse_decode_back_line(s['completion'])
            assert parsed is not None, 'expected every wrong answer to still have a parseable decode-back line'
            decoded_code, _decoded_literal = parsed
            tokens = s['nonliteral_tokens']
            assert tokens and decoded_code.strip().lower() == tokens[-1], (
                'expected the decode-back line to match the model\'s own last tracked token '
                '(a translation error), not diverge from it (a tracking error)')
            mistranslation += 1
    assert total_wrong > 0
    assert mistranslation == total_wrong  # 100% of errors are mistranslations, zero are tracking failures


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
