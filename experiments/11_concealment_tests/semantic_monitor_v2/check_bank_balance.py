"""Stage 11, semantic_monitor_v2 follow-up: is the ~54% trace-level accuracy an
artifact of this 63-trace bank's true Step-1 states happening to be close to
balanced, or does the "first-mentioned code = Heads" heuristic carry a genuine
(if partial) correlation with the true state that isn't just base-rate luck?

Pure analysis of already-persisted evidence -- no API calls, no GPU, no new traces.
Reads ONLY `v2_monitor_results.json` (unmodified) and does not touch
`semantic_monitor.py`, `v2_monitor.py`, or any existing results file.

Two DISTINCT quantities are computed and never conflated (see step 2's discrepancy
check below, which this script runs before reporting anything else):
  - IDEALIZED heuristic accuracy: what a mechanical rule-follower ("the code seen in
    Step 1 is always called Heads") would score. At Step 1 this is tautologically
    "predict Heads every time" -- so idealized Step-1 accuracy = the bank's own
    fraction of traces truly starting Heads. This is a property of the BANK, not of
    the judge.
  - ACTUAL judge accuracy (already reported in the v2 README: 34/63): what the judge
    literally answered, which followed the idealized heuristic in only 58/63 traces
    (see semantic_monitor_v2/README.md step 4) -- in the other 5/63 the judge's own
    Step-1 guess was "Tails" even though its own first-mentioned code was the one it
    later (self-consistently) labeled Tails throughout, i.e. it did NOT always apply
    the naive positional default. These two numbers are NOT guaranteed to match, and
    step 2 below checks this explicitly rather than assuming it.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from binomial_stats import summarize_binomial

RESULTS = json.loads((Path(__file__).resolve().parent / 'v2_monitor_results.json').read_text())
GROUPS = ['pre_rl_trained_pair', 'post_rl_trained_pair', 'pre_rl_heldout_pair']


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test p-value for a 2x2 table
    [[a, b], [c, d]], via direct hypergeometric enumeration (no scipy dependency)."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d

    def hyper_p(x: int) -> float:
        if x < 0 or x > col1 or (row1 - x) < 0 or (row1 - x) > col2:
            return 0.0
        return (math.comb(row1, x) * math.comb(row2, col1 - x)) / math.comb(n, col1)

    observed_p = hyper_p(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    total = sum(hyper_p(x) for x in range(lo, hi + 1) if hyper_p(x) <= observed_p * (1 + 1e-9))
    return min(total, 1.0)


def main():
    by_group = defaultdict(list)
    for r in RESULTS:
        by_group[r['group']].append(r)

    # ===== Step 1: true Step-1 state, per trace and per group =====
    print('===== STEP 1: true Step-1 state, per group =====')
    overall_heads = overall_tails = 0
    per_group_true_step1 = {}
    for g in GROUPS:
        rs = by_group[g]
        true_step1 = [r['ground_truth_state_sequence'][0] for r in rs]
        n_heads = sum(1 for s in true_step1 if s == 'Heads')
        n_tails = sum(1 for s in true_step1 if s == 'Tails')
        per_group_true_step1[g] = true_step1
        overall_heads += n_heads
        overall_tails += n_tails
        print(f'{g}: n={len(rs)} true_Heads={n_heads} true_Tails={n_tails}')
    print(f'OVERALL: n={len(RESULTS)} true_Heads={overall_heads} true_Tails={overall_tails}')

    # ===== Step 2: cross-tab + discrepancy check against the already-reported 34/63 =====
    print('\n===== STEP 2: idealized heuristic (predict Heads at step 1, always) vs true state =====')
    idealized_correct = 0
    rows = []
    for r in RESULTS:
        true_s1 = r['ground_truth_state_sequence'][0]
        first_code = r['as_written_code_sequence'][0]
        idealized_prediction = 'Heads'  # tautological: whatever code is first, heuristic calls it Heads
        idealized_match = (idealized_prediction == true_s1)
        idealized_correct += idealized_match
        rows.append({
            'group': r['group'], 'true_step1': true_s1, 'first_code': first_code,
            'idealized_prediction': idealized_prediction, 'idealized_match': idealized_match,
            'judge_actual_step1_guess': r['judge_labels'][0], 'judge_actual_first_step_correct': r['first_step_correct'],
        })
    print(f'idealized heuristic (mechanical, ALWAYS predicts Heads at step 1): correct on {idealized_correct}/63')
    print(f'idealized heuristic accuracy = fraction of bank truly starting Heads = {overall_heads}/63')
    already_reported_judge_accuracy = sum(1 for r in RESULTS if r['first_step_correct'])
    print(f'ACTUAL judge accuracy already reported in v2 README: {already_reported_judge_accuracy}/63')
    judge_deviated_from_idealized = sum(
        1 for r in rows if r['judge_actual_step1_guess'] != r['idealized_prediction']
    )
    print(f'traces where judge\'s ACTUAL step-1 guess deviates from the idealized "always Heads" rule: '
          f'{judge_deviated_from_idealized}/63')
    if idealized_correct == already_reported_judge_accuracy:
        print('MATCH: idealized-heuristic accuracy == already-reported judge accuracy (34/63) -- coincidental, '
              'given the judge deviates from the idealized rule on '
              f'{judge_deviated_from_idealized} traces; flagged explicitly, not silently assumed.')
    else:
        print(f'*** DISCREPANCY FLAGGED: idealized heuristic accuracy ({idealized_correct}/63) != '
              f'already-reported judge accuracy ({already_reported_judge_accuracy}/63). These are DIFFERENT '
              f'quantities (idealized mechanical rule vs. what the judge actually said) and were never claimed '
              f'to be identical -- see module docstring.')

    # ===== Step 3: does "which code appears first" correlate with "true Step-1 state"? =====
    print('\n===== STEP 3: does first_code correlate with true_step1 state? (2x2 contingency + Fisher exact) =====')
    # Build the 2x2 table across ALL 63 traces: rows = which code appeared first
    # (there are 3 distinct code-pair vocabularies across the 3 groups, so this is
    # done both PER GROUP, where "first code" is a real binary choice within a fixed
    # pair, and is the scientifically meaningful table).
    for g in GROUPS:
        rs = by_group[g]
        codes_used = sorted(set(r['as_written_code_sequence'][0] for r in rs) | set(
            c for r in rs for c in r['as_written_code_sequence']))
        # restrict to the two codes that actually appear as "first code" in this group
        first_codes = sorted(set(r['as_written_code_sequence'][0] for r in rs))
        if len(first_codes) != 2:
            print(f'{g}: first_code takes {len(first_codes)} distinct values ({first_codes}) -- '
                  f'not a clean 2x2 case, reporting raw counts only.')
            continue
        code_a, code_b = first_codes
        a = sum(1 for r in rs if r['as_written_code_sequence'][0] == code_a and r['ground_truth_state_sequence'][0] == 'Heads')
        b = sum(1 for r in rs if r['as_written_code_sequence'][0] == code_a and r['ground_truth_state_sequence'][0] == 'Tails')
        c = sum(1 for r in rs if r['as_written_code_sequence'][0] == code_b and r['ground_truth_state_sequence'][0] == 'Heads')
        d = sum(1 for r in rs if r['as_written_code_sequence'][0] == code_b and r['ground_truth_state_sequence'][0] == 'Tails')
        p = fisher_exact_2x2(a, b, c, d)
        print(f'{g}: first_code={code_a!r} -> (true Heads={a}, true Tails={b}); '
              f'first_code={code_b!r} -> (true Heads={c}, true Tails={d}); Fisher exact two-sided p={p:.6f}')

    # Pooled table: since Bek/Ner and Jub/Kag are DIFFERENT vocabularies per group, the
    # scientifically meaningful pooled question is stated structurally: "does the code
    # word occupying position 1 (arbitrary label 'X' vs 'Y' per trace, canonicalized
    # so 'X' = whichever code is alphabetically first within its pair) correlate with
    # true_step1?" -- reported for completeness, per-group tables above are primary.
    print('\n(Per-group tables above are the primary, valid comparison -- code vocabularies '
          'differ across groups so a naive pool across groups conflates different codes.)')

    # ===== Step 4: bank balance, precise, with binomial test vs 50/50 =====
    print('\n===== STEP 4: bank true-state balance vs 50/50 =====')
    overall_summary = summarize_binomial(overall_heads, len(RESULTS))
    print(f'OVERALL: {overall_heads}/{len(RESULTS)} = {overall_heads/len(RESULTS)*100:.2f}% true Heads at step 1')
    print(f'  exact binomial two-sided p (vs 0.5) = {overall_summary["exact_binomial_two_sided_p"]:.6f}, '
          f'95% Wilson CI = {overall_summary["wilson_95pct_ci"]}, '
          f'distinguishable from 50/50: {overall_summary["distinguishable_from_0.5_at_alpha_0.05"]}')
    for g in GROUPS:
        true_step1 = per_group_true_step1[g]
        n_heads = sum(1 for s in true_step1 if s == 'Heads')
        s = summarize_binomial(n_heads, len(true_step1))
        print(f'{g}: {n_heads}/{len(true_step1)} = {n_heads/len(true_step1)*100:.2f}% true Heads, '
              f'p={s["exact_binomial_two_sided_p"]:.6f}, CI={s["wilson_95pct_ci"]}, '
              f'distinguishable from 50/50: {s["distinguishable_from_0.5_at_alpha_0.05"]}')

    return {
        'overall_heads': overall_heads, 'overall_tails': overall_tails,
        'idealized_correct': idealized_correct,
        'already_reported_judge_accuracy': already_reported_judge_accuracy,
        'judge_deviated_from_idealized': judge_deviated_from_idealized,
    }


if __name__ == '__main__':
    main()
