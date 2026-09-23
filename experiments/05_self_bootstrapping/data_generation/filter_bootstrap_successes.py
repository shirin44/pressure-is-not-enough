"""Classify, filter, and format the Option 1 bootstrap SFT dataset.

Pure CPU, no GPU/Colab needed -- run this locally after downloading the raw
completions JSON produced by notebooks/option1_bootstrap_generation.ipynb
(that notebook does the GPU-only work: reproduce the classifier gate, load
checkpoint 130, generate completions for all 8,160 undeclared prompts from
data/processed/bootstrap_undeclared_fan_valve_v1/prompts.json, save raw
completions to Drive). Classification, filtering, and SFT-dataset
construction need zero GPU, so they are deliberately kept out of the Colab
notebook to avoid spending GPU-hours on CPU-only work.

Reuses src.data.undeclared_generalization.score_undeclared_completion
unchanged -- the same function already verified (see
notebooks/option1_bootstrap_generation.ipynb's reproduction gate, and the
5-known-positive local check run during this session) to reproduce the
original Stage 3.5 10% result. No classification logic is rewritten here.

## Two filtering criteria, reported side by side

1. "verified" -- the EXACT criteria the original 10% figure was computed
   from (both_states_observed AND nonliteral_consistent AND
   decode_back_self_consistent; see
   notebooks/stage35_third_domain_zero_shot.ipynb cell 9). Reported for
   direct comparability against the known 10% baseline.
2. "sft_eligible" -- criteria (1) PLUS answer_correct=True. The original
   10% definition does not require the literal <answer> tag to match
   ground truth (decode_back_self_consistent already requires the
   *decode-back line's stated* physical state to match ground truth, but
   that is textually separate from the <answer> tag). For a research
   audit that gap doesn't matter; for SFT training targets it does --
   training on a self-consistent trace that ends in a wrong <answer> tag
   would actively teach wrong answers. This is a deliberate, minimal,
   documented addition on top of the existing classifier's fields, not a
   change to the classifier itself.

Usage:
    python3 scripts/filter_bootstrap_successes.py <raw_completions.json> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
sys.path.insert(0, str(ROOT))

from src.data.undeclared_generalization import UndeclaredExample, score_undeclared_completion  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "bootstrap_undeclared_fan_valve_v1"
KNOWN_BASELINE_RATE = 0.10  # lamp, n=100, notebooks/stage35_third_domain_zero_shot.ipynb


def _row_to_example(row: dict) -> UndeclaredExample:
    return UndeclaredExample(
        example_id=row["example_id"], domain=row["domain"], initial_state=row["initial_state"],
        operations=tuple(row["operations"]), expected_states=tuple(row["expected_states"]),
        final_answer=row["final_answer"], prompt=row["prompt"])


def classify_and_filter(raw_rows: list[dict]) -> dict:
    scored = []
    for row in raw_rows:
        example = _row_to_example(row)
        score = score_undeclared_completion(example, row["completion"])
        verified = bool(score["both_states_observed"] and score["nonliteral_consistent"]
                         and score["decode_back_self_consistent"])
        sft_eligible = verified and bool(score["answer_correct"])
        scored.append({**row, "score": score, "verified": verified, "sft_eligible": sft_eligible})

    n = len(scored)
    verified_count = sum(r["verified"] for r in scored)
    sft_eligible_count = sum(r["sft_eligible"] for r in scored)
    verified_rate = verified_count / n if n else 0.0
    sft_eligible_rate = sft_eligible_count / n if n else 0.0

    by_domain = {}
    for r in scored:
        d = by_domain.setdefault(r["domain"], {"n": 0, "verified": 0, "sft_eligible": 0})
        d["n"] += 1
        d["verified"] += r["verified"]
        d["sft_eligible"] += r["sft_eligible"]
    for d in by_domain.values():
        d["verified_rate"] = d["verified"] / d["n"] if d["n"] else 0.0
        d["sft_eligible_rate"] = d["sft_eligible"] / d["n"] if d["n"] else 0.0

    ratio_to_baseline = verified_rate / KNOWN_BASELINE_RATE if KNOWN_BASELINE_RATE else None
    same_ballpark = ratio_to_baseline is not None and 0.5 <= ratio_to_baseline <= 2.0

    report = {
        "n_attempts": n, "verified_count": verified_count, "verified_rate": verified_rate,
        "sft_eligible_count": sft_eligible_count, "sft_eligible_rate": sft_eligible_rate,
        "known_baseline_rate_lamp_n100": KNOWN_BASELINE_RATE,
        "ratio_to_baseline": ratio_to_baseline,
        "same_ballpark_as_baseline": same_ballpark,
        "same_ballpark_note": ("verified_rate is between 0.5x and 2x the lamp baseline" if same_ballpark else
            "verified_rate deviates by more than 2x from the lamp baseline in either direction -- worth "
            "reviewing before training on this data, not just proceeding."),
        "by_domain": by_domain,
    }
    return {"scored": scored, "report": report}


def build_sft_dataset(scored: list[dict]) -> list[dict]:
    """Each SFT example: the undeclared prompt (unchanged) and the model's own verbatim successful
    completion as the target. No text is rewritten or cleaned up beyond what the model produced --
    training on the model's actual successful trace, not an edited version of it."""
    eligible = [r for r in scored if r["sft_eligible"]]
    dataset = []
    for r in eligible:
        dataset.append({
            "example_id": r["example_id"], "domain": r["domain"], "split": "bootstrap_sft",
            "initial_state": r["initial_state"], "operations": r["operations"],
            "expected_states": r["expected_states"], "final_answer": r["final_answer"],
            "prompt": r["prompt"], "target": r["completion"],
            "self_invented_token_pair": r["score"].get("token_pair"),
        })
    return dataset


def reverify_sft_dataset(dataset: list[dict]) -> dict:
    """Final audit: re-run the same classifier against every included row one more time, from the
    stored (prompt, target) pair alone, exactly the same way the training-eligible filter was applied.
    This is the 100% semantic verification pass rate check every prior seeding dataset in this project
    requires -- trivial by construction here (every row already passed once), but re-derived rather than
    just trusted, to catch any accidental data-shuffling bug between filtering and dataset construction."""
    failures = []
    for row in dataset:
        example = UndeclaredExample(
            example_id=row["example_id"], domain=row["domain"], initial_state=row["initial_state"],
            operations=tuple(row["operations"]), expected_states=tuple(row["expected_states"]),
            final_answer=row["final_answer"], prompt=row["prompt"])
        score = score_undeclared_completion(example, row["target"])
        ok = (score["both_states_observed"] and score["nonliteral_consistent"]
              and score["decode_back_self_consistent"] and score["answer_correct"])
        if not ok:
            failures.append({"example_id": row["example_id"], "score": score})
    n = len(dataset)
    return {
        "count": n,
        "semantic_pass_rate": 100 * (n - len(failures)) / n if n else 0.0,
        "failures": failures,
        "accepted": n > 0 and not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_completions", type=Path, help="Path to the raw completions JSON from option1_bootstrap_generation.ipynb")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    raw = json.loads(args.raw_completions.read_text())
    raw_rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw

    result = classify_and_filter(raw_rows)
    print("===== CLASSIFICATION REPORT =====")
    print(json.dumps(result["report"], indent=2))

    dataset = build_sft_dataset(result["scored"])
    audit = reverify_sft_dataset(dataset)
    print("\n===== SFT DATASET RE-VERIFICATION (must be 100%) =====")
    print(json.dumps({k: v for k, v in audit.items() if k != "failures"}, indent=2))
    if not audit["accepted"]:
        raise RuntimeError(f"SFT dataset failed re-verification: {audit['failures'][:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "classification_report.json").write_text(json.dumps(result["report"], indent=2))
    (args.out_dir / "bootstrap_sft_dataset.json").write_text(json.dumps({
        "source": str(args.raw_completions), "audit": audit, "count": len(dataset), "rows": dataset,
    }, indent=2))
    print(f"\nWrote {len(dataset)} SFT examples to {args.out_dir / 'bootstrap_sft_dataset.json'}")


if __name__ == "__main__":
    main()
