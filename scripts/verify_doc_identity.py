"""Sanity-check that tracked docs actually contain their own content, not
another file's -- catches silent cross-file overwrites (e.g. a scp/cp
collision landing the wrong file's content at a given path) before they
get committed.

Added 2026-08-24 after two silent-overwrite incidents in one session: a
build script blindly overwriting an executed notebook with a blank
template (fixed by src/notebook_io.py's safe_write_notebook), and a scp
filename collision that silently replaced experiments/05_self_bootstrapping
/README.md's content with experiments/03_demonstration_seeding_multi_domain
/README.md's, undetected because commit-time validation only checked that
JSON files parse, never that Markdown files still say what they should.

Run before every commit that touches experiments/*/README.md or
results/*.json:

    python3 scripts/verify_doc_identity.py

Exits nonzero (and prints every mismatch) if anything fails.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").is_dir())


def check_experiment_readmes() -> list[str]:
    """Each experiments/NN_stage/README.md's first heading, IF it leads with a
    number (most do: '# NN — Title'), must match its own directory's NN
    prefix. A heading with no leading number (e.g. '# Infrastructure') is not
    flagged -- not every stage follows the numbered-title convention -- but a
    heading whose number belongs to a DIFFERENT stage is exactly the signal
    a cross-file collision would produce, and is flagged."""
    errors = []
    for readme in sorted((ROOT / "experiments").glob("*/README.md")):
        stage_dir = readme.parent.name
        dir_match = re.match(r"^(\d+)_", stage_dir)
        if not dir_match:
            continue  # e.g. "archive" -- no numeric prefix to check against
        expected = dir_match.group(1)
        text = readme.read_text()
        first_line = text.splitlines()[0] if text.strip() else ""
        heading_match = re.match(r"^#\s*(\d+)\s*[—-]", first_line)
        if heading_match and heading_match.group(1) != expected:
            errors.append(
                f"{readme.relative_to(ROOT)}: heading is numbered '{heading_match.group(1)}', "
                f"but this file lives under the '{expected}_' stage directory -- "
                f"looks like another stage's content (got: {first_line!r})"
            )
    return errors


def check_results_json() -> list[str]:
    """Each results/NN_stage.json's top-level "stage" field must match its
    own filename (every existing file already follows this convention)."""
    errors = []
    for result_file in sorted((ROOT / "results").glob("*.json")):
        try:
            data = json.loads(result_file.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{result_file.relative_to(ROOT)}: invalid JSON ({exc})")
            continue
        expected_stage = result_file.stem
        actual_stage = data.get("stage")
        if actual_stage != expected_stage:
            errors.append(
                f"{result_file.relative_to(ROOT)}: 'stage' field is {actual_stage!r}, "
                f"expected {expected_stage!r} -- filename/content mismatch"
            )
    return errors


def main() -> None:
    errors = check_experiment_readmes() + check_results_json()
    if errors:
        print("DOC IDENTITY CHECK FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"OK: doc identity check passed "
          f"({len(list((ROOT / 'experiments').glob('*/README.md')))} READMEs, "
          f"{len(list((ROOT / 'results').glob('*.json')))} results files).")


if __name__ == "__main__":
    main()
