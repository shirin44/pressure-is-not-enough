"""Stage 9e, Decision 3: leakage audit for the Same/Different bank -- literal words
and abbreviations ONLY, per Decision 3's explicit scope. Deliberately NOT a synonym/
semantic-leakage classifier (that mechanism is explicitly deferred to a separate,
later, more carefully validated experiment, per the task's own rationale: an
aggressive fuzzy penalty risks reproducing Stage 1's vacuous-collapse failure mode).
The rigid template's structural constraint is the leakage defense for this pass; this
module is the PERMANENT, AUTOMATED check that the bank actually respects it -- run
against every completion in the bank before it is used for anything, not a one-off
human read-through.
"""
from __future__ import annotations

import re
from typing import Any

# Literal state words and single-letter abbreviations -- the SAME banned vocabulary
# already used throughout this project (synthetic_bridge.LITERAL_TOKENS covers
# heads/tails; this adds the single-letter forms h/t explicitly, matching
# token_pool_audit.BANNED_WORDS' own broader list).
BANNED_LITERALS = {'heads', 'tails', 'head', 'tail'}
BANNED_ABBREVIATIONS = {'h', 't'}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def check_completion_for_leakage(completion: str) -> dict[str, Any]:
    """Scans a single completion for literal banned words / single-letter
    abbreviations anywhere in the text (not just inside <answer> tags -- a leak in
    the reasoning trace is still a leak)."""
    words = _word_tokens(completion)
    literal_hits = sorted(set(w for w in words if w in BANNED_LITERALS))
    # Abbreviation check: single-letter word tokens only (avoids false-positiving on
    # every word that happens to CONTAIN 'h' or 't') -- matches how a real
    # letter-only abbreviated answer ("H" or "T" standing in for Heads/Tails) would
    # actually appear as its own token.
    abbrev_hits = sorted(set(w for w in words if w in BANNED_ABBREVIATIONS))
    clean = not literal_hits and not abbrev_hits
    return {'literal_hits': literal_hits, 'abbreviation_hits': abbrev_hits, 'clean': clean}


def audit_bank_for_leakage(completions: list[str]) -> dict[str, Any]:
    """Run the check against every completion in a bank. Returns a summary plus the
    per-completion results for any that failed, so a failure is immediately
    actionable (which completion, which word) rather than just a pass/fail count."""
    results = [check_completion_for_leakage(c) for c in completions]
    n_clean = sum(1 for r in results if r['clean'])
    failures = [{'index': i, **r} for i, r in enumerate(results) if not r['clean']]
    return {
        'n_total': len(completions), 'n_clean': n_clean, 'n_leaked': len(completions) - n_clean,
        'all_clean': n_clean == len(completions), 'failures': failures,
    }
