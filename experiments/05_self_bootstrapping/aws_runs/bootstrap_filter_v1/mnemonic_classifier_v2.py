"""Mnemonic-adjacency classifier for invented codes -- v2.

v1 (original) only checked each invented code against its OWN state's name
(e.g. does 'R' resemble 'Running'?). This missed a second leakage channel:
codes that resemble the DOMAIN name itself (e.g. 'FZ'/'Fx' for fan -- F is
fan's own first letter -- or 'VAL'/'Va'/'Vi' for valve), which don't
identify *which* state but still leak *what task this is*, same underlying
concern as the state-name check, just one level up.

Found by manual inspection of Dataset A's 46 candidate examples: 100% of
both fan and valve rows had at least one code starting with the domain's
own first letter, despite passing the v1 (state-only) arbitrary filter.
Re-running this v2 rule against the full 8,160-completion pool found the
effect is total, not partial: 0/575 verified rows are genuinely clean
under both checks (state AND domain) -- see logs/development_log.md and
experiments/05_self_bootstrapping/README.md for the full investigation.
"""
DOMAIN_NAME = {"fan": "fan", "valve": "valve", "lamp": "lamp"}


def is_mnemonic_adjacent(code: str, real_word: str) -> bool:
    """True if `code` shares its first letter with `real_word`, or one is a
    prefix of the other (recognizable abbreviation in either direction)."""
    code_cf, word_cf = code.casefold(), real_word.casefold()
    if not code_cf or not word_cf:
        return False
    if code_cf[0] == word_cf[0]:
        return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf):
        return True
    return False


def is_leaking(code: str, state_word: str, domain: str) -> bool:
    """True if `code` is mnemonic-adjacent to its own state name (original check)
    OR to the domain name itself (v2 addition, same rule applied one level up)."""
    return is_mnemonic_adjacent(code, state_word) or is_mnemonic_adjacent(code, DOMAIN_NAME[domain])
