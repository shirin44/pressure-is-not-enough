"""Unit tests for the extended mnemonic-adjacency classifier (state-name check + new domain-name check)."""

DOMAIN_NAME = {"fan": "fan", "valve": "valve"}

def is_mnemonic_adjacent(code, real_word):
    code_cf, word_cf = code.casefold(), real_word.casefold()
    if not code_cf or not word_cf:
        return False
    if code_cf[0] == word_cf[0]:
        return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf):
        return True
    return False

def is_leaking(code, state_word, domain):
    """A code 'leaks' if it's mnemonic-adjacent to its own state name (existing check)
    OR mnemonic-adjacent to the domain name itself (new check, same rule applied one level up)."""
    return is_mnemonic_adjacent(code, state_word) or is_mnemonic_adjacent(code, DOMAIN_NAME[domain])

# ===== Regression: must still catch the original state-name leakage cases =====
assert is_leaking("R", "Running", "fan") is True   # first-letter match to state
assert is_leaking("Run", "Running", "fan") is True  # prefix of state
assert is_leaking("O", "Open", "valve") is True

# ===== New: must catch the domain-leakage cases actually found in Dataset A =====
assert is_leaking("VAL", "Open", "valve") is True, "VAL (prefix of 'valve') must now be rejected"
assert is_leaking("Va", "Open", "valve") is True, "Va (prefix of 'valve') must now be rejected"
assert is_leaking("Vi", "Closed", "valve") is True, "Vi (starts with valve's first letter) must now be rejected"
assert is_leaking("FZ", "Stopped", "fan") is True, "FZ (starts with fan's first letter) must now be rejected"
assert is_leaking("Fx", "Running", "fan") is True, "Fx (starts with fan's first letter) must now be rejected"
assert is_leaking("F", "Running", "fan") is True

# ===== ALT was the OTHER code in the VAL/ALT pair -- does it independently leak? =====
# ALT vs domain 'valve': first letter 'a' != 'v', not a prefix either way -> should NOT leak via domain.
# ALT vs state 'Closed': first letter 'a' != 'c', not a prefix -> should NOT leak via state either.
assert is_leaking("ALT", "Closed", "valve") is False, "ALT alone doesn't leak -- the PAIR was rejected because VAL did"

# ===== Must NOT over-reject: F/V appearing mid-word or as an unrelated code should pass =====
assert is_leaking("TAF", "Stopped", "fan") is False, "F not at the start -- must not be flagged"
assert is_leaking("XRV", "Open", "valve") is False, "V not at the start, and X doesnt match Open or valve -- must not be flagged"
assert is_leaking("ZR", "Running", "fan") is False, "genuinely arbitrary code, no relation to fan or Running"
assert is_leaking("L", "Closed", "valve") is False, "genuinely arbitrary code, no relation to valve or Closed"
assert is_leaking("H", "Closed", "valve") is False
assert is_leaking("Zorp", "Stopped", "fan") is False, "classic arbitrary invented word"
assert is_leaking("Blib", "Open", "valve") is False, "classic arbitrary invented word"

print("ALL UNIT TESTS PASSED")
print("  - regression (state-name leakage still caught): OK")
print("  - new domain-name leakage cases (VAL/ALT, Va/Vi, FZ, Fx) correctly rejected: OK")
print("  - no over-rejection (mid-word F/V, ALT alone, genuinely arbitrary codes) confirmed: OK")
