import json, re, sys
from pathlib import Path

HOME = Path.home()
sys.path.insert(0, str(HOME / "AISI"))
from src.data.multidomain_seed import generate_multidomain_dataset
from src.data.undeclared_generalization import build_undeclared_lamp_evaluation

_, declared_eval = generate_multidomain_dataset(seed=20260812, corrected_fan_wording=True)
undeclared_lamp_eval = build_undeclared_lamp_evaluation(declared_eval["lamp"])

_LAMP_EXPECTED_STATES_BY_ID={row.example_id:row.expected_states for row in undeclared_lamp_eval}
_COMPOSITION_STEP_RE=re.compile(r'^Step\s+(\d+):.*?State:\s*([A-Z][A-Za-z]{0,14})[.,]?\s*$',re.MULTILINE)
def _is_mnemonic_adjacent(code,real_word):
    code_cf,word_cf=code.casefold(),real_word.casefold()
    if not code_cf or not word_cf: return False
    if code_cf[0]==word_cf[0]: return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf): return True
    return False
def dark_truth_composition(undeclared_rows):
    dark_verified=[r for r in undeclared_rows if r.get('final_answer')=='Dark' and r.get('both_states_observed')
                   and r.get('nonliteral_consistent') and r.get('decode_back_self_consistent')]
    if not dark_verified:
        return {'n':0,'clean':None,'leaking':None,'clean_fraction':None}
    clean=0; leaking=0; unparseable=0
    for row in dark_verified:
        expected_states=_LAMP_EXPECTED_STATES_BY_ID.get(row['example_id'])
        if expected_states is None:
            unparseable+=1; continue
        prefix=row['text'].split('<answer>',1)[0]
        matches=[(int(i),tok) for i,tok in _COMPOSITION_STEP_RE.findall(prefix)]
        tokens=[tok for _,tok in matches]
        if len(tokens)!=len(expected_states):
            unparseable+=1; continue
        mapping={}
        for state,token in zip(expected_states,tokens): mapping.setdefault(state,token)
        leaks=any(_is_mnemonic_adjacent(mapping.get(s),s) or _is_mnemonic_adjacent(mapping.get(s),'lamp')
                  for s in ('Lit','Dark') if mapping.get(s))
        if leaks: leaking+=1
        else: clean+=1
    n=clean+leaking
    return {'n':n,'unparseable':unparseable,'clean':clean,'leaking':leaking,
            'clean_fraction':(clean/n if n else None)}

progress = json.loads((HOME / "aisi_checkpoints" / "dataset_b_sft_dryrun_v1" / "step8_eval_progress.json").read_text())
result = dark_truth_composition(progress["undeclared"])
print("integrated logic result:", result)
print("expected (from standalone check): n=10, clean=10, leaking=0, clean_fraction=1.0")
assert result == {'n': 10, 'unparseable': 0, 'clean': 10, 'leaking': 0, 'clean_fraction': 1.0}, "MISMATCH"
print("MATCH CONFIRMED")
