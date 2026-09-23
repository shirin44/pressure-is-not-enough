"""Q4: what do the malformed raw-prompt completions contain, and state-slot well-formedness per arm (re-parsing the persisted completion text with the repo's own parsers)."""
import json, re, collections
from common import AWS
from reward_v3 import parse_state_slots, structure_penalty, _extract_answer
ev = json.loads((AWS / 'stage9e-step0-rollout-regen-v1' / 'step0_rollout_regen.json').read_text())
for arm, d in ev['arms'].items():
    rows = d['rows']; n = len(rows)
    slot_ok = sum(structure_penalty(r['completion'], 5) == 0.0 for r in rows)
    both = sum(structure_penalty(r['completion'], 5) == 0.0 and r['format_valid'] for r in rows)
    print(f'{arm}: state slots exactly Step1..5 in order: {slot_ok}/{n}; slots ok AND parseable final answer: {both}/{n}')
rows = ev['arms']['train_format']['rows']; bad = [r for r in rows if not r['format_valid']]
c = collections.Counter()
for r in bad:
    t = r['completion']
    c['total malformed'] += 1
    c['starts with a bare code word (" Bek"/" Ner") before anything else'] += bool(re.match(r'\s*(Bek|Ner)\b', t))
    c['contains a literal "<code>" tag'] += '<code>' in t
    c['contains a complete <answer>...</answer> somewhere (not at end of text)'] += bool(re.search(r'<answer>.*?</answer>', t, re.S))
    c['contains "</answer>" (any)'] += '</answer>' in t
    c['contains a "State:" trace line'] += 'State:' in t
    c['has >=5 parseable state slots'] += len(parse_state_slots(t)) >= 5
    c['ends by hitting 256-token cap'] += r['hit_cap']
    c['ends with EOS'] += r['ended_eos']
    c['contains a run of >=8 repeats of one code word (" Bek Bek ..." loop)'] += bool(re.search(r'\b(Bek|Ner)(?:\s+\1){7,}', t))
    c['no "<answer>" tag at all'] += '<answer>' not in t
for k, v in c.items(): print(f'  raw-arm malformed [{k}]: {v}/{c["total malformed"]}')
def show(r, label):
    print(f'--- {label}: scenario start={r["starting_state"]} ops={r["operations"]} n_tokens={r["n_tokens"]} r_task={r["r_task"]}')
    print(repr(r['completion'][:420]))
show(bad[0], 'raw malformed #1 (looping)')
noans = [r for r in bad if '<answer>' not in r['completion']]
if noans: show(noans[0], 'raw malformed, no <answer> tag')
eosbad = [r for r in bad if r['ended_eos']]
if eosbad: show(eosbad[0], 'raw malformed but ended with EOS')
