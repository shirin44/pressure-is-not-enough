"""Q11: summarise every causal-flip evidence file (SFT-only + post-RL seeds), per direction and step, from the persisted JSONs (no recomputation of the test itself)."""
import json
from pathlib import Path
from common import AWS
runs = [('SFT-only (Bek/Ner)', 'stage9e-llama-causal-flip-v1')] + [(f'MAIN seed{s}', f'stage9e-llama-causal-flip-post-rl-seed{s}-v1') for s in (42, 43, 44, 45)]
print('%-20s %-8s %-22s %-22s %-14s %s' % ('checkpoint', 'n', 'final tracks flip', 'trace==counterfactual', 'ignores flip', 'per step2 / step3 (tracks)'))
for name, d in runs:
    p = AWS / d / 'stage9e_llama_causal_flip.json'
    if not p.exists(): print('%-20s MISSING %s' % (name, d)); continue
    ev = json.loads(p.read_text()); rs = ev['intervention_results']; a = ev['analysis']
    n = len(rs); tr = sum(r['final_answer_classification'] == 'tracks_flip_genuine_causal_use' for r in rs)
    cf = sum(bool(r['propagation_matches_counterfactual']) for r in rs); ig = sum(bool(r.get('propagation_matches_original_ignoring_flip')) for r in rs)
    noop = sum(not r['flip_changes_the_predicted_answer'] for r in rs)
    cls = {}
    for r in rs: cls[r['final_answer_classification']] = cls.get(r['final_answer_classification'], 0) + 1
    per = ' / '.join('%d/%d' % (round(a[s]['all']['final_answer_tracks_flip_rate'] * a[s]['all']['n']), a[s]['all']['n']) for s in ('step_2', 'step_3'))
    dirs = {k: sum(r['final_answer_classification'] == 'tracks_flip_genuine_causal_use' for r in rs if (r['original_token'], r['flipped_token']) == k) for k in (('bek', 'ner'), ('ner', 'bek'))}
    dn = {k: sum((r['original_token'], r['flipped_token']) == k for r in rs) for k in dirs}
    print('%-20s %-8d %-22s %-22s %-14s %s | bek->ner %d/%d ner->bek %d/%d | no-op flips %d | classes %s | sanity %s/%s | adapter %s' % (
        name, n, f'{tr}/{n}', f'{cf}/{n}', f'{ig}/{n}', per, dirs[('bek','ner')], dn[('bek','ner')], dirs[('ner','bek')], dn[('ner','bek')], noop, cls,
        ev['config']['sanity_matches'], ev['config']['sanity_n'], ev['config']['adapter_sha256'][:8]))
