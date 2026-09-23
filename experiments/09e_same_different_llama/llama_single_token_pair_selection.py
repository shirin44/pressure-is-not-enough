"""Stage 9e, Llama-3-8B-Instruct, Part B, Decision 1 (2026-09-07): single-token
code-word pair selection. Single-token status is now a HARD REQUIREMENT (not a
preference, per the task's explicit decision) -- the original 30-candidate pool
(shared with Qwen) contained only ONE single-token candidate on Llama's tokenizer
("Loft"), and it was correctly rejected via mnemonic-adjacency to "lit"
(OTHER_PROJECT_DOMAIN_WORDS). This required an expanded, documented search.

Full methodology (documented for the exhaustive-search disclosure requirement):
1. Generated 2,845 short CVC/CVCC pseudo-word combinations from onset+vowel+coda
   patterns, restricted to first letters NOT already covered by BANNED_WORDS or
   OTHER_PROJECT_DOMAIN_WORDS' own first letters (b,g,j,k,m,n,p,qu,w,x,y,z,em,en,
   ib,ub,ak,ek,ok,uk) -- 375 turned out to be single-token-with-leading-space under
   Llama's tokenizer (a much higher hit rate for real-word-shaped short strings than
   the original hand-invented 30-word pool, consistent with BPE vocabularies being
   built from real-corpus subword frequency).
2. Filtered against a real English dictionary (macOS /usr/share/dict/words, 235,976
   entries) -- excludes the MANY real short English words this systematic generation
   surfaced (bad, get, win, won, yes, yep, zero, pot, war, web, ...) that the existing
   mechanical leakage_check does NOT catch (it only checks mnemonic-adjacency/edit-
   distance against the SPECIFIC Coin Flip banned-word list, not real-word-ness in
   general).
3. Filtered against an explicit answer-adjacent denylist (yes/no/win/won/lose/lost/
   zero/one) -- outsized risk of entangling with the reward signal even if not
   literally in the task's own banned-word list.
4. Ran the existing leakage_check() (mnemonic-adjacency, edit-distance>=3, prompt-
   vocabulary, alpha-only) unchanged.
5. 115 candidates survived steps 2-4 mechanically -- but manual review found this
   still included real proper nouns/place names with sensitive associations (Gaza,
   Zika, Nero, Naz), foreign-language function words not in an English dictionary
   (Que/Qui/Por/Pero = that/who/for/but in Spanish/French/Portuguese), brand/prefix
   associations (Uber, Nano, Moto), a dictionary-check gap (Mom, Box -- ordinary
   English words the specific web2 wordlist didn't happen to flag), a code-adjacent
   collision (Kod, too close to "code" for a code-word task), and several common
   Western first names (Gina, Jana, Jen, Kara) -- excluded via documented manual
   judgment, since no automated check in this project's existing pipeline covers
   these categories. This is disclosed explicitly as a real limitation of the
   existing leakage_check(): it is necessary but not sufficient for candidates drawn
   from a systematic generator (as opposed to the original hand-invented pool, where
   the human generation process itself avoided real words by construction).
6. Final shortlist: 33 candidates, run through this script for base-policy
   log-probability matching -- the same criterion used to pair Nib/Nomo and
   Yelt/Yark on Qwen (within-pair logprob match under the untouched base model).
"""
from __future__ import annotations

import json
import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
for _name in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU visible to this process.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '09_direct_indomain_synthetic_bridge' / 'data_generation'))
from token_pool_audit import tokenizer_audit, leakage_check, logprob_at_state_slot  # noqa: E402

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'

SHORTLIST = [
    'Bak', 'Bek', 'Bez', 'Bik', 'Bir', 'Buk', 'Ekon', 'Emer', 'Geg', 'Jeg', 'Jub',
    'Kab', 'Kag', 'Kap', 'Kas', 'Kem', 'Kes', 'Kub', 'Kur', 'Mez', 'Mur', 'Nem',
    'Ner', 'Nir', 'Nur', 'Peb', 'Weg', 'Yaz', 'Zem', 'Zub', 'Zuk', 'Zur', 'Zus',
]

print('===== LOAD TOKENIZER =====')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('===== LOAD MODEL (8-bit) =====')
quant = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
model.eval()
device = next(model.parameters()).device
print({'model_loaded': True, 'device': str(device)})

print('===== AUDIT SHORTLIST: tokenization + leakage + base-policy logprob =====')
rows = []
for tok in SHORTLIST:
    row = {'token': tok}
    row.update(tokenizer_audit(tokenizer, tok))
    row.update(leakage_check(tok))
    row['base_policy_logprob_at_state_slot'] = logprob_at_state_slot(model, tokenizer, tok, device)
    row['is_single_token_with_leading_space'] = row['n_tokens_with_leading_space'] == 1
    row['eligible'] = row['is_single_token_with_leading_space'] and row['passes_leakage_classifier']
    rows.append(row)

print(f"\n{'token':8s} {'single_tok':11s} {'leak_ok':8s} {'min_ed':7s} {'logprob':10s} {'eligible':9s}")
for r in rows:
    lp = r['base_policy_logprob_at_state_slot']
    lp_str = f'{lp:.3f}' if lp is not None else 'n/a'
    print(f"{r['token']:8s} {str(r['is_single_token_with_leading_space']):11s} {str(r['passes_leakage_classifier']):8s} "
          f"{r['min_edit_distance_to_banned_word']:<7d} {lp_str:10s} {str(r['eligible']):9s}")

eligible = [r for r in rows if r['eligible']]
print(f'\n{len(eligible)}/{len(rows)} shortlist candidates eligible (single-token AND passes leakage).')

print('===== FIND BEST-MATCHED PAIRS (smallest |logprob difference| among eligible candidates) =====')
pairs = []
for i in range(len(eligible)):
    for j in range(i + 1, len(eligible)):
        a, b = eligible[i], eligible[j]
        if a['base_policy_logprob_at_state_slot'] is None or b['base_policy_logprob_at_state_slot'] is None:
            continue
        diff = abs(a['base_policy_logprob_at_state_slot'] - b['base_policy_logprob_at_state_slot'])
        pairs.append((diff, a['token'], b['token']))
pairs.sort(key=lambda x: x[0])
print('Top 10 best-matched pairs by |logprob difference| (ascending):')
for diff, a, b in pairs[:10]:
    print(f'  {a:6s} / {b:6s}  diff={diff:.4f}')

print('===== FINAL REPORT =====')
ROOT = Path.home() / 'aisi_checkpoints'
for version_id in range(1, 1000):
    OUTPUT = ROOT / f'stage9e-llama-single-token-pair-selection-v{version_id}'
    if not OUTPUT.exists():
        break
else:
    raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
report = {
    'config': {'model': MODEL_NAME, 'shortlist': SHORTLIST},
    'audit_rows': rows,
    'n_eligible': len(eligible),
    'best_matched_pairs_top10': [{'token_a': a, 'token_b': b, 'logprob_diff': diff} for diff, a, b in pairs[:10]],
}
EVENT_LOG = OUTPUT / 'stage9e_llama_single_token_pair_selection.json'
EVENT_LOG.write_text(json.dumps(report, indent=2, default=str))
print('\nEvidence:', EVENT_LOG)
print('STAGE 9E LLAMA SINGLE-TOKEN PAIR SELECTION COMPLETE.')
