"""Q1-Q3: run the ACTUAL repo functions on toy strings. No repo code modified."""
import re
from common import *
from reward_v3 import score_completion_v2, ALL_BANNED_PATTERNS, _extract_answer
from same_different_leakage_audit import check_completion_for_leakage

def train_pcot(text, step=150, total=150):
    b = score_completion_v2(text, 'Bek', step, total, num_flips=5)   # MAIN defaults (cot_max_scale=2.0)
    return b['banned_count'], round(b['p_cot'], 4)
def eval_flag(text):
    r = check_completion_for_leakage(text); return (not r['clean']), r['literal_hits'] + r['abbreviation_hits']

CASES = {
 'clean code-word completion':                   'Step 1: The state is tracked. State: Bek\n<answer> Bek</answer>',
 'lowercase "heads" BEFORE answer':              'Step 1: it is heads. State: Bek\n<answer> Bek</answer>',
 'capitalised "Heads" BEFORE answer':            'Step 1: it is Heads. State: Bek\n<answer> Bek</answer>',
 'lowercase "heads" AFTER answer tag':           'Step 1: State: Bek\n<answer> Bek</answer> heads',
 'capitalised "Heads" AFTER answer tag':         'Step 1: State: Bek\n<answer> Bek</answer> Heads',
 '"Heads" INSIDE the answer tag':                'Step 1: State: Bek\n<answer> Heads</answer>',
 'uppercase "HEADS" before answer':              'Step 1: it is HEADS. State: Bek\n<answer> Bek</answer>',
 'lowercase single letter "h" before answer':    'Step 1: h. State: Bek\n<answer> Bek</answer>',
 'capital single letter "T" before answer':      'Step 1: T. State: Bek\n<answer> Bek</answer>',
 'NO delimiter, "Heads" present':                'Step 1: it is Heads. State: Bek',
 'NO delimiter, lowercase "heads" present':      'Step 1: it is heads. State: Bek',
 'NO delimiter, clean':                          'Step 1: State: Bek',
 'uppercase tag <ANSWER> then "Heads" inside it':'Step 1: State: Bek\n<ANSWER> Heads</ANSWER>',
 'apostrophe "don\'t" (single-letter t trap)':   "Step 1: don't worry. State: Bek\n<answer> Bek</answer>",
}
print('%-48s | %-22s | %s' % ('case', 'EVAL scanner (flag,hits)', 'TRAIN pCoT (banned_count, p_cot@step150)'))
for name, text in CASES.items():
    print('%-48s | %-22s | %s' % (name, eval_flag(text), train_pcot(text)))
print()
print('_extract_answer on no-delimiter text ->', _extract_answer('Step 1: State: Bek'), '(training: format invalid => r_task=-5.0)')
b = score_completion_v2('Step 1: it is Heads. State: Bek', 'Bek', 150, 150, num_flips=5)
print('score_completion_v2 no-delimiter r_task=%s p_cot=%s banned_count=%s' % (b['r_task'], round(b['p_cot'],4), b['banned_count']))
print('training patterns:', ALL_BANNED_PATTERNS)
