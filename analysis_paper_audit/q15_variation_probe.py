from common import *
from reward_v3 import state_variation_penalty, parse_state_slots, structure_penalty
from code_word_answer_bank import build_code_word_prompt, build_code_word_completion
ops = ['different','same','different','same','same']; prompt = build_code_word_prompt('Heads', ops); good = build_code_word_completion('Heads', ops)
L = good.split('\n'); ans = L[-1]; steps = L[:-1]
cases = {
 'correct trace': good,
 'missing slot (step 3 deleted)': '\n'.join(steps[:2] + steps[3:] + [ans]),
 'duplicate slot (step 2 twice)': '\n'.join(steps[:2] + [steps[1]] + steps[2:] + [ans]),
 'out of order (steps 2,3 swapped)': '\n'.join([steps[0], steps[2], steps[1]] + steps[3:] + [ans]),
 'extra slot (6 lines)': '\n'.join(steps + ['Step 6: extra. State: Bek', ans]),
 'wrong token change (step 3 same instead of change)': good.replace(steps[2], steps[2].rsplit('State:',1)[0] + 'State: ' + steps[1].rsplit('State: ',1)[1]),
 'slot after <answer> only': '\n'.join(steps[:4] + [ans, steps[4]]),
}
for n, c in cases.items():
    print('%-52s slots=%s var_pen=%.1f struct_pen=%.1f' % (n, [i for i,_ in parse_state_slots(c)], state_variation_penalty(c, prompt, 5), structure_penalty(c, 5)))
