import json, math
from common import AWS
def show(seed, pred):
    ev = json.loads((AWS / f'stage9e-llama-causal-flip-post-rl-seed{seed}-v1' / 'stage9e_llama_causal_flip.json').read_text())
    for r in ev['intervention_results']:
        if pred(r):
            print(f'--- MAIN seed{seed}: start={r["starting_state"]} ops={r["operations"]} flip_at_step={r["flip_at_step"]} {r["original_token"]}->{r["flipped_token"]}')
            print('   class:', r['final_answer_classification'], '| extracted:', r['extracted_answer'], '| true_final:', r['true_final_code'], '| counterfactual_final:', r['counterfactual_final_code'])
            print('   subsequent_tokens       :', r['subsequent_tokens']); print('   expected_counterfactual :', r['expected_counterfactual_tokens']); print('   expected_original       :', r['expected_original_tokens'])
            print('   corrupted completion:', repr(r['full_corrupted_completion'][:300]))
            print('   continuation        :', repr(r['continuation'][:300]))
show(44, lambda r: r['final_answer_classification'] != 'tracks_flip_genuine_causal_use')
show(45, lambda r: not r['propagation_matches_counterfactual'])
# exact Clopper-Pearson lower bounds
def cp_lower(k, n, alpha=0.05):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        # P(X>=k | p=mid) > alpha/2 -> mid too high?  compute upper tail
        tail = sum(math.comb(n, i) * mid**i * (1-mid)**(n-i) for i in range(k, n+1))
        if tail > alpha/2: hi = mid
        else: lo = mid
    return hi
for name, k, n in (('each of SFT/42/43/45 final-token 42/42', 42, 42), ('seed44 41/42', 41, 42), ('pooled MAIN 4 seeds final-answer 167/168', 167, 168), ('pooled MAIN trace==counterfactual 166/168', 166, 168)):
    print(f'{name}: 95% CP lower bound = {cp_lower(k, n):.3f}')
