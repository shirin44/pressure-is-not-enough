import sys, json
from pathlib import Path
REPO = Path('/Users/researcher/Desktop/AISI')
E = REPO / 'experiments'
for p in ('07_positive_signal_annealed_reward', '09e_same_different_llama', '09c_sft_diagnostic', '09d_sft_seeded_adversarial_rl', '09_direct_indomain_synthetic_bridge'):
    sys.path.insert(0, str(E / p))
AWS = E / '09e_same_different_llama' / 'aws_runs'
RUNS = {  # (condition, seed) -> (dir, file)
    ('BASELINE', 42): ('stage9e-llama-rl-baseline-v1', 'stage9e_llama_rl_baseline.json'),
    ('BASELINE', 43): ('stage9e-llama-rl-baseline-v2', 'stage9e_llama_rl_baseline.json'),
    ('BASELINE', 44): ('stage9e-llama-rl-baseline-v3', 'stage9e_llama_rl_baseline.json'),
    ('BASELINE', 45): ('stage9e-llama-rl-baseline-v4', 'stage9e_llama_rl_baseline.json'),
    ('MAIN', 42): ('stage9e-llama-rl-main-v2', 'stage9e_llama_rl_main.json'),
    ('MAIN', 43): ('stage9e-llama-rl-main-v3', 'stage9e_llama_rl_main.json'),
    ('MAIN', 44): ('stage9e-llama-rl-main-v4', 'stage9e_llama_rl_main.json'),
    ('MAIN', 45): ('stage9e-llama-rl-main-v5', 'stage9e_llama_rl_main.json'),
}
def load_run(key):
    d, f = RUNS[key]
    return json.loads((AWS / d / f).read_text())
def load_sft():
    return json.loads((AWS / 'stage9e-llama-sft-code-word-v1' / 'stage9e_llama_sft_code_word.json').read_text())
