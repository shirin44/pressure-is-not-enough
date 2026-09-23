"""Q4/Q5: regenerate step-0 training rollouts from the SFT adapter with the RL sampling settings and re-score with the TRAINING penalty.

Analysis script (does not modify training/eval code). Runs on the GPU instance.
Mirrors llama_code_word_rl.py:
  - model load: 8-bit Llama-3-8B-Instruct, prepare_model_for_kbit_training, fresh get_peft_model(LoraConfig r=8 a=16 q/k/v/o, dropout .05), SFT weights set into 'default' (RL:184-209)
  - sampling: do_sample, temperature .8, top_p .95, top_k 0 (TRL default), max_new_tokens 256, eos/pad/bos from the tokenizer (pad := eos), stop-after-'</answer>' criterion (RL:212-217, 336-341)
  - prompts: the 43 TRAIN scenarios (build_code_word_train_eval_split(seed=20260907, n_eval=21))
Three prompt-format arms, because TRL 1.9.2 tokenises a plain-string prompt with `processing_class(text=prompts)` (grpo_trainer.py:1780; no chat template, BOS added),
while the SFT training and every milestone eval used the chat template (llama_sft_code_word_seeding.py:281-283, llama_code_word_rl.py:238-239):
  arm 'train_format' = raw prompt string exactly as RL fed it; 'chat_eval_harness' = chat-wrapped + tokenizer default BOS (as the milestone eval harness fed it, i.e. two BOS);
  'chat_sft_exact' = chat-wrapped, single BOS (as SFT training saw it).
Scoring uses the real reward_v3.score_completion_v2 (MAIN params) at step 1 (cot_scale 0.2) and step 150 (cot_scale 2.0), plus the eval scanner and a case-insensitive whole-text check."""
import hashlib, json, math, os, random, sys, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList)

REPO = Path.home() / 'stage09_repo'
for sub in ('09_direct_indomain_synthetic_bridge', '07_positive_signal_annealed_reward', '09c_sft_diagnostic', '09e_same_different_llama'):
    sys.path.insert(0, str(REPO / 'experiments' / sub))
from reward_v3 import score_completion_v2, _extract_answer, ALL_BANNED_PATTERNS  # noqa: E402
from code_word_answer_bank import build_code_word_prompt, build_code_word_train_eval_split  # noqa: E402
from same_different_leakage_audit import check_completion_for_leakage  # noqa: E402
import re  # noqa: E402

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
ADAPTER_DIR = Path(os.environ.get('REGEN_ADAPTER_DIR', str(Path.home() / 'aisi_checkpoints' / 'stage9e-llama-sft-code-word-v1' / 'final_adapter'))).expanduser()
N_PER_PROMPT = int(os.environ.get('REGEN_N_PER_PROMPT', '16'))
BATCH = int(os.environ.get('REGEN_BATCH', '48'))
SEED = int(os.environ.get('REGEN_SEED', '20260921'))
MAX_NEW = 256
GROUP = 8
TAG = os.environ.get('REGEN_TAG', 'stage9e-step0-rollout-regen')

weights = ADAPTER_DIR / 'adapter_model.safetensors'
adapter_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
print({'adapter_dir': str(ADAPTER_DIR), 'adapter_sha256': adapter_sha, 'n_per_prompt': N_PER_PROMPT, 'seed': SEED, 'gpu': torch.cuda.get_device_name(0)})

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'
print({'eos_token': tok.eos_token, 'eos_id': tok.eos_token_id, 'pad_id': tok.pad_token_id, 'bos_id': tok.bos_token_id})

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                            device_map='auto', trust_remote_code=False)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM'))
res = set_peft_model_state_dict(model, load_safetensors(str(weights)), adapter_name='default')
assert not getattr(res, 'unexpected_keys', None), res.unexpected_keys
assert list(model.peft_config) == ['default']
model.eval(); model.config.use_cache = True

ANSWER_IDS = tok.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        w = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= w and row[-w:].tolist() == ANSWER_IDS for row in input_ids], device=input_ids.device, dtype=torch.bool)
STOP = StoppingCriteriaList([StopAfterAnswer()])

train_meta, _eval_meta = build_code_word_train_eval_split(seed=20260907, n_eval=21)
scen = [{'starting_state': m['starting_state'], 'operations': m['operations'], 'ground_truth': m['final_answer_code'],
         'prompt': build_code_word_prompt(m['starting_state'], m['operations'])} for m in train_meta]
print({'n_train_scenarios': len(scen)})

def fmt(arm, prompt):
    if arm == 'train_format':
        return prompt  # raw string, exactly what TRL tokenised
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

# add_special_tokens per arm:
#  train_format      True  == TRL processing_class(text=...) (BOS prepended to the raw prompt)
#  chat_eval_harness True  == llama_code_word_rl.py::_generate_batch / SFT eval: tokenizer(chunk, padding=True) on the ALREADY chat-templated text -> a second BOS
#  chat_sft_exact    False == llama_sft_code_word_seeding.py:282 tokenizer.encode(prefix, add_special_tokens=False) -> single BOS as in SFT training
ADD_SPECIAL = {'train_format': True, 'chat_eval_harness': True, 'chat_sft_exact': False}

def generate(arm, texts, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    outs = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        enc = tok(chunk, return_tensors='pt', padding=True, add_special_tokens=ADD_SPECIAL[arm]).to(model.device)
        with torch.inference_mode():
            o = model.generate(**enc, do_sample=True, temperature=.8, top_p=.95, top_k=0, max_new_tokens=MAX_NEW,
                               pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, stopping_criteria=STOP)
        if i == 0:
            print('  first input ids (arm %s):' % arm, enc['input_ids'][0][-400:][:6].tolist() if False else [t for t in enc['input_ids'][0].tolist() if t != tok.pad_token_id][:4], flush=True)
        gen = o[:, enc['input_ids'].shape[1]:].cpu()
        for row in gen:
            ids = row.tolist()
            eos_pos = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else None
            cut = ids[:eos_pos + 1] if eos_pos is not None else ids
            # trailing pads are eos==pad, so strip them when no eos found earlier
            while eos_pos is None and cut and cut[-1] == tok.pad_token_id:
                cut = cut[:-1]
            outs.append({'ids': cut, 'ended_eos': eos_pos is not None,
                         'ended_answer_stop': eos_pos is None and cut[-len(ANSWER_IDS):] == ANSWER_IDS,
                         'hit_cap': eos_pos is None and len(cut) >= MAX_NEW and cut[-len(ANSWER_IDS):] != ANSWER_IDS,
                         'text': tok.decode(cut, skip_special_tokens=True), 'n_tokens': len(cut)})
        print(f'  [{arm}] {min(i + BATCH, len(texts))}/{len(texts)} generated', flush=True)
    return outs

CI_WORD = re.compile(r'(?<![A-Za-z])(heads|tails|head|tail)(?![A-Za-z])', re.I)
def score(c, sc):
    ans, valid = _extract_answer(c['text'])
    b1 = score_completion_v2(c['text'], sc['ground_truth'], 1, 150, prompt=sc['prompt'])
    b150 = score_completion_v2(c['text'], sc['ground_truth'], 150, 150, prompt=sc['prompt'])
    whole_count = sum(len(re.findall(p, c['text'])) for p, _ in ALL_BANNED_PATTERNS)
    return {'format_valid': valid, 'correct': bool(valid and b1['r_task'] == 4.0), 'no_delimiter': '<answer>' not in c['text'],
            'banned_count_train_span': b1['banned_count'], 'banned_count_whole_completion': float(whole_count),
            'p_cot_step1': b1['p_cot'], 'p_cot_step150': b150['p_cot'],
            'reward_main_step1': b1['total'], 'reward_main_step150': b150['total'], 'r_task': b1['r_task'],
            'eval_scanner_flag': not check_completion_for_leakage(c['text'])['clean'], 'ci_word_anywhere': bool(CI_WORD.search(c['text']))}

report = {'config': {'adapter_dir': str(ADAPTER_DIR), 'adapter_sha256': adapter_sha, 'n_per_prompt': N_PER_PROMPT, 'seed': SEED, 'max_new_tokens': MAX_NEW,
                     'temperature': .8, 'top_p': .95, 'top_k': 0, 'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
                     'n_train_scenarios': len(scen)}, 'arms': {}}
t0 = time.time()
for arm in ('train_format', 'chat_eval_harness', 'chat_sft_exact'):
    jobs = [(si, k) for si in range(len(scen)) for k in range(N_PER_PROMPT)]
    texts = [fmt(arm, scen[si]['prompt']) for si, _k in jobs]
    comps = generate(arm, texts, SEED)
    rows = []
    for (si, k), c in zip(jobs, comps):
        s = score(c, scen[si])
        rows.append({'scenario': si, 'sample': k, 'starting_state': scen[si]['starting_state'], 'operations': scen[si]['operations'],
                     'completion': c['text'], 'n_tokens': c['n_tokens'], 'ended_eos': c['ended_eos'],
                     'ended_answer_stop': c['ended_answer_stop'], 'hit_cap': c['hit_cap'], **s})
    n = len(rows)
    frac = lambda f: sum(bool(f(r)) for r in rows) / n
    # pseudo-groups of GROUP consecutive samples of the same scenario, as in one GRPO step
    groups = []
    for si in range(len(scen)):
        rs = [r for r in rows if r['scenario'] == si]
        for g in range(0, len(rs) - GROUP + 1, GROUP):
            groups.append(rs[g:g + GROUP])
    def gstd(g, key):
        v = [r[key] for r in g]; m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    summary = {
        'n_completions': n,
        'format_valid': frac(lambda r: r['format_valid']), 'correct': frac(lambda r: r['correct']), 'no_delimiter': frac(lambda r: r['no_delimiter']),
        'ended_eos': frac(lambda r: r['ended_eos']), 'ended_answer_stop': frac(lambda r: r['ended_answer_stop']), 'hit_cap_256': frac(lambda r: r['hit_cap']),
        'mean_tokens': sum(r['n_tokens'] for r in rows) / n,
        'nonzero_p_cot (banned pattern in TRAIN span)': frac(lambda r: r['banned_count_train_span'] > 0),
        'banned_pattern_anywhere_whole_completion': frac(lambda r: r['banned_count_whole_completion'] > 0),
        'eval_scanner_flag': frac(lambda r: r['eval_scanner_flag']), 'case_insensitive_word_anywhere': frac(lambda r: r['ci_word_anywhere']),
        'n_groups': len(groups),
        'groups_zero_reward_variance_step1': sum(gstd(g, 'reward_main_step1') < 1e-8 for g in groups) / len(groups),
        'groups_with_any_nonzero_p_cot': sum(any(r['banned_count_train_span'] > 0 for r in g) for g in groups) / len(groups),
        'mean_reward_main_step1': sum(r['reward_main_step1'] for r in rows) / n,
    }
    report['arms'][arm] = {'summary': summary, 'rows': rows}
    print(f'===== ARM {arm} =====')
    print(json.dumps(summary, indent=1))
print('elapsed_s', round(time.time() - t0))

root = Path.home() / 'aisi_checkpoints'
for v in range(1, 1000):
    out = root / f'{TAG}-v{v}'
    if not out.exists():
        break
out.mkdir(parents=True)
(out / 'step0_rollout_regen.json').write_text(json.dumps(report, indent=1, default=str))
print('Evidence:', out / 'step0_rollout_regen.json')
print('STEP0 ROLLOUT REGEN COMPLETE.')
