"""Q6 / Q7 (+ stop-criterion tokenisation diagnostic). Analysis script; does not modify training/eval code. Runs on the GPU instance.

Part A  tokenisation diagnostic for the stop-after-'</answer>' criterion (CPU tokenizer only).
Part B  (Q6) SFT adapter, GREEDY, on the 21 held-out scenarios (eval_meta), 256 new tokens:
          chat_harness   chat template + tokenizer default BOS (2 BOS)  == llama_code_word_rl.py::_generate_batch / milestone evals
          chat_single    chat template, single BOS                       == SFT training format
          raw            raw prompt text, BOS added                      == what TRL fed RL
          raw_stop       raw + stop-after-'</answer>' criterion          == RL sampling harness, greedy
Part C  (Q7) teacher-forced sequence log-probs of the 21 persisted SFT chat-template completions (+eos) under each adapter:
          SFT (control), SFT reloaded (noise floor), MAIN seed 42/43/44/45 final adapters, MAIN-43's saved frozen 'ref' adapter (control),
          in two prompt formats: chat_sft (single BOS chat prefix, as SFT trained) and raw (BOS + raw prompt text, as RL rolled out).
          Reports mean per-token log-ratio (adapter - SFT) per run."""
import hashlib, json, math, os, sys, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

REPO = Path.home() / 'stage09_repo'
for sub in ('09_direct_indomain_synthetic_bridge', '07_positive_signal_annealed_reward', '09c_sft_diagnostic', '09e_same_different_llama'):
    sys.path.insert(0, str(REPO / 'experiments' / sub))
from reward_v3 import normalize_state_token, parse_state_slots, _extract_answer, score_completion_v2  # noqa: E402
from synthetic_bridge import _trace  # noqa: E402
from code_word_answer_bank import CODE_FOR, build_code_word_prompt, build_code_word_train_eval_split  # noqa: E402

MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
CK = Path.home() / 'aisi_checkpoints'
SFT_DIR = CK / 'stage9e-llama-sft-code-word-v1' / 'final_adapter'
ADAPTERS = [('sft', SFT_DIR), ('main_seed42', CK / 'stage9e-llama-rl-main-v2' / 'final_adapter'), ('main_seed43', CK / 'stage9e-llama-rl-main-v3' / 'final_adapter'),
            ('main_seed44', CK / 'stage9e-llama-rl-main-v4' / 'final_adapter'), ('main_seed45', CK / 'stage9e-llama-rl-main-v5' / 'final_adapter'),
            ('main_seed43_frozen_ref_adapter', CK / 'stage9e-llama-rl-main-v3' / 'final_adapter' / 'ref'), ('sft_reloaded_noise_floor', SFT_DIR)]
MAX_NEW = 256

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'left'
ANSWER_IDS = tok.encode('</answer>', add_special_tokens=False)

train_meta, eval_meta = build_code_word_train_eval_split(seed=20260907, n_eval=21)
sft_ev = json.loads((CK / 'stage9e-llama-sft-code-word-v1' / 'stage9e_llama_sft_code_word.json').read_text())
tier_b = sft_ev['tier_b_heldout_same_pair']['samples']
key = lambda s: (s['starting_state'], tuple(s['operations']))
assert {key(m) for m in eval_meta} == {key(s) for s in tier_b} and len(eval_meta) == 21, 'tier-b scenarios != eval_meta'

report = {'config': {'max_new': MAX_NEW, 'answer_ids': ANSWER_IDS, 'eos_id': tok.eos_token_id, 'n_heldout': 21}}

# ---------------- Part A: stop-criterion tokenisation diagnostic
def tail_is_answer(ids): return ids[-len(ANSWER_IDS):] == ANSWER_IDS
A = {'answer_ids': ANSWER_IDS, 'answer_ids_decoded': [tok.decode([i]) for i in ANSWER_IDS]}
A['sft_tierb_completion_tokenised_alone_ends_with_ANSWER_IDS'] = sum(tail_is_answer(tok.encode(s['completion'], add_special_tokens=False)) for s in tier_b)
A['sft_tierb_last6_tokens_decoded_first3'] = [[tok.decode([i]) for i in tok.encode(s['completion'], add_special_tokens=False)[-6:]] for s in tier_b[:3]]
A['sft_tierb_with_eos_last6_ids_first1'] = tok.encode(tier_b[0]['completion'] + tok.eos_token, add_special_tokens=False)[-6:]
regen_p = sorted(CK.glob('stage9e-step0-rollout-regen-v*/step0_rollout_regen.json'))
if regen_p:
    rg = json.loads(regen_p[-1].read_text())['arms']['train_format']['rows']
    mid = [r for r in rg if '</answer>' in r['completion']][:40]
    A['raw_arm_n_checked'] = len(mid)
    A['raw_arm_prefix_up_to_first_</answer>_ends_with_ANSWER_IDS'] = sum(tail_is_answer(tok.encode(r['completion'][:r['completion'].index('</answer>') + len('</answer>')], add_special_tokens=False)) for r in mid)
    A['raw_arm_ANSWER_IDS_contiguous_anywhere_in_tokenised_completion'] = sum(any(tok.encode(r['completion'], add_special_tokens=False)[i:i + len(ANSWER_IDS)] == ANSWER_IDS for i in range(len(tok.encode(r['completion'], add_special_tokens=False)))) for r in mid)
report['A_stop_criterion_tokenisation'] = A
print('A:', json.dumps(A, indent=1), flush=True)

# ---------------- model
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map='auto', trust_remote_code=False)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], lora_dropout=.05, bias='none', task_type='CAUSAL_LM'))
model.eval()
dev = next(model.parameters()).device

def load_adapter(path):
    w = Path(path) / 'adapter_model.safetensors'
    sha = hashlib.sha256(w.read_bytes()).hexdigest()
    res = set_peft_model_state_dict(model, load_safetensors(str(w)), adapter_name='default')
    assert not getattr(res, 'unexpected_keys', None), res.unexpected_keys
    return sha

class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        w = len(ANSWER_IDS)
        return torch.tensor([row.numel() >= w and row[-w:].tolist() == ANSWER_IDS for row in input_ids], device=input_ids.device, dtype=torch.bool)

def chat_text(p): return tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)

# ---------------- Part B: greedy, held-out 21, SFT adapter
sha_sft = load_adapter(SFT_DIR); assert sha_sft.startswith('3eb41be3'), sha_sft
model.config.use_cache = True
prompts = [build_code_word_prompt(m['starting_state'], m['operations']) for m in eval_meta]
CFG = {'chat_harness': (lambda p: chat_text(p), True, False), 'chat_single': (lambda p: chat_text(p), False, False),
       'raw': (lambda p: p, True, False), 'raw_stop': (lambda p: p, True, True)}

def score_one(m, text):
    exp = [normalize_state_token(CODE_FOR[s]) for s in _trace(m['starting_state'], m['operations'])]
    got = [t for _i, t in parse_state_slots(text)]
    ans, valid = _extract_answer(text)
    b = score_completion_v2(text, m['final_answer_code'], 1, 150, prompt=build_code_word_prompt(m['starting_state'], m['operations']))
    return {'format_valid': valid, 'correct': bool(valid and ans == normalize_state_token(m['final_answer_code'])), 'intermediate_ok': got == exp, 'r_task': b['r_task']}

B = {}
for name, (fmt, add_sp, stop) in CFG.items():
    texts = [fmt(p) for p in prompts]; rows = []
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=add_sp).to(dev)
    kw = dict(do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if stop: kw['stopping_criteria'] = StoppingCriteriaList([StopAfterAnswer()])
    with torch.inference_mode():
        out = model.generate(**enc, **kw)
    gen = out[:, enc['input_ids'].shape[1]:].cpu().tolist()
    for m, ids in zip(eval_meta, gen):
        eos_pos = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else None
        cut = ids[:eos_pos + 1] if eos_pos is not None else ids
        while eos_pos is None and cut and cut[-1] == tok.pad_token_id: cut = cut[:-1]
        text = tok.decode(cut, skip_special_tokens=True)
        rows.append({'starting_state': m['starting_state'], 'operations': m['operations'], 'completion': text, 'n_tokens': len(cut), 'ended_eos': eos_pos is not None,
                     'hit_cap': eos_pos is None and len(cut) >= MAX_NEW and not tail_is_answer(cut), 'ended_answer_stop': eos_pos is None and tail_is_answer(cut), **score_one(m, text)})
    n = len(rows)
    B[name] = {'summary': {'n': n, 'malformed_rate(no parseable final answer)': sum(not r['format_valid'] for r in rows) / n, 'accuracy(final answer correct)': sum(r['correct'] for r in rows) / n,
                           'intermediate_tracking_ok': sum(r['intermediate_ok'] for r in rows) / n, 'mean_tokens': sum(r['n_tokens'] for r in rows) / n,
                           'hit_cap_256': sum(r['hit_cap'] for r in rows) / n, 'ended_eos': sum(r['ended_eos'] for r in rows) / n, 'ended_answer_stop': sum(r['ended_answer_stop'] for r in rows) / n,
                           'r_task_counts': {str(k): sum(r['r_task'] == k for r in rows) for k in (4.0, -0.5, -5.0)}}, 'rows': rows}
    print(f'B[{name}]:', json.dumps(B[name]['summary']), flush=True)
    # greedy chat_harness should reproduce the persisted SFT tier-b completions
    if name == 'chat_harness':
        per = {key(s): s['completion'] for s in tier_b}
        B[name]['summary']['identical_to_persisted_sft_tier_b'] = sum(per[(r['starting_state'], tuple(r['operations']))] == r['completion'] for r in rows)
report['B_greedy_heldout21_sft_adapter'] = B

# ---------------- Part C: drift via teacher-forced log-probs
model.config.use_cache = False
def seq_lp(prefix_ids, comp_ids):
    ids = torch.tensor([prefix_ids + comp_ids], device=dev)
    with torch.inference_mode():
        lg = model(input_ids=ids, use_cache=False).logits[0]
    s = len(prefix_ids)
    lp = torch.log_softmax(lg[s - 1:-1].float(), dim=-1)
    return lp.gather(1, ids[0, s:, None])[:, 0].cpu().tolist()

def build_examples():
    ex = {'chat_sft': [], 'raw': []}
    for s in tier_b:
        p = build_code_word_prompt(s['starting_state'], s['operations'])
        pre = chat_text(p); pre_ids = tok.encode(pre, add_special_tokens=False)
        full = tok.encode(pre + s['completion'] + tok.eos_token, add_special_tokens=False)
        ex['chat_sft'].append((pre_ids, full[len(pre_ids):]))
        raw_ids = tok(p)['input_ids']                      # BOS + raw prompt, as TRL tokenised it
        ex['raw'].append((raw_ids, tok.encode(s['completion'] + tok.eos_token, add_special_tokens=False)))
    return ex
EX = build_examples()
assert EX['chat_sft'][0][1][-1] == tok.eos_token_id

LP, SHAS = {}, {}
for name, path in ADAPTERS:
    SHAS[name] = load_adapter(path)
    LP[name] = {fmt: [seq_lp(p, c) for p, c in EX[fmt]] for fmt in EX}
    print('C: computed', name, SHAS[name][:12], flush=True)

def stats(name, fmt):
    ref = LP['sft'][fmt]; cur = LP[name][fmt]
    d = [[a - b for a, b in zip(rc, rr)] for rc, rr in zip(cur, ref)]
    flat = [x for s in d for x in s]
    return {'n_sequences': len(d), 'n_tokens': len(flat), 'mean_per_token_logratio(adapter-SFT)': sum(flat) / len(flat),
            'mean_of_per_sequence_means': sum(sum(s) / len(s) for s in d) / len(d), 'mean_abs_logratio': sum(abs(x) for x in flat) / len(flat),
            'max_abs_logratio': max(abs(x) for x in flat), 'frac_tokens_abs_gt_0.04': sum(abs(x) > 0.04 for x in flat) / len(flat),
            'mean_sequence_logprob_sft': sum(sum(s) for s in ref) / len(ref), 'mean_sequence_logprob_adapter': sum(sum(s) for s in cur) / len(cur),
            'mean_per_token_logprob_sft': sum(x for s in ref for x in s) / len(flat), 'mean_per_token_logprob_adapter': sum(x for s in cur for x in s) / len(flat)}
C = {name: {fmt: stats(name, fmt) for fmt in EX} for name, _p in ADAPTERS}
report['C_drift'] = {'adapter_sha256': SHAS, 'stats': C}
for name in C:
    print(f'C[{name}]', {f: round(C[name][f]['mean_per_token_logratio(adapter-SFT)'], 6) for f in C[name]}, flush=True)

out_dir = None
for v in range(1, 1000):
    out_dir = CK / f'stage9e-raw-vs-chat-drift-v{v}'
    if not out_dir.exists(): break
out_dir.mkdir(parents=True)
(out_dir / 'raw_vs_chat_drift.json').write_text(json.dumps(report, indent=1, default=str))
print('Evidence:', out_dir / 'raw_vs_chat_drift.json'); print('RAW VS CHAT DRIFT COMPLETE.')
