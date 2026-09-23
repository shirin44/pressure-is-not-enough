"""Build the checkpoint-500 greedy-decoding preamble-behavior check.

Pure inference, no training, no GRPO, no optimizer -- decides between the two
candidate mitigations (reward-side structural penalty vs. per-token KL clip)
by checking whether checkpoint 500 exhibits the meta-commentary/instructional
preamble failure identified in the per-token KL instrumentation run
(notebooks/checkpoint500_per_token_kl_instrumentation.ipynb) under plain
greedy (do_sample=False) decoding on a diverse batch of ordinary Coin Flip
prompts.

Detector: anchors on the loose occurrence of "Step 1" (case-insensitive,
anywhere in the reasoning text), not the strict single-line State-parser
regex -- the real step-15 example that motivated this check ("1. Step 1:
...") has a numbered-list prefix that the strict parser never matches at
all, so anchoring on the strict parser would leave "preamble before the
first valid line" undefined for exactly the malformed cases most relevant
here. Everything before that anchor, if non-trivial, is flagged.

Verified locally (see conversation) against the three real positive examples
from the instrumented GRPO run (all three flagged correctly) and a clean
negative (not flagged). One caveat surfaced during that local verification,
carried into this notebook's report: the worst rollout from the healthiest
step in that same run (step 1, mean_kl~=0.0) ALSO had a short leading
instructional-sounding line and was still flagged by this detector -- so
"preamble present" is not the same claim as "preamble was KL-catastrophic".
This notebook checks presence/absence only, which is what the two candidate
mitigations actually turn on; it does not attempt to replicate the
ref-vs-policy KL computation without a training step.

Report interpretation (both cases pre-registered before running):
- ANY flagged completion under greedy decoding -> pre-existing latent
  behavior in checkpoint 500 itself, independent of GRPO sampling. Fix
  belongs upstream (base training or a new narrow reward-side structural
  penalty), not in a post-hoc KL clip.
- ZERO flagged completions -> the behavior is not evident under greedy
  decoding across this batch; more consistent with high-temperature
  sampling occasionally landing in a reference-disfavored region. Points
  toward a per-token KL clip or a lower rollout-sampling temperature.
  Caveat: absence across N draws bounds the rate, it does not prove zero
  (e.g. 0/200 flagged bounds the rate at roughly <=1.5% with ~95%
  confidence by the rule of three); this is reported explicitly, not
  glossed over.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "04_stability_investigation" / "notebooks" / "checkpoint500_greedy_preamble_check.ipynb"
COINFLIP = (ROOT / "src" / "data" / "coinflip.py").read_text()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 greedy-decoding preamble-behavior check

Pure inference -- no training, no GRPO, no optimizer. Loads checkpoint 500
adapter weights only and greedily (`do_sample=False`) decodes a diverse
batch of 200 ordinary Coin Flip prompts, checking for the same
meta-commentary/instructional-preamble failure identified in the per-token
KL instrumentation run: content like *"complete example..."*, *"Replace the
question completely."*, *"Note: The reasoning isn't a step..."* appearing
before or instead of genuine step-by-step reasoning.

**Decision this run makes**: any flagged completion under deterministic
greedy decoding means the behavior is latent in checkpoint 500 itself,
independent of GRPO's temperature sampling -- pointing at an upstream fix
(base training or a new narrow reward-side structural penalty). Zero flagged
completions points more toward temperature sampling occasionally landing in
a reference-disfavored region -- pointing at a per-token KL clip or lower
rollout temperature instead. Reported with the statistical caveat that
absence across N draws bounds the rate, it does not prove zero.
"""),
    code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate\n"),
    code(r'''import gc, importlib.metadata, itertools, json, logging, math, random, re
from pathlib import Path
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from google.colab import drive
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
PROMPT_SEED=20260819  # fresh seed, distinct from every training/diagnostic seed used elsewhere in this repo
N_PROMPTS=200
MAX_NEW_TOKENS=256
BATCH_SIZE=8
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes.autograd._functions').disabled=True

def version(name): return importlib.metadata.version(name)
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4 (bf16 requires Ampere+); found {torch.cuda.get_device_name(0)}.')
expected={'transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.50.0'}
actual={k:version(k) for k in expected}
if actual!=expected: raise RuntimeError(f'Version mismatch: expected={expected}, actual={actual}')
print({'gpu':torch.cuda.get_device_name(0),'prompt_seed':PROMPT_SEED,'n_prompts':N_PROMPTS,**actual})
'''),
    code(r'''drive.mount('/content/drive',force_remount=False)
SOURCE=Path('/content/drive/MyDrive/AISI/checkpoints/full-snapshots/step-500')
if not (SOURCE/'adapter_model.safetensors').is_file():
    raise RuntimeError(f'Missing checkpoint-500 adapter: {SOURCE}')
OUTPUT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/checkpoint500-greedy-preamble-check')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT=OUTPUT_DIR/'greedy_preamble_report.json'
print({'checkpoint':str(SOURCE),'output':str(OUTPUT_DIR)})
'''),
    code(COINFLIP),
    code(r'''print('===== PREAMBLE DETECTOR (verified locally against the 3 real positive examples before this notebook was built) =====')
def detect_preamble(completion_text):
    """Flags meta-commentary/instructional content before, or instead of, genuine
    'Step 1' task reasoning. Anchors on the loose occurrence of 'Step 1'
    (case-insensitive, anywhere), not the strict single-line State-parser regex,
    because malformed variants (e.g. a numbered-list prefix like '1. Step 1: ...')
    never match the strict parser at all, which would leave 'preamble before the
    first valid line' undefined for exactly the cases we care about."""
    reasoning = re.split(r'<answer>', completion_text, maxsplit=1, flags=re.IGNORECASE)[0]
    match = re.search(r'(?i)\bstep\s+1\b', reasoning)
    if match is None:
        stripped = reasoning.strip()
        return {'category': 'no_step_structure_at_all' if stripped else 'empty',
                'preamble_text': stripped, 'flagged': bool(stripped)}
    preamble = reasoning[:match.start()].strip()
    return {'category': 'preamble_before_step1' if preamble else 'clean',
            'preamble_text': preamble, 'flagged': bool(preamble) and len(preamble) > 3}

# Self-test against the three REAL positive examples from the instrumented GRPO run, plus a clean negative.
_positive_1 = (" \n\ncomplete example (but without the final answer tag at the end):\n"
               "1. Step 1: The state remains unchanged. State: Heads\n<answer>Tails</answer>")
_positive_2 = " Replace the question completely. \n\nStep 1: The state remains unchanged. State: Heads\n<answer>Tails</answer>"
_positive_3 = (" Note: The reasoning isn't a step, it's an explanation of a step. So for the step, "
               "you don't need to mention the reasoning at all.\nStep 1: The state remains unchanged. "
               "State: Tails\n<answer>Tails</answer>")
_negative_1 = "Step 1: The state remains unchanged. State: Heads\n<answer>Heads</answer>"
assert detect_preamble(_positive_1)['flagged'] is True
assert detect_preamble(_positive_2)['flagged'] is True
assert detect_preamble(_positive_3)['flagged'] is True
assert detect_preamble(_negative_1)['flagged'] is False
print('PASSED: detector correctly flags all 3 known-positive real examples and a clean negative.')
'''),
    code(r'''print('===== BUILD DIVERSE PROMPT BATCH (fresh seed, no training data reused) =====')
def unique_pool(size, start):
    rows = []; seen = set(); seed = start
    while len(rows) < size:
        prompt, truth = generate_coinflip_example(3 + (seed % 6), seed); seed += 1
        if prompt in seen: continue
        seen.add(prompt); rows.append({'prompt': prompt, 'ground_truth': truth})
    return rows
PROMPT_POOL = unique_pool(N_PROMPTS, PROMPT_SEED)
assert len({x['prompt'] for x in PROMPT_POOL}) == N_PROMPTS
flip_counts = sorted({count_flips(x['prompt']) for x in PROMPT_POOL})
print({'n_prompts': N_PROMPTS, 'unique': True, 'flip_count_range': [min(flip_counts), max(flip_counts)]})
'''),
    code(r'''print('===== LOAD CHECKPOINT-500: ADAPTER WEIGHTS ONLY, NO TRAINING =====')
gc.collect(); torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, quantization_config=quant, device_map='auto', trust_remote_code=False)
base_model.config.use_cache = True
lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','k_proj','v_proj','o_proj'],
                   lora_dropout=.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(base_model, lora)
set_peft_model_state_dict(model, load_safetensors(str(SOURCE/'adapter_model.safetensors')), adapter_name='default')
model.eval()
# get_peft_model() leaves LoRA adapter params requires_grad=True by default (it assumes training is coming
# next); this notebook never calls backward(), so explicitly freeze everything rather than asserting a
# default that PEFT does not actually provide.
for p in model.parameters(): p.requires_grad_(False)
meta = [n for n,p in model.named_parameters() if p.device.type=='meta']
if meta: raise RuntimeError(f'Meta tensors after load: {meta[:5]}')
assert not any(p.requires_grad for p in model.parameters())

ANSWER_IDS = tokenizer.encode('</answer>', add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        width = len(ANSWER_IDS)
        return torch.tensor([row.numel()>=width and row[-width:].tolist()==ANSWER_IDS
                             for row in input_ids], device=input_ids.device, dtype=torch.bool)
ANSWER_STOP = StoppingCriteriaList([StopAfterAnswer()])
print({'checkpoint':'500 (adapter weights only, no optimizer/scheduler loaded)', 'meta_parameters': len(meta),
       'trainable_parameters': 0, 'mode': 'eval, greedy decoding only'})
'''),
    code(r'''print(f'===== GREEDY GENERATION: {N_PROMPTS} PROMPTS, do_sample=False =====')
rows = []
for start in range(0, N_PROMPTS, BATCH_SIZE):
    chunk = PROMPT_POOL[start:start+BATCH_SIZE]
    rendered = [tokenizer.apply_chat_template([{'role':'user','content':item['prompt']}],
                tokenize=False, add_generation_prompt=True) for item in chunk]
    batch = tokenizer(rendered, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            stopping_criteria=ANSWER_STOP, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(out[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    for item, text in zip(chunk, texts):
        rows.append({'prompt': item['prompt'], 'ground_truth': item['ground_truth'], 'completion': text})
    if (start // BATCH_SIZE) % 5 == 0:
        print(f'  generated {min(start+BATCH_SIZE, N_PROMPTS)}/{N_PROMPTS}')
print(f'DONE: {len(rows)} greedy completions generated.')
'''),
    code(r'''print('===== CLASSIFY: PREAMBLE DETECTION + TASK ACCURACY =====')
flagged = []
for row in rows:
    detection = detect_preamble(row['completion'])
    answer_text, format_valid = _extract_answer(row['completion'])
    row['detection'] = detection
    row['answer_correct'] = bool(format_valid and answer_text == normalize_state_token(row['ground_truth']))
    row['format_valid'] = format_valid
    if detection['flagged']:
        flagged.append(row)

n = len(rows)
flagged_count = len(flagged)
accuracy = sum(r['answer_correct'] for r in rows) / n
report = {
    'n_prompts': n, 'flagged_count': flagged_count, 'flagged_rate': flagged_count / n,
    'greedy_task_accuracy': accuracy,
    'rule_of_three_upper_bound_if_zero_flagged': round(3.0 / n, 4),
    'decision': (
        'PRE-EXISTING LATENT BEHAVIOR: at least one greedy completion exhibited the meta-commentary/'
        'instructional-preamble pattern independent of GRPO sampling. Fix belongs upstream (base training '
        'or a new narrow reward-side structural penalty), not a post-hoc per-token KL clip.'
        if flagged_count > 0 else
        f'NOT EVIDENT UNDER GREEDY DECODING across {n} diverse prompts. More consistent with high-temperature '
        'sampling occasionally landing in a reference-disfavored region -- points toward a per-token KL clip '
        'or a lower rollout-sampling temperature. Caveat: this bounds the rate at roughly '
        f'{round(3.0/n,4)*100:.1f}% with ~95% confidence (rule of three), it does not prove a zero rate.'
    ),
}
report_path_json = json.dumps(report, indent=2)
print(report_path_json)
print(f'\n===== ALL {flagged_count} FLAGGED COMPLETIONS (full text) =====')
for i, row in enumerate(flagged, 1):
    print('\n' + '-'*100)
    print({'index': i, 'category': row['detection']['category'], 'preamble_text': row['detection']['preamble_text'],
           'answer_correct': row['answer_correct'], 'ground_truth': row['ground_truth']})
    print(row['completion'])
REPORT.write_text(json.dumps({'report': report, 'rows': rows}, indent=2))
print('\nEvidence:', REPORT)
print('CHECK COMPLETE. Pure inference -- no training was run, no model state changed.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
