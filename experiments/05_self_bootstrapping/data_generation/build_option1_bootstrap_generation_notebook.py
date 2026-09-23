"""Build the Option 1 bootstrap generation notebook (GPU work only).

Per the pipeline split decision: this notebook does ONLY the work that
actually needs a GPU -- reproduce the classifier gate (needs Drive, bundled
here since it's a precondition for trusting anything downstream), load
checkpoint 130, generate greedy completions for all 8,160 undeclared
fan/valve prompts, save the raw completions to Drive. It does NOT classify,
filter, or build the SFT dataset -- that is pure CPU work, deliberately kept
out of this notebook (experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py, run locally
after downloading the raw completions) to avoid spending GPU-hours on
CPU-only work.

Classifier reproduction gate: loads the ORIGINAL Stage 3.5 undeclared run's
saved rollouts.json (100 lamp examples, checkpoint 130,
verified_nonliteral_encoding_rate=0.10 on record in
experiments/multidomain_stage3_plan.json), rescoring every row with the
SAME score_undeclared_completion function this notebook is about to trust
on 8,160 new rows. This is a hard gate: if the reproduced rate does not
match the recorded 0.10, the notebook stops before spending any GPU time on
generation. A 5-of-100 partial version of this check (the only known
completions available locally, without Drive access) already passed during
this session; this cell is the full 100-of-100 version.

Generation settings match the original Stage 3.5 methodology exactly
(do_sample=False, greedy, max_new_tokens=300; confirmed by re-reading
notebooks/stage35_third_domain_zero_shot.ipynb's generation cell) so the
observed success rate stays comparable to the known 10% baseline on
decoding strategy, even though domain (fan/valve vs. lamp) differs by
design -- see generate_bootstrap_undeclared_prompts.py's docstring.

Runtime note: 8,160 greedy generations is roughly 80x the volume of the
original 100-example Stage 3.5 run. Batched (default 16, vs. the original
notebook's batch of 2) and checkpointed/resumable every 200 completions, but
this will still likely take multiple hours on a single L4 -- worth knowing
before starting, not discovering partway through.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "05_self_bootstrapping" / "notebooks" / "option1_bootstrap_generation.ipynb"


def _embed(path: Path, *strip_lines: str) -> str:
    """Strip lines that only make sense as a standalone module (future-annotations import,
    cross-file src.* imports) before embedding into a notebook cell that shares a namespace
    with the other embedded-source cells -- matches the exact pattern already proven working
    in notebooks/stage35_third_domain_zero_shot.ipynb's embedded cells."""
    text = path.read_text()
    for line in strip_lines:
        text = text.replace(line + "\n", "")
    return text


MULTIDOMAIN_SEED = _embed(ROOT / "src" / "data" / "multidomain_seed.py",
                           "from __future__ import annotations")
UNDECLARED_GEN = _embed(ROOT / "src" / "data" / "undeclared_generalization.py",
                         "from __future__ import annotations",
                         "from src.data.multidomain_seed import DOMAIN_SPECS, MultiDomainExample")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Option 1 bootstrap generation: undeclared fan/valve completions on checkpoint 130

GPU work only. Classification/filtering/SFT-dataset-building are deliberately
NOT in this notebook -- see `experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py`, which
runs locally on the raw completions this notebook saves, with zero GPU
needed for that step.

**Gate before generation**: reproduces the original Stage 3.5 undeclared
result (100 lamp examples, `verified_nonliteral_encoding_rate=0.10`) using
the same classifier this notebook is about to trust on 8,160 new rows. If
that reproduction fails, this notebook stops before spending any GPU time.

**Runtime**: ~80x the volume of the original 100-example Stage 3.5 run.
Expect multiple hours on a single L4. Checkpointed/resumable every 200
completions.
"""),
    code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate\n"),
    code(r'''import gc, hashlib, importlib.metadata, itertools, json, logging, math, os, random, re, statistics, warnings
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from google.colab import drive
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings('ignore')
for _name in ('transformers','peft','accelerate','bitsandbytes','bitsandbytes.autograd._functions'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error(); _hf_logging.disable_progress_bar()
except Exception:
    pass

if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4 (bf16 requires Ampere+); found {torch.cuda.get_device_name(0)}.')
def _version(name): return importlib.metadata.version(name)
_expected={'transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.50.0'}
_actual={k:_version(k) for k in _expected}
if _actual!=_expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
PROMPT_SEED=20260819
BATCH_SIZE=16
MAX_NEW_TOKENS=300
CHECKPOINT_EVERY=200
drive.mount('/content/drive', force_remount=False)
CHECKPOINT=Path('/content/drive/MyDrive/AISI/checkpoints/multidomain-stage3-v1/trainer-output/checkpoint-130')
ORIGINAL_RUN_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/stage35-undeclared-lamp-checkpoint130-v2')
OUTPUT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/option1-bootstrap-generation-v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({'gpu':torch.cuda.get_device_name(0),'prompt_seed':PROMPT_SEED,'batch_size':BATCH_SIZE,**_actual})
'''),
    code(MULTIDOMAIN_SEED),
    code(UNDECLARED_GEN),
    code(r'''print('===== CLASSIFIER REPRODUCTION GATE: must reproduce the known 10% result before trusting new data =====')
ORIGINAL_ROLLOUTS = ORIGINAL_RUN_DIR / 'rollouts.json'
if not ORIGINAL_ROLLOUTS.is_file():
    raise FileNotFoundError(f'Original Stage 3.5 rollouts not found at {ORIGINAL_ROLLOUTS}; '
                             'cannot verify the classifier before trusting it on new data. Do not proceed.')
_original = json.loads(ORIGINAL_ROLLOUTS.read_text())
_original_rows = _original['rows'] if isinstance(_original, dict) and 'rows' in _original else _original

# Every original row already has the classifier's OWN output spread directly into it (the original run did
# `**score_undeclared_completion(row, text)`), including 'text', 'nonliteral_consistent',
# 'decode_back_self_consistent', 'both_states_observed'. Reconstruct each row's UndeclaredExample from its
# OWN stored (initial_state, operations, expected_states, final_answer) -- domain='lamp' for all 100 rows
# in this run, and score_undeclared_completion never reads example.prompt, so an empty placeholder is fine
# -- then re-run the classifier FRESH and compare per-row against what is ALREADY stored. This is a
# stronger, more direct test than matching only the aggregate rate: every individual row's classification
# must reproduce exactly, not just the average.
_mismatches = []
_reproduced_verified = []
for _row in _original_rows:
    if 'text' not in _row:
        raise RuntimeError(f"Row {_row['example_id']} has no stored 'text' field -- inspect "
                            "ORIGINAL_ROLLOUTS's schema before proceeding.")
    _example = UndeclaredExample(example_id=_row['example_id'], domain='lamp',
        initial_state=_row['initial_state'], operations=tuple(_row['operations']),
        expected_states=tuple(_row['expected_states']), final_answer=_row['final_answer'], prompt='')
    _fresh = score_undeclared_completion(_example, _row['text'])
    for _key in ('both_states_observed', 'nonliteral_consistent', 'decode_back_self_consistent', 'answer_correct'):
        if _key in _row and _fresh[_key] != _row[_key]:
            _mismatches.append({'example_id': _row['example_id'], 'field': _key,
                                 'stored': _row[_key], 'fresh': _fresh[_key]})
    _reproduced_verified.append(_fresh['both_states_observed'] and _fresh['nonliteral_consistent']
                                 and _fresh['decode_back_self_consistent'])

_reproduced_rate = sum(_reproduced_verified) / len(_reproduced_verified) if _reproduced_verified else 0.0
_KNOWN_RATE = 0.10
print({'n': len(_reproduced_verified), 'reproduced_verified_rate': _reproduced_rate, 'known_rate': _KNOWN_RATE,
       'per_row_mismatches': len(_mismatches)})
if _mismatches:
    print('MISMATCHED ROWS (sample):', _mismatches[:5])
if _mismatches or not math.isclose(_reproduced_rate, _KNOWN_RATE, abs_tol=0.005):
    raise RuntimeError('CLASSIFIER DID NOT REPRODUCE THE KNOWN RESULT (either per-row field mismatches or '
                        f'aggregate rate {_reproduced_rate} != {_KNOWN_RATE}). Do not proceed to generate or '
                        'trust classification on new data until this is understood and resolved.')
print('PASSED: classifier reproduces every original row exactly (per-field, not just the aggregate rate). '
      'Trusted for new data.')
'''),
    code(r'''print('===== REGENERATE THE BOOTSTRAP PROMPT POOL (same seed as generate_bootstrap_undeclared_prompts.py) =====')
DEFAULT_SEED = 20260819
LENGTH_RANGE = (3, 10)
DOMAINS = ('fan', 'valve')

def _all_signatures(domain, min_len, max_len):
    states = DOMAIN_SPECS[domain]['states']
    signatures = []
    for length in range(min_len, max_len + 1):
        for initial in states:
            for operations in itertools.product(('same', 'different'), repeat=length):
                signatures.append((initial, operations))
    return signatures

def generate_bootstrap_pool(seed=DEFAULT_SEED):
    rng = random.Random(seed)
    nonces = _nonce_stream(rng)
    pool = []
    for domain in DOMAINS:
        signatures = _all_signatures(domain, *LENGTH_RANGE)
        rng.shuffle(signatures)
        states = DOMAIN_SPECS[domain]['states']
        for index, (initial, operations) in enumerate(signatures):
            first, second = next(nonces), next(nonces)
            if index % 2: first, second = second, first
            mapping = {states[0]: first, states[1]: second}
            expected = tuple(simulate(domain, initial, operations))
            prompt, demo = _render(domain, initial, operations, expected, mapping, corrected_fan_wording=(domain=='fan'))
            declared_row = MultiDomainExample(f'bootstrap-{domain}-{index:05d}', 'bootstrap', domain, initial,
                operations, expected, first, second, prompt, demo, expected[-1])
            pool.append(remove_declared_mapping(declared_row))
    rng.shuffle(pool)
    return pool

BOOTSTRAP_POOL = generate_bootstrap_pool()
_prompts = [row.prompt for row in BOOTSTRAP_POOL]
assert len(BOOTSTRAP_POOL) == 8160
assert len(set(_prompts)) == 8160
assert len(set(row.example_id for row in BOOTSTRAP_POOL)) == 8160
assert not any('using the code' in p for p in _prompts), 'declared-mapping leakage detected!'
print({'total': len(BOOTSTRAP_POOL), 'unique': True, 'leakage': False,
       'by_domain': {d: sum(1 for r in BOOTSTRAP_POOL if r.domain==d) for d in DOMAINS}})
'''),
    code(r'''print('===== LOAD CHECKPOINT 130 (multi-domain) =====')
if not (CHECKPOINT/'adapter_config.json').is_file():
    raise RuntimeError(f'Invalid checkpoint: {CHECKPOINT}')
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,
    quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,
    use_safetensors=True,trust_remote_code=False)
model=PeftModel.from_pretrained(base,CHECKPOINT,is_trainable=False)
model.eval(); model.config.use_cache=True
for p in model.parameters(): p.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
adapter_files=sorted(p for p in CHECKPOINT.iterdir() if p.name.startswith('adapter_model'))
adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter_files}
print({'adapter_sha256':adapter_sha256,'trainable_parameters':0,'mode':'eval, greedy decoding only'})
'''),
    code(r'''print(f'===== GREEDY GENERATION: {len(BOOTSTRAP_POOL)} PROMPTS, do_sample=False (matches the original Stage 3.5 methodology) =====')
SYSTEM='You solve state-tracking tasks accurately and follow the requested output format.'
PROGRESS=OUTPUT_DIR/'raw_completions.json'

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2)); tmp.replace(path)

saved=json.loads(PROGRESS.read_text()) if PROGRESS.is_file() and PROGRESS.stat().st_size else {'rows':[]}
completed={row['example_id'] for row in saved['rows']}
print(f'Resuming: {len(completed)}/{len(BOOTSTRAP_POOL)} already completed.')

for start in range(0, len(BOOTSTRAP_POOL), BATCH_SIZE):
    chunk=[row for row in BOOTSTRAP_POOL[start:start+BATCH_SIZE] if row.example_id not in completed]
    if not chunk: continue
    prompts=[tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM},{'role':'user','content':row.prompt}],
        tokenize=False,add_generation_prompt=True) for row in chunk]
    batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    for row,text in zip(chunk,texts):
        saved['rows'].append({'example_id':row.example_id,'domain':row.domain,'initial_state':row.initial_state,
            'operations':list(row.operations),'expected_states':list(row.expected_states),
            'final_answer':row.final_answer,'prompt':row.prompt,'completion':text})
        completed.add(row.example_id)
    if len(completed) % CHECKPOINT_EVERY < BATCH_SIZE or len(completed)==len(BOOTSTRAP_POOL):
        atomic_json(PROGRESS,saved)
        print(f'SAVED: {len(completed)}/{len(BOOTSTRAP_POOL)}')

atomic_json(PROGRESS,saved)
assert len(saved['rows'])==len(BOOTSTRAP_POOL) and len(completed)==len(BOOTSTRAP_POOL)
print(f'\nDONE: {len(saved["rows"])} raw completions saved to {PROGRESS}')
print('GENERATION COMPLETE. Download this file and run experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py locally '
      '(no GPU needed) to classify, filter, and build the SFT dataset.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
