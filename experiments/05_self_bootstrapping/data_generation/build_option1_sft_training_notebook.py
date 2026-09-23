"""Build (do not run) the Option 1 bootstrap SFT training notebook.

Continues from checkpoint 130 (not a fresh LoRA -- the whole point of
self-bootstrapping is reinforcing the model's own rare successes on top of
what it already knows, not discarding the existing declared-mapping
capability), training on the bootstrap SFT dataset built by
experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py from checkpoint 130's own undeclared
completions.

Follows the exact gated Trainer/TrainerCallback/score/aggregate pattern
already established and verified in
scripts/build_multidomain_stage3_notebook.py (first_gate) --
periodic milestone evaluation, checkpoint verification via valid_checkpoint,
adapter fingerprinting, atomic JSON progress saves -- extended with the
undeclared-condition evaluation as the PRIMARY metric this run, since that
is the actual capability being trained for.

## Stop conditions (bounded, matching the multi-domain Stage 3 pattern)

- MAX_STEPS = 150, milestone evaluation every 30 steps (30/60/90/120/150) --
  same horizon and cadence as multidomain_stage3_plan.json's original run.
- Early-failure: nonfinite loss/grad_norm (same as first_gate's FirstGate
  callback), OR pooled declared-condition accuracy drops more than 15 points
  below checkpoint 130's own recorded baseline (89.3%, see
  experiments/multidomain_stage3_plan.json's step130_result) -- a regression
  guard so chasing the undeclared metric can't silently destroy the
  existing declared-mapping capability.
- Early-success: undeclared_verified_rate on held-out lamp reaches >=0.30 at
  any milestone (3x the original 10% baseline -- a threshold meant to
  represent real, substantial improvement, not milestone-to-milestone
  noise on a small eval set).
- Otherwise: continues to MAX_STEPS, then stops regardless and requires
  human review before any further continuation, matching this project's
  standing rule that no stage authorizes its own continuation.

Same acceptance metrics as every prior stage (physical_final_answer_accuracy,
structural_format_rate, transition_tracking_rate, global_mapping_consistency_rate,
declared_mapping_adherence_rate, decode_back_specific_accuracy) reused
unchanged from the first_gate score()/aggregate() functions, evaluated on
the standard declared fan/valve/lamp sets as a regression check, PLUS the
new undeclared-condition success rate (verified_nonliteral_encoding_rate,
final_answer_accuracy, global_self_consistency_rate,
decode_back_self_consistency_rate) on held-out lamp via the unchanged
score_undeclared_completion, as the primary metric.

Does NOT run automatically -- this script only builds the notebook. Before
running it on Colab: (1) run experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py locally
on the raw completions from experiments/05_self_bootstrapping/notebooks/option1_bootstrap_generation.ipynb,
(2) upload the resulting bootstrap_sft_dataset.json to
/content/drive/MyDrive/AISI/checkpoints/option1-bootstrap-sft-v1/bootstrap_sft_dataset.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "05_self_bootstrapping" / "notebooks" / "option1_sft_training.ipynb"


def _embed(path: Path, *strip_lines: str) -> str:
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
    markdown("""# Option 1 bootstrap SFT training (built, NOT run)

Continues checkpoint 130's adapter, training on the bootstrap SFT dataset
(the model's own filtered, verified undeclared-invention successes on
fan/valve). Undeclared-condition success rate on held-out lamp is the
PRIMARY metric -- that's the actual thing this is trying to move. Standard
declared-condition metrics are also tracked as a regression guard.

**Prerequisite**: upload `bootstrap_sft_dataset.json` (built locally by
`experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py` from
`experiments/05_self_bootstrapping/notebooks/option1_bootstrap_generation.ipynb`'s raw completions) to
`/content/drive/MyDrive/AISI/checkpoints/option1-bootstrap-sft-v1/bootstrap_sft_dataset.json`
before running this.

Bounded at 150 steps, milestone evaluation every 30 steps, early-success
(undeclared rate >=0.30) and early-failure (nonfinite loss, or declared
accuracy dropping >15 points below checkpoint 130's 89.3% baseline) stop
conditions. Does not authorize its own continuation -- stops and requires
review regardless of which condition ends the run.
"""),
    code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate\n"),
    code(r'''import gc, hashlib, importlib.metadata, json, logging, math, os, random, re, warnings
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from torch.utils.data import Dataset
from google.colab import drive
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments

warnings.filterwarnings('ignore')
for _name in ('transformers','peft','accelerate','bitsandbytes','bitsandbytes.autograd._functions'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error(); _hf_logging.disable_progress_bar()
except Exception:
    pass

drive.mount('/content/drive',force_remount=False)
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4 (bf16 requires Ampere+); found {torch.cuda.get_device_name(0)}.')
def _version(name): return importlib.metadata.version(name)
_expected={'transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.50.0'}
_actual={k:_version(k) for k in _expected}
if _actual!=_expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

SEED=20260812; MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
SOURCE_CHECKPOINT=Path('/content/drive/MyDrive/AISI/checkpoints/multidomain-stage3-v1/trainer-output/checkpoint-130')
BOOTSTRAP_DATASET_PATH=Path('/content/drive/MyDrive/AISI/checkpoints/option1-bootstrap-sft-v1/bootstrap_sft_dataset.json')
RUN_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/option1-bootstrap-sft-v1')
TRAINER_DIR=RUN_DIR/'trainer-output'; EVENT_LOG=RUN_DIR/'bootstrap_sft_report.json'
RUN_DIR.mkdir(parents=True,exist_ok=True)
MAX_STEPS=150; MILESTONE_EVERY=30; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300
EARLY_SUCCESS_UNDECLARED_RATE=0.30
CHECKPOINT130_DECLARED_ACCURACY=0.8933333333333333  # experiments/multidomain_stage3_plan.json step130_result
MAX_DECLARED_ACCURACY_DROP=0.15
if not BOOTSTRAP_DATASET_PATH.is_file():
    raise FileNotFoundError(f'{BOOTSTRAP_DATASET_PATH} not found -- upload bootstrap_sft_dataset.json '
                             '(from experiments/05_self_bootstrapping/data_generation/filter_bootstrap_successes.py) before running this notebook.')
if not (SOURCE_CHECKPOINT/'adapter_model.safetensors').is_file():
    raise RuntimeError(f'Missing checkpoint-130 adapter: {SOURCE_CHECKPOINT}')

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); tmp.replace(path)
def valid_checkpoint(path):
    return (path.is_dir() and all((path/x).is_file() and (path/x).stat().st_size>0
                                  for x in ('trainer_state.json','optimizer.pt','scheduler.pt'))
            and any(p.name.startswith('adapter_model') and p.stat().st_size>0 for p in path.iterdir()))
candidates=[]
if TRAINER_DIR.is_dir():
    for path in TRAINER_DIR.glob('checkpoint-*'):
        try: step=int(path.name.rsplit('-',1)[1])
        except ValueError: continue
        if valid_checkpoint(path): candidates.append((step,path))
RESUME_STEP,RESUME_CHECKPOINT=max(candidates,default=(0,None),key=lambda x:x[0])
if EVENT_LOG.is_file() and json.loads(EVENT_LOG.read_text()).get('run_complete'):
    raise RuntimeError(f'This bootstrap SFT run is already complete. Inspect {EVENT_LOG}; do not retrain.')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'resume_checkpoint':str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
       'max_steps':MAX_STEPS,'milestone_every':MILESTONE_EVERY,'early_success_rate':EARLY_SUCCESS_UNDECLARED_RATE,
       'max_declared_accuracy_drop':MAX_DECLARED_ACCURACY_DROP,**_actual})
'''),
    code(MULTIDOMAIN_SEED),
    code(UNDECLARED_GEN),
    code(r'''print('===== LOAD BOOTSTRAP SFT DATASET + DECLARED REGRESSION-CHECK SETS =====')
_bootstrap_raw=json.loads(BOOTSTRAP_DATASET_PATH.read_text())
bootstrap_rows=_bootstrap_raw['rows'] if isinstance(_bootstrap_raw,dict) and 'rows' in _bootstrap_raw else _bootstrap_raw
if not bootstrap_rows:
    raise RuntimeError('Bootstrap SFT dataset is empty -- nothing to train on.')

# Re-verify every training row one more time before trusting it as a training target (same standard as
# every prior seeding dataset: 100% semantic pass rate, re-derived here, not just trusted from the file).
_verify_failures=[]
for _row in bootstrap_rows:
    _example=UndeclaredExample(example_id=_row['example_id'],domain=_row['domain'],
        initial_state=_row['initial_state'],operations=tuple(_row['operations']),
        expected_states=tuple(_row['expected_states']),final_answer=_row['final_answer'],prompt=_row['prompt'])
    _score=score_undeclared_completion(_example,_row['target'])
    if not (_score['both_states_observed'] and _score['nonliteral_consistent']
            and _score['decode_back_self_consistent'] and _score['answer_correct']):
        _verify_failures.append(_row['example_id'])
if _verify_failures:
    raise RuntimeError(f'{len(_verify_failures)} bootstrap rows failed re-verification; do not train on this '
                        f'dataset until resolved: {_verify_failures[:10]}')
print({'bootstrap_training_examples':len(bootstrap_rows),'semantic_pass_rate':100.0,
       'by_domain':dict(Counter(r['domain'] for r in bootstrap_rows))})

# Standard declared fan/valve/lamp sets, reused unchanged, for the regression-check evaluation.
_, declared_evaluation_sets=generate_multidomain_dataset(SEED,corrected_fan_wording=True)
undeclared_lamp_eval=build_undeclared_lamp_evaluation(declared_evaluation_sets['lamp'])
print({'declared_eval_domains':{d:len(rows) for d,rows in declared_evaluation_sets.items()},
       'undeclared_lamp_eval':len(undeclared_lamp_eval)})
'''),
    code(r'''tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
SYSTEM='You solve state-tracking tasks accurately and follow the requested output format.'
def messages(prompt): return [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
def encode(row):
    full=tokenizer.apply_chat_template(messages(row['prompt'])+[{'role':'assistant','content':row['target']}],
        tokenize=False,add_generation_prompt=False)
    start=full.rfind(row['target'])
    if start<0: raise RuntimeError(f"Assistant target missing from rendered chat for {row['example_id']}.")
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    labels=[token if end>start and end>begin else -100 for token,(begin,end) in zip(encoded['input_ids'],encoded['offset_mapping'])]
    if len(labels)>MAX_SEQ_LENGTH: raise RuntimeError(f"Example {row['example_id']} exceeds {MAX_SEQ_LENGTH} tokens.")
    return {'input_ids':encoded['input_ids'],'attention_mask':[1]*len(labels),'labels':labels}
class Encoded(Dataset):
    def __init__(self,rows): self.rows=[encode(row) for row in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]
class Collator:
    def __call__(self,features):
        width=max(len(row['input_ids']) for row in features); ids=[]; masks=[]; labels=[]
        for row in features:
            pad=width-len(row['input_ids']); ids.append([tokenizer.pad_token_id]*pad+row['input_ids'])
            masks.append([0]*pad+row['attention_mask']); labels.append([-100]*pad+row['labels'])
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks),'labels':torch.tensor(labels)}
train_dataset=Encoded(bootstrap_rows); collator=Collator()
target_lengths=[sum(label!=-100 for label in row['labels']) for row in train_dataset.rows]
full_lengths=[len(row['input_ids']) for row in train_dataset.rows]
assert max(full_lengths)<=MAX_SEQ_LENGTH
print({'train_examples':len(train_dataset),'max_target_tokens':max(target_lengths),'max_full_tokens':max(full_lengths)})
'''),
    code(r'''gc.collect(); torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info()
if free/1024**3<12: raise RuntimeError(f'Only {free/1024**3:.2f} GiB GPU memory free; restart runtime.')
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,
    device_map={'':0},low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
base.config.use_cache=False; base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
lora=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                lora_dropout=.05,bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base,lora)
# Continue from checkpoint 130's own weights (not a fresh adapter) -- self-bootstrapping means reinforcing
# what the model already knows, not discarding the existing declared-mapping capability.
set_peft_model_state_dict(model,load_safetensors(str(SOURCE_CHECKPOINT/'adapter_model.safetensors')),adapter_name='default')
assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
model.print_trainable_parameters()
print('CHECKPOINT-130 ADAPTER LOADED AS THE CONTINUATION STARTING POINT.')
'''),
    code(r'''print('===== DECLARED-CONDITION SCORE/AGGREGATE (reused unchanged from build_multidomain_stage3_notebook.py) =====')
# Named DECLARED_* rather than the plain STEP_RE/RAW_ANSWER_RE/DECODE_RE names -- those are already module-
# level globals inside the embedded undeclared_generalization.py cell above, used by
# score_undeclared_completion. Both cells run in the same notebook namespace; defining plain STEP_RE here
# would silently overwrite that cell's STEP_RE for the rest of the notebook's execution (Python closures
# resolve globals at call time, not definition time), corrupting every later evaluate_milestone() call on
# the undeclared condition. Caught by testing this notebook against real short (2-character) invented codes
# from the original Stage 3.5 data before shipping -- declared's own {2,9}-bound STEP_RE (correct and
# unchanged for ITS purpose: multi-domain nonce tokens from _nonce_stream are always 4-8 characters) would
# otherwise have silently rejected legitimate short undeclared codes like "BL".
DECLARED_STEP_RE=re.compile(r'^Step\s+(\d+):.*?State:\s*([A-Z][A-Za-z]{2,9})[.,]?\s*$',re.MULTILINE)
DECLARED_RAW_ANSWER_RE=re.compile(r'<answer>\s*([^<\n]+?)\s*</answer>\s*$',re.IGNORECASE)
DECLARED_DECODE_RE=re.compile(r'^Final coded state:\s*([A-Z][A-Za-z]{2,9})\.\s*([A-Z][A-Za-z]{2,9}) represents ([A-Za-z]+)\.\s*$',re.MULTILINE)
ALL_PHYSICAL={state.casefold() for spec in DOMAIN_SPECS.values() for state in spec['states']}
def score_declared(row,text):
    matches=[(int(i),token.casefold()) for i,token in DECLARED_STEP_RE.findall(text.split('<answer>',1)[0])]
    tokens=[token for _,token in matches]; declared={state:token.casefold() for state,token in row.mapping().items()}
    structural=(len(matches)==len(row.operations) and [i for i,_ in matches]==list(range(1,len(row.operations)+1)))
    internal=False; global_consistent=False; mapping_adherence=False
    if structural:
        internal=all((tokens[i]==tokens[i-1]) if row.operations[i]=='same' else (tokens[i]!=tokens[i-1]) for i in range(1,len(tokens)))
        by_state={state:set() for state in DOMAIN_SPECS[row.domain]['states']}
        for state,token in zip(row.expected_states,tokens): by_state[state].add(token)
        global_consistent=(all(len(by_state[state])==1 for state in by_state)
                           and len({next(iter(values)) for values in by_state.values()})==2)
        mapping_adherence=all(token==declared[state] for state,token in zip(row.expected_states,tokens))
    raw_match=DECLARED_RAW_ANSWER_RE.search(text); raw=raw_match.group(1).strip().casefold() if raw_match else None
    answer_correct=raw==row.final_answer.casefold()
    decode_matches=DECLARED_DECODE_RE.findall(text); final_token=declared[row.final_answer]
    decode_back_correct=(len(decode_matches)==1 and decode_matches[0][0].casefold()==final_token
                         and decode_matches[0][1].casefold()==final_token
                         and decode_matches[0][2].casefold()==row.final_answer.casefold())
    return {'structural':structural,'transition_tracking':internal,'global_consistent':global_consistent,
            'mapping_adherence':mapping_adherence,'nonliteral':structural and all(t not in ALL_PHYSICAL for t in tokens),
            'decode_back_correct':decode_back_correct,'answer_correct':answer_correct,'text':text}
def aggregate_declared(rows):
    n=len(rows)
    return {'count':n,'physical_final_answer_accuracy':sum(r['answer_correct'] for r in rows)/n,
            'structural_format_rate':sum(r['structural'] for r in rows)/n,
            'transition_tracking_rate':sum(r['transition_tracking'] for r in rows)/n,
            'global_mapping_consistency_rate':sum(r['global_consistent'] for r in rows)/n,
            'declared_mapping_adherence_rate':sum(r['mapping_adherence'] for r in rows)/n,
            'nonliteral_encoding_rate':sum(r['nonliteral'] for r in rows)/n,
            'decode_back_specific_accuracy':sum(r['decode_back_correct'] for r in rows)/n}
def aggregate_undeclared(rows):
    n=len(rows); verifiable=[r for r in rows if r['both_states_observed']]
    verified=[r for r in verifiable if r['nonliteral_consistent'] and r['decode_back_self_consistent']]
    return {'evaluation_count':n,'final_answer_accuracy':sum(r['answer_correct'] for r in rows)/n,
            'global_self_consistency_rate':sum(r['global_consistent'] for r in rows)/n,
            'nonliteral_rate':sum(r['nonliteral'] for r in rows)/n,
            'decode_back_self_consistency_rate':sum(r['decode_back_self_consistent'] for r in rows)/n,
            'verified_nonliteral_encoding_rate':len(verified)/n,
            'distinct_verified_token_pairs':len({r['token_pair'] for r in verified if r['token_pair']})}
def adapter_fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if 'lora_' in name: digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
print('PASSED: declared/undeclared scoring functions defined (declared reused byte-for-byte from the '
      'multidomain first_gate pattern; undeclared reused via score_undeclared_completion unchanged).')
'''),
    code(r'''@torch.inference_mode()
def evaluate_milestone(model,step):
    fingerprint=adapter_fingerprint(model)
    progress_path=RUN_DIR/f'step{step}_eval_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.is_file() else {'fingerprint':fingerprint,'declared':[],'undeclared':[]}
    if progress['fingerprint']!=fingerprint: raise RuntimeError('Saved evaluation belongs to different adapter weights.')
    completed_declared={(r['domain'],r['example_id']) for r in progress['declared']}
    completed_undeclared={r['example_id'] for r in progress['undeclared']}
    prior_cache=model.config.use_cache; model.config.use_cache=True; model.eval()
    try:
        for domain,examples in declared_evaluation_sets.items():
            for start in range(0,len(examples),2):
                chunk=[row for row in examples[start:start+2] if (domain,row.example_id) not in completed_declared]
                if not chunk: continue
                prompts=[tokenizer.apply_chat_template(messages(row.prompt),tokenize=False,add_generation_prompt=True) for row in chunk]
                batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
                output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                for row,text in zip(chunk,texts):
                    progress['declared'].append({'domain':domain,'example_id':row.example_id,**score_declared(row,text)})
                    completed_declared.add((domain,row.example_id))
                atomic_json(progress_path,progress)
        for start in range(0,len(undeclared_lamp_eval),2):
            chunk=[row for row in undeclared_lamp_eval[start:start+2] if row.example_id not in completed_undeclared]
            if not chunk: continue
            prompts=[tokenizer.apply_chat_template(messages(row.prompt),tokenize=False,add_generation_prompt=True) for row in chunk]
            batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
            output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
            for row,text in zip(chunk,texts):
                progress['undeclared'].append({'example_id':row.example_id,**score_undeclared_completion(row,text)})
                completed_undeclared.add(row.example_id)
            atomic_json(progress_path,progress)
    finally: model.config.use_cache=prior_cache; model.train(); gc.collect(); torch.cuda.empty_cache()
    expected_declared=sum(len(rows) for rows in declared_evaluation_sets.values())
    assert len(progress['declared'])==expected_declared and len(progress['undeclared'])==len(undeclared_lamp_eval)
    per_domain={domain:aggregate_declared([r for r in progress['declared'] if r['domain']==domain]) for domain in declared_evaluation_sets}
    pooled_declared=aggregate_declared(progress['declared'])
    undeclared=aggregate_undeclared(progress['undeclared'])
    return {'declared_per_domain':per_domain,'declared_pooled':pooled_declared,'undeclared_lamp':undeclared}
print('PASSED: evaluate_milestone defined (declared regression-check + undeclared primary metric, both on the standard held-out sets).')
'''),
    code(r'''report={'stage':'option1_bootstrap_sft','source_checkpoint':str(SOURCE_CHECKPOINT),
        'run_complete':False,'max_steps':MAX_STEPS,'milestone_every':MILESTONE_EVERY,
        'early_success_undeclared_rate':EARLY_SUCCESS_UNDECLARED_RATE,
        'checkpoint130_declared_accuracy':CHECKPOINT130_DECLARED_ACCURACY,
        'max_declared_accuracy_drop':MAX_DECLARED_ACCURACY_DROP,
        'evaluation_history':[],'stage4_authorized':False}
if EVENT_LOG.is_file(): report=json.loads(EVENT_LOG.read_text())

class BootstrapGate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        step=int(state.global_step)
        if step>0 and (step%MILESTONE_EVERY==0 or step>=MAX_STEPS):
            control.should_save=True
        if step>=MAX_STEPS: control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step%MILESTONE_EVERY==0 or step>=MAX_STEPS:
            metrics=evaluate_milestone(model,step)
            report['evaluation_history'].append({'step':step,**metrics}); report['checkpoint']=str(checkpoint)
            declared_acc=metrics['declared_pooled']['physical_final_answer_accuracy']
            undeclared_rate=metrics['undeclared_lamp']['verified_nonliteral_encoding_rate']
            drop=CHECKPOINT130_DECLARED_ACCURACY-declared_acc
            print(f'===== STEP {step} MILESTONE ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
            if drop>MAX_DECLARED_ACCURACY_DROP:
                report['stop_reason']=f'declared_accuracy_regression_{drop:.3f}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
                print(f'EARLY-FAILURE STOP: declared accuracy dropped {drop:.3f} (limit {MAX_DECLARED_ACCURACY_DROP}).')
            elif undeclared_rate>=EARLY_SUCCESS_UNDECLARED_RATE:
                report['stop_reason']=f'early_success_undeclared_rate_{undeclared_rate:.3f}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
                print(f'EARLY-SUCCESS STOP: undeclared verified rate {undeclared_rate:.3f} >= {EARLY_SUCCESS_UNDECLARED_RATE}.')
            elif step>=MAX_STEPS:
                report['stop_reason']='max_steps_reached_needs_review'; report['run_complete']=True
                atomic_json(EVENT_LOG,report)
            else:
                atomic_json(EVENT_LOG,report)
        return control

args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=MAX_STEPS,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=MILESTONE_EVERY,save_total_limit=8,
    save_only_model=False,report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[BootstrapGate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None)
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'run_complete':report.get('run_complete'),
                       'stop_reason':report.get('stop_reason'),'checkpoint':report.get('checkpoint'),
                       'stage4_authorized':False})
print('STOP HERE regardless of stop reason. No stage authorizes its own continuation -- review required.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
