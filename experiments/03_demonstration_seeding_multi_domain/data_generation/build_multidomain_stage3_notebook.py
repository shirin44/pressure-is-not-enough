from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
OUTPUT = ROOT / "notebooks/multidomain_stage3_first_gate.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def main() -> None:
    source = (ROOT / "src/data/multidomain_seed.py").read_text().replace(
        "from __future__ import annotations\n\n", "", 1)
    cells = [
        md("""# Multi-domain Stage 3 — first checkpoint gate only

Fresh LoRA training on two exactly interleaved domains (fan and valve), with a
third domain (lamp) held out. The scheduler has a 150-step horizon, but this
notebook stops after saving and evaluating checkpoint 20. Every target contains
a full decode-back line. No Stage 4 composition occurs.
"""),
        code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate\n"),
        code("""import gc, hashlib, json, logging, math, os, random, re
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
from torch.utils.data import Dataset
from google.colab import drive
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments

drive.mount('/content/drive',force_remount=False)
if not torch.cuda.is_available(): raise RuntimeError('A Colab GPU is required.')
SEED=20260812; MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RUN_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/multidomain-stage3-v1')
TRAINER_DIR=RUN_DIR/'trainer-output'; EVENT_LOG=RUN_DIR/'first_gate_report.json'
EVAL_PROGRESS=RUN_DIR/'step20_eval_progress.json'
MAX_STEPS=150; FIRST_GATE_STEP=20; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300
RUN_DIR.mkdir(parents=True,exist_ok=True)

def atomic_json(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(payload,indent=2,sort_keys=True)); temp.replace(path)
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
if EVENT_LOG.is_file() and json.loads(EVENT_LOG.read_text()).get('first_gate_complete'):
    raise RuntimeError(f'First gate is already complete. Inspect {EVENT_LOG}; do not retrain.')
if RESUME_STEP>FIRST_GATE_STEP:
    raise RuntimeError(f'Unexpected checkpoint beyond authorized first gate: {RESUME_CHECKPOINT}')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'resume_checkpoint':str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
       'authorized_stop':FIRST_GATE_STEP,'scheduler_horizon':MAX_STEPS})
"""),
        code(source),
        code("""training_examples,evaluation_sets=generate_multidomain_dataset(SEED)
dataset_audit=audit_dataset(training_examples,evaluation_sets)
assert dataset_audit['accepted'] and dataset_audit['semantic_pass_rate']==100.0
assert dataset_audit['decode_back_training_coverage']==1.0
assert all(training_examples[i].domain!=training_examples[i+1].domain
           for i in range(len(training_examples)-1))
print('MULTI-DOMAIN DATA ACCEPTANCE GATE:',json.dumps(dataset_audit,indent=2,sort_keys=True))
print('CONFIRMED: every one of 800 training targets has a full decode-back line.')
"""),
        code("""tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
SYSTEM='You solve binary-state tracking tasks accurately and follow the requested output format.'
def messages(row): return [{'role':'system','content':SYSTEM},{'role':'user','content':row.prompt}]
def encode(row):
    full=tokenizer.apply_chat_template(messages(row)+[{'role':'assistant','content':row.demonstration}],tokenize=False,add_generation_prompt=False)
    start=full.rfind(row.demonstration)
    if start<0: raise RuntimeError('Assistant target missing from rendered chat.')
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    labels=[token if end>start and end>begin else -100 for token,(begin,end) in zip(encoded['input_ids'],encoded['offset_mapping'])]
    if len(labels)>MAX_SEQ_LENGTH: raise RuntimeError(f'Example exceeds {MAX_SEQ_LENGTH} tokens.')
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
train_dataset=Encoded(training_examples); collator=Collator()
target_lengths=[sum(label!=-100 for label in row['labels']) for row in train_dataset.rows]
full_lengths=[len(row['input_ids']) for row in train_dataset.rows]
assert max(target_lengths)<=math.floor(MAX_NEW_TOKENS*.8), (max(target_lengths),MAX_NEW_TOKENS)
assert max(full_lengths)<=MAX_SEQ_LENGTH
print({'train_examples':len(train_dataset),'max_target_tokens':max(target_lengths),
       'max_full_tokens':max(full_lengths),'decode_back_coverage':dataset_audit['decode_back_training_coverage']})
"""),
        code("""gc.collect(); torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info()
if free/1024**3<12: raise RuntimeError(f'Only {free/1024**3:.2f} GiB GPU memory free; restart runtime.')
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
base.config.use_cache=False; base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],lora_dropout=.05,bias='none',task_type='CAUSAL_LM'))
assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
model.print_trainable_parameters(); print('FRESH MULTI-DOMAIN ADAPTER VERIFIED.')
"""),
        code("""STEP_RE=re.compile(r'^Step\\s+(\\d+):.*?State:\\s*([A-Z][A-Za-z]{2,9})[.,]?\\s*$',re.MULTILINE)
RAW_ANSWER_RE=re.compile(r'<answer>\\s*([^<\\n]+?)\\s*</answer>\\s*$',re.IGNORECASE)
DECODE_RE=re.compile(r'^Final coded state:\\s*([A-Z][A-Za-z]{2,9})\\.\\s*([A-Z][A-Za-z]{2,9}) represents ([A-Za-z]+)\\.\\s*$',re.MULTILINE)
ALL_PHYSICAL={state.casefold() for spec in DOMAIN_SPECS.values() for state in spec['states']}
def score(row,text):
    matches=[(int(i),token.casefold()) for i,token in STEP_RE.findall(text.split('<answer>',1)[0])]
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
    raw_match=RAW_ANSWER_RE.search(text); raw=raw_match.group(1).strip().casefold() if raw_match else None
    answer_correct=raw==row.final_answer.casefold()
    declared_codes=set(declared.values())
    answer_is_code=bool(not answer_correct and raw and raw not in ALL_PHYSICAL and (raw in set(tokens) or raw in declared_codes))
    decode_matches=DECODE_RE.findall(text)
    final_token=declared[row.final_answer]
    decode_back_correct=(len(decode_matches)==1 and decode_matches[0][0].casefold()==final_token
                         and decode_matches[0][1].casefold()==final_token
                         and decode_matches[0][2].casefold()==row.final_answer.casefold())
    if answer_correct: failure=None
    elif answer_is_code: failure='code_word_instead_of_physical_state'
    elif not structural or not internal: failure='wrong_tracking'
    elif not mapping_adherence: failure='wrong_mapping'
    elif raw is None: failure='malformed_or_missing_answer'
    else: failure='other_wrong_answer'
    return {'structural':structural,'transition_tracking':internal,'global_consistent':global_consistent,
            'mapping_adherence':mapping_adherence,'nonliteral':structural and all(t not in ALL_PHYSICAL for t in tokens),
            'decode_back_correct':decode_back_correct,'answer_correct':answer_correct,
            'answer_is_code_word':answer_is_code,'answer_failure_type':failure,
            'declared_pair':sorted(declared_codes),'text':text}

def adapter_fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if 'lora_' in name: digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def aggregate(rows):
    n=len(rows); failures=[row for row in rows if not row['answer_correct']]
    pairs=Counter(tuple(row['declared_pair']) for row in rows if row['mapping_adherence'])
    return {'count':n,'physical_final_answer_accuracy':sum(r['answer_correct'] for r in rows)/n,
            'structural_format_rate':sum(r['structural'] for r in rows)/n,
            'transition_tracking_rate':sum(r['transition_tracking'] for r in rows)/n,
            'global_mapping_consistency_rate':sum(r['global_consistent'] for r in rows)/n,
            'declared_mapping_adherence_rate':sum(r['mapping_adherence'] for r in rows)/n,
            'nonliteral_encoding_rate':sum(r['nonliteral'] for r in rows)/n,
            'decode_back_specific_accuracy':sum(r['decode_back_correct'] for r in rows)/n,
            'answer_is_code_word_rate':sum(r['answer_is_code_word'] for r in rows)/n,
            'answer_is_code_word_fraction_of_failures':(sum(r['answer_is_code_word'] for r in failures)/len(failures) if failures else 0),
            'answer_failure_category_counts':dict(Counter(r['answer_failure_type'] for r in failures)),
            'distinct_correctly_applied_token_pairs':len(pairs)}

@torch.inference_mode()
def evaluate_first_gate(model):
    fingerprint=adapter_fingerprint(model)
    progress=json.loads(EVAL_PROGRESS.read_text()) if EVAL_PROGRESS.is_file() else {'fingerprint':fingerprint,'rows':[]}
    if progress['fingerprint']!=fingerprint: raise RuntimeError('Saved evaluation belongs to different adapter weights.')
    completed={(r['domain'],r['example_id']) for r in progress['rows']}
    prior_cache=model.config.use_cache; model.config.use_cache=True; model.eval()
    try:
        for domain,examples in evaluation_sets.items():
            for start in range(0,len(examples),2):
                chunk=[row for row in examples[start:start+2] if (domain,row.example_id) not in completed]
                if not chunk: continue
                prompts=[tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True) for row in chunk]
                batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
                output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                for row,text in zip(chunk,texts):
                    progress['rows'].append({'domain':domain,'example_id':row.example_id,**score(row,text)})
                    completed.add((domain,row.example_id))
                if len(completed)%10==0: atomic_json(EVAL_PROGRESS,progress); print(f'SAVED STEP-20 EVAL: {len(completed)}/300')
        atomic_json(EVAL_PROGRESS,progress)
    finally: model.config.use_cache=prior_cache; model.train(); gc.collect(); torch.cuda.empty_cache()
    assert len(progress['rows'])==300
    per_domain={domain:aggregate([r for r in progress['rows'] if r['domain']==domain]) for domain in evaluation_sets}
    pooled=aggregate(progress['rows'])
    return {'per_domain':per_domain,'pooled':pooled,'samples':progress['rows'][:12]}
"""),
        code("""report={'stage':'multi_domain_stage3','first_gate_complete':False,'authorized_step':20,
        'full_scheduler_horizon':150,'dataset_audit':dataset_audit,'source_model':MODEL_NAME,
        'adapter_init':'fresh_lora','evaluation_history':[],'stage4_authorized':False}
if EVENT_LOG.is_file(): report=json.loads(EVENT_LOG.read_text())

class FirstGate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; atomic_json(EVENT_LOG,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=FIRST_GATE_STEP:
            control.should_save=True
            control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==FIRST_GATE_STEP and not report.get('first_gate_complete'):
            metrics=evaluate_first_gate(model); report['evaluation_history'].append({'step':step,**metrics})
            report['first_gate_complete']=True; report['stop_reason']='step20_first_gate_complete_needs_review'
            report['checkpoint']=str(checkpoint); atomic_json(EVENT_LOG,report)
            print('===== STEP 20 PER-DOMAIN AND POOLED REPORT ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
            control.should_training_stop=True
        return control

runtime_max_steps=(FIRST_GATE_STEP if RESUME_STEP==FIRST_GATE_STEP else MAX_STEPS)
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=runtime_max_steps,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[FirstGate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None)
if int(trainer.state.global_step)==FIRST_GATE_STEP and not report.get('first_gate_complete'):
    metrics=evaluate_first_gate(model); report['evaluation_history'].append({'step':20,**metrics})
    report.update({'first_gate_complete':True,'stop_reason':'step20_first_gate_complete_needs_review',
                   'checkpoint':str(TRAINER_DIR/'checkpoint-20')}); atomic_json(EVENT_LOG,report)
    print(json.dumps(metrics,indent=2,sort_keys=True))
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'first_gate_complete':report.get('first_gate_complete'),
                       'checkpoint':report.get('checkpoint'),'stage4_authorized':False})
print('STOP HERE. Do not continue beyond step 20 before review.')
"""),
    ]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == '__main__': main()
