"""Build the self-contained Colab notebook for Stage 3 latch SFT."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
OUTPUT = ROOT / "notebooks" / "latch_seed_stage3_sft.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    latch_source = (ROOT / "src" / "data" / "latch_seed.py").read_text()
    latch_source = latch_source.replace("from __future__ import annotations\n\n", "", 1)
    cells = [
        markdown(
            """# Stage 3 — explicit per-step transition diagnostic

This notebook trains a **fresh LoRA adapter on the untouched Qwen2.5-3B-Instruct base**.
It never loads Coin Flip checkpoint 500. It regenerates and verifies the redesigned
800/200 latch corpus, with a randomized mapping declared in every prompt and an
explicit previous-code citation at every step and a final code-to-physical-state decode-back target, performs
a maximum of 150 optimizer steps, and stops on the
pre-registered success, plateau, or safety gates. It does not merge adapters and
does not run the third-domain or Coin Flip stages.

Recommended runtime: Colab L4. Run top-to-bottom once on a fresh kernel.
"""
        ),
        code(
            """# Install only if the runtime does not already have the pinned environment.
%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 datasets accelerate
"""
        ),
        code(
            """import gc, hashlib, importlib.metadata, json, logging, math, os, random, re, shutil, time
from collections import Counter
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from google.colab import drive
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainerCallback, TrainingArguments, default_data_collator,
)

drive.mount('/content/drive', force_remount=False)
if not torch.cuda.is_available():
    raise RuntimeError('Stage 3 requires a Colab GPU runtime.')

SEED=20260809
MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RUN_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/latch-seed-stage3-decode-back-sft-v3')
TRAINER_DIR=RUN_DIR/'trainer-output'
DATA_DIR=RUN_DIR/'data'
FINAL_ADAPTER=RUN_DIR/'delta-seed-final'
EVENT_LOG=RUN_DIR/'stage3_sft_report.json'
MAX_STEPS=150
EVAL_EVERY=10
MAX_SEQ_LENGTH=1024
# The v5 explicit-transition target reaches 239 tokens. 300 preserves the
# registered 20% headroom (floor(300 * .80) == 240) without the cost of the
# earlier 384-token diagnostic ceiling.
MAX_NEW_TOKENS=300
GRAD_BREAKER=50.0
DATASET_VERSION='v5_mapping_anchor_explicit_per_step_transition'

RUN_DIR.mkdir(parents=True,exist_ok=True)

logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes.autograd._functions').disabled=True

def valid_trainer_checkpoint(path):
    required=('trainer_state.json','optimizer.pt','scheduler.pt')
    return (path.is_dir() and all((path/name).is_file() and (path/name).stat().st_size>0
                                  for name in required)
            and any(x.name.startswith('adapter_model') and x.stat().st_size>0
                    for x in path.rglob('adapter_model*')))

resume_candidates=[]
if TRAINER_DIR.is_dir():
    for candidate in TRAINER_DIR.glob('checkpoint-*'):
        try: step=int(candidate.name.rsplit('-',1)[1])
        except ValueError: continue
        if valid_trainer_checkpoint(candidate): resume_candidates.append((step,candidate))
RESUME_CHECKPOINT=max(resume_candidates,default=(0,None),key=lambda x:x[0])[1]
prior_report=json.loads(EVENT_LOG.read_text()) if EVENT_LOG.is_file() and EVENT_LOG.stat().st_size else None
if prior_report and prior_report.get('accepted'):
    raise RuntimeError(f'Stage 3 is already accepted; inspect {EVENT_LOG}, do not retrain.')
if prior_report and prior_report.get('training_started') and RESUME_CHECKPOINT is None:
    raise RuntimeError('Interrupted training evidence exists but no valid checkpoint is available; refusing silent restart.')

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'run_dir':str(RUN_DIR),'seed':SEED,
       'resume_checkpoint':str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None})
"""
        ),
        markdown("## Stage 1 generator and hard acceptance gate\n"),
        code(latch_source),
        code(
            """train_examples,heldout_examples=generate_latch_seed_dataset(SEED)
dataset_audit=audit_latch_seed_dataset(train_examples,heldout_examples,seed=SEED)
assert dataset_audit['accepted'] is True
assert dataset_audit['semantic_verification_pass_rate']==100.0
assert dataset_audit['prompt_mapping_verification_pass_rate']==100.0
assert dataset_audit['decode_back_verification_pass_rate']==100.0
assert dataset_audit['mapping_anchor_verification_pass_rate']==100.0
assert dataset_audit['explicit_transition_verification_pass_rate']==100.0
assert dataset_audit['unique_token_pair_percentage']==100.0
write_latch_seed_dataset(DATA_DIR,SEED)
print('STAGE 1 DATA REVERIFIED:',json.dumps(dataset_audit,indent=2,sort_keys=True))
"""
        ),
        markdown("## Tokenization and completion-only labels\n"),
        code(
            """tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'

SYSTEM='You solve binary-state tracking tasks accurately and follow the requested output format.'

def prompt_messages(example):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':example.prompt}]

def encode_training_example(example):
    full_messages=prompt_messages(example)+[{'role':'assistant','content':example.demonstration}]
    # Do not assume that add_generation_prompt=True token IDs are an exact
    # prefix of a completed conversation. That is not guaranteed across
    # Transformers/chat-template versions. Render once, locate the exact
    # assistant content boundary, then mask by tokenizer character offsets.
    full_text=tokenizer.apply_chat_template(full_messages,tokenize=False,add_generation_prompt=False)
    assistant_start=full_text.rfind(example.demonstration)
    if assistant_start<0:
        raise RuntimeError('Rendered chat template does not contain the exact assistant demonstration.')
    encoded=tokenizer(full_text,add_special_tokens=False,return_offsets_mapping=True)
    full_ids=encoded['input_ids']; offsets=encoded['offset_mapping']
    if len(full_ids)>MAX_SEQ_LENGTH:
        raise RuntimeError(f'Training example exceeds {MAX_SEQ_LENGTH} tokens.')
    labels=[token_id if end>assistant_start and end>start else -100
            for token_id,(start,end) in zip(full_ids,offsets)]
    if not any(x!=-100 for x in labels): raise RuntimeError('No assistant labels remain.')
    return {'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels}

class EncodedDataset(Dataset):
    def __init__(self,examples): self.rows=[encode_training_example(x) for x in examples]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]

class CompletionCollator:
    def __call__(self,features):
        width=max(len(x['input_ids']) for x in features)
        ids=[]; masks=[]; labels=[]
        for row in features:
            pad=width-len(row['input_ids'])
            ids.append([tokenizer.pad_token_id]*pad+row['input_ids'])
            masks.append([0]*pad+row['attention_mask'])
            labels.append([-100]*pad+row['labels'])
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks),'labels':torch.tensor(labels)}

train_dataset=EncodedDataset(train_examples)
heldout_dataset=EncodedDataset(heldout_examples)
collator=CompletionCollator()
lengths=[len(x['input_ids']) for x in train_dataset.rows]
target_lengths=[sum(label!=-100 for label in row['labels']) for row in train_dataset.rows]
target_budget_limit=math.floor(MAX_NEW_TOKENS*.80)
if max(target_lengths)>target_budget_limit:
    raise RuntimeError(
        f'Longest target is {max(target_lengths)} tokens, exceeding the registered '
        f'80% generation-budget limit ({target_budget_limit}/{MAX_NEW_TOKENS}).'
    )

all_codes=[token for example in train_examples+heldout_examples
           for token in (example.token_for_locked,example.token_for_unlocked)]
code_piece_counts={token:len(tokenizer(token,add_special_tokens=False)['input_ids'])
                   for token in all_codes}
if max(code_piece_counts.values())>8:
    hardest=sorted(code_piece_counts.items(),key=lambda item:item[1],reverse=True)[:10]
    raise RuntimeError(f'Nonce code fragmentation exceeds 8 tokenizer pieces: {hardest}')
locked_mean=sum(code_piece_counts[x.token_for_locked] for x in train_examples)/len(train_examples)
unlocked_mean=sum(code_piece_counts[x.token_for_unlocked] for x in train_examples)/len(train_examples)
if abs(locked_mean-unlocked_mean)>.5:
    raise RuntimeError(f'Code-tokenization imbalance is too large: Locked={locked_mean}, Unlocked={unlocked_mean}')

def mapping_orientation(example):
    return ('locked_first' if example.token_for_locked<example.token_for_unlocked
            else 'unlocked_first')

screen_cells={key:[] for key in
              ((state,orientation) for state in STATES
               for orientation in ('locked_first','unlocked_first'))}
for example in heldout_examples:
    screen_cells[(example.final_answer,mapping_orientation(example))].append(example)
screen_examples=[]
for key,items in sorted(screen_cells.items()):
    items=sorted(items,key=lambda x:(len(x.operations),x.example_id))
    indices=np.linspace(0,len(items)-1,10,dtype=int)
    screen_examples.extend(items[index] for index in indices)
assert len(screen_examples)==40 and len({x.example_id for x in screen_examples})==40
first=train_dataset.rows[0]
first_target=tokenizer.decode([token for token,label in zip(first['input_ids'],first['labels']) if label!=-100])
if train_examples[0].demonstration.splitlines()[0] not in first_target:
    raise RuntimeError('Completion-only mask does not begin with the worked demonstration.')
if 'A door latch starts' in first_target:
    raise RuntimeError('Completion-only mask leaked user-prompt tokens into the loss target.')
print({'train':len(train_dataset),'heldout':len(heldout_dataset),
       'screening_prompts':len(screen_examples),'max_full_tokens':max(lengths),
       'max_target_tokens':max(target_lengths),'generation_budget':MAX_NEW_TOKENS,
       'max_nonce_tokenizer_pieces':max(code_piece_counts.values()),
       'locked_code_mean_pieces':locked_mean,'unlocked_code_mean_pieces':unlocked_mean})
print('COMPLETION-ONLY MASK VERIFIED. First target preview:',repr(first_target[:180]))
"""
        ),
        markdown("## Untouched base model + fresh seed adapter\n"),
        code(
            """gc.collect(); torch.cuda.empty_cache()
free_bytes,total_bytes=torch.cuda.mem_get_info()
free_gib=free_bytes/1024**3
MIN_FREE_GPU_GIB=12.0
if free_gib<MIN_FREE_GPU_GIB:
    raise RuntimeError(
        f'Only {free_gib:.2f} GiB GPU memory is free before model load; '
        f'this 4-bit run requires at least {MIN_FREE_GPU_GIB:.0f} GiB. Restart the runtime before continuing.'
    )
print({'gpu_free_before_load_gib':round(free_gib,2),'gpu_total_gib':round(total_bytes/1024**3,2)})

quant=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map={'':0},
    low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
base_model.config.use_cache=False
base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
lora_config=LoraConfig(
    r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05,bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base_model,lora_config)
model.print_trainable_parameters()
if hasattr(model,'set_adapter'): model.set_adapter('default')
trainable=[name for name,p in model.named_parameters() if p.requires_grad]
assert trainable and all('lora_' in name for name in trainable),trainable[:10]
print('CONFIRMED: no Coin Flip adapter loaded; only fresh LoRA tensors are trainable.')
free_after,_=torch.cuda.mem_get_info()
print({'gpu_allocated_gib':round(torch.cuda.memory_allocated()/1024**3,2),
       'gpu_reserved_gib':round(torch.cuda.memory_reserved()/1024**3,2),
       'gpu_free_after_load_gib':round(free_after/1024**3,2)})
"""
        ),
        markdown("## Generative validation metrics and stopping controller\n"),
        code(
            """STATE_RE=re.compile(r'^\\s*Step\\s+(\\d+)\\s*:\\s*Previous code:\\s*([A-Z][A-Za-z]{2,9})\\.\\s*The latch (?:stays the same|switches)\\.\\s*State:\\s*([A-Z][A-Za-z]{2,9})[.,]?\\s*$',re.MULTILINE)
ANSWER_RE=re.compile(r'<answer>\\s*(Locked|Unlocked)\\s*</answer>\\s*$',re.IGNORECASE)
RAW_ANSWER_RE=re.compile(r'<answer>\\s*([^<\\n]+?)\\s*</answer>\\s*$',re.IGNORECASE)
DECODE_RE=re.compile(r'^Final coded state:\\s*([A-Z][A-Za-z]{2,9})\\.\\s*([A-Z][A-Za-z]{2,9}) represents (Locked|Unlocked)\\.\\s*$',re.MULTILINE)
LITERALS={'locked','unlocked'}

def score_generated(example,text):
    matches=[(int(i),previous,token) for i,previous,token in STATE_RE.findall(text.split('<answer>',1)[0])]
    expected_states=simulate(example.initial_state,example.operations)
    structural=(len(matches)==len(example.operations) and [i for i,_,_ in matches]==list(range(1,len(example.operations)+1)))
    previous_tokens=[token.casefold() for _,token,_ in matches]
    tokens=[token.casefold() for _,_,token in matches]
    declared={'Locked':example.token_for_locked.casefold(),
              'Unlocked':example.token_for_unlocked.casefold()}
    internal=False; global_consistency=False; mapping_adherence=False; nonliteral=False
    previous_code_adherence=False
    first_transition_error_step=None; mapping_failure_subtype=None
    if structural:
        expected_previous_states=[example.initial_state,*expected_states[:-1]]
        previous_code_adherence=all(
            token==declared[state]
            for state,token in zip(expected_previous_states,previous_tokens))
        transition_checks=[
            ((tokens[i]==tokens[i-1]) if example.operations[i]=='same'
             else (tokens[i]!=tokens[i-1]))
            for i in range(1,len(tokens))]
        internal=all(transition_checks)
        if not internal:
            first_transition_error_step=next(
                index+2 for index,passed in enumerate(transition_checks) if not passed)
        by_state={state:set() for state in STATES}
        for state,token in zip(expected_states,tokens): by_state[state].add(token)
        covered=all(by_state[state] for state in STATES)
        global_consistency=(covered and all(len(by_state[state])==1 for state in STATES)
                            and by_state['Locked']!=by_state['Unlocked'])
        mapping_adherence=all(token==declared[state]
                              for state,token in zip(expected_states,tokens))
        if internal and not mapping_adherence:
            statuses=[]
            for state,token in zip(expected_states,tokens):
                opposite='Unlocked' if state=='Locked' else 'Locked'
                if token==declared[state]: statuses.append('correct')
                elif token==declared[opposite]: statuses.append('swapped')
                else: statuses.append('other')
            if all(status=='swapped' for status in statuses):
                mapping_failure_subtype='consistent_declared_mapping_swap'
            elif global_consistency:
                mapping_failure_subtype='consistent_nondeclared_mapping'
            elif statuses[0]=='correct':
                mapping_failure_subtype='starts_correct_then_drifts'
            elif statuses[0]=='swapped':
                mapping_failure_subtype='starts_swapped_then_drifts'
            else:
                mapping_failure_subtype='other_or_immediate_drift'
        nonliteral=all(token not in LITERALS for token in tokens)
    answer=ANSWER_RE.search(text)
    raw_answer_match=RAW_ANSWER_RE.search(text)
    raw_answer=raw_answer_match.group(1).strip() if raw_answer_match else None
    answer_valid=answer is not None
    answer_correct=bool(answer and answer.group(1).casefold()==example.final_answer.casefold())
    decode_matches=list(DECODE_RE.finditer(text))
    decode=decode_matches[0] if len(decode_matches)==1 else None
    expected_final_token=(example.token_for_locked if example.final_answer=='Locked'
                          else example.token_for_unlocked)
    decode_back_correct=bool(
        decode and decode.group(1)==expected_final_token
        and decode.group(2)==expected_final_token
        and decode.group(3)==example.final_answer
        and raw_answer_match and not text[decode.end():raw_answer_match.start()].strip()
    )
    if answer_correct:
        answer_error_type=None
    elif (decode_back_correct and raw_answer
          and raw_answer.casefold()==expected_final_token.casefold()):
        answer_error_type='correct_decode_then_code_copy'
    elif not structural or not internal or not previous_code_adherence:
        answer_error_type='wrong_tracking'
    elif not mapping_adherence:
        answer_error_type='wrong_mapping'
    elif raw_answer is None or not answer_valid:
        answer_error_type='malformed_answer'
    else:
        answer_error_type='wrong_decoded_state_or_other'
    return {'answer_correct':answer_correct,'answer_valid':answer_valid,'structural':structural,
            'internal_correct':internal,'global_consistent':global_consistency,
            'mapping_adherence':mapping_adherence,'decode_back_correct':decode_back_correct,
            'previous_code_adherence':previous_code_adherence,
            'nonliteral':nonliteral,'raw_answer':raw_answer,
            'answer_error_type':answer_error_type,
            'sequence_length':len(example.operations),
            'first_transition_error_step':first_transition_error_step,
            'mapping_failure_subtype':mapping_failure_subtype,'text':text}

_taxonomy_example=heldout_examples[0]
_perfect_score=score_generated(_taxonomy_example,_taxonomy_example.demonstration)
assert (_perfect_score['answer_correct'] and _perfect_score['decode_back_correct']
        and _perfect_score['previous_code_adherence'])
_expected_code=(_taxonomy_example.token_for_locked
                if _taxonomy_example.final_answer=='Locked'
                else _taxonomy_example.token_for_unlocked)
_copy_text=_taxonomy_example.demonstration.replace(
    f'<answer>{_taxonomy_example.final_answer}</answer>',f'<answer>{_expected_code}</answer>')
_copy_score=score_generated(_taxonomy_example,_copy_text)
assert (_copy_score['decode_back_correct'] and not _copy_score['answer_correct']
        and _copy_score['answer_error_type']=='correct_decode_then_code_copy')
_swap_text=(_taxonomy_example.demonstration
            .replace(_taxonomy_example.token_for_locked,'ZZZTEMPZZZ')
            .replace(_taxonomy_example.token_for_unlocked,_taxonomy_example.token_for_locked)
            .replace('ZZZTEMPZZZ',_taxonomy_example.token_for_unlocked))
_swap_score=score_generated(_taxonomy_example,_swap_text)
assert (_swap_score['internal_correct'] and not _swap_score['mapping_adherence']
        and _swap_score['mapping_failure_subtype']=='consistent_declared_mapping_swap')
print('ANSWER-ERROR AND TRACE-DIAGNOSTIC UNIT TESTS PASSED.')

@torch.inference_mode()
def generative_validation(model,examples,batch_size=2,label='evaluation'):
    was_training=model.training; model.eval(); prior_cache=model.config.use_cache; model.config.use_cache=True
    rows=[]
    try:
        for start in range(0,len(examples),batch_size):
            chunk=examples[start:start+batch_size]
            prompts=[tokenizer.apply_chat_template(prompt_messages(x),tokenize=False,add_generation_prompt=True) for x in chunk]
            batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
            output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            prompt_width=batch['input_ids'].shape[1]
            completion_ids=output[:,prompt_width:]
            texts=tokenizer.batch_decode(completion_ids,skip_special_tokens=True)
            for example,text,ids in zip(chunk,texts,completion_ids):
                eos_positions=(ids==tokenizer.eos_token_id).nonzero(as_tuple=False)
                terminated=bool(eos_positions.numel())
                generated_length=(int(eos_positions[0].item())+1 if terminated else int(ids.numel()))
                row=score_generated(example,text)
                row.update({'terminated':terminated,'generated_length':generated_length,
                            'truncated':not terminated and int(ids.numel())>=MAX_NEW_TOKENS})
                rows.append(row)
            if start%20==0:
                print(f'{label}: {min(start+len(chunk),len(examples))}/{len(examples)} generated')
    finally:
        model.config.use_cache=prior_cache
        if was_training: model.train()
        gc.collect(); torch.cuda.empty_cache()
    count=len(rows)
    metrics={key:sum(row[key] for row in rows)/count for key in
             ('answer_correct','answer_valid','structural','internal_correct','global_consistent',
              'mapping_adherence','decode_back_correct','nonliteral','previous_code_adherence')}
    metrics['truncation_rate']=sum(row['truncated'] for row in rows)/count
    metrics['termination_rate']=sum(row['terminated'] for row in rows)/count
    metrics['mean_generated_length']=sum(row['generated_length'] for row in rows)/count
    metrics['max_generated_length']=max(row['generated_length'] for row in rows)
    answer_failures=[row for row in rows if not row['answer_correct']]
    error_counts=Counter(row['answer_error_type'] for row in answer_failures)
    metrics['answer_error_counts']=dict(sorted(error_counts.items()))
    metrics['answer_error_fractions']={
        key:value/len(answer_failures) for key,value in sorted(error_counts.items())
    } if answer_failures else {}
    metrics['answer_failure_count']=len(answer_failures)

    tracking_by_length={}
    for length in sorted({row['sequence_length'] for row in rows}):
        length_rows=[row for row in rows if row['sequence_length']==length]
        tracking_errors=[row for row in length_rows
                         if not row['structural'] or not row['internal_correct']]
        first_steps=Counter(row['first_transition_error_step'] for row in tracking_errors
                            if row['first_transition_error_step'] is not None)
        tracking_by_length[str(length)]={
            'n':len(length_rows),'wrong_tracking':len(tracking_errors),
            'wrong_tracking_rate':len(tracking_errors)/len(length_rows),
            'first_transition_error_step_counts':{
                str(key):value for key,value in sorted(first_steps.items())}
        }
    length_values=np.array([row['sequence_length'] for row in rows],dtype=float)
    tracking_values=np.array([
        float(not row['structural'] or not row['internal_correct']) for row in rows])
    tracking_length_correlation=(float(np.corrcoef(length_values,tracking_values)[0,1])
                                 if np.std(tracking_values)>0 else 0.0)
    mapping_failure_rows=[row for row in rows
                          if row['structural'] and row['internal_correct']
                          and not row['mapping_adherence']]
    mapping_subtypes=Counter(row['mapping_failure_subtype'] for row in mapping_failure_rows)
    metrics['trace_error_diagnostics']={
        'wrong_tracking_total':int(tracking_values.sum()),
        'wrong_tracking_by_sequence_length':tracking_by_length,
        'sequence_length_tracking_error_correlation':tracking_length_correlation,
        'wrong_mapping_total':len(mapping_failure_rows),
        'wrong_mapping_subtype_counts':dict(sorted(mapping_subtypes.items())),
        'wrong_mapping_subtype_fractions':{
            key:value/len(mapping_failure_rows)
            for key,value in sorted(mapping_subtypes.items())
        } if mapping_failure_rows else {},
    }
    metrics['samples']=[row['text'] for row in rows[:5]]
    return metrics

@torch.inference_mode()
def validation_loss(model,dataset,batch_size=2):
    was_training=model.training; model.eval(); losses=[]; weights=[]
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collator)
    try:
        for batch in loader:
            batch={k:v.to(model.device) for k,v in batch.items()}
            weight=int((batch['labels']!=-100).sum())
            loss=float(model(**batch).loss.detach().float().cpu())
            if not math.isfinite(loss): return float('nan')
            losses.append(loss*weight); weights.append(weight)
    finally:
        if was_training: model.train()
        gc.collect(); torch.cuda.empty_cache()
    return sum(losses)/sum(weights)

def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); tmp.replace(path)

report=prior_report if prior_report is not None else {
    'stage':'latch_seed_decode_back_sft','training_started':False,'evaluations':[],
    'stop_reason':None,'accepted':False,
}
previous_dataset_version=report.get('config',{}).get('dataset_version')
if previous_dataset_version and previous_dataset_version!=DATASET_VERSION:
    report.setdefault('superseded_evaluations',[]).extend(report.get('evaluations',[]))
    report['evaluations']=[]
    report.pop('baseline',None); report.pop('baseline_scope',None)
    report['migration_note']=(f'Resuming weights, optimizer, scheduler, and trainer state from '
                              f'{RESUME_CHECKPOINT} while replacing {previous_dataset_version} '
                              f'with {DATASET_VERSION}; prior prompt-version metrics are retained '
                              'only as superseded historical evidence.')
report['dataset_audit']=dataset_audit
report['config']={'max_steps':150,'lr':2e-5,'warmup_steps':10,'scheduler':'cosine',
                  'effective_batch_size':16,'lora_r':8,'lora_alpha':16,
                  'quantization':'4-bit NF4 double-quantization',
                  'dataset_version':DATASET_VERSION,
                  'screening_size':40,'full_evaluation_steps':[50,60,100,130,140,150],
                  'step140_tracking_gate':{
                      'reference_step':130,'reference_wrong_tracking_total':19,
                      'pass_max_wrong_tracking_total':15,
                      'minimum_structure':.95,'minimum_answer_correct':.875,
                      'minimum_previous_code_adherence':.95},
                  'target_budget_margin':.20,'max_nonce_tokenizer_pieces':8,
                  'truncation_stop':'>10% for two consecutive evaluations from step 30',
                  'answer_error_taxonomy':'v1_mutually_exclusive',
                  'step100_and_step130_trace_diagnostics':[
                      'tracking_error_rate_by_sequence_length',
                      'first_transition_error_step',
                      'consistent_mapping_swap_vs_mid_trace_drift'],
                  'checkpoint_every_steps':5,
                  'gradient_breaker':GRAD_BREAKER,'seed':SEED}
report.setdefault('evaluations',[])
report.setdefault('verified_checkpoints',[])
report['stop_reason']=None

BEHAVIOR_KEYS=('answer_correct','answer_valid','structural','internal_correct',
               'global_consistent','mapping_adherence','nonliteral','previous_code_adherence')
def behavioral_success(metrics,threshold=.95):
    return all(metrics.get(key,0)>=threshold for key in BEHAVIOR_KEYS)

class Stage3Controller(TrainerCallback):
    def __init__(self):
        prior=report.get('evaluations',[])
        self.eval_losses=[float(row['validation_loss']) for row in prior if 'validation_loss' in row]
        self.success_streak=0
        self.last_full_step=None
        for row in reversed([x for x in prior if x.get('scope')=='full_200']):
            success=behavioral_success(row)
            if not success: break
            self.success_streak+=1
            if self.success_streak==1: self.last_full_step=int(row['step'])
        self.truncation_streak=0
        self.hard_stop=None
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        loss=logs.get('loss'); grad=logs.get('grad_norm')
        if loss is not None and not math.isfinite(float(loss)): self.hard_stop='nonfinite_loss'
        if grad is not None and (not math.isfinite(float(grad)) or float(grad)>=GRAD_BREAKER):
            self.hard_stop=f'gradient_circuit_breaker:{grad}'
        if self.hard_stop:
            control.should_training_stop=True; report['stop_reason']=self.hard_stop; atomic_json(EVENT_LOG,report)
        return control
    def on_step_end(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step)
        if step==0 or step%EVAL_EVERY: return control
        loss=validation_loss(model,heldout_dataset)
        self.eval_losses.append(loss)
        relative_improvement=None
        if len(self.eval_losses)>=3:
            old,new=self.eval_losses[-3],self.eval_losses[-1]
            relative_improvement=(old-new)/max(abs(old),1e-12)

        screen=generative_validation(model,screen_examples,label=f'step {step} screen')
        near_success=behavioral_success(screen,.93)
        plateau=relative_improvement is not None and relative_improvement<.005
        full_due=(step in (50,60,100,130,140,150) or near_success or plateau)
        metrics=(generative_validation(model,heldout_examples,label=f'step {step} full')
                 if full_due else screen)
        scope='full_200' if full_due else 'screen_40'
        row={'step':step,'scope':scope,'validation_loss':loss,**metrics,
             'screen_metrics':{k:v for k,v in screen.items() if k!='samples'}}
        if relative_improvement is not None:
            row['three_eval_relative_loss_improvement']=relative_improvement
        report['evaluations'].append(row); atomic_json(EVENT_LOG,report)
        print('STAGE 3 EVAL:',{k:v for k,v in row.items() if k!='samples'})

        observed_truncation=max(screen['truncation_rate'],metrics['truncation_rate'])
        if step>=30 and observed_truncation>.10:
            self.truncation_streak+=1
        else:
            self.truncation_streak=0
        row['truncation_concern_streak']=self.truncation_streak
        if self.truncation_streak>=2:
            report['stop_reason']=f'sustained_truncation_above_10_percent_at_step_{step}'
            control.should_training_stop=True
            atomic_json(EVENT_LOG,report)
            return control
        viability_failure=None
        if step==30 and (screen['structural']<.60 or screen['mapping_adherence']<.20):
            viability_failure='step30_format_or_mapping_below_registered_floor'
        elif step==60 and (screen['structural']<.95 or screen['mapping_adherence']<.60
                           or screen['answer_correct']<.60):
            viability_failure='step60_structure_mapping_or_answer_below_registered_floor'
        elif step==100 and (metrics['mapping_adherence']<.85
                            or metrics['internal_correct']<.85
                            or metrics['answer_correct']<.80):
            viability_failure='step100_mapping_transition_or_answer_below_registered_floor'
        if viability_failure:
            report['stop_reason']=viability_failure; control.should_training_stop=True

        success=full_due and behavioral_success(metrics)
        if full_due:
            if success and self.last_full_step is not None and step-self.last_full_step<=20:
                self.success_streak+=1
            else:
                self.success_streak=1 if success else 0
            self.last_full_step=step
        if self.success_streak>=2:
            report['provisional_accepted']=True
            report['stop_reason']='provisional_success_two_consecutive_full_evaluations'
            control.should_training_stop=True
        elif step==140 and not control.should_training_stop:
            tracking_total=int(metrics['trace_error_diagnostics']['wrong_tracking_total'])
            tracking_gate=(tracking_total<=15 and metrics['structural']>=.95
                           and metrics['answer_correct']>=.875
                           and metrics['previous_code_adherence']>=.95)
            report['step140_explicit_transition_diagnostic']={
                'reference_wrong_tracking_total':19,
                'observed_wrong_tracking_total':tracking_total,
                'absolute_reduction':19-tracking_total,
                'relative_reduction':(19-tracking_total)/19,
                'previous_code_adherence':metrics['previous_code_adherence'],
                'passed':tracking_gate,
            }
            if not tracking_gate:
                report['stop_reason']='step140_explicit_transition_tracking_gate_failed_needs_review'
                control.should_training_stop=True
            else:
                report['step140_gate_passed_continuing_to_full_step150_gate']=True
        elif plateau and not success and not control.should_training_stop:
            report['stop_reason']='early_failure_validation_loss_plateau_below_0.5_percent'
            control.should_training_stop=True
        elif step==60 and not control.should_training_stop:
            report['step60_all_metrics_at_least_95_percent']=behavioral_success(metrics)
            report['stop_reason']=('step60_full_gate_passed_once_needs_review' if success
                                   else 'step60_full_gate_below_95_percent_needs_review')
            control.should_training_stop=True
        elif step==100 and not control.should_training_stop:
            report['step100_all_metrics_at_least_95_percent']=behavioral_success(metrics)
            report['stop_reason']=('step100_full_gate_passed_once_needs_review' if success
                                   else 'step100_full_gate_below_95_percent_needs_review')
            control.should_training_stop=True
        elif step==130 and not control.should_training_stop:
            report['step130_all_metrics_at_least_95_percent']=behavioral_success(metrics)
            report['stop_reason']=('step130_full_gate_passed_once_needs_review' if success
                                   else 'step130_full_gate_below_95_percent_needs_review')
            control.should_training_stop=True
        atomic_json(EVENT_LOG,report)
        return control
    def on_save(self,args,state,control,**kwargs):
        checkpoint=Path(args.output_dir)/f'checkpoint-{int(state.global_step)}'
        if not valid_trainer_checkpoint(checkpoint):
            report['stop_reason']=f'checkpoint_verification_failed:{checkpoint}'
            atomic_json(EVENT_LOG,report)
            raise RuntimeError(f'Checkpoint save is incomplete: {checkpoint}')
        record={'step':int(state.global_step),'path':str(checkpoint)}
        if record not in report['verified_checkpoints']:
            report['verified_checkpoints'].append(record)
        atomic_json(EVENT_LOG,report)
        print('VERIFIED RESUMABLE DRIVE CHECKPOINT:',record)
        return control

controller=Stage3Controller()
"""
        ),
        markdown("## Baseline, bounded training, and separate adapter save\n"),
        code(
            """baseline=report.get('baseline')
if baseline is None:
    baseline=generative_validation(model,screen_examples,label='untrained screen baseline')
    report['baseline_scope']='screen_40'
    report['baseline']=baseline; atomic_json(EVENT_LOG,report)
else:
    print('Reusing saved baseline from the interrupted run.')
print('UNTRAINED BASELINE:',{k:v for k,v in baseline.items() if k!='samples'})
print('BASELINE SAMPLES:'); [print(f'--- {i} ---\\n{text}') for i,text in enumerate(baseline['samples'],1)]
"""
        ),
        code(
            """args=TrainingArguments(
    output_dir=str(TRAINER_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_steps=MAX_STEPS,
    learning_rate=2e-5,
    warmup_steps=10,
    lr_scheduler_type='cosine',
    optim='adamw_torch',
    weight_decay=0.0,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    torch_empty_cache_steps=1,
    bf16=True,
    logging_strategy='steps',logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,
    save_strategy='steps',save_steps=5,save_total_limit=4,save_only_model=False,
    eval_strategy='no',report_to='none',disable_tqdm=False,
    seed=SEED,data_seed=SEED,remove_unused_columns=False,
)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[controller])
report['training_started']=True; atomic_json(EVENT_LOG,report)
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None)
final_step=int(trainer.state.global_step)
if report['stop_reason'] is None:
    report['stop_reason']='hard_step_limit_150' if final_step>=150 else 'trainer_stopped_without_registered_reason'
early_screen_stop=any(str(report['stop_reason']).startswith(prefix) for prefix in (
    'step30_','step60_','step100_','step130_','step140_','sustained_truncation_',
    'gradient_circuit_breaker','nonfinite_'))
if early_screen_stop and report.get('evaluations'):
    latest=report['evaluations'][-1]
    metric_keys=BEHAVIOR_KEYS+('decode_back_correct','truncation_rate','termination_rate',
                               'mean_generated_length','max_generated_length',
                               'answer_error_counts','answer_error_fractions',
                               'answer_failure_count','trace_error_diagnostics','samples')
    final_metrics={key:latest[key] for key in metric_keys}
    report['final_metrics_scope']=latest['scope']
    print('Skipping redundant 200-prompt final generation after registered early screen stop.')
else:
    final_metrics=generative_validation(model,heldout_examples,label='final live full evaluation')
    report['final_metrics_scope']='full_200'
report['final_step']=final_step; report['final_metrics']=final_metrics
report['train_result']=dict(result.metrics)
report['live_full_pass']=behavioral_success(final_metrics) and final_metrics['truncation_rate']==0

save_dir=RUN_DIR/f'candidate-adapter-step-{final_step}'
save_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
adapter_files=list(save_dir.glob('adapter_model.*'))
if not adapter_files or not all(path.stat().st_size for path in adapter_files):
    raise RuntimeError('Adapter save verification failed.')
report['candidate_adapter_path']=str(save_dir)

if report.get('provisional_accepted') and report['live_full_pass']:
    del trainer, result, model, base_model
    gc.collect(); torch.cuda.empty_cache()
    reload_base=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map={'':0},
        low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
    model=PeftModel.from_pretrained(reload_base,save_dir,is_trainable=False)
    model.eval(); model.config.use_cache=True
    reload_metrics=generative_validation(model,heldout_examples,label='clean-reload full evaluation')
    report['reload_metrics']=reload_metrics
    report['reload_verified']=behavioral_success(reload_metrics) and reload_metrics['truncation_rate']==0
    if report['reload_verified']:
        FINAL_ADAPTER.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(FINAL_ADAPTER); tokenizer.save_pretrained(FINAL_ADAPTER)
        final_files=list(FINAL_ADAPTER.glob('adapter_model.*'))
        if not final_files or not all(path.stat().st_size for path in final_files):
            raise RuntimeError('Final reload-verified adapter save failed.')
        report['accepted']=True
        report['stop_reason']='accepted_after_two_live_full_passes_and_clean_reload'
        report['adapter_path']=str(FINAL_ADAPTER)
        report['adapter_files']=[str(x) for x in final_files]
    else:
        report['accepted']=False
        report['stop_reason']='clean_reload_failed_acceptance_gate'
        report['adapter_path']=str(save_dir)
        report['adapter_files']=[str(x) for x in adapter_files]
else:
    report['accepted']=False
    report['reload_verified']=False
    report['adapter_path']=str(save_dir)
    report['adapter_files']=[str(x) for x in adapter_files]
report['library_versions']={name:importlib.metadata.version(name) for name in
                            ('torch','transformers','peft','bitsandbytes','accelerate')}
atomic_json(EVENT_LOG,report)
print('===== STAGE 3 FINAL REPORT =====')
print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','final_metrics','dataset_audit')},indent=2,default=str))
print('FINAL METRICS:',{k:v for k,v in final_metrics.items() if k!='samples'})
print('FINAL SAMPLES:'); [print(f'--- {i} ---\\n{text}') for i,text in enumerate(final_metrics['samples'],1)]
print('STOP HERE. Third-domain testing and adapter composition are intentionally absent.')
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
