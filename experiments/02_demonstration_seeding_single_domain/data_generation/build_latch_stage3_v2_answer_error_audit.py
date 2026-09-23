"""Build a read-only Colab audit for Stage 3 v2 final-answer errors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
OUTPUT = ROOT / "notebooks" / "latch_stage3_v2_final_answer_error_audit.ipynb"


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
            """# Stage 3 v2 — read-only final-answer error audit

This notebook does **no training and no checkpoint modification**. It loads the
nonaccepted step-150 explicit-mapping latch adapter, regenerates the exact 200
held-out prompts, verifies that aggregate metrics reproduce the saved report,
and then analyzes final-answer errors by mapping orientation, sequence length,
true final state/expected final code, and emitted final code role.
"""
        ),
        code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate scipy\n"),
        code(
            """import gc, json, math, os, re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
from google.colab import drive
from peft import PeftModel
from scipy.stats import chi2_contingency
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

drive.mount('/content/drive',force_remount=False)
if not torch.cuda.is_available(): raise RuntimeError('A Colab GPU runtime is required.')

SEED=20260809
MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
ADAPTER_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/latch-seed-stage3-explicit-mapping-sft-v2/nonaccepted-adapter-step-150')
AUDIT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/latch-seed-stage3-explicit-mapping-sft-v2/read-only-answer-error-audit')
OUTPUT_JSON=AUDIT_DIR/'answer_error_audit.json'
MAX_NEW_TOKENS=192
if not (ADAPTER_DIR/'adapter_config.json').is_file():
    raise RuntimeError(f'Missing adapter config: {ADAPTER_DIR}')
if not any(p.name.startswith('adapter_model') and p.stat().st_size for p in ADAPTER_DIR.iterdir()):
    raise RuntimeError(f'Missing/non-empty adapter weights: {ADAPTER_DIR}')
AUDIT_DIR.mkdir(parents=True,exist_ok=True)
print({'adapter':str(ADAPTER_DIR),'output':str(OUTPUT_JSON),'gpu':torch.cuda.get_device_name(0)})
"""
        ),
        code(latch_source),
        code(
            """train_examples,heldout_examples=generate_latch_seed_dataset(
    SEED,include_decode_back=False,include_mapping_anchor=False,
    include_explicit_transitions=False)
audit=audit_latch_seed_dataset(train_examples,heldout_examples,SEED)
assert audit['accepted'] and audit['semantic_verification_pass_rate']==100.0
assert audit['prompt_mapping_verification_pass_rate']==100.0
assert len(heldout_examples)==200
print('EXACT HELD-OUT DATA REVERIFIED:',{'count':len(heldout_examples),'semantic_pass_rate':100.0})
"""
        ),
        code(
            """gc.collect(); torch.cuda.empty_cache()
free_gib=torch.cuda.mem_get_info()[0]/1024**3
if free_gib<12: raise RuntimeError(f'Only {free_gib:.2f} GiB free; restart the runtime.')
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,
    quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,
    use_safetensors=True,trust_remote_code=False)
model=PeftModel.from_pretrained(base,ADAPTER_DIR,is_trainable=False)
model.eval(); model.config.use_cache=True
print({'gpu_free_before_gib':round(free_gib,2),'gpu_allocated_gib':round(torch.cuda.memory_allocated()/1024**3,2)})
"""
        ),
        code(
            """SYSTEM='You solve binary-state tracking tasks accurately and follow the requested output format.'
STATE_RE=re.compile(r'^\\s*Step\\s+(\\d+)\\s*:.*?\\bState:\\s*([A-Z][A-Za-z]{2,9})[.,]?\\s*$',re.MULTILINE)
ANSWER_RE=re.compile(r'<answer>\\s*(Locked|Unlocked)\\s*</answer>\\s*$',re.IGNORECASE)
LITERALS={'locked','unlocked'}

def prompt_messages(example):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':example.prompt}]

def score(example,text):
    matches=[(int(i),token) for i,token in STATE_RE.findall(text.split('<answer>',1)[0])]
    expected_states=simulate(example.initial_state,example.operations)
    structural=(len(matches)==len(example.operations) and
                [i for i,_ in matches]==list(range(1,len(example.operations)+1)))
    tokens=[token.casefold() for _,token in matches]
    internal=global_consistent=mapping_adherence=nonliteral=False
    if structural:
        internal=all((tokens[i]==tokens[i-1]) if example.operations[i]=='same'
                     else (tokens[i]!=tokens[i-1]) for i in range(1,len(tokens)))
        by_state={state:set() for state in STATES}
        for state,token in zip(expected_states,tokens): by_state[state].add(token)
        covered=all(by_state[state] for state in STATES)
        global_consistent=(covered and all(len(by_state[state])==1 for state in STATES)
                           and by_state['Locked']!=by_state['Unlocked'])
        declared={'Locked':example.token_for_locked.casefold(),
                  'Unlocked':example.token_for_unlocked.casefold()}
        mapping_adherence=all(token==declared[state]
                              for state,token in zip(expected_states,tokens))
        nonliteral=all(token not in LITERALS for token in tokens)
    answer=ANSWER_RE.search(text)
    predicted=answer.group(1).capitalize() if answer else None
    answer_valid=answer is not None
    answer_correct=predicted==example.final_answer
    expected_last_role=f'{example.final_answer}_code'
    if not matches:
        emitted_last_role='missing_or_invalid'
    else:
        last=matches[-1][1].casefold()
        if last==example.token_for_locked.casefold(): emitted_last_role='Locked_code'
        elif last==example.token_for_unlocked.casefold(): emitted_last_role='Unlocked_code'
        else: emitted_last_role='other_token'
    orientation=('Locked_token_alphabetically_first' if
                 example.token_for_locked<example.token_for_unlocked else
                 'Unlocked_token_alphabetically_first')
    return {
        'example_id':example.example_id,'sequence_length':len(example.operations),
        'mapping_orientation':orientation,'initial_state':example.initial_state,
        'final_state':example.final_answer,'expected_last_token_role':expected_last_role,
        'emitted_last_token_role':emitted_last_role,'predicted_answer':predicted,
        'answer_correct':answer_correct,'answer_valid':answer_valid,'structural':structural,
        'internal_correct':internal,'global_consistent':global_consistent,
        'mapping_adherence':mapping_adherence,'nonliteral':nonliteral,
        'last_emitted_token':matches[-1][1] if matches else None,'completion':text,
    }

@torch.inference_mode()
def evaluate(examples,batch_size=2):
    rows=[]
    for start in range(0,len(examples),batch_size):
        chunk=examples[start:start+batch_size]
        prompts=[tokenizer.apply_chat_template(prompt_messages(x),tokenize=False,
                                               add_generation_prompt=True) for x in chunk]
        batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
        output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        width=batch['input_ids'].shape[1]
        texts=tokenizer.batch_decode(output[:,width:],skip_special_tokens=True)
        rows.extend(score(example,text) for example,text in zip(chunk,texts))
        if start%20==0: print(f'Generated {min(start+len(chunk),len(examples))}/{len(examples)}')
    return rows

rows=evaluate(heldout_examples)
print('GENERATION COMPLETE:',len(rows))
"""
        ),
        code(
            """EXPECTED={'answer_correct':.765,'answer_valid':.985,'structural':.99,
          'internal_correct':.93,'global_consistent':.92,
          'mapping_adherence':.915,'nonliteral':.985}
actual={key:sum(bool(row[key]) for row in rows)/len(rows) for key in EXPECTED}
delta={key:actual[key]-EXPECTED[key] for key in EXPECTED}
TOLERANCE={'answer_correct':.025,'answer_valid':.02,'mapping_adherence':.015,
           'structural':.005,'internal_correct':.005,
           'global_consistent':.005,'nonliteral':.005}
print('===== REPRODUCTION CHECK =====')
print({'expected':EXPECTED,'actual':actual,'delta':delta,'tolerance':TOLERANCE})
violations={key:value for key,value in delta.items()
            if abs(value)>TOLERANCE[key]+1e-12}
if violations:
    raise RuntimeError(f'Audit generations exceed the registered reload tolerances: {violations}')

def grouped(field):
    buckets=defaultdict(list)
    for row in rows: buckets[str(row[field])].append(row)
    result=[]
    for value,items in sorted(buckets.items()):
        errors=sum(not x['answer_correct'] for x in items)
        result.append({'value':value,'n':len(items),'errors':errors,
                       'error_rate':errors/len(items)})
    return result

def association(field):
    table=grouped(field)
    counts=np.array([[x['n']-x['errors'],x['errors']] for x in table])
    chi2,p,_,_=chi2_contingency(counts)
    n=counts.sum(); v=math.sqrt(chi2/(n*max(1,min(counts.shape)-1)))
    return {'groups':table,'chi2':float(chi2),'p_value':float(p),'cramers_v':float(v),
            'max_minus_min_error_rate':max(x['error_rate'] for x in table)-min(x['error_rate'] for x in table)}

fields=('mapping_orientation','sequence_length','final_state',
        'expected_last_token_role','emitted_last_token_role')
clusters={field:association(field) for field in fields}
confusion=Counter((row['final_state'],row['predicted_answer'] or 'INVALID') for row in rows)
report={'read_only':True,'checkpoint_logical_step':150,'rows':rows,
        'reproduction':{'expected':EXPECTED,'actual':actual,'delta':delta,
                        'tolerance':TOLERANCE,'passed':True},
        'clusters':clusters,
        'answer_confusion':{f'{truth}->{prediction}':count
                            for (truth,prediction),count in sorted(confusion.items())}}
OUTPUT_JSON.write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n')

print('===== FINAL-ANSWER ERROR CLUSTERING =====')
for field in fields:
    print('\\n',field)
    for row in clusters[field]['groups']: print(row)
    print({k:clusters[field][k] for k in ('p_value','cramers_v','max_minus_min_error_rate')})
print('\\nANSWER CONFUSION:',report['answer_confusion'])
print('FULL READ-ONLY EVIDENCE SAVED:',OUTPUT_JSON)
print('STOP HERE. Dataset regeneration and training are intentionally absent.')
"""
        ),
        code(
            """print('===== RUN THIS LAST CELL: RECOVER CLUSTERING AFTER THE OLD TOLERANCE STOP =====')
if 'rows' not in globals() or len(rows)!=200:
    raise RuntimeError('The 200 generated rows are not in memory; run the generation cell first.')

EXPECTED={'answer_correct':.765,'answer_valid':.985,'structural':.99,
          'internal_correct':.93,'global_consistent':.92,
          'mapping_adherence':.915,'nonliteral':.985}
TOLERANCE={'answer_correct':.025,'answer_valid':.02,'mapping_adherence':.015,
           'structural':.005,'internal_correct':.005,
           'global_consistent':.005,'nonliteral':.005}
actual={key:sum(bool(row[key]) for row in rows)/len(rows) for key in EXPECTED}
delta={key:actual[key]-EXPECTED[key] for key in EXPECTED}
violations={key:value for key,value in delta.items()
            if abs(value)>TOLERANCE[key]+1e-12}
print('REPRODUCTION:',{'expected':EXPECTED,'actual':actual,'delta':delta,
                       'tolerance':TOLERANCE,'violations':violations})
if violations:
    raise RuntimeError(f'Reload evidence exceeds registered tolerances: {violations}')

def recovery_grouped(field):
    buckets=defaultdict(list)
    for row in rows: buckets[str(row[field])].append(row)
    result=[]
    for value,items in sorted(buckets.items()):
        errors=sum(not item['answer_correct'] for item in items)
        result.append({'value':value,'n':len(items),'errors':errors,
                       'error_rate':errors/len(items)})
    return result

def recovery_association(field):
    table=recovery_grouped(field)
    counts=np.array([[item['n']-item['errors'],item['errors']] for item in table])
    chi2,p,_,_=chi2_contingency(counts)
    n=counts.sum(); v=math.sqrt(chi2/(n*max(1,min(counts.shape)-1)))
    return {'groups':table,'chi2':float(chi2),'p_value':float(p),
            'cramers_v':float(v),
            'max_minus_min_error_rate':max(x['error_rate'] for x in table)-min(x['error_rate'] for x in table)}

fields=('mapping_orientation','sequence_length','final_state',
        'expected_last_token_role','emitted_last_token_role')
clusters={field:recovery_association(field) for field in fields}
confusion=Counter((row['final_state'],row['predicted_answer'] or 'INVALID') for row in rows)
report={'read_only':True,'checkpoint_logical_step':150,'rows':rows,
        'reproduction':{'expected':EXPECTED,'actual':actual,'delta':delta,
                        'tolerance':TOLERANCE,'passed':True},
        'clusters':clusters,
        'answer_confusion':{f'{truth}->{prediction}':count
                            for (truth,prediction),count in sorted(confusion.items())}}
OUTPUT_JSON.write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n')

print('===== FINAL-ANSWER ERROR CLUSTERING =====')
for field in fields:
    print('\\n'+field)
    for item in clusters[field]['groups']: print(item)
    print({key:clusters[field][key] for key in
           ('p_value','cramers_v','max_minus_min_error_rate')})
print('\\nANSWER CONFUSION:',report['answer_confusion'])
print('FULL READ-ONLY EVIDENCE SAVED:',OUTPUT_JSON)
print('DONE. Do not regenerate data or train yet.')
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
