from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
OUTPUT = ROOT / "notebooks" / "stage35_declared_mapping_control.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def main() -> None:
    data_source = (ROOT / "src/data/fan_generalization.py").read_text().replace(
        "from __future__ import annotations\n\n", "", 1
    )
    cells = [
        markdown("""# Stage 3.5 control — supplied mapping, zero-shot

This matched read-only control uses the same 100 greenhouse-fan tasks as the
failed self-selected-mapping test, but declares a unique randomized mapping in
each prompt. It loads checkpoint 130, performs no training, supplies no worked
examples, and never composes adapters. Stage 4 remains blocked.
"""),
        code("%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate\n"),
        code("""import hashlib, json, os, random
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
from google.colab import drive
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

drive.mount('/content/drive',force_remount=False)
if not torch.cuda.is_available(): raise RuntimeError('A Colab GPU is required.')
SEED=20260811
MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
CHECKPOINT=Path('/content/drive/MyDrive/AISI/checkpoints/latch-seed-stage3-decode-back-sft-v3/trainer-output/checkpoint-130')
OUTPUT_DIR=Path('/content/drive/MyDrive/AISI/checkpoints/stage35-declared-mapping-control-v1')
PROGRESS=OUTPUT_DIR/'rollouts.json'; REPORT=OUTPUT_DIR/'report.json'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
def valid_adapter(path):
    return (path.is_dir() and (path/'adapter_config.json').is_file()
            and any(p.name.startswith('adapter_model') and p.stat().st_size>0 for p in path.iterdir()))
if not valid_adapter(CHECKPOINT): raise RuntimeError(f'Invalid checkpoint: {CHECKPOINT}')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'checkpoint':str(CHECKPOINT),
       'training':False,'stage4_blocked':True})
"""),
        code(data_source),
        code("""examples=generate_declared_mapping_control(SEED,100)
audit=verify_declared_control(examples)
assert audit['accepted'] and audit['semantic_pass_rate']==100.0
assert audit['unique_token_pairs']==100
print('DECLARED-MAPPING CONTROL VERIFIED:',json.dumps(audit,indent=2,sort_keys=True))
print('MATCHED PROMPT EXAMPLE (NO DEMONSTRATION):\\n',examples[0].prompt)
"""),
        code("""tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map={'':0},
    low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
model=PeftModel.from_pretrained(base,CHECKPOINT,is_trainable=False)
model.eval(); model.config.use_cache=True
assert not any(p.requires_grad for p in model.parameters())
adapter_files=sorted(p for p in CHECKPOINT.iterdir() if p.name.startswith('adapter_model'))
adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter_files}
print({'adapter_sha256':adapter_sha256,'trainable_parameters':0})
"""),
        code("""SYSTEM='You solve state-tracking tasks accurately and follow the requested output format.'
saved=json.loads(PROGRESS.read_text()) if PROGRESS.is_file() and PROGRESS.stat().st_size else {'rows':[]}
completed={row['example_id'] for row in saved['rows']}
def atomic_json(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(payload,indent=2,sort_keys=True)); temp.replace(path)
for start in range(0,100,2):
    chunk=[row for row in examples[start:start+2] if row.example_id not in completed]
    if not chunk: continue
    prompts=[tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM},{'role':'user','content':row.prompt}],
        tokenize=False,add_generation_prompt=True) for row in chunk]
    batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=300,do_sample=False,
                              pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    for example,text in zip(chunk,texts):
        score=score_declared_completion(example,text)
        saved['rows'].append({'example_id':example.example_id,
                              'declared_pair':[example.token_for_running,example.token_for_stopped],
                              'final_answer':example.base.final_answer,**score})
        completed.add(example.example_id)
    atomic_json(PROGRESS,saved); print(f'SAVED CONTROL PROGRESS: {len(completed)}/100')
assert len(saved['rows'])==100 and len(completed)==100
"""),
        code("""rows=sorted(saved['rows'],key=lambda row:row['example_id']); n=len(rows)
adherent_pairs=Counter(tuple(sorted(row['declared_pair'])) for row in rows if row['declared_mapping_adherence'])
metrics={
    'evaluation_count':n,
    'structural_rate':sum(row['structural'] for row in rows)/n,
    'nonliteral_encoding_rate':sum(row['nonliteral'] for row in rows)/n,
    'global_mapping_consistency_rate':sum(row['global_consistent'] for row in rows)/n,
    'declared_mapping_adherence_rate':sum(row['declared_mapping_adherence'] for row in rows)/n,
    'final_answer_accuracy':sum(row['answer_correct'] for row in rows)/n,
    'answer_is_code_word_rate':sum(row['answer_is_code_word'] for row in rows)/n,
    'latch_coin_vocabulary_leakage_rate':sum(row['latch_coin_leakage'] for row in rows)/n,
    'distinct_correctly_applied_declared_pairs':len(adherent_pairs),
}
answer_failures=[row for row in rows if not row['answer_correct']]
metrics['answer_is_code_word_fraction_of_failures']=(
    sum(row['answer_is_code_word'] for row in answer_failures)/len(answer_failures)
    if answer_failures else 0.0)
metrics['answer_failure_category_counts']=dict(Counter(
    row['answer_failure_type'] for row in answer_failures))
criteria={
    'mapping_application_at_least_80_percent':metrics['declared_mapping_adherence_rate']>=.80,
    'global_consistency_at_least_80_percent':metrics['global_mapping_consistency_rate']>=.80,
    'answer_accuracy_at_least_80_percent':metrics['final_answer_accuracy']>=.80,
    'structure_at_least_90_percent':metrics['structural_rate']>=.90,
    'nonliteral_at_least_90_percent':metrics['nonliteral_encoding_rate']>=.90,
    'no_domain_leakage':metrics['latch_coin_vocabulary_leakage_rate']==0,
}
report={'stage':'3.5_declared_mapping_control','source_checkpoint':str(CHECKPOINT),
        'mode':'zero_shot_no_training','audit':audit,'adapter_sha256':adapter_sha256,
        'metrics':metrics,'criteria':criteria,'passed':all(criteria.values()),'rows':rows,
        'stage4_authorized':False,'next_action':'STOP_FOR_REVIEW'}
atomic_json(REPORT,report)
print('===== DECLARED-MAPPING CONTROL REPORT =====')
print(json.dumps({**metrics,'criteria':criteria,'passed':report['passed']},indent=2,sort_keys=True))
print('\\n===== REPRESENTATIVE RAW COMPLETIONS =====')
for index,row in enumerate(rows[:10],1):
    print(f"\\n--- {index}: {row['example_id']} declared={row['declared_pair']} ---\\n{row['text']}")
print('\\nREPORT SAVED:',REPORT)
print('STOP HERE. No training or Stage 4 composition was performed.')
"""),
    ]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
