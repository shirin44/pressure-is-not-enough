"""Build a no-training audit of checkpoint-500 adapter/evaluation state."""
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET=ROOT/'notebooks'/'checkpoint500_evaluation_path_readonly_audit.ipynb'
COINFLIP=(ROOT/'src'/'data'/'coinflip.py').read_text()

def cell(kind,source):
    c={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
    if kind=='code': c.update({'execution_count':None,'outputs':[]})
    return c

cells=[cell('markdown','''# Checkpoint-500 evaluation-path audit (read-only)

No training, optimizer, scheduler, or gradients. Evaluates the same 25 held-out prompts twice with the checkpoint-500 `default` adapter, once with adapters disabled, then restores `default` and checks byte-identical deterministic output.
'''),
cell('code','%pip install -q transformers==5.13.1 peft==0.19.1 bitsandbytes==0.50.0 accelerate safetensors\n'),
cell('code',r'''import gc,hashlib,importlib.metadata,json,logging,random,re,statistics
from pathlib import Path
import torch
from google.colab import drive
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,
                          StoppingCriteria,StoppingCriteriaList)

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'; RUN_SEED=20260730; MAX_NEW_TOKENS=256
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select an L4; found {torch.cuda.get_device_name(0)}')
expected={'transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.50.0'}
actual={k:importlib.metadata.version(k) for k in expected}
if actual!=expected: raise RuntimeError(f'Version mismatch: {actual}')
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
print({'gpu':torch.cuda.get_device_name(0),**actual})
'''),
cell('code',r'''drive.mount('/content/drive',force_remount=False)
SOURCE=Path('/content/drive/MyDrive/AISI/checkpoints/full-snapshots/step-500')
adapter=SOURCE/'adapter_model.safetensors'
if not adapter.is_file() or not adapter.stat().st_size:
    raise RuntimeError(f'Missing checkpoint-500 adapter: {adapter}')
print('READ-ONLY SOURCE:',SOURCE)
'''),
cell('code',COINFLIP),
cell('code',r'''print('===== BUILD THE EXACT DISJOINT 25-PROMPT HELD-OUT SET =====')
def unique_pool(size,start,excluded=()):
    rows=[]; seen=set(excluded); seed=start
    while len(rows)<size:
        prompt,truth=generate_coinflip_example(3+(seed%6),seed); seed+=1
        if prompt in seen: continue
        seen.add(prompt); rows.append({'prompt':prompt,'ground_truth':truth})
    return rows
train=unique_pool(100,RUN_SEED)
heldout=unique_pool(25,RUN_SEED+1_000_000,{x['prompt'] for x in train})
assert len(heldout)==25 and {x['prompt'] for x in train}.isdisjoint({x['prompt'] for x in heldout})
prompt_digest=hashlib.sha256(json.dumps(heldout,sort_keys=True).encode()).hexdigest()
print({'heldout':25,'prompt_digest':prompt_digest})
# Exact corrected-template gate: every evaluated prompt must contain the
# clarified operations, explicit worked example, and same-line requirement.
required_prompt_fragments=(
    'Put Step and State on the SAME line',
    'Step 1: The state remains unchanged. State: Heads',
    'inside <answer>...</answer> tags.',
)
for fragment in required_prompt_fragments:
    assert all(fragment in row['prompt'] for row in heldout),fragment
for row in heldout:
    instructions=re.findall(r'(?m)^\d+\. (.+)$',row['prompt'])
    assert instructions and all(text in {
        'same as previous (the state does NOT change)',
        'different from previous (the state flips)'} for text in instructions)
print({'prompt_template_verified':True,'required_fragments':required_prompt_fragments,
       'max_new_tokens':MAX_NEW_TOKENS,'stop_string':'</answer>'})
'''),
cell('code',r'''print('===== LOAD CHECKPOINT 500 FOR INFERENCE ONLY =====')
for name in ('model','base_model','ANSWER_STOP'):
    stale=globals().pop(name,None)
    if stale is not None: del stale
gc.collect(); torch.cuda.empty_cache()
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_enable_fp32_cpu_offload=True)
base_model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=False)
model=PeftModel.from_pretrained(base_model,SOURCE,adapter_name='default',is_trainable=False)
model.set_adapter('default'); model.eval()
if model.training: raise RuntimeError('model.eval() did not take effect.')
initially_trainable=[name for name,param in model.named_parameters() if param.requires_grad]
# Some PEFT/Transformers combinations leave adapter parameters marked trainable
# even when loaded with is_trainable=False. No optimizer exists, but freeze them
# explicitly so inference_mode is backed by an unambiguous read-only model.
model.requires_grad_(False); model.eval()
if any(p.requires_grad for p in model.parameters()):
    raise RuntimeError('Explicit model freeze failed.')

def canonical_tensor_hash(mapping):
    digest=hashlib.sha256()
    for name in sorted(mapping):
        value=mapping[name].detach().float().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

reference_state=load_safetensors(str(adapter))
loaded_state=get_peft_model_state_dict(model,adapter_name='default')
if set(reference_state)!=set(loaded_state):
    raise RuntimeError({'missing_from_loaded':sorted(set(reference_state)-set(loaded_state))[:10],
                        'unexpected_loaded':sorted(set(loaded_state)-set(reference_state))[:10]})
mismatched=[name for name in reference_state
            if not torch.equal(reference_state[name].detach().float().cpu(),
                               loaded_state[name].detach().float().cpu())]
if mismatched: raise RuntimeError(f'Loaded adapter differs from checkpoint-500 tensors: {mismatched[:10]}')
reference_content_hash=canonical_tensor_hash(reference_state)
loaded_content_hash=canonical_tensor_hash(loaded_state)
raw_file_hash=hashlib.sha256(adapter.read_bytes()).hexdigest()
assert loaded_content_hash==reference_content_hash

ANSWER_IDS=tokenizer.encode('</answer>',add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self,input_ids,scores,**kwargs):
        n=len(ANSWER_IDS)
        return torch.tensor([row.numel()>=n and row[-n:].tolist()==ANSWER_IDS for row in input_ids],
                            device=input_ids.device,dtype=torch.bool)
ANSWER_STOP=StoppingCriteriaList([StopAfterAnswer()])
print({'active_adapter':str(model.active_adapter),'training':model.training,
       'initially_trainable_parameter_names':initially_trainable,
       'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
       'answer_stop_ids':ANSWER_IDS,'max_new_tokens':MAX_NEW_TOKENS,
       'checkpoint500_raw_file_sha256':raw_file_hash,
       'checkpoint500_tensor_content_sha256':reference_content_hash,
       'loaded_adapter_tensor_content_sha256':loaded_content_hash,
       'loaded_tensor_count':len(loaded_state),'exact_tensor_match':True})
'''),
cell('code',r'''print('===== STRICT PARSER CONFIGURATION GATE =====')
parser_cases={
 'Step 1: reasoning. State: Heads':[(1,'heads')],
 'Step 1: reasoning. State: Heads.':[(1,'heads')],
 'Step 1: reasoning. State: Heads,':[(1,'heads')],
 'Step 1: reasoning. State: Heads. anything':[],
 'Step 1: reasoning. State: the':[],
 'Step 1: reasoning.\nState: Heads':[],
}
for text,expected_slots in parser_cases.items():
    actual_slots=parse_state_slots(text)
    assert actual_slots==expected_slots,(text,actual_slots,expected_slots)
print({'parser_verified':True,'same_line_required':True,
       'accepted_incidental_suffixes':['none','period','comma','whitespace'],
       'token_pattern':'single capitalized alphabetic word, 1-15 characters'})
'''),
cell('code',r'''def evaluate(label):
    model.eval(); rows=[]
    for item in heldout:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':item['prompt']}],
            tokenize=False,add_generation_prompt=True)
        batch=tokenizer(rendered,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                stopping_criteria=ANSWER_STOP,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
        score=score_completion(text,item['ground_truth'],500,prompt=item['prompt'])
        rows.append({'truth':item['ground_truth'],'text':text,'score':score})
    summary={'label':label,'accuracy':statistics.fmean(x['score']['r_task']==4.0 for x in rows),
        'strict_structure':statistics.fmean(x['score']['p_structure']==0.0 for x in rows),
        'valid_format':statistics.fmean(x['score']['r_task']!=-5.0 for x in rows),
        'ended_at_answer':statistics.fmean(x['text'].rstrip().casefold().endswith('</answer>') for x in rows),
        'output_digest':hashlib.sha256(json.dumps([x['text'] for x in rows]).encode()).hexdigest(),
        'active_adapter':str(model.active_adapter),'training':model.training}
    print(summary); return rows,summary

print('===== DEFAULT ADAPTER: REPEATABILITY CHECK =====')
model.set_adapter('default'); default_a,summary_a=evaluate('default_A')
model.set_adapter('default'); default_b,summary_b=evaluate('default_B')
assert [x['text'] for x in default_a]==[x['text'] for x in default_b]
print('PASSED: two consecutive default-adapter evaluations are byte-identical.')
print('===== FIVE FULL RAW CHECKPOINT-500 COMPLETIONS =====')
for i,row in enumerate(default_a[:5],1):
    print('\n'+'-'*100)
    print({'index':i,'truth':row['truth'],'r_task':row['score']['r_task'],
           'p_structure':row['score']['p_structure'],
           'p_state_variation':row['score']['p_state_variation']})
    print(row['text'])
'''),
cell('code',r'''print('===== ADAPTER-DISABLED CONTROL =====')
with model.disable_adapter():
    disabled,summary_disabled=evaluate('adapters_disabled_base_model')
print('===== RESTORE DEFAULT ADAPTER =====')
model.set_adapter('default'); model.eval()
restored,summary_restored=evaluate('default_restored')
restoration_identical=[x['text'] for x in default_a]==[x['text'] for x in restored]
if not restoration_identical:
    raise RuntimeError('Default-adapter output changed after disable/restore round trip.')

report={'source':str(SOURCE),'prompt_digest':prompt_digest,'default_A':summary_a,
        'default_B':summary_b,'adapters_disabled':summary_disabled,
        'default_restored':summary_restored,'default_repeat_identical':True,
        'restore_roundtrip_identical':restoration_identical,'training_performed':False}
OUT=Path('/content/drive/MyDrive/AISI/audits/checkpoint500_evaluation_path_audit.json')
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2))
print('\n===== FINAL READ-ONLY REPORT ====='); print(json.dumps(report,indent=2))
print('Evidence:',OUT)
print('No training occurred. Stop here for review.')
'''),
cell('code',r'''# Only print raw outputs when the checkpoint-500 baseline is not reproduced.
if summary_a['accuracy']!=0.64 or summary_a['strict_structure']!=1.0:
    print('===== UNEXPECTED DEFAULT-ADAPTER OUTPUTS =====')
    for i,row in enumerate(default_a,1):
        print('\n'+'-'*100); print({'index':i,'truth':row['truth'],'score':row['score']}); print(row['text'])
else:
    print('EXPECTED BASELINE REPRODUCED: accuracy=0.64, strict_structure=1.0')
''')]

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET); print(TARGET)
