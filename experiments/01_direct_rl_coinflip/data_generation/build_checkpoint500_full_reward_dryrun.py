"""Build the self-gating checkpoint-500 corrected full-reward dry-run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "notebooks" / "checkpoint500_corrected_full_reward_dryrun.ipynb"
COINFLIP = (ROOT / "src" / "data" / "coinflip.py").read_text()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 corrected full-reward dry-run

Five optimizer updates only. Loads checkpoint 500 adapter weights, creates a fresh optimizer and linear scheduler, verifies the exact reward invariant, uses the corrected same-line prompt/parser and `</answer>` stopping, evaluates the same 25 held-out prompts before/after, then stops for review. **No real continuation is included.**
"""),
    code("%pip install -q transformers==5.13.1 trl==1.9.2 peft==0.19.1 bitsandbytes==0.50.0 accelerate datasets safetensors\n"),
    code(r'''import gc, importlib.metadata, itertools, json, logging, math, os, random, re, statistics
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from datasets import Dataset
from google.colab import drive
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList, TrainerCallback)
from trl import GRPOConfig, GRPOTrainer

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'; RUN_SEED=20260730
GROUP_SIZE=8; MAX_DYNAMIC_ATTEMPTS=3; MAX_NEW_TOKENS=256
DRY_STEPS=5; LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes.autograd._functions').disabled=True

def version(name): return importlib.metadata.version(name)
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4; found {torch.cuda.get_device_name(0)}')
expected={'transformers':'5.13.1','trl':'1.9.2','peft':'0.19.1','bitsandbytes':'0.50.0'}
actual={k:version(k) for k in expected}
if actual!=expected: raise RuntimeError(f'Version mismatch: expected={expected}, actual={actual}')
print({'gpu':torch.cuda.get_device_name(0),**actual})
'''),
    code(r'''drive.mount('/content/drive',force_remount=False)
SOURCE=Path('/content/drive/MyDrive/AISI/checkpoints/full-snapshots/step-500')
ROOT=Path('/content/drive/MyDrive/AISI/checkpoints')
if not (SOURCE/'adapter_model.safetensors').is_file():
    raise RuntimeError(f'Missing checkpoint-500 adapter: {SOURCE}')

def inspect_source_scheduler(checkpoint):
    trainer=json.loads((checkpoint/'trainer_state.json').read_text()) if (checkpoint/'trainer_state.json').is_file() else {}
    scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True) if (checkpoint/'scheduler.pt').is_file() else {}
    optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True) if (checkpoint/'optimizer.pt').is_file() else {}
    last_epoch=int(scheduler.get('last_epoch',trainer.get('global_step',0)))
    horizon=int(scheduler.get('_step_count',last_epoch))
    lrs=[float(g.get('lr',0.0)) for g in optimizer.get('param_groups',[])]
    near_zero=(not lrs) or max(abs(x) for x in lrs)<=1e-8
    return {'last_epoch':last_epoch,'recorded_lr':lrs,'near_zero_lr':near_zero,
            'scheduler_state_present':bool(scheduler),'optimizer_state_present':bool(optimizer)}

source_scheduler=inspect_source_scheduler(SOURCE)
RESUME_MODE='weights_only_fresh_schedule'
if RESUME_MODE!='weights_only_fresh_schedule':
    raise RuntimeError('REFUSED: this phase transition requires weights-only + fresh schedule.')
print('SOURCE SCHEDULER AUDIT:',source_scheduler)
print('LOAD POLICY:',{'loaded':['adapter weights only'],
      'reinitialized':['optimizer','scheduler','trainer state']})

lr_curve={501:1e-5,503:1e-5*(1-2/DRY_STEPS),505:1e-5*(1-4/DRY_STEPS)}
assert lr_curve[501]>lr_curve[503]>lr_curve[505]>0
print('FRESH 5-STEP LINEAR LR CURVE:',lr_curve)

for version_id in range(1,1000):
    OUTPUT=ROOT/f'grpo-checkpoint500-corrected-fullreward-dryrun-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG=OUTPUT/'checkpoint500_fullreward_dryrun.json'
'''),
    code(COINFLIP),
    code(r'''print('===== FINAL REWARD/PARSER PREFLIGHT =====')
# Narrow normalization and same-line behavior.
assert parse_state_slots('Step 1: reasoning. State: Heads.')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads. anything')==[]
assert parse_state_slots('Step 1: reasoning. State: the')==[]
assert parse_state_slots('Step 1: reasoning.\nState: Heads')==[]

# No novelty term is active in this final configuration.
# P<2, L<=.5, S<=.5, V<=.5, B<=.15:
# min(C)=4-2-.5-.5-.5=0.5; max(W)=-.5+.15=-.35 => margin .85
# min(W)=-.5-2-.5-.5-.5=-4; max(M)=-5+.15=-4.85 => margin .85
task={'correct':4.0,'wrong':-0.5,'malformed':-5.0}
aux=list(itertools.product((0.0,2.0),(0.0,0.5),(0.0,0.5),(0.0,0.5),(0.0,0.15)))
totals={k:[r-p-l-s-v+b for p,l,s,v,b in aux] for k,r in task.items()}
margins={'correct_over_wrong':min(totals['correct'])-max(totals['wrong']),
         'wrong_over_malformed':min(totals['wrong'])-max(totals['malformed'])}
assert all(abs(x-0.85)<1e-12 for x in margins.values()),margins

prompt,truth=generate_coinflip_example(3,12345)
good='Step 1: same. State: Heads.\nStep 2: flips. State: Tails\nStep 3: same. State: Tails\n<answer>'+truth+'</answer>'
score=score_completion(good,truth,30,prompt=prompt)
assert score['r_novelty']==0.0
assert math.isclose(2*(1-math.exp(-.075*12.78)),1.2328,abs_tol=.001)
print({'ranges':{k:[min(v),max(v)] for k,v in totals.items()},'margins':margins})
print('PASSED: corrected parser and exact full-reward invariant (0.85/0.85).')
'''),
    code(r'''print('===== BUILD DETERMINISTIC DISJOINT DATA =====')
def unique_pool(size,start,excluded=()):
    rows=[]; seen=set(excluded); seed=start
    while len(rows)<size:
        prompt,truth=generate_coinflip_example(3+(seed%6),seed); seed+=1
        if prompt in seen: continue
        seen.add(prompt); rows.append({'prompt':prompt,'ground_truth':truth})
    return rows
TRAIN_POOL=unique_pool(100,RUN_SEED)
HELDOUT_POOL=unique_pool(25,RUN_SEED+1_000_000,{x['prompt'] for x in TRAIN_POOL})
assert len({x['prompt'] for x in TRAIN_POOL})==100
assert len({x['prompt'] for x in HELDOUT_POOL})==25
assert {x['prompt'] for x in TRAIN_POOL}.isdisjoint({x['prompt'] for x in HELDOUT_POOL})
train_dataset=Dataset.from_list(TRAIN_POOL).shuffle(seed=RUN_SEED)
print({'train':len(TRAIN_POOL),'heldout':len(HELDOUT_POOL),'disjoint':True})
'''),
    code(r'''print('===== LOAD MODEL: CHECKPOINT-500 WEIGHTS ONLY =====')
for name in ('diagnostic_trainer','model','base_model','original_generate','base_generate','ANSWER_STOP'):
    stale=globals().pop(name,None)
    if stale is not None: del stale
gc.collect(); torch.cuda.empty_cache()
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_enable_fp32_cpu_offload=True)
base_model=AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,device_map='auto',trust_remote_code=False)
base_model.config.use_cache=False
base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True)
lora=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                lora_dropout=.05,bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base_model,lora)
set_peft_model_state_dict(model,load_safetensors(str(SOURCE/'adapter_model.safetensors')),adapter_name='default')
meta=[n for n,p in model.named_parameters() if p.device.type=='meta']
if meta: raise RuntimeError(f'Meta tensors after load: {meta[:5]}')

ANSWER_IDS=tokenizer.encode('</answer>',add_special_tokens=False)
class StopAfterAnswer(StoppingCriteria):
    def __call__(self,input_ids,scores,**kwargs):
        width=len(ANSWER_IDS)
        return torch.tensor([row.numel()>=width and row[-width:].tolist()==ANSWER_IDS
                             for row in input_ids],device=input_ids.device,dtype=torch.bool)
ANSWER_STOP=StoppingCriteriaList([StopAfterAnswer()])
print({'answer_stop_token_ids':ANSWER_IDS,'max_new_tokens':MAX_NEW_TOKENS,'meta_parameters':len(meta)})
'''),
    code(r'''print('===== BUILD FRESH TRAINER =====')
args=GRPOConfig(output_dir=str(OUTPUT),per_device_train_batch_size=1,
    gradient_accumulation_steps=GROUP_SIZE,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},torch_empty_cache_steps=1,
    max_steps=DRY_STEPS,learning_rate=1e-5,lr_scheduler_type='linear',warmup_steps=0,
    bf16=True,num_generations=GROUP_SIZE,generation_batch_size=GROUP_SIZE,num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS,temperature=.8,top_p=.95,beta=.04,entropy_coef=.05,
    logging_strategy='steps',logging_steps=1,disable_tqdm=True,save_strategy='steps',save_steps=5,
    save_total_limit=2,report_to='none',remove_unused_columns=False,disable_dropout=True,
    seed=RUN_SEED,data_seed=RUN_SEED)

candidate_calls=[]
def diagnostic_reward(prompts,completions,**kwargs):
    truths=kwargs.get('ground_truth') or kwargs.get('ground_truths')
    step=LOGICAL_OFFSET+int(globals().get('diagnostic_trainer').state.global_step) if globals().get('diagnostic_trainer') else 500
    texts=[completion_to_text(x) for x in completions]; prompt_texts=[prompt_to_text(x) for x in prompts]
    breakdowns=[score_completion(t,y,step,prompt=p) for t,y,p in zip(texts,truths,prompt_texts)]
    call={'texts':texts,'truths':list(truths),'prompts':prompt_texts,'breakdowns':breakdowns,
          'rewards':[x['total'] for x in breakdowns]}
    candidate_calls.append(call); return call['rewards']

diagnostic_trainer=GRPOTrainer(model=model,reward_funcs=diagnostic_reward,args=args,
    train_dataset=train_dataset,processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE

# Inject the same suffix stop into TRL's actual Transformers generate path.
original_generate=model.generate
def generate_stopped(*a,**kw):
    kw.setdefault('stopping_criteria',ANSWER_STOP)
    return original_generate(*a,**kw)
model.generate=generate_stopped
diagnostic_trainer.model.generate=generate_stopped
print({'fresh_optimizer':True,'fresh_scheduler':True,'steps_per_generation':
       diagnostic_trainer.args.steps_per_generation,'max_steps':DRY_STEPS})
'''),
    code(r'''print('===== BASELINE: SAME 25 HELD-OUT PROMPTS =====')
def generate_greedy(item):
    # GRPO may leave the policy in train mode or with a temporary reference
    # adapter active. Evaluation must explicitly restore the trained policy.
    model.set_adapter('default'); model.eval()
    assert str(model.active_adapter)=='default' and not model.training
    rendered=tokenizer.apply_chat_template([{'role':'user','content':item['prompt']}],
        tokenize=False,add_generation_prompt=True)
    batch=tokenizer(rendered,return_tensors='pt').to(model.device)
    with torch.inference_mode(): out=original_generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
        stopping_criteria=ANSWER_STOP,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(out[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    return {'text':text,'score':score_completion(text,item['ground_truth'],500,prompt=item['prompt']),
            'truth':item['ground_truth']}
baseline=[generate_greedy(x) for x in HELDOUT_POOL]
baseline_accuracy=statistics.fmean(x['score']['r_task']==4.0 for x in baseline)
baseline_structure=statistics.fmean(x['score']['p_structure']==0.0 for x in baseline)
print({'baseline_accuracy':baseline_accuracy,'baseline_strict_structure':baseline_structure})
'''),
    code(r'''print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE =====')
event={'config':{'P_CoT':2.0,'task':{'correct':4.0,'wrong':-.5,'malformed':-5.0},
       'max_new_tokens':256,'stop':'</answer>','grad_breaker':50.0,'kl_breaker':5.0},
       'groups':[],'telemetry':[],'baseline_accuracy':baseline_accuracy,
       'baseline_strict_structure':baseline_structure,'training_started':False}
def save_event():
    tmp=EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(event,indent=2)); tmp.replace(EVENT_LOG)
    if not EVENT_LOG.is_file() or not EVENT_LOG.stat().st_size: raise RuntimeError('Evidence save failed.')
save_event()
base_generate=diagnostic_trainer._generate_and_score_completions
def dynamic_generate(inputs):
    for attempt in range(1,MAX_DYNAMIC_ATTEMPTS+1):
        result=base_generate(inputs); call=candidate_calls[-1]
        correct=sum(x['r_task']==4.0 for x in call['breakdowns'])
        structural=sum(x['p_structure']==0.0 for x in call['breakdowns'])
        reasons=[]
        if correct in (0,GROUP_SIZE): reasons.append('correctness')
        if structural<math.ceil(.25*GROUP_SIZE): reasons.append('structure')
        accepted=not reasons or attempt==MAX_DYNAMIC_ATTEMPTS
        advantages=result['advantages'].detach().float().cpu().tolist()
        record={'attempt':attempt,'accepted':accepted,'fallback':accepted and bool(reasons),
            'rejection_reasons':reasons,'correct_count':correct,'structural_passes':structural,
            'reward_mean':statistics.fmean(call['rewards']),'reward_std':statistics.pstdev(call['rewards']),
            'rewards':call['rewards'],'advantages':advantages,
            'all_finite':all(math.isfinite(x) for x in call['rewards']+advantages),
            'rollouts':[{'prompt':p,'truth':y,'completion':t,'breakdown':b,'advantage':a}
                        for p,y,t,b,a in zip(call['prompts'],call['truths'],call['texts'],call['breakdowns'],advantages)]}
        event['groups'].append(record); save_event()
        if accepted: return result
    raise RuntimeError('Dynamic sampling returned no group.')
diagnostic_trainer._generate_and_score_completions=dynamic_generate

def breaker(grad_norm,kl): return grad_norm>=GRAD_BREAKER or kl>=KL_BREAKER
assert breaker(50.0,0.0) and breaker(0.0,5.0) and not breaker(49.99,4.99)
class Safety(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}; row={'physical_step':int(state.global_step),
            **{k:float(v) for k,v in logs.items() if isinstance(v,(int,float))}}
        event['telemetry'].append(row); save_event()
        grad=float(logs.get('grad_norm',0)); kl=float(logs.get('kl',0))
        if not all(math.isfinite(x) for x in (grad,kl)) or breaker(grad,kl):
            event['hard_stop']={'step':int(state.global_step),'grad_norm':grad,'kl':kl}; save_event()
            control.should_training_stop=True
        return control
diagnostic_trainer.add_callback(Safety())
print('PASSED: synthetic grad_norm/KL circuit-breaker tests.')
'''),
    code(r'''print('===== AUTHORIZED FIVE-STEP DRY-RUN ONLY =====')
event['training_started']=True; save_event()
result=diagnostic_trainer.train()
terminal=int(diagnostic_trainer.state.global_step)
if terminal!=DRY_STEPS: raise RuntimeError(f'Dry-run stopped safely at step {terminal}; inspect {EVENT_LOG}')

post=[generate_greedy(x) for x in HELDOUT_POOL]
post_accuracy=statistics.fmean(x['score']['r_task']==4.0 for x in post)
post_structure=statistics.fmean(x['score']['p_structure']==0.0 for x in post)
accepted=[x for x in event['groups'] if x['accepted']]
accepted_rollouts=[r for g in accepted for r in g['rollouts']]
sampled_structure=statistics.fmean(r['breakdown']['p_structure']==0.0 for r in accepted_rollouts)
nondegenerate=sum(g['reward_std']>1e-8 for g in accepted)
all_ended=statistics.fmean(r['completion'].rstrip().casefold().endswith('</answer>') for r in accepted_rollouts)
accuracy_drop=baseline_accuracy-post_accuracy
gate={'baseline_accuracy':baseline_accuracy,'post_accuracy':post_accuracy,
      'accuracy_drop':accuracy_drop,'baseline_strict_structure':baseline_structure,
      'post_strict_structure':post_structure,'sampled_strict_structure':sampled_structure,
      'nondegenerate_groups':nondegenerate,'accepted_groups':len(accepted),
      'ended_at_answer_rate':all_ended,'hard_stop':event.get('hard_stop'),
      'passed':accuracy_drop<=.10 and post_structure>=.80 and sampled_structure>=.80
               and nondegenerate==len(accepted) and all_ended==1.0 and not event.get('hard_stop')}
event['gate']=gate; event['post_rows']=post; save_event()
# Persist evaluation evidence outside the run directory so checkpoint cleanup
# cannot remove the raw completions again.
AUDIT_OUT=Path('/content/drive/MyDrive/AISI/audits')/f'{OUTPUT.name}_pre_post_evaluation.json'
AUDIT_OUT.parent.mkdir(parents=True,exist_ok=True)
AUDIT_OUT.write_text(json.dumps({'baseline':baseline,'post':post,'gate':gate},indent=2))
print(json.dumps(gate,indent=2))
print({'evaluation_active_adapter':str(model.active_adapter),'evaluation_mode':not model.training,
       'raw_evaluation_evidence':str(AUDIT_OUT)})
print('===== FIVE FULL POST-UPDATE COMPLETIONS =====')
for i,row in enumerate(post[:5],1):
    print('\n'+'-'*100); print({'index':i,'truth':row['truth'],'score':row['score']}); print(row['text'])
for i,g in enumerate(accepted[:3],1):
    print(f'GROUP {i}:',{'reward_mean':g['reward_mean'],'reward_std':g['reward_std'],
          'correct':g['correct_count'],'structural_passes':g['structural_passes'],
          'attempt':g['attempt'],'fallback':g['fallback']})
if not gate['passed']:
    raise RuntimeError('DRY-RUN GATE DID NOT PASS. No real continuation is authorized.')
print('DRY-RUN PASSED. STOP HERE FOR REVIEW; no real continuation is present.')
print('Evidence:',EVENT_LOG)
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
