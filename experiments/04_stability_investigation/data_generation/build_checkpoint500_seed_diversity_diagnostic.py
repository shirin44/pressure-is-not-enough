"""Build the checkpoint-500 seed-diversity diagnostic (informational only).

Purely diagnostic: resumes checkpoint 500 adapter weights with a FRESH
optimizer/scheduler (the only resume mode this checkpoint is authorized for;
see src/training/resume_guard.py), reseeds with a run seed distinct from the
20260730 seed used in every prior checkpoint-500 diagnostic, and trains for a
bounded number of steps under the unchanged reward/config. It classifies every
accepted rollout into overt / vacuous / substituted and reports the
distribution. It does not gate on accuracy, does not continue training beyond
N_STEPS, and does not authorize any real continuation.

"overt / vacuous / substituted" is not a label scheme defined anywhere in
src/data/coinflip.py -- only "vacuous" exists as an audit status there. These
three categories are derived here from the existing parse_state_slots() /
literal-token machinery already in coinflip.py:
  - vacuous: no non-empty state-slot tokens at all (matches the existing
    audit_global_state_consistency "vacuous" status)
  - overt: every non-empty state-slot token is a literal Heads/Tails family
    word (heads/tails/head/tail/h/t)
  - substituted: at least one non-empty state-slot token is NOT a literal
    word -- i.e. any attempt at a non-literal encoding, stable or not
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "04_stability_investigation" / "notebooks" / "checkpoint500_seed_diversity_diagnostic.ipynb"
COINFLIP = (ROOT / "src" / "data" / "coinflip.py").read_text()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 seed-diversity diagnostic (informational only)

Resumes checkpoint 500 adapter weights with a fresh optimizer/scheduler
(`weights_only_fresh_schedule`, the only mode this checkpoint is authorized
for), reseeds with `RUN_SEED=20260817` -- distinct from the `20260730` seed
used in every prior checkpoint-500 diagnostic in this repo -- and trains for
`N_STEPS` bounded optimizer updates under the unchanged reward stack and
`entropy_coef=0.05`. Same reward function, same GRPO config, same model.

This does **not** gate on accuracy or structure; it is purely a data point on
whether a different random seed produces different early exploration
diversity (any substituted attempts at all, even unstable ones) in the first
`N_STEPS` steps. It does not continue beyond `N_STEPS` regardless of outcome,
and it does not authorize any real continuation.
"""),
    code("%pip install -q transformers==5.13.1 trl==1.9.2 peft==0.19.1 bitsandbytes==0.50.0 accelerate datasets safetensors\n"),
    code(r'''import gc, importlib.metadata, itertools, json, logging, math, os, random, re, statistics
from collections import Counter
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

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
RUN_SEED=20260817  # intentionally different from the 20260730 diagnostic reference seed
REFERENCE_SEED=20260730
assert RUN_SEED!=REFERENCE_SEED
GROUP_SIZE=8; MAX_DYNAMIC_ATTEMPTS=3; MAX_NEW_TOKENS=256
N_STEPS=50  # informational diagnostic; raise to 100 for more signal if the 50-step read is ambiguous
LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes.autograd._functions').disabled=True

def version(name): return importlib.metadata.version(name)
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4; found {torch.cuda.get_device_name(0)}')
expected={'transformers':'5.13.1','trl':'1.9.2','peft':'0.19.1','bitsandbytes':'0.50.0'}
actual={k:version(k) for k in expected}
if actual!=expected: raise RuntimeError(f'Version mismatch: expected={expected}, actual={actual}')
print({'gpu':torch.cuda.get_device_name(0),'run_seed':RUN_SEED,'reference_seed':REFERENCE_SEED,'n_steps':N_STEPS,**actual})
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
    raise RuntimeError('REFUSED: this diagnostic requires weights-only + fresh schedule.')
print('SOURCE SCHEDULER AUDIT:',source_scheduler)
print('LOAD POLICY:',{'loaded':['adapter weights only'],
      'reinitialized':['optimizer','scheduler','trainer state'],
      'reseeded':['random','numpy','torch','data ordering','GRPO sampling']})

lr_curve={LOGICAL_OFFSET+1:1e-5,LOGICAL_OFFSET+N_STEPS//2:1e-5*(1-(N_STEPS//2)/N_STEPS),LOGICAL_OFFSET+N_STEPS:1e-5*(1/N_STEPS)}
print('FRESH LINEAR LR CURVE (sample points):',lr_curve)

for version_id in range(1,1000):
    OUTPUT=ROOT/f'grpo-checkpoint500-seed-diversity-diagnostic-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG=OUTPUT/'checkpoint500_seed_diversity_diagnostic.json'
'''),
    code(COINFLIP),
    code(r'''print('===== EXPLORATION CLASSIFIER (derived, not a codebase-defined label) =====')
LITERAL_TOKENS={'heads','tails','head','tail','h','t'}
def classify_exploration(text, prompt):
    """overt / vacuous / substituted, derived from parse_state_slots().

    vacuous: no non-empty state-slot tokens at all.
    overt: every non-empty token is a literal Heads/Tails-family word.
    substituted: at least one non-empty token is non-literal (an attempt at
    a non-literal encoding, regardless of whether it is globally stable).
    """
    slots=parse_state_slots(text)
    tokens=[t for _,t in slots if t]
    if not tokens: return 'vacuous'
    if any(t not in LITERAL_TOKENS for t in tokens): return 'substituted'
    return 'overt'

# Self-test before trusting this on real rollouts.
assert classify_exploration('Step 1: x. State: Heads\nStep 2: y. State: Tails','')=='overt'
assert classify_exploration('Step 1: x.\nStep 2: y.','')=='vacuous'
assert classify_exploration('Step 1: x. State: Zoop\nStep 2: y. State: Heads','')=='substituted'
print('PASSED: classifier self-test.')
'''),
    code(r'''print('===== FINAL REWARD/PARSER PREFLIGHT =====')
assert parse_state_slots('Step 1: reasoning. State: Heads.')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads')==[(1,'heads')]
assert parse_state_slots('Step 1: reasoning. State: Heads. anything')==[]
assert parse_state_slots('Step 1: reasoning. State: the')==[]
assert parse_state_slots('Step 1: reasoning.\nState: Heads')==[]
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
print('PASSED: unchanged parser and full-reward invariant (0.85/0.85), same as every prior checkpoint-500 diagnostic.')
'''),
    code(r'''print('===== BUILD DETERMINISTIC DISJOINT DATA (RUN_SEED, not REFERENCE_SEED) =====')
def unique_pool(size,start,excluded=()):
    rows=[]; seen=set(excluded); seed=start
    while len(rows)<size:
        prompt,truth=generate_coinflip_example(3+(seed%6),seed); seed+=1
        if prompt in seen: continue
        seen.add(prompt); rows.append({'prompt':prompt,'ground_truth':truth})
    return rows
TRAIN_POOL=unique_pool(max(200,N_STEPS*GROUP_SIZE//4),RUN_SEED)
HELDOUT_POOL=unique_pool(25,RUN_SEED+1_000_000,{x['prompt'] for x in TRAIN_POOL})
assert {x['prompt'] for x in TRAIN_POOL}.isdisjoint({x['prompt'] for x in HELDOUT_POOL})
train_dataset=Dataset.from_list(TRAIN_POOL).shuffle(seed=RUN_SEED)
print({'train':len(TRAIN_POOL),'heldout':len(HELDOUT_POOL),'disjoint':True,'seed':RUN_SEED})
'''),
    code(r'''print('===== LOAD MODEL: CHECKPOINT-500 WEIGHTS ONLY =====')
for name in ('diagnostic_trainer','model','base_model','original_generate','base_generate','ANSWER_STOP'):
    stale=globals().pop(name,None)
    if stale is not None: del stale
gc.collect(); torch.cuda.empty_cache()
random.seed(RUN_SEED); torch.manual_seed(RUN_SEED); torch.cuda.manual_seed_all(RUN_SEED)
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
    code(r'''print('===== BUILD FRESH TRAINER (same config, same entropy_coef=0.05, new seed) =====')
args=GRPOConfig(output_dir=str(OUTPUT),per_device_train_batch_size=1,
    gradient_accumulation_steps=GROUP_SIZE,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},torch_empty_cache_steps=1,
    max_steps=N_STEPS,learning_rate=1e-5,lr_scheduler_type='linear',warmup_steps=0,
    bf16=True,num_generations=GROUP_SIZE,generation_batch_size=GROUP_SIZE,num_iterations=1,
    max_completion_length=MAX_NEW_TOKENS,temperature=.8,top_p=.95,beta=.04,entropy_coef=.05,
    logging_strategy='steps',logging_steps=1,disable_tqdm=True,save_strategy='steps',save_steps=25,
    save_total_limit=2,report_to='none',remove_unused_columns=False,disable_dropout=True,
    seed=RUN_SEED,data_seed=RUN_SEED)

candidate_calls=[]
def diagnostic_reward(prompts,completions,**kwargs):
    truths=kwargs.get('ground_truth') or kwargs.get('ground_truths')
    step=LOGICAL_OFFSET+int(globals().get('diagnostic_trainer').state.global_step) if globals().get('diagnostic_trainer') else LOGICAL_OFFSET
    texts=[completion_to_text(x) for x in completions]; prompt_texts=[prompt_to_text(x) for x in prompts]
    breakdowns=[score_completion(t,y,step,prompt=p) for t,y,p in zip(texts,truths,prompt_texts)]
    call={'texts':texts,'truths':list(truths),'prompts':prompt_texts,'breakdowns':breakdowns,
          'rewards':[x['total'] for x in breakdowns]}
    candidate_calls.append(call); return call['rewards']

diagnostic_trainer=GRPOTrainer(model=model,reward_funcs=diagnostic_reward,args=args,
    train_dataset=train_dataset,processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE

original_generate=model.generate
def generate_stopped(*a,**kw):
    kw.setdefault('stopping_criteria',ANSWER_STOP)
    return original_generate(*a,**kw)
model.generate=generate_stopped
diagnostic_trainer.model.generate=generate_stopped
print({'fresh_optimizer':True,'fresh_scheduler':True,'run_seed':RUN_SEED,'n_steps':N_STEPS})
'''),
    code(r'''print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE =====')
event={'config':{'P_CoT':2.0,'task':{'correct':4.0,'wrong':-.5,'malformed':-5.0},
       'entropy_coef':.05,'run_seed':RUN_SEED,'reference_seed':REFERENCE_SEED,'n_steps':N_STEPS,
       'max_new_tokens':256,'stop':'</answer>','grad_breaker':50.0,'kl_breaker':5.0},
       'groups':[],'telemetry':[],'training_started':False,
       'purpose':'seed_diversity_diagnostic_informational_only_no_gate'}
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
        classes=[classify_exploration(t,p) for t,p in zip(call['texts'],call['prompts'])]
        record={'attempt':attempt,'accepted':accepted,'fallback':accepted and bool(reasons),
            'rejection_reasons':reasons,'correct_count':correct,'structural_passes':structural,
            'reward_mean':statistics.fmean(call['rewards']),'reward_std':statistics.pstdev(call['rewards']),
            'rewards':call['rewards'],'advantages':advantages,
            'exploration_classes':classes,
            'all_finite':all(math.isfinite(x) for x in call['rewards']+advantages),
            'rollouts':[{'prompt':p,'truth':y,'completion':t,'breakdown':b,'advantage':a,'exploration_class':c}
                        for p,y,t,b,a,c in zip(call['prompts'],call['truths'],call['texts'],call['breakdowns'],advantages,classes)]}
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
print('PASSED: synthetic grad_norm/KL circuit-breaker tests. Safety breakers active; no accuracy/structure gate for this diagnostic.')
'''),
    code(r'''print(f'===== SEED-DIVERSITY DIAGNOSTIC: UP TO {N_STEPS} STEPS, NO CONTINUATION AUTHORIZED =====')
event['training_started']=True; save_event()
result=diagnostic_trainer.train()
terminal=int(diagnostic_trainer.state.global_step)

accepted=[g for g in event['groups'] if g['accepted']]
accepted_rollouts=[r for g in accepted for r in g['rollouts']]
class_counts=Counter(r['exploration_class'] for r in accepted_rollouts)
n=len(accepted_rollouts)
class_rates={k:class_counts.get(k,0)/n for k in ('overt','vacuous','substituted')} if n else {}
substituted_rollouts=[r for r in accepted_rollouts if r['exploration_class']=='substituted']

report={
    'run_seed':RUN_SEED,'reference_seed':REFERENCE_SEED,'entropy_coef':0.05,
    'terminal_step':terminal,'requested_steps':N_STEPS,
    'hard_stop':event.get('hard_stop'),
    'accepted_groups':len(accepted),'total_accepted_rollouts':n,
    'exploration_class_counts':dict(class_counts),
    'exploration_class_rates':class_rates,
    'any_substituted_attempts':len(substituted_rollouts)>0,
    'note':'Informational only. Original-seed reference: across the first ~100 steps of the original 20260730-lineage run and 3200+ audited rollouts overall, the model used literal (overt) or vacuous completions only -- zero verified non-literal (substituted) encodings.',
}
event['report']=report; save_event()
print(json.dumps(report,indent=2))
print('\n===== UP TO 10 SUBSTITUTED-CLASS COMPLETIONS (if any) =====')
for i,r in enumerate(substituted_rollouts[:10],1):
    print('\n'+'-'*100); print({'index':i,'truth':r['truth'],'breakdown':r['breakdown']}); print(r['completion'])
print('\nEvidence:',EVENT_LOG)
print('DIAGNOSTIC COMPLETE. Purely informational -- do not continue this run. No production config changed.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
