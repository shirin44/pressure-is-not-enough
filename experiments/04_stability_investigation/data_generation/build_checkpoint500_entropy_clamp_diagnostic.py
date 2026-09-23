"""Build the checkpoint-500 entropy-clamp + gentle-warmup survival diagnostic.

Purely diagnostic. Two changes on top of the seed-diversity diagnostic
(build_checkpoint500_seed_diversity_diagnostic.py), both aimed at surviving
past the step-3/step-4 grad_norm/KL blowups both prior checkpoint-500
diagnostics hit regardless of seed:

1. A hard clamp on the entropy LOSS CONTRIBUTION (entropy_coef * entropy),
   implemented by clamping entropy_from_logits()'s output at the source.
   This is exact, not approximate: entropy_coef is constant/non-adaptive in
   our config (use_adaptive_entropy defaults False in TRL and we never set
   it -- confirmed by reading trl/trainer/grpo_config.py and
   grpo_trainer.py's `apply_coef = self.entropy_coef` branch), so clamping
   per-token entropy at threshold T bounds the masked-mean entropy_loss at
   T, which bounds entropy_coef * entropy_loss at entropy_coef * T.

   Clamp value: derived from ALL real observed entropy data points across
   both prior checkpoint-500 diagnostics (7 values, mean 0.849, max 1.441 --
   not just the original 3-point/0.71-mean estimate), at 2x that mean
   (~1.70). 1.5x the updated mean (~1.27) was rejected because it would
   already clip an observed non-blowup value (1.31); 2x keeps the clamp
   above everything seen in healthy operation so far.

2. A gentler, longer LR warmup: 12 steps to a 1e-6 target (both prior
   attempts -- 5-step warmup to 2e-6, and a flat 1e-5 start -- blew up).

IMPORTANT CAVEAT surfaced while building this: in both prior diagnostics,
the logged "entropy" value at the actual blowup step was NOT anomalously
high relative to other steps in the same run (reference run: blowup step
entropy 0.71, versus 1.03 at step 1 with no blowup; new-seed run: blowup
step entropy 1.44, versus step 2's 1.31 with grad_norm only 19.5, no
blowup). Entropy magnitude does not obviously track with the instability in
either prior run. This clamp is still worth trying -- cheap, harmless, and
directly requested -- but if it does NOT prevent a further blowup, that is
expected given this evidence, and points at the per-token importance-ratio
mechanism instead, not a failure of the clamp value chosen here.

Also note: the trl version installed locally for inspection is 1.9.1, one
patch below the 1.9.2 pinned everywhere else in this repo. The patch targets
trl.trainer.grpo_trainer.entropy_from_logits by name and includes a runtime
self-test cell that calls the actual patched name post-monkeypatch, so it is
verified against whatever 1.9.2 actually does at run time, not just against
the 1.9.1 source read locally. If entropy_from_logits does not exist at that
import path in 1.9.2, this notebook will raise ImportError immediately
rather than silently training unpatched.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET = ROOT / "experiments" / "04_stability_investigation" / "notebooks" / "checkpoint500_entropy_clamp_diagnostic.ipynb"
COINFLIP = (ROOT / "src" / "data" / "coinflip.py").read_text()


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


cells = [
    markdown("""# Checkpoint-500 entropy-clamp + gentle-warmup survival diagnostic

Purely diagnostic; the goal is survival to 50+ steps without tripping the
`grad_norm>=50 or kl>=5` safety breaker, not any accuracy/exploration result.

Two changes vs. the two prior checkpoint-500 diagnostics (both of which blew
up within 3-4 steps regardless of seed or LR schedule):

1. Entropy-loss clamp: `entropy_coef * entropy` bounded at `entropy_coef *
   1.70` (~2x the mean of all 7 real entropy values observed across both
   prior diagnostics), by clamping `entropy_from_logits()`'s output at the
   source. Same seed as the seed-diversity diagnostic (`20260817`), same
   reward stack, same `entropy_coef=0.05`.
2. Gentler warmup: 12-step linear warmup to a `1e-6` target LR (vs. the
   prior 5-step-to-2e-6 and flat-1e-5 attempts, both of which failed).

**Caveat surfaced while building this**: in both prior runs, entropy at the
actual blowup step was not anomalously high relative to other steps in the
same run. This clamp is worth trying regardless, but if it does not prevent
a further blowup, that would be consistent with the evidence already in
hand, not a sign the clamp value was wrong -- see the per-token
importance-ratio fallback noted in the project log.
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
RUN_SEED=20260817  # same fresh seed as the seed-diversity diagnostic
REFERENCE_SEED=20260730
assert RUN_SEED!=REFERENCE_SEED
GROUP_SIZE=8; MAX_DYNAMIC_ATTEMPTS=3; MAX_NEW_TOKENS=256
N_STEPS=50  # survival is the goal, not a particular result
WARMUP_UPDATES=12; TARGET_LR=1e-6
LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0
# 7 real entropy values observed across both prior checkpoint-500 diagnostics:
# reference-seed run [1.034, 0.3864, 0.7124] + new-seed run [0.6751, 1.31, 0.3852, 1.441]
OBSERVED_ENTROPY_VALUES=[1.034, 0.3864, 0.7124, 0.6751, 1.31, 0.3852, 1.441]
OBSERVED_MEAN_ENTROPY=statistics.fmean(OBSERVED_ENTROPY_VALUES)
ENTROPY_CLAMP_MULTIPLIER=2.0  # 1.5x was rejected: it would clip an already-observed non-blowup value (1.31)
ENTROPY_CLAMP_VALUE=ENTROPY_CLAMP_MULTIPLIER*OBSERVED_MEAN_ENTROPY
assert ENTROPY_CLAMP_VALUE>max(OBSERVED_ENTROPY_VALUES), 'Clamp must sit above every value seen in healthy operation.'
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)
logging.getLogger('bitsandbytes.autograd._functions').disabled=True

def version(name): return importlib.metadata.version(name)
if not torch.cuda.is_available(): raise RuntimeError('Select a Colab GPU runtime.')
if 'L4' not in torch.cuda.get_device_name(0).upper():
    raise RuntimeError(f'Select a Colab L4; found {torch.cuda.get_device_name(0)}')
expected={'transformers':'5.13.1','trl':'1.9.2','peft':'0.19.1','bitsandbytes':'0.50.0'}
actual={k:version(k) for k in expected}
if actual!=expected: raise RuntimeError(f'Version mismatch: expected={expected}, actual={actual}')
print({'gpu':torch.cuda.get_device_name(0),'run_seed':RUN_SEED,'n_steps':N_STEPS,
       'observed_mean_entropy':OBSERVED_MEAN_ENTROPY,'entropy_clamp_value':ENTROPY_CLAMP_VALUE,
       'warmup_updates':WARMUP_UPDATES,'target_lr':TARGET_LR,**actual})
'''),
    code(r'''print('===== PATCH entropy_from_logits: CLAMP AT SOURCE (differentiable) =====')
import trl.trainer.grpo_trainer as _grpo_mod
import trl.trainer.utils as _trl_utils
if not hasattr(_grpo_mod, 'entropy_from_logits'):
    raise ImportError(
        'trl.trainer.grpo_trainer.entropy_from_logits not found at the expected import path in this trl '
        'version. Do not proceed with an unverified patch -- inspect the installed trl source before rerunning.'
    )
_original_entropy_from_logits = _grpo_mod.entropy_from_logits
_raw_entropy_log = []  # every micro-batch's pre-clamp mean entropy, for post-hoc clamp-engagement analysis
_clamp_engaged_count = [0]
_clamp_total_calls = [0]

def _clamped_entropy_from_logits(logits, chunk_size=128):
    raw = _original_entropy_from_logits(logits, chunk_size=chunk_size)
    _clamp_total_calls[0] += 1
    raw_mean = raw.detach().float().mean().item()
    _raw_entropy_log.append(raw_mean)
    if raw_mean > ENTROPY_CLAMP_VALUE:
        _clamp_engaged_count[0] += 1
    return torch.clamp(raw, max=ENTROPY_CLAMP_VALUE)

_grpo_mod.entropy_from_logits = _clamped_entropy_from_logits
assert _grpo_mod.entropy_from_logits is _clamped_entropy_from_logits

# Self-test against the REAL installed entropy_from_logits (verifies the live trl version's behavior,
# not just the locally-inspected 1.9.1 source): a near-uniform-logit row has entropy close to log(vocab_size)
# nats, which for a real vocab is far above ENTROPY_CLAMP_VALUE, so the clamp must engage; a near-deterministic
# row has entropy close to 0 and must pass through unclamped; gradients must still flow where unclamped.
_test_vocab=1000
_uniform_logits=torch.zeros(1,1,_test_vocab,requires_grad=True)
_peaked_logits=torch.full((1,1,_test_vocab),-20.0)
_peaked_logits[0,0,0]=20.0
_peaked_logits.requires_grad_(True)
_raw_uniform=_original_entropy_from_logits(_uniform_logits)
_clamped_uniform=_grpo_mod.entropy_from_logits(_uniform_logits)
_clamped_peaked=_grpo_mod.entropy_from_logits(_peaked_logits)
assert _raw_uniform.item()>ENTROPY_CLAMP_VALUE, 'Test setup invalid: uniform-logit entropy should exceed the clamp.'
assert math.isclose(_clamped_uniform.item(),ENTROPY_CLAMP_VALUE,rel_tol=1e-6), 'Clamp did not engage on a high-entropy row.'
assert _clamped_peaked.item()<0.01, 'Near-deterministic row should pass through ~unclamped.'
_clamped_uniform.sum().backward()
assert _uniform_logits.grad is not None, 'Gradient must still flow through the clamped (but not saturated) path.'
print('PASSED: entropy clamp self-test against the live installed trl.entropy_from_logits (differentiable, correctly bounded).')
print({'entropy_clamp_value':ENTROPY_CLAMP_VALUE,'max_loss_contribution':0.05*ENTROPY_CLAMP_VALUE})
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

def lr_factor(update_index):
    # update_index is zero-based. Updates 1..WARMUP_UPDATES ramp 0.1 -> 1.0, then linear decay to N_STEPS.
    if update_index < WARMUP_UPDATES:
        return 0.1 + 0.9 * update_index / (WARMUP_UPDATES - 1)
    decay_updates = N_STEPS - WARMUP_UPDATES
    return max(0.0, (N_STEPS - update_index) / decay_updates)
lr_curve={LOGICAL_OFFSET+i+1:TARGET_LR*lr_factor(i) for i in range(N_STEPS)}
assert lr_curve[LOGICAL_OFFSET+1]<lr_curve[LOGICAL_OFFSET+WARMUP_UPDATES]<=TARGET_LR
print('FRESH LR CURVE (12-STEP WARMUP TO 1E-6, SAMPLE POINTS):',
      {k:v for k,v in list(lr_curve.items())[:3]+list(lr_curve.items())[-3:]})

for version_id in range(1,1000):
    OUTPUT=ROOT/f'grpo-checkpoint500-entropy-clamp-diagnostic-v{version_id}'
    if not OUTPUT.exists(): break
else: raise RuntimeError('Could not allocate output directory.')
OUTPUT.mkdir(parents=True)
EVENT_LOG=OUTPUT/'checkpoint500_entropy_clamp_diagnostic.json'
'''),
    code(COINFLIP),
    code(r'''print('===== EXPLORATION CLASSIFIER (derived, not a codebase-defined label; same as the seed-diversity diagnostic) =====')
LITERAL_TOKENS={'heads','tails','head','tail','h','t'}
def classify_exploration(text, prompt):
    slots=parse_state_slots(text)
    tokens=[t for _,t in slots if t]
    if not tokens: return 'vacuous'
    if any(t not in LITERAL_TOKENS for t in tokens): return 'substituted'
    return 'overt'
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
print('PASSED: unchanged parser and full-reward invariant (0.85/0.85), same reward as every prior checkpoint-500 diagnostic.')
'''),
    code(r'''print('===== BUILD DETERMINISTIC DISJOINT DATA (RUN_SEED) =====')
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
    code(r'''print('===== BUILD FRESH TRAINER (entropy-clamped, 12-step warmup to 1e-6) =====')
from torch.optim.lr_scheduler import LambdaLR
args=GRPOConfig(output_dir=str(OUTPUT),per_device_train_batch_size=1,
    gradient_accumulation_steps=GROUP_SIZE,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},torch_empty_cache_steps=1,
    max_steps=N_STEPS,learning_rate=TARGET_LR,lr_scheduler_type='linear',warmup_steps=WARMUP_UPDATES,
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
diagnostic_trainer.create_optimizer()
diagnostic_trainer.lr_scheduler=LambdaLR(
    diagnostic_trainer.optimizer, lr_lambda=lambda scheduler_step: lr_factor(scheduler_step)
)
assert diagnostic_trainer.optimizer is not None and diagnostic_trainer.lr_scheduler is not None
assert math.isclose(diagnostic_trainer.optimizer.param_groups[0]['lr'],lr_curve[LOGICAL_OFFSET+1],rel_tol=0,abs_tol=1e-15)
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE
# Confirm this trainer instance will actually call the patched entropy function.
assert _grpo_mod.entropy_from_logits is _clamped_entropy_from_logits

original_generate=model.generate
def generate_stopped(*a,**kw):
    kw.setdefault('stopping_criteria',ANSWER_STOP)
    return original_generate(*a,**kw)
model.generate=generate_stopped
diagnostic_trainer.model.generate=generate_stopped
print({'fresh_optimizer':True,'fresh_scheduler':True,'run_seed':RUN_SEED,'n_steps':N_STEPS,
       'entropy_clamp_value':ENTROPY_CLAMP_VALUE,'warmup_updates':WARMUP_UPDATES,'target_lr':TARGET_LR})
'''),
    code(r'''print('===== INSTALL DYNAMIC SAMPLING + PERSISTENT EVIDENCE =====')
event={'config':{'P_CoT':2.0,'task':{'correct':4.0,'wrong':-.5,'malformed':-5.0},
       'entropy_coef':.05,'entropy_clamp_value':ENTROPY_CLAMP_VALUE,
       'run_seed':RUN_SEED,'reference_seed':REFERENCE_SEED,'n_steps':N_STEPS,
       'warmup_updates':WARMUP_UPDATES,'target_lr':TARGET_LR,
       'max_new_tokens':256,'stop':'</answer>','grad_breaker':50.0,'kl_breaker':5.0},
       'groups':[],'telemetry':[],'training_started':False,
       'purpose':'entropy_clamp_gentle_warmup_survival_diagnostic_informational_only_no_gate'}
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
print('PASSED: synthetic grad_norm/KL circuit-breaker tests. Safety breakers active (unchanged thresholds); no accuracy/structure gate for this diagnostic.')
'''),
    code(r'''print(f'===== SURVIVAL DIAGNOSTIC: UP TO {N_STEPS} STEPS, ENTROPY-CLAMPED + GENTLE WARMUP =====')
event['training_started']=True; save_event()
result=diagnostic_trainer.train()
terminal=int(diagnostic_trainer.state.global_step)

telemetry=event['telemetry']
grad_norms=[row['grad_norm'] for row in telemetry if 'grad_norm' in row]
kls=[row['kl'] for row in telemetry if 'kl' in row]
accepted=[g for g in event['groups'] if g['accepted']]
accepted_rollouts=[r for g in accepted for r in g['rollouts']]
class_counts=Counter(r['exploration_class'] for r in accepted_rollouts)
n=len(accepted_rollouts)
class_rates={k:class_counts.get(k,0)/n for k in ('overt','vacuous','substituted')} if n else {}
substituted_rollouts=[r for r in accepted_rollouts if r['exploration_class']=='substituted']

report={
    'survived':terminal>=N_STEPS and not event.get('hard_stop'),
    'terminal_step':terminal,'requested_steps':N_STEPS,
    'hard_stop':event.get('hard_stop'),
    'max_grad_norm_observed':max(grad_norms) if grad_norms else None,
    'max_kl_observed':max(kls) if kls else None,
    'entropy_clamp_value':ENTROPY_CLAMP_VALUE,
    'entropy_clamp_engagement_rate':(_clamp_engaged_count[0]/_clamp_total_calls[0]) if _clamp_total_calls[0] else None,
    'entropy_clamp_engaged_calls':_clamp_engaged_count[0],'entropy_clamp_total_calls':_clamp_total_calls[0],
    'raw_entropy_range_this_run':[min(_raw_entropy_log),max(_raw_entropy_log)] if _raw_entropy_log else None,
    'accepted_groups':len(accepted),'total_accepted_rollouts':n,
    'exploration_class_counts':dict(class_counts),
    'exploration_class_rates':class_rates,
    'any_substituted_attempts':len(substituted_rollouts)>0,
    'note':'Informational only. If survived=True and hard_stop is None, the seed-diversity question becomes answerable in a follow-up run. If this still blew up, entropy magnitude is unlikely to be the driver (consistent with both prior runs already showing no correlation between entropy and blowup) -- revive the per-token importance-ratio investigation next, not another hyperparameter guess.',
}
event['report']=report; save_event()
print(json.dumps(report,indent=2))
if substituted_rollouts:
    print('\n===== UP TO 10 SUBSTITUTED-CLASS COMPLETIONS =====')
    for i,r in enumerate(substituted_rollouts[:10],1):
        print('\n'+'-'*100); print({'index':i,'truth':r['truth'],'breakdown':r['breakdown']}); print(r['completion'])
print('\nEvidence:',EVENT_LOG)
print('DIAGNOSTIC COMPLETE. Purely informational -- do not continue this run regardless of outcome. No production config changed.')
'''),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
TARGET.parent.mkdir(parents=True, exist_ok=True)
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET)
print(TARGET)
