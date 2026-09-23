"""Build per-token diagnostic for the recurrent checkpoint-500 KL spike."""
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
BASE=ROOT/'notebooks'/'checkpoint500_corrected_full_reward_warmup8_dryrun.ipynb'
TARGET=ROOT/'notebooks'/'checkpoint500_warmup_per_token_kl_diagnostic.ipynb'

def once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError(f'Expected one target, found {text.count(old)}: {old[:80]!r}')
    return text.replace(old,new)

nb=json.loads(BASE.read_text())
for c in nb['cells']:
    c['outputs']=[]; c['execution_count']=None
nb['cells'][0]['source']=['# Checkpoint-500 warmup per-token KL diagnostic\n\n',
    'Replays the unchanged warmup dry-run while proactively saving every TRL tensor, token IDs, adapter fingerprints/norms, and RNG states. Circuit breakers remain unchanged. No real continuation is included.\n']

s=''.join(nb['cells'][3]['source'])
s=s.replace('grpo-checkpoint500-corrected-fullreward-warmup8-v','grpo-checkpoint500-warmup-token-diagnostic-v')
s=s.replace('checkpoint500_fullreward_warmup8_dryrun.json','checkpoint500_warmup_token_diagnostic.json')
nb['cells'][3]['source']=s.splitlines(True)

s=''.join(nb['cells'][10]['source'])
s=s.replace("'groups':[],'telemetry':[]", "'groups':[],'telemetry':[],'adapter_updates':[]")
needle="""def save_event():
    tmp=EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(event,indent=2)); tmp.replace(EVENT_LOG)
    if not EVENT_LOG.is_file() or not EVENT_LOG.stat().st_size: raise RuntimeError('Evidence save failed.')
save_event()
base_generate=diagnostic_trainer._generate_and_score_completions
"""
replacement="""def save_event():
    EVENT_LOG.parent.mkdir(parents=True,exist_ok=True)
    tmp=EVENT_LOG.with_suffix('.tmp'); tmp.write_text(json.dumps(event,indent=2)); tmp.replace(EVENT_LOG)
    if not EVENT_LOG.is_file() or not EVENT_LOG.stat().st_size: raise RuntimeError('Evidence save failed.')

def adapter_state_evidence():
    import hashlib
    digest=hashlib.sha256(); squared=0.0; count=0
    for name,param in model.named_parameters():
        if '.default.' not in name: continue
        value=param.detach().float().cpu().contiguous()
        digest.update(name.encode()); digest.update(value.numpy().tobytes())
        squared+=float(value.square().sum()); count+=value.numel()
    return {'sha256':digest.hexdigest(),'l2_norm':math.sqrt(squared),'parameter_count':count}

def rng_evidence():
    return {'cpu':torch.get_rng_state().cpu().tolist(),
            'cuda':[state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]}

save_event()
base_generate=diagnostic_trainer._generate_and_score_completions
"""
s=once(s,needle,replacement)
needle="""        result=base_generate(inputs); call=candidate_calls[-1]
        correct=sum(x['r_task']==4.0 for x in call['breakdowns'])
"""
replacement="""        result=base_generate(inputs); call=candidate_calls[-1]
        tensor_evidence={}
        for key,value in result.items():
            if torch.is_tensor(value) and value.numel() <= 200000:
                cpu=value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu()
                tensor_evidence[key]={'shape':list(cpu.shape),'dtype':str(cpu.dtype),'values':cpu.tolist()}
        correct=sum(x['r_task']==4.0 for x in call['breakdowns'])
"""
s=once(s,needle,replacement)
s=s.replace("'all_finite':all(math.isfinite(x) for x in call['rewards']+advantages),",
            "'all_finite':all(math.isfinite(x) for x in call['rewards']+advantages),\n            'trl_tensor_evidence':tensor_evidence,")
old="""class Safety(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
"""
new="""class Safety(TrainerCallback):
    def on_step_begin(self,args,state,control,**kwargs):
        event['adapter_updates'].append({'target_physical_step':int(state.global_step)+1,
            'before':adapter_state_evidence(),'rng_before':rng_evidence()})
        save_event(); return control
    def on_step_end(self,args,state,control,**kwargs):
        target=int(state.global_step)
        row=next(x for x in reversed(event['adapter_updates']) if x['target_physical_step']==target)
        row['after']=adapter_state_evidence(); row['rng_after']=rng_evidence()
        row['norm_delta']=row['after']['l2_norm']-row['before']['l2_norm']
        row['hash_changed']=row['after']['sha256']!=row['before']['sha256']
        save_event(); return control
    def on_log(self,args,state,control,logs=None,**kwargs):
"""
s=once(s,old,new)
nb['cells'][10]['source']=s.splitlines(True)

s=''.join(nb['cells'][11]['source'])
old="""terminal=int(diagnostic_trainer.state.global_step)
if terminal!=DRY_STEPS: raise RuntimeError(f'Dry-run stopped safely at step {terminal}; inspect {EVENT_LOG}')

post=[generate_greedy(x) for x in HELDOUT_POOL]
"""
new="""terminal=int(diagnostic_trainer.state.global_step)
print('===== TOKEN DIAGNOSTIC CAPTURE SUMMARY =====')
print({'terminal_step':terminal,'hard_stop':event.get('hard_stop'),
       'tensor_keys_by_group':[sorted(g.get('trl_tensor_evidence',{}))
                               for g in event['groups'] if g.get('accepted')],
       'adapter_updates':[{k:v for k,v in row.items() if not k.startswith('rng_')}
                          for row in event.get('adapter_updates',[])]})
if terminal!=DRY_STEPS:
    raise RuntimeError(f'Instrumented run stopped safely at step {terminal}; full evidence: {EVENT_LOG}')

post=[generate_greedy(x) for x in HELDOUT_POOL]
"""
s=once(s,old,new)
s=s.replace('AUTHORIZED EIGHT-STEP WARMUP DRY-RUN ONLY','INSTRUMENTED WARMUP DIAGNOSTIC: MAXIMUM 8 UPDATES')
nb['cells'][11]['source']=s.splitlines(True)

import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET); print(TARGET)
