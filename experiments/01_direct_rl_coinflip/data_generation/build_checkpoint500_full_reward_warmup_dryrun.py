"""Derive the 8-step LR-warmup dry-run from the validated full-reward notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
BASE=ROOT/'notebooks'/'checkpoint500_corrected_full_reward_dryrun.ipynb'
TARGET=ROOT/'notebooks'/'checkpoint500_corrected_full_reward_warmup8_dryrun.ipynb'

def replace_once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError(f'Expected one replacement, found {text.count(old)}: {old!r}')
    return text.replace(old,new)

nb=json.loads(BASE.read_text())
for cell in nb['cells']:
    cell['outputs']=[]; cell['execution_count']=None

nb['cells'][0]['source']=[
    '# Checkpoint-500 corrected full-reward warmup dry-run\n\n',
    'Eight optimizer updates only. This is byte-for-byte the validated full-reward dry-run except for a fresh LR schedule: target `2e-6`, explicit 5-update linear warmup beginning at `2e-7`, then linear decay through update 8. It stops for review; no real continuation is included.\n'
]

# Constants: only the run length and LR schedule parameters change.
s=''.join(nb['cells'][2]['source'])
s=replace_once(s,"DRY_STEPS=5; LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0",
               "DRY_STEPS=8; LOGICAL_OFFSET=500; GRAD_BREAKER=50.0; KL_BREAKER=5.0\nTARGET_LR=2e-6; WARMUP_UPDATES=5")
s=replace_once(s,"from pathlib import Path", "from pathlib import Path\nfrom torch.optim.lr_scheduler import LambdaLR")
nb['cells'][2]['source']=s.splitlines(True)

# Fresh namespace and exact pre-model schedule preview.
s=''.join(nb['cells'][3]['source'])
s=s.replace('grpo-checkpoint500-corrected-fullreward-dryrun-v',
            'grpo-checkpoint500-corrected-fullreward-warmup8-v')
s=s.replace('checkpoint500_fullreward_dryrun.json',
            'checkpoint500_fullreward_warmup8_dryrun.json')
old="""lr_curve={501:1e-5,503:1e-5*(1-2/DRY_STEPS),505:1e-5*(1-4/DRY_STEPS)}
assert lr_curve[501]>lr_curve[503]>lr_curve[505]>0
print('FRESH 5-STEP LINEAR LR CURVE:',lr_curve)"""
new="""def lr_factor(update_index):
    # update_index is zero-based. Updates 1..5: 0.1 -> 1.0.
    if update_index < WARMUP_UPDATES:
        return 0.1 + 0.9 * update_index / (WARMUP_UPDATES - 1)
    # Updates 6..8 decay after reaching the target at update 5.
    decay_updates = DRY_STEPS - WARMUP_UPDATES
    return max(0.0, (DRY_STEPS - update_index) / decay_updates)

lr_curve={LOGICAL_OFFSET+i+1:TARGET_LR*lr_factor(i) for i in range(DRY_STEPS)}
expected=[2e-7,6.5e-7,1.1e-6,1.55e-6,2e-6,2e-6,4e-6/3,2e-6/3]
assert all(math.isclose(lr_curve[501+i],value,rel_tol=0,abs_tol=1e-15)
           for i,value in enumerate(expected)),lr_curve
assert all(value>0 and value<=TARGET_LR for value in lr_curve.values())
print('FRESH 8-STEP LR CURVE (5-UPDATE WARMUP):',lr_curve)"""
s=replace_once(s,old,new)
nb['cells'][3]['source']=s.splitlines(True)

# Trainer target LR changes; a custom LambdaLR realizes the nonzero warmup.
s=''.join(nb['cells'][8]['source'])
s=replace_once(s,"max_steps=DRY_STEPS,learning_rate=1e-5,lr_scheduler_type='linear',warmup_steps=0,",
               "max_steps=DRY_STEPS,learning_rate=TARGET_LR,lr_scheduler_type='linear',warmup_steps=WARMUP_UPDATES,")
needle="""diagnostic_trainer=GRPOTrainer(model=model,reward_funcs=diagnostic_reward,args=args,
    train_dataset=train_dataset,processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE
"""
replacement="""diagnostic_trainer=GRPOTrainer(model=model,reward_funcs=diagnostic_reward,args=args,
    train_dataset=train_dataset,processing_class=tokenizer)
assert diagnostic_trainer.optimizer is None and diagnostic_trainer.lr_scheduler is None
diagnostic_trainer.create_optimizer()
diagnostic_trainer.lr_scheduler=LambdaLR(
    diagnostic_trainer.optimizer, lr_lambda=lambda scheduler_step: lr_factor(scheduler_step)
)
assert diagnostic_trainer.optimizer is not None and diagnostic_trainer.lr_scheduler is not None
assert math.isclose(diagnostic_trainer.optimizer.param_groups[0]['lr'],2e-7,
                    rel_tol=0,abs_tol=1e-15)
assert int(diagnostic_trainer.args.steps_per_generation)==GROUP_SIZE
"""
s=replace_once(s,needle,replacement)
s=s.replace("'max_steps':DRY_STEPS", "'max_steps':DRY_STEPS,'target_lr':TARGET_LR,'warmup_updates':WARMUP_UPDATES")
nb['cells'][8]['source']=s.splitlines(True)

# Evidence records the schedule; all reward/sampling/safety code is unchanged.
s=''.join(nb['cells'][10]['source'])
s=replace_once(s,"'max_new_tokens':256,'stop':'</answer>'",
               "'max_new_tokens':256,'stop':'</answer>','target_lr':TARGET_LR,'warmup_updates':WARMUP_UPDATES,'lr_curve':lr_curve")
nb['cells'][10]['source']=s.splitlines(True)

s=''.join(nb['cells'][11]['source'])
s=s.replace('AUTHORIZED FIVE-STEP DRY-RUN ONLY','AUTHORIZED EIGHT-STEP WARMUP DRY-RUN ONLY')
s=s.replace('DRY-RUN PASSED. STOP HERE FOR REVIEW; no real continuation is present.',
            'EIGHT-STEP WARMUP DRY-RUN PASSED. STOP HERE FOR REVIEW; no real continuation is present.')
nb['cells'][11]['source']=s.splitlines(True)

import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET); print(TARGET)
