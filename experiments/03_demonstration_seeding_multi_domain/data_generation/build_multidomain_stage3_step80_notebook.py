from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/multidomain_stage3_step40_to60.ipynb"
OUTPUT = ROOT / "notebooks/multidomain_stage3_step60_to80.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence: {old[:100]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    notebook=json.loads(SOURCE.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code': cell['execution_count']=None; cell['outputs']=[]
    notebook['cells'][0]['source']=(
        '# Multi-domain Stage 3 — natural continuation from step 60 to step 80\n\n'
        'This restores checkpoint 60 with all optimizer/scheduler/Trainer/RNG state, '
        'changes no training setting, saves checkpoint 80, evaluates the same 300 '
        'prompts, applies the registered trajectory rule, and stops.\n').splitlines(True)

    config=''.join(notebook['cells'][2]['source'])
    config=replace_once(config,
        "STEP40_REPORT=RUN_DIR/'step40_report.json'\nSTEP60_REPORT=RUN_DIR/'step60_report.json'\nEVAL_PROGRESS=RUN_DIR/'step60_eval_progress.json'\nMAX_STEPS=150; SOURCE_STEP=40; TARGET_STEP=60; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
        "STEP60_REPORT=RUN_DIR/'step60_report.json'\nSTEP80_REPORT=RUN_DIR/'step80_report.json'\n"
        "EVAL_PROGRESS=RUN_DIR/'step80_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=60; TARGET_STEP=80; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300")
    old="""if not STEP40_REPORT.is_file() or not json.loads(STEP40_REPORT.read_text()).get('step40_complete'):
    raise RuntimeError(f'Missing completed step-40 report: {STEP40_REPORT}')
if STEP60_REPORT.is_file() and json.loads(STEP60_REPORT.read_text()).get('step60_complete'):
    raise RuntimeError(f'Step-60 report already exists: {STEP60_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 40 through 60, found: {RESUME_CHECKPOINT}')
"""
    new="""if not STEP60_REPORT.is_file() or not json.loads(STEP60_REPORT.read_text()).get('step60_complete'):
    raise RuntimeError(f'Missing completed step-60 report: {STEP60_REPORT}')
if STEP80_REPORT.is_file() and json.loads(STEP80_REPORT.read_text()).get('step80_complete'):
    raise RuntimeError(f'Step-80 report already exists: {STEP80_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 60 through 80, found: {RESUME_CHECKPOINT}')
"""
    config=replace_once(config,old,new); notebook['cells'][2]['source']=config.splitlines(True)

    last="""step60_report=json.loads(STEP60_REPORT.read_text())
step60_metrics=step60_report['metrics']
step40_metrics=step60_report['step40_reference']
report={'stage':'multi_domain_stage3_step80','source_checkpoint':str(RESUME_CHECKPOINT),
        'step80_complete':False,'target_step':80,'full_scheduler_horizon':150,
        'step60_reference':step60_metrics,'stage4_authorized':False}

def finish_step80(model,checkpoint):
    metrics=evaluate_first_gate(model)
    fan40=step40_metrics['per_domain']['fan']; fan60=step60_metrics['per_domain']['fan']; fan80=metrics['per_domain']['fan']
    valve80=metrics['per_domain']['valve']; lamp80=metrics['per_domain']['lamp']
    prior_gain=fan60['transition_tracking_rate']-fan40['transition_tracking_rate']
    current_gain=fan80['transition_tracking_rate']-fan60['transition_tracking_rate']
    half_prior_gain=.5*prior_gain
    valve_gap=abs(fan80['transition_tracking_rate']-valve80['transition_tracking_rate'])
    lamp_gap=abs(fan80['transition_tracking_rate']-lamp80['transition_tracking_rate'])
    persistent_large_gap=max(valve_gap,lamp_gap)>=.15
    clearly_slowed=current_gain<half_prior_gain
    if not clearly_slowed:
        decision='fan_still_converging_continue_unchanged'
    elif persistent_large_gap:
        decision='fan_decelerated_with_persistent_gap_prepare_fix_do_not_execute'
    else:
        decision='fan_decelerated_but_gap_no_longer_large_continue_unchanged'
    trajectory={
        'step40_fan_tracking':fan40['transition_tracking_rate'],
        'step60_fan_tracking':fan60['transition_tracking_rate'],
        'step80_fan_tracking':fan80['transition_tracking_rate'],
        'step40_to60_gain_points':100*prior_gain,
        'step60_to80_gain_points':100*current_gain,
        'half_prior_gain_threshold_points':100*half_prior_gain,
        'gain_clearly_slowed_below_half_prior':clearly_slowed,
        'step80_fan_valve_gap_points':100*valve_gap,
        'step80_fan_lamp_gap_points':100*lamp_gap,
        'persistent_gap_at_least_15_points':persistent_large_gap,
        'decision':decision,
    }
    report.update({'step80_complete':True,'checkpoint':str(checkpoint),'metrics':metrics,
                   'fan_tracking_trajectory':trajectory,'stop_reason':'step80_complete_needs_review'})
    if decision=='fan_decelerated_with_persistent_gap_prepare_fix_do_not_execute':
        report['prepared_but_not_executed_fan_fix']={
            'scope':'fan transition rendering only; signatures, tokens, valve/lamp, and hyperparameters unchanged',
            'old':'different from previously (the fan changes to its other mode)',
            'proposed':'different from previously (the fan flips to the opposite state: Running becomes Stopped, and Stopped becomes Running)',
            'recovery_checkpoint':str(checkpoint),'execution_authorized':False}
    atomic_json(STEP80_REPORT,report)
    print('===== STEP 80 STANDARD METRICS ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
    print('===== FAN TRACKING TRAJECTORY DECISION ====='); print(json.dumps(trajectory,indent=2,sort_keys=True))
    if 'prepared_but_not_executed_fan_fix' in report:
        print('===== PREPARED FAN FIX — NOT EXECUTED ====='); print(json.dumps(report['prepared_but_not_executed_fan_fix'],indent=2,sort_keys=True))

class Step80Gate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; atomic_json(STEP80_REPORT,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=TARGET_STEP: control.should_save=True; control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==TARGET_STEP and not report.get('step80_complete'):
            finish_step80(model,checkpoint); control.should_training_stop=True
        return control

runtime_max_steps=(TARGET_STEP if RESUME_STEP==TARGET_STEP else MAX_STEPS)
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=runtime_max_steps,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[Step80Gate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))
if int(trainer.state.global_step)==TARGET_STEP and not report.get('step80_complete'):
    finish_step80(model,TRAINER_DIR/'checkpoint-80')
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'step80_complete':report.get('step80_complete'),
                       'checkpoint':report.get('checkpoint'),'decision':report.get('fan_tracking_trajectory',{}).get('decision'),
                       'stage4_authorized':False})
print('STOP HERE. No fan fix and no training beyond step 80 was executed.')
"""
    notebook['cells'][8]['source']=last.splitlines(True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT); print(f'Wrote {OUTPUT}')


if __name__=='__main__': main()
