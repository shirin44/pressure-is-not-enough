from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/multidomain_stage3_step20_to40.ipynb"
OUTPUT = ROOT / "notebooks/multidomain_stage3_step40_to60.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one occurrence: {old[:100]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"][0]["source"] = (
        "# Multi-domain Stage 3 — natural continuation from step 40 to step 60\n\n"
        "This restores the complete checkpoint-40 state and changes no training "
        "setting. It saves and evaluates checkpoint 60, classifies the registered "
        "fan-gap decision, and stops before taking corrective action.\n"
    ).splitlines(True)

    config = "".join(notebook["cells"][2]["source"])
    config = replace_once(
        config,
        "STEP40_REPORT=RUN_DIR/'step40_report.json'\nEVAL_PROGRESS=RUN_DIR/'step40_eval_progress.json'\nMAX_STEPS=150; SOURCE_STEP=20; TARGET_STEP=40; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
        "STEP40_REPORT=RUN_DIR/'step40_report.json'\nSTEP60_REPORT=RUN_DIR/'step60_report.json'\n"
        "EVAL_PROGRESS=RUN_DIR/'step60_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=40; TARGET_STEP=60; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
    )
    old_guard = """if not EVENT_LOG.is_file() or not json.loads(EVENT_LOG.read_text()).get('first_gate_complete'):
    raise RuntimeError(f'Missing completed step-20 report: {EVENT_LOG}')
if STEP40_REPORT.is_file() and json.loads(STEP40_REPORT.read_text()).get('step40_complete'):
    raise RuntimeError(f'Step-40 report already exists: {STEP40_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 20 through 40, found: {RESUME_CHECKPOINT}')
"""
    new_guard = """if not STEP40_REPORT.is_file() or not json.loads(STEP40_REPORT.read_text()).get('step40_complete'):
    raise RuntimeError(f'Missing completed step-40 report: {STEP40_REPORT}')
if STEP60_REPORT.is_file() and json.loads(STEP60_REPORT.read_text()).get('step60_complete'):
    raise RuntimeError(f'Step-60 report already exists: {STEP60_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 40 through 60, found: {RESUME_CHECKPOINT}')
"""
    config = replace_once(config, old_guard, new_guard)
    notebook["cells"][2]["source"] = config.splitlines(True)

    final_cell = """step40_report=json.loads(STEP40_REPORT.read_text())
step40_metrics=step40_report['metrics']
report={'stage':'multi_domain_stage3_step60','source_checkpoint':str(RESUME_CHECKPOINT),
        'step60_complete':False,'target_step':60,'full_scheduler_horizon':150,
        'step40_reference':step40_metrics,'stage4_authorized':False}

def finish_step60(model,checkpoint):
    metrics=evaluate_first_gate(model)
    fan40=step40_metrics['per_domain']['fan']; valve40=step40_metrics['per_domain']['valve']; lamp40=step40_metrics['per_domain']['lamp']
    fan60=metrics['per_domain']['fan']; valve60=metrics['per_domain']['valve']; lamp60=metrics['per_domain']['lamp']
    gap={
        'step40_fan_tracking':fan40['transition_tracking_rate'],
        'step60_fan_tracking':fan60['transition_tracking_rate'],
        'step40_fan_valve_gap_points':100*abs(fan40['transition_tracking_rate']-valve40['transition_tracking_rate']),
        'step60_fan_valve_gap_points':100*abs(fan60['transition_tracking_rate']-valve60['transition_tracking_rate']),
        'step40_fan_lamp_gap_points':100*abs(fan40['transition_tracking_rate']-lamp40['transition_tracking_rate']),
        'step60_fan_lamp_gap_points':100*abs(fan60['transition_tracking_rate']-lamp60['transition_tracking_rate']),
    }
    gap_closed=(gap['step60_fan_valve_gap_points']<=15 and gap['step60_fan_lamp_gap_points']<=15)
    semantic_bottleneck=(.30<=fan60['transition_tracking_rate']<=.40
                         and valve60['transition_tracking_rate']>.80
                         and lamp60['transition_tracking_rate']>.80)
    if gap_closed:
        decision='gap_closed_continue_unchanged_toward_full_gate'
    elif semantic_bottleneck:
        decision='semantic_ambiguity_confirmed_prepare_fix_do_not_execute'
    else:
        decision='inconclusive_neither_registered_branch_fully_met_stop_for_review'
    comparison={'decision':decision,'gap_closed_to_within_15_points':gap_closed,
                'semantic_bottleneck_condition_met':semantic_bottleneck,**gap,
                'step40_lamp_accuracy':lamp40['physical_final_answer_accuracy'],
                'step60_lamp_accuracy':lamp60['physical_final_answer_accuracy'],
                'step40_lamp_consistency':lamp40['global_mapping_consistency_rate'],
                'step60_lamp_consistency':lamp60['global_mapping_consistency_rate']}
    report.update({'step60_complete':True,'checkpoint':str(checkpoint),'metrics':metrics,
                   'registered_decision':comparison,'stop_reason':'step60_complete_needs_review'})
    if semantic_bottleneck:
        report['prepared_but_not_executed_fan_fix']={
            'scope':'fan rendering only; valve/lamp/data distribution/hyperparameters unchanged',
            'old_different_instruction':'different from previously (the fan changes to its other mode)',
            'proposed_different_instruction':'different from previously (the fan flips to the opposite state: Running becomes Stopped, and Stopped becomes Running)',
            'old_demonstration_reason':'The fan changes mode',
            'proposed_demonstration_reason':'The fan flips to its opposite state',
            'regeneration':'render the same 400 fan signatures and same nonce mappings through the existing 100% semantic verifier',
            'recovery_checkpoint':str(checkpoint),
            'execution_authorized':False}
    atomic_json(STEP60_REPORT,report)
    print('===== STEP 60 STANDARD METRICS ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
    print('===== PRE-REGISTERED STEP 60 DECISION ====='); print(json.dumps(comparison,indent=2,sort_keys=True))
    if 'prepared_but_not_executed_fan_fix' in report:
        print('===== PREPARED FAN FIX — NOT EXECUTED =====')
        print(json.dumps(report['prepared_but_not_executed_fan_fix'],indent=2,sort_keys=True))

class Step60Gate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; atomic_json(STEP60_REPORT,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=TARGET_STEP:
            control.should_save=True; control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==TARGET_STEP and not report.get('step60_complete'):
            finish_step60(model,checkpoint); control.should_training_stop=True
        return control

runtime_max_steps=(TARGET_STEP if RESUME_STEP==TARGET_STEP else MAX_STEPS)
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=runtime_max_steps,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[Step60Gate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))
if int(trainer.state.global_step)==TARGET_STEP and not report.get('step60_complete'):
    finish_step60(model,TRAINER_DIR/'checkpoint-60')
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'step60_complete':report.get('step60_complete'),
                       'checkpoint':report.get('checkpoint'),'decision':report.get('registered_decision',{}).get('decision'),
                       'stage4_authorized':False})
print('STOP HERE. No fan fix and no training beyond step 60 was executed.')
"""
    notebook["cells"][8]["source"] = final_cell.splitlines(True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == '__main__': main()
