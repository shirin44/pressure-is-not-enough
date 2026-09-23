from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/multidomain_stage3_step80_to100_fan_fix.ipynb"
OUTPUT = ROOT / "notebooks/multidomain_stage3_step100_to130.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one occurrence of {old[:100]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"][0]["source"] = (
        "# Multi-domain Stage 3 — unchanged continuation, step 100 to 130\n\n"
        "Loads the newest verified full-state checkpoint in [100, 130], keeps the corrected "
        "fan data and all optimization settings unchanged, and stops after the step-130 gate.\n"
    ).splitlines(True)

    config = "".join(notebook["cells"][2]["source"])
    config = replace_once(
        config,
        "STEP80_REPORT=RUN_DIR/'step80_report.json'\nSTEP100_REPORT=RUN_DIR/'step100_fan_fix_report.json'\n"
        "EVAL_PROGRESS=RUN_DIR/'step100_fan_fix_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=80; TARGET_STEP=100; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
        "STEP100_REPORT=RUN_DIR/'step100_fan_fix_report.json'\nSTEP130_REPORT=RUN_DIR/'step130_report.json'\n"
        "EVAL_PROGRESS=RUN_DIR/'step130_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=100; TARGET_STEP=130; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
    )
    old_guard = """if not STEP80_REPORT.is_file() or not json.loads(STEP80_REPORT.read_text()).get('step80_complete'):
    raise RuntimeError(f'Missing completed step-80 report: {STEP80_REPORT}')
if STEP100_REPORT.is_file() and json.loads(STEP100_REPORT.read_text()).get('step100_complete'):
    raise RuntimeError(f'Step-100 report already exists: {STEP100_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 80 through 100, found: {RESUME_CHECKPOINT}')
"""
    new_guard = """if not STEP100_REPORT.is_file() or not json.loads(STEP100_REPORT.read_text()).get('step100_complete'):
    raise RuntimeError(f'Missing completed step-100 report: {STEP100_REPORT}')
if STEP130_REPORT.is_file() and json.loads(STEP130_REPORT.read_text()).get('step130_complete'):
    completed=json.loads(STEP130_REPORT.read_text())
    raise RuntimeError(f'Step 130 is already complete at {completed.get("checkpoint")}. Inspect the report; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 100 through 130, found: {RESUME_CHECKPOINT}')
"""
    config = replace_once(config, old_guard, new_guard)
    notebook["cells"][2]["source"] = config.splitlines(True)

    final = r'''step100_report=json.loads(STEP100_REPORT.read_text())
step100_metrics=step100_report['metrics']
report={'stage':'multi_domain_stage3_step130_unchanged','source_checkpoint':str(RESUME_CHECKPOINT),
        'step130_complete':False,'target_step':130,'full_scheduler_horizon':150,
        'change':'none; corrected fan wording retained','step100_reference':step100_metrics,
        'stage4_authorized':False}

def finish_step130(model,checkpoint):
    metrics=evaluate_first_gate(model)
    tracked=('physical_final_answer_accuracy','structural_format_rate','transition_tracking_rate',
             'global_mapping_consistency_rate','declared_mapping_adherence_rate',
             'decode_back_specific_accuracy','nonliteral_encoding_rate')
    trajectory={}
    for domain in ('fan','valve','lamp'):
        before=step100_metrics['per_domain'][domain]; after=metrics['per_domain'][domain]
        trajectory[domain]={'changes_points':{key:100*(after[key]-before[key]) for key in tracked},
                            'step100_tracking':before['transition_tracking_rate'],
                            'step130_tracking':after['transition_tracking_rate']}
    fan=metrics['per_domain']['fan']; valve=metrics['per_domain']['valve']; lamp=metrics['per_domain']['lamp']
    gaps={'fan_valve_tracking_gap_points':100*abs(fan['transition_tracking_rate']-valve['transition_tracking_rate']),
          'fan_lamp_tracking_gap_points':100*abs(fan['transition_tracking_rate']-lamp['transition_tracking_rate'])}
    gaps['both_within_15_points']=all(value<=15 for key,value in gaps.items() if key.endswith('_points'))
    pooled=metrics['pooled']
    acceptance={'threshold':.95,
                'per_metric':{key:{'value':pooled[key],'passed':pooled[key]>=.95} for key in tracked},
                'zero_code_copy':pooled['answer_is_code_word_rate']==0}
    acceptance['full_pooled_gate_passed']=(all(v['passed'] for v in acceptance['per_metric'].values())
                                            and acceptance['zero_code_copy'])
    report.update({'step130_complete':True,'checkpoint':str(checkpoint),'metrics':metrics,
                   'step100_to130_trajectory':trajectory,'tracking_gaps':gaps,
                   'pooled_acceptance_gate':acceptance,'stop_reason':'step130_complete_needs_review'})
    atomic_json(STEP130_REPORT,report)
    print('===== STEP 130 STANDARD METRICS ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
    print('===== STEP 100 TO 130 TRAJECTORY ====='); print(json.dumps(trajectory,indent=2,sort_keys=True))
    print('===== TRACKING GAPS ====='); print(json.dumps(gaps,indent=2,sort_keys=True))
    print('===== POOLED 95% ACCEPTANCE GATE ====='); print(json.dumps(acceptance,indent=2,sort_keys=True))

class Step130Gate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        for key in ('loss','grad_norm'):
            if key in (logs or {}) and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'
                atomic_json(STEP130_REPORT,report)
                control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=TARGET_STEP:
            control.should_save=True; control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==TARGET_STEP and not report.get('step130_complete'):
            finish_step130(model,checkpoint); control.should_training_stop=True
        return control

# Preserve the original 150-step scheduler and restore its optimizer/scheduler state.
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=MAX_STEPS,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[Step130Gate()])
print({'resume_from':str(RESUME_CHECKPOINT),'resume_step':RESUME_STEP,'target_step':TARGET_STEP,
       'state_restored':['adapter','optimizer','scheduler','trainer','RNG'],'settings_changed':False})
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))
if int(trainer.state.global_step)==TARGET_STEP and not report.get('step130_complete'):
    finish_step130(model,TRAINER_DIR/'checkpoint-130')
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'step130_complete':report.get('step130_complete'),
                       'checkpoint':report.get('checkpoint'),
                       'full_gate':report.get('pooled_acceptance_gate',{}).get('full_pooled_gate_passed'),
                       'stage4_authorized':False})
print('STOP HERE. No Stage 3.5 or Stage 4 was executed.')
'''
    notebook["cells"][8]["source"] = final.splitlines(True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
