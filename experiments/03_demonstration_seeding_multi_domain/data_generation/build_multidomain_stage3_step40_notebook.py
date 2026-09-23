from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/multidomain_stage3_first_gate.ipynb"
OUTPUT = ROOT / "notebooks/multidomain_stage3_step20_to40.ipynb"


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
        "# Multi-domain Stage 3 — natural continuation from step 20 to step 40\n\n"
        "This notebook requires the verified step-20 checkpoint, restores model, "
        "optimizer, scheduler, RNG, and Trainer state, then stops after saving and "
        "evaluating checkpoint 40. It uses the identical data and configuration.\n"
    ).splitlines(True)

    config = "".join(notebook["cells"][2]["source"])
    config = replace_once(
        config,
        "TRAINER_DIR=RUN_DIR/'trainer-output'; EVENT_LOG=RUN_DIR/'first_gate_report.json'\nEVAL_PROGRESS=RUN_DIR/'step20_eval_progress.json'\nMAX_STEPS=150; FIRST_GATE_STEP=20; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
        "TRAINER_DIR=RUN_DIR/'trainer-output'; EVENT_LOG=RUN_DIR/'first_gate_report.json'\n"
        "STEP40_REPORT=RUN_DIR/'step40_report.json'\nEVAL_PROGRESS=RUN_DIR/'step40_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=20; TARGET_STEP=40; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
    )
    old_guard = """if EVENT_LOG.is_file() and json.loads(EVENT_LOG.read_text()).get('first_gate_complete'):
    raise RuntimeError(f'First gate is already complete. Inspect {EVENT_LOG}; do not retrain.')
if RESUME_STEP>FIRST_GATE_STEP:
    raise RuntimeError(f'Unexpected checkpoint beyond authorized first gate: {RESUME_CHECKPOINT}')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'resume_checkpoint':str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
       'authorized_stop':FIRST_GATE_STEP,'scheduler_horizon':MAX_STEPS})
"""
    new_guard = """if not EVENT_LOG.is_file() or not json.loads(EVENT_LOG.read_text()).get('first_gate_complete'):
    raise RuntimeError(f'Missing completed step-20 report: {EVENT_LOG}')
if STEP40_REPORT.is_file() and json.loads(STEP40_REPORT.read_text()).get('step40_complete'):
    raise RuntimeError(f'Step-40 report already exists: {STEP40_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 20 through 40, found: {RESUME_CHECKPOINT}')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'resume_checkpoint':str(RESUME_CHECKPOINT),
       'loaded_state':['adapter','optimizer','scheduler','trainer','RNG'],
       'authorized_stop':TARGET_STEP,'scheduler_horizon':MAX_STEPS})
"""
    config = replace_once(config, old_guard, new_guard)
    notebook["cells"][2]["source"] = config.splitlines(True)

    last = """step20_report=json.loads(EVENT_LOG.read_text())
step20_metrics=step20_report['evaluation_history'][-1]
report={'stage':'multi_domain_stage3_step40','source_checkpoint':str(RESUME_CHECKPOINT),
        'step40_complete':False,'target_step':40,'full_scheduler_horizon':150,
        'step20_reference':step20_metrics,'stage4_authorized':False}

def training_examples_for_domain(domain,count=8):
    rows=[row for row in training_examples if row.domain==domain]
    rows=sorted(rows,key=lambda row:(len(row.operations),row.example_id))
    indices=np.linspace(0,len(rows)-1,count,dtype=int)
    return [{'example_id':rows[i].example_id,'operation_count':len(rows[i].operations),
             'initial_state':rows[i].initial_state,'prompt':rows[i].prompt,
             'demonstration':rows[i].demonstration} for i in indices]

def finish_step40(model,checkpoint):
    metrics=evaluate_first_gate(model)
    fan20=step20_metrics['per_domain']['fan']; valve20=step20_metrics['per_domain']['valve']
    fan40=metrics['per_domain']['fan']; valve40=metrics['per_domain']['valve']
    gaps={
        'step20_structure_gap_points':100*abs(valve20['structural_format_rate']-fan20['structural_format_rate']),
        'step40_structure_gap_points':100*abs(valve40['structural_format_rate']-fan40['structural_format_rate']),
        'step20_tracking_gap_points':100*abs(valve20['transition_tracking_rate']-fan20['transition_tracking_rate']),
        'step40_tracking_gap_points':100*abs(valve40['transition_tracking_rate']-fan40['transition_tracking_rate']),
    }
    gaps['narrowed_to_within_15_points']=(gaps['step40_structure_gap_points']<=15
                                         and gaps['step40_tracking_gap_points']<=15)
    lamp20=step20_metrics['per_domain']['lamp']; lamp40=metrics['per_domain']['lamp']
    lamp_comparison={
        'step20_accuracy':lamp20['physical_final_answer_accuracy'],
        'step40_accuracy':lamp40['physical_final_answer_accuracy'],
        'accuracy_change_points':100*(lamp40['physical_final_answer_accuracy']-lamp20['physical_final_answer_accuracy']),
        'step20_global_consistency':lamp20['global_mapping_consistency_rate'],
        'step40_global_consistency':lamp40['global_mapping_consistency_rate'],
        'consistency_change_points':100*(lamp40['global_mapping_consistency_rate']-lamp20['global_mapping_consistency_rate']),
        'same_100_prompt_evaluation_set':True,
    }
    report.update({'step40_complete':True,'checkpoint':str(checkpoint),'metrics':metrics,
                   'fan_valve_gap_comparison':gaps,'lamp_transfer_comparison':lamp_comparison,
                   'stop_reason':'step40_complete_needs_review'})
    if not gaps['narrowed_to_within_15_points']:
        report['domain_asymmetry_audit']={
            'reason':'fan/valve structure and tracking did not both close to within 15 points',
            'fan_examples':training_examples_for_domain('fan'),
            'valve_examples':training_examples_for_domain('valve'),
            'generator_facts':{
                'examples_per_domain':400,'length_range':[3,8],
                'same_signature_generation_algorithm':True,
                'same_nonce_generator':True,'exact_training_interleaving':True,
            }}
    atomic_json(STEP40_REPORT,report)
    print('===== STEP 40 STANDARD METRICS ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
    print('===== FAN/VALVE GAP COMPARISON ====='); print(json.dumps(gaps,indent=2,sort_keys=True))
    print('===== LAMP TRANSFER COMPARISON ====='); print(json.dumps(lamp_comparison,indent=2,sort_keys=True))
    if 'domain_asymmetry_audit' in report:
        print('===== SIDE-BY-SIDE DOMAIN ASYMMETRY AUDIT =====')
        print(json.dumps(report['domain_asymmetry_audit'],indent=2,sort_keys=True))

class Step40Gate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; atomic_json(STEP40_REPORT,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=TARGET_STEP:
            control.should_save=True; control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==TARGET_STEP and not report.get('step40_complete'):
            finish_step40(model,checkpoint); control.should_training_stop=True
        return control

runtime_max_steps=(TARGET_STEP if RESUME_STEP==TARGET_STEP else MAX_STEPS)
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=runtime_max_steps,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[Step40Gate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))
if int(trainer.state.global_step)==TARGET_STEP and not report.get('step40_complete'):
    finish_step40(model,TRAINER_DIR/'checkpoint-40')
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'step40_complete':report.get('step40_complete'),
                       'checkpoint':report.get('checkpoint'),'stage4_authorized':False})
print('STOP HERE. Do not continue beyond step 40 before review.')
"""
    notebook["cells"][8]["source"] = last.splitlines(True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == '__main__': main()
