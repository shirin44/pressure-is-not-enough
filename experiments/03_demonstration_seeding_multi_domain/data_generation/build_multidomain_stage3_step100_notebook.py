from __future__ import annotations

import json
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
SOURCE = ROOT / "notebooks/multidomain_stage3_step60_to80.ipynb"
OUTPUT = ROOT / "notebooks/multidomain_stage3_step80_to100_fan_fix.ipynb"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence: {old[:100]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    notebook=json.loads(SOURCE.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code': cell['execution_count']=None; cell['outputs']=[]
    notebook['cells'][0]['source']=(
        '# Multi-domain Stage 3 — authorized fan-only correction, step 80 to 100\n\n'
        'This restores checkpoint 80 and changes only fan transition wording. '
        'Signatures, nonce mappings, ordering, valve/lamp records, optimization, and '
        'the 150-step scheduler remain unchanged. It stops after the step-100 report.\n').splitlines(True)

    # Continuation notebooks contain an embedded, self-contained generator.
    # Always refresh it from the repository source when a phase intentionally
    # changes generator behavior; otherwise the call site and embedded function
    # signature can silently drift apart.
    generator_source=(ROOT/'src/data/multidomain_seed.py').read_text().replace(
        'from __future__ import annotations\n\n','',1)
    if 'corrected_fan_wording: bool = False' not in generator_source:
        raise RuntimeError('Current generator lacks the authorized fan-wording option.')
    notebook['cells'][3]['source']=generator_source.splitlines(True)

    config=''.join(notebook['cells'][2]['source'])
    config=replace_once(config,
        "STEP60_REPORT=RUN_DIR/'step60_report.json'\nSTEP80_REPORT=RUN_DIR/'step80_report.json'\nEVAL_PROGRESS=RUN_DIR/'step80_eval_progress.json'\nMAX_STEPS=150; SOURCE_STEP=60; TARGET_STEP=80; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300",
        "STEP80_REPORT=RUN_DIR/'step80_report.json'\nSTEP100_REPORT=RUN_DIR/'step100_fan_fix_report.json'\n"
        "EVAL_PROGRESS=RUN_DIR/'step100_fan_fix_eval_progress.json'\n"
        "MAX_STEPS=150; SOURCE_STEP=80; TARGET_STEP=100; MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300")
    old="""if not STEP60_REPORT.is_file() or not json.loads(STEP60_REPORT.read_text()).get('step60_complete'):
    raise RuntimeError(f'Missing completed step-60 report: {STEP60_REPORT}')
if STEP80_REPORT.is_file() and json.loads(STEP80_REPORT.read_text()).get('step80_complete'):
    raise RuntimeError(f'Step-80 report already exists: {STEP80_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 60 through 80, found: {RESUME_CHECKPOINT}')
"""
    new="""if not STEP80_REPORT.is_file() or not json.loads(STEP80_REPORT.read_text()).get('step80_complete'):
    raise RuntimeError(f'Missing completed step-80 report: {STEP80_REPORT}')
if STEP100_REPORT.is_file() and json.loads(STEP100_REPORT.read_text()).get('step100_complete'):
    raise RuntimeError(f'Step-100 report already exists: {STEP100_REPORT}. Inspect it; do not retrain.')
if not SOURCE_STEP<=RESUME_STEP<=TARGET_STEP:
    raise RuntimeError(f'Expected a valid checkpoint from step 80 through 100, found: {RESUME_CHECKPOINT}')
"""
    config=replace_once(config,old,new); notebook['cells'][2]['source']=config.splitlines(True)

    data=''.join(notebook['cells'][4]['source'])
    data=replace_once(data,
        'training_examples,evaluation_sets=generate_multidomain_dataset(SEED)',
        'training_examples,evaluation_sets=generate_multidomain_dataset(SEED,corrected_fan_wording=True)')
    data += """
old_training,old_evaluations=generate_multidomain_dataset(SEED,corrected_fan_wording=False)
assert len(old_training)==len(training_examples)==800
localized_changes=0
for old,new in zip(old_training,training_examples):
    assert (old.example_id,old.domain,old.initial_state,old.operations,old.expected_states,
            old.token_for_first_state,old.token_for_second_state)==(
            new.example_id,new.domain,new.initial_state,new.operations,new.expected_states,
            new.token_for_first_state,new.token_for_second_state)
    if old.domain=='valve': assert old==new
    elif old.prompt!=new.prompt: localized_changes+=1
assert old_evaluations['valve']==evaluation_sets['valve']
assert old_evaluations['lamp']==evaluation_sets['lamp']
assert localized_changes>0
print({'fan_only_wording_changes':localized_changes,'valve_records_unchanged':True,
       'lamp_eval_records_unchanged':True,'signatures_tokens_order_unchanged':True})
"""
    notebook['cells'][4]['source']=data.splitlines(True)

    last="""step80_report=json.loads(STEP80_REPORT.read_text())
step80_metrics=step80_report['metrics']
step60_metrics=step80_report['step60_reference']
step40_metrics=json.loads(STEP60_REPORT.read_text())['step40_reference'] if 'STEP60_REPORT' in globals() else None
report={'stage':'multi_domain_stage3_step100_fan_fix','source_checkpoint':str(RESUME_CHECKPOINT),
        'step100_complete':False,'target_step':100,'full_scheduler_horizon':150,
        'change':'fan transition wording only','step80_reference':step80_metrics,
        'stage4_authorized':False}

def finish_step100(model,checkpoint):
    metrics=evaluate_first_gate(model)
    fan80=step80_metrics['per_domain']['fan']; fan100=metrics['per_domain']['fan']
    valve80=step80_metrics['per_domain']['valve']; valve100=metrics['per_domain']['valve']
    lamp80=step80_metrics['per_domain']['lamp']; lamp100=metrics['per_domain']['lamp']
    fan_gain=fan100['transition_tracking_rate']-fan80['transition_tracking_rate']
    trajectory={
        'step40_to60_fan_gain_points':31.0,
        'step60_to80_fan_gain_points':10.0,
        'step80_to100_fan_gain_points':100*fan_gain,
        'step100_fan_tracking':fan100['transition_tracking_rate'],
        'step100_fan_consistency':fan100['global_mapping_consistency_rate'],
        'step100_fan_mapping':fan100['declared_mapping_adherence_rate'],
        'wording_fix_response_at_least_8_points':fan_gain>=.08,
        'step100_fan_valve_tracking_gap_points':100*abs(fan100['transition_tracking_rate']-valve100['transition_tracking_rate']),
        'step100_fan_lamp_tracking_gap_points':100*abs(fan100['transition_tracking_rate']-lamp100['transition_tracking_rate']),
        'both_tracking_gaps_within_15_points':(abs(fan100['transition_tracking_rate']-valve100['transition_tracking_rate'])<=.15
                                               and abs(fan100['transition_tracking_rate']-lamp100['transition_tracking_rate'])<=.15),
    }
    tracked=('physical_final_answer_accuracy','structural_format_rate','transition_tracking_rate',
             'global_mapping_consistency_rate','declared_mapping_adherence_rate',
             'decode_back_specific_accuracy','nonliteral_encoding_rate')
    nonregression={}
    for domain,before,after in (('valve',valve80,valve100),('lamp',lamp80,lamp100)):
        changes={key:100*(after[key]-before[key]) for key in tracked}
        nonregression[domain]={'changes_points':changes,
                               'any_drop_over_5_points':any(value < -5 for value in changes.values())}
    pooled=metrics['pooled']
    acceptance_keys=('physical_final_answer_accuracy','structural_format_rate','transition_tracking_rate',
                     'global_mapping_consistency_rate','declared_mapping_adherence_rate',
                     'decode_back_specific_accuracy','nonliteral_encoding_rate')
    acceptance={'threshold':.95,
                'per_metric':{key:{'value':pooled[key],'passed':pooled[key]>=.95} for key in acceptance_keys},
                'zero_code_copy':pooled['answer_is_code_word_rate']==0}
    acceptance['full_pooled_gate_passed']=(all(item['passed'] for item in acceptance['per_metric'].values())
                                            and acceptance['zero_code_copy'])
    report.update({'step100_complete':True,'checkpoint':str(checkpoint),'metrics':metrics,
                   'fan_response_trajectory':trajectory,'valve_lamp_nonregression':nonregression,
                   'pooled_acceptance_gate':acceptance,'stop_reason':'step100_complete_needs_review'})
    atomic_json(STEP100_REPORT,report)
    print('===== STEP 100 STANDARD METRICS ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
    print('===== FAN WORDING-FIX RESPONSE ====='); print(json.dumps(trajectory,indent=2,sort_keys=True))
    print('===== VALVE/LAMP NON-REGRESSION ====='); print(json.dumps(nonregression,indent=2,sort_keys=True))
    print('===== POOLED 95% ACCEPTANCE GATE ====='); print(json.dumps(acceptance,indent=2,sort_keys=True))

class Step100Gate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; atomic_json(STEP100_REPORT,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step)>=TARGET_STEP: control.should_save=True; control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step==TARGET_STEP and not report.get('step100_complete'):
            finish_step100(model,checkpoint); control.should_training_stop=True
        return control

runtime_max_steps=(TARGET_STEP if RESUME_STEP==TARGET_STEP else MAX_STEPS)
args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=10,max_steps=runtime_max_steps,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=6,save_only_model=False,
    report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[Step100Gate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))
if int(trainer.state.global_step)==TARGET_STEP and not report.get('step100_complete'):
    finish_step100(model,TRAINER_DIR/'checkpoint-100')
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'step100_complete':report.get('step100_complete'),
                       'checkpoint':report.get('checkpoint'),'full_gate':report.get('pooled_acceptance_gate',{}).get('full_pooled_gate_passed'),
                       'stage4_authorized':False})
print('STOP HERE. No Stage 3.5 rerun or training beyond step 100 was executed.')
"""
    notebook['cells'][8]['source']=last.splitlines(True)
    import sys as _sys; _sys.path.insert(0, str(ROOT))
    from src.notebook_io import safe_write_notebook
    safe_write_notebook(notebook, OUTPUT); print(f'Wrote {OUTPUT}')


if __name__=='__main__': main()
