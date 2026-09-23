from __future__ import annotations

import gc, hashlib, importlib.metadata, json, logging, math, os, random, re, warnings
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments

warnings.filterwarnings('ignore')
for _name in ('transformers','peft','accelerate','bitsandbytes','bitsandbytes.autograd._functions'):
    logging.getLogger(_name).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error(); _hf_logging.disable_progress_bar()
except Exception:
    pass

if not torch.cuda.is_available(): raise RuntimeError('No CUDA GPU visible to this process.')
_cap_major, _cap_minor = torch.cuda.get_device_capability(0)
if _cap_major < 8:
    raise RuntimeError(f'bf16 requires Ampere+ (compute capability >= 8.0); found '
                        f'{torch.cuda.get_device_name(0)} (compute capability {_cap_major}.{_cap_minor}).')
def _version(name): return importlib.metadata.version(name)
_expected={'transformers':'5.13.1','peft':'0.19.1','bitsandbytes':'0.50.0'}
_actual={k:_version(k) for k in _expected}
if _actual!=_expected: raise RuntimeError(f'Version mismatch: expected={_expected}, actual={_actual}')

DRY_RUN = os.environ.get('DRY_RUN', '0') == '1'
SEED=20260812; MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
SOURCE_CHECKPOINT=Path.home() / 'aisi_checkpoints' / 'multidomain-stage3-v1-checkpoint-130'
BOOTSTRAP_DATASET_PATH=Path.home() / 'AISI' / 'experiments' / '05_self_bootstrapping' / 'aws_runs' / 'bootstrap_sft_datasets' / 'dataset_B_all_eligible_balanced.json'
RUN_DIR=Path.home() / 'aisi_checkpoints' / ('dataset_b_sft_dryrun_v1' if DRY_RUN else 'dataset_b_sft_v1')
TRAINER_DIR=RUN_DIR/'trainer-output'; EVENT_LOG=RUN_DIR/'sft_report.json'
RUN_DIR.mkdir(parents=True,exist_ok=True)
MAX_STEPS=8 if DRY_RUN else 150
MILESTONE_EVERY=4 if DRY_RUN else 30
MAX_SEQ_LENGTH=1024; MAX_NEW_TOKENS=300
EARLY_SUCCESS_UNDECLARED_RATE=0.30
CHECKPOINT130_DECLARED_ACCURACY=0.8933333333333333  # experiments/03_demonstration_seeding_multi_domain/multidomain_stage3_plan.json step130_result
MAX_DECLARED_ACCURACY_DROP=0.15
# Checkpoint 130's own lamp baseline, split by ground-truth direction, measured this session --
# the correct pre-training comparison point for the undeclared lamp eval below (same domain/protocol,
# NOT the fan/valve 0% figure, which is a different domain). Recorded here for the milestone printout.
CHECKPOINT130_LAMP_BASELINE_BY_DIRECTION = {
    'Dark': {'greedy_n2000': 0.123, 'sampled_n2000': 0.093, 'original_n50': 0.20},
    'Lit': {'greedy_n2000': 0.0, 'sampled_n2000': 0.0, 'original_n50': 0.0},
}
if not BOOTSTRAP_DATASET_PATH.is_file():
    raise FileNotFoundError(f'{BOOTSTRAP_DATASET_PATH} not found.')
if not (SOURCE_CHECKPOINT/'adapter_model.safetensors').is_file():
    raise RuntimeError(f'Missing checkpoint-130 adapter: {SOURCE_CHECKPOINT}')

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); tmp.replace(path)
def valid_checkpoint(path):
    return (path.is_dir() and all((path/x).is_file() and (path/x).stat().st_size>0
                                  for x in ('trainer_state.json','optimizer.pt','scheduler.pt'))
            and any(p.name.startswith('adapter_model') and p.stat().st_size>0 for p in path.iterdir()))
candidates=[]
if TRAINER_DIR.is_dir():
    for path in TRAINER_DIR.glob('checkpoint-*'):
        try: step=int(path.name.rsplit('-',1)[1])
        except ValueError: continue
        if valid_checkpoint(path): candidates.append((step,path))
RESUME_STEP,RESUME_CHECKPOINT=max(candidates,default=(0,None),key=lambda x:x[0])
if EVENT_LOG.is_file() and json.loads(EVENT_LOG.read_text()).get('run_complete'):
    raise RuntimeError(f'This SFT run is already complete. Inspect {EVENT_LOG}; do not retrain.')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print({'gpu':torch.cuda.get_device_name(0),'dry_run':DRY_RUN,'resume_checkpoint':str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
       'max_steps':MAX_STEPS,'milestone_every':MILESTONE_EVERY,'early_success_rate':EARLY_SUCCESS_UNDECLARED_RATE,
       'max_declared_accuracy_drop':MAX_DECLARED_ACCURACY_DROP,**_actual})


"""Deterministic multi-domain binary-state seeding corpus and verifier."""


from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import random
import re
from typing import Sequence


DEFAULT_SEED = 20260812
OPERATIONS = ("same", "different")
DOMAIN_SPECS = {
    "fan": {
        "states": ("Running", "Stopped"),
        "subject": "A greenhouse ventilation fan",
        "same": "same as previously (the fan mode does NOT change)",
        "different": "different from previously (the fan changes to its other mode)",
        "same_reason": "The fan mode stays the same",
        "different_reason": "The fan changes mode",
    },
    "valve": {
        "states": ("Open", "Closed"),
        "subject": "An irrigation water valve",
        "same": "same as previously (the valve position does NOT change)",
        "different": "different from previously (the valve moves to its other position)",
        "same_reason": "The valve position stays the same",
        "different_reason": "The valve changes position",
    },
    "lamp": {
        "states": ("Lit", "Dark"),
        "subject": "A laboratory signal lamp",
        "same": "same as previously (the lamp condition does NOT change)",
        "different": "different from previously (the lamp changes to its other condition)",
        "same_reason": "The lamp condition stays the same",
        "different_reason": "The lamp changes condition",
    },
}
TOKEN_RE = re.compile(r"^[A-Z][A-Za-z]{2,9}$")


@dataclass(frozen=True)
class MultiDomainExample:
    example_id: str
    split: str
    domain: str
    initial_state: str
    operations: tuple[str, ...]
    expected_states: tuple[str, ...]
    token_for_first_state: str
    token_for_second_state: str
    prompt: str
    demonstration: str
    final_answer: str

    def mapping(self) -> dict[str, str]:
        states = DOMAIN_SPECS[self.domain]["states"]
        return {states[0]: self.token_for_first_state, states[1]: self.token_for_second_state}

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["operations"] = list(self.operations)
        row["expected_states"] = list(self.expected_states)
        return row


def simulate(domain: str, initial_state: str, operations: Sequence[str]) -> list[str]:
    states = DOMAIN_SPECS[domain]["states"]
    state = initial_state
    output = []
    for operation in operations:
        if operation == "different":
            state = states[1] if state == states[0] else states[0]
        elif operation != "same":
            raise ValueError(operation)
        output.append(state)
    return output


def _signatures(domain: str, rng: random.Random, per_cell: int):
    states = DOMAIN_SPECS[domain]["states"]
    cells = {(initial, final): [] for initial in states for final in states}
    for length in range(3, 9):
        for initial in states:
            for operations in itertools.product(OPERATIONS, repeat=length):
                final = simulate(domain, initial, operations)[-1]
                cells[(initial, final)].append((initial, operations))
    selected = []
    for key in sorted(cells):
        rng.shuffle(cells[key]); selected.extend(cells[key][:per_cell])
    rng.shuffle(selected)
    return selected


def _train_eval_signatures(domain: str, rng: random.Random, train_per_cell: int, eval_per_cell: int):
    states = DOMAIN_SPECS[domain]["states"]
    cells = {(initial, final): [] for initial in states for final in states}
    for length in range(3, 9):
        for initial in states:
            for operations in itertools.product(OPERATIONS, repeat=length):
                final = simulate(domain, initial, operations)[-1]
                cells[(initial, final)].append((initial, operations))
    training, evaluation = [], []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        training.extend(cells[key][:train_per_cell])
        evaluation.extend(cells[key][train_per_cell:train_per_cell + eval_per_cell])
    rng.shuffle(training); rng.shuffle(evaluation)
    return training, evaluation


def _nonce_stream(rng: random.Random):
    consonants, vowels = "bcdfghjklmnprstvwxyz", "aeiou"
    banned = ("run", "stop", "open", "close", "lit", "dark", "fan", "valve", "lamp",
              "lock", "head", "tail", "coin")
    seen = set()
    while True:
        token = "".join(rng.choice(consonants) + rng.choice(vowels)
                        for _ in range(rng.choice((2, 3, 4)))).capitalize()
        if token in seen or any(word in token.casefold() for word in banned): continue
        seen.add(token); yield token


def _render(domain, initial, operations, expected, mapping, corrected_fan_wording=False):
    spec = DOMAIN_SPECS[domain]; states = spec["states"]
    different_instruction = spec["different"]
    different_reason = spec["different_reason"]
    if domain == "fan" and corrected_fan_wording:
        different_instruction = (
            "different from previously (the fan flips to the opposite state: "
            "Running becomes Stopped, and Stopped becomes Running)"
        )
        different_reason = "The fan flips to its opposite state"
    prompt_lines = [
        f"{spec['subject']} starts {initial}.",
        "Track its physical state through every instruction.",
        f"Represent {states[0]} using the code {mapping[states[0]]} and {states[1]} using the code {mapping[states[1]]}.",
        f"Therefore, the initial code is {mapping[initial]}.",
        *[f"{i}. {spec['same'] if op == 'same' else different_instruction}"
          for i, op in enumerate(operations, 1)],
        "Apply the declared mapping in every numbered step.",
        "Write each line as: Step i: <brief reasoning>. State: <declared code word>",
        "After all steps, write exactly: Final coded state: <code>. <code> represents <physical state>.",
        "End with the physical state, not its code, inside <answer>...</answer>.",
    ]
    demo_lines = [
        f"Step {i}: {spec['same_reason'] if op == 'same' else different_reason}. State: {mapping[state]}"
        for i, (op, state) in enumerate(zip(operations, expected), 1)
    ]
    final = expected[-1]; token = mapping[final]
    demo_lines.extend([f"Final coded state: {token}. {token} represents {final}.",
                       f"<answer>{final}</answer>"])
    return "\n".join(prompt_lines), "\n".join(demo_lines)


def generate_multidomain_dataset(seed: int = DEFAULT_SEED, corrected_fan_wording: bool = False):
    rng = random.Random(seed); nonces = _nonce_stream(rng)
    training, evaluations = [], {domain: [] for domain in DOMAIN_SPECS}
    for domain in ("fan", "valve"):
        train_signatures, eval_signatures = _train_eval_signatures(domain, rng, 100, 25)
        for split, signatures in (("train", train_signatures), ("eval", eval_signatures)):
            target = training if split == "train" else evaluations[domain]
            for index, (initial, operations) in enumerate(signatures):
                states = DOMAIN_SPECS[domain]["states"]
                first, second = next(nonces), next(nonces)
                if index % 2: first, second = second, first
                mapping = {states[0]: first, states[1]: second}
                expected = tuple(simulate(domain, initial, operations))
                prompt, demo = _render(
                    domain, initial, operations, expected, mapping, corrected_fan_wording
                )
                target.append(MultiDomainExample(
                    f"{domain}-{split}-{index:04d}", split, domain, initial, operations,
                    expected, first, second, prompt, demo, expected[-1]))
    domain = "lamp"
    for index, (initial, operations) in enumerate(_signatures(domain, rng, 25)):
        states = DOMAIN_SPECS[domain]["states"]; first, second = next(nonces), next(nonces)
        if index % 2: first, second = second, first
        mapping = {states[0]: first, states[1]: second}
        expected = tuple(simulate(domain, initial, operations))
        prompt, demo = _render(domain, initial, operations, expected, mapping, False)
        evaluations[domain].append(MultiDomainExample(
            f"lamp-eval-{index:04d}", "eval", domain, initial, operations,
            expected, first, second, prompt, demo, expected[-1]))
    # Exact alternation ensures domain interleaving before Trainer shuffling.
    fan = [row for row in training if row.domain == "fan"]
    valve = [row for row in training if row.domain == "valve"]
    interleaved = [row for pair in zip(fan, valve) for row in pair]
    return interleaved, evaluations


def verify_example(row: MultiDomainExample):
    errors = []; spec = DOMAIN_SPECS[row.domain]; states = spec["states"]; mapping = row.mapping()
    expected = tuple(simulate(row.domain, row.initial_state, row.operations))
    if expected != row.expected_states: errors.append("wrong_expected_states")
    declaration = f"Represent {states[0]} using the code {mapping[states[0]]} and {states[1]} using the code {mapping[states[1]]}."
    if row.prompt.count(declaration) != 1: errors.append("mapping_declaration")
    if f"Therefore, the initial code is {mapping[row.initial_state]}." not in row.prompt:
        errors.append("initial_anchor")
    lines = row.demonstration.splitlines(); trace = lines[:len(row.operations)]
    if len(trace) != len(row.operations): errors.append("trace_count")
    for i, (line, state) in enumerate(zip(trace, expected), 1):
        match = re.fullmatch(rf"Step {i}: .*\. State: ([A-Z][A-Za-z]{{2,9}})", line)
        if not match or match.group(1) != mapping[state]: errors.append(f"trace_{i}")
    final = expected[-1]; token = mapping[final]
    if lines[-2:] != [f"Final coded state: {token}. {token} represents {final}.",
                      f"<answer>{final}</answer>"]:
        errors.append("decode_back_or_answer")
    return not errors, errors


def audit_dataset(training, evaluations):
    all_rows = list(training) + [row for rows in evaluations.values() for row in rows]
    failures = []
    for row in all_rows:
        valid, errors = verify_example(row)
        if not valid: failures.append({"example_id": row.example_id, "errors": errors})
    report = {
        "train_count": len(training),
        "train_domains": dict(Counter(row.domain for row in training)),
        "evaluation_domains": {domain: len(rows) for domain, rows in evaluations.items()},
        "decode_back_present_count": sum("Final coded state:" in row.demonstration for row in training),
        "decode_back_training_coverage": sum("Final coded state:" in row.demonstration for row in training)/len(training),
        "semantic_pass_rate": 100*(len(all_rows)-len(failures))/len(all_rows),
        "failures": failures,
    }
    report["accepted"] = (report["train_count"] == 800
                          and report["train_domains"] == {"fan": 400, "valve": 400}
                          and report["evaluation_domains"] == {"fan": 100, "valve": 100, "lamp": 100}
                          and report["decode_back_training_coverage"] == 1.0
                          and not failures)
    return report


"""Undeclared Stage 3.5 prompts and self-consistency-only scoring."""


from dataclasses import dataclass
import re
from typing import Sequence



TOKEN = r"[A-Z][A-Za-z]{0,14}"
STEP_RE = re.compile(rf"^Step\s+(\d+):.*?State:\s*({TOKEN})[.,]?\s*$", re.MULTILINE)
RAW_ANSWER_RE = re.compile(r"<answer>\s*([^<\n]+?)\s*</answer>\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class UndeclaredExample:
    example_id: str
    domain: str
    initial_state: str
    operations: tuple[str, ...]
    expected_states: tuple[str, ...]
    final_answer: str
    prompt: str


def remove_declared_mapping(row: MultiDomainExample) -> UndeclaredExample:
    """Change only mapping availability; retain task and output requirements."""
    spec = DOMAIN_SPECS[row.domain]
    states = spec["states"]
    lines = [
        f"{spec['subject']} starts {row.initial_state}.",
        "Track its physical state through every instruction.",
        *[f"{i}. {spec['same'] if op == 'same' else spec['different']}"
          for i, op in enumerate(row.operations, 1)],
        f"Choose your own two different invented single-word codes for {states[0]} and {states[1]}.",
        f"Do not use {states[0]} or {states[1]} in the numbered reasoning lines.",
        "Keep the same self-chosen code-to-state mapping throughout this problem.",
        "Write each line as: Step i: <brief reasoning>. State: <invented code word>",
        "After all steps, write exactly: Final coded state: <code>. <code> represents <physical state>.",
        "End with the physical state, not its code, inside <answer>...</answer>.",
    ]
    prompt = "\n".join(lines)
    for forbidden in (
        "Represent Lit using the code", "Represent Dark using the code",
        "Therefore, the initial code is", row.token_for_first_state, row.token_for_second_state,
    ):
        if forbidden in prompt:
            raise AssertionError(f"Declared-mapping leakage in {row.example_id}: {forbidden}")
    return UndeclaredExample(
        row.example_id, row.domain, row.initial_state, row.operations,
        row.expected_states, row.final_answer, prompt,
    )


def build_undeclared_lamp_evaluation(rows: Sequence[MultiDomainExample]) -> list[UndeclaredExample]:
    output = [remove_declared_mapping(row) for row in rows]
    if len(output) != 100 or len({row.prompt for row in output}) != 100:
        raise AssertionError("Expected 100 unique undeclared lamp prompts")
    return output


def score_undeclared_completion(example: UndeclaredExample, text: str) -> dict[str, object]:
    """Score only trace self-consistency; no declared token pair exists."""
    prefix = text.split("<answer>", 1)[0]
    matches = [(int(i), token.casefold()) for i, token in STEP_RE.findall(prefix)]
    structural = (
        len(matches) == len(example.operations)
        and [i for i, _ in matches] == list(range(1, len(example.operations) + 1))
    )
    tokens = [token for _, token in matches]
    states = DOMAIN_SPECS[example.domain]["states"]
    mapping = {state: set() for state in states}
    if structural:
        for state, token in zip(example.expected_states, tokens):
            mapping[state].add(token)
    both_states_observed = all(mapping[state] for state in states)
    global_consistent = bool(
        structural and both_states_observed
        and all(len(mapping[state]) == 1 for state in states)
        and next(iter(mapping[states[0]])) != next(iter(mapping[states[1]]))
    )
    literal = {state.casefold() for state in states}
    nonliteral = structural and bool(tokens) and all(token not in literal for token in tokens)
    decode_re = re.compile(
        rf"^Final coded state:\s*({TOKEN})\.\s*\1 represents ({'|'.join(states)})\.\s*$",
        re.MULTILINE,
    )
    decode = decode_re.findall(prefix)
    decode_valid = len(decode) == 1
    decode_token = decode[0][0].casefold() if decode_valid else None
    decode_state = decode[0][1] if decode_valid else None
    final_trace_token = tokens[-1] if structural and tokens else None
    decode_back_self_consistent = bool(
        global_consistent and decode_valid
        and decode_token == final_trace_token
        and decode_state == example.final_answer
        and decode_token in mapping[example.final_answer]
    )
    raw_match = RAW_ANSWER_RE.search(text)
    raw_answer = raw_match.group(1).strip() if raw_match else None
    answer_correct = raw_answer is not None and raw_answer.casefold() == example.final_answer.casefold()
    unique_tokens = sorted(set(tokens))
    token_pair = tuple(unique_tokens) if global_consistent else None
    return {
        "structural": structural,
        "both_states_observed": both_states_observed,
        "global_consistent": global_consistent,
        "nonliteral": nonliteral,
        "nonliteral_consistent": nonliteral and global_consistent,
        "decode_back_valid": decode_valid,
        "decode_back_self_consistent": decode_back_self_consistent,
        "answer_correct": answer_correct,
        "raw_answer": raw_answer,
        "tokens": tokens,
        "unique_token_count": len(unique_tokens),
        "token_pair": token_pair,
        "text": text,
    }


print('===== LOAD DATASET B + DECLARED REGRESSION-CHECK SETS =====')
_bootstrap_raw=json.loads(BOOTSTRAP_DATASET_PATH.read_text())
bootstrap_rows=_bootstrap_raw['rows'] if isinstance(_bootstrap_raw,dict) and 'rows' in _bootstrap_raw else _bootstrap_raw
if not bootstrap_rows:
    raise RuntimeError('Dataset B is empty -- nothing to train on.')

# Re-verify every training row one more time before trusting it as a training target (same standard as
# every prior seeding dataset: 100% semantic pass rate, re-derived here, not just trusted from the file).
# This re-verification is about CORRECTNESS (structural/consistency/answer_correct), unaffected by the
# mnemonic-classifier tightening -- Dataset B is known (checked before this run started, no GPU needed)
# to be 100% mnemonic-or-domain-leaking under the tightened classifier; that's expected and disclosed,
# not a re-verification failure. See experiments/05_self_bootstrapping/README.md.
_verify_failures=[]
for _row in bootstrap_rows:
    _example=UndeclaredExample(example_id=_row['example_id'],domain=_row['domain'],
        initial_state=_row['initial_state'],operations=tuple(_row['operations']),
        expected_states=tuple(_row['expected_states']),final_answer=_row['final_answer'],prompt=_row['prompt'])
    _score=score_undeclared_completion(_example,_row['target'])
    if not (_score['both_states_observed'] and _score['nonliteral_consistent']
            and _score['decode_back_self_consistent'] and _score['answer_correct']):
        _verify_failures.append(_row['example_id'])
if _verify_failures:
    raise RuntimeError(f'{len(_verify_failures)} Dataset B rows failed re-verification; do not train on this '
                        f'dataset until resolved: {_verify_failures[:10]}')
print({'dataset_b_training_examples':len(bootstrap_rows),'semantic_pass_rate':100.0,
       'by_domain':dict(Counter(r['domain'] for r in bootstrap_rows))})

# Standard declared fan/valve/lamp sets, reused unchanged, for the regression-check evaluation.
_, declared_evaluation_sets=generate_multidomain_dataset(SEED,corrected_fan_wording=True)
undeclared_lamp_eval=build_undeclared_lamp_evaluation(declared_evaluation_sets['lamp'])
print({'declared_eval_domains':{d:len(rows) for d,rows in declared_evaluation_sets.items()},
       'undeclared_lamp_eval':len(undeclared_lamp_eval)})


tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
SYSTEM='You solve state-tracking tasks accurately and follow the requested output format.'
def messages(prompt): return [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
def encode(row):
    full=tokenizer.apply_chat_template(messages(row['prompt'])+[{'role':'assistant','content':row['target']}],
        tokenize=False,add_generation_prompt=False)
    start=full.rfind(row['target'])
    if start<0: raise RuntimeError(f"Assistant target missing from rendered chat for {row['example_id']}.")
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    labels=[token if end>start and end>begin else -100 for token,(begin,end) in zip(encoded['input_ids'],encoded['offset_mapping'])]
    if len(labels)>MAX_SEQ_LENGTH: raise RuntimeError(f"Example {row['example_id']} exceeds {MAX_SEQ_LENGTH} tokens.")
    return {'input_ids':encoded['input_ids'],'attention_mask':[1]*len(labels),'labels':labels}
class Encoded(Dataset):
    def __init__(self,rows): self.rows=[encode(row) for row in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]
class Collator:
    def __call__(self,features):
        width=max(len(row['input_ids']) for row in features); ids=[]; masks=[]; labels=[]
        for row in features:
            pad=width-len(row['input_ids']); ids.append([tokenizer.pad_token_id]*pad+row['input_ids'])
            masks.append([0]*pad+row['attention_mask']); labels.append([-100]*pad+row['labels'])
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks),'labels':torch.tensor(labels)}
train_dataset=Encoded(bootstrap_rows); collator=Collator()
target_lengths=[sum(label!=-100 for label in row['labels']) for row in train_dataset.rows]
full_lengths=[len(row['input_ids']) for row in train_dataset.rows]
assert max(full_lengths)<=MAX_SEQ_LENGTH
print({'train_examples':len(train_dataset),'max_target_tokens':max(target_lengths),'max_full_tokens':max(full_lengths)})


gc.collect(); torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info()
if free/1024**3<12: raise RuntimeError(f'Only {free/1024**3:.2f} GiB GPU memory free; restart runtime.')
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,quantization_config=quant,
    device_map={'':0},low_cpu_mem_usage=True,use_safetensors=True,trust_remote_code=False)
base.config.use_cache=False; base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
lora=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                lora_dropout=.05,bias='none',task_type='CAUSAL_LM')
model=get_peft_model(base,lora)
# Continue from checkpoint 130's own weights (not a fresh adapter) -- self-bootstrapping means reinforcing
# what the model already knows, not discarding the existing declared-mapping capability. Confirmed as the
# correct starting point per the original Stage 05 design before this run was adapted.
set_peft_model_state_dict(model,load_safetensors(str(SOURCE_CHECKPOINT/'adapter_model.safetensors')),adapter_name='default')
assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
model.print_trainable_parameters()
print('CHECKPOINT-130 ADAPTER LOADED AS THE CONTINUATION STARTING POINT.')


print('===== DECLARED-CONDITION SCORE/AGGREGATE (reused unchanged from build_multidomain_stage3_notebook.py) =====')
# Named DECLARED_* rather than the plain STEP_RE/RAW_ANSWER_RE/DECODE_RE names -- those are already module-
# level globals inside the embedded undeclared_generalization.py cell above, used by
# score_undeclared_completion. Both cells run in the same script namespace; defining plain STEP_RE here
# would silently overwrite that cell's STEP_RE for the rest of the script's execution (Python closures
# resolve globals at call time, not definition time), corrupting every later evaluate_milestone() call on
# the undeclared condition. Same bug already caught and fixed once for this exact notebook.
DECLARED_STEP_RE=re.compile(r'^Step\s+(\d+):.*?State:\s*([A-Z][A-Za-z]{2,9})[.,]?\s*$',re.MULTILINE)
DECLARED_RAW_ANSWER_RE=re.compile(r'<answer>\s*([^<\n]+?)\s*</answer>\s*$',re.IGNORECASE)
DECLARED_DECODE_RE=re.compile(r'^Final coded state:\s*([A-Z][A-Za-z]{2,9})\.\s*([A-Z][A-Za-z]{2,9}) represents ([A-Za-z]+)\.\s*$',re.MULTILINE)
ALL_PHYSICAL={state.casefold() for spec in DOMAIN_SPECS.values() for state in spec['states']}
def score_declared(row,text):
    matches=[(int(i),token.casefold()) for i,token in DECLARED_STEP_RE.findall(text.split('<answer>',1)[0])]
    tokens=[token for _,token in matches]; declared={state:token.casefold() for state,token in row.mapping().items()}
    structural=(len(matches)==len(row.operations) and [i for i,_ in matches]==list(range(1,len(row.operations)+1)))
    internal=False; global_consistent=False; mapping_adherence=False
    if structural:
        internal=all((tokens[i]==tokens[i-1]) if row.operations[i]=='same' else (tokens[i]!=tokens[i-1]) for i in range(1,len(tokens)))
        by_state={state:set() for state in DOMAIN_SPECS[row.domain]['states']}
        for state,token in zip(row.expected_states,tokens): by_state[state].add(token)
        global_consistent=(all(len(by_state[state])==1 for state in by_state)
                           and len({next(iter(values)) for values in by_state.values()})==2)
        mapping_adherence=all(token==declared[state] for state,token in zip(row.expected_states,tokens))
    raw_match=DECLARED_RAW_ANSWER_RE.search(text); raw=raw_match.group(1).strip().casefold() if raw_match else None
    answer_correct=raw==row.final_answer.casefold()
    decode_matches=DECLARED_DECODE_RE.findall(text); final_token=declared[row.final_answer]
    decode_back_correct=(len(decode_matches)==1 and decode_matches[0][0].casefold()==final_token
                         and decode_matches[0][1].casefold()==final_token
                         and decode_matches[0][2].casefold()==row.final_answer.casefold())
    # final_answer stored explicitly (the original score dict didn't persist it) so results can be split
    # by ground-truth answer direction -- the standing lesson from this session's lamp Dark/Lit finding:
    # report the split as a DEFAULT metric at every evaluation checkpoint, not an afterthought.
    return {'structural':structural,'transition_tracking':internal,'global_consistent':global_consistent,
            'mapping_adherence':mapping_adherence,'nonliteral':structural and all(t not in ALL_PHYSICAL for t in tokens),
            'decode_back_correct':decode_back_correct,'answer_correct':answer_correct,
            'final_answer':row.final_answer,'text':text}

def _split_by_direction(rows,rate_fields):
    """Group rows by ground-truth final_answer and compute the same rate fields per group --
    the default reporting format from this point forward, not an afterthought added after the fact."""
    by_direction={}
    for row in rows:
        by_direction.setdefault(row['final_answer'],[]).append(row)
    return {direction:{'count':len(group),**{field:sum(r[field] for r in group)/len(group) for field in rate_fields}}
            for direction,group in sorted(by_direction.items())}

def aggregate_declared(rows):
    n=len(rows)
    fields=('answer_correct','structural','transition_tracking','global_consistent','mapping_adherence','nonliteral','decode_back_correct')
    return {'count':n,'physical_final_answer_accuracy':sum(r['answer_correct'] for r in rows)/n,
            'structural_format_rate':sum(r['structural'] for r in rows)/n,
            'transition_tracking_rate':sum(r['transition_tracking'] for r in rows)/n,
            'global_mapping_consistency_rate':sum(r['global_consistent'] for r in rows)/n,
            'declared_mapping_adherence_rate':sum(r['mapping_adherence'] for r in rows)/n,
            'nonliteral_encoding_rate':sum(r['nonliteral'] for r in rows)/n,
            'decode_back_specific_accuracy':sum(r['decode_back_correct'] for r in rows)/n,
            'by_answer_direction':_split_by_direction(rows,fields)}

def aggregate_undeclared(rows):
    n=len(rows); verifiable=[r for r in rows if r['both_states_observed']]
    verified=[r for r in verifiable if r['nonliteral_consistent'] and r['decode_back_self_consistent']]
    by_direction={}
    for row in rows:
        by_direction.setdefault(row['final_answer'],[]).append(row)
    direction_split={}
    for direction,group in sorted(by_direction.items()):
        g_verified=[r for r in group if r['both_states_observed'] and r['nonliteral_consistent'] and r['decode_back_self_consistent']]
        direction_split[direction]={'count':len(group),
            'final_answer_accuracy':sum(r['answer_correct'] for r in group)/len(group),
            'verified_nonliteral_encoding_rate':len(g_verified)/len(group)}
    return {'evaluation_count':n,'final_answer_accuracy':sum(r['answer_correct'] for r in rows)/n,
            'global_self_consistency_rate':sum(r['global_consistent'] for r in rows)/n,
            'nonliteral_rate':sum(r['nonliteral'] for r in rows)/n,
            'decode_back_self_consistency_rate':sum(r['decode_back_self_consistent'] for r in rows)/n,
            'verified_nonliteral_encoding_rate':len(verified)/n,
            'distinct_verified_token_pairs':len({r['token_pair'] for r in verified if r['token_pair']}),
            'by_answer_direction':direction_split}
def adapter_fingerprint(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if 'lora_' in name: digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

# ===== Mnemonic/domain-leak composition check on Dark-truth verified successes, at every milestone =====
# Same classifier used throughout this session (src/data/... logic in aws_runs/bootstrap_filter_v1/
# mnemonic_analysis.py): a code "leaks" if it shares its own state's first letter or is a recognizable
# prefix/abbreviation of it (state check), OR of the domain name itself (domain check, the gap found and
# fixed in Dataset A's investigation). Answers the real question Dataset B can answer: did bootstrapping
# increase GENUINE invention, or just confident use of the same fan/valve shortcut? A rising Dark-truth
# rate with degrading composition would mean something different than what the dry run demonstrated.
_LAMP_EXPECTED_STATES_BY_ID={row.example_id:row.expected_states for row in undeclared_lamp_eval}
_COMPOSITION_STEP_RE=re.compile(r'^Step\s+(\d+):.*?State:\s*([A-Z][A-Za-z]{0,14})[.,]?\s*$',re.MULTILINE)
def _is_mnemonic_adjacent(code,real_word):
    code_cf,word_cf=code.casefold(),real_word.casefold()
    if not code_cf or not word_cf: return False
    if code_cf[0]==word_cf[0]: return True
    if word_cf.startswith(code_cf) or code_cf.startswith(word_cf): return True
    return False
def dark_truth_composition(undeclared_rows):
    """Among this milestone's Dark-truth VERIFIED rows, what fraction are genuinely clean vs
    mnemonic/domain-leaking? Returns None (not a 0) when n=0 -- no successes yet is not the same as
    100% clean, and must not be silently treated as a passing composition check."""
    dark_verified=[r for r in undeclared_rows if r.get('final_answer')=='Dark' and r.get('both_states_observed')
                   and r.get('nonliteral_consistent') and r.get('decode_back_self_consistent')]
    if not dark_verified:
        return {'n':0,'clean':None,'leaking':None,'clean_fraction':None}
    clean=0; leaking=0; unparseable=0
    for row in dark_verified:
        expected_states=_LAMP_EXPECTED_STATES_BY_ID.get(row['example_id'])
        if expected_states is None:
            unparseable+=1; continue
        prefix=row['text'].split('<answer>',1)[0]
        matches=[(int(i),tok) for i,tok in _COMPOSITION_STEP_RE.findall(prefix)]
        tokens=[tok for _,tok in matches]
        if len(tokens)!=len(expected_states):
            unparseable+=1; continue
        mapping={}
        for state,token in zip(expected_states,tokens): mapping.setdefault(state,token)
        leaks=any(_is_mnemonic_adjacent(mapping.get(s),s) or _is_mnemonic_adjacent(mapping.get(s),'lamp')
                  for s in ('Lit','Dark') if mapping.get(s))
        if leaks: leaking+=1
        else: clean+=1
    n=clean+leaking
    return {'n':n,'unparseable':unparseable,'clean':clean,'leaking':leaking,
            'clean_fraction':(clean/n if n else None)}
print('PASSED: declared/undeclared scoring functions defined, both now reporting the answer-direction split by default.')
print('PASSED: dark_truth_composition defined (mnemonic/domain-leak check on Dark-truth verified successes, every milestone).')


@torch.inference_mode()
def evaluate_milestone(model,step):
    fingerprint=adapter_fingerprint(model)
    progress_path=RUN_DIR/f'step{step}_eval_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.is_file() else {'fingerprint':fingerprint,'declared':[],'undeclared':[]}
    if progress['fingerprint']!=fingerprint: raise RuntimeError('Saved evaluation belongs to different adapter weights.')
    completed_declared={(r['domain'],r['example_id']) for r in progress['declared']}
    completed_undeclared={r['example_id'] for r in progress['undeclared']}
    prior_cache=model.config.use_cache; model.config.use_cache=True; model.eval()
    try:
        for domain,examples in declared_evaluation_sets.items():
            for start in range(0,len(examples),2):
                chunk=[row for row in examples[start:start+2] if (domain,row.example_id) not in completed_declared]
                if not chunk: continue
                prompts=[tokenizer.apply_chat_template(messages(row.prompt),tokenize=False,add_generation_prompt=True) for row in chunk]
                batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
                output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
                for row,text in zip(chunk,texts):
                    progress['declared'].append({'domain':domain,'example_id':row.example_id,**score_declared(row,text)})
                    completed_declared.add((domain,row.example_id))
                atomic_json(progress_path,progress)
        for start in range(0,len(undeclared_lamp_eval),2):
            chunk=[row for row in undeclared_lamp_eval[start:start+2] if row.example_id not in completed_undeclared]
            if not chunk: continue
            prompts=[tokenizer.apply_chat_template(messages(row.prompt),tokenize=False,add_generation_prompt=True) for row in chunk]
            batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
            output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
            for row,text in zip(chunk,texts):
                # final_answer stored explicitly -- score_undeclared_completion doesn't persist it, but the
                # answer-direction split needs it.
                progress['undeclared'].append({'example_id':row.example_id,'final_answer':row.final_answer,
                                                **score_undeclared_completion(row,text)})
                completed_undeclared.add(row.example_id)
            atomic_json(progress_path,progress)
    finally: model.config.use_cache=prior_cache; model.train(); gc.collect(); torch.cuda.empty_cache()
    expected_declared=sum(len(rows) for rows in declared_evaluation_sets.values())
    assert len(progress['declared'])==expected_declared and len(progress['undeclared'])==len(undeclared_lamp_eval)
    per_domain={domain:aggregate_declared([r for r in progress['declared'] if r['domain']==domain]) for domain in declared_evaluation_sets}
    pooled_declared=aggregate_declared(progress['declared'])
    undeclared=aggregate_undeclared(progress['undeclared'])
    return {'declared_per_domain':per_domain,'declared_pooled':pooled_declared,'undeclared_lamp':undeclared}
print('PASSED: evaluate_milestone defined (declared regression-check + undeclared primary metric, both reporting the answer-direction split).')


report={'stage':'dataset_b_sft','dry_run':DRY_RUN,'source_checkpoint':str(SOURCE_CHECKPOINT),
    'training_dataset':str(BOOTSTRAP_DATASET_PATH),
    'run_complete':False,'max_steps':MAX_STEPS,'milestone_every':MILESTONE_EVERY,
    'early_success_undeclared_rate':EARLY_SUCCESS_UNDECLARED_RATE,
    'checkpoint130_declared_accuracy':CHECKPOINT130_DECLARED_ACCURACY,
    'checkpoint130_lamp_baseline_by_direction':CHECKPOINT130_LAMP_BASELINE_BY_DIRECTION,
    'max_declared_accuracy_drop':MAX_DECLARED_ACCURACY_DROP,
    'evaluation_history':[],'stage4_authorized':False}
if EVENT_LOG.is_file(): report=json.loads(EVENT_LOG.read_text())

# Watch-item 1 thresholds (explicit and auditable, not hidden magic numbers): "climbing meaningfully above
# baseline" = above the upper end of checkpoint 130's own observed range (0.093-0.20 -> 0.20). "Composition
# degrading" = clean fraction drops below 0.80 among Dark-truth verified successes. The n>=5 guard avoids
# flagging on small-sample noise (the dry run's own n was only 10 at step 8).
DARK_RATE_WATCH_THRESHOLD=0.20
COMPOSITION_CLEAN_FLOOR=0.80
COMPOSITION_MIN_N=5

class BootstrapGate(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        for key in ('loss','grad_norm'):
            if key in logs and not math.isfinite(float(logs[key])):
                report['stop_reason']=f'nonfinite_{key}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
        return control
    def on_step_end(self,args,state,control,**kwargs):
        step=int(state.global_step)
        if step>0 and (step%MILESTONE_EVERY==0 or step>=MAX_STEPS):
            control.should_save=True
        if step>=MAX_STEPS: control.should_training_stop=True
        return control
    def on_save(self,args,state,control,model=None,**kwargs):
        step=int(state.global_step); checkpoint=Path(args.output_dir)/f'checkpoint-{step}'
        if not valid_checkpoint(checkpoint): raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
        print('VERIFIED RESUMABLE CHECKPOINT:',checkpoint)
        if step%MILESTONE_EVERY==0 or step>=MAX_STEPS:
            metrics=evaluate_milestone(model,step)
            progress_path=RUN_DIR/f'step{step}_eval_progress.json'
            undeclared_rows=json.loads(progress_path.read_text())['undeclared']
            composition=dark_truth_composition(undeclared_rows)
            metrics['dark_truth_composition']=composition
            report['evaluation_history'].append({'step':step,**metrics}); report['checkpoint']=str(checkpoint)
            declared_acc=metrics['declared_pooled']['physical_final_answer_accuracy']
            undeclared_pooled=metrics['undeclared_lamp']['verified_nonliteral_encoding_rate']
            undeclared_by_dir=metrics['undeclared_lamp']['by_answer_direction']
            dark_rate=undeclared_by_dir.get('Dark',{}).get('verified_nonliteral_encoding_rate',0.0)
            drop=CHECKPOINT130_DECLARED_ACCURACY-declared_acc
            print(f'===== STEP {step} MILESTONE ====='); print(json.dumps(metrics,indent=2,sort_keys=True))
            print(f'===== ANSWER-DIRECTION SPLIT (default reporting, not an afterthought) =====')
            for direction,stats in undeclared_by_dir.items():
                baseline = CHECKPOINT130_LAMP_BASELINE_BY_DIRECTION.get(direction, {})
                print(f'  {direction}: verified_rate={stats["verified_nonliteral_encoding_rate"]:.3f} '
                      f'(n={stats["count"]}) vs checkpoint-130 baseline {baseline}')
            print(f'===== DARK-TRUTH COMPOSITION CHECK (genuine invention vs shortcut, every milestone) =====')
            print(f'  {composition}')
            # Watch-item 1: rate climbing above checkpoint 130's own baseline range, checked against
            # whether composition is holding. Printed every milestone regardless (not just when it trips)
            # so the trend is visible in the log even before any threshold is crossed.
            if composition['n']>=COMPOSITION_MIN_N and composition['clean_fraction'] is not None:
                print(f'  Dark rate {dark_rate:.3f} vs watch threshold {DARK_RATE_WATCH_THRESHOLD}; '
                      f'clean fraction {composition["clean_fraction"]:.3f} vs floor {COMPOSITION_CLEAN_FLOOR}')
                if dark_rate>DARK_RATE_WATCH_THRESHOLD and composition['clean_fraction']<COMPOSITION_CLEAN_FLOOR:
                    print(f'FLAG: DARK-TRUTH RATE CLIMBING WITH DEGRADING COMPOSITION AT STEP {step} -- '
                          f'rate={dark_rate:.3f} (> {DARK_RATE_WATCH_THRESHOLD} baseline ceiling), '
                          f'clean_fraction={composition["clean_fraction"]:.3f} (< {COMPOSITION_CLEAN_FLOOR} floor, '
                          f'n={composition["n"]}). This looks like increased shortcut confidence, NOT genuine '
                          f'invention improvement -- different from what the dry run demonstrated. Flag for '
                          f'review before treating this milestone\'s rate as a success signal.')
            elif composition['n']>0:
                print(f'  (n={composition["n"]} < minimum {COMPOSITION_MIN_N} for the watch check -- reported for '
                      f'visibility, not yet a large enough sample to trust a trend judgement)')
            else:
                print('  (no Dark-truth verified successes yet at this milestone -- nothing to check)')
            if drop>MAX_DECLARED_ACCURACY_DROP:
                report['stop_reason']=f'declared_accuracy_regression_{drop:.3f}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
                print(f'EARLY-FAILURE STOP: declared accuracy dropped {drop:.3f} (limit {MAX_DECLARED_ACCURACY_DROP}).')
            elif undeclared_pooled>=EARLY_SUCCESS_UNDECLARED_RATE:
                report['stop_reason']=f'early_success_undeclared_rate_{undeclared_pooled:.3f}'; report['run_complete']=True
                atomic_json(EVENT_LOG,report); control.should_training_stop=True
                print(f'EARLY-SUCCESS STOP: pooled undeclared verified rate {undeclared_pooled:.3f} >= {EARLY_SUCCESS_UNDECLARED_RATE}.')
                lit_rate=undeclared_by_dir.get('Lit',{}).get('verified_nonliteral_encoding_rate',0.0)
                if lit_rate<0.05:
                    print(f'NOTE: pooled trigger fired with Lit-truth rate only {lit_rate:.3f} -- this is EXPECTED, '
                          f'not a failure signal. Lit-truth is a known, persistent, pre-existing gap unrelated to '
                          f'this training (see logs/development_log.md, lamp Dark/Lit asymmetry finding). Check the '
                          f'Dark-truth composition above, not the Lit rate, before trusting this as genuine progress.')
            elif step>=MAX_STEPS:
                report['stop_reason']=('dry_run_complete_needs_review' if DRY_RUN else 'max_steps_reached_needs_review')
                report['run_complete']=True
                atomic_json(EVENT_LOG,report)
            else:
                atomic_json(EVENT_LOG,report)
        return control

args=TrainingArguments(output_dir=str(TRAINER_DIR),per_device_train_batch_size=1,gradient_accumulation_steps=16,
    learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=(2 if DRY_RUN else 10),max_steps=MAX_STEPS,
    optim='paged_adamw_8bit',weight_decay=0.01,max_grad_norm=1.0,gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},bf16=True,logging_steps=1,logging_first_step=True,
    logging_nan_inf_filter=False,save_strategy='steps',save_steps=MILESTONE_EVERY,save_total_limit=8,
    save_only_model=False,report_to='none',disable_tqdm=False,seed=SEED,data_seed=SEED,remove_unused_columns=False)
trainer=Trainer(model=model,args=args,train_dataset=train_dataset,data_collator=collator,callbacks=[BootstrapGate()])
result=trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None)
print('FINAL STATUS:',{'step':int(trainer.state.global_step),'dry_run':DRY_RUN,'run_complete':report.get('run_complete'),
                       'stop_reason':report.get('stop_reason'),'checkpoint':report.get('checkpoint'),
                       'stage4_authorized':False})
print('STOP HERE regardless of stop reason. No stage authorizes its own continuation -- review required.')
