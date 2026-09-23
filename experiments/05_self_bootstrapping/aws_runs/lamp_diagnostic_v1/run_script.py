from __future__ import annotations

import gc, hashlib, importlib.metadata, itertools, json, logging, math, os, random, re, statistics, warnings
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

MODEL_NAME='Qwen/Qwen2.5-3B-Instruct'
PROMPT_SEED=20260824
BATCH_SIZE=64
MAX_NEW_TOKENS=300
CHECKPOINT_EVERY=200
TARGET_POOL_SIZE=2000
CHECKPOINT=Path.home() / 'aisi_checkpoints' / 'multidomain-stage3-v1-checkpoint-130'
ORIGINAL_LAMP_EVAL_DIR=Path.home() / 'aisi_checkpoints' / 'stage35-undeclared-lamp-checkpoint130-v2'
OUTPUT_DIR=Path.home() / 'aisi_checkpoints' / 'lamp-diagnostic-generation-v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({'gpu':torch.cuda.get_device_name(0),'prompt_seed':PROMPT_SEED,'batch_size':BATCH_SIZE,
       'target_pool_size':TARGET_POOL_SIZE,**_actual})


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


print('===== BUILD LAMP UNDECLARED PROMPT POOL, DISJOINT FROM THE ORIGINAL 100-EXAMPLE EVAL SET =====')
LENGTH_RANGE = (3, 10)
DOMAIN = 'lamp'

_original = json.loads((ORIGINAL_LAMP_EVAL_DIR / 'rollouts.json').read_text())
_original_rows = _original['rows'] if isinstance(_original, dict) and 'rows' in _original else _original
# rollouts.json does not store the 'prompt' text field (unneeded for classification), only the
# structural fields prompt is a deterministic function of -- compare by (initial_state, operations)
# signature instead, which is equivalent and does not require re-deriving prompt text from the original.
ORIGINAL_LAMP_SIGNATURES = {(r['initial_state'], tuple(r['operations'])) for r in _original_rows}
print(f'original lamp eval signatures loaded for disjointness check: {len(ORIGINAL_LAMP_SIGNATURES)}')

def _all_signatures(domain, min_len, max_len):
    states = DOMAIN_SPECS[domain]['states']
    signatures = []
    for length in range(min_len, max_len + 1):
        for initial in states:
            for operations in itertools.product(('same', 'different'), repeat=length):
                signatures.append((initial, operations))
    return signatures

def generate_lamp_pool(seed=PROMPT_SEED, target_size=TARGET_POOL_SIZE):
    rng = random.Random(seed)
    nonces = _nonce_stream(rng)
    signatures = _all_signatures(DOMAIN, *LENGTH_RANGE)
    rng.shuffle(signatures)
    states = DOMAIN_SPECS[DOMAIN]['states']
    pool = []
    for index, (initial, operations) in enumerate(signatures):
        if len(pool) >= target_size:
            break
        first, second = next(nonces), next(nonces)
        if index % 2: first, second = second, first
        mapping = {states[0]: first, states[1]: second}
        expected = tuple(simulate(DOMAIN, initial, operations))
        prompt, demo = _render(DOMAIN, initial, operations, expected, mapping, corrected_fan_wording=False)
        if (initial, operations) in ORIGINAL_LAMP_SIGNATURES:
            continue  # skip any accidental overlap with the held-out eval set
        declared_row = MultiDomainExample(f'lampdiag-{index:05d}', 'lampdiag', DOMAIN, initial,
            operations, expected, first, second, prompt, demo, expected[-1])
        pool.append(remove_declared_mapping(declared_row))
    rng.shuffle(pool)
    return pool

LAMP_POOL = generate_lamp_pool()
_prompts = [row.prompt for row in LAMP_POOL]
assert len(set(_prompts)) == len(LAMP_POOL), 'duplicate prompts generated'
assert len(set(row.example_id for row in LAMP_POOL)) == len(LAMP_POOL)
assert not any('using the code' in p for p in _prompts), 'declared-mapping leakage detected!'
_new_signatures = {(row.initial_state, row.operations) for row in LAMP_POOL}
assert ORIGINAL_LAMP_SIGNATURES.isdisjoint(_new_signatures), 'overlap with the original held-out lamp eval set!'
print({'total': len(LAMP_POOL), 'unique': True, 'leakage': False, 'disjoint_from_original_eval': True})


print('===== LOAD CHECKPOINT 130 (multi-domain) =====')
if not (CHECKPOINT/'adapter_config.json').is_file():
    raise RuntimeError(f'Invalid checkpoint: {CHECKPOINT}')
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left'
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.bfloat16,
    quantization_config=quant,device_map={'':0},low_cpu_mem_usage=True,
    use_safetensors=True,trust_remote_code=False)
model=PeftModel.from_pretrained(base,CHECKPOINT,is_trainable=False)
model.eval(); model.config.use_cache=True
for p in model.parameters(): p.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters())
adapter_files=sorted(p for p in CHECKPOINT.iterdir() if p.name.startswith('adapter_model'))
adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter_files}
print({'adapter_sha256':adapter_sha256,'trainable_parameters':0,'mode':'eval, greedy decoding only'})


print(f'===== GREEDY GENERATION: {len(LAMP_POOL)} LAMP PROMPTS, do_sample=False =====')
SYSTEM='You solve state-tracking tasks accurately and follow the requested output format.'
PROGRESS=OUTPUT_DIR/'raw_completions.json'

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2)); tmp.replace(path)

saved=json.loads(PROGRESS.read_text()) if PROGRESS.is_file() and PROGRESS.stat().st_size else {'rows':[]}
completed={row['example_id'] for row in saved['rows']}
print(f'Resuming: {len(completed)}/{len(LAMP_POOL)} already completed.')

for start in range(0, len(LAMP_POOL), BATCH_SIZE):
    chunk=[row for row in LAMP_POOL[start:start+BATCH_SIZE] if row.example_id not in completed]
    if not chunk: continue
    prompts=[tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM},{'role':'user','content':row.prompt}],
        tokenize=False,add_generation_prompt=True) for row in chunk]
    batch=tokenizer(prompts,return_tensors='pt',padding=True).to(model.device)
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    texts=tokenizer.batch_decode(output[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    for row,text in zip(chunk,texts):
        saved['rows'].append({'example_id':row.example_id,'domain':row.domain,'initial_state':row.initial_state,
            'operations':list(row.operations),'expected_states':list(row.expected_states),
            'final_answer':row.final_answer,'prompt':row.prompt,'completion':text})
        completed.add(row.example_id)
    if len(completed) % CHECKPOINT_EVERY < BATCH_SIZE or len(completed)==len(LAMP_POOL):
        atomic_json(PROGRESS,saved)
        print(f'SAVED: {len(completed)}/{len(LAMP_POOL)}')

atomic_json(PROGRESS,saved)
assert len(saved['rows'])==len(LAMP_POOL) and len(completed)==len(LAMP_POOL)
print(f'\nDONE: {len(saved["rows"])} raw completions saved to {PROGRESS}')
print('LAMP DIAGNOSTIC GENERATION COMPLETE.')
