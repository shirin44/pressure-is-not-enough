"""Deterministic, CPU-only confirmation of prompt-token-id format for each construction path found in the Qwen-era code
(no training/eval code modified or executed; tokenizer-only). Uses the actual prompt-building functions from the repo."""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/researcher/Desktop/AISI')
sys.path.insert(0, '/Users/researcher/Desktop/AISI/src/data')
sys.path.insert(0, '/Users/researcher/Desktop/AISI/experiments/09_direct_indomain_synthetic_bridge')
from transformers import AutoTokenizer
from coinflip import generate_coinflip_example
from synthetic_bridge import build_prompt as build_cot_prompt

for name in ('Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    p1, _ = generate_coinflip_example(3, 42)
    p2 = build_cot_prompt('Heads', ['same', 'different', 'same'])
    for label, p in (('coinflip.py prompt', p1), ('synthetic_bridge.py build_cot_prompt', p2)):
        raw_ids = tok(p)['input_ids']  # exactly Dataset.from_list({'prompt': p}) -> GRPOTrainer's plain-string path
        chat_text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
        chat_ids = tok(chat_text)['input_ids']  # exactly what eval / chat_wrap / teacher_forced_ce prefix feeds
        print(f'{name} | {label}')
        print(f'  RAW  (train_dataset[\"prompt\"] tokenised as-is): first8={raw_ids[:8]}  decoded_first3={[tok.decode([i]) for i in raw_ids[:3]]}')
        print(f'  CHAT (apply_chat_template, as SFT/teacher_forced_ce/eval use): first8={chat_ids[:8]}  decoded_first3={[tok.decode([i]) for i in chat_ids[:3]]}')
        print(f'  identical for first 8 ids: {raw_ids[:8] == chat_ids[:8]}')
