from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.coinflip import generate_dataset, make_grpo_reward_fn, score_completion

os.environ['WANDB_DISABLED'] = 'true'

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

raw_examples = generate_dataset(40, (2, 6))
prompt_texts = [prompt for prompt, _ in raw_examples]
prompts = [[{'role': 'user', 'content': prompt}] for prompt in prompt_texts]
answers = [answer for _, answer in raw_examples]

train_dataset = Dataset.from_dict({'prompt': prompts, 'ground_truth': answers})
train_dataset = train_dataset.shuffle(seed=42).select(range(min(20, len(train_dataset))))
print('dataset size', len(train_dataset))

model_name = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
if not tokenizer.chat_template:
    raise RuntimeError(f'{model_name} tokenizer does not provide a chat template')

quantization_config = None
if torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=False,
)
print('tokenizer:', tokenizer.name_or_path)
print('model:', model.config._name_or_path)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

reward_callable = make_grpo_reward_fn(debug=True)
output_dir = str(REPO_ROOT / 'artifacts' / 'grpo-coinflip-dryrun')
os.makedirs(output_dir, exist_ok=True)

training_args = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=3,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=1,
    save_steps=1,
    save_total_limit=3,
    num_generations=16,
    max_completion_length=120,
    temperature=0.8,
    top_p=0.95,
    beta=0.04,
    report_to='none',
    remove_unused_columns=False,
    disable_dropout=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_callable,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()
checkpoint_path = os.path.join(output_dir, 'checkpoint-1')
print('checkpoint exists', os.path.exists(checkpoint_path))

# Re-run three deterministic diagnostics using exactly Qwen's instruction
# template, and decode only newly generated tokens (never the input prompt).
sample_count = min(3, len(train_dataset))
sample_rows = train_dataset.select(range(sample_count))
sample_messages = sample_rows['prompt']
sample_answers = sample_rows['ground_truth']
formatted_prompts = [
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for messages in sample_messages
]
sample_inputs = tokenizer(
    formatted_prompts,
    return_tensors='pt',
    padding=True,
    truncation=True,
).to(model.device)

model.eval()
with torch.no_grad():
    sample_outputs = model.generate(
        **sample_inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

input_width = sample_inputs['input_ids'].shape[1]
sample_completions = tokenizer.batch_decode(
    sample_outputs[:, input_width:],
    skip_special_tokens=True,
)
for idx, (messages, completion, answer) in enumerate(
    zip(sample_messages, sample_completions, sample_answers),
    start=1,
):
    breakdown = score_completion(
        completion,
        answer,
        step=int(trainer.state.global_step),
        prompt=messages,
    )
    print(f'\n===== SAMPLE {idx} =====')
    print('prompt messages:', messages)
    print('formatted prompt:', repr(formatted_prompts[idx - 1]))
    print('ground truth:', repr(answer))
    print('raw completion:', repr(completion))
    print('reward breakdown:', breakdown)
