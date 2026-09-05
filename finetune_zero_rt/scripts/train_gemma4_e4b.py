#!/usr/bin/env python3
"""
train_gemma4_e4b.py
QLoRA / LoRA Fine-Tuning Pipeline for Google Gemma 4 E4B on 1x NVIDIA L4 (24GB VRAM).
Trains specialized model in C99, zero_udt, Tiny Pointers, Linux, and Bash.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "google/gemma-4-E4B-it" # or "google/gemma-4-E2B-it"
DATASET_FILE = "/home/coder/workspace/finetune_zero_rt/data/zero_rt_instructions.jsonl"
OUTPUT_DIR = "/home/coder/workspace/finetune_zero_rt/checkpoints/gemma4-e4b-zero-rt"

def main():
    print("===================================================================")
    print(f"  Fine-Tuning Gemma 4 on NVIDIA L4 (24GB VRAM)")
    print(f"  Target: {MODEL_ID} -> Specialization: zero_rt C99 / Linux / Bash")
    print("===================================================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware Activated: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("[1/4] Loading Dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    print(f"[2/4] Initializing Tokenizer & Model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["prompt"], truncation=True, max_length=1024, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt"])

    # Load in 4-bit / 8-bit for efficient training on 24GB L4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("[3/4] Configuring LoRA / QLoRA Adapters on Active Weights...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        bf16=True,
        save_strategy="epoch",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("\n[4/4] Starting Fine-Tuning Execution...")
    trainer.train()
    print(f"\n[✓] Training complete! Saving adapters to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
