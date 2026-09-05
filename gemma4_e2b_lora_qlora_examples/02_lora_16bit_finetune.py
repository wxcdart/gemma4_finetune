"""
02_lora_16bit_finetune.py
-------------------------
Standard 16-bit LoRA fine-tuning example for Gemma 4 E2B without 4-bit quantization.
Suitable for high precision training when full VRAM (or bfloat16/fp16) is available.

Requirements:
    pip install torch transformers peft datasets trl accelerate
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"  # Replace with target Gemma model ID e.g., google/gemma-4-E2B-it
DATASET_PATH = "./dataset_sample.jsonl"
OUTPUT_DIR = "./output_lora_16bit"
MAX_SEQ_LENGTH = 2048

def main():
    print(f"Starting 16-bit LoRA Fine-tuning for {MODEL_ID}...")

    # 1. Load Tokenizer & Base Model in 16-bit precision (bfloat16 or float16)
    print("Loading model and tokenizer in 16-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. Define LoRA Adapter Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=20,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="no",
        report_to="none",
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 6. Start Fine-Tuning
    print("Beginning training loop...")
    trainer.train()

    # 7. Save Adapters
    print(f"Saving LoRA 16-bit adapters to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("16-bit LoRA Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
