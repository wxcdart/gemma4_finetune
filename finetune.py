"""
finetune.py
Fine-tuning script for Google Gemma 4 E2B using Unsloth.
Supports LoRA / QLoRA 4-bit and 16-bit parameter efficient fine-tuning.
Target tasks: C, Python ML, DL, NLP, SP, Distributed Programming, Automation.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# Gemma 4 E2B (2.3B effective parameters, 128k context)
MODEL_NAME = "google/gemma-4-E2B-it"  # or unsloth quantized version if available
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # Set False for 16-bit LoRA (fits easily on 24GB NVIDIA L4)
DTYPE = None  # None for auto detection (bfloat16 on Ampere/Ada/Hopper)

OUTPUT_DIR = "./gemma4_e2b_specialized"
DATASET_PATH = "./train_data.jsonl"

def train():
    print(f"[1/4] Loading model {MODEL_NAME} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("[2/4] Applying LoRA / PEFT adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Optimized to 0 for Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"[3/4] Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("[4/4] Starting training...")
    trainer_stats = trainer.train()
    print(f"Training completed! Loss stats: {trainer_stats}")

    # Save fine-tuned LoRA adapters and tokenizer
    print(f"Saving fine-tuned adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    train()
