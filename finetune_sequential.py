"""
finetune_sequential.py
Performs sequential fine-tuning (stage 2) on top of the already fine-tuned
Gemma 4 E2B model adapters, or merges existing LoRA weights and continues training.
Target: Adding BusyBox / minimal POSIX sh environment automation & tooling skills.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Base model or previously fine-tuned LoRA checkpoint directory
BASE_MODEL_NAME = "google/gemma-4-E2B-it"
STAGE1_ADAPTER_DIR = "./gemma4_e2b_specialized"  # Checkpoint from Stage 1
STAGE2_OUTPUT_DIR = "./gemma4_e2b_specialized_busybox"
DATASET_PATH = "./busybox_train_data.jsonl"
MAX_SEQ_LENGTH = 2048

def train():
    # If Stage 1 checkpoint exists, load and continue adapting; otherwise fallback to base
    model_to_load = STAGE1_ADAPTER_DIR if os.path.exists(STAGE1_ADAPTER_DIR) else BASE_MODEL_NAME
    print(f"[1/4] Loading checkpoint: {model_to_load}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # If loading fresh base model, configure PEFT; if continuing from Stage 1,
    # FastLanguageModel automatically keeps adapter trainable
    if model_to_load == BASE_MODEL_NAME:
        print("[2/4] Applying LoRA / PEFT adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print("[2/4] Continuing training on existing LoRA weights...")

    print(f"[3/4] Loading BusyBox training dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Lower learning rate (e.g. 5e-5 or 1e-4) to prevent catastrophic forgetting
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=40,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=STAGE2_OUTPUT_DIR,
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

    print("[4/4] Starting sequential fine-tuning for BusyBox...")
    trainer.train()

    print(f"Saving updated adapter weights to {STAGE2_OUTPUT_DIR}...")
    model.save_pretrained(STAGE2_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)
    print("Stage 2 BusyBox training complete!")

if __name__ == "__main__":
    train()
