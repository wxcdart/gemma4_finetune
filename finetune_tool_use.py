"""
finetune_tool_use.py
Trains Gemma 4 E2B to reliably invoke and parse tool calls (BusyBox execution,
system tools) using Unsloth and custom chat tokens (<call:name>, <response:name>).
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

BASE_OR_ADAPTER_PATH = "./gemma4_e2b_specialized" # Can start from Stage 1 adapter or google/gemma-4-E2B-it
OUTPUT_DIR = "./gemma4_e2b_tool_calling"
DATASET_PATH = "./tool_calling_train_data.jsonl"
MAX_SEQ_LENGTH = 2048

def train():
    target_path = BASE_OR_ADAPTER_PATH if os.path.exists(BASE_OR_ADAPTER_PATH) else "google/gemma-4-E2B-it"
    print(f"[1/4] Loading model from {target_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=target_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Add tool-calling control tokens as special tokens to preserve delimiter boundary precision
    special_tokens = {
        "additional_special_tokens": [
            "<start_of_turn>", "<end_of_turn>",
            "<thought>", "</thought>",
            "<call:busybox_exec>", "</call:busybox_exec>",
            "<response:busybox_exec>", "</response:busybox_exec>"
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        print(f"Added {num_added} special tool tokens; resizing token embeddings...")
        model.resize_token_embeddings(len(tokenizer))

    if target_path == "google/gemma-4-E2B-it":
        print("[2/4] Initializing PEFT adapter...")
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

    print(f"[3/4] Loading tool calling dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=50,
        learning_rate=1.5e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
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

    print("[4/4] Starting Tool Calling Fine-Tuning...")
    trainer.train()

    print(f"Saving final model with tool tokens to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Tool calling fine-tuning completed successfully!")

if __name__ == "__main__":
    train()
