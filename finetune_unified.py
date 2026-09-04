"""
finetune_unified.py
Trains Gemma 4 E2B with full unified tool-calling support across:
- C execution & sanitizer checks (c_compile_run)
- Python ML/DL/DSP analysis (python_exec)
- BusyBox container shell automation (busybox_exec)

Compatibility & Architecture Notes:
- Preserves native Gemma 4 Per-Layer Embedding (PLE) dimensions (no out-of-vocab resizing).
- Uses standard structured text delimiters natively tokenized by Gemma's vocabulary.
- Implements frequent checkpointing (save_steps=10) and auto-resumes from last checkpoint.
- Fixes Jinja chat template for multi-turn reasoning and tool invocation.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

BASE_MODEL = "google/gemma-4-E2B-it"
OUTPUT_DIR = "./checkpoints_gemma4_e2b"
FINAL_MODEL_DIR = "./gemma4_e2b_unified_engine"
DATASET_PATH = "./unified_tool_calling_data.jsonl"
JINJA_TEMPLATE_PATH = "./gemma_chat_template.jinja"
MAX_SEQ_LENGTH = 2048

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else None
    if last_checkpoint:
        print(f"[*] Detected existing checkpoint: {last_checkpoint}. Resuming from here...")
    else:
        print(f"[*] No checkpoint found in {OUTPUT_DIR}. Starting from base model: {BASE_MODEL}...")

    print(f"[1/4] Loading model: {BASE_MODEL} with FastLanguageModel...")
    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    if hasattr(processor_or_tokenizer, "tokenizer"):
        tokenizer = processor_or_tokenizer.tokenizer
    else:
        tokenizer = processor_or_tokenizer

    # Apply corrected Jinja chat template without resizing embeddings
    if os.path.exists(JINJA_TEMPLATE_PATH):
        with open(JINJA_TEMPLATE_PATH, "r") as f:
            template_str = f.read()
            tokenizer.chat_template = template_str
            if hasattr(processor_or_tokenizer, "chat_template"):
                processor_or_tokenizer.chat_template = template_str
        print("  -> Applied corrected gemma_chat_template.jinja")

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

    print(f"[3/4] Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
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

    print(f"[4/4] Starting training (resume_from_checkpoint={last_checkpoint is not None})...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving final model and processor/tokenizer to {FINAL_MODEL_DIR}...")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    processor_or_tokenizer.save_pretrained(FINAL_MODEL_DIR)
    if hasattr(processor_or_tokenizer, "tokenizer"):
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("Training complete and final checkpoint saved!")

if __name__ == "__main__":
    train()
