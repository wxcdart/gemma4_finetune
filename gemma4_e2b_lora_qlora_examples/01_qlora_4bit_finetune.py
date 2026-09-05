"""
01_qlora_4bit_finetune.py
-------------------------
Standard Hugging Face + PEFT + bitsandbytes QLoRA (4-bit quantization) fine-tuning example
for Gemma 4 E2B (google/gemma-4-E2B-it or similar Gemma 2B model).

Requirements:
    pip install torch transformers peft bitsandbytes datasets trl accelerate
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MODEL_ID = "google/gemma-2-2b-it"  # Replace with target Gemma model ID e.g., google/gemma-4-E2B-it
DATASET_PATH = "./dataset_sample.jsonl"
OUTPUT_DIR = "./output_qlora_4bit"
MAX_SEQ_LENGTH = 2048

def main():
    print(f"Starting QLoRA (4-bit) Fine-tuning for {MODEL_ID}...")

    # 1. 4-bit Quantization Config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training (freezes base model parameters, handles layer norms)
    model = prepare_model_for_kbit_training(model)

    # 3. Define LoRA Configuration
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

    # 4. Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=20,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="no",
        report_to="none",
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. Start Fine-Tuning
    print("Beginning training loop...")
    trainer.train()

    # 8. Save LoRA Adapters
    print(f"Saving QLoRA adapters to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("QLoRA Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
