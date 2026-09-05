"""
06_qlora_tool_calling.py
------------------------
Specialized QLoRA fine-tuning script for Gemma models on Tool-Calling / Function-Calling tasks.
Teaches the model to format structured JSON function arguments upon system request.

Requirements:
    pip install torch transformers peft bitsandbytes datasets trl
"""

import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR = "./output_qlora_tool_use"

# Tool calling prompt dataset
TOOL_CALLING_DATA = [
    {
        "text": "<bos><start_of_turn>user\n[TOOL_REQUEST] Search the weather forecast for San Francisco tomorrow.\nAvailable tools: {\"name\": \"get_weather\", \"parameters\": {\"location\": \"str\", \"days\": \"int\"}}<end_of_turn>\n<start_of_turn>model\n<call:get_weather>{\"location\": \"San Francisco\", \"days\": 1}</call><end_of_turn>"
    },
    {
        "text": "<bos><start_of_turn>user\n[TOOL_REQUEST] Calculate the square root of 256.\nAvailable tools: {\"name\": \"calculator\", \"parameters\": {\"expression\": \"str\"}}<end_of_turn>\n<start_of_turn>model\n<call:calculator>{\"expression\": \"sqrt(256)\"}</call><end_of_turn>"
    }
]

def main():
    print(f"Starting QLoRA Tool-Calling fine-tuning for {MODEL_ID}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    dataset = Dataset.from_list(TOOL_CALLING_DATA)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        logging_steps=1,
        max_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tool-calling QLoRA fine-tuning completed! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
