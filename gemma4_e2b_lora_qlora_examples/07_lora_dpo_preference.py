"""
07_lora_dpo_preference.py
-------------------------
Direct Preference Optimization (DPO) fine-tuning with PEFT LoRA adapters.
Aligns model responses based on preference data pairs (prompt, chosen, rejected).

Requirements:
    pip install torch transformers peft trl datasets bitsandbytes
"""

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

MODEL_ID = "google/gemma-2-2b-it"
OUTPUT_DIR = "./output_lora_dpo"

# Sample preference dataset (Prompt, Chosen response, Rejected response)
DPO_DATA = [
    {
        "prompt": "<bos><start_of_turn>user\nWrite a clean Python function to filter even numbers.<end_of_turn>\n<start_of_turn>model\n",
        "chosen": "def filter_evens(numbers):\n    return [num for num in numbers if num % 2 == 0]\n<end_of_turn>",
        "rejected": "def filter_evens(numbers):\n    res = []\n    for i in range(len(numbers)):\n        if numbers[i] % 2 == 0:\n            res.append(numbers[i])\n    return res\n<end_of_turn>"
    }
]

def main():
    print(f"Starting DPO Alignment with LoRA adapters for {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    dataset = Dataset.from_list(DPO_DATA)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        max_steps=5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles reference model automatically
        peft_config=peft_config,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=256,
    )

    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"DPO LoRA alignment finished! Adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
