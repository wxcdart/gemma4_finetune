"""
05_merge_and_save.py
--------------------
Merge fine-tuned LoRA adapter weights into the base Gemma model and save
as a standalone 16-bit model (HF format).

Requirements:
    pip install torch transformers peft accelerate
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
BASE_MODEL_ID = "google/gemma-2-2b-it"  # Base Gemma model ID
ADAPTER_DIR = "./output_qlora_4bit"     # Saved adapter directory
MERGED_OUTPUT_DIR = "./gemma4_e2b_merged_model"

def main():
    print(f"Loading base model {BASE_MODEL_ID} in full precision for merging...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    # Base model must be loaded in 16-bit/32-bit (not 4-bit) for merge_and_unload
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="cpu",  # Load to CPU or GPU for merging
        trust_remote_code=True,
    )

    print(f"Loading adapter weights from {ADAPTER_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged standalone model to {MERGED_OUTPUT_DIR}...")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print("Successfully merged and saved standalone model!")

if __name__ == "__main__":
    main()
