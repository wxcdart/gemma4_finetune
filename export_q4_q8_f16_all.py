"""
export_q4_q8_f16_all.py
-----------------------
Exports and uploads fine-tuned Gemma 4 E2B model in:
1. F16 / Merged 16-bit standalone Safetensors: wxcdart/gemma4-e2b-unified-engine-f16
2. Q4_K_M and Q8_0 GGUF Quantizations: wxcdart/gemma4-e2b-unified-engine-gguf
"""

import os
import sys
import torch
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")

HF_USERNAME = "wxcdart"
F16_REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine-f16"
GGUF_REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine-gguf"
MODEL_DIR = "./gemma4_e2b_unified_engine"
MAX_SEQ_LENGTH = 2048

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else HF_TOKEN
    if not token:
        print("Error: Missing HF_TOKEN environment variable or command line argument.")
        sys.exit(1)
        
    os.environ["HF_TOKEN"] = token

    print("==================================================")
    print("   Exporting Gemma 4 E2B Model: F16, Q4, Q8       ")
    print("==================================================\n")

    print(f"[*] Loading model from {MODEL_DIR}...")
    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    # 1. Export & Push Merged F16 Standalone Model
    print(f"\n[1/2] Merging weights and pushing F16 model to {F16_REPO_NAME}...")
    try:
        model.push_to_hub_merged(
            F16_REPO_NAME,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=token
        )
        print(f"-> Successfully uploaded F16 model to https://huggingface.co/{F16_REPO_NAME}")
    except Exception as e:
        print(f"Note on push_to_hub_merged: {e}. Executing manual local merge and upload...")
        merged_dir = "./gemma4_e2b_merged_f16"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        api = HfApi(token=token)
        api.create_repo(repo_id=F16_REPO_NAME, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=F16_REPO_NAME)
        print(f"-> Successfully uploaded F16 model to https://huggingface.co/{F16_REPO_NAME}")

    # 2. Export & Push Q4_K_M and Q8_0 GGUF Models
    print(f"\n[2/2] Quantizing and pushing Q4_K_M and Q8_0 GGUF to {GGUF_REPO_NAME}...")
    quant_methods = ["q4_k_m", "q8_0"]
    model.push_to_hub_gguf(
        GGUF_REPO_NAME,
        tokenizer=tokenizer,
        quantization_method=quant_methods,
        token=token
    )
    print(f"-> Successfully uploaded Q4 & Q8 GGUF models to https://huggingface.co/{GGUF_REPO_NAME}")

    print("\n[OK] F16, Q4, and Q8 model exports and uploads finished successfully!")

if __name__ == "__main__":
    main()
