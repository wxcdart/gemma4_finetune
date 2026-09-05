"""
merge_and_push_f16.py
---------------------
Directly loads base Gemma model + fine-tuned LoRA adapter, merges weights in 16-bit precision,
and uploads standalone safetensors to Hugging Face: wxcdart/gemma4-e2b-unified-engine-f16
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_DIR = "/home/coder/workspace/gemma4_finetune/gemma4_e2b_unified_engine"
F16_REPO_NAME = "wxcdart/gemma4-e2b-unified-engine-f16"
LOCAL_MERGED_DIR = "/home/coder/workspace/gemma4_finetune/gemma4_e2b_merged_f16"

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else HF_TOKEN
    if not token:
        print("Error: Missing HF_TOKEN environment variable or command line argument.")
        sys.exit(1)

    print("==================================================")
    print("   Merging Gemma 4 E2B to 16-bit Standalone Model  ")
    print("==================================================\n")

    print("[1/4] Loading base model and tokenizer in 16-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=token,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    print("[2/4] Attaching LoRA adapter and merging weights...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged_model = peft_model.merge_and_unload()

    print(f"[3/4] Saving 16-bit merged model to {LOCAL_MERGED_DIR}...")
    merged_model.save_pretrained(LOCAL_MERGED_DIR)
    tokenizer.save_pretrained(LOCAL_MERGED_DIR)

    print(f"[4/4] Uploading F16 merged model to Hugging Face ({F16_REPO_NAME})...")
    login(token=token)
    api = HfApi(token=token)
    api.create_repo(repo_id=F16_REPO_NAME, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=LOCAL_MERGED_DIR,
        repo_id=F16_REPO_NAME,
        repo_type="model"
    )

    print(f"\n[OK] F16 model successfully uploaded to: https://huggingface.co/{F16_REPO_NAME}")

if __name__ == "__main__":
    main()
