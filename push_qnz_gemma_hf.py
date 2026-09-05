"""
push_qnz_gemma_hf.py
--------------------
Upload fine-tuned Gemma 4 E2B model adapters & tokenizer to Hugging Face Hub.
Repo: wxcdart/gemma4-e2b-unified-engine
"""

import os
import sys
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "wxcdart/gemma4-e2b-unified-engine"
MODEL_DIR = "/home/coder/workspace/gemma4_finetune/gemma4_e2b_unified_engine"

def main():
    token = sys.argv[1] if len(sys.argv) > 1 else HF_TOKEN
    if not token:
        print("Error: Missing HF_TOKEN environment variable or command line argument.")
        sys.exit(1)

    print(f"[*] Logging in to Hugging Face Hub...")
    login(token=token)
    api = HfApi(token=token)

    print(f"[*] Creating repository '{REPO_ID}' (if not already existing)...")
    api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")

    print(f"[*] Uploading folder '{MODEL_DIR}' to Hugging Face Hub...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"\n[OK] Model successfully published to: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
