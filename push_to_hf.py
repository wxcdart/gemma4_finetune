"""
push_to_hf.py
Uploads the fine-tuned Gemma 4 E2B unified tool-calling model:
1. LoRA adapter & Tokenizer
2. Merged 16-bit Full Safetensors (Transformers-compatible standalone model)
3. GGUF Quantizations (q4_k_m, q8_0)
Repo: wxcdart/gemma4-e2b-unified-engine
"""

import os
import sys
from unsloth import FastLanguageModel

HF_USERNAME = "wxcdart"
REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine"
MERGED_REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine-safetensors"
GGUF_REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine-gguf"
MODEL_DIR = "./gemma4_e2b_unified_engine"
MAX_SEQ_LENGTH = 2048

def export_and_upload(token=None):
    if token:
        os.environ["HF_TOKEN"] = token
    elif "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"].strip():
        token = input("Enter Hugging Face Write Token: ").strip()
        os.environ["HF_TOKEN"] = token

    print(f"[*] Loading model from {MODEL_DIR}...")
    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    # 1. Push LoRA adapter and tokenizer
    print(f"\n[1/3] Pushing LoRA adapter and tokenizer to {REPO_NAME}...")
    model.push_to_hub(REPO_NAME, token=os.environ["HF_TOKEN"])
    tokenizer.push_to_hub(REPO_NAME, token=os.environ["HF_TOKEN"])
    if hasattr(processor_or_tokenizer, "push_to_hub"):
        try:
            processor_or_tokenizer.push_to_hub(REPO_NAME, token=os.environ["HF_TOKEN"])
        except Exception as e:
            print(f"Note: Processor push note: {e}")
    print(f"-> Successfully uploaded LoRA to https://huggingface.co/{REPO_NAME}")

    # 2. Push full merged safetensors version (16-bit standalone)
    print(f"\n[2/3] Merging LoRA into base weights and pushing full Safetensors to {MERGED_REPO_NAME}...")
    try:
        model.push_to_hub_merged(
            MERGED_REPO_NAME,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"]
        )
        print(f"-> Successfully uploaded Safetensors to https://huggingface.co/{MERGED_REPO_NAME}")
    except Exception as e:
        print(f"Note on push_to_hub_merged: {e}. Trying save_pretrained_merged then upload...")
        merged_dir = "./gemma4_e2b_merged_16bit"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id=MERGED_REPO_NAME, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=MERGED_REPO_NAME)
        print(f"-> Successfully uploaded Safetensors to https://huggingface.co/{MERGED_REPO_NAME}")

    # 3. Convert and push GGUF formats
    print(f"\n[3/3] Exporting and pushing GGUF quants to {GGUF_REPO_NAME}...")
    quant_methods = ["q4_k_m", "q8_0"]
    print(f"Target quantizations: {quant_methods}")
    model.push_to_hub_gguf(
        GGUF_REPO_NAME,
        tokenizer=tokenizer,
        quantization_method=quant_methods,
        token=os.environ["HF_TOKEN"]
    )
    print(f"-> Successfully uploaded GGUF to https://huggingface.co/{GGUF_REPO_NAME}")

if __name__ == "__main__":
    cli_token = sys.argv[1] if len(sys.argv) > 1 else None
    export_and_upload(cli_token)
