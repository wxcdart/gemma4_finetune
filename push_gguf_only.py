"""
push_gguf_only.py
Converts the trained model to GGUF and uploads to wxcdart/gemma4-e2b-unified-engine-gguf
"""

import os
import sys
from unsloth import FastLanguageModel

HF_USERNAME = "wxcdart"
GGUF_REPO_NAME = f"{HF_USERNAME}/gemma4-e2b-unified-engine-gguf"
MODEL_DIR = "./gemma4_e2b_unified_engine"
MAX_SEQ_LENGTH = 2048

def run_gguf_export(token):
    os.environ["HF_TOKEN"] = token
    print(f"[*] Loading model from {MODEL_DIR}...")
    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    print(f"\n[*] Exporting and pushing GGUF quants to {GGUF_REPO_NAME}...")
    quant_methods = ["q4_k_m", "q8_0"]
    print(f"Target quantizations: {quant_methods}")
    model.push_to_hub_gguf(
        GGUF_REPO_NAME,
        tokenizer=tokenizer,
        quantization_method=quant_methods,
        token=token
    )
    print(f"\n-> Successfully uploaded GGUF to https://huggingface.co/{GGUF_REPO_NAME}")

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Missing token")
        sys.exit(1)
    run_gguf_export(token)
