"""
04_inference.py
---------------
Script to load fine-tuned LoRA / QLoRA adapter weights on top of base Gemma model
and run inference on prompts.

Requirements:
    pip install torch transformers peft accelerate bitsandbytes
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
BASE_MODEL_ID = "google/gemma-2-2b-it"  # Base model path or HF ID
ADAPTER_DIR = "./output_qlora_4bit"     # Path to saved LoRA checkpoint

def main():
    print(f"Loading base model {BASE_MODEL_ID} and adapter from {ADAPTER_DIR}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    # Load base model in bfloat16 / float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Attach PEFT adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    prompt = "<bos><start_of_turn>user\nWrite a Python function to check if a string is a palindrome.<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    print("\n--- Generating response ---")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(response)

if __name__ == "__main__":
    main()
