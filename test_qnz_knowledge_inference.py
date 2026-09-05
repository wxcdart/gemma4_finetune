"""
test_qnz_knowledge_inference.py
--------------------------------
Inference script to test if the fine-tuned `wxcdart/gemma4-e2b-unified-engine` model
has specialized knowledge of QNZ, zero.h, zero_cc, zujs, zero_df, and zero_signal.

Execution:
    python3 test_qnz_knowledge_inference.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LOCAL_ADAPTER = "/home/coder/workspace/gemma4_finetune/gemma4_e2b_unified_engine"
MODEL_REPO = LOCAL_ADAPTER if os.path.exists(LOCAL_ADAPTER) else "wxcdart/gemma4-e2b-unified-engine"
BASE_MODEL = "google/gemma-2-2b-it"

# Prompts testing QNZ specialized knowledge
TEST_PROMPTS = [
    "Explain what QNZ zero.h is and how to create a ZeroArena allocator in C.",
    "Write a C code snippet using zero_df to parse a CSV string into a DataFrame.",
    "Describe the zujs JavaScript engine and how it executes freestanding QLoRA fine-tuning."
]

def main():
    print("==================================================================")
    print("   Testing QNZ Knowledge in wxcdart/gemma4-e2b-unified-engine     ")
    print("==================================================================\n")

    token = os.environ.get("HF_TOKEN")
    print("[1/3] Loading Tokenizer and Base Model in 16-bit precision...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=token,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"[2/3] Attaching fine-tuned adapter weights from '{MODEL_REPO}'...")
    model = PeftModel.from_pretrained(base_model, MODEL_REPO)
    model.eval()

    print("\n[3/3] Generating Responses for QNZ Knowledge Verification:\n")

    for i, prompt_text in enumerate(TEST_PROMPTS, 1):
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        print(f"--- Prompt #{i}: \"{prompt_text}\" ---")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Model Response:\n{response.strip()}\n")
        print("=" * 66 + "\n")

if __name__ == "__main__":
    main()
