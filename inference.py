"""
inference.py
Inference & test script to prompt the fine-tuned Gemma 4 E2B model across all targeted domains.
"""

import torch
from unsloth import FastLanguageModel

MODEL_DIR = "./gemma4_e2b_specialized"
MAX_SEQ_LENGTH = 2048

def run_test(prompt: str, model, tokenizer):
    FastLanguageModel.for_inference(model)
    formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer([formatted], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.2,
        top_p=0.95
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(f"PROMPT: {prompt}")
    print("=" * 60)
    print(decoded)

def main():
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    test_queries = [
        "Write a C function to implement a lock-free single-producer single-consumer ring buffer using atomic operations.",
        "How do I train an audio classification model in PyTorch using STFT spectrograms?",
        "Write an asyncio script in Python that concurrently pings a list of Redis workers and restarts any worker that times out.",
        "Show a PyTorch FSDP (Fully Sharded Data Parallel) training setup script."
    ]

    for q in test_queries:
        run_test(q, model, tokenizer)

if __name__ == "__main__":
    main()
