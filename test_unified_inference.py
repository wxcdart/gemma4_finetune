"""
test_unified_inference.py
Validates the fine-tuned Gemma 4 E2B model across:
- C concurrency & memory verification (c_compile_run)
- Python DSP & ML (python_exec)
- BusyBox container shell automation (busybox_exec)
"""

import torch
from unsloth import FastLanguageModel

MODEL_DIR = "./gemma4_e2b_unified_engine"
MAX_SEQ_LENGTH = 2048

def test_model():
    print(f"Loading fine-tuned model and processor from {MODEL_DIR}...")
    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)

    prompts = [
        "How do I check available disk space on root in a minimal BusyBox container?",
        "Write a thread-safe atomic counter in C and verify it with ThreadSanitizer.",
        "Design a 50Hz notch filter in Python using scipy.signal and test its rejection ratio."
    ]

    for p in prompts:
        print("\n" + "=" * 70)
        print(f"USER QUERY: {p}")
        print("=" * 70)

        # Render turn with Gemma chat template
        messages = [{"role": "user", "content": p}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            use_cache=True,
            temperature=0.2,
            top_p=0.9
        )
        # Decode only generated response
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(response)

if __name__ == "__main__":
    test_model()
