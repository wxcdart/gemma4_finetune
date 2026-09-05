"""
03_qnz_lora_qlora.py
--------------------
Fast fine-tuning script for Gemma 4 E2B using the QNZ engine (QNZ / zujs / zero ecosystem).
Executes freestanding QLoRA matrix operations and SIMD AdamW parameter optimization.

Execution:
    python3 03_qnz_lora_qlora.py
"""

import os
import subprocess
import sys

QNZ_FINETUNE_JS = "/home/coder/workspace/qnz/tools/zusloth_finetune.js"

def main():
    print("==================================================")
    print("   Gemma 4 E2B QLoRA Fine-Tuning - QNZ Engine     ")
    print("==================================================\n")

    if not os.path.exists(QNZ_FINETUNE_JS):
        print(f"Error: QNZ engine fine-tuner not found at {QNZ_FINETUNE_JS}")
        sys.exit(1)

    zujs_bin = "/home/coder/workspace/qnz/bin/zujs"
    if not os.path.exists(zujs_bin):
        zujs_bin = "node"

    print("[1/2] Invoking QNZ Engine Freestanding SLM Fine-Tuner...")
    try:
        res = subprocess.run([zujs_bin, QNZ_FINETUNE_JS], capture_output=True, text=True, check=True)
        print("-------------------- QNZ STDOUT --------------------")
        print(res.stdout.strip())
        print("----------------------------------------------------")
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)

    print("\n[2/2] QNZ QLoRA Fine-Tuning benchmark completed successfully!")

if __name__ == "__main__":
    main()
