# Gemma 4 E2B LoRA & QLoRA Fine-Tuning Guide & Examples (QNZ Engine)

This directory provides complete, end-to-end Python examples for fine-tuning **Gemma 4 E2B** using **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA in 4-bit precision) powered by the **QNZ Engine** (`qnz`, `zero.h`, `zero_cc`, and `zujs`).

---

## Directory Overview

| File | Purpose | Key Libraries |
| :--- | :--- | :--- |
| `01_qlora_4bit_finetune.py` | 4-bit NF4 QLoRA fine-tuning for low memory GPUs (< 8GB VRAM) | `bitsandbytes`, `peft`, `trl` |
| `02_lora_16bit_finetune.py` | Full 16-bit precision LoRA fine-tuning for standard GPUs (16GB - 24GB VRAM) | `peft`, `trl`, `transformers` |
| `03_qnz_lora_qlora.py` | Accelerated LoRA/QLoRA training using QNZ freestanding engine | `qnz`, `zujs` |
| `04_inference.py` | Load fine-tuned LoRA adapters on top of base model for generation | `peft`, `transformers` |
| `05_merge_and_save.py` | Merge adapter weights into base model to produce a standalone model | `peft`, `transformers` |
| `06_qlora_tool_calling.py` | Structured JSON Tool-Calling / Function-Calling QLoRA fine-tuning | `peft`, `bitsandbytes`, `trl` |
| `07_lora_dpo_preference.py` | Direct Preference Optimization (DPO / Alignment) with LoRA | `trl` (`DPOTrainer`), `peft` |
| `dataset_sample.jsonl` | Sample training dataset formatted in Gemma ChatML prompt structure | `jsonl` |

---

## Technical Details: LoRA vs. QLoRA (QNZ Powered)

### 1. QLoRA (4-bit NF4)
- **Quantization**: Base model frozen in 4-bit NormalFloat (NF4) with Double Quantization (`use_double_quant=True`).
- **Memory Footprint**: ~4.5 GB - 6 GB GPU VRAM.
- **Compute Precision**: Adapter activations calculated in `bfloat16` or `float16`.
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### 2. QNZ Engine Optimization
- High-throughput freestanding SIMD AdamW parameter optimization.
- Native C99/C23 foundation via `zero.h` and `zero_cc.h`.
- Embedded JS matrix forward pass scaling via `zujs`.

---

## How to Run

### Step 1: Fine-Tune using QNZ Engine
```bash
python3 03_qnz_lora_qlora.py
```

### Step 2: Standard 4-bit QLoRA
```bash
python3 01_qlora_4bit_finetune.py
```

### Step 3: Run Inference
```bash
python3 04_inference.py
```
