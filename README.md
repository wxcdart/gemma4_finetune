# Fine-Tuning Google Gemma 4 E2B with Unsloth

This repository contains the complete pipeline to fine-tune Google's **Gemma 4 E2B** model (2.3B effective / 5.1B total parameters with Per-Layer Embeddings and 128k context window) for **unified tool-use, chain-of-thought reasoning, and domain-specialized engineering tasks**.

Supported domains and tool execution capabilities:
1. **C Systems Programming & ThreadSanitizer**: Memory diagnostics, race condition detection, and POSIX concurrency.
2. **Containerized Linux / BusyBox Execution**: Safe execution of shell diagnostics, file manipulation, and sandboxed testing.
3. **Signal Processing & DSP**: Digital filtering (Butterworth/Chebyshev), FFT, and spectrogram analysis using SciPy and Librosa.
4. **Machine Learning & Deep Learning**: PyTorch architectures, scikit-learn, XGBoost pipelines, and optimization routines.
5. **Distributed Programming & Automation**: PyTorch DDP/FSDP, asyncio networking, and workflow scripting.

---

## 1. Environment & Hardware Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (NVIDIA L4 24GB, A10G 24GB, RTX 3090/4090, or A100 recommended).
- **CUDA**: Version 12.1 to 12.6+ supported by PyTorch 2.6.
- **System Memory (RAM)**: Minimum 32GB RAM (recommended 64GB RAM when compiling or quantizing large Per-Layer Embedding models to GGUF).
- **Disk Space**: At least 35GB of free space for weights, checkpoints, and GGUF conversions.

### System Packages (Linux / Ubuntu)
Ensure base build tools and `git-lfs` are installed:
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    curl \
    python3-dev \
    python3-pip \
    python3-venv
```

---

## 2. Python Environment Setup

We recommend using Python 3.10, 3.11, or 3.12 within a virtual environment.

### Option A: Create Virtual Environment (venv)
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### Option B: Conda / Mamba Environment
```bash
conda create -n gemma4 python=3.12 -y
conda activate gemma4
```

### Install Dependencies
Install PyTorch (CUDA 12.4 / 12.6) and Unsloth with its optimized dependencies:

```bash
# 1. Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install Unsloth and Hugging Face ecosystem
pip install unsloth unsloth_zoo
pip install "transformers>=5.0.0" trl peft accelerate bitsandbytes datasets xformers

# 3. Install domain tool & validation libraries
pip install scipy numpy librosa scikit-learn pydantic huggingface_hub

# 4. torchao version notes:
# - During fine-tuning/training: a lower version (e.g. torchao==0.7.0) is typically used for PyTorch 2.6.0 stability.
# - During GGUF export / quantization: Unsloth and unsloth_zoo require torchao > 0.7 (recommended: torchao==0.9.0):
pip install torchao==0.9.0
```

### Hugging Face Authentication
To download Google Gemma 4 base weights or push fine-tuned models:
1. Accept the Gemma 4 license on [Hugging Face](https://huggingface.co/google/gemma-4-E2B-it).
2. Login using your Hugging Face user access token:
```bash
huggingface-cli login
# Or export the environment variable:
export HF_TOKEN="hf_your_token_here"
```

---

## 3. Dataset Architecture & Scaling Roadmap

### Current Dataset Overview (Proof-of-Concept Scaffold)
The repository currently includes a curated, high-signal seed dataset in [`prepare_unified_tool_dataset.py`](prepare_unified_tool_dataset.py) (`unified_tool_calling_data.jsonl`).

> [!NOTE]
> **Dataset Status & Sparsity Notice**:
> The seed dataset is intentionally lightweight (3 multi-turn conversations) designed to serve as a **structural format-adaptation scaffold** rather than a full pretraining corpus. It verifies that Gemma 4 E2B can learn the exact token syntax (`<thought>`, `<call:tool_name>`, `<response:tool_name>`), converge with Unsloth QLoRA, and cleanly export to Safetensors and GGUF without corruption.

Generate the seed unified dataset:
```bash
python3 prepare_unified_tool_dataset.py
```
This produces:
- `unified_tool_calling_data.jsonl`: Formatted conversational turns for training.
- `unified_tools_schema.json`: Standard JSON Schema definitions for tools (`busybox_exec`, `python_exec`, `c_compile_run`).

### Scaling to Production (2,000 – 10,000 Examples)

For production deployment and generalized zero-shot tool usage, the dataset should be scaled to a minimum of **2,000 to 5,000 examples** across the following distribution:

| Category | Recommended Volume | Role & Objective |
| :--- | :--- | :--- |
| **Multi-Turn Tool Loops** | 1,500 – 2,500 | Multi-step agent trajectories (e.g. compile C code $\rightarrow$ analyze sanitizer output $\rightarrow$ apply fix). |
| **Error Recovery & Diagnostics** | 800 – 1,200 | Teaches the model to handle compilation errors, runtime exceptions, and non-zero exit codes. |
| **Direct Answers (No Tool)** | 1,000 – 1,500 | Teaches the model *when not to invoke a tool* for standard conversational or conceptual queries. |
| **Safety & Command Rejections** | 300 – 500 | Rejects destructive commands (`rm -rf /`, dangerous fork bombs, raw block device writes). |
| **Cross-Domain Coverage** | 1,500+ | Balanced coverage across C/TSan, Python ML/DSP, container shell commands, and distributed pipelines. |

#### Data Acquisition Strategies:
1. **Synthetic Generation via Frontier Models**: Automated prompt-driven generation of multi-turn tool calling traces adhering to [`unified_tools_schema.json`](unified_tools_schema.json).
2. **Open-Source Dataset Adaptation**: Converting and filtering programming/system subsets from **Glaive Function Calling v2** and **ToolBench** into Gemma 4's native `<start_of_turn>` and `<call:...>` tokens.
3. **Execution-Validated Self-Play**: Running generated Python, C, and Shell code in isolated sandboxes to capture authentic stdout/stderr responses.

---

## 4. Training (Fine-Tuning)

Execute fine-tuning using Unsloth's QLoRA engine:
```bash
python3 finetune_unified.py
```

### Key Training Highlights:
- **Base Model**: `google/gemma-4-E2B-it` loaded in 4-bit (`load_in_4bit=True`).
- **LoRA Config**: `r=16`, `lora_alpha=16`, targeting all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Optimizer**: 8-bit AdamW (`adamw_8bit`) with linear learning rate scheduling and `bfloat16` mixed precision.
- **Output Artifacts**: Checkpoints stored in `./checkpoints_gemma4_e2b/` and the final merged adapter saved to `./gemma4_e2b_unified_engine/`.

---

## 5. Inference & Validation

Run inference to verify that the fine-tuned model correctly produces structured reasoning and valid tool calls:

```bash
python3 test_unified_inference.py
```

Expected output includes chain-of-thought `<thought>` reasoning blocks followed by executable tool actions like:
```
call:run_c_code{
  "code": "...",
  "enable_tsan": true
}
```

---

## 6. Export & Hugging Face Deployment

The fine-tuned model artifacts are published and available on Hugging Face:

| Artifact | Format | Precision / Quant | Hugging Face Repository |
| :--- | :--- | :--- | :--- |
| **LoRA Adapter** | PEFT / LoRA | 16-bit LoRA adapter weights | [`wxcdart/gemma4-e2b-unified-engine`](https://huggingface.co/wxcdart/gemma4-e2b-unified-engine) |
| **Merged Standalone** | Safetensors | Full merged 16-bit weights | [`wxcdart/gemma4-e2b-unified-engine-safetensors`](https://huggingface.co/wxcdart/gemma4-e2b-unified-engine-safetensors) |
| **GGUF Models** | GGUF | `q4_k_m`, `q8_0` | [`wxcdart/gemma4-e2b-unified-engine-gguf`](https://huggingface.co/wxcdart/gemma4-e2b-unified-engine-gguf) |

### Exporting and Uploading Scripts

1. **LoRA Adapter & Full Merged Safetensors**:
   ```bash
   python3 push_to_hf.py <HF_TOKEN>
   ```

2. **Quantized GGUF Models (`q4_k_m`, `q8_0`)**:
   ```bash
   python3 push_gguf_only.py <HF_TOKEN>
   ```

### Running with llama.cpp or Ollama

Once the GGUF weights are downloaded:

#### llama.cpp
```bash
./llama-cli \
    -m gemma4-e2b-unified-engine-q4_k_m.gguf \
    -p "<start_of_turn>user\nDiagnose race conditions in this POSIX thread code...<end_of_turn>\n<start_of_turn>model\n" \
    -n 512
```

#### Ollama Modelfile
```dockerfile
FROM ./gemma4-e2b-unified-engine-q4_k_m.gguf
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
```
Create and run the model:
```bash
ollama create gemma4-unified -f Modelfile
ollama run gemma4-unified
```
