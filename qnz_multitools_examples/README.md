# QNZ Multitools & Multi-Language Examples

This directory provides coding examples demonstrating the **QNZ Multitools ecosystem** across **C99/C23**, **JavaScript (zujs)**, **POSIX Dash (`/bin/dash`)**, and **Python**.

---

## Workspace Directory Structure

| File | Language | Target Feature / Multitool |
| :--- | :--- | :--- |
| `01_c_programming_zero_h.c` | C99 / C23 | Custom Arena Allocator, DataFrames, ZeroSQL Queries (`zero.h`) |
| `02_js_zujs_multitool.js` | JavaScript | High-performance Data Pipelines, DOM Parsing (`zujs` engine) |
| `03_dash_multitool.sh` | POSIX Dash | Automated header integrity checks, toolchain launcher |
| `04_python_qnz_interop.py` | Python 3 | QNZ Header inspection, sub-process pipeline driver |
| `05_zero_ml_dl_signal_suite.c` | C99 / C23 | `zero_ml` (Regress/Cluster), `zero_dl` (Autograd/Vision), `zero_signal` (FFT/Windows) |
| `06_zero_ml_dl_signal_demo.py` | Python 3 | Automated compilation and execution runner for C zero suites |
| `07_qnz_master_all_modules.c` | C99 / C23 | Master Suite covering ALL QNZ modules (`zero_arena`, `zero_df`, `zero_ml`, `zero_dl`, `zero_signal`, `zero_crdt`) |
| `08_qnz_ocr_image_pipeline.c` | C99 / C23 | Optical Character Recognition (OCR), Otsu Binarization, Connected-Component Table Grid Extraction (`zero_ocr`, `zero_image`) |
| `09_qnz_ytdownloader_media.py` | Python 3 | QNZ YouTube InnerTube Downloader (`zuytdownloader`), format deciphering & unthrottled streaming URL extraction |
| `10_qnz_model_exporter.js` | JavaScript | QNZ Engine Model Exporter & GGUF / Safetensors Inspector for Q4, Q8, and F16 quantization profiles (`zujs`) |
| `download_and_test_q4_gemma.js` | JavaScript | QNZ Engine Q4 GGUF Gemma model & LoRA adapter fetcher and forward pass test runner (`zujs`) |
| `test_wxcdart_gemma4_qnz.js` | JavaScript | QNZ Engine test script verifying downloaded `wxcdart/gemma4-e2b-unified-engine` Hugging Face model snapshot (`zujs`) |

---

## Execution Guide

### 1. POSIX Dash Script Execution
```bash
dash 03_dash_multitool.sh
```

### 2. JavaScript (zujs) Multitool Execution
```bash
node 02_js_zujs_multitool.js
```

### 3. C99/C23 QNZ Foundation Compilation & Execution
```bash
gcc -O2 -std=c99 01_c_programming_zero_h.c -I/home/coder/workspace/qnz -lpthread -lm -o qnz_c_demo
./qnz_c_demo
```

### 4. Python Interop Script Execution
```bash
python3 04_python_qnz_interop.py
```
