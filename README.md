# Fine-Tuning Google Gemma 4 E2B with Unsloth

This workspace provides an end-to-end pipeline to fine-tune Google's **Gemma 4 E2B** model (2.3B effective / 5.1B total parameters with Per-Layer Embeddings and 128k context) across multiple specialized engineering domains:

1. **C Programming**: Systems programming, memory management, POSIX concurrency, and kernel/socket APIs.
2. **Python Machine Learning**: scikit-learn, XGBoost, LightGBM, feature engineering pipelines.
3. **Deep Learning**: Custom PyTorch architectures, attention mechanisms, optimizers, and loss formulations.
4. **Natural Language Processing (NLP)**: Tokenization, embeddings, text classification, and seq2seq tasks.
5. **Signal Processing (SP)**: FFT, Butterworth/Chebyshev digital filtering, spectrogram analysis via `scipy.signal` and `librosa`.
6. **Distributed Programming**: PyTorch DDP/FSDP, MPI, Ray, multiprocessing, and message queues.
7. **Automation & Scripting**: Async networking (`asyncio`, `httpx`), cron automation, workflow scripts, and tooling.

---

## 1. Setup & Environment
The project runs on an **NVIDIA L4 (24GB VRAM)** GPU using **Unsloth** for 2-5x faster training and 70% reduced memory footprint with QLoRA/LoRA.

```bash
pip install unsloth "transformers>=4.48.0" "datasets" "trl" "peft" "accelerate" "bitsandbytes"
```

## 2. Prepare Training Data
Run the dataset builder to generate domain-curated samples in the official Gemma chat turn format (`<start_of_turn>user ... <end_of_turn><start_of_turn>model ... <end_of_turn>`):

```bash
python3 prepare_dataset.py
```
This generates `train_data.jsonl`. You can augment this file with your own open-source code datasets (e.g., CodeAlpaca, Magicoder, OpenHermes, or your private codebases).

## 3. Run Fine-Tuning
Execute the training script:

```bash
python3 finetune.py
```

Key features of `finetune.py`:
- Loads `google/gemma-4-E2B-it` (or `unsloth` 4-bit quantized variant).
- Fast LoRA rank (`r=16`, `lora_alpha=16`) on all linear projection layers (`q, k, v, o, gate, up, down`).
- 8-bit AdamW optimizer with bfloat16 mixed precision.
- Saves the trained adapters to `./gemma4_e2b_specialized`.

## 4. Inference & Evaluation
Test the adapted model across test scenarios spanning all target domains:

```bash
python3 inference.py
```
