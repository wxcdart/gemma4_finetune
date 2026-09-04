"""
prepare_dataset.py
Dataset preparation and mixing pipeline for:
- C Programming (systems programming, memory management, POSIX, concurrency)
- Python Machine Learning (scikit-learn, XGBoost, pandas/numpy pipelines)
- Deep Learning (PyTorch training loops, custom modules, attention, optimizers)
- Natural Language Processing (tokenization, embeddings, seq2seq, prompt engineering)
- Signal Processing (FFT, filtering, audio analysis, scipy.signal, librosa)
- Distributed Programming (MPI, multiprocessing, Ray, PyTorch DDP/FSDP, Celery)
- Automation & Scripting (Bash, automation tools, workflow orchestration, APIs)
"""

import json
from datasets import Dataset

DOMAIN_EXAMPLES = [
    # 1. C Programming
    {
        "domain": "c_programming",
        "instruction": "Write a thread-safe circular ring buffer in C using POSIX mutexes and condition variables.",
        "input": "",
        "output": """```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>

typedef struct {
    int *buffer;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} RingBuffer;

RingBuffer* rb_create(size_t capacity) {
    RingBuffer *rb = malloc(sizeof(RingBuffer));
    if (!rb) return NULL;
    rb->buffer = malloc(sizeof(int) * capacity);
    rb->capacity = capacity;
    rb->head = 0;
    rb->tail = 0;
    rb->count = 0;
    pthread_mutex_init(&rb->lock, NULL);
    pthread_cond_init(&rb->not_empty, NULL);
    pthread_cond_init(&rb->not_full, NULL);
    return rb;
}

void rb_push(RingBuffer *rb, int item) {
    pthread_mutex_lock(&rb->lock);
    while (rb->count == rb->capacity) {
        pthread_cond_wait(&rb->not_full, &rb->lock);
    }
    rb->buffer[rb->tail] = item;
    rb->tail = (rb->tail + 1) % rb->capacity;
    rb->count++;
    pthread_cond_signal(&rb->not_empty);
    pthread_mutex_unlock(&rb->lock);
}

int rb_pop(RingBuffer *rb) {
    pthread_mutex_lock(&rb->lock);
    while (rb->count == 0) {
        pthread_cond_wait(&rb->not_empty, &rb->lock);
    }
    int item = rb->buffer[rb->head];
    rb->head = (rb->head + 1) % rb->capacity;
    rb->count--;
    pthread_cond_signal(&rb->not_full);
    pthread_mutex_unlock(&rb->lock);
    return item;
}

void rb_free(RingBuffer *rb) {
    if (!rb) return;
    pthread_mutex_destroy(&rb->lock);
    pthread_cond_destroy(&rb->not_empty);
    pthread_cond_destroy(&rb->not_full);
    free(rb->buffer);
    free(rb);
}
```"""
    },
    # 2. Python ML
    {
        "domain": "python_ml",
        "instruction": "Build an end-to-end scikit-learn Pipeline with median imputation, one-hot encoding, feature scaling, and XGBoost classification with cross-validation.",
        "input": "",
        "output": """```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

def build_ml_pipeline(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])
    
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        ))
    ])
    return model_pipeline
```"""
    },
    # 3. Deep Learning (PyTorch)
    {
        "domain": "deep_learning",
        "instruction": "Implement Multi-Head Self-Attention with causal masking from scratch in PyTorch.",
        "input": "",
        "output": """```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```"""
    },
    # 4. Natural Language Processing (NLP)
    {
        "domain": "nlp",
        "instruction": "Write a PyTorch function to calculate sentence embedding similarity using cosine similarity and batched forward passes.",
        "input": "",
        "output": """```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def compute_sentence_embeddings(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def pairwise_similarity(embeddings_a, embeddings_b):
    return torch.mm(embeddings_a, embeddings_b.transpose(0, 1))
```"""
    },
    # 5. Signal Processing (SP)
    {
        "domain": "signal_processing",
        "instruction": "Design a Butterworth bandpass filter in Python using scipy.signal and apply it to a noisy multi-frequency signal.",
        "input": "",
        "output": """```python
import numpy as np
import scipy.signal as signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data
```"""
    },
    # 6. Distributed Programming
    {
        "domain": "distributed_programming",
        "instruction": "Show how to set up a PyTorch DistributedDataParallel (DDP) training boilerplate with torch.distributed and DistributedSampler.",
        "input": "",
        "output": """```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()
```"""
    },
    # 7. Automation
    {
        "domain": "automation",
        "instruction": "Write an asynchronous Python script using asyncio and httpx to monitor endpoints, alert on failures, and log latency metrics.",
        "input": "",
        "output": """```python
import asyncio
import logging
import time
from typing import List, Dict
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def check_endpoint(client: httpx.AsyncClient, url: str) -> Dict[str, any]:
    start = time.perf_counter()
    try:
        response = await client.get(url, timeout=5.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"url": url, "status": response.status_code, "latency_ms": elapsed_ms, "healthy": response.status_code == 200}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"url": url, "status": None, "latency_ms": elapsed_ms, "healthy": False, "error": str(exc)}
```"""
    }
]

def format_prompt(example):
    """
    Format into standard Gemma chat template:
    <start_of_turn>user
    {instruction}<end_of_turn>
    <start_of_turn>model
    {output}<end_of_turn>
    """
    instruction = example["instruction"]
    if example.get("input") and example["input"].strip():
        instruction += f"\n\nContext:\n{example['input']}"
    
    formatted_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
    return {"text": formatted_text}

def main():
    print(f"Loaded {len(DOMAIN_EXAMPLES)} domain seed samples.")
    dataset = Dataset.from_list(DOMAIN_EXAMPLES)
    formatted_dataset = dataset.map(format_prompt)
    output_path = "/home/coder/workspace/gemma4_finetune/train_data.jsonl"
    formatted_dataset.to_json(output_path)
    print(f"Saved formatted dataset to {output_path}")

if __name__ == "__main__":
    main()
