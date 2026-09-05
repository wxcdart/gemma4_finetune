#!/usr/bin/env python3
"""
generate_dataset.py
Generates domain-specific instruction-tuning dataset for zero_rt:
  Specializations:
    1. ISO C99 Mechanical-Sympathy Systems Programming
    2. zero_udt data structures (ZeroDict TPHT, ZeroArray, ZeroArena, ZeroWal, ZeroChunkedPool)
    3. Tiny Pointers (zero_tinyptr.h / zero_tinyptr_var.h) & Succinct Trees (zero_succinct_tree.h)
    4. zero_dl & zero_infer neural networks and .ztensors serialization
    5. High-performance Linux Bash pipelines, epoll, and eBPF kernel hooks
    6. Python zero_udt C-extension interop bindings
"""

import json
import pathlib

DATASET_PATH = pathlib.Path("/home/coder/workspace/finetune_zero_rt/data/zero_rt_instructions.jsonl")

EXAMPLES = [
    {
        "instruction": "How do you allocate and reset an activation scratchpad in zero_udt without heap allocation overhead?",
        "response": """To achieve 1-cycle activation scratchpad allocation in zero_udt, use `ZeroArena` (defined in `portable/core/zero_arena.h`):

```c
#include "portable/core/zero_arena.h"

// 1. Initialize arena buffer with 64KB backing memory
ZeroArena arena;
uint8_t buffer[65536];
zero_arena_init(&arena, buffer, sizeof(buffer));

// 2. Bump-allocate temporary activations during forward inference
float* layer1_out = (float*)zero_arena_alloc(&arena, 16 * sizeof(float));
float* layer2_out = (float*)zero_arena_alloc(&arena, 8 * sizeof(float));

// 3. Reset all scratchpad memory in 1 CPU clock cycle (offset = 0)
zero_arena_reset(&arena);
```
This guarantees zero heap `malloc`/`free` calls on the critical inference path."""
    },
    {
        "instruction": "Write a pure ISO C99 function using zero_ztensor.h to look up a neural network weight tensor in under 30 nanoseconds.",
        "response": """In `zero_ztensor.h`, tensor metadata is indexed using Two-Phase Quotient Hashing (TPHT) to achieve ~21.7 ns lookups:

```c
#include "portable/containers/zero_ztensor.h"

const float* get_layer_weights(const ZeroZTensorFile* model, const char* layer_name, size_t* out_len) {
    if (!model || !layer_name) return NULL;

    // TPHT O(1) single-lookup resolution
    const ZeroZTensorDesc* desc = zero_ztensor_get_desc(model, layer_name);
    if (!desc) return NULL;

    if (out_len) *out_len = desc->data_len / sizeof(float);
    return (const float*)zero_ztensor_get_data(model, desc);
}
```"""
    },
    {
        "instruction": "Explain how 2-tier variable-length Tiny Pointers in zero_tinyptr_var.h eliminate the log log log n theoretical lower bound.",
        "response": """Standard fixed-size tiny pointers require $\\Theta(\\log \\log \\log n + \\log \\delta^{-1})$ bits for an array at load factor $1-\\delta$. 

`zero_tinyptr_var.h` implements a **2-Tier Variable-Length Tiny Pointer Table**:
1. **Tier-1 Compact (8-bit)**: Used for 95%+ of elements where offset delta fits in $[-127, +127]$.
2. **Tier-2 Extended (16-bit / 32-bit)**: Used as an escape mechanism for rare collision clusters.

Because the amortized expected bits per entry is constant $O(1)$ regardless of total elements $n$, the $\\log \\log \\log n$ lower bound is eliminated while maintaining 100% stable index dereferencing."""
    },
    {
        "instruction": "Write a bash one-liner using zero_toolbox tools to find all C99 struct definitions across zero_rt submodules and count them.",
        "response": """Using `zero_toolbox` native tools (`zugrep` and `zuwc`):

```bash
./bin/zugrep -r "typedef struct" /home/coder/workspace/zero_rt/ | ./bin/zuwc -l
```

Or using standard Linux pipeline with zero_udt filter:
```bash
find /home/coder/workspace/zero_rt -name "*.h" -exec ./bin/zugrep "typedef struct" {} + | wc -l
```"""
    },
    {
        "instruction": "How do you construct a frozen backbone transfer learning model in zero_dl?",
        "response": """Use `zero_dl_transfer.h`:

```c
#include "zero_dl_transfer.h"

ZeroDLTransferModel model;
// Loads backbone from .ztensors with is_trainable = 0, attaches Xavier task head with is_trainable = 1
int ok = zero_dl_transfer_init_from_ztensors(
    &model,
    "models/backbone.ztensors",
    "backbone.weight",
    "backbone.bias",
    64, /* Input feature dimension */
    16, /* Backbone latent dimension */
    1   /* Task head output classes */
);

// Perform fine-tuning step (autograd skips frozen backbone weights)
float loss = zero_dl_transfer_train_step(&model, x_sample, y_target, 0.01f);
```"""
    }
]

def main():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        for ex in EXAMPLES:
            entry = {
                "prompt": f"<start_of_turn>user\n{ex['instruction']}<end_of_turn>\n<start_of_turn>model\n<thought>\nAnalyzing zero_rt systems mechanics, zero_udt primitives, and memory layout.\n</thought>\n{ex['response']}<end_of_turn>"
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[✓] Generated {len(EXAMPLES)} curated domain instructions at {DATASET_PATH}")

if __name__ == "__main__":
    main()
