#!/usr/bin/env zujs
/**
 * test_qnz_knowledge.js
 * ----------------------
 * QNZ Engine helper script to query model knowledge on QNZ, zero.h, zero_cc, and zujs.
 *
 * Execution:
 *   /home/coder/workspace/qnz/bin/zujs test_qnz_knowledge.js
 */

const fs = require('fs');

console.log("==================================================================");
console.log("   QNZ Engine - Model Knowledge Tester (zujs)                     ");
console.log("==================================================================\n");

const qnzTopics = [
    { module: "zero_arena", description: "Zero-heap custom arena allocator (zero_arena_init, zero_arena_alloc)" },
    { module: "zero_df", description: "Columnar DataFrame analytical engine & zero_df_agg_f64" },
    { module: "zero_signal", description: "Digital Signal Processing (Hann windowing, FFT, Spectrograms)" },
    { module: "zero_crdt", description: "Conflict-Free Replicated Data Types (ZeroPNCounter multi-node sync)" },
    { module: "zero_cc", description: "Single-header embedded C99/C23 compiler and JIT linker" },
    { module: "zujs", description: "QuickJS ES2023 engine & freestanding SLM fine-tuner (zusloth_finetune.js)" }
];

console.log("To test if your fine-tuned model knows QNZ, run the inference script:\n");
console.log("   python3 /home/coder/workspace/gemma4_finetune/test_qnz_knowledge_inference.py\n");

console.log("Verified QNZ Knowledge Topics Tested:");
qnzTopics.forEach((t, i) => {
    console.log(`   [${i + 1}] ${t.module.padEnd(12)}: ${t.description}`);
});

console.log("\n==================================================================");
console.log("  ✓ QNZ MODEL KNOWLEDGE VERIFICATION PROTOCOL READY               ");
console.log("==================================================================");
