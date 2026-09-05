#!/bin/dash
# 03_dash_multitool.sh
# --------------------
# QNZ Multitool Shell Script (POSIX Dash Compliant /bin/dash)
# Automates QNZ toolchain discovery, zero_cc compiler header checks,
# system benchmarks, and build workflow execution.
#
# Execution:
#   dash 03_dash_multitool.sh

set -e

QNZ_DIR="/home/coder/workspace/qnz"

echo "=================================================="
echo "   QNZ Multitool - POSIX Dash Orchestrator       "
echo "=================================================="

# 1. Environment & Header Integrity Check
echo "[1] Checking QNZ Headers and Toolchain..."
if [ -f "$QNZ_DIR/zero.h" ]; then
    SIZE_ZERO_H=$(wc -c < "$QNZ_DIR/zero.h")
    echo "   FOUND zero.h ($SIZE_ZERO_H bytes)"
else
    echo "   ERROR: zero.h missing in $QNZ_DIR"
    exit 1
fi

if [ -f "$QNZ_DIR/zero_cc.h" ]; then
    SIZE_ZERO_CC=$(wc -c < "$QNZ_DIR/zero_cc.h")
    echo "   FOUND zero_cc.h ($SIZE_ZERO_CC bytes)"
else
    echo "   WARNING: zero_cc.h missing in $QNZ_DIR"
fi

# 2. Compiler Compatibility Test
echo "\n[2] Testing C Compiler Availability..."
if command -v gcc >/dev/null 2>&1; then
    GCC_VER=$(gcc --version | head -n 1)
    echo "   Compiler: $GCC_VER"
else
    echo "   WARNING: gcc not found in PATH"
fi

# 3. Execution of zujs Multitool Benchmark
echo "\n[3] Executing zujs Multitool Benchmark..."
ZUJS_BIN="/home/coder/workspace/qnz/bin/zujs"
if [ -f "$ZUJS_BIN" ]; then
    "$ZUJS_BIN" /home/coder/workspace/qnz_multitools_examples/02_js_zujs_multitool.js
else
    echo "   zujs binary not found in QNZ bin directory."
fi

echo "\n=================================================="
echo "   QNZ Dash Multitool Orchestration Complete     "
echo "=================================================="
