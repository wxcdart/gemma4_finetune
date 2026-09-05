"""
06_zero_ml_dl_signal_demo.py
----------------------------
Python wrapper and demonstration script for QNZ zero_ml, zero_dl, and zero_signal suites.
Compiles and invokes the C native binary `zero_suite_demo` and processes the execution results.

Execution:
    python3 06_zero_ml_dl_signal_demo.py
"""

import os
import subprocess
import sys

EXAMPLES_DIR = "/home/coder/workspace/qnz_multitools_examples"
C_SOURCE = os.path.join(EXAMPLES_DIR, "05_zero_ml_dl_signal_suite.c")
BINARY_PATH = os.path.join(EXAMPLES_DIR, "zero_suite_demo")
QNZ_DIR = "/home/coder/workspace/qnz"

def build_suite():
    print("[1] Compiling Native QNZ Suite (zero_ml, zero_dl, zero_signal)...")
    cmd = [
        "gcc", "-O2", "-std=c99", C_SOURCE,
        f"-I{QNZ_DIR}",
        f"-I{os.path.join(QNZ_DIR, 'include')}",
        f"-I{os.path.join(QNZ_DIR, 'src', 'bearssl')}",
        "-lpthread", "-lm",
        "-o", BINARY_PATH
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation Error:\n{res.stderr}")
        sys.exit(1)
    print("   Compilation successful!")

def run_suite():
    print("\n[2] Executing Native QNZ Tool Suites Binary...")
    res = subprocess.run([BINARY_PATH], capture_output=True, text=True)
    print("-------------------- STDOUT --------------------")
    print(res.stdout.strip())
    print("------------------------------------------------")

def main():
    print("==================================================")
    print("   QNZ zero_ml / zero_dl / zero_signal Python    ")
    print("==================================================\n")
    build_suite()
    run_suite()
    print("\n[OK] QNZ Python zero suite execution complete.")

if __name__ == "__main__":
    main()
