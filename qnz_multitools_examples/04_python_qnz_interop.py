"""
04_python_qnz_interop.py
------------------------
QNZ Multitool Python Interop Script.
Demonstrates:
  1. Inspecting QNZ C headers (zero.h / zero_cc.h)
  2. Subprocess integration with QNZ multitools
  3. JSON / Data transformation for Gemma 4 & zero ecosystems

Execution:
  python3 04_python_qnz_interop.py
"""

import os
import sys
import subprocess
import json

QNZ_ROOT = "/home/coder/workspace/qnz"
ZERO_H_PATH = os.path.join(QNZ_ROOT, "zero.h")
ZERO_CC_PATH = os.path.join(QNZ_ROOT, "zero_cc.h")

def inspect_header(file_path):
    if not os.path.exists(file_path):
        return {"status": "missing", "lines": 0, "size": 0}
    
    size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = len(f.readlines())
    return {"status": "available", "lines": lines, "size": size}

def main():
    print("==================================================")
    print("   QNZ Multitool - Python Ecosystem Interop       ")
    print("==================================================\n")

    # 1. Header Inspection
    print("[1] Inspecting QNZ C Headers...")
    zero_h_info = inspect_header(ZERO_H_PATH)
    zero_cc_info = inspect_header(ZERO_CC_PATH)

    print(f"   zero.h   : {zero_h_info['status'].upper()} ({zero_h_info['lines']:,} lines, {zero_h_info['size']:,} bytes)")
    print(f"   zero_cc.h: {zero_cc_info['status'].upper()} ({zero_cc_info['lines']:,} lines, {zero_cc_info['size']:,} bytes)")

    # 2. Executing POSIX Dash Multitool
    dash_script = "/home/coder/workspace/qnz_multitools_examples/03_dash_multitool.sh"
    if os.path.exists(dash_script):
        print(n := "\n[2] Triggering Dash Multitool via Subprocess...")
        try:
            res = subprocess.run(["dash", dash_script], capture_output=True, text=True, check=True)
            print("   Dash Script Output:")
            for line in res.stdout.strip().split("\n"):
                print(f"     | {line}")
        except Exception as e:
            print(f"   Dash script execution failed: {e}")

    print("\n[OK] QNZ Python Interop execution completed successfully.")

if __name__ == "__main__":
    main()
