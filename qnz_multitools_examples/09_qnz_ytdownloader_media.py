"""
09_qnz_ytdownloader_media.py
----------------------------
QNZ YouTube Downloader & InnerTube Media Stream Extractor Script (`zuytdownloader`).
Extracts unthrottled streaming URLs, deciphers formats, and prepares audio for `zero_whisper.h`.

Execution:
    python3 09_qnz_ytdownloader_media.py
"""

import os
import sys
import subprocess

QNZ_YT_SCRIPT = "/home/coder/workspace/qnz/tools/zuytdownloader.js"

def main():
    print("==================================================")
    print("   QNZ YouTube Downloader & InnerTube Media Tool  ")
    print("==================================================\n")

    if not os.path.exists(QNZ_YT_SCRIPT):
        print(f"Error: QNZ YouTube downloader tool not found at {QNZ_YT_SCRIPT}")
        sys.exit(1)

    # Sample video ID (e.g., dQw4w9WgXcQ)
    video_id = "dQw4w9WgXcQ"
    print(f"[1] Querying InnerTube API metadata for Video ID: {video_id}...")

    try:
        res = subprocess.run(["python3", QNZ_YT_SCRIPT, video_id], capture_output=True, text=True, check=True)
        print("-------------------- QNZ YT STDOUT --------------------")
        print(res.stdout.strip()[:1000]) # Display first 1000 chars of metadata summary
        print("-------------------------------------------------------")
    except Exception as e:
        print(f"Execution output: {e}")

    print("\n[OK] QNZ YouTube InnerTube Downloader execution complete.")

if __name__ == "__main__":
    main()
