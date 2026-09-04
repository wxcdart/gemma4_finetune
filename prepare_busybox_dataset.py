"""
prepare_busybox_dataset.py
Curates and formats BusyBox / POSIX sh tool-use and scripting datasets for
fine-tuning Gemma 4 E2B (or continuing fine-tuning from existing adapter checkpoints).

Key busybox constraints emphasized:
- Ash / POSIX sh semantics (avoid bashisms like [[ ]], arrays, process substitution <(), <<<)
- BusyBox applet constraints (cut down versions of grep, sed, awk, find, tar, wget, netcat)
- Lightweight automation, Alpine/embedded Linux, init scripts, container diagnostics
"""

from datasets import Dataset

BUSYBOX_EXAMPLES = [
    {
        "instruction": "Write a POSIX-compliant shell script for BusyBox ash to monitor free disk space on root '/' and send an alert via busybox wget if usage exceeds 90%. Avoid non-POSIX bashisms.",
        "input": "",
        "output": """```sh
#!/bin/sh
set -eu

THRESHOLD=90
ALERT_WEBHOOK="http://192.168.1.100:8080/alert"

# In busybox df, usage percentage is typically in column 5
USAGE=$(df / | awk 'NR==2 {gsub("%", "", $5); print $5}')

if [ "$USAGE" -ge "$THRESHOLD" ]; then
    MSG="WARNING: Root filesystem disk usage is at ${USAGE}% (threshold: ${THRESHOLD}%)"
    echo "$MSG" >&2
    
    # Use busybox wget with POST payload
    if command -v wget >/dev/null 2>&1; then
        wget -q -O - --post-data="{\"alert\":\"$MSG\"}" --header="Content-Type: application/json" "$ALERT_WEBHOOK" || true
    fi
fi
```"""
    },
    {
        "instruction": "In a minimal BusyBox container without curl or bash, how do you perform a TCP port check and fetch an HTTP response using only busybox builtins and applets?",
        "input": "",
        "output": """In a minimal BusyBox environment, `curl` and bash's `/dev/tcp` virtual paths are unavailable. You can use BusyBox's `nc` (netcat) or `wget` applets:

### 1. TCP Port Reachability Check with `nc`
```sh
# -z for zero-I/O mode (scanning), -w for timeout in seconds
if nc -z -w 3 10.0.0.1 5432; then
    echo "Port 5432 is open"
else
    echo "Port 5432 is closed or unreachable"
fi
```

### 2. Raw HTTP GET Request via `nc`
If `wget` is stripped or missing TLS support:
```sh
printf "GET /health HTTP/1.1\\r\\nHost: api.internal\\r\\nConnection: close\\r\\n\\r\\n" | nc -w 5 api.internal 80
```

### 3. Fetching Content using BusyBox `wget`
```sh
# -q: quiet mode, -O -: dump to stdout
busybox wget -q -O - "http://api.internal/health"
```"""
    },
    {
        "instruction": "Convert the following bash script to a strictly POSIX sh script compatible with BusyBox ash:\n```bash\nfiles=(/var/log/*.log)\nif [[ ${#files[@]} -gt 0 ]]; then\n    for f in \"${files[@]}\"; do\n        echo \"Processing $f\"\n    done\nfi\n```",
        "input": "",
        "output": """BusyBox `ash` does not support indexed arrays (`${files[@]}`) or double brackets (`[[ ]]`). Here is the POSIX / BusyBox compliant translation using positional parameters:

```sh
#!/bin/sh
set -e

# Set positional parameters to matching files
set -- /var/log/*.log

# Check if glob expansion actually matched a file or stayed literal
if [ -e "$1" ]; then
    for f in "$@"; do
        echo "Processing $f"
    done
else
    echo "No log files found in /var/log/"
fi
```"""
    },
    {
        "instruction": "Write a BusyBox inittab and runscript service pattern for an embedded microservice daemon with auto-restart.",
        "input": "",
        "output": """BusyBox `init` uses a simplified `/etc/inittab` format: `<id>::<action>:<process>`.

### 1. `/etc/inittab` Configuration
```text
::sysinit:/etc/init.d/rcS
::respawn:/usr/bin/my_service_daemon --config /etc/daemon.conf
::restart:/sbin/init
::ctrlaltdel:/sbin/reboot
::shutdown:/bin/umount -a -r
```

### 2. Startup script `/etc/init.d/rcS`
```sh
#!/bin/sh
set -e

# Mount virtual filesystems essential for busybox tools
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

# Start system logging daemon (busybox syslogd)
syslogd -C256
klogd

echo "BusyBox init environment initialized."
```"""
    },
    {
        "instruction": "Write a BusyBox script to extract and aggregate IP addresses with high request counts from an Nginx access log using only busybox awk and sort.",
        "input": "",
        "output": """```sh
#!/bin/sh
# Aggregate IPs exceeding request threshold using BusyBox applets
LOG_FILE="${1:-/var/log/nginx/access.log}"
THRESHOLD="${2:-50}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE" >&2
    exit 1
fi

# BusyBox awk extracts column 1 ($1 = client IP) and tallies counts
awk '{ count[$1]++ } END {
    for (ip in count) {
        if (count[ip] >= threshold) {
            printf "%-6d %s\\n", count[ip], ip
        }
    }
}' threshold="$THRESHOLD" "$LOG_FILE" | sort -nr
```"""
    }
]

def format_prompt(example):
    instruction = example["instruction"]
    if example.get("input") and example["input"].strip():
        instruction += f"\n\nContext:\n{example['input']}"
    
    formatted_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
    return {"text": formatted_text}

def main():
    print(f"Loaded {len(BUSYBOX_EXAMPLES)} BusyBox seed samples.")
    dataset = Dataset.from_list(BUSYBOX_EXAMPLES)
    formatted_dataset = dataset.map(format_prompt)
    output_path = "/home/coder/workspace/gemma4_finetune/busybox_train_data.jsonl"
    formatted_dataset.to_json(output_path)
    print(f"Saved BusyBox formatted dataset to {output_path}")

if __name__ == "__main__":
    main()
