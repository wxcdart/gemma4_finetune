"""
prepare_tool_use_dataset.py
Formats training data with structured tool calling conventions for Gemma 4 / BusyBox.
Supports both:
1. Standard HuggingFace / Gemma chat template with `tools` schema.
2. Structured XML / Markdown execution blocks:
   <start_of_turn>model
   <thought>
   ...
   </thought>
   <call:busybox_sh>
   {"command": "df -h /"}
   </call:busybox_sh><end_of_turn>
   <start_of_turn>tool
   <response:busybox_sh>
   Filesystem Size Used Avail Use% Mounted on
   /dev/root  20G  14G  4.8G  75% /
   </response:busybox_sh><end_of_turn>
   <start_of_turn>model
   ...
"""

import json
from datasets import Dataset

# Define tool declarations (OpenAI / HuggingFace format)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "busybox_exec",
            "description": "Executes a POSIX shell command in a BusyBox container/environment. Only standard busybox applets and /bin/sh syntax are permitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute in BusyBox ash."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Execution timeout limit in seconds.",
                        "default": 10
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# Multi-turn tool calling training conversations
TOOL_CALLING_CONVERSATIONS = [
    {
        "id": "busybox_disk_check",
        "messages": [
            {
                "role": "user",
                "content": "Check available disk space on the root filesystem in the BusyBox environment and report if it is above 80%."
            },
            {
                "role": "model",
                "thought": "I need to inspect the disk space of the root filesystem using the BusyBox 'df' applet.",
                "tool_calls": [
                    {
                        "name": "busybox_exec",
                        "arguments": {"command": "df -h /"}
                    }
                ]
            },
            {
                "role": "tool",
                "name": "busybox_exec",
                "content": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        30G   26G  2.2G  92% /"
            },
            {
                "role": "model",
                "thought": "The root filesystem is currently at 92% capacity, which is above the 80% threshold. I will formulate a warning message.",
                "content": "The root filesystem (`/`) is currently at **92% capacity** (26GB used out of 30GB, 2.2GB available). This exceeds the 80% threshold. You may want to clean up package caches, temporary files in `/tmp`, or old logs."
            }
        ]
    },
    {
        "id": "busybox_tcp_port_probe",
        "messages": [
            {
                "role": "user",
                "content": "Test whether the redis server on 10.0.0.15:6379 is reachable using busybox."
            },
            {
                "role": "model",
                "thought": "Curl and bash /dev/tcp are not available in BusyBox. I should use the BusyBox 'nc' (netcat) applet in zero-I/O scanning mode (-z) with a 3-second timeout.",
                "tool_calls": [
                    {
                        "name": "busybox_exec",
                        "arguments": {"command": "nc -z -w 3 10.0.0.15 6379 && echo 'PORT_OPEN' || echo 'PORT_CLOSED'"}
                    }
                ]
            },
            {
                "role": "tool",
                "name": "busybox_exec",
                "content": "PORT_OPEN"
            },
            {
                "role": "model",
                "thought": "The command returned PORT_OPEN, indicating the Redis port is accepting connections.",
                "content": "The Redis server at `10.0.0.15:6379` is **reachable and accepting connections**."
            }
        ]
    },
    {
        "id": "busybox_memory_inspection",
        "messages": [
            {
                "role": "user",
                "content": "Find the top 3 processes consuming the most memory using busybox ps and sort."
            },
            {
                "role": "model",
                "thought": "BusyBox ps has limited flags compared to GNU procps. Typically 'busybox ps -o pid,user,vsz,comm' or 'busybox ps aux' works. Let's run 'ps -o pid,vsz,comm' and sort by VSZ column.",
                "tool_calls": [
                    {
                        "name": "busybox_exec",
                        "arguments": {"command": "ps -o pid,vsz,comm | sort -k2 -nr | head -n 3"}
                    }
                ]
            },
            {
                "role": "tool",
                "name": "busybox_exec",
                "content": " 1420 819200 python3\n 1102 450120 node\n  890 120400 redis-server"
            },
            {
                "role": "model",
                "thought": "I have the top 3 memory-consuming processes. I will format the response cleanly.",
                "content": "The top 3 processes by virtual memory usage (`VSZ`) in the BusyBox environment are:\n\n1. **PID 1420 (`python3`)**: ~800 MB\n2. **PID 1102 (`node`)**: ~440 MB\n3. **PID 890 (`redis-server`)**: ~118 MB"
            }
        ]
    }
]

def format_turn_based_tool_calling(conv):
    """
    Formats multi-turn tool calling into the standard Gemma turn tokens:
    <start_of_turn>user ... <end_of_turn>
    <start_of_turn>model
    <thought>...</thought>
    <call:tool_name>
    {arguments}
    </call:tool_name><end_of_turn>
    <start_of_turn>tool
    <response:tool_name>
    {tool_output}
    </response:tool_name><end_of_turn>
    <start_of_turn>model
    ...<end_of_turn>
    """
    text_chunks = []
    
    for msg in conv["messages"]:
        role = msg["role"]
        
        if role == "user":
            text_chunks.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
            
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            content = msg["content"]
            turn_str = f"<start_of_turn>tool\n<response:{tool_name}>\n{content}\n</response:{tool_name}><end_of_turn>"
            text_chunks.append(turn_str)
            
        elif role == "model":
            parts = []
            if "thought" in msg and msg["thought"]:
                parts.append(f"<thought>\n{msg['thought']}\n</thought>")
                
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    args_json = json.dumps(tc["arguments"], ensure_ascii=False)
                    parts.append(f"<call:{tc['name']}>\n{args_json}\n</call:{tc['name']}>")
                    
            if "content" in msg and msg["content"]:
                parts.append(msg["content"])
                
            model_turn = f"<start_of_turn>model\n" + "\n".join(parts) + "\n<end_of_turn>"
            text_chunks.append(model_turn)

    full_text = "\n".join(text_chunks)
    return {"text": full_text}

def main():
    print(f"Loaded {len(TOOL_CALLING_CONVERSATIONS)} tool-calling conversations.")
    dataset = Dataset.from_list(TOOL_CALLING_CONVERSATIONS)
    formatted = dataset.map(format_turn_based_tool_calling)
    
    out_path = "/home/coder/workspace/gemma4_finetune/tool_calling_train_data.jsonl"
    formatted.to_json(out_path)
    print(f"Saved formatted tool-calling dataset to {out_path}")

    # Also save the schema reference
    schema_path = "/home/coder/workspace/gemma4_finetune/tools_schema.json"
    with open(schema_path, "w") as f:
        json.dump(TOOLS_SCHEMA, f, indent=2)
    print(f"Saved tools schema to {schema_path}")

if __name__ == "__main__":
    main()
