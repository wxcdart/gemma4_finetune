"""
apply_chat_template_patch.py
Utility to inspect, validate, and patch the tokenizer's chat_template
to fix the upstream Gemma Jinja template bugs for tool calling and multi-turn turns.
"""

from transformers import AutoTokenizer

def patch_tokenizer_chat_template(tokenizer, template_path="gemma_chat_template.jinja"):
    with open(template_path, "r") as f:
        fixed_jinja_template = f.read()

    print("[*] Applying fixed Jinja chat template to tokenizer...")
    tokenizer.chat_template = fixed_jinja_template
    return tokenizer

def test_template(tokenizer):
    sample_messages = [
        {"role": "user", "content": "Check free disk space with busybox."},
        {
            "role": "model",
            "thought": "Need to check root disk usage using df -h.",
            "tool_calls": [{"name": "busybox_exec", "arguments": {"command": "df -h /"}}],
        },
        {
            "role": "tool",
            "name": "busybox_exec",
            "content": "Filesystem Size Used Avail Use% Mounted on\n/dev/root 20G 15G 4G 79% /"
        },
        {
            "role": "model",
            "thought": "Root usage is 79%, within safe operational limits.",
            "content": "The root filesystem is currently at 79% capacity with 4GB free."
        }
    ]

    rendered = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    print("\n--- RENDERED JINJA CHAT TEMPLATE TEST ---")
    print(rendered)
    print("------------------------------------------\n")
    return rendered

if __name__ == "__main__":
    # If unsloth/transformers is ready, test locally
    try:
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        patch_tokenizer_chat_template(tok)
        test_template(tok)
    except Exception as e:
        print(f"Test note: {e}")
