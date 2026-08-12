"""Lightweight chat-template definitions shared by training and offline tools."""

# From HuggingFaceTB/SmolLM3-3B.  MemGen deliberately forces this template for
# both training/evaluation and Phase 1 student rollout collection.
CONVERSATION_TEMPLATE = r"""
{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if (message.role == "user") or (message.role == "system") -%}
        {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {%- generation -%}
        {{ "<|im_start|>assistant\n" + content + "<|im_end|>\n" }}
        {%- endgeneration -%}
    {%- elif message.role == "tool" -%}
    {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
""".strip()
