"""Canonical GSM8K prompt contract shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class GSM8KPromptContract:
    """Versioned rendering rules for the project's GSM8K baseline.

    The lack of whitespace between the format instruction and ``Question:``
    preserves the prompt emitted by :class:`data.gsm8k.builder.GSM8KBuilder`.
    Qwen2.5-1.5B is sensitive to seemingly harmless punctuation changes here,
    so every research path must use this object instead of rebuilding the text.
    """

    version: str = "gsm8k-memgen-builder-v1"
    format_instruction: str = (
        "Solve the math problem with proper reasoning, and make sure to put the "
        "FINAL ANSWER inside \\boxed{}."
    )
    question_template: str = "Question: {question}\n"
    max_new_tokens: int = 1024

    def user_content(self, question: str) -> str:
        normalized = str(question).strip()
        if not normalized:
            raise ValueError("GSM8K question must not be empty")
        return self.format_instruction + self.question_template.format(
            question=normalized
        )

    def messages(self, question: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.user_content(question)}]

    def render(self, tokenizer: Any, question: str) -> str:
        return str(tokenizer.apply_chat_template(
            self.messages(question),
            tokenize=False,
            add_generation_prompt=True,
        ))

    def token_ids(self, tokenizer: Any, question: str) -> list[int]:
        return [
            int(value)
            for value in tokenizer.encode(
                self.render(tokenizer, question), add_special_tokens=False
            )
        ]

    def metadata(self, *, chat_template: str) -> dict[str, Any]:
        contract = {
            "version": self.version,
            "format_instruction": self.format_instruction,
            "question_template": self.question_template,
            "max_new_tokens": self.max_new_tokens,
        }
        serialized = json.dumps(
            contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            **contract,
            "contract_sha256": hashlib.sha256(serialized).hexdigest(),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
        }


GSM8K_PROMPT_CONTRACT = GSM8KPromptContract()
