from __future__ import annotations

import unittest

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE


class GSM8KPromptContractTests(unittest.TestCase):
    def test_preserves_the_repository_baseline_prompt_exactly(self) -> None:
        self.assertEqual(
            GSM8K_PROMPT_CONTRACT.user_content("  How many?  "),
            "Solve the math problem with proper reasoning, and make sure to put "
            "the FINAL ANSWER inside \\boxed{}.Question: How many?\n",
        )

    def test_metadata_is_stable_and_records_the_generation_budget(self) -> None:
        first = GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        )
        second = GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        )
        self.assertEqual(first, second)
        self.assertEqual(first["max_new_tokens"], 1024)
        self.assertEqual(len(first["contract_sha256"]), 64)
        self.assertEqual(len(first["chat_template_sha256"]), 64)

    def test_rejects_an_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            GSM8K_PROMPT_CONTRACT.messages("  ")


if __name__ == "__main__":
    unittest.main()
