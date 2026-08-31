from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v4_runtime = _load_module(
    "memgen_v4_runtime_test_module", "memgen/model/v4_runtime.py"
)
V4EpisodeConfig = v4_runtime.V4EpisodeConfig
V4MemoryEpisodeController = v4_runtime.V4MemoryEpisodeController

v4_selector = (
    None
    if torch is None
    else _load_module(
        "memgen_v4_selector_test_module", "memgen/model/v4_selector.py"
    )
)


@unittest.skipIf(torch is None, "Torch is required for V4 selector tests")
class V4SelectorTests(unittest.TestCase):
    def _bank(self, bank_id, positive, negative):
        positive_tensor = torch.nn.functional.normalize(
            torch.tensor(positive, dtype=torch.float32), dim=-1
        )
        negative_tensor = torch.nn.functional.normalize(
            torch.tensor(negative, dtype=torch.float32), dim=-1
        )
        return v4_selector.V4AnchorBank(
            bank_id=bank_id,
            positive_keys=positive_tensor,
            negative_keys=negative_tensor,
            positive_anchor_ids=tuple(
                f"{bank_id}-positive-{index}"
                for index in range(len(positive))
            ),
            negative_anchor_ids=tuple(
                f"{bank_id}-negative-{index}"
                for index in range(len(negative))
            ),
        )

    def test_selector_uses_size_normalized_positive_minus_negative_evidence(self):
        first = self._bank(
            "bank-a",
            positive=[[1.0, 0.0], [1.0, 0.0]],
            negative=[[0.0, 1.0]],
        )
        second = self._bank(
            "bank-b",
            positive=[[0.0, 1.0]],
            negative=[[1.0, 0.0], [1.0, 0.0]],
        )
        selector = v4_selector.V4RepairSelector(
            banks=(second, first),
            config=v4_selector.V4SelectorConfig(
                absolute_threshold=0.5,
                margin_threshold=0.1,
            ),
        )
        decision = selector.select(torch.tensor([1.0, 0.0]))
        self.assertEqual(decision.outcome, "selected")
        self.assertEqual(decision.selected_bank_id, "bank-a")
        self.assertAlmostEqual(decision.top1_score, 1.0, places=6)
        self.assertAlmostEqual(decision.top2_score, -1.0, places=6)
        self.assertEqual(
            decision.ranked_scores[0].positive_anchor_count,
            2,
        )

    def test_selector_abstains_on_absolute_and_margin_thresholds(self):
        bank_a = self._bank(
            "bank-a", positive=[[1.0, 0.0]], negative=[[0.0, 1.0]]
        )
        bank_b = self._bank(
            "bank-b", positive=[[0.9, 0.1]], negative=[[0.0, 1.0]]
        )
        absolute = v4_selector.V4RepairSelector(
            banks=(bank_a, bank_b),
            config=v4_selector.V4SelectorConfig(
                absolute_threshold=2.0,
                margin_threshold=0.0,
            ),
        ).select(torch.tensor([1.0, 0.0]))
        self.assertEqual(absolute.outcome, "abstained_absolute_threshold")
        self.assertIsNone(absolute.selected_bank_id)

        margin = v4_selector.V4RepairSelector(
            banks=(bank_a, bank_b),
            config=v4_selector.V4SelectorConfig(
                absolute_threshold=-2.0,
                margin_threshold=0.2,
            ),
        ).select(torch.tensor([1.0, 0.0]))
        self.assertEqual(margin.outcome, "abstained_margin_threshold")
        self.assertIsNone(margin.selected_bank_id)

    def test_local_query_uses_only_last_valid_reasoning_window(self):
        states = torch.tensor(
            [[100.0, 100.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
        )
        mask = torch.tensor([True, True, True, False])
        query = v4_selector.pool_v4_local_reasoning_query(
            states,
            reasoning_start_index=1,
            valid_token_mask=mask,
        )
        expected = torch.nn.functional.normalize(torch.tensor([0.5, 0.5]), dim=0)
        self.assertTrue(torch.allclose(query, expected))

    def test_anchor_bank_requires_normalized_disjoint_keys(self):
        with self.assertRaisesRegex(ValueError, "L2 normalized"):
            v4_selector.V4AnchorBank(
                bank_id="bank-a",
                positive_keys=torch.tensor([[2.0, 0.0]]),
                negative_keys=torch.tensor([[0.0, 1.0]]),
                positive_anchor_ids=("positive",),
                negative_anchor_ids=("negative",),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            v4_selector.V4AnchorBank(
                bank_id="bank-a",
                positive_keys=torch.tensor([[1.0, 0.0]]),
                negative_keys=torch.tensor([[0.0, 1.0]]),
                positive_anchor_ids=("same",),
                negative_anchor_ids=("same",),
            )


class V4EpisodeRuntimeTests(unittest.TestCase):
    def test_selected_memory_recovers_then_later_episode_can_reselect(self) -> None:
        controller = V4MemoryEpisodeController()
        selected = controller.apply_selection("bank-a")
        self.assertEqual(selected.state_before, "ARMED")
        self.assertEqual(selected.state_after, "ACTIVE")
        self.assertEqual(selected.activation_bank_id, "bank-a")
        self.assertTrue(controller.memory_should_be_active)

        self.assertIsNone(controller.observe_decoded_token(low_entropy=False))
        self.assertIsNone(controller.observe_decoded_token(low_entropy=True))
        recovered = controller.observe_decoded_token(low_entropy=True)
        self.assertEqual(recovered.reason, "recovery_low_entropy_hysteresis")
        self.assertTrue(recovered.deactivate_memory)
        self.assertEqual(controller.state, "ARMED")
        self.assertFalse(controller.memory_should_be_active)

        repeated = controller.apply_selection("bank-a")
        self.assertEqual(repeated.attempt_count, 2)
        self.assertEqual(controller.active_bank_id, "bank-a")

    def test_memory_window_is_bounded_and_requires_cooldown_before_rearm(self) -> None:
        controller = V4MemoryEpisodeController()
        controller.apply_selection("bank-a")
        transition = None
        for _ in range(32):
            transition = controller.observe_decoded_token(low_entropy=False)
        self.assertIsNotNone(transition)
        self.assertEqual(transition.reason, "maximum_active_window")
        self.assertEqual(controller.state, "COOLDOWN")
        self.assertTrue(transition.deactivate_memory)
        self.assertEqual(transition.active_step_count_before, 31)
        self.assertEqual(transition.active_step_count_after, 0)
        with self.assertRaisesRegex(RuntimeError, "only while ARMED"):
            controller.apply_selection("bank-b")
        self.assertIsNone(controller.observe_decoded_token(low_entropy=True))
        rearmed = controller.observe_decoded_token(low_entropy=True)
        self.assertEqual(rearmed.event, "gate_rearmed")
        self.assertEqual(controller.state, "ARMED")

    def test_abstain_is_nonterminal_until_attempt_budget_is_exhausted(self) -> None:
        controller = V4MemoryEpisodeController()
        for attempt in range(1, 4):
            transition = controller.apply_selection(None)
            self.assertEqual(transition.attempt_count, attempt)
            if attempt < 3:
                self.assertEqual(controller.state, "COOLDOWN")
                controller.observe_decoded_token(low_entropy=True)
                controller.observe_decoded_token(low_entropy=True)
                self.assertEqual(controller.state, "ARMED")
            else:
                self.assertEqual(controller.state, "EXHAUSTED")
        self.assertFalse(controller.gate_enabled)

    def test_answer_marker_closes_and_deactivates_active_memory(self) -> None:
        controller = V4MemoryEpisodeController()
        controller.apply_selection("bank-a")
        closed = controller.observe_decoded_token(
            low_entropy=False,
            answer_marker_seen=True,
        )
        self.assertEqual(closed.state_after, "CLOSED")
        self.assertTrue(closed.deactivate_memory)
        self.assertIsNone(controller.active_bank_id)

    def test_lifecycle_contract_is_frozen(self) -> None:
        config = V4EpisodeConfig()
        self.assertEqual(config.max_selector_attempts, 3)
        self.assertEqual(config.recovery_low_token_count, 2)
        self.assertEqual(config.max_active_steps, 32)
        with self.assertRaisesRegex(ValueError, "thirty-two"):
            V4EpisodeConfig(max_active_steps=1)


if __name__ == "__main__":
    unittest.main()
