from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_phase2_conditional_actions.py"
SPEC = importlib.util.spec_from_file_location("conditional_action_audit", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class ConditionalActionAuditTests(unittest.TestCase):
    def test_action_direction_diagnostics_distinguishes_collinear_and_orthogonal_actions(self) -> None:
        collinear = AUDIT.action_direction_diagnostics([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertEqual(collinear["nonzero_count"], 3)
        self.assertAlmostEqual(collinear["effective_rank"], 1.0)
        self.assertAlmostEqual(collinear["pairwise_cosine"]["mean"], 1.0)

        orthogonal = AUDIT.action_direction_diagnostics([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(orthogonal["effective_rank"], 2.0)
        self.assertAlmostEqual(orthogonal["pairwise_cosine"]["mean"], 0.0)

    def test_selection_concentration_reports_effective_action_count(self) -> None:
        concentration = AUDIT.selection_concentration([0, 0, 1], ["a", "b"])
        self.assertEqual(concentration["query_count"], 3)
        self.assertEqual(concentration["unique_selected_action_count"], 2)
        self.assertEqual(concentration["selected_counts"], {"a": 2, "b": 1})
        self.assertAlmostEqual(concentration["effective_selected_action_count"], 1.8)


if __name__ == "__main__":
    unittest.main()
