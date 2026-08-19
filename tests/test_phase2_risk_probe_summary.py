from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "scripts" / "summarize_phase2_entropy_risk_probe.py"
SPEC = importlib.util.spec_from_file_location("phase2_risk_probe_summary", SUMMARY_PATH)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class Phase2RiskProbeSummaryTests(unittest.TestCase):
    def test_paired_bootstrap_uses_only_shared_sample_ids(self) -> None:
        real = {
            "a": {"entropy_delta_to_next_candidate": -0.4},
            "b": {"entropy_delta_to_next_candidate": -0.3},
            "real-only": {"entropy_delta_to_next_candidate": -0.2},
        }
        control = {
            "a": {"entropy_delta_to_next_candidate": -0.1},
            "b": {"entropy_delta_to_next_candidate": -0.1},
            "control-only": {"entropy_delta_to_next_candidate": -0.1},
        }
        report = SUMMARY.paired_bootstrap(real, control, seed=42, resamples=100)
        self.assertEqual(report["paired_sample_count"], 2)
        self.assertEqual(report["real_only_count"], 1)
        self.assertEqual(report["control_only_count"], 1)
        self.assertLess(report["mean_real_minus_control"], 0.0)
        self.assertLess(report["bootstrap_95_ci"][1], 0.0)

    def test_triggered_selection_excludes_unselected_candidates(self) -> None:
        records = [{
            "sample_id": "a",
            "intervention_trace": [
                {"entropy_triggered": False, "entropy_delta_to_next_candidate": 0.2},
                {"entropy_triggered": True, "entropy_delta_to_next_candidate": -0.2},
            ],
        }]
        selected = SUMMARY.selected_entropy_events(records, mode="triggered")
        self.assertEqual(selected["a"]["entropy_delta_to_next_candidate"], -0.2)


if __name__ == "__main__":
    unittest.main()
