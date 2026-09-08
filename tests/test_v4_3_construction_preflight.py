from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from memgen.experience import v4_3_bank as bank
from scripts.diagnose_v4_3_construction import construction_diagnostics, require_qualified
from scripts import compile_v4_3_side_kv as compiler
from tests.test_v4_3_bank import fixture, rebind


class V43ConstructionPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = fixture()
        cls.qualified = bank.build_outputs(**data)
        for i, packet in enumerate(data["packets"]):
            for e in packet["evidence"]:
                e["semantic_signature"]["repair_operator"] = "The answer is \\boxed{37}."
            rebind(data, i)
        cls.quarantined = bank.build_outputs(**data)

    def test_zero_banks_reports_actual_clause_failures_without_raw_sources(self):
        report = construction_diagnostics(self.quarantined)
        self.assertEqual(report["qualified_tier_counts"], {"primary": 0, "conditional": 0})
        self.assertEqual(report["qualification_failure_counts"]["repair_operator:no_process_only_candidate"], 17)
        for r in report["banks"]:
            self.assertEqual(r["fields"]["repair_operator"]["support_count"], 0)
            self.assertIn("numeric_constant", r["fields"]["repair_operator"]["candidate_issue_counts"])
        self.assertNotIn("boxed", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Construction is not ready"):
            require_qualified(self.quarantined)

    def test_compiler_rejects_empty_bank_before_reading_reasoner_or_loading_model(self):
        args = ["compile", "--bank-dir", "/tmp/v43-bank", "--reasoner-manifest", "/tmp/v43-old/manifest.json",
                "--output-dir", "/tmp/v43-compiled"]
        with patch("sys.argv", args), patch.object(compiler, "load_construction", return_value=self.quarantined), patch.object(compiler, "read_json") as read:
            with self.assertRaisesRegex(ValueError, "Construction is not ready"):
                compiler.main()
            read.assert_not_called()

    def test_smoke_requires_both_tiers_but_compiler_can_compile_one_tier(self):
        require_qualified(self.qualified, both_tiers=True)
        partial = deepcopy(self.qualified)
        partial["conditional_bank_records.jsonl"] = []
        require_qualified(partial)
        with self.assertRaisesRegex(ValueError, "both tiers"):
            require_qualified(partial, both_tiers=True)


if __name__ == "__main__":
    unittest.main()
