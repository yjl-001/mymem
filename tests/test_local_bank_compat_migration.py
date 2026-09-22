"""The legacy resume lane must stay pinned to the exact production implementation."""
import unittest

from memgen.experience.bank_construction.sources import implementation_hashes
from scripts.resume_local_bank_cache_compat import BASELINE, TARGET, verify_implementation_migration


class CompatibilityMigrationTests(unittest.TestCase):
    def test_target_hashes_match_current_implementation(self):
        current = implementation_hashes()
        self.assertEqual({path: current.get(path) for path in TARGET}, TARGET)
        recorded = dict(current)
        for path, value in BASELINE.items():
            if value is None:
                recorded.pop(path, None)
            else:
                recorded[path] = value
        migration = verify_implementation_migration(recorded, current)
        self.assertEqual(migration["mode"], "pre-two-phase-production-compatibility")

    def test_unrelated_drift_is_rejected(self):
        current = implementation_hashes()
        recorded = dict(current)
        recorded["data/gsm8k/prompt.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_implementation_migration(recorded, current)


if __name__ == "__main__":
    unittest.main()
