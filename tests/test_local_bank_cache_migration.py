"""Exact implementation-drift checks for an active pre-cache-compat run."""
import unittest

from scripts.resume_local_bank_cache_compat import BASELINE, TARGET, verify_implementation_migration


class CacheMigrationTests(unittest.TestCase):
    def test_accepts_only_the_recorded_cache_api_migration(self):
        recorded = {"unchanged.py": "same", **{key: value for key, value in BASELINE.items()
                                                 if value is not None}}
        current = {"unchanged.py": "same", **TARGET}
        result = verify_implementation_migration(recorded, current)
        self.assertEqual(result["mode"], "pre-two-phase-production-compatibility")
        self.assertEqual(set(result["changes"]), set(BASELINE))

    def test_rejects_unrelated_or_unknown_baseline_drift(self):
        recorded = {key: value for key, value in BASELINE.items() if value is not None}
        current = dict(TARGET)
        recorded["other.py"] = "old"
        current["other.py"] = "new"
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_implementation_migration(recorded, current)
        recorded.pop("other.py")
        current.pop("other.py")
        recorded[next(iter(BASELINE))] = "unknown"
        with self.assertRaisesRegex(ValueError, "not on"):
            verify_implementation_migration(recorded, current)

    def test_exact_current_profile_needs_no_migration(self):
        self.assertEqual(verify_implementation_migration(TARGET, TARGET)["changes"], {})


if __name__ == "__main__":
    unittest.main()
