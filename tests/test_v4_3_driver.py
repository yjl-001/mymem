"""Exercise persisted plans, resumable cases, and smoke→full authorization locally."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_bank import canonical_hash, seal
from memgen.experience.v4_3_audit import build_plan
import scripts.audit_v4_3_unified_memory as driver
from tests.test_v4_3_audit import audit_fixture, result_for


class V43DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.outputs, cls.events, cls.controls, cls.binding = audit_fixture()

    def prepared(self, mode):
        plan = build_plan(candidates=self.outputs["candidate_bank_records.jsonl"], events=self.events,
                          controls=self.controls, binding=self.binding, mode=mode)
        experiment = {"test_fixture_only": True}
        profile = seal({"experiment": experiment, "experiment_identity_sha256": canonical_hash(experiment),
                        "plan_sha256": plan["plan_sha256"]}, "profile_sha256")
        return ({}, {}, None, [], {}, self.binding, self.controls, plan, profile)

    def args(self, root, mode="smoke", **kwargs):
        values = dict(mode=mode, output_dir=root / mode, bank_dir=root / "bank", side_kv_dir=root / "side",
                      cache_manifest=root / "source" / "cache.json", plan_only=True, resume=True,
                      validate_only=False, smoke_report=root / "smoke" / "v4_3_audit_report.json")
        values.update(kwargs)
        return SimpleNamespace(**values)

    def call(self, args, prepared):
        with patch.object(driver, "parse_args", return_value=args), patch.object(driver, "prepare", return_value=prepared):
            driver.main()

    def seed_cases(self, args, prepared, omit=0):
        plan, profile = prepared[-2:]
        for case in plan["cases"][omit:]:
            atomic_json(args.output_dir / "cases" / (case["case_id"] + ".json"),
                        result_for(case, profile["profile_sha256"]), immutable=True)

    def test_resume_reconstructs_complete_smoke_then_full_and_validates_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("smoke", "full"):
                prepared, args = self.prepared(mode), self.args(root, mode)
                self.call(args, prepared)
                self.seed_cases(args, prepared)
                args.plan_only = False
                self.call(args, prepared)  # Complete resume never imports/loads a model.
                report = read_json(args.output_dir / "v4_3_audit_report.json")
                self.assertTrue(report["complete"])
                args.validate_only = True
                before = {p: p.stat().st_mtime_ns for p in args.output_dir.rglob("*.json")}
                self.call(args, prepared)
                self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def test_full_without_authenticated_smoke_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp), "full", plan_only=False)
            with self.assertRaisesRegex(ValueError, "completed smoke"):
                self.call(args, self.prepared("full"))
            self.assertFalse(args.output_dir.exists())

    def test_incomplete_resume_is_not_a_passed_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared, args = self.prepared("smoke"), self.args(Path(tmp))
            self.call(args, prepared)
            self.seed_cases(args, prepared, omit=1)
            args.plan_only, args.validate_only = False, True
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.call(args, prepared)

    def test_resealed_smoke_profile_cannot_substitute_another_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared, args = self.prepared("smoke"), self.args(Path(tmp))
            self.call(args, prepared)
            self.seed_cases(args, prepared)
            args.plan_only = False
            self.call(args, prepared)
            profile = deepcopy(prepared[-1])
            profile["experiment"] = {"another_fixture": True}
            atomic_json(args.output_dir / "v4_3_audit_profile.json", seal(profile, "profile_sha256"))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                driver.authenticate_smoke(args.smoke_report, prepared[-1]["experiment_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
