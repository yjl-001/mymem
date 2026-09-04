from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_bank import V4_BANK_RECORD_SCHEMA
from memgen.experience.v4_2_curated import (
    V4_2_CURATED_BANK_MANIFEST_SCHEMA,
    V4_2_CURATED_CONSTRUCTION_VERSION,
    V4_2_CURATED_EXPECTED_DECISION_COUNTS,
    V4_2_CURATED_POLICY_SCHEMA,
    V42CuratedProfile,
    build_curated_manifest,
    build_curated_record,
    load_and_validate_curation_policy,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
    V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
)
import scripts.curate_v4_2_local_direct_bank as builder


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "configs/experiments/gsm8k/v4_2_local_curation_policy.json"
)


def source_manifest_from_policy() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    bank_ids = [item["bank_id"] for item in policy["decisions"]]
    return {
        "schema_version": V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
        "manifest_sha256": policy["source_manifest_sha256"],
        "profile_sha256": policy["source_profile_sha256"],
        "record_order_sha256": policy["source_record_order_sha256"],
        "record_count": policy["expected_source_record_count"],
        "bank_ids": bank_ids,
    }


def local_direct_record(bank_id: str, support: int = 7) -> dict:
    record = {
        "schema_version": V4_BANK_RECORD_SCHEMA,
        "construction_version": V4_2_LOCAL_DIRECT_CONSTRUCTION_VERSION,
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "quality_tier": "provisional_local_direct",
        "cluster": {"cluster_key": f"cluster-{bank_id}"},
        "process_card": {"fixture": bank_id},
        "construction": {
            "sample_ids": [f"{bank_id}-sample-{index}" for index in range(support)],
            "distinct_sample_count": support,
        },
        "roles": {
            "target_online_injectable": True,
            "reference_online_injectable": False,
            "auxiliary": None,
        },
        "compiler_contract": {
            "layer_number": 24,
            "all_kv_groups": True,
            "canonical_pre_rope": True,
            "relative_phase_delta": 0,
            "attention_backend": "sdpa",
        },
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


class V42CuratedBankTests(unittest.TestCase):
    def test_policy_is_complete_hash_bound_and_has_frozen_counts(self) -> None:
        source_manifest = source_manifest_from_policy()
        policy, decisions = load_and_validate_curation_policy(
            POLICY_PATH, source_manifest=source_manifest
        )
        self.assertEqual(policy["schema_version"], V4_2_CURATED_POLICY_SCHEMA)
        self.assertEqual(len(decisions), 24)
        observed = {
            name: sum(item["decision"] == name for item in decisions)
            for name in V4_2_CURATED_EXPECTED_DECISION_COUNTS
        }
        self.assertEqual(observed, V4_2_CURATED_EXPECTED_DECISION_COUNTS)
        self.assertEqual(
            sum(item["decision"] in {"primary", "conditional"} for item in decisions),
            17,
        )

        drifted = dict(source_manifest)
        drifted["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest_sha256 binding drifted"):
            load_and_validate_curation_policy(POLICY_PATH, source_manifest=drifted)

    def test_profile_states_exactly_which_review_was_and_was_not_done(self) -> None:
        profile = V42CuratedProfile()
        self.assertTrue(profile.static_process_card_review_performed)
        self.assertFalse(profile.full_construction_evidence_review_performed)
        self.assertFalse(profile.independent_review_performed)
        self.assertFalse(profile.semantic_api_audit_performed)
        self.assertEqual(profile.expected_source_record_count, 24)
        self.assertEqual(profile.expected_retained_record_count, 17)
        self.assertEqual(profile.expected_retained_evidence_count, 116)
        self.assertEqual(profile.injection_layer, 24)
        with self.assertRaisesRegex(ValueError, "must not overclaim"):
            V42CuratedProfile(independent_review_performed=True)

    def test_curated_record_preserves_semantic_id_but_rehashes_provenance(self) -> None:
        profile = V42CuratedProfile()
        source = local_direct_record("bank-a", support=7)
        record = build_curated_record(
            source_record=source,
            decision={
                "bank_id": "bank-a",
                "decision": "primary",
                "reason": "an actionable repair",
                "semantic_category": "rate scaling",
            },
            policy_sha256="policy-hash",
            profile=profile,
        )
        self.assertEqual(record["bank_id"], source["bank_id"])
        self.assertNotEqual(record["record_sha256"], source["record_sha256"])
        self.assertEqual(record["construction_version"], V4_2_CURATED_CONSTRUCTION_VERSION)
        self.assertEqual(record["quality_tier"], "provisional_local_curated")
        self.assertTrue(record["curation"]["bank_identity_preserved"])
        self.assertEqual(
            record["curation"]["source_record_sha256"], source["record_sha256"]
        )
        self.assertEqual(
            record["record_sha256"],
            canonical_json_sha256(
                {key: value for key, value in record.items() if key != "record_sha256"}
            ),
        )

    def test_manifest_binds_source_policy_and_exact_17_by_116_bank(self) -> None:
        policy, all_decisions = load_and_validate_curation_policy(
            POLICY_PATH, source_manifest=source_manifest_from_policy()
        )
        retained = [
            item
            for item in all_decisions
            if item["decision"] in {"primary", "conditional"}
        ]
        # Fourteen support-seven and three support-six fixtures total 116.
        supports = [7] * 14 + [6] * 3
        profile = V42CuratedProfile()
        records = [
            build_curated_record(
                source_record=local_direct_record(decision["bank_id"], support=support),
                decision=decision,
                policy_sha256=canonical_json_sha256(policy),
                profile=profile,
            )
            for decision, support in zip(retained, supports)
        ]
        source_manifest = source_manifest_from_policy()
        source_manifest["inputs"] = {
            "experiences_sha256": "experiences",
            "split_manifest_sha256": "split",
            "repository": {"implementation_sha256": {"old": "source"}},
        }
        source_manifest["source_signature_teacher"] = {"model": "historical"}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            paths = {
                name: temporary / name
                for name in ("manifest.json", "records.jsonl", "profile.json", "report.json")
            }
            for path in paths.values():
                path.write_text("fixture\n", encoding="utf-8")
            manifest = build_curated_manifest(
                records=records,
                source_manifest=source_manifest,
                source_manifest_path=paths["manifest.json"],
                source_records_path=paths["records.jsonl"],
                source_profile_path=paths["profile.json"],
                source_report_path=paths["report.json"],
                policy_path=POLICY_PATH,
                policy_sha256=canonical_json_sha256(policy),
                decisions=all_decisions,
                profile=profile,
                implementation_sha256={"curator.py": "hash"},
            )
        self.assertEqual(manifest["schema_version"], V4_2_CURATED_BANK_MANIFEST_SCHEMA)
        self.assertEqual(manifest["record_count"], 17)
        self.assertEqual(manifest["evidence_count"], 116)
        self.assertEqual(manifest["curation"]["decision_counts"]["hard_reject"], 4)
        self.assertEqual(len(manifest["curation"]["excluded_bank_ids"]), 7)
        self.assertFalse(manifest["qualified_for_online_use"])
        self.assertEqual(manifest["external_api_calls_made"], 0)
        self.assertEqual(
            manifest["manifest_sha256"],
            canonical_json_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            ),
        )

    def test_curator_has_no_model_network_or_credential_interface(self) -> None:
        source = inspect.getsource(builder)
        self.assertNotIn("TeacherClient", source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("urllib", source)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        argument_source = inspect.getsource(builder.parse_args)
        self.assertNotIn("api-key", argument_source)
        self.assertNotIn("model", argument_source)
        self.assertNotIn("base-url", argument_source)

    def test_one_command_script_is_valid_and_runs_the_three_offline_steps(self) -> None:
        script = ROOT / "scripts/experiments/gsm8k/run_v4_2_curated_offline.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text(encoding="utf-8")
        self.assertIn("curate_v4_2_local_direct_bank.py", source)
        self.assertIn("compile_v4_side_kv.py", source)
        self.assertIn("compile_v4_selector_anchors.py", source)
        self.assertIn("--layer 24", source)
        self.assertNotIn("build_v4_2_semantic_bank.py", source)


if __name__ == "__main__":
    unittest.main()
