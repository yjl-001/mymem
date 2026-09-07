from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v4_source_state import (
    FAILURE_TENSORS,
    PROMPT_TENSORS,
    SUCCESS_GATE_TENSORS,
    V4_SOURCE_STATE_TENSORS,
    V4SourceStateCache,
    build_gate_reachability_report,
    build_source_state_manifest,
    finalize_event,
    independent_support,
    load_source_state_cache,
    validate_events,
    validate_tensor_alignment,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeRow:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeTensor:
    def __init__(self, shape, *, dtype="bfloat16", rows=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.rows = rows

    def __getitem__(self, index):
        if self.rows is None:
            raise TypeError("Fake state tensors are shape-only")
        return FakeRow(self.rows[index])


def common(sample_id: str, bank_id: str, *, medoid: bool = False) -> dict:
    return {
        "experience_id": f"experience-{sample_id}",
        "sample_id": sample_id,
        "independent_sample_id": f"independent-{sample_id}",
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "logical_split": "bank-source",
        "dataset_split": "train",
        "source_index": int(sample_id[-1]),
        "question_sha256": f"question-{sample_id}",
        "is_medoid": medoid,
        "curation_tier": "primary" if bank_id == "bank-a" else "conditional",
        "construction_profile_sha256": "construction-profile",
        "bank_record_sha256": f"record-{bank_id}",
        "contrast_pair": {
            "target_episode_id": f"success-{sample_id}",
            "reference_episode_id": f"failure-{sample_id}",
            "paired_success_failure": True,
        },
        "completion_hashes": {
            "verified_success_completion_sha256": f"success-hash-{sample_id}",
            "verified_failure_completion_sha256": f"failure-hash-{sample_id}",
        },
    }


def prompt_event(
    sample_id: str,
    bank_id: str,
    row: int,
    *,
    failure_attempts: int,
    success_attempts: int,
    medoid: bool = False,
) -> dict:
    return finalize_event(
        {
            **common(sample_id, bank_id, medoid=medoid),
            "event_id": f"experience-{sample_id}::prompt-semantic",
            "event_kind": "prompt_semantic",
            "online_reachable_safety_negative": False,
            "tensor_rows": {name: row for name in PROMPT_TENSORS},
            "prompt_token_count": 10,
            "prompt_token_ids_sha256": f"prompt-{sample_id}",
            "question_token_start": 3,
            "question_token_end_exclusive": 8,
            "question_token_count": 5,
            "failure_gate_eligible": failure_attempts > 0,
            "failure_gate_attempt_count": failure_attempts,
            "failure_gate_rejection_reason": (
                None if failure_attempts else "failure_has_no_joint_gate"
            ),
            "success_gate_eligible": success_attempts > 0,
            "success_gate_attempt_count": success_attempts,
            "success_gate_rejection_reason": (
                None if success_attempts else "success_has_no_joint_gate"
            ),
        }
    )


def gate_event(
    sample_id: str,
    bank_id: str,
    row: int,
    attempt: int,
    *,
    success: bool,
    medoid: bool = False,
) -> dict:
    kind = "success_gate_attempt" if success else "failure_gate_attempt"
    tensor_names = SUCCESS_GATE_TENSORS if success else FAILURE_TENSORS
    value = {
        **common(sample_id, bank_id, medoid=medoid),
        "event_id": f"experience-{sample_id}::{kind}::attempt-{attempt}",
        "event_kind": kind,
        "online_reachable_safety_negative": success,
        "attempt_number": attempt,
        "reasoning_rank": attempt + 1,
        "candidate_rank": attempt + 2,
        "token_position": 11 + attempt,
        "window_token_count": attempt + 2,
        "tensor_rows": {name: row for name in tensor_names},
        "gate_diagnostics": {
            "gate_eligible": True,
            "gate_rejection_reason": None,
            "attention_entropy": 2.0,
            "persistence_risk": 0.2,
            "high_entropy_threshold": 1.5,
            "low_entropy_threshold": 1.0,
            "risk_threshold": 0.1,
            "state_before": "ARMED",
            "state_after": "DISARMED",
            "logit_summary": {
                "maximum_logit": 5.0,
                "top1_top2_logit_gap": 0.5,
                "logsumexp": 6.0,
                "predictive_entropy": 1.0,
            },
        },
        "prefix_alignment": {
            "prefix_token_count": 12 + attempt,
            "prefix_token_ids_sha256": f"prefix-{sample_id}-{kind}-{attempt}",
            "prefix_includes_current_token": True,
            "token_position_matches_prefix_end": True,
        },
    }
    if not success:
        value["matched_success_alignment"] = {
            "state_role": "offline_repair_direction_control",
            "online_reachable_safety_negative": False,
            "alignment_method": "normalized_reasoning_progress_endpoint_preserving",
            "reasoning_rank": attempt,
            "token_position": 20 + attempt,
            "window_token_count": attempt + 1,
            "prefix_token_ids_sha256": f"aligned-{sample_id}-{attempt}",
        }
    return finalize_event(value)


def fixture_events() -> list[dict]:
    specifications = (
        ("a1", "bank-a", 2, 1, True),
        ("a2", "bank-a", 1, 0, False),
        ("a3", "bank-a", 0, 0, False),
        ("b1", "bank-b", 1, 1, True),
        ("b2", "bank-b", 1, 0, False),
    )
    events: list[dict] = []
    failure_row = 0
    success_row = 0
    for prompt_row, (sample_id, bank_id, failures, successes, medoid) in enumerate(
        specifications
    ):
        events.append(
            prompt_event(
                sample_id,
                bank_id,
                prompt_row,
                failure_attempts=failures,
                success_attempts=successes,
                medoid=medoid,
            )
        )
        for attempt in range(1, failures + 1):
            events.append(
                gate_event(
                    sample_id,
                    bank_id,
                    failure_row,
                    attempt,
                    success=False,
                    medoid=medoid,
                )
            )
            failure_row += 1
        for attempt in range(1, successes + 1):
            events.append(
                gate_event(
                    sample_id,
                    bank_id,
                    success_row,
                    attempt,
                    success=True,
                    medoid=medoid,
                )
            )
            success_row += 1
    return events


def fake_tensors(events: list[dict]) -> dict[str, FakeTensor]:
    counts = {
        name: sum(name in event["tensor_rows"] for event in events)
        for name in V4_SOURCE_STATE_TENSORS
    }
    result: dict[str, FakeTensor] = {}
    for name, count in counts.items():
        if name.endswith("_masks"):
            rows = [[False] * 29 + [True] * 3 for _ in range(count)]
            result[name] = FakeTensor((count, 32), dtype="bool", rows=rows)
        elif name.endswith("_windows"):
            result[name] = FakeTensor((count, 32, 4))
        else:
            result[name] = FakeTensor((count, 4))
    return result


def build_cache_fixture(directory: Path) -> tuple[Path, list[dict]]:
    events = fixture_events()
    tensors = fake_tensors(events)
    validate_tensor_alignment(events=events, tensors=tensors)
    tensor_path = directory / "v4_source_states.safetensors"
    event_path = directory / "v4_source_state_events.jsonl"
    reachability_path = directory / "gate_reachability_report.json"
    tensor_path.write_bytes(b"fixture-tensors")
    event_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    reachability = build_gate_reachability_report(
        events=events, bank_ids=("bank-a", "bank-b")
    )
    reachability_path.write_text(
        json.dumps(reachability, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = build_source_state_manifest(
        tensors=tensors,
        events=events,
        reachability_report=reachability,
        tensor_path=tensor_path,
        event_path=event_path,
        reachability_path=reachability_path,
        repository_revision="fixture-revision",
        reasoner={
            "model_name": "fixture-model",
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
        },
        configuration={
            "layer_number": 24,
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "maximum_gate_attempts": 3,
            "maximum_hidden_window": 32,
            "support_unit": "independent_sample",
        },
        provenance={
            "construction_profile_sha256": "construction-profile",
            "bank_manifest_logical_sha256": "bank-manifest",
            "side_kv_manifest_logical_sha256": "side-kv-manifest",
        },
        implementation_sha256={"fixture.py": "fixture-sha"},
    )
    manifest_path = directory / "v4_source_state_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, events


class V4SourceStateCacheTests(unittest.TestCase):
    def test_schema_separates_aligned_success_from_actual_success_gate(self) -> None:
        events = fixture_events()
        validate_events(events)
        failure = next(
            row for row in events if row["event_kind"] == "failure_gate_attempt"
        )
        success = next(
            row for row in events if row["event_kind"] == "success_gate_attempt"
        )
        self.assertFalse(
            failure["matched_success_alignment"][
                "online_reachable_safety_negative"
            ]
        )
        self.assertTrue(success["online_reachable_safety_negative"])
        self.assertNotIn(
            "success_gate_windows", failure["tensor_rows"]
        )
        self.assertNotIn(
            "aligned_success_windows", success["tensor_rows"]
        )

    def test_multiple_attempts_do_not_inflate_independent_support(self) -> None:
        events = fixture_events()
        failures = independent_support(
            events, event_kind="failure_gate_attempt"
        )
        self.assertEqual(failures, {"bank-a": 2, "bank-b": 2})
        reachability = build_gate_reachability_report(
            events=events, bank_ids=("bank-a", "bank-b")
        )
        self.assertEqual(reachability["failure_gate_event_count"], 5)
        self.assertEqual(
            reachability["failure_gate_reachable_independent_sample_count"], 4
        )
        self.assertEqual(
            reachability["failure_gate_unreachable_independent_sample_count"], 1
        )
        self.assertEqual(
            reachability["per_bank"]["bank-a"][
                "failure_gate_reachable_independent_sample_count"
            ],
            2,
        )

    def test_tensor_indices_must_cover_contiguous_rows(self) -> None:
        events = fixture_events()
        misaligned = dict(
            next(
                row
                for row in events
                if row["event_kind"] == "failure_gate_attempt"
            )
        )
        misaligned.pop("record_sha256")
        misaligned["tensor_rows"] = dict(misaligned["tensor_rows"])
        misaligned["tensor_rows"]["aligned_success_masks"] += 1
        with self.assertRaisesRegex(ValueError, "paired tensor rows"):
            finalize_event(misaligned)

        broken = [dict(row) for row in events]
        index = next(
            i for i, row in enumerate(broken) if row["event_kind"] == "failure_gate_attempt"
        )
        broken[index] = dict(broken[index])
        broken[index]["tensor_rows"] = dict(broken[index]["tensor_rows"])
        broken[index]["tensor_rows"] = {name: 99 for name in FAILURE_TENSORS}
        broken[index] = finalize_event(broken[index])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_tensor_alignment(events=broken, tensors=fake_tensors(events))

    def test_manifest_authenticates_metadata_without_loading_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, events = build_cache_fixture(Path(temporary))
            cache = load_source_state_cache(manifest_path, load_tensors=False)
            self.assertIsNone(cache.tensors)
            self.assertEqual(len(cache.events), len(events))
            self.assertTrue(cache.manifest["offline_only"])
            self.assertFalse(cache.manifest["qualified_for_online_use"])
            self.assertFalse(cache.manifest["contains_reward_or_answer_signal"])

            event_path = Path(temporary) / "v4_source_state_events.jsonl"
            event_path.write_text(event_path.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or corrupted"):
                load_source_state_cache(manifest_path, load_tensors=False)

    def test_manifest_logical_hash_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = build_cache_fixture(Path(temporary))
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["qualified_for_online_use"] = True
            manifest_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safety contract|hash mismatch"):
                load_source_state_cache(manifest_path, load_tensors=False)

    def test_cache_rejects_answer_or_reward_fields(self) -> None:
        value = prompt_event(
            "a9",
            "bank-a",
            0,
            failure_attempts=0,
            success_attempts=0,
        )
        value.pop("record_sha256")
        value["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "forbidden answer/reward"):
            finalize_event(value)

    def test_cpu_auditor_contains_no_model_loading_path(self) -> None:
        source = (ROOT / "scripts/audit_v4_source_state_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("AutoModel", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("transformers", source)
        self.assertIn('"reasoner_loaded": False', source)
        self.assertIn('"online_selector_tensor": None', source)
        self.assertIn('"bank_size_bias"', source)
        self.assertIn("independent_sample_macro", source)

    def test_bank_size_bias_diagnostic_is_cpu_and_sample_explicit(self) -> None:
        from scripts.audit_v4_source_state_cache import _bank_size_bias

        rows = [
            {
                "top1_bank_id": "bank-a",
                "independent_sample_id": f"independent-{index}",
            }
            for index in range(4)
        ]
        diagnostic = _bank_size_bias(
            rows,
            bank_ids=("bank-a", "bank-b"),
            failure_support={"bank-a": 4, "bank-b": 2},
        )
        self.assertTrue(diagnostic["potential_bank_size_bias_detected"])
        self.assertEqual(diagnostic["zero_top1_selection_bank_count"], 1)
        self.assertEqual(
            diagnostic["per_bank"]["bank-b"][
                "failure_gate_independent_sample_support"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
