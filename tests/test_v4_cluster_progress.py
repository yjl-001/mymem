from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_bank import (
    V4_CARD_PROMPT_VERSION,
    V4_CARD_REVIEW_PROMPT_VERSION,
    V4_CLUSTER_PROMPT_VERSION,
    V4_SIGNATURE_PROMPT_VERSION,
    V4ConstructionProfile,
    V4RepairSignature,
)
from scripts.build_v4_repair_bank import (
    CLUSTER_MAP_SCHEMA,
    CLUSTER_REDUCE_SCHEMA,
    CLUSTER_UNIT_RECORD_SCHEMA,
    MAX_CLUSTER_REDUCE_ROUNDS,
    MAX_CLUSTER_REQUEST_CHARACTERS,
    SIGNATURE_RECORD_SCHEMA,
    _bounded_batches,
    _merge_exact_prototypes,
    _semantic_sort_key,
    _signature_sort_key,
    parse_cluster_map_payload,
)
from scripts.inspect_v4_cluster_progress import inspect_cluster_progress


TEACHER = {
    "model": "deepseek-v4-flash",
    "base_url": "https://api.example.test",
    "temperature": 0.0,
    "thinking": "disabled",
}


def signature(suffix: str, *, second_group: bool = False) -> V4RepairSignature:
    return V4RepairSignature(
        experience_id=f"experience-{suffix}",
        sample_id=f"sample-{suffix}",
        experience_type="answer_correctness",
        problem_structure=(
            "a chain of retained dependent states"
            if second_group
            else "a sequence of dependent quantity updates"
        ),
        decision_point="before applying the following dependent update",
        failure_mechanism=(
            "a dependent state is dropped before reuse"
            if second_group
            else "an intermediate state is discarded too early"
        ),
        repair_operator=(
            "retain each state for the following update"
            if second_group
            else "carry each intermediate state into the next update"
        ),
        verification_operator="check every update against the preceding state",
        applicable=True,
        rejection_reason=None,
        source_provenance_sha256=f"provenance-{suffix}",
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def unit_record(
    *,
    stage: str,
    unit_id: str,
    input_value: object,
    payload: dict,
) -> dict:
    record = {
        "schema_version": CLUSTER_UNIT_RECORD_SCHEMA,
        "prompt_version": V4_CLUSTER_PROMPT_VERSION,
        "stage": stage,
        "unit_id": unit_id,
        "created_at": "fixture",
        "teacher": TEACHER,
        "construction_input_sha256": canonical_json_sha256(input_value),
        "request_character_count": 100,
        "payload": payload,
        "payload_sha256": canonical_json_sha256(payload),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def build_fixture(directory: Path, *, include_reduce: bool) -> tuple[Path, ...]:
    profile = V4ConstructionProfile()
    write_json(
        directory / "construction_profile.json",
        {
            "profile": profile.to_dict(),
            "profile_sha256": profile.profile_sha256,
            "teacher": TEACHER,
            "prompt_versions": {
                "signature": V4_SIGNATURE_PROMPT_VERSION,
                "cluster": V4_CLUSTER_PROMPT_VERSION,
                "card": V4_CARD_PROMPT_VERSION,
                "review": V4_CARD_REVIEW_PROMPT_VERSION,
            },
            "clustering": {
                "method": "bounded_map_reduce",
                "map_batch_size": 3,
                "reduce_batch_size": 10,
                "max_request_characters": MAX_CLUSTER_REQUEST_CHARACTERS,
                "max_reduce_rounds": MAX_CLUSTER_REDUCE_ROUNDS,
            },
        },
    )
    signatures = tuple(signature(suffix) for suffix in "abc") + tuple(
        signature(suffix, second_group=True) for suffix in "def"
    )
    signature_records = [
        {
            "schema_version": SIGNATURE_RECORD_SCHEMA,
            "prompt_version": V4_SIGNATURE_PROMPT_VERSION,
            "created_at": "fixture",
            "generation_status": "teacher_validated",
            "teacher": TEACHER,
            "construction_input_sha256": f"input-{item.experience_id}",
            "signature": item.to_dict(),
            "signature_sha256": item.signature_sha256,
        }
        for item in signatures
    ]
    write_jsonl(directory / "repair_signatures.jsonl", signature_records)

    map_records = []
    map_prototypes = []
    batches = _bounded_batches(signatures, batch_size=3, key=_signature_sort_key)
    for batch_index, batch in enumerate(batches):
        unit_id = f"map-{batch[0].experience_type}-{batch_index:05d}"
        label = "first-group" if batch_index == 0 else "second-group"
        payload = {
            "schema_version": CLUSTER_MAP_SCHEMA,
            "assignments": {item.experience_id: label for item in batch},
        }
        map_records.append(
            unit_record(
                stage="map",
                unit_id=unit_id,
                input_value=[item.to_dict() for item in batch],
                payload=payload,
            )
        )
        parsed, _ = parse_cluster_map_payload(
            payload,
            signatures=batch,
            unit_id=unit_id,
        )
        map_prototypes.extend(parsed)
    write_jsonl(directory / "cluster_map_shards.jsonl", map_records)

    reduce_records: list[dict] = []
    if include_reduce:
        prototypes = _merge_exact_prototypes(map_prototypes, unit_id="post-map")
        reduce_batch = _bounded_batches(
            prototypes,
            batch_size=10,
            key=_semantic_sort_key,
        )[0]
        unit_id = "reduce-00-answer_correctness-00000"
        payload = {
            "schema_version": CLUSTER_REDUCE_SCHEMA,
            "assignments": {
                item["prototype_id"]: "merged-group" for item in reduce_batch
            },
        }
        reduce_records.append(
            unit_record(
                stage="reduce",
                unit_id=unit_id,
                input_value=[dict(item) for item in reduce_batch],
                payload=payload,
            )
        )
    write_jsonl(directory / "cluster_reduce_batches.jsonl", reduce_records)
    return tuple(
        directory / name
        for name in (
            "construction_profile.json",
            "repair_signatures.jsonl",
            "cluster_map_shards.jsonl",
            "cluster_reduce_batches.jsonl",
        )
    )


class V4ClusterProgressTests(unittest.TestCase):
    def test_reconstructs_complete_global_round_without_mutating_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            paths = build_fixture(directory, include_reduce=True)
            before = {path.name: file_sha256(path) for path in paths}

            report = inspect_cluster_progress(
                directory,
                top_limit=10,
                sample_limit=10,
                near_duplicate_limit=10,
            )
            repeated = inspect_cluster_progress(
                directory,
                top_limit=10,
                sample_limit=10,
                near_duplicate_limit=10,
            )

            after = {path.name: file_sha256(path) for path in paths}
            self.assertEqual(after, before)
            self.assertEqual(repeated["report_sha256"], report["report_sha256"])
            self.assertEqual(report["map"]["status"], "complete")
            self.assertEqual(report["map"]["completed_unit_count"], 2)
            self.assertEqual(report["reduce"]["status"], "global_complete")
            self.assertEqual(report["reduce"]["completed_round_count"], 1)
            summary = report["reduce"]["last_complete_state"]["summary"]
            self.assertEqual(summary["prototype_count"], 1)
            self.assertEqual(summary["qualified_prototype_count"], 1)
            self.assertEqual(
                summary["support_histogram"]["five_to_nine"][
                    "prototype_count"
                ],
                1,
            )
            qualified = report["reduce"]["last_complete_state"][
                "qualified_prototypes"
            ]
            self.assertEqual(len(qualified), 1)
            self.assertEqual(qualified[0]["distinct_sample_count"], 6)

    def test_reports_incomplete_reduce_round_and_last_complete_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            build_fixture(directory, include_reduce=False)

            report = inspect_cluster_progress(directory)

            self.assertEqual(report["reduce"]["status"], "round_incomplete")
            self.assertEqual(report["reduce"]["completed_round_count"], 0)
            self.assertEqual(report["reduce"]["last_complete_round"], -1)
            in_progress = report["reduce"]["in_progress_round"]
            self.assertEqual(in_progress["round"], 0)
            self.assertEqual(in_progress["required_api_batch_count"], 1)
            self.assertEqual(in_progress["completed_api_batch_count"], 0)
            summary = report["reduce"]["last_complete_state"]["summary"]
            self.assertEqual(summary["prototype_count"], 2)
            self.assertEqual(summary["qualified_prototype_count"], 0)
            self.assertEqual(
                summary["support_histogram"]["two_to_four"][
                    "prototype_count"
                ],
                2,
            )

    def test_rejects_tampered_cluster_unit_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            build_fixture(directory, include_reduce=False)
            map_path = directory / "cluster_map_shards.jsonl"
            records = [json.loads(line) for line in map_path.read_text().splitlines()]
            records[0]["payload"]["assignments"]["experience-a"] = "tampered"
            write_jsonl(map_path, records)

            with self.assertRaisesRegex(ValueError, "record hash mismatch"):
                inspect_cluster_progress(directory)


if __name__ == "__main__":
    unittest.main()
