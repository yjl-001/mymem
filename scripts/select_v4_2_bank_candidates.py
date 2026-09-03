#!/usr/bin/env python3
"""Select a small, high-quality V4.2 synthesis basis without external APIs.

The completed local cluster plan remains an immutable discovery artifact.  This
command never attempts to recover unsupported atoms and never optimizes source
coverage.  It authenticates the local clusters and their three embedding views,
ranks recurring candidates by weakest-view cohesion, suppresses near-duplicate
candidate centroids, and writes at most forty-eight compact synthesis packets.

No teacher client is constructed, no credential is read, and no paid stage is
started.  The target runtime-bank cap is recorded for later card/anchor stages;
shortlist membership alone never qualifies a memory for online use.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_2_bank import (
    V4_2_DEFAULT_MAX_SYNTHESIS_CANDIDATES,
    V4_2_DEFAULT_MIN_SUPPORT_COHESION_QUANTILE,
    V4_2_DEFAULT_PREFERRED_SUPPORT,
    V4_2_DEFAULT_REDUNDANCY_APPLICABILITY_THRESHOLD,
    V4_2_DEFAULT_REDUNDANCY_MECHANISM_THRESHOLD,
    V4_2_DEFAULT_REDUNDANCY_REPAIR_THRESHOLD,
    V4_2_DEFAULT_REVIEW_BATCH_SIZE,
    V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE,
    V4_2_DEFAULT_TARGET_RUNTIME_BANK_CAP,
    V4_2_EMBEDDING_MANIFEST_SCHEMA,
    V4_2_LOCAL_CLUSTER_PLAN_SCHEMA,
    V4_2_REVIEW_PACKET_SCHEMA,
    V4_2_SHORTLIST_MANIFEST_SCHEMA,
    V4_2_SHORTLIST_PREFLIGHT_SCHEMA,
    V4_2_SYNTHESIS_SEMANTIC_FIELDS,
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
    V42ShortlistProfile,
    validate_v4_2_cluster_payload,
    validate_v4_2_local_atom_payload,
)


LOCAL_ATOM_RECORD_SCHEMA = "memgen-v4.2-local-repair-atom-record-v1"
LOCAL_CLUSTER_RECORD_SCHEMA = "memgen-v4.2-local-cluster-candidate-record-v1"
SHORTLIST_PROFILE_RECORD_SCHEMA = "memgen-v4.2-shortlist-profile-record-v1"
CANDIDATE_QUALITY_REPORT_SCHEMA = "memgen-v4.2-candidate-quality-report-v1"
REDUNDANCY_EDGE_SCHEMA = "memgen-v4.2-candidate-redundancy-edge-v1"
SELECTED_CANDIDATE_SCHEMA = "memgen-v4.2-selected-synthesis-candidate-v1"
REJECTED_CANDIDATE_SCHEMA = "memgen-v4.2-rejected-shortlist-candidate-v1"
EMBEDDING_VIEW_NAMES = ("mechanism", "repair", "applicability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-construction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preferred-support", type=int, default=V4_2_DEFAULT_PREFERRED_SUPPORT
    )
    parser.add_argument(
        "--minimum-support-cohesion-quantile",
        type=float,
        default=V4_2_DEFAULT_MIN_SUPPORT_COHESION_QUANTILE,
    )
    parser.add_argument(
        "--redundancy-mechanism-threshold",
        type=float,
        default=V4_2_DEFAULT_REDUNDANCY_MECHANISM_THRESHOLD,
    )
    parser.add_argument(
        "--redundancy-repair-threshold",
        type=float,
        default=V4_2_DEFAULT_REDUNDANCY_REPAIR_THRESHOLD,
    )
    parser.add_argument(
        "--redundancy-applicability-threshold",
        type=float,
        default=V4_2_DEFAULT_REDUNDANCY_APPLICABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--max-synthesis-candidates",
        type=int,
        default=V4_2_DEFAULT_MAX_SYNTHESIS_CANDIDATES,
    )
    parser.add_argument(
        "--target-runtime-bank-cap",
        type=int,
        default=V4_2_DEFAULT_TARGET_RUNTIME_BANK_CAP,
    )
    parser.add_argument(
        "--synthesis-batch-size",
        type=int,
        default=V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE,
    )
    parser.add_argument(
        "--review-batch-size",
        type=int,
        default=V4_2_DEFAULT_REVIEW_BATCH_SIZE,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Authenticate and reuse a completed shortlist; rebuild partial output.",
    )
    return parser.parse_args()


def _profile(args: argparse.Namespace) -> V42ShortlistProfile:
    return V42ShortlistProfile(
        preferred_distinct_support=args.preferred_support,
        minimum_support_cohesion_quantile=(
            args.minimum_support_cohesion_quantile
        ),
        redundancy_mechanism_threshold=args.redundancy_mechanism_threshold,
        redundancy_repair_threshold=args.redundancy_repair_threshold,
        redundancy_applicability_threshold=(
            args.redundancy_applicability_threshold
        ),
        max_synthesis_candidates=args.max_synthesis_candidates,
        target_runtime_bank_cap=args.target_runtime_bank_cap,
        synthesis_batch_size=args.synthesis_batch_size,
        review_batch_size=args.review_batch_size,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def _implementation_state() -> dict[str, str]:
    paths = (
        "memgen/experience/phase1.py",
        "memgen/experience/v4_bank.py",
        "memgen/experience/v4_2_bank.py",
        "scripts/build_v4_2_local_clusters.py",
        "scripts/select_v4_2_bank_candidates.py",
    )
    return {relative: file_sha256(PROJECT_ROOT / relative) for relative in paths}


def _logical_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return canonical_json_sha256(
        {key: item for key, item in value.items() if key != hash_field}
    )


def _normalized_rows(value: np.ndarray, *, owner: str) -> np.ndarray:
    if value.dtype != np.float32 or value.ndim != 2:
        raise ValueError(f"{owner} must be a float32 matrix")
    norms = np.linalg.norm(value, axis=1)
    if not np.isfinite(value).all() or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"{owner} must contain finite unit vectors")
    return value


def load_authenticated_local_construction(
    directory: Path,
) -> tuple[
    V42ConstructionProfile,
    dict[str, Any],
    tuple[V42LocalRepairAtom, ...],
    tuple[V42LocalClusterCandidate, ...],
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Authenticate every local artifact used by shortlist selection."""

    directory = directory.expanduser().resolve()
    required = (
        "construction_profile.json",
        "local_atoms.jsonl",
        "multiview_embeddings_manifest.json",
        "mechanism_embeddings.npy",
        "repair_embeddings.npy",
        "applicability_embeddings.npy",
        "local_clusters.jsonl",
        "cluster_review_packets.jsonl",
        "local_cluster_plan.json",
    )
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f"Missing V4.2 local construction artifact: {name}")

    construction_record = _load_json(directory / "construction_profile.json")
    construction_profile = V42ConstructionProfile(
        **construction_record.get("profile", {})
    )
    if construction_record.get("profile_sha256") != construction_profile.profile_sha256:
        raise ValueError("V4.2 local construction profile hash mismatch")
    if construction_record.get("external_api_calls") != 0:
        raise ValueError("V4.2 local construction unexpectedly records API calls")

    plan = _load_json(directory / "local_cluster_plan.json")
    if plan.get("schema_version") != V4_2_LOCAL_CLUSTER_PLAN_SCHEMA:
        raise ValueError("Unexpected V4.2 local cluster-plan schema")
    if plan.get("plan_sha256") != _logical_hash(plan, "plan_sha256"):
        raise ValueError("V4.2 local cluster-plan hash mismatch")
    if plan.get("profile_sha256") != construction_profile.profile_sha256:
        raise ValueError("V4.2 local plan/profile binding mismatch")
    if plan.get("external_api_calls") != 0 or plan.get("qualified_for_online_use") is not False:
        raise ValueError("V4.2 local plan has an invalid qualification state")
    artifact_hashes = plan.get("artifacts")
    if not isinstance(artifact_hashes, Mapping) or not set(required[:-1]).issubset(
        artifact_hashes
    ):
        raise ValueError("V4.2 local plan has incomplete artifact bindings")
    for name, expected_sha256 in artifact_hashes.items():
        path = directory / str(name)
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f"V4.2 local artifact hash mismatch: {name}")

    atoms: list[V42LocalRepairAtom] = []
    seen_atom_ids: set[str] = set()
    for record in _iter_jsonl(directory / "local_atoms.jsonl"):
        if record.get("schema_version") != LOCAL_ATOM_RECORD_SCHEMA:
            raise ValueError("Unexpected V4.2 local atom-record schema")
        payload = record.get("atom")
        if not isinstance(payload, Mapping):
            raise ValueError("V4.2 local atom record is missing its payload")
        atom = validate_v4_2_local_atom_payload(payload)
        if record.get("atom_sha256") != canonical_json_sha256(atom.to_dict()):
            raise ValueError(f"V4.2 local atom hash mismatch: {atom.experience_id}")
        if atom.experience_id in seen_atom_ids:
            raise ValueError(f"Duplicate V4.2 local atom: {atom.experience_id}")
        atoms.append(atom)
        seen_atom_ids.add(atom.experience_id)
    if tuple(sorted(atoms, key=lambda item: item.experience_id)) != tuple(atoms):
        raise ValueError("V4.2 local atom order drifted")
    if not atoms:
        raise ValueError("V4.2 local construction has no reasoning atoms")

    embedding_manifest = _load_json(
        directory / "multiview_embeddings_manifest.json"
    )
    if embedding_manifest.get("schema_version") != V4_2_EMBEDDING_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V4.2 embedding-manifest schema")
    if embedding_manifest.get("manifest_sha256") != _logical_hash(
        embedding_manifest, "manifest_sha256"
    ):
        raise ValueError("V4.2 embedding-manifest hash mismatch")
    if (
        embedding_manifest.get("model") != construction_profile.embedding_model
        or embedding_manifest.get("revision")
        != construction_profile.embedding_revision
    ):
        raise ValueError("V4.2 embedding model binding drifted")
    atom_order = [item.atom_id for item in atoms]
    if embedding_manifest.get("atom_order_sha256") != canonical_json_sha256(
        atom_order
    ):
        raise ValueError("V4.2 embedding atom order mismatch")
    embeddings: dict[str, np.ndarray] = {}
    dimensions: set[int] = set()
    for name in EMBEDDING_VIEW_NAMES:
        metadata = embedding_manifest.get("views", {}).get(name, {})
        path = directory / f"{name}_embeddings.npy"
        if metadata.get("path") != path.name:
            raise ValueError(f"V4.2 {name} embedding path drifted")
        if metadata.get("tensor_sha256") != file_sha256(path):
            raise ValueError(f"V4.2 {name} embedding hash mismatch")
        value = _normalized_rows(
            np.load(path, allow_pickle=False), owner=f"V4.2 {name} embeddings"
        )
        if list(value.shape) != metadata.get("shape") or value.shape[0] != len(atoms):
            raise ValueError(f"V4.2 {name} embedding shape mismatch")
        embeddings[name] = value
        dimensions.add(int(value.shape[1]))
    if len(dimensions) != 1:
        raise ValueError("V4.2 embedding views have different dimensions")

    candidates: list[V42LocalClusterCandidate] = []
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(directory / "local_clusters.jsonl"):
        if record.get("schema_version") != LOCAL_CLUSTER_RECORD_SCHEMA:
            raise ValueError("Unexpected V4.2 local cluster-record schema")
        payload = record.get("candidate")
        if not isinstance(payload, Mapping):
            raise ValueError("V4.2 local cluster record is missing its payload")
        candidate = validate_v4_2_cluster_payload(payload)
        if record.get("candidate_sha256") != canonical_json_sha256(
            candidate.to_dict()
        ):
            raise ValueError(f"V4.2 candidate hash mismatch: {candidate.candidate_id}")
        if candidate.candidate_id in candidate_payloads:
            raise ValueError(f"Duplicate V4.2 candidate: {candidate.candidate_id}")
        candidates.append(candidate)
        candidate_payloads[candidate.candidate_id] = candidate.to_dict()
    candidates.sort(key=lambda item: item.candidate_id)
    plan_candidates = plan.get("clusters")
    if not isinstance(plan_candidates, list):
        raise ValueError("V4.2 local plan is missing clusters")
    if any(not isinstance(item, Mapping) for item in plan_candidates):
        raise ValueError("V4.2 local plan contains an invalid cluster")
    if {
        item.candidate_id: item.to_dict() for item in candidates
    } != {
        item["candidate_id"]: item for item in plan_candidates
    } or len(plan_candidates) != len(candidates):
        raise ValueError("V4.2 local candidate records differ from the plan")

    atoms_by_id = {item.experience_id: item for item in atoms}
    review_packets: dict[str, dict[str, Any]] = {}
    for packet in _iter_jsonl(directory / "cluster_review_packets.jsonl"):
        if packet.get("schema_version") != V4_2_REVIEW_PACKET_SCHEMA:
            raise ValueError("Unexpected V4.2 review-packet schema")
        if packet.get("packet_sha256") != _logical_hash(packet, "packet_sha256"):
            raise ValueError("V4.2 review-packet hash mismatch")
        representatives = packet.get("representatives")
        if not isinstance(representatives, list):
            raise ValueError("V4.2 review packet is missing representatives")
        if packet.get("representative_evidence_sha256") != canonical_json_sha256(
            representatives
        ):
            raise ValueError("V4.2 review-packet evidence hash mismatch")
        candidate_id = str(packet.get("candidate_id", ""))
        candidate_payload = candidate_payloads.get(candidate_id)
        if candidate_payload is None or candidate_id in review_packets:
            raise ValueError(f"Unexpected or duplicate V4.2 review packet: {candidate_id}")
        candidate = validate_v4_2_cluster_payload(candidate_payload)
        if packet.get("membership_sha256") != candidate.membership_sha256:
            raise ValueError("V4.2 review packet membership drifted")
        representative_ids = [str(item.get("experience_id", "")) for item in representatives]
        if tuple(representative_ids) != candidate.representative_experience_ids:
            raise ValueError("V4.2 review packet representative order drifted")
        if len({str(item.get("sample_id", "")) for item in representatives}) != len(
            representatives
        ):
            raise ValueError("V4.2 review packet reuses a construction sample")
        for value in representatives:
            atom = atoms_by_id.get(str(value.get("experience_id", "")))
            if atom is None or atom.to_dict() != value:
                raise ValueError("V4.2 review packet contains an unauthenticated atom")
        review_packets[candidate_id] = packet
    if set(review_packets) != set(candidate_payloads):
        raise ValueError("V4.2 review packets do not cover all local candidates")

    source_info = {
        "directory": str(directory),
        "construction_profile_file_sha256": file_sha256(
            directory / "construction_profile.json"
        ),
        "construction_profile_sha256": construction_profile.profile_sha256,
        "local_cluster_plan_file_sha256": file_sha256(
            directory / "local_cluster_plan.json"
        ),
        "local_cluster_plan_sha256": plan["plan_sha256"],
        "embedding_manifest_file_sha256": file_sha256(
            directory / "multiview_embeddings_manifest.json"
        ),
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "candidate_count": len(candidates),
        "reasoning_atom_count": len(atoms),
    }
    return (
        construction_profile,
        plan,
        tuple(atoms),
        tuple(candidates),
        review_packets,
        embeddings,
        source_info,
    )


def candidate_centroids(
    candidates: Sequence[V42LocalClusterCandidate],
    atoms: Sequence[V42LocalRepairAtom],
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    index_by_id = {item.experience_id: index for index, item in enumerate(atoms)}
    result: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        indices = [index_by_id[item] for item in candidate.member_experience_ids]
        by_view: dict[str, np.ndarray] = {}
        for name in EMBEDDING_VIEW_NAMES:
            centroid = np.mean(embeddings[name][indices], axis=0, dtype=np.float64)
            norm = float(np.linalg.norm(centroid))
            if not np.isfinite(centroid).all() or norm <= 0.0:
                raise ValueError(f"Invalid {name} centroid: {candidate.candidate_id}")
            by_view[name] = (centroid / norm).astype(np.float32)
        result[candidate.candidate_id] = by_view
    return result


def _normalized_margin(value: float, threshold: float) -> float:
    if not threshold < 1.0:
        return 1.0 if value >= 1.0 else 0.0
    return (float(value) - threshold) / (1.0 - threshold)


def build_candidate_quality(
    candidates: Sequence[V42LocalClusterCandidate],
    *,
    construction_profile: V42ConstructionProfile,
    shortlist_profile: V42ShortlistProfile,
) -> tuple[dict[str, dict[str, Any]], float]:
    quality: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        margins = {
            "mechanism": _normalized_margin(
                candidate.mechanism_similarity_min,
                construction_profile.mechanism_threshold,
            ),
            "repair": _normalized_margin(
                candidate.repair_similarity_min,
                construction_profile.repair_threshold,
            ),
            "applicability": _normalized_margin(
                candidate.applicability_similarity_min,
                construction_profile.applicability_threshold,
            ),
        }
        weakest = min(margins.values())
        quality[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "distinct_sample_count": candidate.distinct_sample_count,
            "member_count": len(candidate.member_experience_ids),
            "support_tier": (
                "preferred"
                if candidate.distinct_sample_count
                >= shortlist_profile.preferred_distinct_support
                else "minimum"
            ),
            "normalized_minimum_margins": margins,
            "weakest_normalized_minimum_margin": weakest,
            "mechanism_similarity_min": candidate.mechanism_similarity_min,
            "repair_similarity_min": candidate.repair_similarity_min,
            "applicability_similarity_min": candidate.applicability_similarity_min,
            "joint_similarity_min": candidate.joint_similarity_min,
            "joint_similarity_mean": candidate.joint_similarity_mean,
        }
    values = [item["weakest_normalized_minimum_margin"] for item in quality.values()]
    threshold = (
        float(np.quantile(np.asarray(values, dtype=np.float64), shortlist_profile.minimum_support_cohesion_quantile))
        if values
        else math.inf
    )
    return quality, threshold


def _quality_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(value["distinct_sample_count"]),
        -float(value["weakest_normalized_minimum_margin"]),
        -float(value["joint_similarity_min"]),
        -float(value["joint_similarity_mean"]),
        str(value["candidate_id"]),
    )


def build_redundancy_geometry(
    candidates: Sequence[V42LocalClusterCandidate],
    centroids: Mapping[str, Mapping[str, np.ndarray]],
    *,
    construction_profile: V42ConstructionProfile,
    shortlist_profile: V42ShortlistProfile,
) -> tuple[
    tuple[dict[str, Any], ...],
    dict[tuple[str, str], dict[str, float]],
]:
    ordered = tuple(sorted(item.candidate_id for item in candidates))
    all_pairs: dict[tuple[str, str], dict[str, float]] = {}
    redundancy_edges: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            similarities = {
                name: float(np.clip(np.dot(centroids[left][name], centroids[right][name]), -1.0, 1.0))
                for name in EMBEDDING_VIEW_NAMES
            }
            joint = (
                construction_profile.mechanism_weight * similarities["mechanism"]
                + construction_profile.repair_weight * similarities["repair"]
                + construction_profile.applicability_weight
                * similarities["applicability"]
            )
            all_pairs[(left, right)] = {**similarities, "joint": joint}
            if (
                similarities["mechanism"]
                < shortlist_profile.redundancy_mechanism_threshold
                or similarities["repair"]
                < shortlist_profile.redundancy_repair_threshold
                or similarities["applicability"]
                < shortlist_profile.redundancy_applicability_threshold
            ):
                continue
            edge = {
                "schema_version": REDUNDANCY_EDGE_SCHEMA,
                "edge_id": (
                    "v42-redundancy-"
                    f"{canonical_json_sha256([left, right])[:20]}"
                ),
                "left_candidate_id": left,
                "right_candidate_id": right,
                "mechanism_similarity": similarities["mechanism"],
                "repair_similarity": similarities["repair"],
                "applicability_similarity": similarities["applicability"],
                "joint_similarity": joint,
            }
            edge["edge_sha256"] = canonical_json_sha256(edge)
            redundancy_edges.append(edge)
    return tuple(redundancy_edges), all_pairs


def _pair_value(
    left: str,
    right: str,
    pairs: Mapping[tuple[str, str], Mapping[str, float]],
) -> Mapping[str, float]:
    if left == right:
        return {
            "mechanism": 1.0,
            "repair": 1.0,
            "applicability": 1.0,
            "joint": 1.0,
        }
    return pairs[tuple(sorted((left, right)))]


def select_synthesis_shortlist(
    candidates: Sequence[V42LocalClusterCandidate],
    quality: Mapping[str, Mapping[str, Any]],
    redundancy_edges: Sequence[Mapping[str, Any]],
    all_pairs: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    profile: V42ShortlistProfile,
    minimum_support_cohesion_threshold: float,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    candidate_ids = {item.candidate_id for item in candidates}
    adjacency: dict[str, dict[str, Mapping[str, Any]]] = {
        item: {} for item in candidate_ids
    }
    for edge in redundancy_edges:
        left = str(edge["left_candidate_id"])
        right = str(edge["right_candidate_id"])
        if left not in adjacency or right not in adjacency:
            raise ValueError("V4.2 redundancy edge references an unknown candidate")
        adjacency[left][right] = edge
        adjacency[right][left] = edge

    selected: list[str] = []
    decisions: dict[str, dict[str, Any]] = {}

    def conflict(candidate_id: str) -> tuple[str, Mapping[str, Any]] | None:
        values = [
            (kept, adjacency[candidate_id][kept])
            for kept in selected
            if kept in adjacency[candidate_id]
        ]
        if not values:
            return None
        return min(
            values,
            key=lambda item: (
                -float(item[1]["joint_similarity"]),
                selected.index(item[0]),
                item[0],
            ),
        )

    def reject_redundant(
        candidate_id: str,
        duplicate: tuple[str, Mapping[str, Any]],
    ) -> None:
        kept, edge = duplicate
        decisions[candidate_id] = {
            "decision": "rejected",
            "reason": "candidate_centroid_redundant",
            "selection_rank": None,
            "redundant_with_candidate_id": kept,
            "redundancy_edge_id": edge["edge_id"],
            "nearest_selected_joint_similarity": float(edge["joint_similarity"]),
        }

    preferred = sorted(
        (
            candidate_id
            for candidate_id in candidate_ids
            if int(quality[candidate_id]["distinct_sample_count"])
            >= profile.preferred_distinct_support
        ),
        key=lambda item: _quality_sort_key(quality[item]),
    )
    for candidate_id in preferred:
        duplicate = conflict(candidate_id)
        if duplicate is not None:
            reject_redundant(candidate_id, duplicate)
        elif len(selected) >= profile.max_synthesis_candidates:
            decisions[candidate_id] = {
                "decision": "rejected",
                "reason": "synthesis_candidate_budget_exceeded",
                "selection_rank": None,
                "redundant_with_candidate_id": None,
                "redundancy_edge_id": None,
                "nearest_selected_joint_similarity": None,
            }
        else:
            selected.append(candidate_id)
            decisions[candidate_id] = {
                "decision": "selected",
                "reason": "preferred_support_nonredundant",
                "selection_rank": len(selected),
                "redundant_with_candidate_id": None,
                "redundancy_edge_id": None,
                "nearest_selected_joint_similarity": (
                    max(
                        float(_pair_value(candidate_id, kept, all_pairs)["joint"])
                        for kept in selected[:-1]
                    )
                    if len(selected) > 1
                    else None
                ),
            }

    minimum_support = {
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in decisions
    }
    eligible: set[str] = set()
    for candidate_id in sorted(minimum_support):
        if (
            float(quality[candidate_id]["weakest_normalized_minimum_margin"])
            < minimum_support_cohesion_threshold
        ):
            decisions[candidate_id] = {
                "decision": "rejected",
                "reason": "minimum_support_below_cohesion_quantile",
                "selection_rank": None,
                "redundant_with_candidate_id": None,
                "redundancy_edge_id": None,
                "nearest_selected_joint_similarity": None,
            }
        else:
            eligible.add(candidate_id)

    while eligible and len(selected) < profile.max_synthesis_candidates:
        for candidate_id in sorted(tuple(eligible)):
            duplicate = conflict(candidate_id)
            if duplicate is not None:
                reject_redundant(candidate_id, duplicate)
                eligible.remove(candidate_id)
        if not eligible:
            break

        def diversity_key(candidate_id: str) -> tuple[Any, ...]:
            maximum_similarity = (
                max(
                    float(_pair_value(candidate_id, kept, all_pairs)["joint"])
                    for kept in selected
                )
                if selected
                else -1.0
            )
            return (
                maximum_similarity,
                *_quality_sort_key(quality[candidate_id]),
            )

        chosen = min(eligible, key=diversity_key)
        nearest_similarity = (
            max(
                float(_pair_value(chosen, kept, all_pairs)["joint"])
                for kept in selected
            )
            if selected
            else None
        )
        selected.append(chosen)
        decisions[chosen] = {
            "decision": "selected",
            "reason": "minimum_support_high_cohesion_diversity",
            "selection_rank": len(selected),
            "redundant_with_candidate_id": None,
            "redundancy_edge_id": None,
            "nearest_selected_joint_similarity": nearest_similarity,
        }
        eligible.remove(chosen)

    for candidate_id in sorted(eligible):
        duplicate = conflict(candidate_id)
        if duplicate is not None:
            reject_redundant(candidate_id, duplicate)
        else:
            decisions[candidate_id] = {
                "decision": "rejected",
                "reason": "synthesis_candidate_budget_exceeded",
                "selection_rank": None,
                "redundant_with_candidate_id": None,
                "redundancy_edge_id": None,
                "nearest_selected_joint_similarity": (
                    max(
                        float(_pair_value(candidate_id, kept, all_pairs)["joint"])
                        for kept in selected
                    )
                    if selected
                    else None
                ),
            }

    if set(decisions) != candidate_ids:
        raise ValueError("V4.2 shortlist decisions do not cover every candidate")
    if len(selected) != len(set(selected)) or len(selected) > profile.max_synthesis_candidates:
        raise ValueError("V4.2 shortlist selection is duplicated or over budget")
    for candidate_id in selected:
        if any(other in adjacency[candidate_id] for other in selected if other != candidate_id):
            raise ValueError("V4.2 selected shortlist contains a redundant pair")
    return tuple(selected), decisions


def _semantic_packet(
    candidate: V42LocalClusterCandidate,
    packet: Mapping[str, Any],
    quality: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_evidence = []
    representative_provenance = []
    for index, atom in enumerate(packet["representatives"], start=1):
        evidence_id = f"evidence-{index}"
        semantic_evidence.append(
            {
                "evidence_id": evidence_id,
                **{field: atom[field] for field in V4_2_SYNTHESIS_SEMANTIC_FIELDS},
            }
        )
        representative_provenance.append(
            {
                "evidence_id": evidence_id,
                "experience_id": atom["experience_id"],
                "sample_id": atom["sample_id"],
                "source_experience_type": atom["source_experience_type"],
                "source_signature_sha256": atom["source_signature_sha256"],
            }
        )
    payload = {
        "schema_version": SELECTED_CANDIDATE_SCHEMA,
        "selection_rank": decision["selection_rank"],
        "candidate": candidate.to_dict(),
        "quality": dict(quality),
        "semantic_evidence": semantic_evidence,
        "representative_provenance": representative_provenance,
        "source_review_packet_sha256": packet["packet_sha256"],
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def build_outputs(
    *,
    candidates: Sequence[V42LocalClusterCandidate],
    review_packets: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
    redundancy_edges: Sequence[Mapping[str, Any]],
    minimum_support_cohesion_threshold: float,
    profile: V42ShortlistProfile,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    candidate_by_id = {item.candidate_id: item for item in candidates}
    quality_rows = []
    for candidate_id in sorted(candidate_by_id):
        quality_rows.append(
            {
                **dict(quality[candidate_id]),
                **dict(decisions[candidate_id]),
            }
        )
    decision_counts = Counter(item["reason"] for item in decisions.values())
    quality_report = {
        "schema_version": CANDIDATE_QUALITY_REPORT_SCHEMA,
        "construction_version": "v4.2",
        "status": "high_quality_shortlist_complete",
        "source_candidate_count": len(candidates),
        "selected_candidate_count": len(selected_ids),
        "rejected_candidate_count": len(candidates) - len(selected_ids),
        "minimum_support_cohesion_threshold": (
            minimum_support_cohesion_threshold
        ),
        "minimum_support_cohesion_quantile": (
            profile.minimum_support_cohesion_quantile
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "redundancy_edge_count": len(redundancy_edges),
        "selected_candidate_ids": list(selected_ids),
        "candidates": quality_rows,
        "profile_sha256": profile.profile_sha256,
    }
    quality_report["report_sha256"] = canonical_json_sha256(quality_report)

    selected_records = tuple(
        _semantic_packet(
            candidate_by_id[candidate_id],
            review_packets[candidate_id],
            quality[candidate_id],
            decisions[candidate_id],
        )
        for candidate_id in selected_ids
    )
    rejected_records = []
    for candidate_id in sorted(candidate_by_id):
        decision = decisions[candidate_id]
        if decision["decision"] == "selected":
            continue
        record = {
            "schema_version": REJECTED_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "membership_sha256": candidate_by_id[candidate_id].membership_sha256,
            "quality": dict(quality[candidate_id]),
            "decision": dict(decision),
        }
        record["record_sha256"] = canonical_json_sha256(record)
        rejected_records.append(record)
    if len(selected_records) + len(rejected_records) != len(candidates):
        raise ValueError("V4.2 shortlist outputs lost candidate coverage")
    return quality_report, selected_records, tuple(rejected_records)


def _profile_record(
    profile: V42ShortlistProfile,
    *,
    source_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SHORTLIST_PROFILE_RECORD_SCHEMA,
        "construction_version": "v4.2",
        "stage": "high_quality_synthesis_shortlist",
        "external_api_calls": 0,
        "api_key_read": False,
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "source_local_construction": dict(source_info),
        "implementation_sha256": _implementation_state(),
    }


def _write_or_validate_profile(
    path: Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if path.is_file():
        if _load_json(path) != expected:
            raise ValueError("V4.2 shortlist profile drifted")
        if not resume:
            raise ValueError("V4.2 shortlist output exists; pass --resume")
        return
    _write_json(path, expected)


def build_manifest(
    *,
    profile: V42ShortlistProfile,
    source_info: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    rejected_records: Sequence[Mapping[str, Any]],
    redundancy_edges: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    selected_ids = [str(item["candidate"]["candidate_id"]) for item in selected_records]
    manifest = {
        "schema_version": V4_2_SHORTLIST_MANIFEST_SCHEMA,
        "construction_version": "v4.2",
        "status": "synthesis_shortlist_complete_api_not_started",
        "qualified_for_online_use": False,
        "external_api_calls_made": 0,
        "api_key_read": False,
        "automatic_paid_stage_transition": False,
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "source_local_construction": dict(source_info),
        "source_candidate_count": len(selected_records) + len(rejected_records),
        "selected_candidate_count": len(selected_records),
        "rejected_candidate_count": len(rejected_records),
        "redundancy_edge_count": len(redundancy_edges),
        "selected_candidate_ids": selected_ids,
        "target_runtime_bank_cap": profile.target_runtime_bank_cap,
        "artifacts": {
            name: file_sha256(output_dir / name)
            for name in (
                "construction_profile.json",
                "candidate_quality_report.json",
                "candidate_redundancy_edges.jsonl",
                "selected_synthesis_candidates.jsonl",
                "rejected_or_redundant_candidates.jsonl",
            )
        },
        "implementation_sha256": _implementation_state(),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def build_preflight(
    *,
    profile: V42ShortlistProfile,
    selected_records: Sequence[Mapping[str, Any]],
    quality_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_characters = sum(
        len(
            json.dumps(
                item["semantic_evidence"], ensure_ascii=False, sort_keys=True
            )
        )
        for item in selected_records
    )
    selected_count = len(selected_records)
    report = {
        "schema_version": V4_2_SHORTLIST_PREFLIGHT_SCHEMA,
        "construction_version": "v4.2",
        "status": "synthesis_shortlist_complete_api_not_started",
        "qualified_for_online_use": False,
        "external_api_calls_made": 0,
        "api_key_read": False,
        "automatic_paid_stage_transition": False,
        "source_candidate_count": quality_report["source_candidate_count"],
        "selected_synthesis_candidate_count": selected_count,
        "rejected_candidate_count": quality_report["rejected_candidate_count"],
        "decision_counts": quality_report["decision_counts"],
        "minimum_support_cohesion_threshold": quality_report[
            "minimum_support_cohesion_threshold"
        ],
        "redundancy_edge_count": quality_report["redundancy_edge_count"],
        "max_synthesis_candidates": profile.max_synthesis_candidates,
        "target_runtime_bank_cap": profile.target_runtime_bank_cap,
        "synthesis_batch_size": profile.synthesis_batch_size,
        "review_batch_size": profile.review_batch_size,
        "planned_initial_synthesis_requests": math.ceil(
            selected_count / profile.synthesis_batch_size
        ),
        "maximum_followup_review_requests": math.ceil(
            selected_count / profile.review_batch_size
        ),
        "maximum_total_paid_requests": (
            math.ceil(selected_count / profile.synthesis_batch_size)
            + math.ceil(selected_count / profile.review_batch_size)
        ),
        "semantic_evidence_characters": semantic_characters,
        "estimated_semantic_evidence_tokens_at_three_chars_per_token": (
            math.ceil(semantic_characters / 3)
        ),
        "within_synthesis_candidate_guardrail": (
            0 < selected_count <= profile.max_synthesis_candidates
        ),
        "synthesis_blocked_reason": (
            None
            if 0 < selected_count <= profile.max_synthesis_candidates
            else "the local high-quality shortlist is empty or over budget"
        ),
        "profile_sha256": profile.profile_sha256,
        "quality_report_sha256": quality_report["report_sha256"],
        "shortlist_manifest_sha256": manifest["manifest_sha256"],
        "note": (
            "Request counts are upper bounds for a future explicit paid stage. "
            "Rejected local candidates and unsupported atoms are never sent."
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def validate_completed_output(
    output_dir: Path,
    *,
    expected_profile_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = output_dir / "synthesis_shortlist_manifest.json"
    preflight_path = output_dir / "api_preflight_report.json"
    if not manifest_path.is_file() or not preflight_path.is_file():
        return None
    if _load_json(output_dir / "construction_profile.json") != expected_profile_record:
        raise ValueError("Completed V4.2 shortlist profile drifted")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != V4_2_SHORTLIST_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V4.2 shortlist-manifest schema")
    if manifest.get("manifest_sha256") != _logical_hash(
        manifest, "manifest_sha256"
    ):
        raise ValueError("V4.2 shortlist-manifest hash mismatch")
    if (
        manifest.get("profile_sha256")
        != expected_profile_record["profile_sha256"]
        or manifest.get("source_local_construction")
        != expected_profile_record["source_local_construction"]
        or manifest.get("implementation_sha256")
        != expected_profile_record["implementation_sha256"]
    ):
        raise ValueError("V4.2 shortlist manifest provenance drifted")
    if manifest.get("source_candidate_count") != expected_profile_record[
        "source_local_construction"
    ]["candidate_count"]:
        raise ValueError("V4.2 shortlist source-candidate count drifted")
    for name, expected_sha256 in manifest.get("artifacts", {}).items():
        path = output_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f"V4.2 shortlist artifact hash mismatch: {name}")
    preflight = _load_json(preflight_path)
    if preflight.get("schema_version") != V4_2_SHORTLIST_PREFLIGHT_SCHEMA:
        raise ValueError("Unexpected V4.2 shortlist-preflight schema")
    if preflight.get("report_sha256") != _logical_hash(preflight, "report_sha256"):
        raise ValueError("V4.2 shortlist-preflight hash mismatch")
    if preflight.get("shortlist_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("V4.2 shortlist preflight/manifest binding mismatch")
    quality_report = _load_json(output_dir / "candidate_quality_report.json")
    if quality_report.get("report_sha256") != _logical_hash(
        quality_report, "report_sha256"
    ):
        raise ValueError("V4.2 candidate-quality report hash mismatch")
    if preflight.get("quality_report_sha256") != quality_report["report_sha256"]:
        raise ValueError("V4.2 shortlist preflight/quality binding mismatch")

    selected_ids: list[str] = []
    selected_ranks: list[int] = []
    for record in _iter_jsonl(output_dir / "selected_synthesis_candidates.jsonl"):
        if record.get("schema_version") != SELECTED_CANDIDATE_SCHEMA:
            raise ValueError("Unexpected V4.2 selected-candidate schema")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError("V4.2 selected-candidate hash mismatch")
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("V4.2 selected record is missing its candidate")
        validate_v4_2_cluster_payload(candidate)
        semantic_evidence = record.get("semantic_evidence")
        provenance = record.get("representative_provenance")
        if (
            not isinstance(semantic_evidence, list)
            or not isinstance(provenance, list)
            or len(semantic_evidence) != 5
            or len(provenance) != 5
            or [item.get("evidence_id") for item in semantic_evidence]
            != [item.get("evidence_id") for item in provenance]
        ):
            raise ValueError("V4.2 selected candidate has invalid compact evidence")
        selected_ids.append(str(candidate["candidate_id"]))
        selected_ranks.append(int(record["selection_rank"]))
    if (
        selected_ids != manifest.get("selected_candidate_ids")
        or selected_ranks != list(range(1, len(selected_ids) + 1))
        or len(selected_ids) != manifest.get("selected_candidate_count")
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError("V4.2 selected-candidate order or coverage drifted")

    rejected_ids: set[str] = set()
    for record in _iter_jsonl(
        output_dir / "rejected_or_redundant_candidates.jsonl"
    ):
        if record.get("schema_version") != REJECTED_CANDIDATE_SCHEMA:
            raise ValueError("Unexpected V4.2 rejected-candidate schema")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError("V4.2 rejected-candidate hash mismatch")
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id or candidate_id in rejected_ids:
            raise ValueError("V4.2 rejected candidates contain a duplicate ID")
        rejected_ids.add(candidate_id)
    if (
        len(rejected_ids) != manifest.get("rejected_candidate_count")
        or set(selected_ids) & rejected_ids
        or len(selected_ids) + len(rejected_ids)
        != manifest.get("source_candidate_count")
    ):
        raise ValueError("V4.2 shortlist terminal candidate coverage drifted")

    redundancy_count = 0
    for edge in _iter_jsonl(output_dir / "candidate_redundancy_edges.jsonl"):
        if edge.get("schema_version") != REDUNDANCY_EDGE_SCHEMA:
            raise ValueError("Unexpected V4.2 redundancy-edge schema")
        if edge.get("edge_sha256") != _logical_hash(edge, "edge_sha256"):
            raise ValueError("V4.2 redundancy-edge hash mismatch")
        redundancy_count += 1
    if redundancy_count != manifest.get("redundancy_edge_count"):
        raise ValueError("V4.2 redundancy-edge count drifted")
    if (
        preflight.get("selected_synthesis_candidate_count") != len(selected_ids)
        or preflight.get("rejected_candidate_count") != len(rejected_ids)
        or preflight.get("source_candidate_count")
        != manifest.get("source_candidate_count")
        or preflight.get("profile_sha256") != manifest.get("profile_sha256")
    ):
        raise ValueError("V4.2 shortlist preflight counts or profile drifted")
    if (
        manifest.get("external_api_calls_made") != 0
        or manifest.get("api_key_read") is not False
        or manifest.get("automatic_paid_stage_transition") is not False
        or manifest.get("qualified_for_online_use") is not False
        or preflight.get("external_api_calls_made") != 0
        or preflight.get("api_key_read") is not False
        or preflight.get("automatic_paid_stage_transition") is not False
        or preflight.get("qualified_for_online_use") is not False
    ):
        raise ValueError("V4.2 shortlist has an invalid qualification state")
    return preflight


def main() -> None:
    args = parse_args()
    shortlist_profile = _profile(args)
    local_dir = args.local_construction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == local_dir:
        raise ValueError("V4.2 shortlist output must differ from local construction")
    (
        construction_profile,
        _plan,
        atoms,
        candidates,
        review_packets,
        embeddings,
        source_info,
    ) = load_authenticated_local_construction(local_dir)
    expected_profile_record = _profile_record(
        shortlist_profile, source_info=source_info
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed = validate_completed_output(
            output_dir, expected_profile_record=expected_profile_record
        )
        if completed is not None:
            print(
                f"[v4.2-shortlist] reused selected="
                f"{completed['selected_synthesis_candidate_count']} "
                f"api_calls=0",
                flush=True,
            )
            print(
                f"[v4.2-shortlist] preflight="
                f"{output_dir / 'api_preflight_report.json'}",
                flush=True,
            )
            return
    _write_or_validate_profile(
        output_dir / "construction_profile.json",
        expected_profile_record,
        resume=args.resume,
    )
    centroids = candidate_centroids(candidates, atoms, embeddings)
    quality, cohesion_threshold = build_candidate_quality(
        candidates,
        construction_profile=construction_profile,
        shortlist_profile=shortlist_profile,
    )
    redundancy_edges, all_pairs = build_redundancy_geometry(
        candidates,
        centroids,
        construction_profile=construction_profile,
        shortlist_profile=shortlist_profile,
    )
    selected_ids, decisions = select_synthesis_shortlist(
        candidates,
        quality,
        redundancy_edges,
        all_pairs,
        profile=shortlist_profile,
        minimum_support_cohesion_threshold=cohesion_threshold,
    )
    quality_report, selected_records, rejected_records = build_outputs(
        candidates=candidates,
        review_packets=review_packets,
        quality=quality,
        decisions=decisions,
        selected_ids=selected_ids,
        redundancy_edges=redundancy_edges,
        minimum_support_cohesion_threshold=cohesion_threshold,
        profile=shortlist_profile,
    )
    _write_json(output_dir / "candidate_quality_report.json", quality_report)
    _write_jsonl(
        output_dir / "candidate_redundancy_edges.jsonl", redundancy_edges
    )
    _write_jsonl(
        output_dir / "selected_synthesis_candidates.jsonl", selected_records
    )
    _write_jsonl(
        output_dir / "rejected_or_redundant_candidates.jsonl",
        rejected_records,
    )
    manifest = build_manifest(
        profile=shortlist_profile,
        source_info=source_info,
        selected_records=selected_records,
        rejected_records=rejected_records,
        redundancy_edges=redundancy_edges,
        output_dir=output_dir,
    )
    _write_json(output_dir / "synthesis_shortlist_manifest.json", manifest)
    preflight = build_preflight(
        profile=shortlist_profile,
        selected_records=selected_records,
        quality_report=quality_report,
        manifest=manifest,
    )
    _write_json(output_dir / "api_preflight_report.json", preflight)
    print(
        f"[v4.2-shortlist] complete source={len(candidates)} "
        f"redundancy_edges={len(redundancy_edges)} "
        f"selected={len(selected_records)} rejected={len(rejected_records)} "
        f"api_calls=0",
        flush=True,
    )
    print(
        f"[v4.2-shortlist] preflight="
        f"{output_dir / 'api_preflight_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.2-shortlist] error: {exc}", file=sys.stderr)
        raise
