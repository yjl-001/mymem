#!/usr/bin/env python3
"""Build an authenticated V4.2 repair-cluster plan without external APIs.

This command deliberately stops before semantic synthesis.  It reuses the
completed, authenticated V4 repair signatures, deterministically quarantines
verified format-only failures, embeds three process views with one pinned local
BGE checkpoint, and forms mutual-kNN complete-link groups.  Only groups with at
least five distinct construction samples become synthesis candidates.

The script never reads an API key, never constructs a teacher client, and never
continues into a paid stage.  Its final preflight report states how many future
cluster-level synthesis requests would be needed under the chosen batch size.
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
from memgen.experience.v4_bank import V4RepairSignature
from memgen.experience.v4_2_bank import (
    V4_2_DEFAULT_APPLICABILITY_THRESHOLD,
    V4_2_DEFAULT_APPLICABILITY_WEIGHT,
    V4_2_DEFAULT_MAX_API_CANDIDATES,
    V4_2_DEFAULT_MECHANISM_THRESHOLD,
    V4_2_DEFAULT_MECHANISM_WEIGHT,
    V4_2_DEFAULT_NEIGHBOR_COUNT,
    V4_2_DEFAULT_REPAIR_THRESHOLD,
    V4_2_DEFAULT_REPAIR_WEIGHT,
    V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE,
    V4_2_EMBEDDING_MANIFEST_SCHEMA,
    V4_2_LOCAL_CLUSTER_PLAN_SCHEMA,
    V4_2_POSITIVE_EDGE_SCHEMA,
    V4_2_PREFLIGHT_REPORT_SCHEMA,
    V4_2_REVIEW_PACKET_SCHEMA,
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
)
from scripts.build_v4_1_repair_bank import load_authenticated_signatures
from scripts.build_v4_repair_bank import (
    _validate_split_manifest,
    load_v4_experiences,
)


LOCAL_ATOM_RECORD_SCHEMA = "memgen-v4.2-local-repair-atom-record-v1"
LOCAL_CLUSTER_RECORD_SCHEMA = "memgen-v4.2-local-cluster-candidate-record-v1"
POSITIVE_EDGE_ARTIFACT_SCHEMA = "memgen-v4.2-positive-edge-artifact-v1"
EMBEDDING_VIEW_NAMES = ("mechanism", "repair", "applicability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-signatures", type=Path, required=True)
    parser.add_argument("--source-construction-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--neighbor-count", type=int, default=V4_2_DEFAULT_NEIGHBOR_COUNT
    )
    parser.add_argument(
        "--mechanism-threshold",
        type=float,
        default=V4_2_DEFAULT_MECHANISM_THRESHOLD,
    )
    parser.add_argument(
        "--repair-threshold", type=float, default=V4_2_DEFAULT_REPAIR_THRESHOLD
    )
    parser.add_argument(
        "--applicability-threshold",
        type=float,
        default=V4_2_DEFAULT_APPLICABILITY_THRESHOLD,
    )
    parser.add_argument(
        "--mechanism-weight", type=float, default=V4_2_DEFAULT_MECHANISM_WEIGHT
    )
    parser.add_argument(
        "--repair-weight", type=float, default=V4_2_DEFAULT_REPAIR_WEIGHT
    )
    parser.add_argument(
        "--applicability-weight",
        type=float,
        default=V4_2_DEFAULT_APPLICABILITY_WEIGHT,
    )
    parser.add_argument(
        "--synthesis-batch-size",
        type=int,
        default=V4_2_DEFAULT_SYNTHESIS_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-api-candidates",
        type=int,
        default=V4_2_DEFAULT_MAX_API_CANDIDATES,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only fully authenticated local embedding artifacts.",
    )
    return parser.parse_args()


def _profile(args: argparse.Namespace) -> V42ConstructionProfile:
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    return V42ConstructionProfile(
        neighbor_count=args.neighbor_count,
        mechanism_threshold=args.mechanism_threshold,
        repair_threshold=args.repair_threshold,
        applicability_threshold=args.applicability_threshold,
        mechanism_weight=args.mechanism_weight,
        repair_weight=args.repair_weight,
        applicability_weight=args.applicability_weight,
        synthesis_batch_size=args.synthesis_batch_size,
        max_api_candidates=args.max_api_candidates,
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


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _implementation_state() -> dict[str, str]:
    paths = (
        "memgen/experience/v4_bank.py",
        "memgen/experience/v4_2_bank.py",
        "scripts/build_v4_1_repair_bank.py",
        "scripts/build_v4_2_local_clusters.py",
    )
    return {relative: file_sha256(PROJECT_ROOT / relative) for relative in paths}


def _write_or_validate_profile(
    path: Path,
    *,
    profile: V42ConstructionProfile,
    source_signature_info: Mapping[str, Any],
    resume: bool,
) -> None:
    expected = {
        "schema_version": profile.schema_version,
        "construction_version": "v4.2",
        "stage": "local_cluster_only",
        "external_api_calls": 0,
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "source_signatures": dict(source_signature_info),
        "implementation_sha256": _implementation_state(),
    }
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise ValueError("V4.2 local construction profile drifted")
        if not resume:
            raise ValueError("V4.2 output exists; pass --resume to authenticate it")
        return
    _write_json(path, expected)


def partition_signatures(
    signatures: Sequence[V4RepairSignature],
) -> tuple[
    tuple[V42LocalRepairAtom, ...],
    dict[str, tuple[str, ...]],
]:
    """Apply only authenticated, deterministic role boundaries."""

    atoms: list[V42LocalRepairAtom] = []
    nonapplicable: list[str] = []
    answer_serialization: list[str] = []
    for signature in signatures:
        if not signature.applicable:
            nonapplicable.append(signature.experience_id)
        elif signature.experience_type == "format_compliance":
            answer_serialization.append(signature.experience_id)
        else:
            atoms.append(V42LocalRepairAtom.from_signature(signature))
    atoms.sort(key=lambda item: item.experience_id)
    archive = {
        "source_nonapplicable_experience_ids": tuple(sorted(nonapplicable)),
        "answer_serialization_experience_ids": tuple(
            sorted(answer_serialization)
        ),
    }
    covered = {
        item.experience_id for item in atoms
    } | set(nonapplicable) | set(answer_serialization)
    if covered != {item.experience_id for item in signatures}:
        raise ValueError("V4.2 signature partition lost source coverage")
    if len(atoms) + len(nonapplicable) + len(answer_serialization) != len(signatures):
        raise ValueError("V4.2 signature partition overlaps")
    return tuple(atoms), archive


def _view_texts(
    atoms: Sequence[V42LocalRepairAtom],
) -> dict[str, list[str]]:
    return {
        "mechanism": [item.mechanism_text for item in atoms],
        "repair": [item.repair_text for item in atoms],
        "applicability": [item.applicability_text for item in atoms],
    }


def embed_view_texts(
    atoms: Sequence[V42LocalRepairAtom],
    *,
    profile: V42ConstructionProfile,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Encode all three views in one local frozen-model pass."""

    if not atoms:
        raise ValueError("V4.2 has no local reasoning atoms to embed")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on GPU server
        raise RuntimeError("V4.2 embedding requires torch and transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        profile.embedding_model,
        revision=profile.embedding_revision,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        profile.embedding_model,
        revision=profile.embedding_revision,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    by_view = _view_texts(atoms)
    flat_texts = [text for name in EMBEDDING_VIEW_NAMES for text in by_view[name]]
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat_texts), batch_size):
            encoded = tokenizer(
                flat_texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            rows.append(hidden.cpu().numpy().astype(np.float32, copy=False))
    flat = np.concatenate(rows, axis=0)
    expected_rows = len(atoms) * len(EMBEDDING_VIEW_NAMES)
    if flat.ndim != 2 or flat.shape[0] != expected_rows:
        raise RuntimeError("V4.2 embedding output shape mismatch")
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(EMBEDDING_VIEW_NAMES):
        value = flat[index * len(atoms) : (index + 1) * len(atoms)]
        norms = np.linalg.norm(value, axis=1)
        if not np.isfinite(value).all() or not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(f"V4.2 {name} embeddings are not finite unit vectors")
        result[name] = value
    return result


def load_or_build_embeddings(
    atoms: Sequence[V42LocalRepairAtom],
    *,
    profile: V42ConstructionProfile,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    manifest_path = output_dir / "multiview_embeddings_manifest.json"
    tensor_paths = {
        name: output_dir / f"{name}_embeddings.npy" for name in EMBEDDING_VIEW_NAMES
    }
    atom_order = [item.atom_id for item in atoms]
    texts = _view_texts(atoms)
    expected_text_hashes = {
        name: canonical_json_sha256(texts[name]) for name in EMBEDDING_VIEW_NAMES
    }
    artifact_exists = manifest_path.exists() or any(
        path.exists() for path in tensor_paths.values()
    )
    if artifact_exists and not args.resume:
        raise ValueError("Refusing to overwrite V4.2 embedding artifacts without --resume")
    if artifact_exists:
        if not manifest_path.is_file() or not all(
            path.is_file() for path in tensor_paths.values()
        ):
            raise ValueError("V4.2 embedding artifact set is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logical = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if (
            manifest.get("schema_version") != V4_2_EMBEDDING_MANIFEST_SCHEMA
            or manifest.get("manifest_sha256") != canonical_json_sha256(logical)
            or manifest.get("model") != profile.embedding_model
            or manifest.get("revision") != profile.embedding_revision
            or manifest.get("atom_order_sha256")
            != canonical_json_sha256(atom_order)
            or manifest.get("view_text_sha256") != expected_text_hashes
        ):
            raise ValueError("V4.2 embedding manifest authentication failed")
        result: dict[str, np.ndarray] = {}
        for name, path in tensor_paths.items():
            metadata = manifest.get("views", {}).get(name, {})
            if metadata.get("tensor_sha256") != file_sha256(path):
                raise ValueError(f"V4.2 {name} embedding hash mismatch")
            value = np.load(path, allow_pickle=False)
            if value.dtype != np.float32 or list(value.shape) != metadata.get("shape"):
                raise ValueError(f"V4.2 {name} embedding shape or dtype drifted")
            norms = np.linalg.norm(value, axis=1)
            if value.shape[0] != len(atoms) or not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError(f"V4.2 {name} embeddings are invalid")
            result[name] = value
        print(
            f"[v4.2-local] embeddings reused atoms={len(atoms)} "
            f"model={profile.embedding_model}",
            flush=True,
        )
        return result

    result = embed_view_texts(
        atoms,
        profile=profile,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    views: dict[str, Any] = {}
    for name, value in result.items():
        path = tensor_paths[name]
        _write_npy(path, value)
        views[name] = {
            "path": path.name,
            "shape": list(value.shape),
            "dtype": "float32",
            "normalization": "l2",
            "tensor_sha256": file_sha256(path),
        }
    manifest = {
        "schema_version": V4_2_EMBEDDING_MANIFEST_SCHEMA,
        "model": profile.embedding_model,
        "revision": profile.embedding_revision,
        "pooling": "cls",
        "max_length": 512,
        "atom_order_sha256": canonical_json_sha256(atom_order),
        "view_text_sha256": expected_text_hashes,
        "views": views,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    print(
        f"[v4.2-local] embeddings complete atoms={len(atoms)} "
        f"model={profile.embedding_model}",
        flush=True,
    )
    return result


def _validate_embedding_bundle(
    atoms: Sequence[V42LocalRepairAtom],
    embeddings: Mapping[str, np.ndarray],
) -> None:
    expected_rows = len(atoms)
    dimensions: set[int] = set()
    for name in EMBEDDING_VIEW_NAMES:
        value = embeddings.get(name)
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError(f"V4.2 {name} embedding matrix is missing")
        if value.shape[0] != expected_rows or value.dtype != np.float32:
            raise ValueError(f"V4.2 {name} embedding matrix shape or dtype is invalid")
        norms = np.linalg.norm(value, axis=1)
        if not np.isfinite(value).all() or not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"V4.2 {name} embeddings must be finite unit vectors")
        dimensions.add(int(value.shape[1]))
    if len(dimensions) != 1:
        raise ValueError("V4.2 embedding views have different dimensions")


def build_joint_embeddings(
    embeddings: Mapping[str, np.ndarray],
    *,
    profile: V42ConstructionProfile,
) -> np.ndarray:
    value = np.concatenate(
        (
            embeddings["mechanism"] * math.sqrt(profile.mechanism_weight),
            embeddings["repair"] * math.sqrt(profile.repair_weight),
            embeddings["applicability"]
            * math.sqrt(profile.applicability_weight),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norms <= 0):
        raise ValueError("V4.2 joint embeddings are invalid")
    return (value / norms).astype(np.float32, copy=False)


def _stable_top_k(
    scores: np.ndarray,
    *,
    self_index: int,
    identifiers: Sequence[str],
    count: int,
) -> tuple[int, ...]:
    """Top-k with deterministic identifier tie-breaking at the cutoff."""

    if scores.ndim != 1 or scores.shape[0] != len(identifiers):
        raise ValueError("V4.2 top-k score shape mismatch")
    available = len(identifiers) - 1
    if available <= 0:
        return ()
    count = min(count, available)
    values = scores.copy()
    values[self_index] = -np.inf
    selected = np.argpartition(-values, count - 1)[:count]
    cutoff = float(np.min(values[selected]))
    higher = [
        index
        for index, value in enumerate(values)
        if index != self_index and float(value) > cutoff
    ]
    tied = sorted(
        (
            index
            for index, value in enumerate(values)
            if index != self_index and float(value) == cutoff
        ),
        key=lambda index: identifiers[index],
    )
    chosen = higher + tied[: count - len(higher)]
    return tuple(
        sorted(
            chosen,
            key=lambda index: (-float(values[index]), identifiers[index]),
        )
    )


def build_multiview_positive_edges(
    atoms: Sequence[V42LocalRepairAtom],
    embeddings: Mapping[str, np.ndarray],
    *,
    profile: V42ConstructionProfile,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], np.ndarray]:
    """Build thresholded mutual-kNN edges; local similarity is merge authority."""

    _validate_embedding_bundle(atoms, embeddings)
    if len(atoms) < 2:
        return (), {
            "mutual_knn_pair_count": 0,
            "positive_edge_count": 0,
            "below_mechanism_threshold": 0,
            "below_repair_threshold": 0,
            "below_applicability_threshold": 0,
            "nodes_without_positive_edges": len(atoms),
        }, build_joint_embeddings(embeddings, profile=profile)
    identifiers = [item.experience_id for item in atoms]
    similarities = {
        name: np.clip(value @ value.T, -1.0, 1.0).astype(np.float32, copy=False)
        for name, value in embeddings.items()
    }
    joint_similarity = (
        profile.mechanism_weight * similarities["mechanism"]
        + profile.repair_weight * similarities["repair"]
        + profile.applicability_weight * similarities["applicability"]
    ).astype(np.float32, copy=False)
    top = tuple(
        frozenset(
            _stable_top_k(
                joint_similarity[index],
                self_index=index,
                identifiers=identifiers,
                count=profile.neighbor_count,
            )
        )
        for index in range(len(atoms))
    )
    counters = Counter()
    edges: list[dict[str, Any]] = []
    degree = [0 for _item in atoms]
    for left in range(len(atoms)):
        for right in sorted(top[left]):
            if right <= left or left not in top[right]:
                continue
            counters["mutual_knn_pair_count"] += 1
            mechanism = float(similarities["mechanism"][left, right])
            repair = float(similarities["repair"][left, right])
            applicability = float(similarities["applicability"][left, right])
            if mechanism < profile.mechanism_threshold:
                counters["below_mechanism_threshold"] += 1
            if repair < profile.repair_threshold:
                counters["below_repair_threshold"] += 1
            if applicability < profile.applicability_threshold:
                counters["below_applicability_threshold"] += 1
            if (
                mechanism < profile.mechanism_threshold
                or repair < profile.repair_threshold
                or applicability < profile.applicability_threshold
            ):
                continue
            left_id = identifiers[left]
            right_id = identifiers[right]
            pair_id = f"v42-edge-{canonical_json_sha256([left_id, right_id])[:20]}"
            edge = {
                "schema_version": V4_2_POSITIVE_EDGE_SCHEMA,
                "pair_id": pair_id,
                "left_experience_id": left_id,
                "right_experience_id": right_id,
                "mechanism_similarity": mechanism,
                "repair_similarity": repair,
                "applicability_similarity": applicability,
                "joint_similarity": float(joint_similarity[left, right]),
                "retrieval": "mutual_joint_top_k",
            }
            edge["edge_sha256"] = canonical_json_sha256(edge)
            edges.append(edge)
            degree[left] += 1
            degree[right] += 1
    edges.sort(key=lambda item: (item["left_experience_id"], item["right_experience_id"]))
    counters["positive_edge_count"] = len(edges)
    counters["nodes_without_positive_edges"] = sum(value == 0 for value in degree)
    positive_joint = [float(item["joint_similarity"]) for item in edges]
    diagnostics = {
        key: int(counters.get(key, 0))
        for key in (
            "mutual_knn_pair_count",
            "positive_edge_count",
            "below_mechanism_threshold",
            "below_repair_threshold",
            "below_applicability_threshold",
            "nodes_without_positive_edges",
        )
    }
    diagnostics["positive_joint_similarity"] = _distribution(positive_joint)
    return tuple(edges), diagnostics, build_joint_embeddings(
        embeddings, profile=profile
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def form_complete_link_groups(
    atoms: Sequence[V42LocalRepairAtom],
    edges: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    """Greedily partition atoms into deterministic positive-edge cliques."""

    identifiers = {item.experience_id for item in atoms}
    adjacency = {identifier: set() for identifier in identifiers}
    joint_scores: dict[tuple[str, str], float] = {}
    for edge in edges:
        left = str(edge["left_experience_id"])
        right = str(edge["right_experience_id"])
        if left not in identifiers or right not in identifiers or left >= right:
            raise ValueError("V4.2 positive edge has invalid endpoints")
        pair = (left, right)
        if pair in joint_scores:
            raise ValueError("V4.2 positive edge is duplicated")
        adjacency[left].add(right)
        adjacency[right].add(left)
        joint_scores[pair] = float(edge["joint_similarity"])
    order = sorted(identifiers, key=lambda item: (-len(adjacency[item]), item))
    groups: list[list[str]] = []
    for identifier in order:
        compatible: list[tuple[int, int, float, tuple[str, ...]]] = []
        for index, group in enumerate(groups):
            if all(member in adjacency[identifier] for member in group):
                minimum = min(
                    joint_scores[tuple(sorted((identifier, member)))]
                    for member in group
                )
                compatible.append((index, len(group), minimum, tuple(group)))
        if compatible:
            target = min(
                compatible,
                key=lambda item: (-item[1], -item[2], item[3]),
            )[0]
            groups[target].append(identifier)
            groups[target].sort()
        else:
            groups.append([identifier])
    result = tuple(sorted((tuple(group) for group in groups), key=lambda item: item))
    flattened = [item for group in result for item in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != identifiers:
        raise ValueError("V4.2 complete-link groups do not partition atoms")
    for group in result:
        for left_index, left in enumerate(group):
            if any(right not in adjacency[left] for right in group[left_index + 1 :]):
                raise ValueError("V4.2 complete-link group is not a clique")
    return result


def _representatives(
    member_ids: Sequence[str],
    *,
    atoms_by_id: Mapping[str, V42LocalRepairAtom],
    joint_by_id: Mapping[str, np.ndarray],
    count: int,
) -> tuple[str, ...]:
    """Choose one medoid then farthest-first examples with distinct samples."""

    by_sample: dict[str, list[str]] = {}
    for experience_id in sorted(member_ids):
        by_sample.setdefault(atoms_by_id[experience_id].sample_id, []).append(
            experience_id
        )
    if len(by_sample) < count:
        raise ValueError("V4.2 representative selection lacks distinct samples")
    all_members = tuple(sorted(member_ids))

    def mean_similarity(experience_id: str, comparison: Sequence[str]) -> float:
        vector = joint_by_id[experience_id]
        return float(
            np.mean([float(np.dot(vector, joint_by_id[item])) for item in comparison])
        )

    candidates = [
        min(
            values,
            key=lambda item: (-mean_similarity(item, all_members), item),
        )
        for _sample, values in sorted(by_sample.items())
    ]
    selected = [
        min(
            candidates,
            key=lambda item: (-mean_similarity(item, candidates), item),
        )
    ]
    while len(selected) < count:
        remaining = [item for item in candidates if item not in selected]

        def min_distance(experience_id: str) -> float:
            vector = joint_by_id[experience_id]
            return min(
                1.0 - float(np.dot(vector, joint_by_id[chosen]))
                for chosen in selected
            )

        selected.append(
            min(remaining, key=lambda item: (-min_distance(item), item))
        )
    return tuple(selected)


def build_local_cluster_candidates(
    groups: Sequence[Sequence[str]],
    *,
    atoms: Sequence[V42LocalRepairAtom],
    edges: Sequence[Mapping[str, Any]],
    joint_embeddings: np.ndarray,
    profile: V42ConstructionProfile,
) -> tuple[
    tuple[V42LocalClusterCandidate, ...],
    tuple[dict[str, Any], ...],
]:
    atoms_by_id = {item.experience_id: item for item in atoms}
    if joint_embeddings.shape[0] != len(atoms):
        raise ValueError("V4.2 joint embedding coverage mismatch")
    joint_by_id = {
        item.experience_id: joint_embeddings[index]
        for index, item in enumerate(atoms)
    }
    edge_by_pair = {
        (str(item["left_experience_id"]), str(item["right_experience_id"])): item
        for item in edges
    }
    candidates: list[V42LocalClusterCandidate] = []
    unsupported: list[dict[str, Any]] = []
    for group in groups:
        members = tuple(sorted(str(item) for item in group))
        distinct_samples = {
            atoms_by_id[item].sample_id for item in members
        }
        distribution = tuple(
            sorted(
                Counter(
                    atoms_by_id[item].source_experience_type for item in members
                ).items()
            )
        )
        if len(distinct_samples) < profile.min_distinct_support:
            unsupported.append(
                {
                    "group_id": (
                        "v42-unsupported-"
                        f"{canonical_json_sha256(list(members))[:20]}"
                    ),
                    "member_experience_ids": list(members),
                    "member_count": len(members),
                    "distinct_sample_count": len(distinct_samples),
                    "source_experience_type_distribution": dict(distribution),
                    "reason": "fewer than five distinct construction samples",
                }
            )
            continue
        pair_values: list[Mapping[str, Any]] = []
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                edge = edge_by_pair.get((left, right))
                if edge is None:
                    raise ValueError("V4.2 qualified group is missing a complete-link edge")
                pair_values.append(edge)
        if not pair_values:
            raise ValueError("V4.2 qualified group has no pair evidence")

        def metric(name: str) -> tuple[float, float]:
            values = [float(item[name]) for item in pair_values]
            return min(values), float(np.mean(values))

        mechanism_min, mechanism_mean = metric("mechanism_similarity")
        repair_min, repair_mean = metric("repair_similarity")
        applicability_min, applicability_mean = metric(
            "applicability_similarity"
        )
        joint_min, joint_mean = metric("joint_similarity")
        representatives = _representatives(
            members,
            atoms_by_id=atoms_by_id,
            joint_by_id=joint_by_id,
            count=profile.representative_count,
        )
        membership = {
            "member_experience_ids": list(members),
            "source_signature_sha256": {
                item: atoms_by_id[item].source_signature_sha256 for item in members
            },
            "profile_sha256": profile.profile_sha256,
        }
        membership_sha256 = canonical_json_sha256(membership)
        candidate = V42LocalClusterCandidate(
            candidate_id=f"v42-candidate-{membership_sha256[:20]}",
            member_experience_ids=members,
            representative_experience_ids=representatives,
            distinct_sample_count=len(distinct_samples),
            source_experience_type_distribution=distribution,
            mechanism_similarity_min=mechanism_min,
            mechanism_similarity_mean=mechanism_mean,
            repair_similarity_min=repair_min,
            repair_similarity_mean=repair_mean,
            applicability_similarity_min=applicability_min,
            applicability_similarity_mean=applicability_mean,
            joint_similarity_min=joint_min,
            joint_similarity_mean=joint_mean,
            membership_sha256=membership_sha256,
        )
        candidates.append(candidate)
    return (
        tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        tuple(sorted(unsupported, key=lambda item: item["group_id"])),
    )


def _candidate_record(candidate: V42LocalClusterCandidate) -> dict[str, Any]:
    payload = candidate.to_dict()
    return {
        "schema_version": LOCAL_CLUSTER_RECORD_SCHEMA,
        "candidate": payload,
        "candidate_sha256": canonical_json_sha256(payload),
    }


def _atom_record(atom: V42LocalRepairAtom) -> dict[str, Any]:
    payload = atom.to_dict()
    return {
        "schema_version": LOCAL_ATOM_RECORD_SCHEMA,
        "atom": payload,
        "atom_sha256": canonical_json_sha256(payload),
    }


def build_review_packet(
    candidate: V42LocalClusterCandidate,
    *,
    atoms_by_id: Mapping[str, V42LocalRepairAtom],
) -> dict[str, Any]:
    representatives = [
        atoms_by_id[experience_id].to_dict()
        for experience_id in candidate.representative_experience_ids
    ]
    evidence_sha256 = canonical_json_sha256(representatives)
    packet = {
        "schema_version": V4_2_REVIEW_PACKET_SCHEMA,
        "candidate_id": candidate.candidate_id,
        "membership_sha256": candidate.membership_sha256,
        "member_count": len(candidate.member_experience_ids),
        "distinct_sample_count": candidate.distinct_sample_count,
        "source_experience_type_distribution": dict(
            candidate.source_experience_type_distribution
        ),
        "similarity_floor": {
            "mechanism": candidate.mechanism_similarity_min,
            "repair": candidate.repair_similarity_min,
            "applicability": candidate.applicability_similarity_min,
            "joint": candidate.joint_similarity_min,
        },
        "representatives": representatives,
        "representative_evidence_sha256": evidence_sha256,
    }
    packet["packet_sha256"] = canonical_json_sha256(packet)
    return packet


def build_preflight_report(
    *,
    candidates: Sequence[V42LocalClusterCandidate],
    atoms: Sequence[V42LocalRepairAtom],
    profile: V42ConstructionProfile,
    diagnostics: Mapping[str, Any],
    cluster_plan_sha256: str,
) -> dict[str, Any]:
    atoms_by_id = {item.experience_id: item for item in atoms}
    evidence_characters = 0
    for candidate in candidates:
        evidence = [
            atoms_by_id[item].to_dict()
            for item in candidate.representative_experience_ids
        ]
        evidence_characters += len(
            json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        )
    initial_requests = math.ceil(len(candidates) / profile.synthesis_batch_size)
    within_guardrail = len(candidates) <= profile.max_api_candidates
    report = {
        "schema_version": V4_2_PREFLIGHT_REPORT_SCHEMA,
        "construction_version": "v4.2",
        "status": "local_cluster_complete_api_not_started",
        "external_api_calls_made": 0,
        "api_key_read": False,
        "automatic_paid_stage_transition": False,
        "qualified_candidate_count": len(candidates),
        "synthesis_batch_size": profile.synthesis_batch_size,
        "planned_initial_synthesis_requests": initial_requests,
        "max_api_candidates": profile.max_api_candidates,
        "within_candidate_guardrail": within_guardrail,
        "synthesis_blocked_reason": (
            None
            if within_guardrail
            else "qualified candidate count exceeds the authenticated API guardrail"
        ),
        "representatives_per_candidate": profile.representative_count,
        "representative_signature_evidence_characters": evidence_characters,
        "estimated_evidence_tokens_at_three_chars_per_token": math.ceil(
            evidence_characters / 3
        ),
        "note": (
            "The token estimate covers compact representative signatures only "
            "and is not a tokenizer-derived upper bound; "
            "a future synthesis command must add its exact prompt and bounded "
            "success/failure evidence before requesting explicit API approval."
        ),
        "local_graph_diagnostics": dict(diagnostics),
        "profile_sha256": profile.profile_sha256,
        "cluster_plan_sha256": cluster_plan_sha256,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def build_cluster_plan(
    *,
    signatures: Sequence[V4RepairSignature],
    atoms: Sequence[V42LocalRepairAtom],
    signature_archive: Mapping[str, Sequence[str]],
    unsupported_groups: Sequence[Mapping[str, Any]],
    candidates: Sequence[V42LocalClusterCandidate],
    edge_diagnostics: Mapping[str, Any],
    groups: Sequence[Sequence[str]],
    profile: V42ConstructionProfile,
    source_signature_info: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    unsupported_ids = sorted(
        str(experience_id)
        for group in unsupported_groups
        for experience_id in group["member_experience_ids"]
    )
    archive = {
        "source_nonapplicable_experience_ids": list(
            signature_archive["source_nonapplicable_experience_ids"]
        ),
        "answer_serialization_experience_ids": list(
            signature_archive["answer_serialization_experience_ids"]
        ),
        "unsupported_local_cluster_experience_ids": unsupported_ids,
    }
    payload = {
        "schema_version": V4_2_LOCAL_CLUSTER_PLAN_SCHEMA,
        "construction_version": "v4.2",
        "status": "local_clusters_complete_not_teacher_synthesized",
        "qualified_for_online_use": False,
        "external_api_calls": 0,
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "source_signatures": dict(source_signature_info),
        "clusters": [item.to_dict() for item in candidates],
        "archive": archive,
        "unsupported_groups": list(unsupported_groups),
        "diagnostics": {
            "source_signature_count": len(signatures),
            "local_reasoning_atom_count": len(atoms),
            "complete_link_group_count": len(groups),
            "qualified_cluster_count": len(candidates),
            "qualified_member_count": sum(
                len(item.member_experience_ids) for item in candidates
            ),
            "unsupported_group_count": len(unsupported_groups),
            "unsupported_member_count": len(unsupported_ids),
            "cross_source_type_cluster_count": sum(
                len(item.source_experience_type_distribution) > 1
                for item in candidates
            ),
            "cluster_member_count_distribution": _distribution(
                [len(item.member_experience_ids) for item in candidates]
            ),
            "cluster_distinct_support_distribution": _distribution(
                [item.distinct_sample_count for item in candidates]
            ),
            "edge": dict(edge_diagnostics),
        },
        "artifacts": {
            name: file_sha256(output_dir / name)
            for name in (
                "local_atoms.jsonl",
                "multiview_embeddings_manifest.json",
                "mechanism_embeddings.npy",
                "repair_embeddings.npy",
                "applicability_embeddings.npy",
                "positive_edges.jsonl",
                "positive_edge_manifest.json",
                "local_clusters.jsonl",
                "cluster_review_packets.jsonl",
            )
        },
        "implementation_sha256": _implementation_state(),
    }
    all_ids = {item.experience_id for item in signatures}
    terminal_ids = {
        item
        for values in archive.values()
        for item in values
    } | {
        item
        for candidate in candidates
        for item in candidate.member_experience_ids
    }
    terminal_count = sum(len(values) for values in archive.values()) + sum(
        len(item.member_experience_ids) for item in candidates
    )
    if terminal_ids != all_ids or terminal_count != len(all_ids):
        raise ValueError("V4.2 cluster plan terminal categories are incomplete or overlap")
    payload["plan_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> None:
    args = parse_args()
    profile = _profile(args)
    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    signatures, source_signature_info = load_authenticated_signatures(
        args.source_signatures,
        source_profile_path=args.source_construction_profile,
        experiences=experiences,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_profile(
        output_dir / "construction_profile.json",
        profile=profile,
        source_signature_info=source_signature_info,
        resume=args.resume,
    )
    atoms, signature_archive = partition_signatures(signatures)
    if len(atoms) < profile.min_distinct_support:
        raise RuntimeError("V4.2 has fewer than five local reasoning atoms")
    _write_jsonl(
        output_dir / "local_atoms.jsonl",
        (_atom_record(item) for item in atoms),
    )
    print(
        f"[v4.2-local] inputs PASS signatures={len(signatures)} "
        f"reasoning={len(atoms)} "
        f"serialization={len(signature_archive['answer_serialization_experience_ids'])} "
        f"nonapplicable={len(signature_archive['source_nonapplicable_experience_ids'])}",
        flush=True,
    )
    embeddings = load_or_build_embeddings(
        atoms,
        profile=profile,
        output_dir=output_dir,
        args=args,
    )
    edges, edge_diagnostics, joint_embeddings = build_multiview_positive_edges(
        atoms,
        embeddings,
        profile=profile,
    )
    _write_jsonl(output_dir / "positive_edges.jsonl", edges)
    edge_artifact = {
        "schema_version": POSITIVE_EDGE_ARTIFACT_SCHEMA,
        "positive_edge_count": len(edges),
        "positive_edge_file": "positive_edges.jsonl",
        "positive_edge_file_sha256": file_sha256(
            output_dir / "positive_edges.jsonl"
        ),
        "edge_diagnostics": edge_diagnostics,
        "profile_sha256": profile.profile_sha256,
    }
    edge_artifact["artifact_sha256"] = canonical_json_sha256(edge_artifact)
    _write_json(output_dir / "positive_edge_manifest.json", edge_artifact)
    groups = form_complete_link_groups(atoms, edges)
    candidates, unsupported_groups = build_local_cluster_candidates(
        groups,
        atoms=atoms,
        edges=edges,
        joint_embeddings=joint_embeddings,
        profile=profile,
    )
    _write_jsonl(
        output_dir / "local_clusters.jsonl",
        (_candidate_record(item) for item in candidates),
    )
    atoms_by_id = {item.experience_id: item for item in atoms}
    _write_jsonl(
        output_dir / "cluster_review_packets.jsonl",
        (
            build_review_packet(item, atoms_by_id=atoms_by_id)
            for item in candidates
        ),
    )
    plan = build_cluster_plan(
        signatures=signatures,
        atoms=atoms,
        signature_archive=signature_archive,
        unsupported_groups=unsupported_groups,
        candidates=candidates,
        edge_diagnostics=edge_diagnostics,
        groups=groups,
        profile=profile,
        source_signature_info=source_signature_info,
        output_dir=output_dir,
    )
    _write_json(output_dir / "local_cluster_plan.json", plan)
    preflight = build_preflight_report(
        candidates=candidates,
        atoms=atoms,
        profile=profile,
        diagnostics=plan["diagnostics"],
        cluster_plan_sha256=plan["plan_sha256"],
    )
    _write_json(output_dir / "api_preflight_report.json", preflight)
    print(
        f"[v4.2-local] complete atoms={len(atoms)} edges={len(edges)} "
        f"groups={len(groups)} qualified={len(candidates)} "
        f"unsupported={len(unsupported_groups)} api_calls=0",
        flush=True,
    )
    print(
        f"[v4.2-local] preflight={output_dir / 'api_preflight_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.2-local] error: {exc}", file=sys.stderr)
        raise
