"""Applicability-aware dual-key retrieval for MemGen V3.5.

The V3.5 selector deliberately keeps three artifacts separate:

* an applicability key copied from, and exactly reproduced against, the
  authenticated V3 ``when_facing`` key bank;
* a newly compiled dynamic key containing only ``when_facing`` and the
  independently sanitized Phase-1 ``transferable_decision``; and
* the existing layer-24 side-KV value bank.

All joins and alignments in this module fail closed.  Online retrieval first
builds a fixed question-only applicability shortlist, then performs exact
cosine reranking only inside that shortlist.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from memgen.experience.memory import (
    ApprovedMemorySourceSelector,
    MemoryRecord,
    MemoryRecordRejected,
    MemorySanitizerConfig,
    PayloadSanitizer,
    Phase1MemorySource,
)
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    text_sha256,
)
from memgen.experience.v3 import (
    ApplicabilityAwareRetrievalDecision,
    V34_QUERY_POOLING_CURRENT_TOKEN,
    query_embedding_token_index,
)
from memgen.experience.v3_5_selector import V35_DUAL_KEY_BANK_SCHEMA
from memgen.experience.v3_artifacts import validate_cross_bank_metadata
from memgen.model.retrieval_keys import (
    RetrievalKeyBankLoader,
    encode_last_layer_token,
    tensor_sha256,
)
from memgen.model.side_kv import SIDE_KV_ATTENTION_BACKEND


V35_DUAL_KEY_TENSOR_FILE = "dual_retrieval_key_bank.safetensors"
V35_DUAL_KEY_MANIFEST_FILE = "dual_retrieval_key_manifest.json"
V35_APPLICABILITY_TENSOR_NAME = "applicability_key_embeddings"
V35_DYNAMIC_TENSOR_NAME = "dynamic_key_embeddings"
V35_STATIC_QUESTION_QUERY_SCHEMA = (
    "experience-memory-v3.5-static-question-query-v1"
)
V35_STATIC_SHORTLIST_SCHEMA = "experience-memory-v3.5-static-shortlist-v1"

V35_APPLICABILITY_KEY_SOURCE = "sanitized_fields.when_facing"
V35_DYNAMIC_KEY_SOURCE = (
    "sanitized_fields.when_facing_plus_sanitized_"
    "bank.target.transferable_decision"
)
V35_DYNAMIC_DECISION_PATH = "bank.target.transferable_decision"

_REQUIRED_PROVENANCE_FIELDS = (
    "memory_records_sha256",
    "side_kv_manifest_sha256",
    "e0_final_report_sha256",
    "v3_retrieval_key_manifest_sha256",
    "v3_retrieval_key_tensor_sha256",
    "v3_offline_report_sha256",
    "phase1_approved_bank_sha256",
    "verified_experiences_sha256",
    "split_manifest_sha256",
    "split_manifest_logical_sha256",
    "dataset_revision",
    "compiler_git_revision",
    "compiler_tracked_diff_sha256",
    "compiler_implementation_set_sha256",
)

# This is intentionally the same implementation surface hashed by the V3.5
# evaluator.  File hashes cover uncommitted and untracked implementation files;
# the separately recorded scoped git diff makes the worktree state inspectable.
V35_IMPLEMENTATION_PATHS = (
    "memgen/experience/risk.py",
    "memgen/experience/v3.py",
    "memgen/experience/v3_5_selector.py",
    "memgen/experience/v3_selector.py",
    "memgen/experience/v3_artifacts.py",
    "memgen/experience/v3_eval.py",
    "memgen/model/retrieval_keys.py",
    "memgen/model/v3_5_retrieval.py",
    "memgen/model/e1_runtime.py",
    "memgen/model/side_kv.py",
    "memgen/model/v3_runtime.py",
    "scripts/compile_v3_5_dual_selector.py",
    "scripts/calibrate_v3_5_dynamic_selector.py",
    "scripts/analyze_v3_evaluation.py",
    "scripts/compare_v3_5_applicability_selector.py",
    "scripts/qualify_v3_5_dev.py",
    "scripts/run_online_experience_memory_v3.py",
    "scripts/evaluate_v3_experience_memory.py",
    "scripts/experiments/gsm8k/run_v3_5_applicability_selector_experiment.sh",
)

_V35_PROHIBITED_DYNAMIC_BOILERPLATE = re.compile(
    r"(?:\\(?:boxed|fbox)\b|\bboxed\b|\bfinal(?:[\s-]+)answer\b)",
    flags=re.IGNORECASE,
)
_V35_UNIT_NORM_ATOL = 1e-5


def v35_implementation_files_sha256() -> dict[str, str]:
    """Hash the complete frozen V3.5 implementation surface by relative path."""

    project_root = Path(__file__).resolve().parents[2]
    missing = [
        relative
        for relative in V35_IMPLEMENTATION_PATHS
        if not (project_root / relative).is_file()
    ]
    if missing:
        raise ValueError(
            f"V3.5 implementation set is incomplete: {sorted(missing)}"
        )
    return {
        relative: file_sha256(project_root / relative)
        for relative in V35_IMPLEMENTATION_PATHS
    }


def validate_v35_split_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate Phase-1 split semantics, not merely the input file bytes."""

    if not isinstance(value, Mapping):
        raise ValueError("V3.5 split manifest must be a JSON object")
    manifest = dict(value)
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("Unexpected V3.5 Phase-1 split manifest schema")
    expected_hash = manifest.get("manifest_sha256")
    actual_hash = canonical_json_sha256({
        key: item
        for key, item in manifest.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError("V3.5 Phase-1 split manifest logical hash mismatch")

    dataset = manifest.get("dataset")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("name") != "openai/gsm8k"
        or dataset.get("configuration") != "main"
        or not isinstance(dataset.get("revision"), str)
        or not str(dataset["revision"]).strip()
    ):
        raise ValueError("V3.5 Phase-1 split dataset provenance is invalid")
    overlap = manifest.get("overlap_check")
    if (
        not isinstance(overlap, Mapping)
        or overlap.get("passed") is not True
        or int(overlap.get("overlap_count", -1)) != 0
    ):
        raise ValueError("V3.5 Phase-1 split overlap audit did not pass")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("V3.5 Phase-1 split manifest has no samples")
    allowed_splits = {
        "bank-source": "train",
        "calibration-val": "train",
        "dev-test": "train",
        "final-test": "test",
    }
    seen_sample_ids: set[str] = set()
    seen_members: set[tuple[str, int]] = set()
    hashes_by_split: dict[str, set[str]] = {
        split: set() for split in allowed_splits
    }
    actual_counts = {split: 0 for split in allowed_splits}
    train_size = dataset.get("train_size")
    test_size = dataset.get("test_size")
    if (
        isinstance(train_size, bool)
        or not isinstance(train_size, int)
        or train_size <= 0
        or isinstance(test_size, bool)
        or not isinstance(test_size, int)
        or test_size < 0
    ):
        raise ValueError("V3.5 Phase-1 split dataset sizes are invalid")
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("V3.5 Phase-1 split sample is malformed")
        sample_id = str(sample.get("sample_id", ""))
        logical_split = str(sample.get("logical_split", ""))
        dataset_split = str(sample.get("dataset_split", ""))
        source_index = sample.get("source_index")
        question_hash = str(sample.get("question_sha256", ""))
        answer_hash = str(sample.get("answer_sha256", ""))
        if (
            not sample_id
            or sample_id in seen_sample_ids
            or logical_split not in allowed_splits
            or dataset_split != allowed_splits[logical_split]
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or not question_hash
            or not answer_hash
        ):
            raise ValueError("V3.5 Phase-1 split sample identity is invalid")
        size = train_size if dataset_split == "train" else test_size
        member_key = (dataset_split, source_index)
        if source_index >= size or member_key in seen_members:
            raise ValueError("V3.5 Phase-1 split source membership is invalid")
        seen_sample_ids.add(sample_id)
        seen_members.add(member_key)
        hashes_by_split[logical_split].add(question_hash)
        actual_counts[logical_split] += 1

    for left_index, left in enumerate(sorted(hashes_by_split)):
        for right in sorted(hashes_by_split)[left_index + 1 :]:
            if hashes_by_split[left] & hashes_by_split[right]:
                raise ValueError("V3.5 Phase-1 split contains question leakage")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(split) != count for split, count in actual_counts.items()
    ):
        raise ValueError("V3.5 Phase-1 split counts differ from its samples")
    if actual_counts["bank-source"] <= 0:
        raise ValueError("V3.5 Phase-1 split has no bank-source members")
    return manifest


def canonicalize_v35_query_embedding(
    value: torch.Tensor, *, expected_width: int, owner: str
) -> torch.Tensor:
    """Return one canonical query, preserving already-unit float32 bits."""

    query = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    if query.shape != (expected_width,):
        raise ValueError(f"V3.5 {owner} query/key embedding widths differ")
    if not torch.isfinite(query).all():
        raise ValueError(f"V3.5 {owner} query embedding is invalid")
    norm = float(query.norm().item())
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"V3.5 {owner} query embedding is invalid")
    if math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_V35_UNIT_NORM_ATOL):
        return query
    return F.normalize(query, dim=0).contiguous()


def validate_v35_dynamic_text_component(*, owner: str, text: str) -> None:
    """Reject V3.5-forbidden final-answer/verifier boilerplate verbatim."""

    if _V35_PROHIBITED_DYNAMIC_BOILERPLATE.search(text):
        raise ValueError(
            f"V3.5 {owner} contains prohibited final-answer/boxed boilerplate"
        )


def _logical_manifest_sha256(value: Mapping[str, Any]) -> str:
    return canonical_json_sha256({
        key: item for key, item in value.items() if key != "manifest_sha256"
    })


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("The V3.5 encoder model exposes no parameter device") from exc


def _require_unique_ids(
    values: Iterable[Mapping[str, Any]], *, owner: str
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in values:
        experience_id = str(value.get("experience_id", ""))
        if not experience_id or experience_id in by_id:
            raise ValueError(
                f"V3.5 {owner} has a missing or duplicate experience_id: "
                f"{experience_id!r}"
            )
        by_id[experience_id] = value
    if not by_id:
        raise ValueError(f"V3.5 {owner} is empty")
    return by_id


def _read_mapping_path(root: Mapping[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


@dataclass(frozen=True)
class DualRetrievalKeyCompilerConfig:
    """Frozen representation and text contracts for both V3.5 keys."""

    layer_number: int = 24
    hidden_state_tuple_index: int = 24
    representation: str = "decoder_layer_output"
    applicability_key_source: str = V35_APPLICABILITY_KEY_SOURCE
    dynamic_key_source: str = V35_DYNAMIC_KEY_SOURCE
    pooling: str = "last_valid_token"
    normalization: str = "l2"
    attention_backend: str = SIDE_KV_ATTENTION_BACKEND
    model_compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.layer_number != 24:
            raise ValueError("V3.5 dual retrieval is frozen to layer 24")
        if (
            self.hidden_state_tuple_index != 24
            or self.representation != "decoder_layer_output"
        ):
            raise ValueError("V3.5 keys require layer-24 decoder output")
        if self.applicability_key_source != V35_APPLICABILITY_KEY_SOURCE:
            raise ValueError("V3.5 applicability source must be when_facing")
        if self.dynamic_key_source != V35_DYNAMIC_KEY_SOURCE:
            raise ValueError("V3.5 dynamic key source contract drifted")
        if self.pooling != "last_valid_token" or self.normalization != "l2":
            raise ValueError("V3.5 dual keys require last-valid pooling and L2")
        if self.attention_backend != SIDE_KV_ATTENTION_BACKEND:
            raise ValueError("V3.5 dual keys require SDPA")
        if self.model_compute_dtype != "bfloat16":
            raise ValueError("V3.5 dual-key model compute dtype is frozen to bfloat16")


@dataclass(frozen=True)
class CompiledDualRetrievalKeyBank:
    """Two aligned unit-vector matrices plus their authenticated manifest."""

    applicability_embeddings: torch.Tensor
    dynamic_embeddings: torch.Tensor
    manifest: Mapping[str, Any]

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / V35_DUAL_KEY_TENSOR_FILE
        manifest_path = output_dir / V35_DUAL_KEY_MANIFEST_FILE
        save_file(
            {
                V35_APPLICABILITY_TENSOR_NAME: (
                    self.applicability_embeddings.contiguous().cpu()
                ),
                V35_DYNAMIC_TENSOR_NAME: self.dynamic_embeddings.contiguous().cpu(),
            },
            str(tensor_path),
            metadata={"schema_version": V35_DUAL_KEY_BANK_SCHEMA},
        )
        manifest = dict(self.manifest)
        manifest["tensor_artifact"] = {
            "path": tensor_path.name,
            "sha256": file_sha256(tensor_path),
        }
        manifest["manifest_sha256"] = _logical_manifest_sha256(manifest)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return tensor_path, manifest_path


@dataclass(frozen=True)
class _JoinedSource:
    record: MemoryRecord
    source: Phase1MemorySource
    transferable_decision: str
    split_member: Mapping[str, Any]


class DualRetrievalKeyCompiler:
    """Compile the V3.5 dynamic bank and reproduce every reused V3 key."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        reasoner_name: str,
        reasoner_revision: str,
        tokenizer_revision: str,
        config: DualRetrievalKeyCompilerConfig | None = None,
        source_selector: ApprovedMemorySourceSelector | None = None,
        sanitizer: PayloadSanitizer | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reasoner_name = str(reasoner_name)
        self.reasoner_revision = str(reasoner_revision)
        self.tokenizer_revision = str(tokenizer_revision)
        self.config = config or DualRetrievalKeyCompilerConfig()
        self.source_selector = source_selector or ApprovedMemorySourceSelector(
            allowed_experience_types=("answer_correctness",)
        )
        self.sanitizer = sanitizer or PayloadSanitizer(
            MemorySanitizerConfig(forbid_numeric_literals=True)
        )

    @torch.inference_mode()
    def compile(
        self,
        *,
        records: Sequence[MemoryRecord],
        approved_records: Sequence[Mapping[str, Any]],
        verified_experiences: Sequence[Mapping[str, Any]],
        applicability_key_bank: RetrievalKeyBankLoader,
        side_kv_manifest: Mapping[str, Any],
        split_manifest: Mapping[str, Any],
        artifact_provenance: Mapping[str, Any],
    ) -> CompiledDualRetrievalKeyBank:
        if not records:
            raise ValueError("Cannot compile an empty V3.5 dual-key bank")
        provenance = self._validate_provenance(artifact_provenance)
        authenticated_split = validate_v35_split_manifest(split_manifest)
        if (
            provenance["split_manifest_logical_sha256"]
            != authenticated_split["manifest_sha256"]
            or provenance["dataset_revision"]
            != authenticated_split["dataset"]["revision"]
        ):
            raise ValueError(
                "V3.5 provenance does not bind the authenticated Phase-1 split"
            )
        if _model_device(self.model).type != "meta" and (
            next(self.model.parameters()).dtype != torch.bfloat16
        ):
            raise ValueError("V3.5 dual-key compilation requires a bfloat16 model")
        if file_sha256(applicability_key_bank.manifest_path) != provenance[
            "v3_retrieval_key_manifest_sha256"
        ]:
            raise ValueError(
                "V3.5 provenance does not bind the reused V3 key manifest"
            )
        old_tensor_sha256 = str(
            applicability_key_bank.manifest.get("tensor_artifact", {}).get(
                "sha256", ""
            )
        )
        if old_tensor_sha256 != provenance["v3_retrieval_key_tensor_sha256"]:
            raise ValueError(
                "V3.5 provenance does not bind the reused V3 key tensor"
            )
        self._validate_record_and_bank_alignment(
            records=records,
            applicability_key_bank=applicability_key_bank,
            side_kv_manifest=side_kv_manifest,
        )
        joined, source_join_audit = self._join_sources(
            records=records,
            approved_records=approved_records,
            verified_experiences=verified_experiences,
            split_manifest=authenticated_split,
        )

        was_training = bool(self.model.training)
        self.model.eval()
        compiled: list[dict[str, Any]] = []
        try:
            for index, item in enumerate(joined):
                compiled.append(
                    self._compile_one(
                        index=index,
                        joined=item,
                        old_bank=applicability_key_bank,
                    )
                )
        finally:
            self.model.train(was_training)

        applicability = applicability_key_bank.embeddings.detach().float().cpu().clone()
        dynamic = torch.stack(
            [item["dynamic_embedding"] for item in compiled], dim=0
        ).float()
        self._validate_embedding_matrix(applicability, owner="applicability")
        self._validate_embedding_matrix(dynamic, owner="dynamic")
        if applicability.shape != dynamic.shape:
            raise RuntimeError("V3.5 applicability/dynamic embedding shapes differ")

        side_entries = list(side_kv_manifest["records"])
        entries: list[dict[str, Any]] = []
        for index, (item, side_entry) in enumerate(zip(compiled, side_entries)):
            record = item["record"]
            applicability_embedding = applicability[index]
            dynamic_embedding = dynamic[index]
            entries.append({
                "index": index,
                "memory_id": record.memory_id,
                "source_experience_id": record.source_experience_id,
                "payload_hash": record.payload_hash,
                "payload_token_count": record.token_count,
                "kv_layer": record.kv_layer,
                "kv_valid_slot_count": int(side_entry["kv_valid_slot_count"]),
                "applicability_key_source": self.config.applicability_key_source,
                "applicability_key_text_sha256": text_sha256(
                    item["applicability_text"]
                ),
                "applicability_key_token_count": len(
                    item["applicability_token_ids"]
                ),
                "applicability_key_token_ids_sha256": canonical_json_sha256(
                    item["applicability_token_ids"]
                ),
                "applicability_key_embedding_sha256": tensor_sha256(
                    applicability_embedding
                ),
                "applicability_key_embedding_norm": float(
                    applicability_embedding.norm().item()
                ),
                "reproduced_applicability_key_embedding_sha256": tensor_sha256(
                    item["reproduced_applicability_embedding"]
                ),
                "applicability_embedding_exact_reproduction": True,
                "dynamic_key_source": self.config.dynamic_key_source,
                "dynamic_key_text_sha256": text_sha256(item["dynamic_text"]),
                "dynamic_key_token_count": len(item["dynamic_token_ids"]),
                "dynamic_key_token_ids_sha256": canonical_json_sha256(
                    item["dynamic_token_ids"]
                ),
                "dynamic_key_embedding_sha256": tensor_sha256(dynamic_embedding),
                "dynamic_key_embedding_norm": float(dynamic_embedding.norm().item()),
                "source_record_sha256": record.source_record_sha256,
                "phase1_provenance_sha256": record.phase1_provenance_sha256,
                "review_provenance_sha256": record.review_provenance_sha256,
                "review_validation_profile": item["review_validation_profile"],
                "source_sample_id": str(item["split_member"]["sample_id"]),
                "source_dataset_revision": provenance["dataset_revision"],
                "source_dataset_split": str(
                    item["split_member"]["dataset_split"]
                ),
                "source_logical_split": str(
                    item["split_member"]["logical_split"]
                ),
                "source_index": int(item["split_member"]["source_index"]),
                "source_question_sha256": str(
                    item["split_member"]["question_sha256"]
                ),
                "source_split_manifest_sha256": provenance[
                    "split_manifest_logical_sha256"
                ],
                "split_member_sha256": canonical_json_sha256(
                    item["split_member"]
                ),
            })

        memory_ids = [record.memory_id for record in records]
        record_order_hash = canonical_json_sha256(memory_ids)
        manifest = {
            "schema_version": V35_DUAL_KEY_BANK_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reasoner": {
                "model_name": self.reasoner_name,
                "model_revision": self.reasoner_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "attention_implementation": self.config.attention_backend,
            },
            "compiler": asdict(self.config),
            "model_compute_dtype": self.config.model_compute_dtype,
            "sanitizer": asdict(self.sanitizer.config),
            "input_artifacts": provenance,
            "record_count": len(records),
            "record_order_sha256": record_order_hash,
            "ordered_memory_ids_sha256": record_order_hash,
            "embedding_shape": list(applicability.shape),
            "embedding_dtype": str(applicability.dtype),
            "tensor_names": {
                "applicability": V35_APPLICABILITY_TENSOR_NAME,
                "dynamic": V35_DYNAMIC_TENSOR_NAME,
            },
            "source_join": {
                "policy": "approved_verified_memory_one_to_one_fail_closed",
                "joined_record_count": len(joined),
                "dynamic_decision_path": V35_DYNAMIC_DECISION_PATH,
                **source_join_audit,
            },
            "phase1_split_audit": {
                "schema_version": authenticated_split["schema_version"],
                "manifest_logical_sha256": authenticated_split[
                    "manifest_sha256"
                ],
                "dataset_revision": authenticated_split["dataset"]["revision"],
                "overlap_check_verified": True,
                "joined_bank_source_member_count": len(joined),
                "authenticated_valid_source_member_count": (
                    source_join_audit["validated_source_count"]
                ),
                "all_sources_match_authenticated_members": True,
            },
            "applicability_reproduction_audit": {
                "source_schema": applicability_key_bank.manifest["schema_version"],
                "record_count": len(records),
                "exact_reproduction_count": len(records),
                "all_exact": True,
            },
            "alignment": validate_cross_bank_metadata(
                records=records,
                side_manifest=side_kv_manifest,
                key_manifest=applicability_key_bank.manifest,
            ),
            "records": entries,
        }
        return CompiledDualRetrievalKeyBank(
            applicability_embeddings=applicability,
            dynamic_embeddings=dynamic.cpu(),
            manifest=manifest,
        )

    @staticmethod
    def _validate_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(value)
        missing = [
            field
            for field in _REQUIRED_PROVENANCE_FIELDS
            if not isinstance(normalized.get(field), str)
            or not str(normalized[field]).strip()
        ]
        if missing:
            raise ValueError(
                f"V3.5 dual-key provenance is incomplete: {sorted(missing)}"
            )
        implementation_files = normalized.get(
            "compiler_implementation_files_sha256"
        )
        if not isinstance(implementation_files, Mapping):
            raise ValueError(
                "V3.5 compiler implementation-file provenance is missing"
            )
        implementation_files = {
            str(path): str(digest)
            for path, digest in implementation_files.items()
        }
        current_files = v35_implementation_files_sha256()
        if (
            implementation_files != current_files
            or normalized["compiler_implementation_set_sha256"]
            != canonical_json_sha256(implementation_files)
        ):
            raise ValueError("V3.5 compiler implementation identity drifted")
        normalized["compiler_implementation_files_sha256"] = (
            implementation_files
        )
        return normalized

    def _validate_record_and_bank_alignment(
        self,
        *,
        records: Sequence[MemoryRecord],
        applicability_key_bank: RetrievalKeyBankLoader,
        side_kv_manifest: Mapping[str, Any],
    ) -> None:
        memory_ids = [record.memory_id for record in records]
        if (
            any(not memory_id for memory_id in memory_ids)
            or len(set(memory_ids)) != len(memory_ids)
        ):
            raise ValueError("V3.5 MemoryRecord IDs are missing or duplicated")
        source_ids = [record.source_experience_id for record in records]
        if (
            any(not source_id for source_id in source_ids)
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError("V3.5 MemoryRecord source join is not one-to-one")
        key_ids = [str(entry["memory_id"]) for entry in applicability_key_bank.entries]
        if key_ids != memory_ids:
            raise ValueError("V3.5 records and reused applicability bank order differ")
        validate_cross_bank_metadata(
            records=records,
            side_manifest=side_kv_manifest,
            key_manifest=applicability_key_bank.manifest,
        )
        compiler = applicability_key_bank.manifest.get("compiler", {})
        if (
            compiler.get("key_source") != V35_APPLICABILITY_KEY_SOURCE
            or int(compiler.get("layer_number", -1)) != 24
            or int(compiler.get("hidden_state_tuple_index", -1)) != 24
            or compiler.get("representation") != "decoder_layer_output"
            or compiler.get("pooling") != "last_valid_token"
            or compiler.get("normalization") != "l2"
        ):
            raise ValueError("Reused V3 applicability encoder contract drifted")
        reasoner = applicability_key_bank.manifest.get("reasoner", {})
        expected_reasoner = {
            "model_name": self.reasoner_name,
            "model_revision": self.reasoner_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "attention_implementation": self.config.attention_backend,
        }
        if any(
            reasoner.get(field) != expected
            for field, expected in expected_reasoner.items()
        ):
            raise ValueError("Reused applicability bank reasoner provenance differs")

    def _join_sources(
        self,
        *,
        records: Sequence[MemoryRecord],
        approved_records: Sequence[Mapping[str, Any]],
        verified_experiences: Sequence[Mapping[str, Any]],
        split_manifest: Mapping[str, Any],
    ) -> tuple[list[_JoinedSource], dict[str, Any]]:
        approved_by_id = _require_unique_ids(approved_records, owner="approved bank")
        verified_by_id = _require_unique_ids(
            verified_experiences, owner="verified experiences"
        )
        if not set(approved_by_id).issubset(verified_by_id):
            raise ValueError(
                "V3.5 approved sources are not covered by verified experiences"
            )
        sources, source_trace = self.source_selector.join(
            approved_records, verified_experiences
        )
        source_by_id = {source.experience_id: source for source in sources}
        if len(source_by_id) != len(sources):
            raise ValueError("V3.5 validated Phase-1 source join is not unique")
        ordered_record_source_ids = [
            record.source_experience_id for record in records
        ]
        record_source_ids = set(ordered_record_source_ids)
        if not record_source_ids.issubset(source_by_id):
            raise ValueError(
                "V3.5 MemoryRecord references a rejected or missing Phase-1 source"
            )

        bank_members: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        for member in split_manifest["samples"]:
            if member["logical_split"] != "bank-source":
                continue
            key = (
                str(member["dataset_split"]),
                int(member["source_index"]),
                str(member["question_sha256"]),
            )
            if key in bank_members:
                raise ValueError(
                    "V3.5 Phase-1 split bank membership is ambiguous"
                )
            bank_members[key] = member

        split_member_by_id = {
            source.experience_id: self._validate_split_source(
                source=source,
                split_manifest=split_manifest,
                bank_members=bank_members,
            )
            for source in sources
        }

        joined: list[_JoinedSource] = []
        for record in records:
            source = source_by_id.get(record.source_experience_id)
            approved = approved_by_id.get(record.source_experience_id)
            if source is None or approved is None:
                raise ValueError(
                    f"{record.memory_id} has no uniquely validated Phase-1 source"
                )
            split_member = split_member_by_id[source.experience_id]
            self._validate_record_source(record=record, source=source)
            sanitized_when_facing = self._sanitize_when_facing(source)
            if (
                str(record.sanitized_fields.get("when_facing", "")).strip()
                != sanitized_when_facing
            ):
                raise ValueError(
                    f"{record.memory_id} sanitized when_facing differs from Phase-1"
                )
            raw_decision = _read_mapping_path(approved, V35_DYNAMIC_DECISION_PATH)
            try:
                decision = self.sanitizer.sanitize_field(
                    path=V35_DYNAMIC_DECISION_PATH,
                    value=raw_decision,
                    source=source,
                )
            except MemoryRecordRejected as exc:
                raise ValueError(
                    f"{record.memory_id} transferable_decision sanitizer failed: "
                    f"{exc}"
                ) from exc
            if not decision.strip():
                raise ValueError(f"{record.memory_id} has an empty dynamic decision")
            validate_v35_dynamic_text_component(
                owner=f"{record.memory_id} when_facing",
                text=str(record.sanitized_fields.get("when_facing", "")),
            )
            validate_v35_dynamic_text_component(
                owner=f"{record.memory_id} transferable_decision",
                text=decision,
            )
            joined.append(
                _JoinedSource(
                    record=record,
                    source=source,
                    transferable_decision=decision,
                    split_member=split_member,
                )
            )
        valid_source_ids = set(source_by_id)
        unselected_valid_ids = sorted(valid_source_ids - record_source_ids)
        rejected_ids = sorted(
            trace.experience_id
            for trace in source_trace
            if trace.status == "rejected_selection"
        )
        audit = {
            "approved_input_count": len(approved_by_id),
            "verified_input_count": len(verified_by_id),
            "validated_source_count": len(valid_source_ids),
            "selector_rejected_source_count": len(rejected_ids),
            "selected_memory_source_count": len(record_source_ids),
            "unselected_valid_source_count": len(unselected_valid_ids),
            "validated_source_ids_sha256": canonical_json_sha256(
                sorted(valid_source_ids)
            ),
            "selected_memory_source_ids_sha256": canonical_json_sha256(
                ordered_record_source_ids
            ),
            "unselected_valid_source_ids_sha256": canonical_json_sha256(
                unselected_valid_ids
            ),
            "selector_rejected_source_ids_sha256": canonical_json_sha256(
                rejected_ids
            ),
        }
        return joined, audit

    @staticmethod
    def _validate_split_source(
        *,
        source: Phase1MemorySource,
        split_manifest: Mapping[str, Any],
        bank_members: Mapping[tuple[str, int, str], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        approved_source = source.approved_record.get("source")
        verified_source = source.verified_experience.get("source")
        if not isinstance(approved_source, Mapping) or not isinstance(
            verified_source, Mapping
        ):
            raise ValueError(
                f"{source.experience_id} Phase-1 split source is missing"
            )
        # ApprovedMemorySourceSelector already requires equality.  Keep the
        # explicit comparison here because this audit is persisted separately.
        if approved_source != verified_source:
            raise ValueError(
                f"{source.experience_id} approved/verified split sources differ"
            )
        source_index = approved_source.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(
                f"{source.experience_id} Phase-1 source index is invalid"
            )
        logical_hash = str(split_manifest["manifest_sha256"])
        dataset_revision = str(split_manifest["dataset"]["revision"])
        if (
            approved_source.get("dataset") != "openai/gsm8k"
            or approved_source.get("dataset_revision") != dataset_revision
            or approved_source.get("logical_split") != "bank-source"
            or approved_source.get("dataset_split") != "train"
            or approved_source.get("split_manifest_sha256") != logical_hash
        ):
            raise ValueError(
                f"{source.experience_id} Phase-1 split provenance differs"
            )
        question_hash = str(approved_source.get("question_sha256", ""))
        member = bank_members.get(("train", source_index, question_hash))
        sample_id = str(source.verified_experience.get("sample_id", ""))
        context = source.verified_experience.get("context")
        if (
            member is None
            or not sample_id
            or sample_id != member.get("sample_id")
            or not isinstance(context, str)
            or text_sha256(context.strip()) != question_hash
        ):
            raise ValueError(
                f"{source.experience_id} is not an authenticated split member"
            )
        return member

    def _sanitize_when_facing(self, source: Phase1MemorySource) -> str:
        values: list[str] = []
        seen: set[str] = set()
        for path in (
            "bank.target.situation_signature",
            "bank.target.applicability_boundary",
        ):
            try:
                normalized = self.sanitizer.sanitize_field(
                    path=path,
                    value=_read_mapping_path(source.approved_record, path),
                    source=source,
                )
            except MemoryRecordRejected as exc:
                raise ValueError(
                    f"{source.experience_id} when_facing sanitizer failed: {exc}"
                ) from exc
            key = normalized.casefold().rstrip(". ")
            if key not in seen:
                seen.add(key)
                values.append(normalized)
        rendered = " ".join(values).strip()
        if not rendered:
            raise ValueError(f"{source.experience_id} has empty when_facing")
        return rendered

    def _validate_record_source(
        self, *, record: MemoryRecord, source: Phase1MemorySource
    ) -> None:
        approved = source.approved_record
        gate = approved.get("ai_review_gate", {})
        student = approved.get("student", {})
        expected = {
            "source_record_sha256": canonical_json_sha256(approved),
            "phase1_provenance_sha256": str(approved.get("provenance_sha256", "")),
            "review_provenance_sha256": str(
                gate.get("review_provenance_sha256", "")
                if isinstance(gate, Mapping)
                else ""
            ),
            "reasoner_name": self.reasoner_name,
            "reasoner_revision": self.reasoner_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }
        for field, expected_value in expected.items():
            if getattr(record, field) != expected_value or not expected_value:
                raise ValueError(
                    f"{record.memory_id} source provenance mismatch: {field}"
                )
        if (
            record.source_experience_id != source.experience_id
            or record.approved_route != "ai_approved"
            or record.experience_type != approved.get("experience_type")
            or record.source_logical_split
            != approved.get("source", {}).get("logical_split")
        ):
            raise ValueError(
                f"{record.memory_id} MemoryRecord/Phase-1 identity metadata differs"
            )
        if not isinstance(student, Mapping) or any(
            student.get(field) != expected_value
            for field, expected_value in (
                ("model_name", self.reasoner_name),
                ("model_revision", self.reasoner_revision),
                ("tokenizer_revision", self.tokenizer_revision),
            )
        ):
            raise ValueError(
                f"{record.memory_id} Phase-1 student provenance differs from reasoner"
            )
        if record.kv_layer != 24:
            raise ValueError(f"{record.memory_id} is not a layer-24 MemoryRecord")

    def _compile_one(
        self,
        *,
        index: int,
        joined: _JoinedSource,
        old_bank: RetrievalKeyBankLoader,
    ) -> dict[str, Any]:
        record = joined.record
        applicability_text = str(
            record.sanitized_fields.get("when_facing", "")
        ).strip()
        if not applicability_text:
            raise ValueError(f"{record.memory_id} has empty when_facing")
        dynamic_text = (
            f"When facing: {applicability_text}\n"
            f"Prefer: {joined.transferable_decision}"
        )
        validate_v35_dynamic_text_component(
            owner=f"{record.memory_id} dynamic key", text=dynamic_text
        )
        applicability_token_ids = [
            int(value)
            for value in self.tokenizer.encode(
                applicability_text, add_special_tokens=False
            )
        ]
        dynamic_token_ids = [
            int(value)
            for value in self.tokenizer.encode(dynamic_text, add_special_tokens=False)
        ]
        if not applicability_token_ids or not dynamic_token_ids:
            raise ValueError(f"{record.memory_id} produced an empty V3.5 key")
        if len(dynamic_token_ids) > record.model_sequence_limit:
            raise ValueError(
                f"{record.memory_id} dynamic key exceeds the model sequence limit"
            )
        reproduced = encode_last_layer_token(
            model=self.model,
            token_ids=applicability_token_ids,
            layer_number=24,
            device=_model_device(self.model),
        ).cpu()
        old_entry = old_bank.entries[index]
        old_embedding = old_bank.embeddings[index]
        expected_text_hash = text_sha256(applicability_text)
        expected_token_hash = canonical_json_sha256(applicability_token_ids)
        if (
            old_entry.get("key_source") != V35_APPLICABILITY_KEY_SOURCE
            or old_entry.get("key_text_sha256") != expected_text_hash
            or int(old_entry.get("key_token_count", -1))
            != len(applicability_token_ids)
            or old_entry.get("key_token_ids_sha256") != expected_token_hash
            or old_entry.get("key_embedding_sha256") != tensor_sha256(old_embedding)
            or tensor_sha256(reproduced) != tensor_sha256(old_embedding)
            or not torch.equal(reproduced, old_embedding)
        ):
            raise ValueError(
                f"{record.memory_id} reused applicability embedding reproduction failed"
            )
        dynamic_embedding = encode_last_layer_token(
            model=self.model,
            token_ids=dynamic_token_ids,
            layer_number=24,
            device=_model_device(self.model),
        ).cpu()
        return {
            "record": record,
            "applicability_text": applicability_text,
            "applicability_token_ids": applicability_token_ids,
            "reproduced_applicability_embedding": reproduced,
            "dynamic_text": dynamic_text,
            "dynamic_token_ids": dynamic_token_ids,
            "dynamic_embedding": dynamic_embedding,
            "review_validation_profile": joined.source.review_validation_profile,
            "split_member": joined.split_member,
        }

    @staticmethod
    def _validate_embedding_matrix(value: torch.Tensor, *, owner: str) -> None:
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
            raise RuntimeError(f"V3.5 {owner} keys must be a non-empty matrix")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"V3.5 {owner} keys contain non-finite values")
        norms = value.norm(dim=-1)
        if not torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5, rtol=0.0
        ):
            raise RuntimeError(f"V3.5 {owner} keys are not L2 normalized")


class DualRetrievalKeyBankLoader:
    """Content-addressed loader that rejects old single-key artifacts."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_reasoner_name: str | None = None,
        expected_reasoner_revision: str | None = None,
        expected_tokenizer_revision: str | None = None,
        expected_input_hashes: Mapping[str, str] | None = None,
    ):
        from safetensors.torch import load_file

        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_reasoner_name=expected_reasoner_name,
            expected_reasoner_revision=expected_reasoner_revision,
            expected_tokenizer_revision=expected_tokenizer_revision,
            expected_input_hashes=expected_input_hashes,
        )
        tensor_info = self.manifest.get("tensor_artifact", {})
        if not isinstance(tensor_info, Mapping):
            raise ValueError("V3.5 dual-key tensor descriptor is missing")
        relative_tensor_path = Path(str(tensor_info.get("path", "")))
        if (
            not relative_tensor_path.parts
            or relative_tensor_path.is_absolute()
            or ".." in relative_tensor_path.parts
            or relative_tensor_path.as_posix() != V35_DUAL_KEY_TENSOR_FILE
        ):
            raise ValueError("V3.5 dual-key tensor path is unsafe")
        artifact_root = self.manifest_path.parent.resolve()
        tensor_path = (artifact_root / relative_tensor_path).resolve()
        if artifact_root not in tensor_path.parents:
            raise ValueError("V3.5 dual-key tensor escaped its artifact directory")
        if (
            not tensor_path.is_file()
            or file_sha256(tensor_path) != tensor_info.get("sha256")
        ):
            raise ValueError("V3.5 dual-key tensor is missing or has a hash mismatch")
        tensors = load_file(str(tensor_path), device="cpu")
        if set(tensors) != {
            V35_APPLICABILITY_TENSOR_NAME,
            V35_DYNAMIC_TENSOR_NAME,
        }:
            raise ValueError("V3.5 dual-key tensor names drifted")
        self.applicability_embeddings = tensors[
            V35_APPLICABILITY_TENSOR_NAME
        ].float()
        self.dynamic_embeddings = tensors[V35_DYNAMIC_TENSOR_NAME].float()
        self.entries = tuple(self.manifest["records"])
        self.entry_by_id = {
            str(entry["memory_id"]): entry for entry in self.entries
        }
        self.index_by_id = {
            str(entry["memory_id"]): int(entry["index"])
            for entry in self.entries
        }
        self._validate_tensors()

    @property
    def input_artifacts(self) -> Mapping[str, Any]:
        return self.manifest["input_artifacts"]

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    def _validate_manifest(
        self,
        *,
        expected_reasoner_name: str | None,
        expected_reasoner_revision: str | None,
        expected_tokenizer_revision: str | None,
        expected_input_hashes: Mapping[str, str] | None,
    ) -> None:
        if self.manifest.get("schema_version") != V35_DUAL_KEY_BANK_SCHEMA:
            raise ValueError("Unexpected V3.5 dual-key manifest schema")
        if self.manifest.get("manifest_sha256") != _logical_manifest_sha256(
            self.manifest
        ):
            raise ValueError("V3.5 dual-key manifest hash mismatch")
        config = DualRetrievalKeyCompilerConfig(
            **self.manifest.get("compiler", {})
        )
        if self.manifest.get("sanitizer") != asdict(
            MemorySanitizerConfig(forbid_numeric_literals=True)
        ):
            raise ValueError("V3.5 dual-key sanitizer contract drifted")
        if self.manifest.get("model_compute_dtype") != "bfloat16":
            raise ValueError("V3.5 dual-key manifest compute dtype drifted")
        reasoner = self.manifest.get("reasoner", {})
        for field, expected in (
            ("model_name", expected_reasoner_name),
            ("model_revision", expected_reasoner_revision),
            ("tokenizer_revision", expected_tokenizer_revision),
        ):
            if expected is not None and reasoner.get(field) != expected:
                raise ValueError(f"V3.5 dual-key reasoner {field} mismatch")
        if reasoner.get("attention_implementation") != config.attention_backend:
            raise ValueError("V3.5 dual-key attention backend drifted")
        inputs = self.manifest.get("input_artifacts")
        if not isinstance(inputs, Mapping):
            raise ValueError("V3.5 dual-key input provenance is missing")
        missing = [
            field
            for field in _REQUIRED_PROVENANCE_FIELDS
            if not isinstance(inputs.get(field), str) or not str(inputs[field]).strip()
        ]
        if missing:
            raise ValueError(f"V3.5 dual-key provenance is incomplete: {missing}")
        implementation_files = inputs.get("compiler_implementation_files_sha256")
        if not isinstance(implementation_files, Mapping):
            raise ValueError(
                "V3.5 compiler implementation-file provenance is missing"
            )
        implementation_files = {
            str(path): str(digest)
            for path, digest in implementation_files.items()
        }
        if (
            any(
                not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                or not digest
                for path, digest in implementation_files.items()
            )
            or implementation_files != v35_implementation_files_sha256()
            or inputs.get("compiler_implementation_set_sha256")
            != canonical_json_sha256(implementation_files)
        ):
            raise ValueError("V3.5 compiler implementation identity drifted")
        if expected_input_hashes is not None:
            for field, expected in expected_input_hashes.items():
                if inputs.get(field) != expected:
                    raise ValueError(
                        f"V3.5 dual-key input artifact mismatch: {field}"
                    )
        records = self.manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("V3.5 dual-key manifest has no records")
        if self.manifest.get("record_count") != len(records):
            raise ValueError("V3.5 dual-key record count mismatch")
        indices = [int(entry.get("index", -1)) for entry in records]
        if indices != list(range(len(records))):
            raise ValueError("V3.5 dual-key record indices are not contiguous")
        memory_ids = [str(entry.get("memory_id", "")) for entry in records]
        source_ids = [
            str(entry.get("source_experience_id", "")) for entry in records
        ]
        if (
            any(not value for value in memory_ids + source_ids)
            or len(set(memory_ids)) != len(memory_ids)
            or len(set(source_ids)) != len(source_ids)
        ):
            raise ValueError("V3.5 dual-key IDs are missing or duplicated")
        order_hash = canonical_json_sha256(memory_ids)
        if (
            self.manifest.get("record_order_sha256") != order_hash
            or self.manifest.get("ordered_memory_ids_sha256") != order_hash
        ):
            raise ValueError("V3.5 dual-key record order hash mismatch")
        if self.manifest.get("embedding_dtype") != "torch.float32":
            raise ValueError("V3.5 dual-key tensors must be stored as float32")
        if self.manifest.get("tensor_names") != {
            "applicability": V35_APPLICABILITY_TENSOR_NAME,
            "dynamic": V35_DYNAMIC_TENSOR_NAME,
        }:
            raise ValueError("V3.5 dual-key tensor-name manifest drifted")
        reproduction = self.manifest.get("applicability_reproduction_audit", {})
        if (
            reproduction.get("source_schema")
            != "experience-memory-retrieval-key-bank-v1"
            or reproduction.get("all_exact") is not True
            or int(reproduction.get("exact_reproduction_count", -1))
            != len(records)
        ):
            raise ValueError("V3.5 applicability reproduction audit did not pass")
        source_join = self.manifest.get("source_join", {})
        selected_source_hash = canonical_json_sha256(source_ids)
        if (
            source_join.get("policy")
            != "approved_verified_memory_one_to_one_fail_closed"
            or int(source_join.get("joined_record_count", -1)) != len(records)
            or source_join.get("dynamic_decision_path")
            != V35_DYNAMIC_DECISION_PATH
            or int(source_join.get("approved_input_count", -1))
            != int(source_join.get("validated_source_count", -2))
            + int(source_join.get("selector_rejected_source_count", -3))
            or int(source_join.get("verified_input_count", -1))
            < int(source_join.get("approved_input_count", 0))
            or int(source_join.get("validated_source_count", -1))
            != int(source_join.get("selected_memory_source_count", -2))
            + int(source_join.get("unselected_valid_source_count", -3))
            or int(source_join.get("selected_memory_source_count", -1))
            != len(records)
            or source_join.get("selected_memory_source_ids_sha256")
            != selected_source_hash
            or any(
                not str(source_join.get(field, ""))
                for field in (
                    "validated_source_ids_sha256",
                    "unselected_valid_source_ids_sha256",
                    "selector_rejected_source_ids_sha256",
                )
            )
        ):
            raise ValueError("V3.5 dual-key source-join contract drifted")
        split_audit = self.manifest.get("phase1_split_audit", {})
        if (
            split_audit.get("schema_version") != SPLIT_MANIFEST_SCHEMA
            or split_audit.get("manifest_logical_sha256")
            != inputs.get("split_manifest_logical_sha256")
            or split_audit.get("dataset_revision")
            != inputs.get("dataset_revision")
            or split_audit.get("overlap_check_verified") is not True
            or int(split_audit.get("joined_bank_source_member_count", -1))
            != len(records)
            or int(
                split_audit.get(
                    "authenticated_valid_source_member_count", -1
                )
            )
            != int(source_join.get("validated_source_count", -2))
            or split_audit.get("all_sources_match_authenticated_members") is not True
        ):
            raise ValueError("V3.5 Phase-1 split audit drifted")

    def _validate_tensors(self) -> None:
        expected_shape = self.manifest.get("embedding_shape")
        for owner, tensor in (
            ("applicability", self.applicability_embeddings),
            ("dynamic", self.dynamic_embeddings),
        ):
            if tensor.ndim != 2 or list(tensor.shape) != expected_shape:
                raise ValueError(f"V3.5 {owner} embedding shape differs from manifest")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"V3.5 {owner} embeddings contain non-finite values")
            norms = tensor.norm(dim=-1)
            if not torch.allclose(
                norms, torch.ones_like(norms), atol=1e-5, rtol=0.0
            ):
                raise ValueError(f"V3.5 {owner} embeddings are not L2 normalized")
        if self.applicability_embeddings.shape != self.dynamic_embeddings.shape:
            raise ValueError("V3.5 applicability and dynamic tensor shapes differ")
        for entry, applicability, dynamic in zip(
            self.entries,
            self.applicability_embeddings,
            self.dynamic_embeddings,
        ):
            required = (
                "memory_id",
                "source_experience_id",
                "payload_hash",
                "applicability_key_text_sha256",
                "applicability_key_token_ids_sha256",
                "dynamic_key_text_sha256",
                "dynamic_key_token_ids_sha256",
                "source_record_sha256",
                "phase1_provenance_sha256",
                "review_provenance_sha256",
                "review_validation_profile",
                "source_sample_id",
                "source_dataset_revision",
                "source_dataset_split",
                "source_logical_split",
                "source_question_sha256",
                "source_split_manifest_sha256",
                "split_member_sha256",
            )
            if any(not str(entry.get(field, "")) for field in required):
                raise ValueError("V3.5 dual-key record metadata is incomplete")
            if (
                entry.get("applicability_key_source")
                != V35_APPLICABILITY_KEY_SOURCE
                or entry.get("dynamic_key_source") != V35_DYNAMIC_KEY_SOURCE
                or int(entry.get("kv_layer", -1)) != 24
                or int(entry.get("payload_token_count", 0)) <= 0
                or int(entry.get("kv_valid_slot_count", 0)) <= 0
                or int(entry.get("applicability_key_token_count", 0)) <= 0
                or int(entry.get("dynamic_key_token_count", 0)) <= 0
                or int(entry.get("payload_token_count", -1))
                != int(entry.get("kv_valid_slot_count", -2))
                or entry.get("applicability_embedding_exact_reproduction") is not True
                or entry.get("source_dataset_revision")
                != self.input_artifacts.get("dataset_revision")
                or entry.get("source_dataset_split") != "train"
                or entry.get("source_logical_split") != "bank-source"
                or isinstance(entry.get("source_index"), bool)
                or not isinstance(entry.get("source_index"), int)
                or int(entry.get("source_index", -1)) < 0
                or entry.get("source_split_manifest_sha256")
                != self.input_artifacts.get("split_manifest_logical_sha256")
            ):
                raise ValueError("V3.5 dual-key per-record contract drifted")
            if (
                entry.get("applicability_key_embedding_sha256")
                != tensor_sha256(applicability)
                or entry.get("reproduced_applicability_key_embedding_sha256")
                != tensor_sha256(applicability)
                or entry.get("dynamic_key_embedding_sha256")
                != tensor_sha256(dynamic)
            ):
                raise ValueError("V3.5 per-record embedding hash mismatch")
            for field, tensor in (
                ("applicability_key_embedding_norm", applicability),
                ("dynamic_key_embedding_norm", dynamic),
            ):
                if not math.isclose(
                    float(entry.get(field, -1.0)),
                    float(tensor.norm().item()),
                    abs_tol=1e-5,
                    rel_tol=0.0,
                ):
                    raise ValueError("V3.5 per-record embedding norm mismatch")


@dataclass(frozen=True)
class QuestionOnlyQuery:
    """One pure question-only layer-24 query and its reproducibility trace."""

    text: str
    token_ids: tuple[int, ...]
    embedding: torch.Tensor
    side_kv_disabled: bool = True
    layer_number: int = 24
    pooling: str = "last_valid_token"
    normalization: str = "l2"
    schema_version: str = V35_STATIC_QUESTION_QUERY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "static_question_text_sha256": text_sha256(self.text),
            "static_question_token_count": len(self.token_ids),
            "static_question_token_ids_sha256": canonical_json_sha256(
                list(self.token_ids)
            ),
            "static_question_embedding_sha256": tensor_sha256(self.embedding),
            "static_question_embedding_norm": float(self.embedding.norm().item()),
            "static_question_token_ids": list(self.token_ids),
            "layer_number": self.layer_number,
            "representation": "decoder_layer_output",
            "pooling": self.pooling,
            "normalization": self.normalization,
            "side_kv_disabled": self.side_kv_disabled,
            "chat_wrapper_included": False,
            "prompt_boilerplate_included": False,
            "add_special_tokens": False,
        }


class QuestionOnlyEncoder:
    """Encode exactly ``question.strip()`` without prompt/chat wrappers."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: torch.device | str,
        layer_number: int = 24,
        controller: Any | None = None,
    ):
        if layer_number != 24:
            raise ValueError("V3.5 question-only encoding is frozen to layer 24")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_number = layer_number
        self.controller = controller

    @torch.inference_mode()
    def encode(self, question: str) -> QuestionOnlyQuery:
        if not isinstance(question, str):
            raise TypeError("V3.5 question-only query must be text")
        text = question.strip()
        if not text:
            raise ValueError("V3.5 question-only query is empty")
        token_ids = tuple(
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False)
        )
        if not token_ids:
            raise ValueError("V3.5 question-only tokenizer produced no tokens")
        context = (
            self.controller.suspend_memory()
            if self.controller is not None
            else nullcontext()
        )
        with context:
            embedding = encode_last_layer_token(
                model=self.model,
                token_ids=token_ids,
                layer_number=self.layer_number,
                device=self.device,
            ).cpu()
        return QuestionOnlyQuery(
            text=text,
            token_ids=token_ids,
            embedding=embedding,
            side_kv_disabled=True,
            layer_number=self.layer_number,
        )


@dataclass(frozen=True)
class V35MemoryChoice:
    """Selected side-KV metadata with an unmodified cosine audit score.

    The legacy ``MemoryChoice`` predates embedding retrieval and rejects
    non-positive scores.  V3.5 has no dynamic absolute-score gate, so an exact
    cosine top-1 may legitimately be negative while its calibrated margin
    passes.  This V3.5-specific duck type preserves that raw score.
    """

    memory_id: str
    payload_hash: str
    token_count: int
    kv_valid_slot_count: int
    retrieval_score: float
    retrieval_rank: int = 1

    def __post_init__(self) -> None:
        if not self.memory_id or not self.payload_hash:
            raise ValueError("V3.5 memory choice requires memory and payload IDs")
        if self.token_count <= 0 or self.kv_valid_slot_count <= 0:
            raise ValueError("V3.5 memory choice token/slot counts must be positive")
        if self.token_count != self.kv_valid_slot_count:
            raise ValueError("V3.5 memory choice payload tokens and KV slots differ")
        if self.retrieval_rank <= 0:
            raise ValueError("V3.5 retrieval rank must be positive")
        if (
            not math.isfinite(float(self.retrieval_score))
            or float(self.retrieval_score) < -1.00001
            or float(self.retrieval_score) > 1.00001
        ):
            raise ValueError("V3.5 retrieval score must be a finite raw cosine")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V35StaticShortlist:
    """Fixed per-question applicability shortlist used for every attempt."""

    query: Mapping[str, Any]
    score_floor: float
    shortlist_k: int
    pre_floor_top_k: tuple[Mapping[str, Any], ...]
    post_floor_shortlist: tuple[Mapping[str, Any], ...]
    unavailable_reason: str | None
    applicability_bank_manifest_sha256: str
    schema_version: str = V35_STATIC_SHORTLIST_SCHEMA

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(
            str(item["memory_id"]) for item in self.post_floor_shortlist
        )

    @property
    def shortlist_nonempty(self) -> bool:
        return bool(self.post_floor_shortlist)

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and len(self.memory_ids) >= 2

    @property
    def static_selector_unavailable(self) -> bool:
        return not self.available

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query": dict(self.query),
            "score_floor": self.score_floor,
            "score_floor_tie_policy": (
                "retain_score_greater_than_or_equal_to_floor"
            ),
            "shortlist_k": self.shortlist_k,
            "pre_floor_top_k": [dict(item) for item in self.pre_floor_top_k],
            "post_floor_shortlist": [
                dict(item) for item in self.post_floor_shortlist
            ],
            "shortlist_memory_ids": list(self.memory_ids),
            "shortlist_nonempty": self.shortlist_nonempty,
            "static_selector_unavailable": self.static_selector_unavailable,
            "unavailable_reason": self.unavailable_reason,
            "applicability_bank_manifest_sha256": (
                self.applicability_bank_manifest_sha256
            ),
            "shortlist_fixed_for_generation": True,
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
        }


class ApplicabilityAwareMemoryRetriever:
    """Question applicability shortlist followed by in-shortlist reranking."""

    def __init__(
        self,
        *,
        key_bank: DualRetrievalKeyBankLoader,
        records: Sequence[MemoryRecord],
        kv_valid_slot_counts: Mapping[str, int],
        question_encoder: QuestionOnlyEncoder,
        shortlist_k: int,
        applicability_score_floor: float,
        dynamic_min_top1_top2_margin: float | None,
        profile: Any | None = None,
    ):
        if isinstance(shortlist_k, bool) or not isinstance(shortlist_k, int):
            raise ValueError("V3.5 shortlist_k must be an integer")
        if not 1 <= int(shortlist_k) <= 32:
            raise ValueError("V3.5 shortlist_k must be in [1, 32]")
        if not math.isfinite(float(applicability_score_floor)) or not (
            -1.0 <= float(applicability_score_floor) <= 1.0
        ):
            raise ValueError("V3.5 applicability floor must be finite in [-1, 1]")
        if dynamic_min_top1_top2_margin is not None and (
            not math.isfinite(float(dynamic_min_top1_top2_margin))
            or float(dynamic_min_top1_top2_margin) < 0.0
        ):
            raise ValueError("V3.5 dynamic margin must be finite and non-negative")
        self.key_bank = key_bank
        self.question_encoder = question_encoder
        self.shortlist_k = int(shortlist_k)
        self.applicability_score_floor = float(applicability_score_floor)
        self.dynamic_min_top1_top2_margin = (
            None
            if dynamic_min_top1_top2_margin is None
            else float(dynamic_min_top1_top2_margin)
        )
        self.profile = profile
        if profile is None:
            self.dynamic_layer_number = 24
            self.dynamic_query_context = "question_plus_full_partial_cot"
            self.dynamic_query_encoder_state = (
                "pure_prefix_reencode_side_kv_disabled"
            )
            self.dynamic_query_pooling = V34_QUERY_POOLING_CURRENT_TOKEN
            self.dynamic_query_normalization = "l2"
            self.include_reproduction_token_ids = True
        else:
            self.dynamic_layer_number = int(getattr(profile, "layer_number", -1))
            self.dynamic_query_context = str(
                getattr(profile, "query_context", "")
            )
            self.dynamic_query_encoder_state = str(
                getattr(profile, "query_encoder_state", "")
            )
            self.dynamic_query_pooling = str(
                getattr(profile, "query_pooling", "")
            )
            self.dynamic_query_normalization = str(
                getattr(profile, "query_normalization", "")
            )
            self.include_reproduction_token_ids = bool(
                getattr(profile, "calibration_trace_only", False)
            )
        if (
            self.dynamic_layer_number != 24
            or self.dynamic_query_context != "question_plus_full_partial_cot"
            or self.dynamic_query_encoder_state
            != "pure_prefix_reencode_side_kv_disabled"
            or self.dynamic_query_pooling != V34_QUERY_POOLING_CURRENT_TOKEN
            or self.dynamic_query_normalization != "l2"
        ):
            raise ValueError("V3.5 dynamic query encoder/profile contract drifted")
        self.record_by_id = {record.memory_id: record for record in records}
        if len(self.record_by_id) != len(records):
            raise ValueError("V3.5 MemoryRecords contain duplicate IDs")
        self.kv_valid_slot_counts = {
            str(memory_id): int(count)
            for memory_id, count in kv_valid_slot_counts.items()
        }
        ids = set(key_bank.entry_by_id)
        if set(self.record_by_id) != ids or set(self.kv_valid_slot_counts) != ids:
            raise ValueError("V3.5 text/key/side-KV banks cover different IDs")
        for memory_id in sorted(ids):
            entry = key_bank.entry_by_id[memory_id]
            record = self.record_by_id[memory_id]
            if (
                entry.get("payload_hash") != record.payload_hash
                or int(entry.get("payload_token_count", -1)) != record.token_count
                or int(entry.get("kv_layer", -1)) != record.kv_layer
                or int(entry.get("kv_valid_slot_count", -1))
                != self.kv_valid_slot_counts[memory_id]
                or self.kv_valid_slot_counts[memory_id] <= 0
            ):
                raise ValueError(
                    f"V3.5 side-KV metadata differs for {memory_id}"
                )
        self.embedding_space_audit = {
            "schema_version": (
                "experience-memory-v3.5-dual-retrieval-space-audit-v1"
            ),
            "transform": "none",
            "applicability_key_embeddings_sha256": tensor_sha256(
                key_bank.applicability_embeddings
            ),
            "dynamic_key_embeddings_sha256": tensor_sha256(
                key_bank.dynamic_embeddings
            ),
            "applicability_key_embedding_norm_min": float(
                key_bank.applicability_embeddings.norm(dim=-1).min().item()
            ),
            "applicability_key_embedding_norm_max": float(
                key_bank.applicability_embeddings.norm(dim=-1).max().item()
            ),
            "dynamic_key_embedding_norm_min": float(
                key_bank.dynamic_embeddings.norm(dim=-1).min().item()
            ),
            "dynamic_key_embedding_norm_max": float(
                key_bank.dynamic_embeddings.norm(dim=-1).max().item()
            ),
            "dual_key_manifest_sha256": key_bank.manifest_sha256,
        }

    @torch.inference_mode()
    def prepare_question(self, question: str) -> V35StaticShortlist:
        encoded = self.question_encoder.encode(question)
        embeddings = self.key_bank.applicability_embeddings
        if embeddings.shape[0] == 0:
            return V35StaticShortlist(
                query=encoded.to_dict(),
                score_floor=self.applicability_score_floor,
                shortlist_k=self.shortlist_k,
                pre_floor_top_k=(),
                post_floor_shortlist=(),
                unavailable_reason="empty_bank",
                applicability_bank_manifest_sha256=self.key_bank.manifest_sha256,
            )
        query = self._normalize_query(
            encoded.embedding,
            expected_width=int(embeddings.shape[1]),
            owner="static question",
        )
        scores = torch.mv(embeddings, query)
        ranked_indices = sorted(
            range(int(scores.numel())),
            key=lambda index: (
                -float(scores[index].item()),
                str(self.key_bank.entries[index]["memory_id"]),
            ),
        )
        ranked_hits: list[dict[str, Any]] = []
        for global_rank, index in enumerate(ranked_indices, start=1):
            entry = self.key_bank.entries[index]
            ranked_hits.append({
                "memory_id": str(entry["memory_id"]),
                "payload_hash": str(entry["payload_hash"]),
                "static_score": float(scores[index].item()),
                "original_global_rank": global_rank,
                "applicability_key_embedding_sha256": str(
                    entry["applicability_key_embedding_sha256"]
                ),
            })
        pre_floor = tuple(ranked_hits[: self.shortlist_k])
        eligible = [
            hit
            for hit in ranked_hits
            if float(hit["static_score"]) >= self.applicability_score_floor
        ]
        shortlist = tuple(eligible[: self.shortlist_k])
        if len(shortlist) >= 2:
            unavailable_reason = None
        elif not ranked_hits:
            unavailable_reason = "empty_bank"
        elif not shortlist:
            unavailable_reason = "below_applicability_floor"
        else:
            unavailable_reason = "insufficient_shortlist"
        query_audit = encoded.to_dict()
        if not self.include_reproduction_token_ids:
            query_audit.pop("static_question_token_ids", None)
        query_audit["static_question_embedding_sha256"] = tensor_sha256(query)
        query_audit["static_question_embedding_norm"] = float(query.norm().item())
        return V35StaticShortlist(
            query=query_audit,
            score_floor=self.applicability_score_floor,
            shortlist_k=self.shortlist_k,
            pre_floor_top_k=pre_floor,
            post_floor_shortlist=shortlist,
            unavailable_reason=unavailable_reason,
            applicability_bank_manifest_sha256=self.key_bank.manifest_sha256,
        )

    # Adapter name retained for callers that make the two-stage operation
    # explicit.  It is semantically identical to ``prepare_question``.
    prepare_static_shortlist = prepare_question

    @torch.inference_mode()
    def retrieve(
        self,
        *,
        query_embedding: torch.Tensor,
        query_token_ids: Sequence[int],
        prompt_token_count: int,
        static_context: V35StaticShortlist,
    ) -> ApplicabilityAwareRetrievalDecision:
        self._validate_static_context(static_context)
        token_ids = [int(value) for value in query_token_ids]
        if (
            not token_ids
            or isinstance(prompt_token_count, bool)
            or not isinstance(prompt_token_count, int)
            or prompt_token_count <= 0
            or prompt_token_count >= len(token_ids)
        ):
            raise ValueError(
                "V3.5 dynamic query must contain prompt plus full partial CoT"
            )
        query = self._normalize_query(
            query_embedding,
            expected_width=int(self.key_bank.dynamic_embeddings.shape[1]),
            owner="dynamic",
        )
        if not static_context.available:
            status = static_context.unavailable_reason or "static_shortlist_unavailable"
            query = self._dynamic_query_audit(
                query_embedding=query,
                query_token_ids=token_ids,
                prompt_token_count=prompt_token_count,
                hits=(),
                static_context=static_context,
                status=status,
            )
            return ApplicabilityAwareRetrievalDecision(
                status=status,
                query=query,
                hits=(),
                matched_memory=None,
                static_shortlist=static_context.post_floor_shortlist,
            )

        shortlist_ids = static_context.memory_ids
        shortlist_indices = [
            self.key_bank.index_by_id[memory_id] for memory_id in shortlist_ids
        ]
        shortlist_embeddings = self.key_bank.dynamic_embeddings[shortlist_indices]
        scores = torch.mv(shortlist_embeddings, query)
        ranked_local_indices = sorted(
            range(len(shortlist_indices)),
            key=lambda local_index: (
                -float(scores[local_index].item()),
                shortlist_ids[local_index],
            ),
        )
        static_by_id = {
            str(hit["memory_id"]): hit
            for hit in static_context.post_floor_shortlist
        }
        hits: list[dict[str, Any]] = []
        for dynamic_rank, local_index in enumerate(
            ranked_local_indices[:2], start=1
        ):
            bank_index = shortlist_indices[local_index]
            entry = self.key_bank.entries[bank_index]
            memory_id = str(entry["memory_id"])
            static_hit = static_by_id[memory_id]
            hits.append({
                "memory_id": memory_id,
                "payload_hash": str(entry["payload_hash"]),
                "payload_token_count": int(entry["payload_token_count"]),
                "kv_layer": int(entry["kv_layer"]),
                "kv_valid_slot_count": int(entry["kv_valid_slot_count"]),
                "score": float(scores[local_index].item()),
                "dynamic_score": float(scores[local_index].item()),
                "dynamic_rank": dynamic_rank,
                "static_score": float(static_hit["static_score"]),
                "static_global_rank": int(static_hit["original_global_rank"]),
                "dynamic_key_embedding_sha256": str(
                    entry["dynamic_key_embedding_sha256"]
                ),
            })
        if len(hits) < 2:
            status = "insufficient_shortlist"
        else:
            margin = float(hits[0]["score"]) - float(hits[1]["score"])
            if self.dynamic_min_top1_top2_margin is not None and (
                margin < self.dynamic_min_top1_top2_margin
            ):
                status = "below_dynamic_margin"
            elif float(hits[0]["static_score"]) < self.applicability_score_floor:
                status = "below_applicability_floor"
            else:
                status = "selected"
        query_audit = self._dynamic_query_audit(
            query_embedding=query,
            query_token_ids=token_ids,
            prompt_token_count=prompt_token_count,
            hits=hits,
            static_context=static_context,
            status=status,
        )
        matched: V35MemoryChoice | None = None
        if status == "selected":
            top = hits[0]
            memory_id = str(top["memory_id"])
            if memory_id not in shortlist_ids:
                raise RuntimeError("V3.5 selected memory escaped static shortlist")
            record = self.record_by_id[memory_id]
            matched = V35MemoryChoice(
                memory_id=memory_id,
                payload_hash=record.payload_hash,
                token_count=record.token_count,
                kv_valid_slot_count=self.kv_valid_slot_counts[memory_id],
                retrieval_score=float(top["score"]),
                retrieval_rank=1,
            )
        return ApplicabilityAwareRetrievalDecision(
            status=status,
            query=query_audit,
            hits=tuple(hits),
            matched_memory=matched,
            static_shortlist=static_context.post_floor_shortlist,
        )

    def _validate_static_context(self, value: V35StaticShortlist) -> None:
        if not isinstance(value, V35StaticShortlist):
            raise TypeError("V3.5 retrieve requires a V35StaticShortlist")
        if value.applicability_bank_manifest_sha256 != self.key_bank.manifest_sha256:
            raise ValueError("V3.5 static shortlist belongs to a different bank")
        if value.shortlist_k != self.shortlist_k or not math.isclose(
            value.score_floor,
            self.applicability_score_floor,
            abs_tol=0.0,
            rel_tol=0.0,
        ):
            raise ValueError("V3.5 static shortlist calibration drifted")
        if len(set(value.memory_ids)) != len(value.memory_ids):
            raise ValueError("V3.5 static shortlist contains duplicate IDs")
        if any(memory_id not in self.key_bank.entry_by_id for memory_id in value.memory_ids):
            raise ValueError("V3.5 static shortlist contains an unknown memory")
        if any(
            float(hit["static_score"]) < self.applicability_score_floor
            for hit in value.post_floor_shortlist
        ):
            raise ValueError("V3.5 static shortlist contains a below-floor candidate")

    @staticmethod
    def _normalize_query(
        value: torch.Tensor, *, expected_width: int, owner: str
    ) -> torch.Tensor:
        return canonicalize_v35_query_embedding(
            value, expected_width=expected_width, owner=owner
        )

    @staticmethod
    def _validate_canonical_query(
        value: torch.Tensor, *, expected_width: int, owner: str
    ) -> torch.Tensor:
        query = (
            value.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
            .contiguous()
        )
        if query.shape != (expected_width,):
            raise ValueError(f"V3.5 {owner} query/key embedding widths differ")
        norm = float(query.norm().item())
        if (
            not torch.isfinite(query).all()
            or not math.isfinite(norm)
            or not math.isclose(
                norm, 1.0, rel_tol=0.0, abs_tol=_V35_UNIT_NORM_ATOL
            )
        ):
            raise ValueError(f"V3.5 {owner} canonical query is not unit norm")
        return query

    def _dynamic_query_audit(
        self,
        *,
        query_embedding: torch.Tensor,
        query_token_ids: Sequence[int],
        prompt_token_count: int,
        hits: Sequence[Mapping[str, Any]],
        static_context: V35StaticShortlist,
        status: str,
    ) -> dict[str, Any]:
        canonical_query = self._validate_canonical_query(
            query_embedding,
            expected_width=int(self.key_bank.dynamic_embeddings.shape[1]),
            owner="dynamic",
        )
        top1 = float(hits[0]["score"]) if hits else None
        top2 = float(hits[1]["score"]) if len(hits) > 1 else None
        margin = top1 - top2 if top1 is not None and top2 is not None else None
        selected_static_score = (
            float(hits[0]["static_score"]) if hits else None
        )
        static_passed = (
            selected_static_score is not None
            and selected_static_score >= self.applicability_score_floor
        )
        margin_passed = (
            margin is not None
            and (
                self.dynamic_min_top1_top2_margin is None
                or margin >= self.dynamic_min_top1_top2_margin
            )
        )
        embedding_token_index = query_embedding_token_index(
            token_count=len(query_token_ids),
            pooling=self.dynamic_query_pooling,
        )
        audit = {
            "method": "exact_cosine_within_static_applicability_shortlist",
            "context": self.dynamic_query_context,
            "encoder_state": self.dynamic_query_encoder_state,
            "pooling": self.dynamic_query_pooling,
            "normalization": self.dynamic_query_normalization,
            "layer_number": self.dynamic_layer_number,
            "side_kv_disabled": True,
            "query_token_count": len(query_token_ids),
            "prompt_token_count": prompt_token_count,
            "partial_cot_token_count": len(query_token_ids) - prompt_token_count,
            "encoded_full_prefix_token_count": len(query_token_ids),
            "query_token_ids_sha256": canonical_json_sha256(
                list(query_token_ids)
            ),
            "query_embedding_token_index": embedding_token_index,
            "query_embedding_token_id": int(query_token_ids[embedding_token_index]),
            "query_embedding_causal_context_token_count": embedding_token_index + 1,
            "query_embedding_sha256": tensor_sha256(canonical_query),
            "query_embedding_norm": float(canonical_query.norm().item()),
            "static_shortlist_ids": list(static_context.memory_ids),
            "static_shortlist_memory_ids": list(static_context.memory_ids),
            "static_shortlist_size": len(static_context.memory_ids),
            "static_shortlist_fixed_for_generation": True,
            "dynamic_search_candidate_count": len(static_context.memory_ids),
            "dynamic_search_restricted_to_static_shortlist": True,
            "top1_score": top1,
            "top2_score": top2,
            "top1_top2_margin": margin,
            "minimum_top1_top2_margin": self.dynamic_min_top1_top2_margin,
            "selected_memory_static_score": selected_static_score,
            "minimum_applicability_score": self.applicability_score_floor,
            "static_condition_passed": static_passed,
            "dynamic_margin_condition_passed": margin_passed,
            "joint_admission_passed": status == "selected",
            "selected_memory_kv_metadata_aligned": (
                status != "selected"
                or (
                    bool(hits)
                    and int(hits[0]["kv_layer"]) == 24
                    and int(hits[0]["kv_valid_slot_count"])
                    == self.kv_valid_slot_counts[str(hits[0]["memory_id"])]
                    and str(hits[0]["payload_hash"])
                    == self.record_by_id[str(hits[0]["memory_id"])].payload_hash
                )
            ),
            "decision_reason": status,
        }
        if self.include_reproduction_token_ids:
            audit["query_token_ids"] = list(query_token_ids)
        return audit


# Short aliases make integration with older naming in experiment scripts less
# brittle while keeping the explicit public class names above canonical.
DualKeyRetrievalBankLoader = DualRetrievalKeyBankLoader
ApplicabilityAwareRetriever = ApplicabilityAwareMemoryRetriever


__all__ = [
    "ApplicabilityAwareMemoryRetriever",
    "ApplicabilityAwareRetriever",
    "CompiledDualRetrievalKeyBank",
    "DualKeyRetrievalBankLoader",
    "DualRetrievalKeyBankLoader",
    "DualRetrievalKeyCompiler",
    "DualRetrievalKeyCompilerConfig",
    "QuestionOnlyEncoder",
    "QuestionOnlyQuery",
    "V35_IMPLEMENTATION_PATHS",
    "V35_APPLICABILITY_KEY_SOURCE",
    "V35_APPLICABILITY_TENSOR_NAME",
    "V35_DUAL_KEY_MANIFEST_FILE",
    "V35_DUAL_KEY_TENSOR_FILE",
    "V35_DYNAMIC_KEY_SOURCE",
    "V35_DYNAMIC_TENSOR_NAME",
    "V35MemoryChoice",
    "V35StaticShortlist",
    "canonicalize_v35_query_embedding",
    "v35_implementation_files_sha256",
    "validate_v35_dynamic_text_component",
    "validate_v35_split_manifest",
]
