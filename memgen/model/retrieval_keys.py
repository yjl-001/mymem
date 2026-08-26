"""Layer-24 embedding keys and exact-cosine retrieval for MemGen V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from memgen.experience.e1 import MemoryChoice
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import canonical_json_sha256, file_sha256, text_sha256
from memgen.experience.v3 import EmbeddingRetrievalDecision, ExperienceMemoryV3Profile
from memgen.model.side_kv import DecoderLayerResolver, SIDE_KV_ATTENTION_BACKEND


RETRIEVAL_KEY_BANK_SCHEMA = "experience-memory-retrieval-key-bank-v1"
RETRIEVAL_KEY_TENSOR_NAME = "retrieval_key_embeddings"


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor values using a stable CPU float32 representation."""

    normalized = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_json_sha256(list(normalized.shape)).encode("ascii"))
    digest.update(normalized.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalKeyCompilerConfig:
    """Frozen offline/online encoder contract for retrieval keys."""

    layer_number: int = 24
    hidden_state_tuple_index: int = 24
    representation: str = "decoder_layer_output"
    key_source: str = "sanitized_fields.when_facing"
    pooling: str = "last_valid_token"
    normalization: str = "l2"
    attention_backend: str = SIDE_KV_ATTENTION_BACKEND

    def __post_init__(self) -> None:
        if self.layer_number != 24:
            raise ValueError("The current V3 retrieval key is frozen to layer 24")
        if (
            self.hidden_state_tuple_index != self.layer_number
            or self.representation != "decoder_layer_output"
        ):
            raise ValueError("V3 retrieval representation must be layer-24 output")
        if self.key_source != "sanitized_fields.when_facing":
            raise ValueError("V3 key source must be sanitized when_facing")
        if self.pooling != "last_valid_token" or self.normalization != "l2":
            raise ValueError("V3 retrieval uses last-token pooling and L2 normalization")
        if self.attention_backend != SIDE_KV_ATTENTION_BACKEND:
            raise ValueError("V3 retrieval key compilation requires SDPA")


@dataclass(frozen=True)
class CompiledRetrievalKeyBank:
    embeddings: torch.Tensor
    manifest: Mapping[str, Any]

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "retrieval_key_bank.safetensors"
        manifest_path = output_dir / "retrieval_key_manifest.json"
        save_file(
            {RETRIEVAL_KEY_TENSOR_NAME: self.embeddings.contiguous()},
            str(tensor_path),
            metadata={"schema_version": RETRIEVAL_KEY_BANK_SCHEMA},
        )
        manifest = dict(self.manifest)
        manifest["tensor_artifact"] = {
            "path": tensor_path.name,
            "sha256": file_sha256(tensor_path),
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return tensor_path, manifest_path


class RetrievalKeyCompiler:
    """Encode ``when_facing`` with the same frozen reasoner used online."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        reasoner_name: str,
        reasoner_revision: str,
        tokenizer_revision: str,
        config: RetrievalKeyCompilerConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reasoner_name = reasoner_name
        self.reasoner_revision = reasoner_revision
        self.tokenizer_revision = tokenizer_revision
        self.config = config or RetrievalKeyCompilerConfig()
        layers = DecoderLayerResolver.resolve(model)
        if self.config.layer_number > len(layers):
            raise ValueError("V3 retrieval layer exceeds the reasoner depth")

    @torch.inference_mode()
    def compile(self, records: Sequence[MemoryRecord]) -> CompiledRetrievalKeyBank:
        if not records:
            raise ValueError("Cannot compile an empty V3 retrieval key bank")
        self._validate_records(records)
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            compiled = [self._compile_one(record) for record in records]
        finally:
            self.model.train(was_training)
        embeddings = torch.stack([item[0] for item in compiled], dim=0).float()
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Compiled V3 retrieval keys contain non-finite values")
        norms = embeddings.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
            raise RuntimeError("Compiled V3 retrieval keys are not unit normalized")
        memory_ids = [record.memory_id for record in records]
        entries = []
        for index, (record, (embedding, key_text, token_ids)) in enumerate(
            zip(records, compiled)
        ):
            entries.append({
                "index": index,
                "memory_id": record.memory_id,
                "payload_hash": record.payload_hash,
                "payload_token_count": record.token_count,
                "key_source": self.config.key_source,
                "key_text_sha256": text_sha256(key_text),
                "key_token_count": len(token_ids),
                "key_token_ids_sha256": canonical_json_sha256(token_ids),
                "key_embedding_sha256": tensor_sha256(embedding),
                "key_embedding_norm": float(embedding.norm().item()),
            })
        manifest = {
            "schema_version": RETRIEVAL_KEY_BANK_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reasoner": {
                "model_name": self.reasoner_name,
                "model_revision": self.reasoner_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "attention_implementation": self.config.attention_backend,
            },
            "compiler": asdict(self.config),
            "record_count": len(records),
            "record_order_sha256": canonical_json_sha256(memory_ids),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "records": entries,
        }
        return CompiledRetrievalKeyBank(embeddings=embeddings.cpu(), manifest=manifest)

    def _validate_records(self, records: Sequence[MemoryRecord]) -> None:
        memory_ids = [record.memory_id for record in records]
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("V3 retrieval records contain duplicate memory IDs")
        for record in records:
            if record.kv_layer != self.config.layer_number:
                raise ValueError("MemoryRecord layer differs from V3 retrieval layer")
            if (
                record.reasoner_name != self.reasoner_name
                or record.reasoner_revision != self.reasoner_revision
                or record.tokenizer_revision != self.tokenizer_revision
            ):
                raise ValueError("MemoryRecord reasoner provenance differs from compiler")
            if not str(record.sanitized_fields.get("when_facing", "")).strip():
                raise ValueError(f"{record.memory_id} has an empty when_facing key")

    def _compile_one(
        self, record: MemoryRecord
    ) -> tuple[torch.Tensor, str, list[int]]:
        key_text = str(record.sanitized_fields["when_facing"]).strip()
        token_ids = list(self.tokenizer.encode(key_text, add_special_tokens=False))
        if not token_ids:
            raise ValueError(f"{record.memory_id} produced an empty key token sequence")
        vector = encode_last_layer_token(
            model=self.model,
            token_ids=token_ids,
            layer_number=self.config.layer_number,
            device=next(self.model.parameters()).device,
        )
        return vector.cpu(), key_text, token_ids


@torch.inference_mode()
def encode_last_layer_token(
    *,
    model: Any,
    token_ids: Sequence[int],
    layer_number: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Pure-prefix re-encode and return a normalized layer state."""

    if not token_ids:
        raise ValueError("V3 query encoding requires a non-empty token sequence")
    inputs = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    output = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None or layer_number >= len(hidden_states):
        raise RuntimeError("Requested V3 retrieval hidden-state layer is unavailable")
    vector = hidden_states[layer_number][0, -1, :].detach().float()
    if not torch.isfinite(vector).all() or float(vector.norm().item()) <= 0.0:
        raise RuntimeError("V3 retrieval encoder produced an invalid vector")
    return F.normalize(vector, dim=0)


class FullPrefixQueryEncoder:
    """Encode question + every generated partial-CoT token from scratch."""

    def __init__(self, *, model: Any, device: str, layer_number: int = 24):
        if layer_number != 24:
            raise ValueError("The current V3 query encoder is frozen to layer 24")
        self.model = model
        self.device = device
        self.layer_number = layer_number

    @torch.inference_mode()
    def encode(self, token_ids: Sequence[int]) -> torch.Tensor:
        return encode_last_layer_token(
            model=self.model,
            token_ids=token_ids,
            layer_number=self.layer_number,
            device=self.device,
        )


class RetrievalKeyBankLoader:
    """Content-addressed loader for a V3 embedding key bank."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_reasoner_name: str | None = None,
        expected_reasoner_revision: str | None = None,
        expected_tokenizer_revision: str | None = None,
    ):
        from safetensors.torch import load_file

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_reasoner_name=expected_reasoner_name,
            expected_reasoner_revision=expected_reasoner_revision,
            expected_tokenizer_revision=expected_tokenizer_revision,
        )
        tensor_info = self.manifest.get("tensor_artifact", {})
        tensor_path = manifest_path.parent / str(tensor_info.get("path", ""))
        if not tensor_path.is_file() or file_sha256(tensor_path) != tensor_info.get("sha256"):
            raise ValueError("V3 retrieval tensor is missing or has a hash mismatch")
        tensors = load_file(str(tensor_path), device="cpu")
        self.embeddings = tensors[RETRIEVAL_KEY_TENSOR_NAME].float()
        self._validate_tensors()
        self.entries = tuple(self.manifest["records"])
        self.entry_by_id = {str(entry["memory_id"]): entry for entry in self.entries}

    def _validate_manifest(
        self,
        *,
        expected_reasoner_name: str | None,
        expected_reasoner_revision: str | None,
        expected_tokenizer_revision: str | None,
    ) -> None:
        if self.manifest.get("schema_version") != RETRIEVAL_KEY_BANK_SCHEMA:
            raise ValueError("Unexpected V3 retrieval key manifest schema")
        expected_hash = self.manifest.get("manifest_sha256")
        actual_hash = canonical_json_sha256({
            key: value
            for key, value in self.manifest.items()
            if key != "manifest_sha256"
        })
        if expected_hash != actual_hash:
            raise ValueError("V3 retrieval key manifest hash mismatch")
        config = RetrievalKeyCompilerConfig(**self.manifest.get("compiler", {}))
        reasoner = self.manifest.get("reasoner", {})
        expected = (
            ("model_name", expected_reasoner_name),
            ("model_revision", expected_reasoner_revision),
            ("tokenizer_revision", expected_tokenizer_revision),
        )
        for field_name, expected_value in expected:
            if expected_value is not None and reasoner.get(field_name) != expected_value:
                raise ValueError(f"V3 retrieval reasoner {field_name} mismatch")
        if reasoner.get("attention_implementation") != config.attention_backend:
            raise ValueError("V3 retrieval attention backend drifted")
        records = self.manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("V3 retrieval key manifest has no records")
        if self.manifest.get("record_count") != len(records):
            raise ValueError("V3 retrieval key record count mismatch")
        memory_ids = [str(entry.get("memory_id", "")) for entry in records]
        if any(not value for value in memory_ids) or len(set(memory_ids)) != len(memory_ids):
            raise ValueError("V3 retrieval memory IDs are missing or duplicated")
        if [int(entry.get("index", -1)) for entry in records] != list(range(len(records))):
            raise ValueError("V3 retrieval record indices are not contiguous")
        if self.manifest.get("record_order_sha256") != canonical_json_sha256(memory_ids):
            raise ValueError("V3 retrieval record order hash mismatch")

    def _validate_tensors(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("V3 retrieval embeddings must use [record, hidden]")
        if list(self.embeddings.shape) != self.manifest.get("embedding_shape"):
            raise ValueError("V3 retrieval embedding shape differs from manifest")
        if self.manifest.get("embedding_dtype") != "torch.float32":
            raise ValueError("V3 retrieval embeddings must be stored as float32")
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("V3 retrieval embeddings contain non-finite values")
        norms = self.embeddings.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
            raise ValueError("V3 retrieval embeddings are not L2 normalized")
        for entry, embedding in zip(self.manifest["records"], self.embeddings):
            if entry.get("key_embedding_sha256") != tensor_sha256(embedding):
                raise ValueError("V3 retrieval per-record embedding hash mismatch")
            if not math.isclose(
                float(entry.get("key_embedding_norm", -1.0)),
                float(embedding.norm().item()),
                abs_tol=1e-5,
                rel_tol=0.0,
            ):
                raise ValueError("V3 retrieval per-record embedding norm mismatch")


class EmbeddingMemoryRetriever:
    """Exact top-k cosine selector over compiled embedding keys."""

    def __init__(
        self,
        *,
        key_bank: RetrievalKeyBankLoader,
        records: Sequence[MemoryRecord],
        kv_valid_slot_counts: Mapping[str, int],
        profile: ExperienceMemoryV3Profile,
    ):
        self.key_bank = key_bank
        self.profile = profile
        self.record_by_id = {record.memory_id: record for record in records}
        self.kv_valid_slot_counts = {
            str(memory_id): int(count)
            for memory_id, count in kv_valid_slot_counts.items()
        }
        key_ids = set(key_bank.entry_by_id)
        if set(self.record_by_id) != key_ids or set(self.kv_valid_slot_counts) != key_ids:
            raise ValueError("V3 text, embedding, and side-KV banks cover different IDs")
        if (
            profile.retrieval_abstention_policy == "top1_top2_margin"
            and len(key_ids) < 2
        ):
            raise ValueError("V3 margin abstention requires at least two memories")
        for memory_id, record in self.record_by_id.items():
            entry = key_bank.entry_by_id[memory_id]
            if entry.get("payload_hash") != record.payload_hash:
                raise ValueError("V3 retrieval key and payload hashes differ")
            if self.kv_valid_slot_counts[memory_id] <= 0:
                raise ValueError("V3 side-KV slot counts must be positive")

    @torch.inference_mode()
    def retrieve(
        self,
        *,
        query_embedding: torch.Tensor,
        query_token_ids: Sequence[int],
        prompt_token_count: int,
    ) -> EmbeddingRetrievalDecision:
        query = query_embedding.detach().float().cpu().reshape(-1)
        if query.shape[0] != self.key_bank.embeddings.shape[1]:
            raise ValueError("V3 query and key embedding widths differ")
        if not torch.isfinite(query).all() or float(query.norm().item()) <= 0.0:
            raise ValueError("V3 query embedding is invalid")
        query = F.normalize(query, dim=0)
        scores = torch.mv(self.key_bank.embeddings, query)
        top_k = min(self.profile.retrieval_top_k, int(scores.numel()))
        if top_k == 0:
            return EmbeddingRetrievalDecision(
                status="empty_bank",
                query=self._query_audit(query, query_token_ids, prompt_token_count, ()),
                hits=(),
                matched_memory=None,
            )
        # Bank order is the stable tie breaker; torch.topk does not guarantee a
        # cross-platform order for equal scores.
        ranked_indices = sorted(
            range(int(scores.numel())),
            key=lambda index: (-float(scores[index].item()), index),
        )[:top_k]
        hits = []
        for rank, index in enumerate(ranked_indices, start=1):
            entry = self.key_bank.entries[int(index)]
            hits.append({
                "memory_id": str(entry["memory_id"]),
                "payload_hash": str(entry["payload_hash"]),
                "payload_token_count": int(entry["payload_token_count"]),
                "score": float(scores[index].item()),
                "rank": rank,
                "key_embedding_sha256": str(entry["key_embedding_sha256"]),
            })
        top = hits[0]
        memory_id = str(top["memory_id"])
        record = self.record_by_id[memory_id]
        query_audit = self._query_audit(
            query, query_token_ids, prompt_token_count, hits
        )
        if (
            self.profile.retrieval_abstention_policy == "top1_top2_margin"
            and float(query_audit["top1_top2_margin"])
            < float(self.profile.retrieval_min_top1_top2_margin)
        ):
            return EmbeddingRetrievalDecision(
                status="below_margin",
                query=query_audit,
                hits=tuple(hits),
                matched_memory=None,
            )
        return EmbeddingRetrievalDecision(
            status="selected",
            query=query_audit,
            hits=tuple(hits),
            matched_memory=MemoryChoice(
                memory_id=memory_id,
                payload_hash=record.payload_hash,
                token_count=record.token_count,
                kv_valid_slot_count=self.kv_valid_slot_counts[memory_id],
                retrieval_score=float(top["score"]),
                retrieval_rank=1,
            ),
        )

    def _query_audit(
        self,
        query: torch.Tensor,
        token_ids: Sequence[int],
        prompt_token_count: int,
        hits: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        top1 = float(hits[0]["score"]) if hits else None
        top2 = float(hits[1]["score"]) if len(hits) > 1 else None
        return {
            "method": self.profile.retrieval_method,
            "context": self.profile.query_context,
            "encoder_state": self.profile.query_encoder_state,
            "pooling": self.profile.query_pooling,
            "normalization": self.profile.query_normalization,
            "abstention_policy": self.profile.retrieval_abstention_policy,
            "minimum_top1_top2_margin": (
                self.profile.retrieval_min_top1_top2_margin
            ),
            "query_token_count": len(token_ids),
            "prompt_token_count": prompt_token_count,
            "partial_cot_token_count": len(token_ids) - prompt_token_count,
            "query_token_ids_sha256": canonical_json_sha256(list(token_ids)),
            "query_embedding_sha256": tensor_sha256(query),
            "query_embedding_norm": float(query.norm().item()),
            "top_k_requested": self.profile.retrieval_top_k,
            "top1_score": top1,
            "top2_score": top2,
            "top1_top2_margin": top1 - top2 if top2 is not None else None,
            "margin_qualified": (
                None
                if self.profile.retrieval_abstention_policy == "disabled"
                else (
                    top2 is not None
                    and top1 - top2
                    >= float(self.profile.retrieval_min_top1_top2_margin)
                )
            ),
        }
