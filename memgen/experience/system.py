"""Pure contracts for the end-to-end experience-memory system.

The module owns the versioned runtime profile and semantic retrieval decision.
It deliberately has no Torch dependency; gate execution and side-KV generation
live in :mod:`memgen.model.experience_system`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from memgen.experience.e1 import MemoryChoice
from memgen.experience.memory import MemoryRecordRejected
from memgen.experience.retrieval import (
    BM25Hit,
    BM25MemoryIndex,
    RetrievalQueryBuilder,
)


EXPERIENCE_MEMORY_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v1"
SEMANTIC_RETRIEVAL_DECISION_SCHEMA = "semantic-memory-retrieval-decision-v1"


@dataclass(frozen=True)
class ExperienceMemorySystemProfile:
    """Frozen reference configuration for the complete online system."""

    layer_number: int = 24
    partial_cot_window_tokens: int = 96
    retrieval_top_k: int = 2
    selected_memory_count: int = 1
    memory_score_normalization: str = "log_valid_slots"
    memory_score_bias: float = math.log(10.0)
    gate_policy: str = "first_joint_entropy_and_risk_boundary"
    injection_policy: str = "persistent_from_trigger_through_eos"
    schema_version: str = EXPERIENCE_MEMORY_SYSTEM_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIENCE_MEMORY_SYSTEM_PROFILE_SCHEMA:
            raise ValueError("Unexpected experience-memory system profile schema")
        if self.layer_number <= 0 or self.partial_cot_window_tokens <= 0:
            raise ValueError("System layer and partial-CoT window must be positive")
        if self.retrieval_top_k < 2:
            raise ValueError("System retrieval must preserve top-2 diagnostics")
        if self.selected_memory_count != 1:
            raise ValueError("Reference system currently supports exactly one memory")
        if self.memory_score_normalization != "log_valid_slots":
            raise ValueError("Reference system requires log_valid_slots normalization")
        if not math.isfinite(self.memory_score_bias):
            raise ValueError("System memory score bias must be finite")
        if self.gate_policy != "first_joint_entropy_and_risk_boundary":
            raise ValueError("Unexpected system gate policy")
        if self.injection_policy != "persistent_from_trigger_through_eos":
            raise ValueError("Unexpected system injection policy")

    @property
    def memory_odds_multiplier(self) -> float:
        return math.exp(self.memory_score_bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "memory_odds_multiplier": self.memory_odds_multiplier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceMemorySystemProfile":
        if value.get("schema_version") != EXPERIENCE_MEMORY_SYSTEM_PROFILE_SCHEMA:
            raise ValueError("Missing or unexpected system profile schema")
        data = dict(value)
        multiplier = data.pop("memory_odds_multiplier", None)
        profile = cls(**data)
        if multiplier is not None and not math.isclose(
            float(multiplier), profile.memory_odds_multiplier, rel_tol=1e-12
        ):
            raise ValueError("System profile memory odds multiplier drifted")
        return profile


@dataclass(frozen=True)
class SemanticRetrievalDecision:
    """Auditable output of one question + partial-CoT BM25 request."""

    status: str
    query: Mapping[str, Any] | None
    hits: tuple[Mapping[str, Any], ...]
    matched_memory: MemoryChoice | None
    rejection_reasons: tuple[str, ...] = ()
    schema_version: str = SEMANTIC_RETRIEVAL_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_RETRIEVAL_DECISION_SCHEMA:
            raise ValueError("Unexpected semantic retrieval decision schema")
        allowed = {"selected", "no_positive_hit", "query_rejected"}
        if self.status not in allowed:
            raise ValueError(f"Unexpected semantic retrieval status: {self.status}")
        if self.status == "selected":
            if self.matched_memory is None or not self.hits or self.query is None:
                raise ValueError("Selected retrieval requires query, hits, and memory")
            if self.hits[0].get("memory_id") != self.matched_memory.memory_id:
                raise ValueError("Top retrieval hit and selected memory differ")
        elif self.matched_memory is not None:
            raise ValueError("Abstaining retrieval cannot carry a selected memory")
        if self.status == "query_rejected" and not self.rejection_reasons:
            raise ValueError("Rejected retrieval requires stable reason codes")

    @property
    def selected(self) -> bool:
        return self.matched_memory is not None

    @property
    def abstain_reason(self) -> str | None:
        return None if self.selected else f"retrieval_{self.status}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "query": dict(self.query) if self.query is not None else None,
            "hits": [dict(hit) for hit in self.hits],
            "matched_memory": (
                self.matched_memory.to_dict()
                if self.matched_memory is not None
                else None
            ),
            "rejection_reasons": list(self.rejection_reasons),
        }


class SemanticMemoryRetriever:
    """Select one compiled memory using answer-blind lexical semantics."""

    def __init__(
        self,
        *,
        index: BM25MemoryIndex,
        query_builder: RetrievalQueryBuilder,
        kv_valid_slot_counts: Mapping[str, int],
        profile: ExperienceMemorySystemProfile,
    ):
        self.index = index
        self.query_builder = query_builder
        self.kv_valid_slot_counts = {
            str(memory_id): int(count)
            for memory_id, count in kv_valid_slot_counts.items()
        }
        self.profile = profile
        record_ids = {record.memory_id for record in index.records}
        if set(self.kv_valid_slot_counts) != record_ids:
            raise ValueError("BM25 and side-KV banks cover different memory IDs")
        if any(count <= 0 for count in self.kv_valid_slot_counts.values()):
            raise ValueError("Semantic retriever received a non-positive KV slot count")
        if (
            query_builder.config.partial_cot_window_tokens
            != profile.partial_cot_window_tokens
        ):
            raise ValueError("Query builder and system partial-CoT windows differ")

    @staticmethod
    def _hit_dict(hit: BM25Hit) -> dict[str, Any]:
        return {
            "memory_id": hit.memory_id,
            "payload_hash": hit.payload_hash,
            "payload_token_count": hit.token_count,
            "score": hit.score,
            "rank": hit.rank,
        }

    def retrieve(
        self,
        *,
        question: str,
        partial_cot_token_ids: Sequence[int],
    ) -> SemanticRetrievalDecision:
        try:
            query = self.query_builder.build(
                question=question,
                partial_cot_token_ids=partial_cot_token_ids,
            )
        except MemoryRecordRejected as exc:
            return SemanticRetrievalDecision(
                status="query_rejected",
                query=None,
                hits=(),
                matched_memory=None,
                rejection_reasons=exc.reasons,
            )
        hits = tuple(
            self.index.search(query.query_text, top_k=self.profile.retrieval_top_k)
        )
        query_audit = query.to_dict(include_text=False)
        query_audit.update({
            "method": "bm25",
            "top_k_requested": self.profile.retrieval_top_k,
            "top1_score": hits[0].score if hits else None,
            "top2_score": hits[1].score if len(hits) > 1 else None,
            "top1_top2_margin": (
                hits[0].score - hits[1].score if len(hits) > 1 else None
            ),
        })
        if not hits:
            return SemanticRetrievalDecision(
                status="no_positive_hit",
                query=query_audit,
                hits=(),
                matched_memory=None,
            )
        top = hits[0]
        return SemanticRetrievalDecision(
            status="selected",
            query=query_audit,
            hits=tuple(self._hit_dict(hit) for hit in hits),
            matched_memory=MemoryChoice(
                memory_id=top.memory_id,
                payload_hash=top.payload_hash,
                token_count=top.token_count,
                kv_valid_slot_count=self.kv_valid_slot_counts[top.memory_id],
                retrieval_score=float(top.score),
                retrieval_rank=top.rank,
            ),
        )
