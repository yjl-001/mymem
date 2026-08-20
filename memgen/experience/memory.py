"""Deterministic contracts for experience-backed side-KV memory.

This module deliberately has no Torch or Transformers dependency.  It owns the
auditable transformation from a Phase-1 approved abstraction to a runtime-safe
``MemoryRecord``.  Retrieval lives in ``memgen.experience.retrieval`` and
model-specific KV compilation/attention integration live in
``memgen.model.side_kv``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any, Iterable, Mapping, Protocol, Sequence

from memgen.experience.phase1 import (
    AI_REVIEW_PAIR_ASSESSMENTS,
    TEACHER_BANK_REQUIRED_FIELDS,
    canonical_json_sha256,
)


MEMORY_RECORD_SCHEMA = "experience-memory-record-v1"
MEMORY_BUILD_TRACE_SCHEMA = "experience-memory-build-trace-v1"
MEMORY_BUILD_REPORT_SCHEMA = "experience-memory-build-report-v1"
MEMORY_ARTIFACT_AUDIT_SCHEMA = "experience-memory-artifact-audit-v1"
STRUCTURED_PRO_REVIEW_SCHEMA = "phase1-ai-review-record-v2"
STRUCTURED_PRO_REVIEW_PROMPT = "phase1-ai-review-v2-field-evidence-rubric"
LEGACY_PRO_REVIEW_SCHEMA = "phase1-ai-review-record-v1"
LEGACY_PRO_REVIEW_PROMPT = "phase1-ai-review-v1-independent-evidence"
LEGACY_PRO_REVIEW_CRITERIA = (
    "target_supported",
    "reference_supported",
    "target_reference_distinct",
    "factually_consistent",
    "failure_type_aligned",
    "transferable_without_instance_leakage",
)
PAYLOAD_FIELD_LINEAGE: dict[str, tuple[str, ...]] = {
    "when_facing": (
        "bank.target.situation_signature",
        "bank.target.applicability_boundary",
    ),
    "prefer": (
        "bank.target.transferable_decision",
        "bank.target.verification_rule",
    ),
    "avoid": (
        "bank.reference.competing_pattern",
        "bank.reference.failure_signal",
        "bank.reference.failure_mechanism",
    ),
}

# Generic phrases such as "check the final answer" describe a transferable
# verification action and do not disclose an answer.  Concrete GSM8K values are
# blocked independently by the numeric/math audit; explicit answer containers
# remain forbidden even when malformed or empty.
_FINAL_ANSWER_RE = re.compile(r"(?:\\boxed|\\fbox)", re.IGNORECASE)
_NUMERIC_LITERAL_RE = re.compile(
    r"(?:\d|\\frac|\\dfrac|\\tfrac|[$€£¥]\s*\w|\b\w+\s*%|[=<>±×÷])",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


class TokenizerLike(Protocol):
    """Small tokenizer surface needed by the model-independent compiler."""

    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str: ...


class MemoryRecordRejected(ValueError):
    """Fail-closed rejection carrying stable, machine-readable reason codes."""

    def __init__(self, reasons: Sequence[str]):
        unique = tuple(sorted(set(str(reason) for reason in reasons if reason)))
        if not unique:
            unique = ("unspecified_rejection",)
        self.reasons = unique
        super().__init__(", ".join(unique))


@dataclass(frozen=True)
class MemorySanitizerConfig:
    """Frozen policy for converting reviewed fields into online text."""

    max_payload_tokens: int
    source_overlap_ngram_tokens: int = 8
    evidence_overlap_ngram_tokens: int = 6
    forbid_numeric_literals: bool = True

    def __post_init__(self) -> None:
        if self.max_payload_tokens <= 0:
            raise ValueError("max_payload_tokens must be positive")
        if self.source_overlap_ngram_tokens < 2:
            raise ValueError("source_overlap_ngram_tokens must be at least 2")
        if self.evidence_overlap_ngram_tokens < 2:
            raise ValueError("evidence_overlap_ngram_tokens must be at least 2")


@dataclass(frozen=True)
class MemoryRecord:
    """Runtime-safe text record with provenance kept outside model input."""

    memory_id: str
    source_experience_id: str
    experience_type: str
    approved_route: str
    source_logical_split: str
    phase1_provenance_sha256: str
    review_provenance_sha256: str
    source_record_sha256: str
    reasoner_name: str
    reasoner_revision: str
    tokenizer_revision: str
    sanitized_fields: Mapping[str, str]
    sanitized_retrieval_key: str
    sanitized_contrast_payload: str
    payload_hash: str
    token_ids_sha256: str
    token_count: int
    token_budget: int
    kv_layer: int = 24
    canonical_pre_rope_kv: Mapping[str, Any] = field(
        default_factory=lambda: {
            "compiled": False,
            "relative_phase_delta": 0,
        }
    )
    schema_version: str = MEMORY_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["field_lineage"] = {
            field_name: list(paths)
            for field_name, paths in PAYLOAD_FIELD_LINEAGE.items()
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryRecord":
        """Load a serialized record while rejecting schema/field drift."""

        data = dict(value)
        data.pop("field_lineage", None)
        if data.get("schema_version") != MEMORY_RECORD_SCHEMA:
            raise ValueError("Unexpected MemoryRecord schema_version")
        expected_fields = set(cls.__dataclass_fields__)
        if set(data) != expected_fields:
            missing = sorted(expected_fields - set(data))
            extra = sorted(set(data) - expected_fields)
            raise ValueError(
                f"MemoryRecord fields drifted: missing={missing}, extra={extra}"
            )
        return cls(**data)


@dataclass(frozen=True)
class MemoryBuildTrace:
    source_index: int
    experience_id: str
    status: str
    reasons: tuple[str, ...] = ()
    memory_id: str | None = None
    payload_hash: str | None = None
    token_count: int | None = None
    schema_version: str = MEMORY_BUILD_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryBuildResult:
    records: tuple[MemoryRecord, ...]
    trace: tuple[MemoryBuildTrace, ...]
    report: Mapping[str, Any]


@dataclass(frozen=True)
class Phase1MemorySource:
    """An approved abstraction joined to verifier-backed raw evidence."""

    approved_record: Mapping[str, Any]
    verified_experience: Mapping[str, Any]
    review_validation_profile: str

    @property
    def experience_id(self) -> str:
        return str(self.approved_record.get("experience_id", ""))


@dataclass(frozen=True)
class ProReviewValidation:
    """Result of validating one frozen Pro-review schema."""

    profile: str | None
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.profile is not None and not self.reasons


class ProReviewProfile(Protocol):
    """Versioned validation strategy for an immutable Phase-1 Pro review."""

    name: str

    def matches(self, gate: Mapping[str, Any]) -> bool: ...

    def rejection_reasons(
        self,
        gate: Mapping[str, Any],
        review: Mapping[str, Any],
    ) -> Sequence[str]: ...


class StructuredFieldEvidenceReviewProfile:
    """Validate the current v2 field-evidence Pro-review contract."""

    name = "structured_field_evidence_v2"

    def matches(self, gate: Mapping[str, Any]) -> bool:
        return (
            gate.get("schema_version") == STRUCTURED_PRO_REVIEW_SCHEMA
            and gate.get("prompt_version") == STRUCTURED_PRO_REVIEW_PROMPT
        )

    def rejection_reasons(
        self,
        gate: Mapping[str, Any],
        review: Mapping[str, Any],
    ) -> Sequence[str]:
        del gate
        reasons: list[str] = []
        if review.get("decision") != "approve":
            reasons.append("structured_pro_decision_not_approve")
        confidence = review.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            reasons.append("structured_pro_confidence_invalid")
        if not isinstance(review.get("issues"), list):
            reasons.append("structured_pro_issues_invalid")

        field_assessments = review.get("field_assessments")
        if not isinstance(field_assessments, Mapping):
            return reasons + ["missing_pro_field_assessments"]
        for section, fields in TEACHER_BANK_REQUIRED_FIELDS.items():
            assessments = field_assessments.get(section)
            if not isinstance(assessments, Mapping):
                reasons.append(f"missing_pro_{section}_assessments")
                continue
            for field_name in fields:
                if field_name == "confidence":
                    continue
                assessment = assessments.get(field_name)
                if not isinstance(assessment, Mapping):
                    reasons.append(f"missing_pro_{section}_{field_name}_assessment")
                    continue
                if assessment.get("status") != "supported":
                    reasons.append(f"pro_{section}_{field_name}_not_supported")
                if not self._nonempty_string(assessment.get("evidence")):
                    reasons.append(f"pro_{section}_{field_name}_evidence_missing")

        pair_assessments = review.get("pair_assessments")
        if not isinstance(pair_assessments, Mapping):
            reasons.append("missing_pro_pair_assessments")
        else:
            for field_name in AI_REVIEW_PAIR_ASSESSMENTS:
                assessment = pair_assessments.get(field_name)
                if not isinstance(assessment, Mapping):
                    reasons.append(f"missing_pro_pair_{field_name}_assessment")
                    continue
                if assessment.get("status") != "supported":
                    reasons.append(f"pro_pair_{field_name}_not_supported")
                if not self._nonempty_string(assessment.get("evidence")):
                    reasons.append(f"pro_pair_{field_name}_evidence_missing")
        return reasons

    @staticmethod
    def _nonempty_string(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())


class LegacyCriteriaReviewProfile:
    """Validate the frozen v1 Pro review without inventing v2 assessments."""

    name = "legacy_independent_criteria_v1"

    def matches(self, gate: Mapping[str, Any]) -> bool:
        return (
            gate.get("schema_version") == LEGACY_PRO_REVIEW_SCHEMA
            and gate.get("prompt_version") == LEGACY_PRO_REVIEW_PROMPT
        )

    def rejection_reasons(
        self,
        gate: Mapping[str, Any],
        review: Mapping[str, Any],
    ) -> Sequence[str]:
        reasons: list[str] = []
        if review.get("decision") != "approve":
            reasons.append("legacy_pro_decision_not_approve")

        confidence = review.get("confidence")
        threshold = gate.get("routing_confidence_threshold")
        if not self._valid_probability(confidence):
            reasons.append("legacy_pro_confidence_invalid")
        if not self._valid_probability(threshold):
            reasons.append("legacy_pro_routing_confidence_threshold_invalid")
        elif self._valid_probability(confidence) and float(confidence) < float(
            threshold
        ):
            reasons.append("legacy_pro_confidence_below_routing_threshold")

        criteria = review.get("criteria")
        if not isinstance(criteria, Mapping):
            reasons.append("missing_legacy_pro_criteria")
        else:
            for field_name in LEGACY_PRO_REVIEW_CRITERIA:
                value = criteria.get(field_name)
                if value is not True:
                    reasons.append(f"legacy_pro_{field_name}_not_supported")

        evidence = review.get("evidence")
        if not isinstance(evidence, Mapping):
            reasons.append("missing_legacy_pro_evidence")
        else:
            for field_name in ("target", "reference"):
                value = evidence.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    reasons.append(f"missing_legacy_pro_{field_name}_evidence")
        if not isinstance(review.get("issues"), list):
            reasons.append("legacy_pro_issues_invalid")
        if not isinstance(review.get("uncertainty_reason"), str):
            reasons.append("legacy_pro_uncertainty_reason_invalid")

        reasons.extend(self._audit_reasons(gate))
        return reasons

    @staticmethod
    def _valid_probability(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) <= 1.0
        )

    @staticmethod
    def _audit_reasons(gate: Mapping[str, Any]) -> list[str]:
        deterministic = gate.get("deterministic_audit")
        if isinstance(deterministic, Mapping):
            reasons: list[str] = []
            if deterministic.get("integrity_passed") is not True:
                reasons.append("legacy_pro_integrity_audit_not_passed")
            integrity_reasons = deterministic.get("integrity_reasons")
            if not isinstance(integrity_reasons, Sequence) or isinstance(
                integrity_reasons, (str, bytes)
            ):
                reasons.append("legacy_pro_integrity_reasons_invalid")
            elif integrity_reasons:
                reasons.append("legacy_pro_integrity_reasons_present")
            return reasons

        automatic = gate.get("automatic_gate")
        if not isinstance(automatic, Mapping):
            return ["missing_legacy_pro_deterministic_audit"]
        reasons = []
        if automatic.get("passed") is not True:
            reasons.append("legacy_pro_automatic_gate_not_passed")
        automatic_reasons = automatic.get("reasons")
        if not isinstance(automatic_reasons, Sequence) or isinstance(
            automatic_reasons, (str, bytes)
        ):
            reasons.append("legacy_pro_automatic_reasons_invalid")
        elif automatic_reasons:
            reasons.append("legacy_pro_automatic_reasons_present")
        return reasons


class ProReviewValidator:
    """Dispatch validation only to explicitly supported frozen review schemas."""

    def __init__(self, profiles: Sequence[ProReviewProfile] | None = None):
        self.profiles = tuple(
            profiles
            or (
                StructuredFieldEvidenceReviewProfile(),
                LegacyCriteriaReviewProfile(),
            )
        )
        if not self.profiles:
            raise ValueError("ProReviewValidator requires at least one profile")

    def validate(self, gate: Mapping[str, Any]) -> ProReviewValidation:
        review = gate.get("ai_review")
        if not isinstance(review, Mapping):
            return ProReviewValidation(None, ("missing_pro_review",))
        for profile in self.profiles:
            if profile.matches(gate):
                reasons = tuple(sorted(set(profile.rejection_reasons(gate, review))))
                return ProReviewValidation(profile.name, reasons)
        return ProReviewValidation(None, ("unsupported_pro_review_schema",))


class ApprovedMemorySourceSelector:
    """Validate the immutable Phase-1 quality/provenance gate for E0."""

    def __init__(
        self,
        *,
        allowed_experience_types: Sequence[str] = ("answer_correctness",),
        review_validator: ProReviewValidator | None = None,
    ):
        allowed = frozenset(str(value) for value in allowed_experience_types)
        if not allowed:
            raise ValueError("allowed_experience_types must not be empty")
        self.allowed_experience_types = allowed
        self.review_validator = review_validator or ProReviewValidator()

    def join(
        self,
        approved_records: Iterable[Mapping[str, Any]],
        verified_experiences: Iterable[Mapping[str, Any]],
    ) -> tuple[list[Phase1MemorySource], list[MemoryBuildTrace]]:
        experience_by_id: dict[str, Mapping[str, Any]] = {}
        for experience in verified_experiences:
            experience_id = str(experience.get("experience_id", ""))
            if not experience_id or experience_id in experience_by_id:
                raise ValueError(
                    f"Missing or duplicate verified experience_id: {experience_id!r}"
                )
            experience_by_id[experience_id] = experience

        sources: list[Phase1MemorySource] = []
        trace: list[MemoryBuildTrace] = []
        seen_ids: set[str] = set()
        for source_index, approved in enumerate(approved_records):
            experience_id = str(approved.get("experience_id", ""))
            if not experience_id or experience_id in seen_ids:
                raise ValueError(
                    f"Missing or duplicate approved experience_id: {experience_id!r}"
                )
            seen_ids.add(experience_id)
            experience = experience_by_id.get(experience_id)
            validation = self._validate_source(approved, experience)
            reasons = validation.reasons
            if reasons:
                trace.append(
                    MemoryBuildTrace(
                        source_index=source_index,
                        experience_id=experience_id,
                        status="rejected_selection",
                        reasons=tuple(reasons),
                    )
                )
                continue
            assert experience is not None
            assert validation.review_profile is not None
            sources.append(
                Phase1MemorySource(
                    approved_record=approved,
                    verified_experience=experience,
                    review_validation_profile=validation.review_profile,
                )
            )
        return sources, trace

    @dataclass(frozen=True)
    class SourceValidation:
        reasons: tuple[str, ...]
        review_profile: str | None

    def _validate_source(
        self,
        approved: Mapping[str, Any],
        experience: Mapping[str, Any] | None,
    ) -> SourceValidation:
        reasons: list[str] = []
        experience_id = str(approved.get("experience_id", ""))
        if experience is None:
            return self.SourceValidation(("missing_verified_experience",), None)
        gate = approved.get("ai_review_gate")
        review_profile: str | None = None
        if not isinstance(gate, Mapping) or gate.get("route") != "ai_approved":
            reasons.append("route_not_ai_approved")
        if isinstance(gate, Mapping):
            review_validation = self.review_validator.validate(gate)
            review_profile = review_validation.profile
            reasons.extend(review_validation.reasons)
        else:
            reasons.append("missing_pro_review")
        if approved.get("experience_type") not in self.allowed_experience_types:
            reasons.append("approved_experience_type_not_allowed")
        if experience.get("experience_type") not in self.allowed_experience_types:
            reasons.append("verified_experience_type_not_allowed")
        if approved.get("reference_evidence") != "verified_failure":
            reasons.append("approved_reference_not_verified_failure")
        if experience.get("reference_evidence") != "verified_failure":
            reasons.append("verified_reference_not_verified_failure")
        if approved.get("source", {}).get("logical_split") != "bank-source":
            reasons.append("approved_source_not_bank_source")
        if experience.get("source", {}).get("logical_split") != "bank-source":
            reasons.append("verified_source_not_bank_source")
        for key in ("experience_id", "provenance_sha256", "source", "student", "experience_type"):
            if approved.get(key) != experience.get(key):
                reasons.append(f"phase1_{key}_mismatch")
        expected_episode_ids = {
            "target": experience.get("target_episode_id"),
            "reference": experience.get("reference_episode_id"),
        }
        if approved.get("source_episode_ids") != expected_episode_ids:
            reasons.append("source_episode_ids_mismatch")
        if experience.get("reference_verifier", {}).get("reward") != 0.0:
            reasons.append("reference_verifier_not_failure")
        bank = approved.get("bank")
        if not isinstance(bank, Mapping):
            reasons.append("missing_reviewed_bank")
        else:
            if bank.get("experience_type") != approved.get("experience_type"):
                reasons.append("reviewed_bank_experience_type_mismatch")
            approved_failure_types = list(approved.get("reference_failure_types") or [])
            verified_failure_types = list(experience.get("reference_failure_types") or [])
            if approved_failure_types != verified_failure_types:
                reasons.append("phase1_reference_failure_types_mismatch")
            if list(bank.get("failure_types") or []) != verified_failure_types:
                reasons.append("reviewed_bank_failure_types_mismatch")
        if not str(approved.get("provenance_sha256", "")):
            reasons.append("missing_phase1_provenance_sha256")
        if not isinstance(gate, Mapping) or not str(
            gate.get("review_provenance_sha256", "")
        ):
            reasons.append("missing_review_provenance_sha256")
        if not experience_id:
            reasons.append("missing_experience_id")
        return self.SourceValidation(tuple(sorted(set(reasons))), review_profile)


class PayloadSanitizer:
    """Conservative sanitizer that never invents or paraphrases semantics."""

    def __init__(self, config: MemorySanitizerConfig):
        self.config = config

    @staticmethod
    def normalize_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = _CONTROL_RE.sub(" ", normalized)
        return _SPACE_RE.sub(" ", normalized).strip()

    def sanitize_field(
        self,
        *,
        path: str,
        value: Any,
        source: Phase1MemorySource,
    ) -> str:
        reasons: list[str] = []
        if not isinstance(value, str):
            raise MemoryRecordRejected([f"{path}:not_string"])
        normalized = self.normalize_text(value)
        if not normalized:
            reasons.append(f"{path}:empty")
        if _FINAL_ANSWER_RE.search(normalized):
            reasons.append(f"{path}:final_answer_marker")
        if self.config.forbid_numeric_literals and _NUMERIC_LITERAL_RE.search(normalized):
            reasons.append(f"{path}:numeric_or_math_literal")

        verified = source.verified_experience
        for label, raw in (
            ("context", verified.get("context")),
            ("target_trajectory", verified.get("trajectory")),
            ("reference_trajectory", verified.get("reference_trajectory")),
        ):
            if self._has_ngram_overlap(
                normalized,
                raw,
                self.config.source_overlap_ngram_tokens,
            ):
                reasons.append(f"{path}:overlaps_{label}")

        bank = source.approved_record.get("bank", {})
        evidence = bank.get("evidence", {}) if isinstance(bank, Mapping) else {}
        if isinstance(evidence, Mapping):
            for label in ("target_observation", "reference_observation"):
                raw = evidence.get(label)
                if self._same_nonempty_text(normalized, raw) or self._has_ngram_overlap(
                    normalized,
                    raw,
                    self.config.evidence_overlap_ngram_tokens,
                ):
                    reasons.append(f"{path}:overlaps_bank_evidence_{label}")

        gate = source.approved_record.get("ai_review_gate", {})
        review = gate.get("ai_review", {}) if isinstance(gate, Mapping) else {}
        for evidence_text in self._review_evidence_texts(review):
            if self._same_nonempty_text(normalized, evidence_text) or self._has_ngram_overlap(
                normalized,
                evidence_text,
                self.config.evidence_overlap_ngram_tokens,
            ):
                reasons.append(f"{path}:overlaps_pro_evidence")
                break

        if reasons:
            raise MemoryRecordRejected(reasons)
        return normalized

    @staticmethod
    def _same_nonempty_text(left: str, right: Any) -> bool:
        if not isinstance(right, str) or not right.strip():
            return False
        return PayloadSanitizer.normalize_text(left).casefold() == PayloadSanitizer.normalize_text(
            right
        ).casefold()

    @staticmethod
    def _word_tokens(value: Any) -> list[str]:
        if not isinstance(value, str):
            return []
        return [match.group(0).casefold() for match in _WORD_RE.finditer(value)]

    @classmethod
    def _has_ngram_overlap(cls, left: str, right: Any, width: int) -> bool:
        left_tokens = cls._word_tokens(left)
        right_tokens = cls._word_tokens(right)
        if len(left_tokens) < width or len(right_tokens) < width:
            return False
        left_ngrams = {
            tuple(left_tokens[index : index + width])
            for index in range(len(left_tokens) - width + 1)
        }
        return any(
            tuple(right_tokens[index : index + width]) in left_ngrams
            for index in range(len(right_tokens) - width + 1)
        )

    @staticmethod
    def _review_evidence_texts(review: Any) -> list[str]:
        if not isinstance(review, Mapping):
            return []
        values: list[str] = []
        for section in ("field_assessments", "pair_assessments"):
            root = review.get(section)
            if not isinstance(root, Mapping):
                continue
            stack = list(root.values())
            while stack:
                item = stack.pop()
                if isinstance(item, Mapping):
                    evidence = item.get("evidence")
                    if isinstance(evidence, str):
                        values.append(evidence)
                    stack.extend(item.values())
        return values


class MemoryRecordCompiler:
    """Compile one approved Phase-1 source into a deterministic text record."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        sanitizer: PayloadSanitizer,
        reasoner_name: str,
        reasoner_revision: str,
        tokenizer_revision: str,
        kv_layer: int = 24,
    ):
        if kv_layer <= 0:
            raise ValueError("kv_layer must be positive")
        self.tokenizer = tokenizer
        self.sanitizer = sanitizer
        self.reasoner_name = reasoner_name
        self.reasoner_revision = reasoner_revision
        self.tokenizer_revision = tokenizer_revision
        self.kv_layer = kv_layer

    def compile(self, source: Phase1MemorySource) -> MemoryRecord:
        approved = source.approved_record
        student = approved.get("student", {})
        revision_reasons = []
        if student.get("model_name") != self.reasoner_name:
            revision_reasons.append("reasoner_name_differs_from_phase1_student")
        if student.get("model_revision") != self.reasoner_revision:
            revision_reasons.append("reasoner_revision_differs_from_phase1_student")
        if student.get("tokenizer_revision") != self.tokenizer_revision:
            revision_reasons.append("tokenizer_revision_differs_from_phase1_student")
        if revision_reasons:
            raise MemoryRecordRejected(revision_reasons)
        bank = approved.get("bank")
        if not isinstance(bank, Mapping):
            raise MemoryRecordRejected(["missing_reviewed_bank"])

        sanitized_by_path: dict[str, str] = {}
        reasons: list[str] = []
        for paths in PAYLOAD_FIELD_LINEAGE.values():
            for path in paths:
                try:
                    sanitized_by_path[path] = self.sanitizer.sanitize_field(
                        path=path,
                        value=self._read_path(approved, path),
                        source=source,
                    )
                except MemoryRecordRejected as exc:
                    reasons.extend(exc.reasons)
        if reasons:
            raise MemoryRecordRejected(reasons)

        fields = {
            field_name: self._join_unique(sanitized_by_path[path] for path in paths)
            for field_name, paths in PAYLOAD_FIELD_LINEAGE.items()
        }
        payload = (
            f"When facing: {fields['when_facing']}\n"
            f"Prefer: {fields['prefer']}\n"
            f"Avoid: {fields['avoid']}"
        )
        if _FINAL_ANSWER_RE.search(payload) or (
            self.sanitizer.config.forbid_numeric_literals
            and _NUMERIC_LITERAL_RE.search(payload)
        ):
            raise MemoryRecordRejected(["rendered_payload_failed_forbidden_pattern_audit"])

        token_ids = [
            int(value)
            for value in self.tokenizer.encode(payload, add_special_tokens=False)
        ]
        if not token_ids:
            raise MemoryRecordRejected(["rendered_payload_has_no_tokens"])
        if len(token_ids) > self.sanitizer.config.max_payload_tokens:
            raise MemoryRecordRejected(["payload_exceeds_token_budget"])

        retrieval_key = " ".join(fields.values())
        payload_hash = canonical_json_sha256(
            {"schema_version": MEMORY_RECORD_SCHEMA, "payload": payload}
        )
        source_experience_id = source.experience_id
        memory_id = "mem-" + canonical_json_sha256(
            {
                "schema_version": MEMORY_RECORD_SCHEMA,
                "source_experience_id": source_experience_id,
                "payload_hash": payload_hash,
                "kv_layer": self.kv_layer,
            }
        )[:24]
        gate = approved.get("ai_review_gate", {})
        return MemoryRecord(
            memory_id=memory_id,
            source_experience_id=source_experience_id,
            experience_type=str(approved.get("experience_type")),
            approved_route=str(gate.get("route")),
            source_logical_split=str(approved.get("source", {}).get("logical_split")),
            phase1_provenance_sha256=str(approved.get("provenance_sha256", "")),
            review_provenance_sha256=str(gate.get("review_provenance_sha256", "")),
            source_record_sha256=canonical_json_sha256(approved),
            reasoner_name=self.reasoner_name,
            reasoner_revision=self.reasoner_revision,
            tokenizer_revision=self.tokenizer_revision,
            sanitized_fields=fields,
            sanitized_retrieval_key=retrieval_key,
            sanitized_contrast_payload=payload,
            payload_hash=payload_hash,
            token_ids_sha256=canonical_json_sha256(token_ids),
            token_count=len(token_ids),
            token_budget=self.sanitizer.config.max_payload_tokens,
            kv_layer=self.kv_layer,
        )

    @staticmethod
    def _read_path(root: Mapping[str, Any], path: str) -> Any:
        value: Any = root
        for part in path.split("."):
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value

    @staticmethod
    def _join_unique(values: Iterable[str]) -> str:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold().rstrip(". ")
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)
        return " ".join(unique)


class MemoryBankBuilder:
    """Orchestrate Phase-1 selection, text compilation, and audit reporting."""

    def __init__(
        self,
        *,
        selector: ApprovedMemorySourceSelector,
        compiler: MemoryRecordCompiler,
    ):
        self.selector = selector
        self.compiler = compiler

    def build(
        self,
        approved_records: Iterable[Mapping[str, Any]],
        verified_experiences: Iterable[Mapping[str, Any]],
    ) -> MemoryBuildResult:
        approved = list(approved_records)
        verified = list(verified_experiences)
        sources, selection_trace = self.selector.join(approved, verified)
        trace = list(selection_trace)
        records: list[MemoryRecord] = []
        seen_payload_hashes: dict[str, str] = {}
        accepted_review_profiles: Counter[str] = Counter()
        source_index_by_id = {
            str(record.get("experience_id")): index
            for index, record in enumerate(approved)
        }
        for source in sources:
            source_index = source_index_by_id[source.experience_id]
            try:
                record = self.compiler.compile(source)
            except MemoryRecordRejected as exc:
                trace.append(
                    MemoryBuildTrace(
                        source_index=source_index,
                        experience_id=source.experience_id,
                        status="rejected_payload",
                        reasons=exc.reasons,
                    )
                )
                continue
            duplicate_of = seen_payload_hashes.get(record.payload_hash)
            if duplicate_of is not None:
                trace.append(
                    MemoryBuildTrace(
                        source_index=source_index,
                        experience_id=source.experience_id,
                        status="rejected_duplicate_payload",
                        reasons=(f"duplicate_of:{duplicate_of}",),
                        memory_id=record.memory_id,
                        payload_hash=record.payload_hash,
                        token_count=record.token_count,
                    )
                )
                continue
            seen_payload_hashes[record.payload_hash] = record.memory_id
            records.append(record)
            accepted_review_profiles[source.review_validation_profile] += 1
            trace.append(
                MemoryBuildTrace(
                    source_index=source_index,
                    experience_id=source.experience_id,
                    status="accepted",
                    memory_id=record.memory_id,
                    payload_hash=record.payload_hash,
                    token_count=record.token_count,
                )
            )

        trace.sort(key=lambda item: item.source_index)
        token_counts = sorted(record.token_count for record in records)
        status_counts = Counter(item.status for item in trace)
        reason_counts = Counter(reason for item in trace for reason in item.reasons)
        report = {
            "schema_version": MEMORY_BUILD_REPORT_SCHEMA,
            "input_approved_count": len(approved),
            "input_verified_experience_count": len(verified),
            "selected_source_count": len(sources),
            "accepted_record_count": len(records),
            "selected_review_profile_counts": dict(
                sorted(
                    Counter(
                        source.review_validation_profile for source in sources
                    ).items()
                )
            ),
            "accepted_review_profile_counts": dict(
                sorted(accepted_review_profiles.items())
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "token_count": self._distribution(token_counts),
            "payload_hash_unique_count": len(seen_payload_hashes),
            "policy": asdict(self.compiler.sanitizer.config),
            "record_set_sha256": canonical_json_sha256(
                [record.to_dict() for record in records]
            ),
        }
        return MemoryBuildResult(tuple(records), tuple(trace), report)

    @staticmethod
    def _distribution(values: Sequence[int]) -> dict[str, int | float | None]:
        if not values:
            return {"min": None, "median": None, "p95": None, "max": None}

        def percentile(q: float) -> float:
            position = (len(values) - 1) * q
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            weight = position - lower
            return values[lower] * (1 - weight) + values[upper] * weight

        return {
            "min": values[0],
            "median": percentile(0.5),
            "p95": percentile(0.95),
            "max": values[-1],
        }


class MemoryArtifactAuditor:
    """Independently re-check serialized payload and tokenizer invariants."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        expected_token_budget: int,
        expected_kv_layer: int = 24,
        expected_kv_compiled: bool | None = None,
    ):
        if expected_token_budget <= 0:
            raise ValueError("expected_token_budget must be positive")
        if expected_kv_layer <= 0:
            raise ValueError("expected_kv_layer must be positive")
        self.tokenizer = tokenizer
        self.expected_token_budget = expected_token_budget
        self.expected_kv_layer = expected_kv_layer
        self.expected_kv_compiled = expected_kv_compiled

    def audit(self, records: Sequence[MemoryRecord]) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        seen_payload_hashes: set[str] = set()
        for record in records:
            reasons = self._record_reasons(record)
            if record.memory_id in seen_memory_ids:
                reasons.append("duplicate_memory_id")
            if record.payload_hash in seen_payload_hashes:
                reasons.append("duplicate_payload_hash")
            seen_memory_ids.add(record.memory_id)
            seen_payload_hashes.add(record.payload_hash)
            if reasons:
                violations.append(
                    {
                        "memory_id": record.memory_id,
                        "reasons": sorted(set(reasons)),
                    }
                )
        reason_counts = Counter(
            reason for violation in violations for reason in violation["reasons"]
        )
        return {
            "schema_version": MEMORY_ARTIFACT_AUDIT_SCHEMA,
            "status": "passed" if not violations and records else "failed",
            "record_count": len(records),
            "violation_count": len(violations),
            "violations": violations,
            "violation_reason_counts": dict(sorted(reason_counts.items())),
            "expected_token_budget": self.expected_token_budget,
            "expected_kv_layer": self.expected_kv_layer,
            "expected_kv_compiled": self.expected_kv_compiled,
            "record_set_sha256": canonical_json_sha256(
                [record.to_dict() for record in records]
            ),
        }

    def assert_valid(self, records: Sequence[MemoryRecord]) -> dict[str, Any]:
        report = self.audit(records)
        if report["status"] != "passed":
            raise ValueError(
                "Memory artifact audit failed: "
                f"reasons={report['violation_reason_counts']}, "
                f"report_sha256={canonical_json_sha256(report)}"
            )
        return report

    def _record_reasons(self, record: MemoryRecord) -> list[str]:
        reasons: list[str] = []
        if record.schema_version != MEMORY_RECORD_SCHEMA:
            reasons.append("schema_version_mismatch")
        if record.approved_route != "ai_approved":
            reasons.append("approved_route_mismatch")
        if record.experience_type != "answer_correctness":
            reasons.append("experience_type_mismatch")
        if record.source_logical_split != "bank-source":
            reasons.append("source_logical_split_mismatch")
        for field_name, value in (
            ("source_experience_id", record.source_experience_id),
            ("phase1_provenance_sha256", record.phase1_provenance_sha256),
            ("review_provenance_sha256", record.review_provenance_sha256),
            ("source_record_sha256", record.source_record_sha256),
            ("reasoner_name", record.reasoner_name),
            ("reasoner_revision", record.reasoner_revision),
            ("tokenizer_revision", record.tokenizer_revision),
        ):
            if not value:
                reasons.append(f"{field_name}_missing")
        fields_are_valid = isinstance(record.sanitized_fields, Mapping) and set(
            record.sanitized_fields
        ) == set(PAYLOAD_FIELD_LINEAGE)
        if not fields_are_valid:
            reasons.append("sanitized_field_set_mismatch")
        else:
            fields_are_nonempty_strings = not any(
                not isinstance(value, str) or not value.strip()
                for value in record.sanitized_fields.values()
            )
            if not fields_are_nonempty_strings:
                reasons.append("sanitized_field_empty")
            else:
                expected_payload = (
                    f"When facing: {record.sanitized_fields['when_facing']}\n"
                    f"Prefer: {record.sanitized_fields['prefer']}\n"
                    f"Avoid: {record.sanitized_fields['avoid']}"
                )
                if record.sanitized_contrast_payload != expected_payload:
                    reasons.append("payload_render_mismatch")
                expected_retrieval_key = " ".join(
                    record.sanitized_fields[field_name]
                    for field_name in PAYLOAD_FIELD_LINEAGE
                )
                if record.sanitized_retrieval_key != expected_retrieval_key:
                    reasons.append("retrieval_key_render_mismatch")
        if _FINAL_ANSWER_RE.search(record.sanitized_contrast_payload):
            reasons.append("payload_final_answer_marker")
        if _NUMERIC_LITERAL_RE.search(record.sanitized_contrast_payload):
            reasons.append("payload_numeric_or_math_literal")
        expected_payload_hash = canonical_json_sha256(
            {
                "schema_version": MEMORY_RECORD_SCHEMA,
                "payload": record.sanitized_contrast_payload,
            }
        )
        if record.payload_hash != expected_payload_hash:
            reasons.append("payload_hash_mismatch")
        expected_memory_id = "mem-" + canonical_json_sha256(
            {
                "schema_version": MEMORY_RECORD_SCHEMA,
                "source_experience_id": record.source_experience_id,
                "payload_hash": record.payload_hash,
                "kv_layer": record.kv_layer,
            }
        )[:24]
        if record.memory_id != expected_memory_id:
            reasons.append("memory_id_mismatch")
        token_ids = [
            int(value)
            for value in self.tokenizer.encode(
                record.sanitized_contrast_payload,
                add_special_tokens=False,
            )
        ]
        if record.token_ids_sha256 != canonical_json_sha256(token_ids):
            reasons.append("token_ids_hash_mismatch")
        if record.token_count != len(token_ids):
            reasons.append("token_count_mismatch")
        if record.token_budget != self.expected_token_budget:
            reasons.append("token_budget_mismatch")
        if not token_ids or len(token_ids) > self.expected_token_budget:
            reasons.append("token_budget_violated")
        if record.kv_layer != self.expected_kv_layer:
            reasons.append("kv_layer_mismatch")
        if not isinstance(record.canonical_pre_rope_kv, Mapping):
            reasons.append("canonical_pre_rope_kv_invalid")
        elif self.expected_kv_compiled is not None:
            compiled = record.canonical_pre_rope_kv.get("compiled")
            if compiled is not self.expected_kv_compiled:
                reasons.append("kv_compiled_status_mismatch")
            if self.expected_kv_compiled:
                required_kv_fields = {
                    "artifact",
                    "artifact_sha256",
                    "manifest",
                    "manifest_sha256",
                    "record_index",
                    "kv_valid_slot_count",
                    "tensor_layout",
                }
                if not required_kv_fields.issubset(record.canonical_pre_rope_kv):
                    reasons.append("compiled_kv_reference_incomplete")
        return reasons
