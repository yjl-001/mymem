#!/usr/bin/env python3
"""Build the MemGen V4.1 tensor-free repair bank from frozen V4 signatures.

V4.1 deliberately does not regenerate the expensive per-experience signatures.
It authenticates a completed V4 signature checkpoint, maps applicable signatures
to bounded process atoms, retrieves semantic neighbours across the verifier's
``experience_type`` labels, judges candidate edges, forms only clique-consistent
groups, and audits five-to-ten independent examples before writing process cards.

Every teacher stage is append-checkpointed.  HTTP/payment/authentication failures
therefore stop safely without discarding validated work.  Malformed JSON batches
are split recursively; a persistently malformed singleton is archived rather
than aborting thousands of otherwise valid examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl
from memgen.experience.v4_bank import (
    V4_SIGNATURE_PROMPT_VERSION,
    V4ConstructionProfile,
    V4CardReview,
    V4ProcessCard,
    V4RepairSignature,
    parse_v4_card_review,
    parse_v4_process_card,
    parse_v4_repair_signature,
)
from memgen.experience.v4_1_bank import (
    V4_1_APPLICABILITY_FAMILIES,
    V4_1_AUDIT_PROMPT_VERSION,
    V4_1_CANONICAL_ATOM_SCHEMA,
    V4_1_CANONICAL_PAYLOAD_SCHEMA,
    V4_1_CANONICAL_PROMPT_VERSION,
    V4_1_CARD_PROMPT_VERSION,
    V4_1_CLUSTER_PLAN_SCHEMA,
    V4_1_DEFAULT_CANONICAL_BATCH_SIZE,
    V4_1_DEFAULT_NEIGHBOR_COUNT,
    V4_1_DEFAULT_PAIR_BATCH_SIZE,
    V4_1_EMBEDDING_MODEL,
    V4_1_EMBEDDING_REVISION,
    V4_1_MECHANISM_FAMILIES,
    V4_1_MEMORY_ROLES,
    V4_1_PAIR_PAYLOAD_SCHEMA,
    V4_1_PAIR_PROMPT_VERSION,
    V4_1_REPAIR_FAMILIES,
    V4_1_REVIEW_PROMPT_VERSION,
    V4_1_STATE_SCOPES,
    V41CanonicalRepairAtom,
    V41ClusterAudit,
    V41ConstructionProfile,
    V41PairJudgment,
    V41RepairCluster,
    build_v4_1_bank_manifest,
    build_v4_1_bank_record,
    parse_v4_1_canonical_atom,
    parse_v4_1_cluster_audit,
    parse_v4_1_pair_judgment,
)
from scripts.build_teacher_bank import TeacherClient, TeacherInvalidResponseError
from scripts.build_v4_repair_bank import (
    SIGNATURE_RECORD_SCHEMA,
    _parse_json_object,
    _validate_split_manifest,
    attach_official_solutions,
    load_v4_experiences,
)


CANONICAL_UNIT_RECORD_SCHEMA = "memgen-v4.1-canonical-unit-record-v1"
PAIR_UNIT_RECORD_SCHEMA = "memgen-v4.1-pair-unit-record-v1"
AUDIT_UNIT_RECORD_SCHEMA = "memgen-v4.1-audit-unit-record-v1"
CARD_RECORD_SCHEMA = "memgen-v4.1-process-card-record-v1"
REVIEW_RECORD_SCHEMA = "memgen-v4.1-process-card-review-record-v1"
EMBEDDING_MANIFEST_SCHEMA = "memgen-v4.1-canonical-embeddings-v1"
CANDIDATE_PAIR_SCHEMA = "memgen-v4.1-candidate-pairs-v1"
MAX_REQUEST_CHARACTERS = 200_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repository_state() -> dict[str, Any]:
    paths = (
        "memgen/experience/v4_bank.py",
        "memgen/experience/v4_1_bank.py",
        "scripts/build_v4_1_repair_bank.py",
        "scripts/build_teacher_bank.py",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V4.1 construction requires a git revision") from exc
    if not revision:
        raise RuntimeError("V4.1 construction resolved an empty git revision")
    return {
        "git_revision": revision,
        "implementation_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in paths
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-signatures", type=Path, required=True)
    parser.add_argument("--source-construction-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--stage",
        choices=("cluster", "cards", "all"),
        default="all",
        help="Run clustering only, resume from its artifacts for cards, or both.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_TEACHER_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--canonical-batch-size", type=int, default=V4_1_DEFAULT_CANONICAL_BATCH_SIZE
    )
    parser.add_argument("--pair-batch-size", type=int, default=V4_1_DEFAULT_PAIR_BATCH_SIZE)
    parser.add_argument("--neighbor-count", type=int, default=V4_1_DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument("--canonical-max-tokens", type=int, default=9000)
    parser.add_argument("--pair-max-tokens", type=int, default=9000)
    parser.add_argument("--audit-max-tokens", type=int, default=2200)
    parser.add_argument("--card-max-tokens", type=int, default=2200)
    parser.add_argument("--review-max-tokens", type=int, default=1400)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--proxy-retries", type=int, default=20)
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only hash- and teacher-compatible append checkpoints.",
    )
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace) -> V41ConstructionProfile:
    profile = V41ConstructionProfile(
        teacher_model=args.model,
        temperature=args.temperature,
        thinking=args.thinking,
        neighbor_count=args.neighbor_count,
        canonical_batch_size=args.canonical_batch_size,
        pair_batch_size=args.pair_batch_size,
    )
    for owner in (
        "canonical_max_tokens",
        "pair_max_tokens",
        "audit_max_tokens",
        "card_max_tokens",
        "review_max_tokens",
        "embedding_batch_size",
    ):
        if getattr(args, owner) <= 0:
            raise ValueError(f"--{owner.replace('_', '-')} must be positive")
    parsed = urlsplit(args.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("V4.1 DeepSeek base URL is invalid or contains credentials")
    return profile


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _client(args: argparse.Namespace, *, api_key: str, max_tokens: int) -> TeacherClient:
    return TeacherClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        proxy_retries=args.proxy_retries,
        proxy_retry_initial_seconds=args.proxy_retry_initial_seconds,
        proxy_retry_max_seconds=args.proxy_retry_max_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        thinking=args.thinking,
    )


def load_authenticated_signatures(
    path: Path,
    *,
    source_profile_path: Path,
    experiences: Sequence[Mapping[str, Any]],
) -> tuple[tuple[V4RepairSignature, ...], dict[str, Any]]:
    """Authenticate and reuse the completed V4 signatures without API calls."""

    source_profile_record = json.loads(source_profile_path.read_text(encoding="utf-8"))
    source_profile = V4ConstructionProfile(**source_profile_record.get("profile", {}))
    if source_profile_record.get("profile_sha256") != source_profile.profile_sha256:
        raise ValueError("Source V4 construction profile hash mismatch")
    prompt_versions = source_profile_record.get("prompt_versions", {})
    if prompt_versions.get("signature") != V4_SIGNATURE_PROMPT_VERSION:
        raise ValueError("Source V4 signature prompt version drifted")
    teacher = source_profile_record.get("teacher", {})
    if (
        teacher.get("model") != source_profile.teacher_model
        or teacher.get("temperature") != source_profile.temperature
        or teacher.get("thinking") != source_profile.thinking
    ):
        raise ValueError("Source V4 signature teacher binding drifted")

    experiences_by_id = {str(item["experience_id"]): item for item in experiences}
    signatures: list[V4RepairSignature] = []
    seen: set[str] = set()
    for record in iter_jsonl(path):
        if record.get("schema_version") != SIGNATURE_RECORD_SCHEMA:
            raise ValueError("Unexpected source V4 signature-record schema")
        if record.get("prompt_version") != V4_SIGNATURE_PROMPT_VERSION:
            raise ValueError("Source V4 signature-record prompt drifted")
        if record.get("teacher") != teacher:
            raise ValueError("Source V4 signature-record teacher binding drifted")
        payload = record.get("signature", {})
        experience_id = str(payload.get("experience_id", ""))
        if not experience_id or experience_id in seen:
            raise ValueError(f"Missing or duplicate source signature ID: {experience_id!r}")
        experience = experiences_by_id.get(experience_id)
        if experience is None:
            raise ValueError(f"Source signature is not in V4 experiences: {experience_id}")
        signature = parse_v4_repair_signature(
            payload,
            experience_id=experience_id,
            sample_id=str(experience["sample_id"]),
            experience_type=str(experience["experience_type"]),
            source_provenance_sha256=str(experience["provenance_sha256"]),
        )
        if record.get("signature_sha256") != signature.signature_sha256:
            raise ValueError(f"Source V4 signature hash mismatch: {experience_id}")
        signatures.append(signature)
        seen.add(experience_id)
    if seen != set(experiences_by_id):
        missing = sorted(set(experiences_by_id) - seen)
        raise ValueError(f"Source V4 signature checkpoint is incomplete: {missing[:5]}")
    return tuple(sorted(signatures, key=lambda item: item.experience_id)), {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "profile_path": str(source_profile_path.resolve()),
        "profile_file_sha256": file_sha256(source_profile_path),
        "profile_sha256": source_profile.profile_sha256,
        "prompt_version": V4_SIGNATURE_PROMPT_VERSION,
        "teacher": teacher,
        "count": len(signatures),
        "applicable_count": sum(item.applicable for item in signatures),
    }


def canonical_atom_messages(signatures: Sequence[V4RepairSignature]) -> list[dict[str, str]]:
    compact = [
        {
            "experience_id": item.experience_id,
            "source_experience_type": item.experience_type,
            "problem_structure": item.problem_structure,
            "decision_point": item.decision_point,
            "failure_mechanism": item.failure_mechanism,
            "repair_operator": item.repair_operator,
            "verification_operator": item.verification_operator,
        }
        for item in signatures
    ]
    system = f"""You are the V4.1 repair-atom canonicalizer. Return JSON only.
Map every supplied grounded signature to exactly one bounded process atom.
The verifier's source_experience_type is provenance, never a semantic category.
Use reasoning_process only for a reusable change to reasoning state or action.
Use answer_serialization for boxing, labels, units attached only at output,
currency symbols, decimal display, or any other final-answer representation.
Use unusable when the signature does not ground one clear reusable transition.
For answer_serialization use exactly the dedicated answer_serialization scope,
output_representation mechanism, canonicalize_final_answer repair, and
answer_serialization applicability. For unusable use other for all four fields.
An authenticated format_compliance source must be answer_serialization because
its verifier already established that only final representation failed.

Choose category strings only from these exact lists:
memory_role={json.dumps(V4_1_MEMORY_ROLES)}
state_scope={json.dumps(V4_1_STATE_SCOPES)}
mechanism_family={json.dumps(V4_1_MECHANISM_FAMILIES)}
repair_family={json.dumps(V4_1_REPAIR_FAMILIES)}
applicability_family={json.dumps(V4_1_APPLICABILITY_FAMILIES)}

Write concise English process text. Remove names, story objects, quantities,
answers, digits, equations, formulas, and source-solution traces. Canonicalize
synonyms where the transition is genuinely the same; do not erase meaningful
differences in operation, state, applicability, or verification. A reasoning
atom may not use any answer-serialization category. Excluded atoms must give
a nonempty exclusion_reason; reasoning atoms must use null."""
    user = f"""Signatures:
{json.dumps(compact, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "schema_version": "{V4_1_CANONICAL_PAYLOAD_SCHEMA}",
  "atoms": {{
    "exact-experience-id": {{
      "memory_role": "one allowed value",
      "state_scope": "one allowed value",
      "mechanism_family": "one allowed value",
      "repair_family": "one allowed value",
      "applicability_family": "one allowed value",
      "failure_transition": "canonical process-only failure transition",
      "repair_action": "canonical process-only corrective action",
      "applicability_condition": "canonical applicability boundary",
      "verification_action": "canonical post-repair check",
      "exclusion_reason": null
    }}
  }}
}}
The atoms object must contain every supplied experience_id exactly once and no others."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_canonical_payload(
    content: str, *, signatures: Sequence[V4RepairSignature]
) -> dict[str, Any]:
    payload = _parse_json_object(content)
    if payload.get("schema_version") != V4_1_CANONICAL_PAYLOAD_SCHEMA:
        raise ValueError("Unexpected V4.1 canonical batch schema")
    values = payload.get("atoms")
    if not isinstance(values, Mapping):
        raise ValueError("V4.1 canonical payload is missing atoms")
    expected = {item.experience_id for item in signatures}
    if set(values) != expected:
        raise ValueError("V4.1 canonical payload does not cover its exact input")
    by_id = {item.experience_id: item for item in signatures}
    normalized_atoms: dict[str, dict[str, Any]] = {}
    for experience_id, value in values.items():
        if not isinstance(value, Mapping):
            raise ValueError("V4.1 canonical atom must be an object")
        atom = parse_v4_1_canonical_atom(value, signature=by_id[experience_id])
        normalized = {
            key: atom.to_dict()[key]
            for key in (
                "memory_role",
                "state_scope",
                "mechanism_family",
                "repair_family",
                "applicability_family",
                "failure_transition",
                "repair_action",
                "applicability_condition",
                "verification_action",
                "exclusion_reason",
            )
        }
        normalized["normalization_flags"] = [
            key
            for key in normalized
            if key != "normalization_flags" and value.get(key) != normalized[key]
        ]
        normalized_atoms[experience_id] = normalized
    return {
        "schema_version": V4_1_CANONICAL_PAYLOAD_SCHEMA,
        "atoms": normalized_atoms,
    }


def _fallback_unusable_atom(signature: V4RepairSignature) -> V41CanonicalRepairAtom:
    if signature.experience_type == "format_compliance":
        return V41CanonicalRepairAtom(
            experience_id=signature.experience_id,
            sample_id=signature.sample_id,
            source_experience_type=signature.experience_type,
            memory_role="answer_serialization",
            state_scope="answer_serialization",
            mechanism_family="output_representation",
            repair_family="canonicalize_final_answer",
            applicability_family="answer_serialization",
            failure_transition="the final representation violates the required form",
            repair_action="render the completed result in the required final form",
            applicability_condition="the reasoning is correct and only representation remains",
            verification_action="check final representation without changing the reasoning",
            source_signature_sha256=signature.signature_sha256,
            exclusion_reason="verified format compliance is outside the reasoning process bank",
        )
    return V41CanonicalRepairAtom(
        experience_id=signature.experience_id,
        sample_id=signature.sample_id,
        source_experience_type=signature.experience_type,
        memory_role="unusable",
        state_scope="other",
        mechanism_family="other",
        repair_family="other",
        applicability_family="other",
        failure_transition="no schema compliant canonical transition was recovered",
        repair_action="exclude this unvalidated abstraction from the runtime bank",
        applicability_condition="only validated process atoms may enter clustering",
        verification_action="require a grounded schema compliant canonical atom",
        source_signature_sha256=signature.signature_sha256,
        exclusion_reason="teacher output remained invalid after bounded retries",
    )


def _load_unit_records(path: Path, *, schema: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for record in iter_jsonl(path):
        if record.get("schema_version") != schema:
            raise ValueError(f"Unexpected checkpoint schema in {path}")
        unit_id = str(record.get("unit_id", ""))
        if not unit_id or unit_id in result:
            raise ValueError(f"Missing or duplicate checkpoint unit: {unit_id!r}")
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != canonical_json_sha256(logical):
            raise ValueError(f"Checkpoint record hash mismatch: {unit_id}")
        if record.get("payload_sha256") != canonical_json_sha256(record.get("payload")):
            raise ValueError(f"Checkpoint payload hash mismatch: {unit_id}")
        result[unit_id] = record
    return result


def _unit_matches(
    record: Mapping[str, Any],
    *,
    prompt_version: str,
    input_sha256: str,
    args: argparse.Namespace,
) -> bool:
    return bool(
        record.get("prompt_version") == prompt_version
        and record.get("input_sha256") == input_sha256
        and record.get("teacher", {}).get("model") == args.model
        and record.get("teacher", {}).get("base_url") == args.base_url
        and record.get("teacher", {}).get("temperature") == args.temperature
        and record.get("teacher", {}).get("thinking") == args.thinking
        and isinstance(record.get("payload"), Mapping)
    )


def _checkpoint_record(
    *,
    schema: str,
    unit_id: str,
    prompt_version: str,
    input_sha256: str,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    generation_status: str = "teacher_validated",
) -> dict[str, Any]:
    record = {
        "schema_version": schema,
        "unit_id": unit_id,
        "prompt_version": prompt_version,
        "created_at": utc_now(),
        "generation_status": generation_status,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
        },
        "input_sha256": input_sha256,
        "payload": dict(payload),
        "payload_sha256": canonical_json_sha256(payload),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _store_unit_record(
    path: Path,
    records: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    """Durably append a new unit or replace one incompatible unit in place."""

    unit_id = str(record["unit_id"])
    replacing = unit_id in records
    records[unit_id] = record
    if replacing:
        _write_jsonl(path, (records[key] for key in sorted(records)))
    else:
        _append_jsonl(path, record)


def canonicalize_signatures(
    signatures: Sequence[V4RepairSignature],
    *,
    checkpoint_path: Path,
    client: TeacherClient,
    args: argparse.Namespace,
) -> tuple[V41CanonicalRepairAtom, ...]:
    eligible = tuple(item for item in signatures if item.applicable)
    existing = _load_unit_records(checkpoint_path, schema=CANONICAL_UNIT_RECORD_SCHEMA)
    if not args.resume and existing:
        raise ValueError(f"Refusing to overwrite canonical checkpoint: {checkpoint_path}")
    atoms: list[V41CanonicalRepairAtom] = []

    def resolve(unit_id: str, batch: tuple[V4RepairSignature, ...]) -> None:
        input_value = [item.to_dict() for item in batch]
        input_sha256 = canonical_json_sha256(input_value)
        record = existing.get(unit_id)
        payload: dict[str, Any] | None = None
        if record is not None and _unit_matches(
            record,
            prompt_version=V4_1_CANONICAL_PROMPT_VERSION,
            input_sha256=input_sha256,
            args=args,
        ) and record.get("generation_status") in {
            "teacher_validated",
            "deterministic_unusable_after_normalized_invalid_teacher_response",
        }:
            payload = dict(record["payload"])
            _parse_canonical_payload(
                json.dumps(payload, ensure_ascii=False), signatures=batch
            )
        elif any(key.startswith(f"{unit_id}-") for key in existing):
            middle = len(batch) // 2
            if middle <= 0:
                raise ValueError(f"Incomplete split checkpoint for {unit_id}")
            resolve(f"{unit_id}-l", batch[:middle])
            resolve(f"{unit_id}-r", batch[middle:])
            return
        else:
            messages = canonical_atom_messages(batch)
            request_size = sum(len(item["content"]) for item in messages)
            if request_size > MAX_REQUEST_CHARACTERS:
                raise ValueError(f"Canonical unit {unit_id} exceeds request limit")
            try:
                payload = client.call(
                    messages,
                    response_parser=lambda content: _parse_canonical_payload(
                        content, signatures=batch
                    ),
                    request_label="v4.1-canonical",
                    expose_parser_error=True,
                    repair_parser_errors=True,
                )
            except TeacherInvalidResponseError:
                if len(batch) > 1:
                    middle = len(batch) // 2
                    resolve(f"{unit_id}-l", batch[:middle])
                    resolve(f"{unit_id}-r", batch[middle:])
                    return
                atom = _fallback_unusable_atom(batch[0])
                payload = {
                    "schema_version": V4_1_CANONICAL_PAYLOAD_SCHEMA,
                    "atoms": {batch[0].experience_id: atom.to_dict()},
                }
                status = (
                    "deterministic_unusable_after_normalized_invalid_teacher_response"
                )
            else:
                status = "teacher_validated"
            record = _checkpoint_record(
                schema=CANONICAL_UNIT_RECORD_SCHEMA,
                unit_id=unit_id,
                prompt_version=V4_1_CANONICAL_PROMPT_VERSION,
                input_sha256=input_sha256,
                payload=payload,
                args=args,
                generation_status=status,
            )
            _store_unit_record(checkpoint_path, existing, record)
        assert payload is not None
        by_id = {item.experience_id: item for item in batch}
        atoms.extend(
            parse_v4_1_canonical_atom(payload["atoms"][experience_id], signature=by_id[experience_id])
            for experience_id in sorted(by_id)
        )

    batches = tuple(
        eligible[start : start + args.canonical_batch_size]
        for start in range(0, len(eligible), args.canonical_batch_size)
    )
    for index, batch in enumerate(batches):
        resolve(f"canonical-{index:05d}", batch)
        print(
            f"[v4.1-bank] canonical {index + 1}/{len(batches)} input={len(batch)}",
            flush=True,
        )
    result = tuple(sorted(atoms, key=lambda item: item.experience_id))
    if len(result) != len(eligible) or len({item.experience_id for item in result}) != len(result):
        raise ValueError("V4.1 canonicalization coverage mismatch")
    return result


def embed_atom_texts(
    atoms: Sequence[V41CanonicalRepairAtom],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Embed canonical process text with the immutable BGE checkpoint."""

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError("torch and transformers are required for V4.1 embedding") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        V4_1_EMBEDDING_MODEL,
        revision=V4_1_EMBEDDING_REVISION,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        V4_1_EMBEDDING_MODEL,
        revision=V4_1_EMBEDDING_REVISION,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    rows: list[np.ndarray] = []
    texts = [item.embedding_text for item in atoms]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            rows.append(hidden.cpu().numpy().astype(np.float32, copy=False))
    if not rows:
        raise ValueError("V4.1 has no reasoning atoms to embed")
    return np.concatenate(rows, axis=0)


def load_or_build_embeddings(
    atoms: Sequence[V41CanonicalRepairAtom],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> np.ndarray:
    tensor_path = output_dir / "canonical_embeddings.npy"
    manifest_path = output_dir / "canonical_embeddings_manifest.json"
    atom_order = [item.atom_id for item in atoms]
    text_hash = canonical_json_sha256([item.embedding_text for item in atoms])
    if args.resume and tensor_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logical = {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
        if (
            manifest.get("schema_version") == EMBEDDING_MANIFEST_SCHEMA
            and manifest.get("manifest_sha256") == canonical_json_sha256(logical)
            and manifest.get("model") == V4_1_EMBEDDING_MODEL
            and manifest.get("revision") == V4_1_EMBEDDING_REVISION
            and manifest.get("atom_order_sha256") == canonical_json_sha256(atom_order)
            and manifest.get("embedding_text_sha256") == text_hash
            and manifest.get("tensor_sha256") == file_sha256(tensor_path)
        ):
            values = np.load(tensor_path, allow_pickle=False)
            expected_shape = tuple(manifest.get("shape", ()))
            if values.dtype == np.float32 and values.shape == expected_shape:
                return values
    elif not args.resume and (tensor_path.exists() or manifest_path.exists()):
        raise ValueError("Refusing to overwrite V4.1 embedding artifacts")

    values = embed_atom_texts(
        atoms,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    if values.ndim != 2 or values.shape[0] != len(atoms):
        raise ValueError("V4.1 embedding output shape mismatch")
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(values)) or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("V4.1 embeddings are not finite unit vectors")
    np.save(tensor_path, values, allow_pickle=False)
    manifest = {
        "schema_version": EMBEDDING_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "model": V4_1_EMBEDDING_MODEL,
        "revision": V4_1_EMBEDDING_REVISION,
        "pooling": "cls",
        "normalization": "l2",
        "dtype": "float32",
        "shape": list(values.shape),
        "atom_order_sha256": canonical_json_sha256(atom_order),
        "embedding_text_sha256": text_hash,
        "tensor_sha256": file_sha256(tensor_path),
    }
    logical = {key: value for key, value in manifest.items() if key != "created_at"}
    manifest["manifest_sha256"] = canonical_json_sha256(logical)
    _write_json(manifest_path, manifest)
    return values


def _distinct_representatives(
    member_ids: Sequence[str],
    *,
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    embedding_by_id: Mapping[str, np.ndarray],
    limit: int,
) -> tuple[str, ...]:
    """Deterministic farthest-first representatives with distinct samples."""

    candidates: list[str] = []
    seen_samples: set[str] = set()
    for experience_id in sorted(member_ids):
        sample_id = atoms_by_id[experience_id].sample_id
        if sample_id not in seen_samples:
            candidates.append(experience_id)
            seen_samples.add(sample_id)
    if not candidates:
        return ()
    selected = [candidates[0]]
    while len(selected) < min(limit, len(candidates)):
        remaining = [item for item in candidates if item not in selected]

        def min_distance(experience_id: str) -> float:
            vector = embedding_by_id[experience_id]
            return min(
                1.0 - float(np.dot(vector, embedding_by_id[chosen]))
                for chosen in selected
            )

        selected.append(min(remaining, key=lambda item: (-min_distance(item), item)))
    return tuple(selected)


def build_exact_seeds(
    atoms: Sequence[V41CanonicalRepairAtom], embeddings: np.ndarray
) -> tuple[dict[str, Any], ...]:
    if embeddings.shape[0] != len(atoms):
        raise ValueError("V4.1 atom and embedding counts differ")
    reasoning = [item for item in atoms if item.memory_role == "reasoning_process"]
    index_by_id = {item.experience_id: index for index, item in enumerate(atoms)}
    groups: dict[tuple[str, ...], list[V41CanonicalRepairAtom]] = {}
    for atom in reasoning:
        groups.setdefault(atom.canonical_key, []).append(atom)
    result: list[dict[str, Any]] = []
    for canonical_key in sorted(groups):
        members = sorted(groups[canonical_key], key=lambda item: item.experience_id)
        member_ids = [item.experience_id for item in members]
        matrix = np.stack([embeddings[index_by_id[item]] for item in member_ids])
        centroid = matrix.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("V4.1 seed centroid is invalid")
        centroid = (centroid / norm).astype(np.float32, copy=False)
        seed_id = f"seed-{canonical_json_sha256(list(canonical_key))[:20]}"
        result.append(
            {
                "seed_id": seed_id,
                "canonical_key": list(canonical_key),
                "state_scope": members[0].state_scope,
                "mechanism_family": members[0].mechanism_family,
                "repair_family": members[0].repair_family,
                "applicability_family": members[0].applicability_family,
                "member_experience_ids": member_ids,
                "distinct_sample_count": len({item.sample_id for item in members}),
                "source_experience_type_distribution": dict(
                    sorted(Counter(item.source_experience_type for item in members).items())
                ),
                "centroid": centroid,
            }
        )
    if len({item["seed_id"] for item in result}) != len(result):
        raise ValueError("V4.1 deterministic seed ID collision")
    return tuple(sorted(result, key=lambda item: item["seed_id"]))


def build_candidate_pairs(
    seeds: Sequence[Mapping[str, Any]], *, neighbor_count: int
) -> tuple[dict[str, Any], ...]:
    """Retrieve cross-type candidates; retrieval proposes but never merges."""

    if neighbor_count <= 0:
        raise ValueError("V4.1 neighbor count must be positive")
    if len(seeds) < 2:
        return ()
    vectors = np.stack([np.asarray(item["centroid"], dtype=np.float32) for item in seeds])
    similarity = vectors @ vectors.T
    candidates: dict[tuple[int, int], set[str]] = {}

    def add_top(index: int, pool: Sequence[int], source: str) -> None:
        ordered = sorted(
            (other for other in pool if other != index),
            key=lambda other: (-float(similarity[index, other]), str(seeds[other]["seed_id"])),
        )[:neighbor_count]
        for other in ordered:
            pair = tuple(sorted((index, other)))
            candidates.setdefault(pair, set()).add(source)

    all_indices = tuple(range(len(seeds)))
    for index, seed in enumerate(seeds):
        add_top(index, all_indices, "global_semantic")
        add_top(
            index,
            tuple(
                other
                for other, value in enumerate(seeds)
                if value["repair_family"] == seed["repair_family"]
            ),
            "same_repair_family",
        )
        add_top(
            index,
            tuple(
                other
                for other, value in enumerate(seeds)
                if value["mechanism_family"] == seed["mechanism_family"]
            ),
            "same_mechanism_family",
        )
    result = []
    for (left_index, right_index), sources in sorted(
        candidates.items(),
        key=lambda item: (
            str(seeds[item[0][0]]["seed_id"]),
            str(seeds[item[0][1]]["seed_id"]),
        ),
    ):
        left = str(seeds[left_index]["seed_id"])
        right = str(seeds[right_index]["seed_id"])
        if left > right:
            left, right = right, left
            left_index, right_index = right_index, left_index
        pair_id = f"pair-{canonical_json_sha256([left, right])[:20]}"
        result.append(
            {
                "pair_id": pair_id,
                "left_seed_id": left,
                "right_seed_id": right,
                "cosine_similarity": float(similarity[left_index, right_index]),
                "retrieval_sources": sorted(sources),
            }
        )
    return tuple(result)


def _seed_summary(
    seed: Mapping[str, Any],
    *,
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    embedding_by_id: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    representatives = _distinct_representatives(
        seed["member_experience_ids"],
        atoms_by_id=atoms_by_id,
        embedding_by_id=embedding_by_id,
        limit=3,
    )
    return {
        key: seed[key]
        for key in (
            "seed_id",
            "state_scope",
            "mechanism_family",
            "repair_family",
            "applicability_family",
            "distinct_sample_count",
            "source_experience_type_distribution",
        )
    } | {
        "representative_atoms": [atoms_by_id[item].to_dict() for item in representatives]
    }


def pair_judgment_messages(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seeds_by_id: Mapping[str, Mapping[str, Any]],
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    embedding_by_id: Mapping[str, np.ndarray],
) -> list[dict[str, str]]:
    compact = []
    for pair in pairs:
        compact.append(
            {
                **dict(pair),
                "left": _seed_summary(
                    seeds_by_id[str(pair["left_seed_id"])],
                    atoms_by_id=atoms_by_id,
                    embedding_by_id=embedding_by_id,
                ),
                "right": _seed_summary(
                    seeds_by_id[str(pair["right_seed_id"])],
                    atoms_by_id=atoms_by_id,
                    embedding_by_id=embedding_by_id,
                ),
            }
        )
    system = """You are the V4.1 repair-cluster edge judge. Return JSON only.
For every retrieved pair, decide whether both seeds describe the same reusable
reasoning failure transition, the same corrective action, and compatible
applicability. The cosine score is recall metadata, never proof. The original
verifier outcome types are provenance and may differ. Reject topical similarity,
same broad arithmetic, merely related repairs, output formatting, or any pair
whose representatives reveal materially different state transitions.

Set merge true exactly when all four component booleans are true. A merged pair
must have no issues. A rejected pair must list at least one concise issue.
Evidence and issues must remain process-only, with no names, digits, answers,
equations, formulas, or story-specific details."""
    user = f"""Candidate pairs:
{json.dumps(compact, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "schema_version": "{V4_1_PAIR_PAYLOAD_SCHEMA}",
  "judgments": {{
    "exact-pair-id": {{
      "same_failure_mechanism": true,
      "same_repair_action": true,
      "compatible_applicability": true,
      "process_only": true,
      "merge": true,
      "evidence": "concise rationale",
      "issues": []
    }}
  }}
}}
Cover every supplied pair_id exactly once and no others."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_pair_payload(
    content: str, *, pairs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload = _parse_json_object(content)
    if payload.get("schema_version") != V4_1_PAIR_PAYLOAD_SCHEMA:
        raise ValueError("Unexpected V4.1 pair payload schema")
    values = payload.get("judgments")
    if not isinstance(values, Mapping):
        raise ValueError("V4.1 pair payload is missing judgments")
    expected = {str(item["pair_id"]) for item in pairs}
    if set(values) != expected:
        raise ValueError("V4.1 pair payload does not cover its exact input")
    pairs_by_id = {str(item["pair_id"]): item for item in pairs}
    for pair_id, value in values.items():
        if not isinstance(value, Mapping):
            raise ValueError("V4.1 pair judgment must be an object")
        pair = pairs_by_id[pair_id]
        parse_v4_1_pair_judgment(
            value,
            pair_id=pair_id,
            left_seed_id=str(pair["left_seed_id"]),
            right_seed_id=str(pair["right_seed_id"]),
        )
    return payload


def _fallback_pair_judgment(pair: Mapping[str, Any]) -> V41PairJudgment:
    return V41PairJudgment(
        pair_id=str(pair["pair_id"]),
        left_seed_id=str(pair["left_seed_id"]),
        right_seed_id=str(pair["right_seed_id"]),
        same_failure_mechanism=False,
        same_repair_action=False,
        compatible_applicability=False,
        process_only=False,
        merge=False,
        evidence="no schema compliant pair judgment was recovered",
        issues=("the candidate edge was not validated",),
    )


def judge_candidate_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[Mapping[str, Any]],
    atoms: Sequence[V41CanonicalRepairAtom],
    embeddings: np.ndarray,
    checkpoint_path: Path,
    client: TeacherClient,
    args: argparse.Namespace,
) -> tuple[V41PairJudgment, ...]:
    existing = _load_unit_records(checkpoint_path, schema=PAIR_UNIT_RECORD_SCHEMA)
    if not args.resume and existing:
        raise ValueError(f"Refusing to overwrite pair checkpoint: {checkpoint_path}")
    seeds_by_id = {str(item["seed_id"]): item for item in seeds}
    atoms_by_id = {item.experience_id: item for item in atoms}
    embedding_by_id = {
        item.experience_id: embeddings[index] for index, item in enumerate(atoms)
    }
    judgments: list[V41PairJudgment] = []

    def resolve(unit_id: str, batch: tuple[Mapping[str, Any], ...]) -> None:
        input_value = [dict(item) for item in batch]
        input_sha256 = canonical_json_sha256(input_value)
        record = existing.get(unit_id)
        payload: dict[str, Any] | None = None
        if record is not None and _unit_matches(
            record,
            prompt_version=V4_1_PAIR_PROMPT_VERSION,
            input_sha256=input_sha256,
            args=args,
        ):
            payload = dict(record["payload"])
            _parse_pair_payload(json.dumps(payload, ensure_ascii=False), pairs=batch)
        elif any(key.startswith(f"{unit_id}-") for key in existing):
            middle = len(batch) // 2
            if middle <= 0:
                raise ValueError(f"Incomplete split checkpoint for {unit_id}")
            resolve(f"{unit_id}-l", batch[:middle])
            resolve(f"{unit_id}-r", batch[middle:])
            return
        else:
            messages = pair_judgment_messages(
                batch,
                seeds_by_id=seeds_by_id,
                atoms_by_id=atoms_by_id,
                embedding_by_id=embedding_by_id,
            )
            request_size = sum(len(item["content"]) for item in messages)
            if request_size > MAX_REQUEST_CHARACTERS:
                raise ValueError(f"Pair unit {unit_id} exceeds request limit")
            try:
                payload = client.call(
                    messages,
                    response_parser=lambda content: _parse_pair_payload(content, pairs=batch),
                    request_label="v4.1-pair",
                    expose_parser_error=True,
                    repair_parser_errors=True,
                )
            except TeacherInvalidResponseError:
                if len(batch) > 1:
                    middle = len(batch) // 2
                    resolve(f"{unit_id}-l", batch[:middle])
                    resolve(f"{unit_id}-r", batch[middle:])
                    return
                fallback = _fallback_pair_judgment(batch[0])
                payload = {
                    "schema_version": V4_1_PAIR_PAYLOAD_SCHEMA,
                    "judgments": {fallback.pair_id: fallback.to_dict()},
                }
                status = "deterministic_rejection_after_invalid_teacher_response"
            else:
                status = "teacher_validated"
            record = _checkpoint_record(
                schema=PAIR_UNIT_RECORD_SCHEMA,
                unit_id=unit_id,
                prompt_version=V4_1_PAIR_PROMPT_VERSION,
                input_sha256=input_sha256,
                payload=payload,
                args=args,
                generation_status=status,
            )
            _store_unit_record(checkpoint_path, existing, record)
        assert payload is not None
        by_id = {str(item["pair_id"]): item for item in batch}
        for pair_id in sorted(by_id):
            pair = by_id[pair_id]
            judgments.append(
                parse_v4_1_pair_judgment(
                    payload["judgments"][pair_id],
                    pair_id=pair_id,
                    left_seed_id=str(pair["left_seed_id"]),
                    right_seed_id=str(pair["right_seed_id"]),
                )
            )

    batches = tuple(
        tuple(pairs[start : start + args.pair_batch_size])
        for start in range(0, len(pairs), args.pair_batch_size)
    )
    for index, batch in enumerate(batches):
        resolve(f"pair-{index:05d}", batch)
        print(
            f"[v4.1-bank] pair judge {index + 1}/{len(batches)} input={len(batch)}",
            flush=True,
        )
    result = tuple(sorted(judgments, key=lambda item: item.pair_id))
    if len(result) != len(pairs) or len({item.pair_id for item in result}) != len(result):
        raise ValueError("V4.1 pair-judgment coverage mismatch")
    return result


def form_clique_candidates(
    seeds: Sequence[Mapping[str, Any]],
    judgments: Sequence[V41PairJudgment],
) -> tuple[dict[str, Any], ...]:
    """Greedily partition positive graph into deterministic complete-link groups."""

    positive = {
        (item.left_seed_id, item.right_seed_id)
        for item in judgments
        if item.merge
    }
    order = sorted(
        (str(item["seed_id"]) for item in seeds),
        key=lambda seed_id: (
            -next(
                int(seed["distinct_sample_count"])
                for seed in seeds
                if seed["seed_id"] == seed_id
            ),
            seed_id,
        ),
    )
    groups: list[list[str]] = []
    for seed_id in order:
        compatible: list[tuple[int, tuple[str, ...]]] = []
        for index, group in enumerate(groups):
            if all(tuple(sorted((seed_id, other))) in positive for other in group):
                compatible.append((index, tuple(group)))
        if compatible:
            target = min(compatible, key=lambda item: (-len(item[1]), item[1]))[0]
            groups[target].append(seed_id)
            groups[target].sort()
        else:
            groups.append([seed_id])
    seeds_by_id = {str(item["seed_id"]): item for item in seeds}
    result: list[dict[str, Any]] = []
    for group in groups:
        members = sorted(
            {
                str(experience_id)
                for seed_id in group
                for experience_id in seeds_by_id[seed_id]["member_experience_ids"]
            }
        )
        result.append(
            {
                "candidate_id": f"candidate-{canonical_json_sha256(group)[:20]}",
                "seed_ids": sorted(group),
                "member_experience_ids": members,
            }
        )
    return tuple(sorted(result, key=lambda item: item["candidate_id"]))


def cluster_audit_messages(
    candidate: Mapping[str, Any],
    *,
    representatives: Sequence[str],
    seeds_by_id: Mapping[str, Mapping[str, Any]],
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    signatures_by_id: Mapping[str, V4RepairSignature],
) -> list[dict[str, str]]:
    compact_seeds = [
        {
            key: seeds_by_id[seed_id][key]
            for key in (
                "seed_id",
                "state_scope",
                "mechanism_family",
                "repair_family",
                "applicability_family",
                "distinct_sample_count",
                "source_experience_type_distribution",
            )
        }
        for seed_id in candidate["seed_ids"]
    ]
    evidence = [
        {
            "canonical_atom": atoms_by_id[experience_id].to_dict(),
            "source_signature": signatures_by_id[experience_id].to_dict(),
        }
        for experience_id in representatives
    ]
    system = """You are the final V4.1 repair-cluster coherence auditor.
Return JSON only. Inspect five to ten independent representative signatures.
Approve only when all examples share one reasoning-state failure mechanism,
one corrective operator, and a compatible applicability boundary. Reject a
bundle that merely shares a topic, broad operation family, or endpoint. Reject
all final-answer serialization, output-format compliance, question-specific
content, and any unsupported abstraction. The source experience_type labels
are verifier provenance and must not control the decision.

For an approved candidate, write one concise process-only title, shared failure
mechanism, repair operator, and scope summary. Remove names, digits, answers,
equations, formulas, quantities, and story traces. Set approve true exactly
when all five component checks are true. Approved output has no issues;
rejected output lists at least one issue."""
    user = f"""Candidate:
{json.dumps(dict(candidate), ensure_ascii=False, sort_keys=True)}

Canonical seeds:
{json.dumps(compact_seeds, ensure_ascii=False, sort_keys=True)}

Representative evidence:
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "coherent": true,
  "process_only": true,
  "transferable": true,
  "serialization_free": true,
  "leakage_free": true,
  "approve": true,
  "title": "short process title",
  "failure_mechanism": "one shared failure transition",
  "repair_operator": "one shared corrective action",
  "scope_summary": "shared applicability boundary",
  "evidence": "concise audit rationale",
  "issues": []
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _fallback_cluster_audit(candidate_id: str) -> V41ClusterAudit:
    return V41ClusterAudit(
        candidate_id=candidate_id,
        coherent=False,
        process_only=False,
        transferable=False,
        serialization_free=False,
        leakage_free=False,
        approve=False,
        title="unvalidated candidate",
        failure_mechanism="no validated shared failure mechanism was recovered",
        repair_operator="exclude the unvalidated candidate from runtime use",
        scope_summary="only audited coherent process clusters may enter the bank",
        evidence="no schema compliant coherence audit was recovered",
        issues=("the candidate audit was not validated",),
    )


def _parse_audit_payload(content: str, *, candidate_id: str) -> dict[str, Any]:
    payload = _parse_json_object(content)
    parse_v4_1_cluster_audit(payload, candidate_id=candidate_id)
    return payload


def audit_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[Mapping[str, Any]],
    atoms: Sequence[V41CanonicalRepairAtom],
    signatures: Sequence[V4RepairSignature],
    embeddings: np.ndarray,
    checkpoint_path: Path,
    client: TeacherClient,
    args: argparse.Namespace,
) -> tuple[tuple[V41RepairCluster, ...], dict[str, Any]]:
    """Audit supported cliques and fall back to supported seeds after rejection."""

    existing = _load_unit_records(checkpoint_path, schema=AUDIT_UNIT_RECORD_SCHEMA)
    if not args.resume and existing:
        raise ValueError(f"Refusing to overwrite audit checkpoint: {checkpoint_path}")
    seeds_by_id = {str(item["seed_id"]): item for item in seeds}
    atoms_by_id = {item.experience_id: item for item in atoms}
    signatures_by_id = {item.experience_id: item for item in signatures}
    embedding_by_id = {
        item.experience_id: embeddings[index] for index, item in enumerate(atoms)
    }
    approved: list[V41RepairCluster] = []
    unsupported: set[str] = set()
    audit_rejected: set[str] = set()
    audit_summaries: list[dict[str, Any]] = []

    def audit_one(candidate: Mapping[str, Any], *, fallback: bool) -> V41ClusterAudit:
        member_ids = tuple(sorted(str(item) for item in candidate["member_experience_ids"]))
        representatives = _distinct_representatives(
            member_ids,
            atoms_by_id=atoms_by_id,
            embedding_by_id=embedding_by_id,
            limit=10,
        )
        if len(representatives) < 5:
            raise ValueError("V4.1 attempted to audit fewer than five distinct examples")
        input_value = {
            "candidate": dict(candidate),
            "representatives": list(representatives),
            "source_signature_sha256": {
                item: signatures_by_id[item].signature_sha256 for item in representatives
            },
            "canonical_atom_sha256": {
                item: canonical_json_sha256(atoms_by_id[item].to_dict())
                for item in representatives
            },
        }
        input_sha256 = canonical_json_sha256(input_value)
        candidate_id = str(candidate["candidate_id"])
        unit_id = f"audit-{candidate_id}"
        record = existing.get(unit_id)
        audit: V41ClusterAudit | None = None
        if record is not None and _unit_matches(
            record,
            prompt_version=V4_1_AUDIT_PROMPT_VERSION,
            input_sha256=input_sha256,
            args=args,
        ):
            audit = parse_v4_1_cluster_audit(record["payload"], candidate_id=candidate_id)
        else:
            messages = cluster_audit_messages(
                candidate,
                representatives=representatives,
                seeds_by_id=seeds_by_id,
                atoms_by_id=atoms_by_id,
                signatures_by_id=signatures_by_id,
            )
            request_size = sum(len(item["content"]) for item in messages)
            if request_size > MAX_REQUEST_CHARACTERS:
                raise ValueError(f"Audit unit {candidate_id} exceeds request limit")
            try:
                payload = client.call(
                    messages,
                    response_parser=lambda content: _parse_audit_payload(
                        content, candidate_id=candidate_id
                    ),
                    request_label="v4.1-audit",
                    expose_parser_error=True,
                    repair_parser_errors=True,
                )
                audit = parse_v4_1_cluster_audit(payload, candidate_id=candidate_id)
                status = "teacher_validated"
            except TeacherInvalidResponseError:
                audit = _fallback_cluster_audit(candidate_id)
                payload = audit.to_dict()
                status = "deterministic_rejection_after_invalid_teacher_response"
            record = _checkpoint_record(
                schema=AUDIT_UNIT_RECORD_SCHEMA,
                unit_id=unit_id,
                prompt_version=V4_1_AUDIT_PROMPT_VERSION,
                input_sha256=input_sha256,
                payload=payload,
                args=args,
                generation_status=status,
            )
            _store_unit_record(checkpoint_path, existing, record)
        assert audit is not None
        audit_summaries.append(
            {
                "candidate_id": candidate_id,
                "fallback_seed_audit": fallback,
                "member_count": len(member_ids),
                "distinct_sample_count": len(
                    {atoms_by_id[item].sample_id for item in member_ids}
                ),
                "representative_experience_ids": list(representatives),
                "audit": audit.to_dict(),
                "audit_sha256": canonical_json_sha256(audit.to_dict()),
            }
        )
        if audit.approve:
            distribution = tuple(
                sorted(Counter(atoms_by_id[item].source_experience_type for item in member_ids).items())
            )
            semantic = {
                "candidate_id": candidate_id,
                "seed_ids": list(candidate["seed_ids"]),
                "members": list(member_ids),
                "audit": audit.to_dict(),
            }
            cluster_key = f"repair-{canonical_json_sha256(semantic)[:20]}"
            approved.append(
                V41RepairCluster(
                    cluster_key=cluster_key,
                    candidate_id=candidate_id,
                    title=audit.title,
                    failure_mechanism=audit.failure_mechanism,
                    repair_operator=audit.repair_operator,
                    scope_summary=audit.scope_summary,
                    member_experience_ids=member_ids,
                    representative_experience_ids=representatives,
                    source_experience_type_distribution=distribution,
                    canonical_seed_ids=tuple(sorted(str(item) for item in candidate["seed_ids"])),
                    audit_sha256=canonical_json_sha256(audit.to_dict()),
                )
            )
        return audit

    for index, candidate in enumerate(candidates, start=1):
        member_ids = tuple(str(item) for item in candidate["member_experience_ids"])
        distinct = {atoms_by_id[item].sample_id for item in member_ids}
        if len(distinct) < 5:
            unsupported.update(member_ids)
            continue
        audit = audit_one(candidate, fallback=False)
        if not audit.approve:
            if len(candidate["seed_ids"]) == 1:
                audit_rejected.update(member_ids)
            else:
                for seed_id in candidate["seed_ids"]:
                    seed = seeds_by_id[str(seed_id)]
                    seed_members = tuple(str(item) for item in seed["member_experience_ids"])
                    seed_distinct = {atoms_by_id[item].sample_id for item in seed_members}
                    if len(seed_distinct) < 5:
                        unsupported.update(seed_members)
                        continue
                    fallback_candidate = {
                        "candidate_id": (
                            "candidate-seed-"
                            f"{canonical_json_sha256([seed_id])[:20]}"
                        ),
                        "seed_ids": [seed_id],
                        "member_experience_ids": list(seed_members),
                    }
                    seed_audit = audit_one(fallback_candidate, fallback=True)
                    if not seed_audit.approve:
                        audit_rejected.update(seed_members)
        print(
            f"[v4.1-bank] audit {index}/{len(candidates)} "
            f"candidate={candidate['candidate_id']} approved={audit.approve}",
            flush=True,
        )

    approved = sorted(approved, key=lambda item: item.cluster_key)
    assigned = [
        experience_id
        for cluster in approved
        for experience_id in cluster.member_experience_ids
    ]
    if len(set(assigned)) != len(assigned):
        raise ValueError("V4.1 approved clusters overlap")
    reasoning_ids = {
        item.experience_id for item in atoms if item.memory_role == "reasoning_process"
    }
    covered = set(assigned) | unsupported | audit_rejected
    if covered != reasoning_ids:
        raise ValueError(
            "V4.1 reasoning coverage mismatch: "
            f"missing={sorted(reasoning_ids - covered)[:5]} "
            f"extra={sorted(covered - reasoning_ids)[:5]}"
        )
    return tuple(approved), {
        "audits": sorted(audit_summaries, key=lambda item: item["candidate_id"]),
        "unsupported_reasoning_experience_ids": sorted(unsupported),
        "audit_rejected_experience_ids": sorted(audit_rejected),
    }


def _serialize_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in seed.items() if key != "centroid"}


def build_cluster_plan(
    *,
    signatures: Sequence[V4RepairSignature],
    atoms: Sequence[V41CanonicalRepairAtom],
    seeds: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    judgments: Sequence[V41PairJudgment],
    clusters: Sequence[V41RepairCluster],
    audit_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    nonapplicable = sorted(item.experience_id for item in signatures if not item.applicable)
    answer_serialization = sorted(
        item.experience_id for item in atoms if item.memory_role == "answer_serialization"
    )
    unusable = sorted(item.experience_id for item in atoms if item.memory_role == "unusable")
    payload = {
        "schema_version": V4_1_CLUSTER_PLAN_SCHEMA,
        "clusters": [item.to_dict() for item in clusters],
        "archive": {
            "source_nonapplicable_experience_ids": nonapplicable,
            "answer_serialization_experience_ids": answer_serialization,
            "unusable_experience_ids": unusable,
            "unsupported_reasoning_experience_ids": list(
                audit_diagnostics["unsupported_reasoning_experience_ids"]
            ),
            "audit_rejected_experience_ids": list(
                audit_diagnostics["audit_rejected_experience_ids"]
            ),
        },
        "diagnostics": {
            "source_signature_count": len(signatures),
            "applicable_signature_count": sum(item.applicable for item in signatures),
            "canonical_atom_count": len(atoms),
            "memory_role_counts": dict(sorted(Counter(item.memory_role for item in atoms).items())),
            "exact_seed_count": len(seeds),
            "candidate_pair_count": len(pairs),
            "positive_pair_count": sum(item.merge for item in judgments),
            "approved_cluster_count": len(clusters),
            "approved_member_count": sum(len(item.member_experience_ids) for item in clusters),
            "cross_type_cluster_count": sum(
                len(item.source_experience_type_distribution) > 1 for item in clusters
            ),
            "audit_count": len(audit_diagnostics["audits"]),
        },
        "audits": list(audit_diagnostics["audits"]),
    }
    all_ids = {item.experience_id for item in signatures}
    covered = {
        item
        for values in payload["archive"].values()
        for item in values
    } | {
        item
        for cluster in clusters
        for item in cluster.member_experience_ids
    }
    if covered != all_ids:
        raise ValueError("V4.1 cluster plan does not account for every source signature")
    if sum(len(values) for values in payload["archive"].values()) + sum(
        len(item.member_experience_ids) for item in clusters
    ) != len(all_ids):
        raise ValueError("V4.1 cluster plan categories overlap")
    return payload


def _representative_evidence(
    cluster: V41RepairCluster,
    *,
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "experience_id": experience_id,
            "sample_id": examples_by_id[experience_id]["sample_id"],
            "question": examples_by_id[experience_id]["question"],
            "official_solution": examples_by_id[experience_id]["official_solution"],
            "verified_success_trajectory": examples_by_id[experience_id][
                "verified_success_trajectory"
            ],
            "verified_failure_trajectory": examples_by_id[experience_id][
                "verified_failure_trajectory"
            ],
            "reference_verifier": examples_by_id[experience_id]["reference_verifier"],
        }
        for experience_id in cluster.representative_experience_ids
    ]


def process_card_messages(
    cluster: V41RepairCluster,
    *,
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    signatures_by_id: Mapping[str, V4RepairSignature],
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    representatives = [
        {
            "canonical_atom": atoms_by_id[item].to_dict(),
            "source_signature": signatures_by_id[item].to_dict(),
        }
        for item in cluster.representative_experience_ids
    ]
    evidence = _representative_evidence(cluster, examples_by_id=examples_by_id)
    system = """You are the offline process-card synthesizer for MemGen V4.1.
Return JSON only. The cluster has already passed a multi-example coherence audit.
Synthesize one target process card and one descriptive contrastive reference.

Ground target claims only in official solutions and verified-success trajectories.
Ground reference claims only in paired verified-failure trajectories. The target
must specify one reusable reasoning-state correction; the reference must describe
the recurring undesired process and must never instruct the model to perform it.
Remove all question text, names, story details, answers, digits, equations,
constants, formulas, and source-solution traces. Do not introduce output-format
or answer-serialization advice. Keep the applicability boundary and verification.
Do not combine unrelated repairs."""
    user = f"""Audited cluster:
{json.dumps(cluster.to_dict(), ensure_ascii=False, sort_keys=True)}

Representative process abstractions:
{json.dumps(representatives, ensure_ascii=False, sort_keys=True)}

Representative construction evidence:
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "target": {{
    "scope": "when this process is relevant",
    "diagnosis": "what structural reasoning risk is present",
    "action": "the reusable corrective process",
    "verification": "how to check the repaired reasoning",
    "do_not_use_when": "a concrete applicability boundary"
  }},
  "reference": {{
    "undesired_pattern": "descriptive recurring failed process",
    "failure_signal": "observable sign of that process",
    "failure_mechanism": "why that process fails",
    "contrast_boundary": "how it differs from the target process"
  }},
  "support_summary": "why the card is shared across independent examples",
  "target_reference_distinction": "the exact process-level contrast"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def card_review_messages(
    cluster: V41RepairCluster,
    card: V4ProcessCard,
    *,
    atoms_by_id: Mapping[str, V41CanonicalRepairAtom],
    signatures_by_id: Mapping[str, V4RepairSignature],
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    representatives = [
        {
            "canonical_atom": atoms_by_id[item].to_dict(),
            "source_signature": signatures_by_id[item].to_dict(),
        }
        for item in cluster.representative_experience_ids
    ]
    evidence = _representative_evidence(cluster, examples_by_id=examples_by_id)
    system = """You are a strict offline auditor for a V4.1 process card.
Return JSON only and do not rewrite the card. Check target grounding against
correct construction evidence, reference grounding against failed trajectories,
one-process coherence, target/reference contrast, transferability, and absence
of names, answers, digits, equations, formulas, story traces, output formatting,
or answer serialization. Approve exactly when all checks are true. Approved
reviews have no issues; rejected reviews list at least one concise issue."""
    user = f"""Audited cluster:
{json.dumps(cluster.to_dict(), ensure_ascii=False, sort_keys=True)}

Representative abstractions:
{json.dumps(representatives, ensure_ascii=False, sort_keys=True)}

Representative evidence:
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}

Candidate card:
{json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "target_grounded": true,
  "reference_grounded": true,
  "process_only": true,
  "target_reference_distinct": true,
  "transferable": true,
  "leakage_free": true,
  "approve": true,
  "evidence": "concise audit rationale",
  "issues": []
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _record_hash_valid(record: Mapping[str, Any]) -> bool:
    logical = {key: value for key, value in record.items() if key != "record_sha256"}
    return record.get("record_sha256") == canonical_json_sha256(logical)


def _load_keyed_records(
    path: Path, *, schema: str, key: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for record in iter_jsonl(path):
        item_id = str(record.get(key, ""))
        if record.get("schema_version") != schema or not item_id or item_id in result:
            raise ValueError(f"Invalid or duplicate record in {path}: {item_id!r}")
        if not _record_hash_valid(record):
            raise ValueError(f"Record hash mismatch in {path}: {item_id}")
        result[item_id] = record
    return result


def _card_record(
    *,
    cluster_key: str,
    construction_input_sha256: str,
    card: V4ProcessCard | None,
    args: argparse.Namespace,
    generation_status: str,
) -> dict[str, Any]:
    record = {
        "schema_version": CARD_RECORD_SCHEMA,
        "prompt_version": V4_1_CARD_PROMPT_VERSION,
        "created_at": utc_now(),
        "generation_status": generation_status,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
        },
        "cluster_key": cluster_key,
        "construction_input_sha256": construction_input_sha256,
        "card": None if card is None else card.to_dict(),
        "card_sha256": None if card is None else card.card_sha256,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _review_record(
    *,
    cluster_key: str,
    construction_input_sha256: str,
    card: V4ProcessCard,
    review: V4CardReview,
    args: argparse.Namespace,
    generation_status: str,
) -> dict[str, Any]:
    record = {
        "schema_version": REVIEW_RECORD_SCHEMA,
        "prompt_version": V4_1_REVIEW_PROMPT_VERSION,
        "created_at": utc_now(),
        "generation_status": generation_status,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
        },
        "cluster_key": cluster_key,
        "construction_input_sha256": construction_input_sha256,
        "card_sha256": card.card_sha256,
        "review": review.to_dict(),
        "review_sha256": canonical_json_sha256(review.to_dict()),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _record_teacher_matches(record: Mapping[str, Any], args: argparse.Namespace) -> bool:
    teacher = record.get("teacher", {})
    return bool(
        teacher.get("model") == args.model
        and teacher.get("base_url") == args.base_url
        and teacher.get("temperature") == args.temperature
        and teacher.get("thinking") == args.thinking
    )


def synthesize_cards(
    clusters: Sequence[V41RepairCluster],
    *,
    signatures: Sequence[V4RepairSignature],
    atoms: Sequence[V41CanonicalRepairAtom],
    examples: Sequence[Mapping[str, Any]],
    profile: V41ConstructionProfile,
    output_dir: Path,
    api_key: str,
    args: argparse.Namespace,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    cards_path = output_dir / "process_cards.jsonl"
    reviews_path = output_dir / "card_reviews.jsonl"
    existing_cards = _load_keyed_records(cards_path, schema=CARD_RECORD_SCHEMA, key="cluster_key")
    existing_reviews = _load_keyed_records(
        reviews_path, schema=REVIEW_RECORD_SCHEMA, key="cluster_key"
    )
    if not args.resume and (existing_cards or existing_reviews):
        raise ValueError("Refusing to overwrite V4.1 card checkpoints")
    atoms_by_id = {item.experience_id: item for item in atoms}
    signatures_by_id = {item.experience_id: item for item in signatures}
    examples_by_id = {str(item["experience_id"]): item for item in examples}
    bank_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    with _client(args, api_key=api_key, max_tokens=args.card_max_tokens) as card_client, _client(
        args, api_key=api_key, max_tokens=args.review_max_tokens
    ) as review_client:
        for index, cluster in enumerate(clusters, start=1):
            construction_input_sha256 = canonical_json_sha256(
                {
                    "cluster": cluster.to_dict(),
                    "examples": {
                        experience_id: examples_by_id[experience_id]["construction_input_sha256"]
                        for experience_id in cluster.member_experience_ids
                    },
                }
            )
            card_record = existing_cards.get(cluster.cluster_key)
            card: V4ProcessCard | None = None
            permanently_invalid = False
            if (
                card_record is not None
                and card_record.get("prompt_version") == V4_1_CARD_PROMPT_VERSION
                and card_record.get("construction_input_sha256") == construction_input_sha256
                and _record_teacher_matches(card_record, args)
            ):
                if card_record.get("generation_status") == "invalid_teacher_response":
                    permanently_invalid = True
                elif isinstance(card_record.get("card"), Mapping):
                    card = parse_v4_process_card(
                        card_record["card"], cluster_key=cluster.cluster_key
                    )
                    if card_record.get("card_sha256") != card.card_sha256:
                        raise ValueError("V4.1 card checkpoint hash mismatch")
            if card is None and not permanently_invalid:
                try:
                    payload = card_client.call(
                        process_card_messages(
                            cluster,
                            atoms_by_id=atoms_by_id,
                            signatures_by_id=signatures_by_id,
                            examples_by_id=examples_by_id,
                        ),
                        response_parser=lambda content: _parse_card_payload(
                            content, cluster_key=cluster.cluster_key
                        ),
                        request_label="v4.1-card",
                        expose_parser_error=True,
                        repair_parser_errors=True,
                    )
                    card = parse_v4_process_card(payload, cluster_key=cluster.cluster_key)
                    card_record = _card_record(
                        cluster_key=cluster.cluster_key,
                        construction_input_sha256=construction_input_sha256,
                        card=card,
                        args=args,
                        generation_status="teacher_validated",
                    )
                except TeacherInvalidResponseError:
                    permanently_invalid = True
                    card_record = _card_record(
                        cluster_key=cluster.cluster_key,
                        construction_input_sha256=construction_input_sha256,
                        card=None,
                        args=args,
                        generation_status="invalid_teacher_response",
                    )
                _append_jsonl(cards_path, card_record)
                existing_cards[cluster.cluster_key] = card_record
            if card is None:
                rejected.append(
                    {
                        "cluster_key": cluster.cluster_key,
                        "stage": "card",
                        "reason": "teacher output remained invalid after bounded retries",
                    }
                )
                continue

            review_record = existing_reviews.get(cluster.cluster_key)
            review: V4CardReview | None = None
            if (
                review_record is not None
                and review_record.get("prompt_version") == V4_1_REVIEW_PROMPT_VERSION
                and review_record.get("construction_input_sha256") == construction_input_sha256
                and review_record.get("card_sha256") == card.card_sha256
                and _record_teacher_matches(review_record, args)
            ):
                review = parse_v4_card_review(
                    review_record.get("review", {}), cluster_key=cluster.cluster_key
                )
                if review_record.get("review_sha256") != canonical_json_sha256(review.to_dict()):
                    raise ValueError("V4.1 review checkpoint hash mismatch")
            if review is None:
                try:
                    payload = review_client.call(
                        card_review_messages(
                            cluster,
                            card,
                            atoms_by_id=atoms_by_id,
                            signatures_by_id=signatures_by_id,
                            examples_by_id=examples_by_id,
                        ),
                        response_parser=lambda content: _parse_review_payload(
                            content, cluster_key=cluster.cluster_key
                        ),
                        request_label="v4.1-card-review",
                        expose_parser_error=True,
                        repair_parser_errors=True,
                    )
                    review = parse_v4_card_review(payload, cluster_key=cluster.cluster_key)
                    status = "teacher_validated"
                except TeacherInvalidResponseError:
                    review = V4CardReview(
                        cluster_key=cluster.cluster_key,
                        target_grounded=False,
                        reference_grounded=False,
                        process_only=False,
                        target_reference_distinct=False,
                        transferable=False,
                        leakage_free=False,
                        approve=False,
                        evidence="no schema compliant card review was recovered",
                        issues=("the card review was not validated",),
                    )
                    status = "deterministic_rejection_after_invalid_teacher_response"
                review_record = _review_record(
                    cluster_key=cluster.cluster_key,
                    construction_input_sha256=construction_input_sha256,
                    card=card,
                    review=review,
                    args=args,
                    generation_status=status,
                )
                _append_jsonl(reviews_path, review_record)
                existing_reviews[cluster.cluster_key] = review_record
            if review.approve:
                bank_records.append(
                    build_v4_1_bank_record(
                        cluster=cluster,
                        card=card,
                        review=review,
                        signatures=signatures,
                        atoms=atoms,
                        construction_input_sha256=construction_input_sha256,
                        profile=profile,
                    )
                )
            else:
                rejected.append(
                    {
                        "cluster_key": cluster.cluster_key,
                        "stage": "review",
                        "card_sha256": card.card_sha256,
                        "review": review.to_dict(),
                    }
                )
            print(
                f"[v4.1-bank] cards {index}/{len(clusters)} "
                f"{cluster.cluster_key} approved={review.approve}",
                flush=True,
            )
    return tuple(bank_records), tuple(rejected)


def _parse_card_payload(content: str, *, cluster_key: str) -> dict[str, Any]:
    payload = _parse_json_object(content)
    parse_v4_process_card(payload, cluster_key=cluster_key)
    return payload


def _parse_review_payload(content: str, *, cluster_key: str) -> dict[str, Any]:
    payload = _parse_json_object(content)
    parse_v4_card_review(payload, cluster_key=cluster_key)
    return payload


def _profile_record(
    profile: V41ConstructionProfile,
    *,
    source_signature_info: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
        },
        "source_signatures": dict(source_signature_info),
        "prompt_versions": {
            "source_signature": V4_SIGNATURE_PROMPT_VERSION,
            "canonical": V4_1_CANONICAL_PROMPT_VERSION,
            "pair": V4_1_PAIR_PROMPT_VERSION,
            "audit": V4_1_AUDIT_PROMPT_VERSION,
            "card": V4_1_CARD_PROMPT_VERSION,
            "review": V4_1_REVIEW_PROMPT_VERSION,
        },
        "clustering": {
            "method": "canonical_atom_candidate_graph_clique_audit",
            "source_experience_type_is_boundary": False,
            "semantic_retrieval_is_merge_decision": False,
            "linkage": "complete_positive_edge",
            "minimum_distinct_support": 5,
            "maximum_audit_representatives": 10,
            "max_request_characters": MAX_REQUEST_CHARACTERS,
        },
    }


def _write_or_validate_profile(
    path: Path,
    *,
    expected: Mapping[str, Any],
    resume: bool,
) -> None:
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise ValueError("V4.1 construction profile differs from existing output")
        if not resume:
            raise ValueError("V4.1 output already exists; pass --resume to authenticate it")
        return
    _write_json(path, expected)


def _atom_record(atom: V41CanonicalRepairAtom) -> dict[str, Any]:
    return {
        "schema_version": V4_1_CANONICAL_ATOM_SCHEMA,
        "atom": atom.to_dict(),
        "atom_sha256": canonical_json_sha256(atom.to_dict()),
    }


def _load_atoms(
    path: Path, *, signatures: Sequence[V4RepairSignature]
) -> tuple[V41CanonicalRepairAtom, ...]:
    signatures_by_id = {item.experience_id: item for item in signatures}
    result: list[V41CanonicalRepairAtom] = []
    seen: set[str] = set()
    for record in iter_jsonl(path):
        if record.get("schema_version") != V4_1_CANONICAL_ATOM_SCHEMA:
            raise ValueError("Unexpected V4.1 canonical-atom artifact schema")
        payload = record.get("atom", {})
        experience_id = str(payload.get("experience_id", ""))
        signature = signatures_by_id.get(experience_id)
        if signature is None or experience_id in seen:
            raise ValueError("V4.1 canonical-atom artifact has unknown or duplicate ID")
        atom = parse_v4_1_canonical_atom(payload, signature=signature)
        if record.get("atom_sha256") != canonical_json_sha256(atom.to_dict()):
            raise ValueError("V4.1 canonical-atom artifact hash mismatch")
        result.append(atom)
        seen.add(experience_id)
    expected = {item.experience_id for item in signatures if item.applicable}
    if seen != expected:
        raise ValueError("V4.1 canonical-atom artifact coverage mismatch")
    return tuple(sorted(result, key=lambda item: item.experience_id))


def _cluster_from_dict(value: Mapping[str, Any]) -> V41RepairCluster:
    distribution = value.get("source_experience_type_distribution")
    if not isinstance(distribution, Mapping):
        raise ValueError("V4.1 cluster is missing type distribution")
    return V41RepairCluster(
        cluster_key=value.get("cluster_key"),
        candidate_id=value.get("candidate_id"),
        title=value.get("title"),
        failure_mechanism=value.get("failure_mechanism"),
        repair_operator=value.get("repair_operator"),
        scope_summary=value.get("scope_summary"),
        member_experience_ids=tuple(value.get("member_experience_ids", ())),
        representative_experience_ids=tuple(
            value.get("representative_experience_ids", ())
        ),
        source_experience_type_distribution=tuple(
            sorted((str(key), int(count)) for key, count in distribution.items())
        ),
        canonical_seed_ids=tuple(value.get("canonical_seed_ids", ())),
        audit_sha256=value.get("audit_sha256"),
    )


def run_clustering(
    *,
    signatures: Sequence[V4RepairSignature],
    profile: V41ConstructionProfile,
    output_dir: Path,
    api_key: str,
    args: argparse.Namespace,
) -> tuple[tuple[V41CanonicalRepairAtom, ...], tuple[V41RepairCluster, ...]]:
    canonical_checkpoint = output_dir / "canonicalization_units.jsonl"
    pair_checkpoint = output_dir / "pair_judgment_units.jsonl"
    audit_checkpoint = output_dir / "cluster_audit_units.jsonl"
    atoms_path = output_dir / "canonical_atoms.jsonl"
    seeds_path = output_dir / "exact_seeds.json"
    pairs_path = output_dir / "candidate_pairs.json"
    judgments_path = output_dir / "pair_judgments.jsonl"
    candidates_path = output_dir / "clique_candidates.json"
    plan_path = output_dir / "cluster_plan.json"

    with ExitStack() as stack:
        canonical_client = stack.enter_context(
            _client(args, api_key=api_key, max_tokens=args.canonical_max_tokens)
        )
        atoms = canonicalize_signatures(
            signatures,
            checkpoint_path=canonical_checkpoint,
            client=canonical_client,
            args=args,
        )
    _write_jsonl(atoms_path, (_atom_record(item) for item in atoms))
    reasoning_atoms = tuple(item for item in atoms if item.memory_role == "reasoning_process")
    if not reasoning_atoms:
        raise RuntimeError("V4.1 canonicalization produced no reasoning-process atoms")
    embeddings = load_or_build_embeddings(reasoning_atoms, output_dir=output_dir, args=args)
    seeds = build_exact_seeds(reasoning_atoms, embeddings)
    _write_json(
        seeds_path,
        {
            "schema_version": "memgen-v4.1-exact-seeds-v1",
            "atom_order_sha256": canonical_json_sha256(
                [item.atom_id for item in reasoning_atoms]
            ),
            "seeds": [_serialize_seed(item) for item in seeds],
        },
    )
    pairs = build_candidate_pairs(seeds, neighbor_count=profile.neighbor_count)
    estimated_pair_requests = (
        len(pairs) + profile.pair_batch_size - 1
    ) // profile.pair_batch_size
    print(
        f"[v4.1-bank] graph reasoning_atoms={len(reasoning_atoms)} "
        f"categorical_seeds={len(seeds)} candidate_pairs={len(pairs)} "
        f"pair_requests={estimated_pair_requests}",
        flush=True,
    )
    pair_artifact = {
        "schema_version": CANDIDATE_PAIR_SCHEMA,
        "neighbor_count": profile.neighbor_count,
        "seed_order_sha256": canonical_json_sha256(
            [str(item["seed_id"]) for item in seeds]
        ),
        "pairs": list(pairs),
    }
    pair_artifact["artifact_sha256"] = canonical_json_sha256(pair_artifact)
    _write_json(pairs_path, pair_artifact)
    with _client(args, api_key=api_key, max_tokens=args.pair_max_tokens) as pair_client:
        judgments = judge_candidate_pairs(
            pairs,
            seeds=seeds,
            atoms=reasoning_atoms,
            embeddings=embeddings,
            checkpoint_path=pair_checkpoint,
            client=pair_client,
            args=args,
        )
    if not pair_checkpoint.exists():
        _write_jsonl(pair_checkpoint, ())
    _write_jsonl(judgments_path, (item.to_dict() for item in judgments))
    candidates = form_clique_candidates(seeds, judgments)
    _write_json(
        candidates_path,
        {
            "schema_version": "memgen-v4.1-clique-candidates-v1",
            "candidates": list(candidates),
        },
    )
    with _client(args, api_key=api_key, max_tokens=args.audit_max_tokens) as audit_client:
        clusters, audit_diagnostics = audit_candidates(
            candidates,
            seeds=seeds,
            atoms=reasoning_atoms,
            signatures=signatures,
            embeddings=embeddings,
            checkpoint_path=audit_checkpoint,
            client=audit_client,
            args=args,
        )
    if not audit_checkpoint.exists():
        _write_jsonl(audit_checkpoint, ())
    plan = build_cluster_plan(
        signatures=signatures,
        atoms=atoms,
        seeds=seeds,
        pairs=pairs,
        judgments=judgments,
        clusters=clusters,
        audit_diagnostics=audit_diagnostics,
    )
    record = {
        "schema_version": V4_1_CLUSTER_PLAN_SCHEMA,
        "prompt_versions": {
            "canonical": V4_1_CANONICAL_PROMPT_VERSION,
            "pair": V4_1_PAIR_PROMPT_VERSION,
            "audit": V4_1_AUDIT_PROMPT_VERSION,
        },
        "created_at": utc_now(),
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
        },
        "profile_sha256": profile.profile_sha256,
        "source_signature_sha256": canonical_json_sha256(
            [item.signature_sha256 for item in signatures]
        ),
        "payload": plan,
        "artifacts": {
            path.name: file_sha256(path)
            for path in (
                canonical_checkpoint,
                atoms_path,
                output_dir / "canonical_embeddings.npy",
                output_dir / "canonical_embeddings_manifest.json",
                seeds_path,
                pairs_path,
                pair_checkpoint,
                judgments_path,
                candidates_path,
                audit_checkpoint,
            )
        },
    }
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(plan_path, record)
    print(
        f"[v4.1-bank] cluster complete atoms={len(atoms)} seeds={len(seeds)} "
        f"pairs={len(pairs)} approved={len(clusters)}",
        flush=True,
    )
    return atoms, clusters


def load_cluster_artifacts(
    *,
    signatures: Sequence[V4RepairSignature],
    profile: V41ConstructionProfile,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[tuple[V41CanonicalRepairAtom, ...], tuple[V41RepairCluster, ...]]:
    atoms_path = output_dir / "canonical_atoms.jsonl"
    plan_path = output_dir / "cluster_plan.json"
    if not atoms_path.is_file() or not plan_path.is_file():
        raise ValueError("V4.1 card stage requires completed cluster artifacts")
    atoms = _load_atoms(atoms_path, signatures=signatures)
    record = json.loads(plan_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != V4_1_CLUSTER_PLAN_SCHEMA or not _record_hash_valid(record):
        raise ValueError("V4.1 cluster-plan record is invalid")
    if record.get("profile_sha256") != profile.profile_sha256:
        raise ValueError("V4.1 cluster-plan profile drifted")
    if record.get("source_signature_sha256") != canonical_json_sha256(
        [item.signature_sha256 for item in signatures]
    ):
        raise ValueError("V4.1 cluster-plan source signatures drifted")
    if not _record_teacher_matches(record, args):
        raise ValueError("V4.1 cluster-plan teacher binding drifted")
    for name, expected_sha256 in record.get("artifacts", {}).items():
        path = output_dir / name
        if not path.is_file() or file_sha256(path) != expected_sha256:
            raise ValueError(f"V4.1 cluster artifact drifted: {name}")
    payload = record.get("payload", {})
    if payload.get("schema_version") != V4_1_CLUSTER_PLAN_SCHEMA:
        raise ValueError("Unexpected V4.1 cluster-plan payload schema")
    clusters = tuple(_cluster_from_dict(item) for item in payload.get("clusters", ()))
    return atoms, tuple(sorted(clusters, key=lambda item: item.cluster_key))


def main() -> None:
    args = parse_args()
    profile = _validate_cli(args)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before V4.1 bank construction")
    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(args.experiences, split_manifest=split_manifest)
    signatures, source_signature_info = load_authenticated_signatures(
        args.source_signatures,
        source_profile_path=args.source_construction_profile,
        experiences=experiences,
    )
    if sum(item.applicable for item in signatures) < 5:
        raise RuntimeError("Source V4 checkpoint has fewer than five applicable signatures")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_profile(
        output_dir / "construction_profile.json",
        expected=_profile_record(
            profile, source_signature_info=source_signature_info, args=args
        ),
        resume=args.resume,
    )

    if args.stage in {"cluster", "all"}:
        atoms, clusters = run_clustering(
            signatures=signatures,
            profile=profile,
            output_dir=output_dir,
            api_key=api_key,
            args=args,
        )
    else:
        atoms, clusters = load_cluster_artifacts(
            signatures=signatures,
            profile=profile,
            output_dir=output_dir,
            args=args,
        )
    if args.stage == "cluster":
        return
    if not clusters:
        raise RuntimeError("No V4.1 cluster passed coherence audit")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError("datasets is required for V4.1 card construction") from exc
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=args.dataset_revision,
    )
    examples = attach_official_solutions(
        experiences,
        split_manifest=split_manifest,
        dataset_revision=args.dataset_revision,
        dataset=dataset,
    )
    bank_records, rejected_cards = synthesize_cards(
        clusters,
        signatures=signatures,
        atoms=atoms,
        examples=examples,
        profile=profile,
        output_dir=output_dir,
        api_key=api_key,
        args=args,
    )
    _write_jsonl(output_dir / "rejected_clusters.jsonl", rejected_cards)
    _write_jsonl(output_dir / "bank_records.jsonl", bank_records)
    if not bank_records:
        raise RuntimeError("No V4.1 process card passed independent review")
    plan_path = output_dir / "cluster_plan.json"
    manifest = build_v4_1_bank_manifest(
        records=bank_records,
        profile=profile,
        inputs={
            "experiences_path": str(args.experiences.resolve()),
            "experiences_sha256": file_sha256(args.experiences),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
            "dataset_revision": args.dataset_revision,
            "construction_example_count": len(experiences),
            "source_signatures": source_signature_info,
            "cluster_plan_path": str(plan_path.resolve()),
            "cluster_plan_sha256": file_sha256(plan_path),
            "canonical_atoms_sha256": file_sha256(output_dir / "canonical_atoms.jsonl"),
            "candidate_pairs_sha256": file_sha256(output_dir / "candidate_pairs.json"),
            "repository": _repository_state(),
        },
        teacher={
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "prompt_versions": {
                "source_signature": V4_SIGNATURE_PROMPT_VERSION,
                "canonical": V4_1_CANONICAL_PROMPT_VERSION,
                "pair": V4_1_PAIR_PROMPT_VERSION,
                "audit": V4_1_AUDIT_PROMPT_VERSION,
                "card": V4_1_CARD_PROMPT_VERSION,
                "review": V4_1_REVIEW_PROMPT_VERSION,
            },
            "embedding": {
                "model": V4_1_EMBEDDING_MODEL,
                "revision": V4_1_EMBEDDING_REVISION,
                "role": "candidate_recall_only",
            },
        },
    )
    manifest["created_at"] = utc_now()
    _write_json(output_dir / "bank_manifest.json", manifest)
    print(
        f"[v4.1-bank] complete records={len(bank_records)} "
        f"manifest={output_dir / 'bank_manifest.json'} "
        f"sha256={manifest['manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.1-bank] error: {exc}", file=sys.stderr)
        raise
