#!/usr/bin/env python3
"""Audit V3.5 dynamic-key retrieval on authenticated source trajectories.

This diagnostic never runs generation or side-KV treatment.  It teacher-forces
the verified success and failure trajectories that produced each memory,
reconstructs the frozen V3.4 gate timeline, and ranks the memory's own V3.5
dynamic key from runtime-identical full-prefix layer-24 queries.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import ApprovedMemorySourceSelector, MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.risk import (
    TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
    deterministic_train_partition,
)
from memgen.experience.v3_5_selector import deterministic_source_pair_partition
from memgen.experience.v3_5_source_alignment import (
    CounterfactualGateObservation,
    V35_SOURCE_ALIGNMENT_EVIDENCE_SCHEMA,
    V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT,
    V35_SOURCE_ALIGNMENT_PERMUTATION_SEED,
    V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR,
    V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA,
    V35_SOURCE_ALIGNMENT_RECALL_KS,
    V35_SOURCE_ALIGNMENT_REPORT_SCHEMA,
    counterfactual_attempts,
    percentile_linear,
    permutation_null,
    rank_metrics,
    score_query,
)


V35_OFFLINE_REPORT_SCHEMA = "experience-memory-v3.5-offline-report-v1"
EVIDENCE_FILE = "source_state_evidence.jsonl"
QUERY_FILE = "first_gate_query_embeddings.safetensors"
REPORT_FILE = "alignment_report.json"
MARKDOWN_FILE = "alignment_report.md"


@dataclass(frozen=True)
class TrajectoryTokens:
    ids: tuple[int, ...]
    reasoning_indices: tuple[int, ...]
    completion_token_count: int
    pre_answer_token_count: int


@dataclass(frozen=True)
class SourcePair:
    memory_record: MemoryRecord
    experience: Mapping[str, Any]
    target: TrajectoryTokens
    reference: TrajectoryTokens
    selector_partition: str
    risk_partition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--dual-key-manifest", type=Path, required=True)
    parser.add_argument("--v35-offline-report", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16",),
        default="bfloat16",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--permutation-count",
        type=int,
        default=V35_SOURCE_ALIGNMENT_PERMUTATION_COUNT,
    )
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument(
        "--skip-exact-anchor-reencode",
        action="store_true",
        help=(
            "Debug only: omit runtime-identical independent prefix re-encoding. "
            "Reports produced with this flag are not formal audit artifacts."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("source alignment audit requires a git revision") from exc
    if not value:
        raise RuntimeError("source alignment audit resolved an empty git revision")
    return value


def model_context_limit(model: Any) -> int | None:
    values = [
        int(value)
        for value in (
            getattr(model.config, "max_position_embeddings", None),
            getattr(model.config, "n_positions", None),
            getattr(model.config, "max_sequence_length", None),
        )
        if isinstance(value, int) and value > 0
    ]
    return min(values) if values else None


def tokenize_trajectory(
    tokenizer: Any, prompt_ids: Sequence[int], completion: str
) -> TrajectoryTokens:
    """Reuse the V3.4 pre-answer source-trajectory token contract."""

    completion_ids = tuple(
        int(value)
        for value in tokenizer.encode(completion, add_special_tokens=False)
    )
    ids = tuple(int(value) for value in prompt_ids) + completion_ids
    marker_offsets = [
        offset
        for marker in ("\\boxed", "\\fbox", "final answer", "answer is")
        if (offset := completion.lower().find(marker.lower())) >= 0
    ]
    upper = len(ids)
    if marker_offsets:
        prefix = completion[: min(marker_offsets)]
        upper = len(prompt_ids) + len(
            tokenizer.encode(prefix, add_special_tokens=False)
        )
    reasoning_indices = tuple(range(len(prompt_ids), upper))
    return TrajectoryTokens(
        ids=ids,
        reasoning_indices=reasoning_indices,
        completion_token_count=len(completion_ids),
        pre_answer_token_count=len(reasoning_indices),
    )


def pad_pair(
    *, tokenizer: Any, target: TrajectoryTokens, reference: TrajectoryTokens,
    device: str,
) -> tuple[Any, Any, tuple[int, int]]:
    """Right-pad two causal trajectories without shifting native positions.

    The production runtime encodes an unpadded prefix whose first prompt token
    is at position zero.  Explicit right padding preserves those positions for
    both source trajectories; masked future pads cannot affect any valid causal
    token.  The returned offsets are therefore always zero.
    """

    import torch

    rows = (target.ids, reference.ids)
    length = max(len(row) for row in rows)
    if tokenizer.pad_token_id is None:
        raise ValueError("source alignment tokenizer has no pad token")
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    offsets: list[int] = []
    for row in rows:
        pad = length - len(row)
        input_ids.append(list(row) + [int(tokenizer.pad_token_id)] * pad)
        attention_masks.append([1] * len(row) + [0] * pad)
        offsets.append(0)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.long, device=device),
        (offsets[0], offsets[1]),
    )


def _resolved_revision(value: Any, fallback: str) -> str:
    result = str(value or fallback)
    if not result:
        raise ValueError("source alignment reasoner revision is empty")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_n < 2:
        raise ValueError("--top-n must be at least two")
    if args.permutation_count <= 0:
        raise ValueError("--permutation-count must be positive")
    if args.max_sequence_length < 0:
        raise ValueError("--max-sequence-length must be non-negative")
    for path in (
        args.approved_bank,
        args.verified_experiences,
        args.split_manifest,
        args.memory_records,
        args.dual_key_manifest,
        args.v35_offline_report,
        args.token_risk_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _validate_inputs(
    *,
    args: argparse.Namespace,
    records: Sequence[MemoryRecord],
    approved_records: Sequence[Mapping[str, Any]],
    verified_experiences: Sequence[Mapping[str, Any]],
    key_bank: Any,
    risk_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v3_5_retrieval import validate_v35_split_manifest

    split_manifest = validate_v35_split_manifest(
        json.loads(args.split_manifest.read_text(encoding="utf-8"))
    )
    expected_hashes = {
        "memory_records_sha256": file_sha256(args.memory_records),
        "phase1_approved_bank_sha256": file_sha256(args.approved_bank),
        "verified_experiences_sha256": file_sha256(args.verified_experiences),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
        "dataset_revision": split_manifest["dataset"]["revision"],
    }
    for field, expected in expected_hashes.items():
        if key_bank.input_artifacts.get(field) != expected:
            raise ValueError(f"source alignment input differs from dual bank: {field}")

    offline = json.loads(args.v35_offline_report.read_text(encoding="utf-8"))
    offline_logical = dict(offline)
    offline_stored_hash = offline_logical.pop("report_sha256", None)
    if (
        offline.get("schema_version") != V35_OFFLINE_REPORT_SCHEMA
        or offline_stored_hash != canonical_json_sha256(offline_logical)
        or int(offline.get("record_count", -1)) != len(records)
        or offline.get("artifacts", {}).get("dual_key_manifest", {}).get("sha256")
        != file_sha256(args.dual_key_manifest)
        or offline.get("task_accuracy_used") is not False
        or offline.get("answer_or_reward_used") is not False
    ):
        raise ValueError("source alignment cannot authenticate V3.5 offline report")

    if (
        risk_artifact.get("schema_version") != TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA
        or risk_artifact.get("status") != "passed"
        or risk_artifact.get("qualification", {}).get("passed") is not True
        or risk_artifact.get("inputs", {}).get("approved_bank_sha256")
        != file_sha256(args.approved_bank)
        or risk_artifact.get("inputs", {}).get("verified_experiences_sha256")
        != file_sha256(args.verified_experiences)
        or risk_artifact.get("prompt_contract")
        != GSM8K_PROMPT_CONTRACT.metadata(chat_template=CONVERSATION_TEMPLATE)
    ):
        raise ValueError("source alignment requires the authenticated V3.4 risk artifact")
    # Reuse the production loader so the token-level observation scope,
    # stable-low label policy, layer, thresholds, and two-token re-arm contract
    # cannot silently drift in this diagnostic.
    EntropyHysteresisGate.from_token_artifact(risk_artifact)

    reasoner = key_bank.manifest.get("reasoner", {})
    risk_reasoner = risk_artifact.get("reasoner", {})
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_reasoner.get(field) != reasoner.get(field):
            raise ValueError("source alignment reasoner provenance differs")

    memory_ids = [record.memory_id for record in records]
    source_ids = [record.source_experience_id for record in records]
    if (
        memory_ids != [str(entry["memory_id"]) for entry in key_bank.entries]
        or source_ids
        != [str(entry["source_experience_id"]) for entry in key_bank.entries]
    ):
        raise ValueError("source alignment record/key order drifted")

    selector = ApprovedMemorySourceSelector()
    sources, rejected = selector.join(approved_records, verified_experiences)
    source_by_id = {source.experience_id: source for source in sources}
    if any(source_id not in source_by_id for source_id in source_ids):
        raise ValueError("source alignment lost an approved verified source")
    for record in records:
        source = source_by_id[record.source_experience_id]
        if (
            record.source_record_sha256
            != canonical_json_sha256(source.approved_record)
            or record.phase1_provenance_sha256
            != str(source.approved_record.get("provenance_sha256", ""))
            or not str(source.verified_experience.get("context", "")).strip()
            or not str(source.verified_experience.get("trajectory", "")).strip()
            or not str(
                source.verified_experience.get("reference_trajectory", "")
            ).strip()
        ):
            raise ValueError(
                f"source alignment provenance drifted for {record.memory_id}"
            )
    return {
        "split_manifest": split_manifest,
        "source_by_id": source_by_id,
        "selector_rejected_count": len(rejected),
        "offline_status": str(offline.get("status")),
        "offline_formal_passed": bool(offline.get("formal_v3_5_offline_passed")),
    }


def _build_pairs(
    *,
    records: Sequence[MemoryRecord],
    source_by_id: Mapping[str, Any],
    tokenizer: Any,
    context_limit: int | None,
    risk_split_seed: int,
    risk_train_fraction: float,
) -> tuple[list[SourcePair], list[dict[str, Any]]]:
    pairs: list[SourcePair] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        source = source_by_id[record.source_experience_id]
        experience = source.verified_experience
        question = str(experience["context"]).strip()
        prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
        target = tokenize_trajectory(
            tokenizer, prompt_ids, str(experience["trajectory"])
        )
        reference = tokenize_trajectory(
            tokenizer, prompt_ids, str(experience["reference_trajectory"])
        )
        reason = None
        if not target.reasoning_indices or not reference.reasoning_indices:
            reason = "no_pre_answer_tokens"
        elif context_limit and max(len(target.ids), len(reference.ids)) > context_limit:
            reason = "context_limit"
        if reason is not None:
            skipped.append({
                "memory_id": record.memory_id,
                "source_experience_id": record.source_experience_id,
                "status": f"skipped_{reason}",
                "target_token_count": len(target.ids),
                "reference_token_count": len(reference.ids),
            })
            continue
        pairs.append(SourcePair(
            memory_record=record,
            experience=experience,
            target=target,
            reference=reference,
            selector_partition=deterministic_source_pair_partition(
                record.memory_id, record.source_experience_id
            ),
            risk_partition=(
                "train"
                if deterministic_train_partition(
                    record.source_experience_id,
                    seed=risk_split_seed,
                    train_fraction=risk_train_fraction,
                )
                else "holdout"
            ),
        ))
    return pairs, skipped


def _risk_scores(states: Any, *, recovery: Any, persistence: Any) -> Any:
    import torch.nn.functional as F

    normalized = F.normalize(states.detach().float(), dim=-1)
    recovery = F.normalize(recovery.detach().float().to(states.device), dim=0)
    persistence = F.normalize(
        persistence.detach().float().to(states.device), dim=0
    )
    return normalized @ persistence - normalized @ recovery


def _selection_gini(counts: Sequence[int]) -> float:
    if not counts or sum(counts) == 0:
        return 0.0
    ordered = sorted(int(value) for value in counts)
    total = sum(ordered)
    count = len(ordered)
    return (
        sum((2 * index - count - 1) * value for index, value in enumerate(ordered, 1))
        / (count * total)
    )


def _hubness(rows: Sequence[Mapping[str, Any]], memory_ids: Sequence[str]) -> dict[str, Any]:
    counts = Counter(str(row["top1_memory_id"]) for row in rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(rows)
    return {
        "query_count": total,
        "selected_memory_count": len(counts),
        "top1_share": ordered[0][1] / total if total else 0.0,
        "selection_gini_over_full_bank": _selection_gini(
            [counts.get(memory_id, 0) for memory_id in memory_ids]
        ),
        "top_memories": [
            {"memory_id": memory_id, "top1_count": count, "top1_share": count / total}
            for memory_id, count in ordered[:10]
        ] if total else [],
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any] | None:
    normalized = [float(value) for value in values]
    if not normalized:
        return None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("source alignment summary values must be finite")
    return {
        "count": len(normalized),
        "minimum": min(normalized),
        "p05": percentile_linear(normalized, 0.05),
        "median": percentile_linear(normalized, 0.50),
        "mean": sum(normalized) / len(normalized),
        "p95": percentile_linear(normalized, 0.95),
        "maximum": max(normalized),
    }


def _anchor_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    memory_ids: Sequence[str],
    expected_pair_count: int,
    permutation_count: int,
) -> dict[str, Any]:
    by_partition = {
        partition: [
            int(row["own_memory_rank"])
            for row in rows
            if row["selector_partition"] == partition
        ]
        for partition in ("train", "holdout")
    }
    by_risk_partition = {
        partition: [
            int(row["own_memory_rank"])
            for row in rows
            if row["risk_partition"] == partition
        ]
        for partition in ("train", "holdout")
    }
    result = {
        "eligible_count": len(rows),
        "no_event_count": expected_pair_count - len(rows),
        "eligible_fraction": len(rows) / expected_pair_count if expected_pair_count else 0.0,
        "all": rank_metrics(
            [int(row["own_memory_rank"]) for row in rows],
            memory_count=len(memory_ids),
        ),
        "train": rank_metrics(by_partition["train"], memory_count=len(memory_ids)),
        "holdout": rank_metrics(by_partition["holdout"], memory_count=len(memory_ids)),
        "risk_fit_train": rank_metrics(
            by_risk_partition["train"], memory_count=len(memory_ids)
        ),
        "risk_fit_holdout": rank_metrics(
            by_risk_partition["holdout"], memory_count=len(memory_ids)
        ),
        "score_geometry": {
            field: _numeric_summary([float(row[field]) for row in rows])
            for field in (
                "own_memory_score",
                "own_minus_best_other_score",
                "top1_top2_margin",
            )
        },
        "hubness": _hubness(rows, memory_ids),
        "permutation_null": permutation_null(
            rows,
            memory_count=len(memory_ids),
            iterations=permutation_count,
            seed=V35_SOURCE_ALIGNMENT_PERMUTATION_SEED,
        ),
    }
    return result


def _paired_anchor_comparison(
    *,
    target_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_by_id = {str(row["memory_id"]): row for row in target_rows}
    reference_by_id = {str(row["memory_id"]): row for row in reference_rows}
    common = sorted(set(target_by_id) & set(reference_by_id))
    deltas = [
        int(reference_by_id[memory_id]["own_memory_rank"])
        - int(target_by_id[memory_id]["own_memory_rank"])
        for memory_id in common
    ]
    return {
        "paired_count": len(common),
        "target_only_count": len(set(target_by_id) - set(reference_by_id)),
        "reference_only_count": len(set(reference_by_id) - set(target_by_id)),
        "target_better_count": sum(delta > 0 for delta in deltas),
        "reference_better_count": sum(delta < 0 for delta in deltas),
        "rank_tie_count": sum(delta == 0 for delta in deltas),
        "reference_minus_target_rank": _numeric_summary(deltas),
    }


def _all_token_summary(
    rows: Sequence[Mapping[str, Any]], *, memory_count: int
) -> dict[str, Any]:
    by_memory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_memory[str(row["memory_id"])].append(row)
    macro: dict[str, list[float]] = {
        f"top_{k}_token_fraction": [] for k in V35_SOURCE_ALIGNMENT_RECALL_KS
    }
    earliest: dict[str, list[float]] = {
        f"earliest_top_{k}_normalized_position": []
        for k in V35_SOURCE_ALIGNMENT_RECALL_KS
    }
    for memory_rows in by_memory.values():
        ordered = sorted(memory_rows, key=lambda row: int(row["reasoning_rank"]))
        denominator = len(ordered)
        for k in V35_SOURCE_ALIGNMENT_RECALL_KS:
            qualifying = [
                row for row in ordered if int(row["own_memory_rank"]) <= k
            ]
            macro[f"top_{k}_token_fraction"].append(
                len(qualifying) / denominator
            )
            if qualifying:
                earliest[f"earliest_top_{k}_normalized_position"].append(
                    float(qualifying[0]["normalized_trajectory_position"])
                )
    result: dict[str, Any] = {
        "token_count": len(rows),
        "memory_count": len(by_memory),
        "token_weighted_rank_metrics": rank_metrics(
            [int(row["own_memory_rank"]) for row in rows],
            memory_count=memory_count,
        ),
    }
    for key, values in macro.items():
        result[f"macro_mean_{key}"] = sum(values) / len(values) if values else None
    for key, values in earliest.items():
        result[key] = {
            "memory_count_ever_reached": len(values),
            "median": percentile_linear(values, 0.5) if values else None,
            "p95": percentile_linear(values, 0.95) if values else None,
        }
    return result


def _clean_row_for_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"rank_by_memory_id", "query_embedding"}
    }


def _markdown(report: Mapping[str, Any]) -> str:
    primary = report.get("primary", {})
    reference = report.get("secondary", {}).get("reference_all_tokens", {})
    target = report.get("secondary", {}).get("target_first_gate", {})
    paired = report.get("secondary", {}).get(
        "paired_target_vs_reference_first_gate", {}
    )
    lines = [
        "# MemGen V3.5 Dynamic Source-State Alignment Audit",
        "",
        f"- Status: `{report.get('status')}`",
        "- Formal V3.5 qualification changed: `false`",
        "- Task accuracy used: `false`",
        "- Answer or reward used: `false`",
        f"- Memory count: `{report.get('memory_count')}`",
        f"- Primary anchor: `{V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR}`",
        f"- Reference first-gate eligible: `{primary.get('eligible_count')}`",
        f"- Reference first-gate no event: `{primary.get('no_event_count')}`",
        f"- Reference first-gate metrics: `{json.dumps(primary.get('all'), sort_keys=True)}`",
        "- Reference score geometry: "
        f"`{json.dumps(primary.get('score_geometry'), sort_keys=True)}`",
        f"- Target first-gate metrics: `{json.dumps(target.get('all'), sort_keys=True)}`",
        f"- Paired target/reference first-gate ranks: `{json.dumps(paired, sort_keys=True)}`",
        f"- Reference all-token summary: `{json.dumps(reference, sort_keys=True)}`",
        "",
        "This is an in-source, answer-blind alignment sanity check. A strong result is",
        "necessary but not sufficient for cross-problem usefulness; a weak result rejects",
        "the current abstract dynamic-key/runtime-prefix alignment before online treatment.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import encode_layer_token, tensor_sha256
    from memgen.model.side_kv import SDPAAttentionEntropyObserver
    from memgen.model.v3_5_retrieval import DualRetrievalKeyBankLoader

    approved_records = list(iter_jsonl(args.approved_bank))
    verified_experiences = list(iter_jsonl(args.verified_experiences))
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    key_bank = DualRetrievalKeyBankLoader(
        manifest_path=args.dual_key_manifest,
    )
    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    authenticated = _validate_inputs(
        args=args,
        records=records,
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        key_bank=key_bank,
        risk_artifact=risk_artifact,
    )

    reasoner = key_bank.manifest["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    if (
        _resolved_revision(
            getattr(model.config, "_commit_hash", None),
            str(reasoner["model_revision"]),
        )
        != reasoner["model_revision"]
        or _resolved_revision(
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            str(reasoner["tokenizer_revision"]),
        )
        != reasoner["tokenizer_revision"]
    ):
        raise ValueError("source alignment resolved reasoner/tokenizer drifted")

    context_limit = args.max_sequence_length or model_context_limit(model)
    risk_split_seed = int(risk_artifact["inputs"]["risk_split_seed"])
    risk_train_fraction = float(risk_artifact["inputs"]["risk_train_fraction"])
    pairs, skipped = _build_pairs(
        records=records,
        source_by_id=authenticated["source_by_id"],
        tokenizer=tokenizer,
        context_limit=context_limit,
        risk_split_seed=risk_split_seed,
        risk_train_fraction=risk_train_fraction,
    )
    if not pairs:
        raise RuntimeError("source alignment has no context-eligible pairs")

    memory_ids = tuple(str(entry["memory_id"]) for entry in key_bank.entries)
    dynamic_embeddings = key_bank.dynamic_embeddings.to(args.device)
    gate_config = risk_artifact["construction"]
    risk_config = risk_artifact["risk_gate"]
    high_threshold = float(gate_config["high_entropy_threshold"])
    low_threshold = float(gate_config["low_entropy_threshold"])
    risk_threshold = float(risk_config["threshold"])
    recovery_center = risk_config["recovery_center"]
    persistence_center = risk_config["persistence_center"]

    evidence_rows: list[dict[str, Any]] = []
    anchor_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_token_rows: dict[str, list[dict[str, Any]]] = {
        "target": [],
        "reference": [],
    }
    sidecar_tensors: dict[str, Any] = {}
    sidecar_order: list[dict[str, Any]] = []

    observer = SDPAAttentionEntropyObserver(
        model=model,
        sink_token_count=int(gate_config["sink_token_count"]),
    )
    try:
        for pair_index, pair in enumerate(pairs):
            input_ids, attention_mask, offsets = pad_pair(
                tokenizer=tokenizer,
                target=pair.target,
                reference=pair.reference,
                device=args.device,
            )
            with torch.inference_mode(), observer.capture():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            entropy_by_query = observer.observation.entropy_by_query
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) <= 24:
                raise RuntimeError("source alignment model has no layer-24 states")
            states = hidden_states[24]

            for row_index, (side, trajectory) in enumerate((
                ("target", pair.target),
                ("reference", pair.reference),
            )):
                positions = [
                    offsets[row_index] + index
                    for index in trajectory.reasoning_indices
                ]
                side_states = states[row_index, positions, :].detach().float()
                query_embeddings = F.normalize(side_states, dim=-1)
                score_matrix = query_embeddings @ dynamic_embeddings.T
                query_embeddings_cpu = query_embeddings.detach().cpu()
                score_matrix_values = score_matrix.detach().cpu().tolist()
                entropies = [
                    float(value)
                    for value in entropy_by_query[
                        row_index, positions
                    ].detach().cpu().tolist()
                ]
                risks = [
                    float(value)
                    for value in _risk_scores(
                        side_states,
                        recovery=recovery_center,
                        persistence=persistence_center,
                    ).detach().cpu().tolist()
                ]
                if not all(math.isfinite(value) for value in entropies + risks):
                    raise RuntimeError("source alignment gate diagnostics are non-finite")
                attempts = counterfactual_attempts(
                    [
                        CounterfactualGateObservation(
                            reasoning_rank=rank,
                            attention_entropy=entropy,
                            risk_score=risk,
                        )
                        for rank, (entropy, risk) in enumerate(zip(entropies, risks))
                    ],
                    high_entropy_threshold=high_threshold,
                    low_entropy_threshold=low_threshold,
                    risk_threshold=risk_threshold,
                    rearm_low_token_count=2,
                    maximum_attempts=3,
                )
                attempt_by_rank = {
                    int(item["reasoning_rank"]): int(item["attempt_number"])
                    for item in attempts
                }
                question_hash = text_sha256(
                    str(pair.experience["context"]).strip()
                )
                trajectory_hash = text_sha256(
                    str(
                        pair.experience[
                            "trajectory" if side == "target" else "reference_trajectory"
                        ]
                    )
                )
                side_rows: list[dict[str, Any]] = []
                for reasoning_rank, token_index in enumerate(
                    trajectory.reasoning_indices
                ):
                    attempt_number = attempt_by_rank.get(reasoning_rank)
                    score_values = [
                        float(value)
                        for value in score_matrix_values[reasoning_rank]
                    ]
                    scored = score_query(
                        memory_ids=memory_ids,
                        scores=score_values,
                        own_memory_id=pair.memory_record.memory_id,
                        top_n=args.top_n,
                        include_rank_lookup=attempt_number is not None,
                    )
                    embedding = query_embeddings_cpu[reasoning_rank]
                    partial_count = reasoning_rank + 1
                    prefix_ids = trajectory.ids[: token_index + 1]
                    row: dict[str, Any] = {
                        "schema_version": V35_SOURCE_ALIGNMENT_EVIDENCE_SCHEMA,
                        "memory_id": pair.memory_record.memory_id,
                        "source_experience_id": pair.memory_record.source_experience_id,
                        "selector_partition": pair.selector_partition,
                        "risk_partition": pair.risk_partition,
                        "trajectory_side": side,
                        "query_anchor": (
                            f"{side}_counterfactual_attempt_{attempt_number}"
                            if attempt_number is not None
                            else "all_pre_answer_token_curve"
                        ),
                        "counterfactual_attempt_number": attempt_number,
                        "source_question_sha256": question_hash,
                        "source_trajectory_sha256": trajectory_hash,
                        "prompt_token_count": token_index - reasoning_rank,
                        "partial_cot_token_count": partial_count,
                        "encoded_full_prefix_token_count": len(prefix_ids),
                        "full_prefix_token_ids_sha256": canonical_json_sha256(
                            list(prefix_ids)
                        ),
                        "reasoning_rank": reasoning_rank,
                        "normalized_trajectory_position": (
                            reasoning_rank / max(1, trajectory.pre_answer_token_count - 1)
                        ),
                        "query_embedding_token_index": len(prefix_ids) - 1,
                        "query_embedding_token_id": int(trajectory.ids[token_index]),
                        "query_embedding_token_text": tokenizer.decode(
                            [int(trajectory.ids[token_index])],
                            skip_special_tokens=False,
                        ),
                        "query_embedding_sha256": tensor_sha256(embedding),
                        "query_embedding_norm": float(embedding.norm().item()),
                        "query_embedding_source": "batched_causal_full_sequence",
                        "exact_anchor_reencoded": False,
                        "layer_number": 24,
                        "pooling": "current_generated_token",
                        "normalization": "l2",
                        "side_kv_disabled": True,
                        "attention_entropy": entropies[reasoning_rank],
                        "persistence_risk": risks[reasoning_rank],
                        "high_entropy_threshold": high_threshold,
                        "low_entropy_threshold": low_threshold,
                        "risk_threshold": risk_threshold,
                        "joint_signal_qualified": (
                            entropies[reasoning_rank] >= high_threshold
                            and risks[reasoning_rank] > risk_threshold
                        ),
                        **scored,
                    }
                    if attempt_number is not None:
                        row["query_embedding"] = embedding
                    if (
                        args.skip_exact_anchor_reencode
                        and attempt_number == 1
                    ):
                        tensor_name = f"{side}_{pair_index:04d}"
                        sidecar_tensors[tensor_name] = embedding.contiguous()
                        sidecar_order.append({
                            "tensor_name": tensor_name,
                            "memory_id": pair.memory_record.memory_id,
                            "source_experience_id": pair.memory_record.source_experience_id,
                            "trajectory_side": side,
                            "query_embedding_sha256": tensor_sha256(embedding),
                        })
                        row["query_sidecar_tensor_name"] = tensor_name
                    side_rows.append(row)

                if not args.skip_exact_anchor_reencode:
                    for attempt in attempts:
                        reasoning_rank = int(attempt["reasoning_rank"])
                        row = side_rows[reasoning_rank]
                        token_index = trajectory.reasoning_indices[reasoning_rank]
                        prefix_ids = trajectory.ids[: token_index + 1]
                        exact = encode_layer_token(
                            model=model,
                            token_ids=prefix_ids,
                            layer_number=24,
                            token_index=len(prefix_ids) - 1,
                            device=args.device,
                        ).cpu()
                        exact_scores = dynamic_embeddings @ exact.to(args.device)
                        scored = score_query(
                            memory_ids=memory_ids,
                            scores=[
                                float(value)
                                for value in exact_scores.detach().cpu().tolist()
                            ],
                            own_memory_id=pair.memory_record.memory_id,
                            top_n=args.top_n,
                            include_rank_lookup=True,
                        )
                        batched = row["query_embedding"]
                        row.update({
                            **scored,
                            "query_embedding_sha256": tensor_sha256(exact),
                            "query_embedding_norm": float(exact.norm().item()),
                            "query_embedding_source": "independent_exact_full_prefix_reencode",
                            "exact_anchor_reencoded": True,
                            "batched_exact_embedding_cosine": float(
                                F.cosine_similarity(batched, exact, dim=0).item()
                            ),
                            "batched_exact_embedding_max_abs_delta": float(
                                (batched - exact).abs().max().item()
                            ),
                            "query_embedding": exact,
                        })
                        if int(attempt["attempt_number"]) == 1:
                            tensor_name = f"{side}_{pair_index:04d}"
                            sidecar_tensors[tensor_name] = exact.contiguous()
                            sidecar_order.append({
                                "tensor_name": tensor_name,
                                "memory_id": pair.memory_record.memory_id,
                                "source_experience_id": pair.memory_record.source_experience_id,
                                "trajectory_side": side,
                                "query_embedding_sha256": tensor_sha256(exact),
                            })
                            row["query_sidecar_tensor_name"] = tensor_name

                evidence_rows.extend(side_rows)
                all_token_rows[side].extend(side_rows)
                for attempt_number in (1, 2, 3):
                    matching = [
                        row for row in side_rows
                        if row["counterfactual_attempt_number"] == attempt_number
                    ]
                    if len(matching) > 1:
                        raise RuntimeError("source alignment produced duplicate attempts")
                    if matching:
                        anchor_rows[(side, attempt_number)].append(matching[0])

            del output, hidden_states, states
            if args.device.startswith("cuda") and (pair_index + 1) % 16 == 0:
                torch.cuda.empty_cache()
            print(
                f"[v3.5-source-alignment] {pair_index + 1}/{len(pairs)} "
                f"{pair.memory_record.memory_id}",
                flush=True,
            )
    finally:
        observer.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / EVIDENCE_FILE
    evidence_count = write_jsonl(
        evidence_path,
        (_clean_row_for_evidence(row) for row in evidence_rows),
    )
    query_path = args.output_dir / QUERY_FILE
    if sidecar_tensors:
        save_file(
            sidecar_tensors,
            str(query_path),
            metadata={
                "schema_version": V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA,
                "ordered_queries_sha256": canonical_json_sha256(sidecar_order),
            },
        )
    else:
        raise RuntimeError("source alignment produced no first-gate query sidecar")

    primary_rows = anchor_rows[("reference", 1)]
    primary = _anchor_summary(
        primary_rows,
        memory_ids=memory_ids,
        expected_pair_count=len(pairs),
        permutation_count=args.permutation_count,
    )
    secondary: dict[str, Any] = {
        "target_first_gate": _anchor_summary(
            anchor_rows[("target", 1)],
            memory_ids=memory_ids,
            expected_pair_count=len(pairs),
            permutation_count=args.permutation_count,
        ),
        "reference_attempt_2": _anchor_summary(
            anchor_rows[("reference", 2)],
            memory_ids=memory_ids,
            expected_pair_count=len(pairs),
            permutation_count=args.permutation_count,
        ),
        "reference_attempt_3": _anchor_summary(
            anchor_rows[("reference", 3)],
            memory_ids=memory_ids,
            expected_pair_count=len(pairs),
            permutation_count=args.permutation_count,
        ),
        "target_attempt_2": _anchor_summary(
            anchor_rows[("target", 2)],
            memory_ids=memory_ids,
            expected_pair_count=len(pairs),
            permutation_count=args.permutation_count,
        ),
        "target_attempt_3": _anchor_summary(
            anchor_rows[("target", 3)],
            memory_ids=memory_ids,
            expected_pair_count=len(pairs),
            permutation_count=args.permutation_count,
        ),
        "paired_target_vs_reference_first_gate": _paired_anchor_comparison(
            target_rows=anchor_rows[("target", 1)],
            reference_rows=anchor_rows[("reference", 1)],
        ),
        "reference_all_tokens": _all_token_summary(
            all_token_rows["reference"], memory_count=len(memory_ids)
        ),
        "target_all_tokens": _all_token_summary(
            all_token_rows["target"], memory_count=len(memory_ids)
        ),
    }

    implementation_paths = (
        "memgen/experience/v3_5_source_alignment.py",
        "scripts/audit_v3_5_dynamic_source_alignment.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    report: dict[str, Any] = {
        "schema_version": V35_SOURCE_ALIGNMENT_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "formal_v3_5_qualification_changed": False,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "answer_or_reward_scope": (
            "not_used_for_query_ranking_threshold_or_result_selection_after_"
            "authenticated_source_bank_selection"
        ),
        "verified_success_failure_roles_reused_from_source_provenance": True,
        "primary_anchor": V35_SOURCE_ALIGNMENT_PRIMARY_ANCHOR,
        "memory_count": len(memory_ids),
        "context_eligible_pair_count": len(pairs),
        "skipped_pair_count": len(skipped),
        "skipped_pairs": skipped,
        "configuration": {
            "trajectory_sides": ["target", "reference"],
            "pre_answer_policy": "before_first_box_fbox_final_answer_or_answer_is_marker",
            "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
                chat_template=CONVERSATION_TEMPLATE
            ),
            "dynamic_query_context": "canonical_prompt_plus_full_partial_cot",
            "dynamic_query_pooling": "current_generated_token",
            "dynamic_query_layer": 24,
            "dynamic_query_normalization": "l2",
            "dynamic_query_encoder_state": "pure_prefix_reencode_side_kv_disabled",
            "batched_gate_padding_policy": (
                "explicit_right_padding_preserves_unpadded_prefix_positions"
            ),
            "retrieval_scope": "all_dynamic_keys_static_selector_bypassed",
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
            "top_n": args.top_n,
            "gate_policy": "frozen_v3.4_counterfactual_native_replay",
            "high_entropy_threshold": high_threshold,
            "low_entropy_threshold": low_threshold,
            "risk_threshold": risk_threshold,
            "risk_split_seed": risk_split_seed,
            "risk_train_fraction": risk_train_fraction,
            "rearm_low_token_count": 2,
            "maximum_counterfactual_attempts": 3,
            "exact_anchor_reencode_required": True,
            "exact_anchor_reencode_performed": not args.skip_exact_anchor_reencode,
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.verified_experiences),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
            "dual_key_manifest_logical_sha256": key_bank.manifest_sha256,
            "v35_offline_report_sha256": file_sha256(args.v35_offline_report),
            "v35_offline_status": authenticated["offline_status"],
            "v35_offline_formal_passed": authenticated["offline_formal_passed"],
            "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
            "git_revision": git_revision(),
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(implementation_hashes),
        },
        "artifacts": {
            "evidence": {
                "path": evidence_path.name,
                "sha256": file_sha256(evidence_path),
                "row_count": evidence_count,
            },
            "first_gate_query_embeddings": {
                "path": query_path.name,
                "sha256": file_sha256(query_path),
                "tensor_count": len(sidecar_tensors),
                "ordered_queries_sha256": canonical_json_sha256(sidecar_order),
                "ordered_queries": sidecar_order,
            },
        },
        "primary": primary,
        "secondary": secondary,
        "requirements": {
            "source_join_authenticated": True,
            "dynamic_bank_authenticated": True,
            "v35_offline_artifact_bound_even_when_not_qualified": True,
            "v34_gate_artifact_authenticated": True,
            "runtime_prompt_contract_reused": True,
            "full_prefix_current_token_query_contract_reused": True,
            "static_selector_bypassed_for_dynamic_isolation": True,
            "side_kv_disabled": True,
            "target_and_reference_trajectories_both_audited": True,
            "reference_first_gate_anchor_pre_registered": True,
            "best_rank_not_used_as_primary": True,
            "permutation_null_preserves_score_geometry": True,
            "exact_anchor_reencoded": not args.skip_exact_anchor_reencode,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used_after_authenticated_source_selection": True,
            "formal_v3_5_qualification_unchanged": True,
        },
        "interpretation_contract": {
            "weak_result": (
                "reject_current_abstract_dynamic_key_runtime_prefix_alignment"
            ),
            "target_strong_reference_weak": (
                "confirmation_retrieval_not_corrective_retrieval"
            ),
            "strong_result": (
                "necessary_source_alignment_only_not_cross_problem_utility"
            ),
            "other_memories_are_not_strict_negatives": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / REPORT_FILE
    write_json(report_path, report)
    write_text(args.output_dir / MARKDOWN_FILE, _markdown(report))
    print(
        "[v3.5-source-alignment] "
        f"status={report['status']} pairs={len(pairs)} "
        f"primary_eligible={primary['eligible_count']} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
