#!/usr/bin/env python3
"""Exhaustive causal sweep of the authenticated bank on V3.7 failures.

The diagnostic reuses authenticated V3.7 candidate treatments, then evaluates
every remaining V3.6 state-key memory on every gate-eligible V3.7 baseline
failure.  It writes a complete query-by-memory utility matrix and attributes
the observed loss to bank/value coverage, candidate retrieval, or top-1
reranking without fitting a selector or changing formal V3.5 qualification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v3_7_cross_problem import (
    V37_CAUSAL_QUERY_SCHEMA,
    V37_CAUSAL_REPORT_SCHEMA,
    V37_CAUSAL_TREATMENT_SCHEMA,
    V37_RETRIEVAL_VARIANTS,
    causal_utility,
    summarize_causal_rows,
)
from memgen.experience.v3_8_full_bank import (
    V38_PROFILE_SCHEMA,
    V38_REPORT_SCHEMA,
    V38_TREATMENT_SCHEMA,
    V38_UTILITY_MATRIX_SCHEMA,
    build_utility_matrix,
    summarize_full_bank_matrix,
)
from scripts.audit_v3_7_cross_problem_causal_applicability import (
    _bank_question_hashes,
    _encode_query_variants,
    _generate_continuous_observation,
    _load_split,
    _load_state_key_bank,
    _load_v36_inputs,
    _processed_solution,
    _retrieval_for_query,
    _score_condition,
    _trace_summary,
)


PROFILE_FILE = "full_bank_profile.json"
TREATMENT_FILE = "full_bank_treatments.jsonl"
MATRIX_FILE = "utility_matrix.json"
REPORT_FILE = "full_bank_report.json"
MARKDOWN_FILE = "full_bank_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--dual-key-manifest", type=Path, required=True)
    parser.add_argument("--source-alignment-evidence", type=Path, required=True)
    parser.add_argument("--v36-report", type=Path, required=True)
    parser.add_argument("--state-key-manifest", type=Path, required=True)
    parser.add_argument("--v37-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GSM8K_PROMPT_CONTRACT.max_new_tokens,
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.is_file() else []


def _logical_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _validate_args(args: argparse.Namespace) -> None:
    required = (
        args.split_manifest,
        args.memory_records,
        args.side_kv_manifest,
        args.e0_final_report,
        args.token_risk_artifact,
        args.dual_key_manifest,
        args.source_alignment_evidence,
        args.v36_report,
        args.state_key_manifest,
        args.v37_dir / "causal_profile.json",
        args.v37_dir / "causal_queries.jsonl",
        args.v37_dir / "causal_treatments.jsonl",
        args.v37_dir / "causal_report.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V3.8 inputs are missing: {missing}")
    if args.diagnosis_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("V3.8 received invalid numeric arguments")
    if args.dtype != "bfloat16":
        raise ValueError("V3.8 is frozen to bfloat16 model compute")
    if args.max_new_tokens != GSM8K_PROMPT_CONTRACT.max_new_tokens:
        raise ValueError("V3.8 requires the canonical GSM8K token budget")


def _authenticate_v37(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    profile_path = args.v37_dir / "causal_profile.json"
    query_path = args.v37_dir / "causal_queries.jsonl"
    treatment_path = args.v37_dir / "causal_treatments.jsonl"
    report_path = args.v37_dir / "causal_report.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    query_rows = load_jsonl(query_path)
    treatment_rows = load_jsonl(treatment_path)
    if (
        profile.get("schema_version")
        != "experience-memory-v3.7-cross-problem-causal-applicability-profile-v1"
        or profile.get("profile_sha256")
        != canonical_json_sha256(
            {
                key: value
                for key, value in profile.items()
                if key not in {"created_at", "profile_sha256"}
            }
        )
    ):
        raise ValueError("V3.8 received an unauthenticated V3.7 profile")
    if (
        report.get("schema_version") != V37_CAUSAL_REPORT_SCHEMA
        or report.get("status") != "completed_diagnostic"
        or report.get("qualified_for_online_use") is not False
        or report.get("same_question_memory_permitted") is not False
        or report.get("cross_problem_enforced") is not True
        or report.get("profile_sha256") != profile.get("profile_sha256")
        or report.get("report_sha256") != _logical_hash(report, "report_sha256")
    ):
        raise ValueError("V3.8 received an unauthenticated V3.7 report")
    requirements = report.get("requirements", {})
    if (
        not isinstance(requirements, Mapping)
        or not requirements
        or not all(requirements.values())
    ):
        raise ValueError("V3.7 requirements are incomplete or failed")
    artifacts = report.get("artifacts", {})
    for name, path in (
        ("profile", profile_path),
        ("queries", query_path),
        ("treatments", treatment_path),
    ):
        if artifacts.get(name, {}).get("sha256") != file_sha256(path):
            raise ValueError(f"V3.7 {name} artifact hash drifted")
    if (
        int(artifacts.get("queries", {}).get("row_count", -1)) != len(query_rows)
        or int(artifacts.get("treatments", {}).get("row_count", -1))
        != len(treatment_rows)
    ):
        raise ValueError("V3.7 artifact row count drifted")
    inputs = report.get("inputs", {})
    expected_inputs = {
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "memory_records_sha256": file_sha256(args.memory_records),
        "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        "e0_final_report_sha256": file_sha256(args.e0_final_report),
        "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
        "source_alignment_evidence_sha256": file_sha256(
            args.source_alignment_evidence
        ),
        "v36_report_sha256": file_sha256(args.v36_report),
        "state_key_manifest_sha256": file_sha256(args.state_key_manifest),
    }
    if any(inputs.get(name) != digest for name, digest in expected_inputs.items()):
        raise ValueError("V3.8 base input identity differs from V3.7")
    implementation = inputs.get("implementation_files_sha256", {})
    if (
        not isinstance(implementation, Mapping)
        or not implementation
        or any(
            Path(path).is_absolute()
            or ".." in Path(path).parts
            or not (PROJECT_ROOT / path).is_file()
            or file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in implementation.items()
        )
        or inputs.get("implementation_set_sha256")
        != canonical_json_sha256(implementation)
    ):
        raise ValueError("V3.7 implementation identity drifted")
    for row in query_rows:
        if (
            row.get("schema_version") != V37_CAUSAL_QUERY_SCHEMA
            or row.get("profile_sha256") != profile["profile_sha256"]
        ):
            raise ValueError("V3.7 query evidence differs from its profile")
    for row in treatment_rows:
        if (
            row.get("schema_version") != V37_CAUSAL_TREATMENT_SCHEMA
            or row.get("profile_sha256") != profile["profile_sha256"]
        ):
            raise ValueError("V3.7 treatment evidence differs from its profile")
    reproduced = summarize_causal_rows(
        query_rows=query_rows,
        treatment_rows=treatment_rows,
        candidate_top_k=int(profile["candidate_top_k"]),
    )
    if reproduced != report.get("summary"):
        raise ValueError("V3.7 saved summary is not reproducible")
    return profile, report, query_rows, treatment_rows


def _convert_v37_treatment(
    *, row: Mapping[str, Any], profile_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": V38_TREATMENT_SCHEMA,
        "profile_sha256": profile_sha256,
        "sample_id": str(row["sample_id"]),
        "question_sha256": str(row["question_sha256"]),
        "memory_id": str(row["memory_id"]),
        "memory_source_question_sha256": str(
            row["memory_source_question_sha256"]
        ),
        "cross_problem": True,
        "same_question": False,
        "rank_by_variant": {
            variant: int(row["rank_by_variant"][variant])
            for variant in V37_RETRIEVAL_VARIANTS
        },
        "score_by_variant": {
            variant: float(row["score_by_variant"][variant])
            for variant in V37_RETRIEVAL_VARIANTS
        },
        "baseline_reward": float(row["baseline_reward"]),
        "treatment_reward": float(row["treatment_reward"]),
        "causal_utility": int(row["causal_utility"]),
        "causal_label": str(row["causal_label"]),
        "prefix_token_ids_sha256": str(row["prefix_token_ids_sha256"]),
        "payload_hash": str(row["payload_hash"]),
        "treatment": row["treatment"],
        "side_kv_trace": row["side_kv_trace"],
        "evidence_origin": "v3.7_authenticated_reuse",
        "v37_treatment_row_sha256": canonical_json_sha256(row),
    }


def _validate_treatment_row(
    *,
    row: Mapping[str, Any],
    profile_sha256: str,
    failure_ids: set[str],
    memory_ids: set[str],
) -> tuple[str, str]:
    pair = (str(row.get("sample_id", "")), str(row.get("memory_id", "")))
    if (
        row.get("schema_version") != V38_TREATMENT_SCHEMA
        or row.get("profile_sha256") != profile_sha256
        or pair[0] not in failure_ids
        or pair[1] not in memory_ids
        or row.get("cross_problem") is not True
        or row.get("same_question") is not False
        or float(row.get("baseline_reward", -1.0)) != 0.0
        or float(row.get("treatment_reward", -1.0)) not in {0.0, 1.0}
        or int(row.get("causal_utility", -1))
        != int(float(row.get("treatment_reward", -1.0)))
    ):
        raise ValueError("V3.8 treatment resume evidence is invalid")
    ranks = row.get("rank_by_variant")
    scores = row.get("score_by_variant")
    if (
        not isinstance(ranks, Mapping)
        or set(ranks) != set(V37_RETRIEVAL_VARIANTS)
        or not isinstance(scores, Mapping)
        or set(scores) != set(V37_RETRIEVAL_VARIANTS)
        or any(
            int(ranks[variant]) < 1
            or int(ranks[variant]) > len(memory_ids)
            or not math.isfinite(float(scores[variant]))
            for variant in V37_RETRIEVAL_VARIANTS
        )
        or row.get("evidence_origin")
        not in {"v3.7_authenticated_reuse", "v3.8_full_bank_generation"}
    ):
        raise ValueError("V3.8 treatment retrieval evidence is invalid")
    return pair


def _progress_report(
    *, profile_sha256: str, expected_pairs: int, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": V38_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "running",
        "profile_sha256": profile_sha256,
        "expected_treatment_pair_count": expected_pairs,
        "completed_treatment_pair_count": len(rows),
        "qualified_for_online_use": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    coverage = summary["causal_coverage"]
    diagnosis_k = int(summary["bottleneck_attribution"]["diagnosis_k"])
    lines = [
        "# MemGen V3.8 Failure-Only Full-Bank Causal Audit",
        "",
        f"- Status: `{report['status']}`",
        "- Diagnostic only: `true`",
        "- Qualified for online use: `false`",
        "- Same-question memory permitted: `false`",
        f"- Gate-eligible baseline failures: `{summary['failure_query_count']}`",
        f"- Authenticated state-key memories: `{summary['authenticated_state_key_memory_count']}`",
        f"- Exhaustive treatment pairs: `{summary['observed_treatment_pair_count']}`",
        f"- Recoverable failures: `{coverage['recoverable_failure_count']}`",
        f"- Recoverable failure fraction: `{coverage['recoverable_failure_fraction']}`",
        f"- Helpful pairs: `{coverage['helpful_pair_count']}`",
        f"- Helpful pair fraction: `{coverage['helpful_pair_fraction']}`",
        f"- Helpful memories used by multiple failures: `{coverage['memory_count_helpful_for_multiple_queries']}`",
        "",
        f"| Variant | MRR helpful | Hit@1 | Hit@{diagnosis_k} | Random expected@{diagnosis_k} | Largest unresolved gap |",
        "| ------- | ----------: | ----: | -----: | --------------------: | ---------------------- |",
    ]
    for variant in V37_RETRIEVAL_VARIANTS:
        value = summary["retrieval_variants"][variant]
        hit1 = value["helpful_hit_at_k"]["1"]
        hitk = value["helpful_hit_at_k"][str(diagnosis_k)]
        lines.append(
            f"| {variant} | {value['mrr_first_helpful_on_recoverable']:.6f} "
            f"| {hit1['count']} "
            f"| {hitk['count']} "
            f"| {hitk['uniform_random_expected_count']:.6f} "
            f"| {value['largest_observed_unresolved_gap']} |"
        )
    lines.extend([
        "",
        "## P2 bottleneck decomposition",
        "",
        f"Each row below partitions all failures at fixed K=`{diagnosis_k}` into four mutually exclusive outcomes.",
        "",
        "| Variant | No helpful in bank | Helpful missed by Top-K | Helpful in Top-K, missed Top-1 | Helpful at Top-1 |",
        "| ------- | -----------------: | ----------------------: | -----------------------------: | ---------------: |",
    ])
    for variant in V37_RETRIEVAL_VARIANTS:
        value = summary["retrieval_variants"][variant][
            "pipeline_decomposition_at_diagnosis_k"
        ]
        lines.append(
            f"| {variant} | {value['no_helpful_in_authenticated_bank']} "
            f"| {value['helpful_exists_but_missed_top_k']} "
            f"| {value['helpful_in_top_k_but_missed_top1']} "
            f"| {value['helpful_at_top1']} |"
        )
    lines.extend([
        "",
        f"- Cross-variant largest-gap consensus: `{summary['bottleneck_attribution']['cross_variant_consensus']}`",
        "- Online variant selected: `false`",
        "- Threshold fitted: `false`",
        "",
        "`Full bank` here means the complete authenticated V3.6 state-key universe,",
        "not every original side-KV record. The failure-only sweep can establish",
        "helpful coverage and retrieval recall, but cannot measure treatment harm",
        "because strict-reward baselines are already zero. Non-helpful treatment may",
        "reflect the value, the fixed persistent-to-EOS injection policy, or both.",
        "",
    ])
    return "\n".join(lines)


def _verify_completed_report(
    *,
    report_path: Path,
    profile_sha256: str,
    treatment_path: Path,
    matrix_path: Path,
    expected_pairs: int,
) -> bool:
    if not report_path.is_file() or not treatment_path.is_file() or not matrix_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed_diagnostic":
        return False
    if (
        report.get("schema_version") != V38_REPORT_SCHEMA
        or report.get("profile_sha256") != profile_sha256
        or report.get("report_sha256") != _logical_hash(report, "report_sha256")
        or report.get("qualified_for_online_use") is not False
        or report.get("formal_v3_5_qualification_changed") is not False
    ):
        raise ValueError("completed V3.8 report failed logical authentication")
    artifacts = report.get("artifacts", {})
    if (
        artifacts.get("treatments", {}).get("sha256") != file_sha256(treatment_path)
        or artifacts.get("matrix", {}).get("sha256") != file_sha256(matrix_path)
        or int(artifacts.get("treatments", {}).get("row_count", -1))
        != expected_pairs
    ):
        raise ValueError("completed V3.8 artifact authentication failed")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (
        matrix.get("schema_version") != V38_UTILITY_MATRIX_SCHEMA
        or matrix.get("profile_sha256") != profile_sha256
        or matrix.get("matrix_sha256") != _logical_hash(matrix, "matrix_sha256")
    ):
        raise ValueError("completed V3.8 utility matrix authentication failed")
    return True


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.experience.memory import MemoryRecord
    from memgen.experience.v3 import ExperienceMemoryV3Profile
    from memgen.experience.v3_artifacts import (
        authenticate_e0_inputs,
        load_formal_e0_report,
        validate_cross_bank_metadata,
    )
    from memgen.model.e1_runtime import GreedyE1Runtime
    from memgen.model.retrieval_keys import tensor_sha256
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader
    from memgen.model.v3_5_retrieval import DualRetrievalKeyBankLoader
    from memgen.model.v3_runtime import EntropyHysteresisGate

    split = _load_split(args.split_manifest)
    v36_report, state_manifest = _load_v36_inputs(args)
    v37_profile, v37_report, v37_queries, v37_treatments = _authenticate_v37(
        args
    )
    failure_queries = [
        row
        for row in v37_queries
        if row.get("gate_eligible") is True
        and float(row.get("baseline", {}).get("strict_reward", -1.0)) == 0.0
    ]
    failure_ids = tuple(str(row["sample_id"]) for row in failure_queries)
    if not failure_ids or len(set(failure_ids)) != len(failure_ids):
        raise ValueError("V3.8 requires unique V3.7 gate-eligible failures")
    split_by_id = {str(row["sample_id"]): row for row in split["samples"]}
    if any(
        sample_id not in split_by_id
        or split_by_id[sample_id].get("logical_split") != "dev-test"
        for sample_id in failure_ids
    ):
        raise ValueError("V3.8 failures are outside authenticated dev-test")

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    validate_cross_bank_metadata(records=records, side_manifest=side_manifest)
    reasoner = side_manifest["reasoner"]
    memory_ids, state_keys = _load_state_key_bank(
        args.state_key_manifest, state_manifest
    )
    if args.diagnosis_k > len(memory_ids):
        raise ValueError("V3.8 diagnosis-k exceeds the authenticated bank")
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    state_entries = {
        str(entry["memory_id"]): entry for entry in state_manifest["records"]
    }
    record_ids = {record.memory_id for record in records}
    if not set(memory_ids).issubset(side_entries) or not set(memory_ids).issubset(
        record_ids
    ):
        raise ValueError("V3.8 state-key IDs are not bound to side-KV values")
    for memory_id in memory_ids:
        if (
            str(state_entries[memory_id].get("payload_hash"))
            != str(side_entries[memory_id].get("payload_hash"))
            or int(state_entries[memory_id].get("kv_valid_slot_count", -1))
            != int(side_entries[memory_id].get("kv_valid_slot_count", -2))
        ):
            raise ValueError(f"V3.8 state/value binding drifted: {memory_id}")
    bank_question_hash = _bank_question_hashes(
        args.source_alignment_evidence, memory_ids
    )
    if any(
        bank_question_hash[memory_id]
        == str(split_by_id[sample_id]["question_sha256"])
        for sample_id in failure_ids
        for memory_id in memory_ids
    ):
        raise ValueError("V3.8 found same-question memory leakage")

    dual_bank = DualRetrievalKeyBankLoader(
        manifest_path=args.dual_key_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    if not set(memory_ids).issubset(dual_bank.entry_by_id):
        raise ValueError("V3.8 text control does not cover the state-key universe")
    applicability_keys = F.normalize(
        torch.stack([
            dual_bank.applicability_embeddings[
                int(dual_bank.entry_by_id[memory_id]["index"])
            ]
            for memory_id in memory_ids
        ]).double(),
        dim=-1,
    ).cpu().contiguous()

    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    gate_wrapper = EntropyHysteresisGate.from_token_artifact(risk_artifact)
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V3.8 risk and side-KV reasoner provenance differs")
    side_profile = ExperienceMemoryV3Profile.continuous_token_joint()

    implementation_paths = (
        "data/gsm8k/prompt.py",
        "data/utils/math_utils.py",
        "memgen/experience/v3_6_state_keys.py",
        "memgen/experience/v3_7_cross_problem.py",
        "memgen/experience/v3_8_full_bank.py",
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_5_retrieval.py",
        "memgen/model/v3_runtime.py",
        "scripts/audit_v3_7_cross_problem_causal_applicability.py",
        "scripts/audit_v3_8_failure_full_bank_causal.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    input_hashes = {
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "memory_records_sha256": file_sha256(args.memory_records),
        "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
        "e0_final_report_sha256": file_sha256(args.e0_final_report),
        "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
        "source_alignment_evidence_sha256": file_sha256(
            args.source_alignment_evidence
        ),
        "v36_report_sha256": file_sha256(args.v36_report),
        "state_key_manifest_sha256": file_sha256(args.state_key_manifest),
        "v37_profile_sha256": file_sha256(args.v37_dir / "causal_profile.json"),
        "v37_queries_sha256": file_sha256(args.v37_dir / "causal_queries.jsonl"),
        "v37_treatments_sha256": file_sha256(
            args.v37_dir / "causal_treatments.jsonl"
        ),
        "v37_report_sha256": file_sha256(args.v37_dir / "causal_report.json"),
    }
    profile_material = {
        "schema_version": V38_PROFILE_SCHEMA,
        "git_revision": git_revision(),
        "inputs": input_hashes,
        "implementation_files_sha256": implementation_hashes,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_hashes
        ),
        "v37_profile_logical_sha256": v37_profile["profile_sha256"],
        "v37_report_logical_sha256": v37_report["report_sha256"],
        "v36_report_logical_sha256": v36_report["report_sha256"],
        "state_key_manifest_logical_sha256": state_manifest["manifest_sha256"],
        "selected_failure_sample_ids": list(failure_ids),
        "failure_selection": "v37_gate_eligible_and_strict_baseline_reward_zero",
        "memory_ids": list(memory_ids),
        "memory_universe": "complete_authenticated_v36_state_key_bank",
        "diagnosis_k": args.diagnosis_k,
        "retrieval_variants": list(V37_RETRIEVAL_VARIANTS),
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "same_question_memory_policy": "strictly_excluded_and_fail_closed",
        "treatment": "persistent_existing_full_when_facing_prefer_avoid_side_kv",
        "utility": "strict_reward_treatment_minus_same_prefix_baseline",
        "v37_candidate_treatment_reuse": "authenticated_exact_evidence_reuse",
        "answer_or_reward_scope": (
            "v37_failure_query_selection_and_post_full_universe_treatment_label_only"
        ),
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "gate": gate_wrapper.config.to_dict(),
        "side_kv_profile": side_profile.to_dict(),
    }
    profile_sha256 = canonical_json_sha256(profile_material)
    run_profile = {
        **profile_material,
        "created_at": utc_now(),
        "profile_sha256": profile_sha256,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / PROFILE_FILE
    treatment_path = args.output_dir / TREATMENT_FILE
    matrix_path = args.output_dir / MATRIX_FILE
    report_path = args.output_dir / REPORT_FILE
    markdown_path = args.output_dir / MARKDOWN_FILE
    if profile_path.is_file():
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
        if existing.get("profile_sha256") != profile_sha256:
            raise ValueError("V3.8 output directory belongs to another profile")
    else:
        write_json_atomic(profile_path, run_profile)

    expected_pairs = len(failure_ids) * len(memory_ids)
    if _verify_completed_report(
        report_path=report_path,
        profile_sha256=profile_sha256,
        treatment_path=treatment_path,
        matrix_path=matrix_path,
        expected_pairs=expected_pairs,
    ):
        print(f"[v3.8] reusing authenticated completed report: {report_path}")
        return

    treatment_rows = load_jsonl(treatment_path)
    completed_pairs: set[tuple[str, str]] = set()
    treatment_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in treatment_rows:
        pair = _validate_treatment_row(
            row=row,
            profile_sha256=profile_sha256,
            failure_ids=set(failure_ids),
            memory_ids=set(memory_ids),
        )
        if pair in completed_pairs:
            raise ValueError("V3.8 treatment resume evidence is duplicated")
        completed_pairs.add(pair)
        treatment_by_pair[pair] = row

    v37_failure_treatments = [
        row for row in v37_treatments if str(row["sample_id"]) in set(failure_ids)
    ]
    v37_failure_by_pair = {
        (str(row["sample_id"]), str(row["memory_id"])): row
        for row in v37_failure_treatments
    }
    if len(v37_failure_by_pair) != len(v37_failure_treatments):
        raise ValueError("V3.7 failure treatment evidence is duplicated")
    for row in treatment_rows:
        if row.get("evidence_origin") != "v3.7_authenticated_reuse":
            continue
        pair = (str(row["sample_id"]), str(row["memory_id"]))
        source_row = v37_failure_by_pair.get(pair)
        if source_row is None or row != _convert_v37_treatment(
            row=source_row, profile_sha256=profile_sha256
        ):
            raise ValueError("V3.8 reused V3.7 treatment evidence drifted")
    with treatment_path.open("a", encoding="utf-8") as handle:
        for source_row in v37_failure_treatments:
            pair = (str(source_row["sample_id"]), str(source_row["memory_id"]))
            if pair in completed_pairs:
                continue
            converted = _convert_v37_treatment(
                row=source_row, profile_sha256=profile_sha256
            )
            _validate_treatment_row(
                row=converted,
                profile_sha256=profile_sha256,
                failure_ids=set(failure_ids),
                memory_ids=set(memory_ids),
            )
            append_jsonl(handle, converted)
            treatment_rows.append(converted)
            completed_pairs.add(pair)
            treatment_by_pair[pair] = converted

    write_json_atomic(
        report_path,
        _progress_report(
            profile_sha256=profile_sha256,
            expected_pairs=expected_pairs,
            rows=treatment_rows,
        ),
    )
    if len(completed_pairs) < expected_pairs:
        tokenizer = AutoTokenizer.from_pretrained(
            reasoner["model_name"], revision=reasoner["tokenizer_revision"]
        )
        tokenizer.chat_template = CONVERSATION_TEMPLATE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            reasoner["model_name"],
            revision=reasoner["model_revision"],
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(args.device)
        model.eval()
        resolved_model = str(
            getattr(model.config, "_commit_hash", None)
            or reasoner["model_revision"]
        )
        resolved_tokenizer = str(
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or reasoner["tokenizer_revision"]
        )
        if (
            resolved_model != reasoner["model_revision"]
            or resolved_tokenizer != reasoner["tokenizer_revision"]
        ):
            raise ValueError("V3.8 resolved model/tokenizer revision drifted")
        side_loader = SideKVBankLoader(
            manifest_path=args.side_kv_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
        )
        controller = SideKVAttentionController(
            model=model,
            layer_number=side_profile.layer_number,
            audit_canonical_rope=False,
            memory_score_normalization=side_profile.memory_score_normalization,
            memory_score_bias=side_profile.memory_score_bias,
        )
        runtime = GreedyE1Runtime(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            revision=split["dataset"]["revision"],
        )
        v37_query_by_id = {
            str(row["sample_id"]): row for row in failure_queries
        }
        try:
            with treatment_path.open("a", encoding="utf-8") as handle:
                for query_index, sample_id in enumerate(failure_ids, start=1):
                    remaining = [
                        memory_id
                        for memory_id in memory_ids
                        if (sample_id, memory_id) not in completed_pairs
                    ]
                    if not remaining:
                        continue
                    entry = split_by_id[sample_id]
                    source = dataset[int(entry["source_index"])]
                    question = str(source["question"]).strip()
                    answer = str(source["answer"]).strip()
                    question_hash = text_sha256(question)
                    if (
                        question_hash != entry.get("question_sha256")
                        or text_sha256(answer) != entry.get("answer_sha256")
                    ):
                        raise ValueError(f"V3.8 dataset drift: {sample_id}")
                    prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(
                        tokenizer, question
                    )
                    observation = _generate_continuous_observation(
                        runtime=runtime,
                        prompt_token_ids=prompt_ids,
                        gate=gate_wrapper.diagnostic_gate,
                    )
                    stored_query = v37_query_by_id[sample_id]
                    prefix_ids = tuple(int(value) for value in observation.prefix_token_ids)
                    ground_truth = _processed_solution(answer)
                    baseline = _score_condition(
                        tokenizer=tokenizer,
                        completion_ids=observation.completion_token_ids,
                        ground_truth=ground_truth,
                    )
                    if (
                        observation.gate_observation is None
                        or baseline["strict_reward"] != 0.0
                        or baseline["completion_token_ids_sha256"]
                        != stored_query["baseline"]["completion_token_ids_sha256"]
                        or canonical_json_sha256(list(prefix_ids))
                        != stored_query["prefix_token_ids_sha256"]
                        or observation.gate_observation.to_dict()
                        != stored_query["gate_observation"]
                    ):
                        raise ValueError(f"V3.8 could not reproduce V3.7 failure: {sample_id}")
                    query_vectors, _ = _encode_query_variants(
                        model=model,
                        prompt_ids=prompt_ids,
                        prefix_ids=prefix_ids,
                        device=args.device,
                        layer_number=side_profile.layer_number,
                    )
                    if {
                        component: tensor_sha256(vector)
                        for component, vector in query_vectors.items()
                    } != stored_query["query_embedding_sha256"]:
                        raise ValueError(f"V3.8 query embedding drift: {sample_id}")
                    _, _, rankings, ranks, scores = _retrieval_for_query(
                        memory_ids=memory_ids,
                        state_keys=state_keys,
                        applicability_keys=applicability_keys,
                        query_vectors=query_vectors,
                        candidate_top_k=int(v37_profile["candidate_top_k"]),
                        random_count=int(v37_profile["random_candidates"]),
                        rrf_rank_constant=int(v37_profile["rrf_rank_constant"]),
                        seed=int(v37_profile["seed"]),
                        sample_id=sample_id,
                    )
                    if any(
                        list(rankings[variant][: int(v37_profile["candidate_top_k"])])
                        != stored_query["candidate_top_memory_ids"][variant]
                        for variant in V37_RETRIEVAL_VARIANTS
                    ):
                        raise ValueError(f"V3.8 V3.7 retrieval reproduction drift: {sample_id}")
                    partial_length = len(prefix_ids) - len(prompt_ids)
                    if partial_length >= len(observation.completion_token_ids):
                        raise ValueError("V3.8 gate prefix has no baseline next token")
                    expected_first = int(
                        observation.completion_token_ids[partial_length]
                    )
                    for memory_index, memory_id in enumerate(remaining, start=1):
                        memory = side_loader.get(
                            memory_id,
                            device=args.device,
                            dtype=next(model.parameters()).dtype,
                        )
                        started = time.perf_counter()
                        result = runtime.generate_from_trigger_with_persistent_memory(
                            prefix_token_ids=prefix_ids,
                            prompt_token_count=len(prompt_ids),
                            memory=memory,
                            controller=controller,
                        )
                        treatment = _score_condition(
                            tokenizer=tokenizer,
                            completion_ids=result.completion_token_ids,
                            ground_truth=ground_truth,
                        )
                        treatment["runtime_seconds"] = time.perf_counter() - started
                        utility = causal_utility(
                            baseline_reward=baseline["strict_reward"],
                            treatment_reward=treatment["strict_reward"],
                        )
                        row = {
                            "schema_version": V38_TREATMENT_SCHEMA,
                            "profile_sha256": profile_sha256,
                            "sample_id": sample_id,
                            "question_sha256": question_hash,
                            "memory_id": memory_id,
                            "memory_source_question_sha256": bank_question_hash[memory_id],
                            "cross_problem": True,
                            "same_question": False,
                            "rank_by_variant": {
                                variant: int(ranks[variant][memory_id])
                                for variant in V37_RETRIEVAL_VARIANTS
                            },
                            "score_by_variant": {
                                variant: float(scores[variant][memory_id])
                                for variant in V37_RETRIEVAL_VARIANTS
                            },
                            "baseline_reward": 0.0,
                            "treatment_reward": float(treatment["strict_reward"]),
                            "causal_utility": utility,
                            "causal_label": "helpful" if utility == 1 else "neutral",
                            "prefix_token_ids_sha256": canonical_json_sha256(
                                list(prefix_ids)
                            ),
                            "payload_hash": memory.payload_hash,
                            "treatment": treatment,
                            "side_kv_trace": _trace_summary(
                                result=result,
                                expected_memory_id=memory_id,
                                expected_baseline_first_token=expected_first,
                            ),
                            "evidence_origin": "v3.8_full_bank_generation",
                        }
                        append_jsonl(handle, row)
                        treatment_rows.append(row)
                        completed_pairs.add((sample_id, memory_id))
                        treatment_by_pair[(sample_id, memory_id)] = row
                        print(
                            f"[v3.8] query={query_index}/{len(failure_ids)} "
                            f"{sample_id} remaining={memory_index}/{len(remaining)} "
                            f"memory={memory_id} utility={utility:+d}",
                            flush=True,
                        )
                    write_json_atomic(
                        report_path,
                        _progress_report(
                            profile_sha256=profile_sha256,
                            expected_pairs=expected_pairs,
                            rows=treatment_rows,
                        ),
                    )
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        finally:
            controller.close()

    ordered_rows = [
        treatment_by_pair[(sample_id, memory_id)]
        for sample_id in failure_ids
        for memory_id in memory_ids
    ]
    matrix = build_utility_matrix(
        query_ids=failure_ids,
        memory_ids=memory_ids,
        treatment_rows=ordered_rows,
    )
    matrix_artifact = {
        "schema_version": V38_UTILITY_MATRIX_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "profile_sha256": profile_sha256,
        **matrix,
    }
    matrix_artifact["matrix_sha256"] = canonical_json_sha256(matrix_artifact)
    write_json_atomic(matrix_path, matrix_artifact)
    summary = summarize_full_bank_matrix(
        query_ids=failure_ids,
        memory_ids=memory_ids,
        treatment_rows=ordered_rows,
        diagnosis_k=args.diagnosis_k,
    )
    v37_harm = v37_report["summary"]["treatment_pair_counts"]
    report: dict[str, Any] = {
        "schema_version": V38_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "qualified_for_online_use": False,
        "formal_v3_5_qualification_changed": False,
        "reasoner_forward_and_generation_run": True,
        "side_kv_treatment_run": True,
        "side_kv_payload_changed": False,
        "task_accuracy_used": True,
        "answer_or_reward_used": True,
        "answer_or_reward_scope": (
            "authenticated_v37_failure_query_selection_and_post_fixed_full_bank_"
            "treatment_labels_never_query_key_rank_or_memory_selection"
        ),
        "same_question_memory_permitted": False,
        "cross_problem_enforced": True,
        "variant_selected": False,
        "threshold_fitted": False,
        "profile_sha256": profile_sha256,
        "configuration": {
            "query_scope": "v37_gate_eligible_strict_baseline_failures",
            "memory_scope": "complete_authenticated_v36_state_key_bank",
            "original_side_kv_memory_count": len(side_manifest["records"]),
            "authenticated_state_key_memory_count": len(memory_ids),
            "excluded_memory_count_without_authenticated_state_key": (
                len(side_manifest["records"]) - len(memory_ids)
            ),
            "diagnosis_k": args.diagnosis_k,
            "retrieval_variants": list(V37_RETRIEVAL_VARIANTS),
            "state_key_trajectory": "reference_failure_first_gate",
            "runtime_query_trajectory": "heldout_vanilla_generation_first_gate",
            "treatment": "persistent_existing_full_when_facing_prefer_avoid_side_kv",
            "utility": "strict_reward_treatment_minus_same_prefix_baseline",
            "full_authenticated_bank_exhaustively_treated": True,
            "full_original_side_kv_bank_exhaustively_treated": False,
            "stable_tie_break": "memory_id_ascending",
        },
        "summary": summary,
        "inherited_v37_risk_context": {
            "scope": "candidate_pool_on_gate_eligible_failures_and_successes",
            "helpful_pair_count": int(v37_harm["helpful"]),
            "harmful_pair_count": int(v37_harm["harmful"]),
            "neutral_pair_count": int(v37_harm["neutral"]),
            "not_full_bank_harm_estimate": True,
        },
        "interpretation_contract": {
            "recoverable_failure": "at_least_one_authenticated_memory_has_positive_causal_utility",
            "unrecoverable_under_audit": (
                "no_authenticated_state_key_memory_improves_strict_reward_under_"
                "the_fixed_persistent_injection_policy"
            ),
            "coverage_gap_is_not_pure_key_evidence": True,
            "coverage_gap_may_be_value_or_injection_policy": True,
            "retrieval_gap_conditions_on_helpful_memory_existing": True,
            "failure_only_matrix_cannot_estimate_harm": True,
            "diagnostic_does_not_select_online_variant": True,
        },
        "inputs": {
            **input_hashes,
            "git_revision": run_profile["git_revision"],
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(
                implementation_hashes
            ),
        },
        "artifacts": {
            "profile": {
                "path": PROFILE_FILE,
                "sha256": file_sha256(profile_path),
            },
            "treatments": {
                "path": TREATMENT_FILE,
                "sha256": file_sha256(treatment_path),
                "row_count": len(ordered_rows),
            },
            "matrix": {
                "path": MATRIX_FILE,
                "sha256": file_sha256(matrix_path),
                "shape": [len(failure_ids), len(memory_ids)],
            },
        },
        "requirements": {
            "v37_inputs_authenticated": True,
            "v37_gate_eligible_baseline_failures_only": True,
            "heldout_dev_test_only": True,
            "bank_source_and_query_questions_disjoint": True,
            "same_question_memory_excluded": True,
            "state_key_value_binding_authenticated": True,
            "complete_authenticated_state_key_bank_treated": True,
            "strict_reward_never_used_for_query_encoding_key_rank_or_within_query_memory_selection": True,
            "baseline_and_treatment_share_exact_prefix": True,
            "baseline_first_token_parity_checked_for_generated_treatments": True,
            "reused_treatments_authenticated_by_v37_artifact": True,
            "existing_side_kv_payload_preserved": True,
            "failure_only_harm_limit_disclosed": True,
            "variant_not_selected": True,
            "threshold_not_fitted": True,
            "formal_v3_5_qualification_unchanged": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json_atomic(report_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(
        f"[v3.8] status={report['status']} failures={len(failure_ids)} "
        f"memories={len(memory_ids)} treatments={len(ordered_rows)} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
