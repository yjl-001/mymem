#!/usr/bin/env python3
"""Re-encode V3 calibration first attempts and audit frozen pooling candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v3 import V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
from memgen.experience.v3_pooling import (
    V3_POOLING_AUDIT_SCHEMA,
    V3_POOLING_BASELINE,
    V3_POOLING_CANDIDATES,
    V3_POOLING_EMBEDDING_SCHEMA,
    V3_POOLING_FULL_MEAN,
    V3_POOLING_PARTIAL_MEAN,
    V3_POOLING_PRE_BOUNDARY,
    V3_POOLING_SAMPLE_SCHEMA,
    qualify_pooling_candidate,
    rank_qualified_pooling_candidates,
    reconstruct_first_attempt_prefix,
    stable_top_indices,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    numeric_summary,
    selection_concentration,
)


EVALUATION_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-profile", type=Path, required=True)
    parser.add_argument("--selector-calibration", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evaluation_profile_sha256(value: Mapping[str, Any]) -> str:
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "repository", "profile_sha256"}
    }
    repository = value.get("repository", {})
    material["code_identity"] = {
        "git_revision": repository.get("git_revision"),
        "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        "implementation_set_sha256": repository.get(
            "implementation_set_sha256"
        ),
    }
    return canonical_json_sha256(material)


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    system = value.get("system_profile", {})
    retrieval_transform = system.get(
        "retrieval_embedding_transform",
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    if (
        value.get("schema_version") != EVALUATION_PROFILE_SCHEMA
        or value.get("profile_sha256") != evaluation_profile_sha256(value)
        or value.get("logical_split") != "calibration-val"
        or system.get("retrieval_abstention_policy") != "disabled"
        or system.get("retrieval_min_top1_top2_margin") not in {None, ""}
        or retrieval_transform != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
        or value.get("slice") != {"offset": 0, "limit": 0}
    ):
        raise ValueError("Pooling audit requires the authenticated V3.1 raw calibration run")
    return value


def load_split_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("Pooling audit split manifest differs from the source run")
    value = json.loads(path.read_text(encoding="utf-8"))
    actual = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if (
        value.get("schema_version") != SPLIT_MANIFEST_SCHEMA
        or value.get("manifest_sha256") != actual
        or value.get("overlap_check", {}).get("passed") is not True
    ):
        raise ValueError("Invalid pooling-audit split manifest")
    return value


def load_first_attempt_rows(
    path: Path,
    *,
    profile: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    source = calibration.get("source", {})
    if (
        file_sha256(path) != source.get("results_file_sha256")
        or profile.get("profile_sha256") != source.get("run_profile_sha256")
    ):
        raise ValueError("Pooling audit inputs differ from selector calibration")
    values = []
    row_count = 0
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            actual_hash = canonical_json_sha256({
                key: item
                for key, item in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            if (
                row.get("schema_version") != EVALUATION_ROW_SCHEMA
                or row.get("profile_sha256") != profile.get("profile_sha256")
                or row.get("row_sha256") != actual_hash
                or not sample_id
                or sample_id in seen
            ):
                raise ValueError(f"Invalid pooling source row at line {line_number}")
            seen.add(sample_id)
            row_count += 1
            attempts = row["conditions"]["v3"]["runtime_trace"][
                "retrieval_attempts"
            ]
            if attempts:
                values.append({"row": row, "first_attempt": attempts[0]})
    if row_count != int(profile.get("selected_sample_count", -1)):
        raise ValueError("Pooling audit source results are incomplete")
    expected_first_attempts = int(
        calibration.get("calibration", {}).get("sample_count", -1)
    )
    if len(values) != expected_first_attempts:
        raise ValueError("Pooling audit first-attempt count differs from calibration")
    return values, row_count


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def safe_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "experience_type": record.experience_type,
        "when_facing": record.sanitized_fields.get("when_facing"),
        "prefer": record.sanitized_fields.get("prefer"),
        "avoid": record.sanitized_fields.get("avoid"),
        "payload_hash": record.payload_hash,
    }


def candidate_specs() -> dict[str, dict[str, str]]:
    return {
        V3_POOLING_BASELINE: {
            "key_pooling": "last_valid_token",
            "query_pooling": "boundary_last_token",
        },
        V3_POOLING_PRE_BOUNDARY: {
            "key_pooling": "last_valid_token",
            "query_pooling": "last_token_before_trigger_boundary",
        },
        V3_POOLING_PARTIAL_MEAN: {
            "key_pooling": "float32_mean_all_key_tokens",
            "query_pooling": "float32_mean_partial_cot_excluding_boundary",
        },
        V3_POOLING_FULL_MEAN: {
            "key_pooling": "float32_mean_all_key_tokens",
            "query_pooling": "float32_mean_full_prefix_excluding_boundary",
        },
    }


def vector_for_query_pool(raw_vector: Any) -> Any:
    """Mirror query encoder GPU normalization plus retriever CPU normalization."""

    import torch.nn.functional as F

    once = F.normalize(raw_vector.detach().float(), dim=0)
    return F.normalize(once.cpu(), dim=0).contiguous()


def vector_for_key_pool(raw_vector: Any) -> Any:
    """Mirror retrieval-key compiler GPU float32 L2 normalization."""

    import torch.nn.functional as F

    return F.normalize(raw_vector.detach().float(), dim=0).cpu().contiguous()


def encode_layer_hidden(
    *, model: Any, token_ids: Sequence[int], layer_number: int, device: str
) -> Any:
    import torch

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
        raise RuntimeError("Pooling audit layer-24 hidden state is unavailable")
    hidden = hidden_states[layer_number][0]
    if hidden.shape[0] != len(token_ids):
        raise RuntimeError("Pooling audit hidden-state length drifted")
    return hidden


def score_candidate(
    *,
    key_embeddings: Any,
    query_embedding: Any,
    entries: Sequence[Mapping[str, Any]],
    key_hashes: Sequence[str],
) -> tuple[list[dict[str, Any]], float]:
    import torch

    scores = torch.mv(key_embeddings, query_embedding)
    indices = stable_top_indices(scores.tolist(), top_k=2)
    if len(indices) < 2:
        raise ValueError("Pooling audit requires at least two memories")
    hits = []
    for rank, index in enumerate(indices, start=1):
        entry = entries[index]
        hits.append({
            "memory_id": str(entry["memory_id"]),
            "payload_hash": str(entry["payload_hash"]),
            "score": float(scores[index].item()),
            "rank": rank,
            "key_embedding_sha256": key_hashes[index],
        })
    return hits, float(hits[0]["score"] - hits[1]["score"])


def boundary_strata(
    *,
    sample_logs: Sequence[Mapping[str, Any]],
    candidate: str,
    complete_memory_ids: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for sample in sample_logs:
        key = (
            int(sample["boundary_token_id"]),
            str(sample["boundary_token_text"]),
        )
        groups[key].append(
            str(sample["candidates"][candidate]["hits"][0]["memory_id"])
        )
    values = []
    for (token_id, token_text), memory_ids in groups.items():
        values.append({
            "boundary_token_id": token_id,
            "boundary_token_text": token_text,
            "sample_count": len(memory_ids),
            "selection_concentration": selection_concentration(
                memory_ids,
                complete_memory_ids=complete_memory_ids,
            ),
        })
    values.sort(key=lambda item: (-int(item["sample_count"]), int(item["boundary_token_id"])))
    return values


def markdown_report(value: Mapping[str, Any]) -> str:
    reproduction = value["baseline_reproduction"]
    qualification = value["qualification"]
    lines = [
        "# MemGen V3.3 pooling geometry audit",
        "",
        f"- Status: `{value['status']}`",
        f"- First-attempt samples: {value['sample_count']}",
        f"- Memory count: {value['memory_count']}",
        f"- Baseline top-1 reproduction: "
        f"{reproduction['top1_exact_match_count']} / {value['sample_count']}",
        f"- Query embedding hash reproduction: "
        f"{reproduction['query_embedding_hash_match_count']} / {value['sample_count']}",
        f"- Key embedding hash reproduction: "
        f"{reproduction['key_embedding_hash_match_count']} / {value['memory_count']}",
        f"- Recommended candidate: `{qualification['recommended_candidate']}`",
        "",
        "## Answer-blind candidate geometry",
        "",
        "| Candidate | Top-1 share | Gini | Selected memories | Normalized entropy | Top-5 share | Median margin | Dominant memory | Qualified |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name in V3_POOLING_CANDIDATES:
        summary = value["candidates"][name]["selection_concentration"]
        dominant = summary["top_by_frequency"][0]
        qualified = (
            False
            if name == V3_POOLING_BASELINE
            else qualification["candidates"][name]["qualified"]
        )
        lines.append(
            f"| {name} | {summary['top1_share']} | {summary['gini']} | "
            f"{summary['selected_memory_count']} | {summary['normalized_entropy']} | "
            f"{summary['top5_share']} | "
            f"{value['candidates'][name]['top1_top2_margin']['median']} | "
            f"{dominant['memory_id']} ({dominant['count']}) | {qualified} |"
        )
    lines.extend([
        "",
        "## Boundary-token strata",
        "",
        "| Candidate | Boundary token ID | Boundary text | Samples | Top-1 share | Selected memories | Gini |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for name in V3_POOLING_CANDIDATES:
        for stratum in value["candidates"][name]["boundary_strata"][:5]:
            concentration = stratum["selection_concentration"]
            boundary_text = json.dumps(
                stratum["boundary_token_text"], ensure_ascii=False
            ).replace("|", "\\|")
            lines.append(
                f"| {name} | {stratum['boundary_token_id']} | "
                f"`{boundary_text}` | {stratum['sample_count']} | "
                f"{concentration['top1_share']} | "
                f"{concentration['selected_memory_count']} | "
                f"{concentration['gini']} |"
            )
    lines.extend([
        "",
        "This audit is answer-blind and does not run dev-test or final-test.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("Pooling audit progress interval must be positive")

    import torch
    from datasets import load_dataset
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import (
        RetrievalKeyBankLoader,
        tensor_sha256,
    )

    profile = load_profile(args.run_profile)
    inputs = profile.get("inputs", {})
    calibration = load_margin_selector_calibration(args.selector_calibration)
    calibration_source = calibration.get("source", {})
    if (
        file_sha256(args.run_profile)
        != calibration_source.get("run_profile_file_sha256")
        or inputs.get("selector_calibration_sha256") not in {None, ""}
    ):
        raise ValueError("Pooling audit selector artifacts are not source-bound")
    # The raw calibration run itself has no selector artifact in its inputs;
    # the calibration artifact binds that run in the opposite direction.
    if calibration_source.get("retrieval_embedding_transform", "none") != "none":
        raise ValueError("Pooling audit baseline calibration is not raw cosine")
    if calibration.get("task_accuracy_used") is not False or calibration.get(
        "answer_or_reward_used"
    ) is not False:
        raise ValueError("Pooling audit requires answer-blind selector calibration")

    split_manifest = load_split_manifest(
        args.split_manifest,
        expected_sha256=str(inputs.get("split_manifest_sha256", "")),
    )
    for path, field_name in (
        (args.memory_records, "memory_records_sha256"),
        (args.retrieval_key_manifest, "retrieval_key_manifest_sha256"),
    ):
        if file_sha256(path) != inputs.get(field_name):
            raise ValueError(f"Pooling audit input differs: {field_name}")
    if (
        calibration_source.get("retrieval_key_manifest_sha256")
        != inputs.get("retrieval_key_manifest_sha256")
    ):
        raise ValueError("Pooling calibration and evaluation use different key banks")

    first_attempt_rows, source_row_count = load_first_attempt_rows(
        args.results,
        profile=profile,
        calibration=calibration,
    )
    selected_split_entries = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == "calibration-val"
    ]
    split_entries = {
        str(item["sample_id"]): item for item in selected_split_entries
    }
    if len(split_entries) != int(profile["selected_sample_count"]):
        raise ValueError("Pooling audit split coverage differs from source profile")
    if canonical_json_sha256([
        str(item["sample_id"]) for item in selected_split_entries
    ]) != profile.get("selected_sample_ids_sha256"):
        raise ValueError("Pooling audit split sample order differs from source profile")

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    record_by_id = {record.memory_id: record for record in records}
    reasoner = profile["reasoner"]
    if reasoner.get("runtime_dtype") != args.dtype:
        raise ValueError("Pooling audit dtype differs from raw calibration")
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    resolved_model = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != reasoner["model_revision"]
        or resolved_tokenizer != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved pooling-audit reasoner revision drifted")

    bank = RetrievalKeyBankLoader(
        manifest_path=args.retrieval_key_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    memory_ids = [str(entry["memory_id"]) for entry in bank.entries]
    if list(record_by_id) != memory_ids:
        raise ValueError("Pooling audit memory and key order differ")

    layer_number = int(profile["system_profile"]["layer_number"])
    if layer_number != 24:
        raise ValueError("Pooling audit is frozen to layer 24")
    key_last_vectors = []
    key_mean_vectors = []
    key_last_hash_matches = 0
    for index, (record, entry) in enumerate(zip(records, bank.entries), start=1):
        key_text = str(record.sanitized_fields["when_facing"]).strip()
        token_ids = list(tokenizer.encode(key_text, add_special_tokens=False))
        if (
            len(token_ids) != int(entry["key_token_count"])
            or canonical_json_sha256(token_ids)
            != entry["key_token_ids_sha256"]
        ):
            raise ValueError(f"Pooling audit key tokenization drifted: {record.memory_id}")
        hidden = encode_layer_hidden(
            model=model,
            token_ids=token_ids,
            layer_number=layer_number,
            device=args.device,
        )
        key_last = vector_for_key_pool(hidden[-1])
        key_mean = vector_for_key_pool(hidden.float().mean(dim=0))
        key_last_vectors.append(key_last)
        key_mean_vectors.append(key_mean)
        key_last_hash_matches += int(
            tensor_sha256(key_last) == entry["key_embedding_sha256"]
        )
        if index % args.progress_every == 0 or index == len(records):
            print(
                f"[v3.3-pooling] encoded_keys={index}/{len(records)}",
                flush=True,
            )
    key_last_matrix = torch.stack(key_last_vectors, dim=0).float()
    key_mean_matrix = torch.stack(key_mean_vectors, dim=0).float()
    if key_last_hash_matches != len(records):
        raise RuntimeError(
            "Pooling audit could not exactly reproduce the raw key bank"
        )
    key_hashes = {
        "last": [str(entry["key_embedding_sha256"]) for entry in bank.entries],
        "mean": [tensor_sha256(value) for value in key_mean_matrix],
    }

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=profile["dataset_split"],
        revision=profile["dataset_revision"],
    )
    sample_logs = []
    query_vectors: dict[str, list[Any]] = {
        "boundary_last": [],
        "pre_boundary": [],
        "partial_mean": [],
        "full_mean": [],
    }
    prefix_hash_matches = 0
    query_hash_matches = 0
    baseline_top1_matches = 0
    candidate_memory_ids: dict[str, list[str]] = {
        name: [] for name in V3_POOLING_CANDIDATES
    }
    candidate_margins: dict[str, list[float]] = {
        name: [] for name in V3_POOLING_CANDIDATES
    }
    candidate_top1_scores: dict[str, list[float]] = {
        name: [] for name in V3_POOLING_CANDIDATES
    }
    started = time.perf_counter()
    for position, value in enumerate(first_attempt_rows, start=1):
        row = value["row"]
        attempt = value["first_attempt"]
        sample_id = str(row["sample_id"])
        split_entry = split_entries.get(sample_id)
        if split_entry is None:
            raise ValueError(f"Pooling source sample is absent from split: {sample_id}")
        if int(split_entry["source_index"]) != int(row["source_index"]):
            raise ValueError(f"Pooling sample source index drifted: {sample_id}")
        source = dataset[int(row["source_index"])]
        question = str(source["question"]).strip()
        if (
            text_sha256(question) != row["question_sha256"]
            or text_sha256(question) != split_entry["question_sha256"]
        ):
            raise ValueError(f"Pooling source question hash drifted: {sample_id}")
        prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
        if (
            canonical_json_sha256(prompt_ids) != row["prompt_token_ids_sha256"]
            or len(prompt_ids) != int(row["prompt_token_count"])
        ):
            raise ValueError(f"Pooling source prompt drifted: {sample_id}")
        query_audit = attempt["retrieval_decision"]["query"]
        boundary_index = int(attempt["generated_boundary_index"])
        partial_count = boundary_index + 1
        completion_ids = [
            int(token)
            for token in row["conditions"]["v3"]["completion_token_ids"]
        ]
        try:
            prefix_ids = list(reconstruct_first_attempt_prefix(
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                generated_boundary_index=boundary_index,
                query_audit=query_audit,
            ))
        except ValueError as error:
            raise ValueError(f"{error}: {sample_id}") from error
        prefix_hash_matches += 1

        hidden = encode_layer_hidden(
            model=model,
            token_ids=prefix_ids,
            layer_number=layer_number,
            device=args.device,
        )
        boundary_query = vector_for_query_pool(hidden[-1])
        pre_boundary_query = vector_for_query_pool(hidden[-2])
        partial_mean_query = vector_for_query_pool(
            hidden[len(prompt_ids):-1].float().mean(dim=0)
        )
        full_mean_query = vector_for_query_pool(
            hidden[:-1].float().mean(dim=0)
        )
        per_sample_vectors = {
            "boundary_last": boundary_query,
            "pre_boundary": pre_boundary_query,
            "partial_mean": partial_mean_query,
            "full_mean": full_mean_query,
        }
        for name, vector in per_sample_vectors.items():
            query_vectors[name].append(vector)
        query_hash_matches += int(
            tensor_sha256(boundary_query)
            == query_audit["query_embedding_sha256"]
        )
        if query_hash_matches != position:
            raise RuntimeError(
                f"Pooling query embedding reproduction failed: {sample_id}"
            )

        matrices = {
            V3_POOLING_BASELINE: (key_last_matrix, boundary_query, key_hashes["last"]),
            V3_POOLING_PRE_BOUNDARY: (key_last_matrix, pre_boundary_query, key_hashes["last"]),
            V3_POOLING_PARTIAL_MEAN: (key_mean_matrix, partial_mean_query, key_hashes["mean"]),
            V3_POOLING_FULL_MEAN: (key_mean_matrix, full_mean_query, key_hashes["mean"]),
        }
        candidate_logs = {}
        for candidate, (keys, query, hashes) in matrices.items():
            hits, margin = score_candidate(
                key_embeddings=keys,
                query_embedding=query,
                entries=bank.entries,
                key_hashes=hashes,
            )
            top_memory_id = str(hits[0]["memory_id"])
            candidate_memory_ids[candidate].append(top_memory_id)
            candidate_margins[candidate].append(margin)
            candidate_top1_scores[candidate].append(float(hits[0]["score"]))
            candidate_logs[candidate] = {
                "query_embedding_sha256": tensor_sha256(query),
                "hits": hits,
                "top1_top2_margin": margin,
            }
        logged_top1 = str(attempt["selected_memory_id"])
        baseline_top1_matches += int(
            candidate_logs[V3_POOLING_BASELINE]["hits"][0]["memory_id"]
            == logged_top1
        )
        if baseline_top1_matches != position:
            raise RuntimeError(
                f"Pooling raw top-1 reproduction failed: {sample_id}"
            )
        sample_log = {
            "schema_version": V3_POOLING_SAMPLE_SCHEMA,
            "sample_id": sample_id,
            "source_index": int(row["source_index"]),
            "prompt_token_count": len(prompt_ids),
            "partial_cot_token_count": partial_count,
            "query_token_count": len(prefix_ids),
            "query_token_ids_sha256": canonical_json_sha256(prefix_ids),
            "boundary_token_id": int(prefix_ids[-1]),
            "boundary_token_text": tokenizer.decode(
                [int(prefix_ids[-1])], skip_special_tokens=False
            ),
            "logged_baseline_memory_id": logged_top1,
            "logged_query_embedding_sha256": query_audit[
                "query_embedding_sha256"
            ],
            "candidates": candidate_logs,
        }
        sample_log["sample_sha256"] = canonical_json_sha256(sample_log)
        sample_logs.append(sample_log)
        if position % args.progress_every == 0 or position == len(first_attempt_rows):
            print(
                f"[v3.3-pooling] encoded_queries={position}/{len(first_attempt_rows)} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

    logged_concentration = selection_concentration(
        [str(value["first_attempt"]["selected_memory_id"]) for value in first_attempt_rows],
        complete_memory_ids=memory_ids,
    )
    frozen_concentration = calibration["first_attempt_selection_concentration"]
    candidate_summaries = {}
    for candidate in V3_POOLING_CANDIDATES:
        concentration = selection_concentration(
            candidate_memory_ids[candidate],
            complete_memory_ids=memory_ids,
        )
        candidate_summaries[candidate] = {
            "specification": candidate_specs()[candidate],
            "selection_concentration": concentration,
            "top1_score": numeric_summary(candidate_top1_scores[candidate]),
            "top1_top2_margin": numeric_summary(candidate_margins[candidate]),
            "boundary_strata": boundary_strata(
                sample_logs=sample_logs,
                candidate=candidate,
                complete_memory_ids=memory_ids,
            ),
            "top_memory_payloads": [
                safe_payload(record_by_id[str(item["memory_id"])])
                | {"selection_count": int(item["count"])}
                for item in concentration["top_by_frequency"][:10]
            ],
        }
    geometry_summaries = {
        name: value["selection_concentration"]
        for name, value in candidate_summaries.items()
    }
    ranked_qualified = rank_qualified_pooling_candidates(geometry_summaries)
    candidate_qualification = {
        name: qualify_pooling_candidate(
            baseline=geometry_summaries[V3_POOLING_BASELINE],
            candidate=geometry_summaries[name],
        )
        for name in V3_POOLING_CANDIDATES
        if name != V3_POOLING_BASELINE
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "pooling_audit_samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for sample in sample_logs:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    embedding_path = args.output_dir / "pooling_embeddings.safetensors"
    tensors = {
        "key_last_reencoded": key_last_matrix.contiguous(),
        "key_mean": key_mean_matrix.contiguous(),
        "query_boundary_last": torch.stack(
            query_vectors["boundary_last"], dim=0
        ).float().contiguous(),
        "query_pre_boundary": torch.stack(
            query_vectors["pre_boundary"], dim=0
        ).float().contiguous(),
        "query_partial_mean": torch.stack(
            query_vectors["partial_mean"], dim=0
        ).float().contiguous(),
        "query_full_mean": torch.stack(
            query_vectors["full_mean"], dim=0
        ).float().contiguous(),
    }
    save_file(
        tensors,
        str(embedding_path),
        metadata={
            "schema_version": V3_POOLING_EMBEDDING_SCHEMA,
            "sample_order_sha256": canonical_json_sha256([
                sample["sample_id"] for sample in sample_logs
            ]),
            "memory_order_sha256": canonical_json_sha256(memory_ids),
        },
    )

    requirements = {
        "source_run_is_complete_authenticated_calibration_val": (
            source_row_count == int(profile["selected_sample_count"])
        ),
        "source_artifact_is_answer_blind": True,
        "prefix_hash_reproduction_is_exact": (
            prefix_hash_matches == len(sample_logs)
        ),
        "key_embedding_hash_reproduction_is_exact": (
            key_last_hash_matches == len(records)
        ),
        "query_embedding_hash_reproduction_is_exact": (
            query_hash_matches == len(sample_logs)
        ),
        "baseline_top1_reproduction_is_exact": (
            baseline_top1_matches == len(sample_logs)
        ),
        "logged_concentration_matches_frozen_calibration": (
            logged_concentration == frozen_concentration
        ),
        "recomputed_baseline_concentration_matches_frozen_calibration": (
            geometry_summaries[V3_POOLING_BASELINE]
            == frozen_concentration
        ),
        "layer_24_is_fixed": layer_number == 24,
        "task_accuracy_not_used": True,
        "answer_or_reward_not_used": True,
    }
    if not all(requirements.values()):
        failed = [name for name, passed in requirements.items() if not passed]
        raise RuntimeError(f"Pooling audit reproduction failed: {failed}")

    report = {
        "schema_version": V3_POOLING_AUDIT_SCHEMA,
        "created_at": utc_now(),
        "status": "passed",
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "sample_count": len(sample_logs),
        "memory_count": len(records),
        "layer_number": layer_number,
        "reasoner": dict(reasoner),
        "source": {
            "logical_split": "calibration-val",
            "scope": "first_retrieval_attempt_per_triggered_question",
            "source_row_count": source_row_count,
            "profile_sha256": profile["profile_sha256"],
            "selector_calibration_artifact_sha256": calibration[
                "artifact_sha256"
            ],
        },
        "pooling_contract": {
            "hidden_state_representation": "decoder_layer_output",
            "normalization": "l2",
            "retrieval_method": "exact_cosine",
            "embedding_transform": "none",
            "query_context": "question_plus_full_semantic_partial_cot",
            "trigger_boundary_token_excluded_only_where_candidate_states_so": True,
            "candidates": candidate_specs(),
        },
        "baseline_reproduction": {
            "prefix_hash_match_count": prefix_hash_matches,
            "key_embedding_hash_match_count": key_last_hash_matches,
            "query_embedding_hash_match_count": query_hash_matches,
            "top1_exact_match_count": baseline_top1_matches,
            "logged_selection_concentration": logged_concentration,
            "frozen_selection_concentration": frozen_concentration,
        },
        "candidates": candidate_summaries,
        "qualification": {
            "policy": (
                "lower_top1_and_gini_nonlower_support_higher_normalized_entropy"
            ),
            "candidates": candidate_qualification,
            "ranked_qualified_candidates": list(ranked_qualified),
            "recommended_candidate": (
                ranked_qualified[0] if ranked_qualified else None
            ),
        },
        "artifacts": {
            "sample_traces": {
                "path": sample_path.name,
                "sha256": file_sha256(sample_path),
                "schema_version": V3_POOLING_SAMPLE_SCHEMA,
            },
            "embeddings": {
                "path": embedding_path.name,
                "sha256": file_sha256(embedding_path),
                "schema_version": V3_POOLING_EMBEDDING_SCHEMA,
                "tensor_shapes": {
                    name: list(tensor.shape) for name, tensor in tensors.items()
                },
                "tensor_sha256": {
                    name: tensor_sha256(tensor) for name, tensor in tensors.items()
                },
            },
        },
        "requirements": requirements,
        "implementation": {
            "git_revision": git_revision(),
            "files_sha256": {
                "memgen/experience/v3_pooling.py": file_sha256(
                    PROJECT_ROOT / "memgen/experience/v3_pooling.py"
                ),
                "memgen/model/retrieval_keys.py": file_sha256(
                    PROJECT_ROOT / "memgen/model/retrieval_keys.py"
                ),
                "scripts/audit_v3_pooling_candidates.py": file_sha256(
                    PROJECT_ROOT / "scripts/audit_v3_pooling_candidates.py"
                ),
            },
        },
        "inputs": {
            "results_sha256": file_sha256(args.results),
            "run_profile_sha256": file_sha256(args.run_profile),
            "selector_calibration_sha256": file_sha256(
                args.selector_calibration
            ),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
        },
    }
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "created_at"
    })
    report_path = args.output_dir / "pooling_audit.json"
    write_json_atomic(report_path, report)
    markdown_path = args.output_dir / "pooling_audit.md"
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"[v3.3-pooling] status=passed samples={len(sample_logs)} "
        f"recommended={report['qualification']['recommended_candidate']} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
