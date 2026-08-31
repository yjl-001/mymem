#!/usr/bin/env python3
"""Resumable held-out cross-problem side-KV causal applicability audit.

The audit finds the first frozen V3.4 joint gate on disjoint GSM8K dev-test
problems, constructs state-space queries without labels, and evaluates a shared
candidate pool by branching from the exact same prefix with one foreign memory
active.  Strict GSM8K reward is used only after retrieval to define treatment
utility; it is never part of a query, key, candidate score, or candidate pool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import GSM8K_VERIFIER_VERSION, diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v3_6_state_keys import V36_STATE_KEY_BANK_SCHEMA
from memgen.experience.v3_7_cross_problem import (
    V37_CAUSAL_PROFILE_SCHEMA,
    V37_CAUSAL_QUERY_SCHEMA,
    V37_CAUSAL_REPORT_SCHEMA,
    V37_CAUSAL_TREATMENT_SCHEMA,
    V37_FUSION_VARIANT,
    V37_RETRIEVAL_VARIANTS,
    V37_STATE_COMPONENT_BY_VARIANT,
    candidate_union,
    causal_utility,
    reciprocal_rank_fusion_scores,
    stable_rank,
    summarize_causal_rows,
)


PROFILE_FILE = "causal_profile.json"
QUERY_FILE = "causal_queries.jsonl"
TREATMENT_FILE = "causal_treatments.jsonl"
REPORT_FILE = "causal_report.json"
MARKDOWN_FILE = "causal_report.md"
_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)", re.IGNORECASE
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Number of pre-registered dev-test samples; 0 uses the remainder.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=4)
    parser.add_argument("--rrf-rank-constant", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3617)
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
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V3.7 inputs are missing: {missing}")
    if (
        args.offset < 0
        or args.limit < 0
        or args.candidate_top_k <= 0
        or args.random_candidates < 0
        or args.rrf_rank_constant <= 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("V3.7 received invalid numeric arguments")
    if args.dtype != "bfloat16":
        raise ValueError("V3.7 is frozen to bfloat16 model compute")
    if args.max_new_tokens != GSM8K_PROMPT_CONTRACT.max_new_tokens:
        raise ValueError("V3.7 requires the canonical GSM8K token budget")


def _load_split(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if (
        value.get("schema_version") != SPLIT_MANIFEST_SCHEMA
        or value.get("manifest_sha256") != canonical_json_sha256(logical)
        or value.get("overlap_check", {}).get("passed") is not True
    ):
        raise ValueError("V3.7 split manifest authentication failed")
    return value


def _logical_report_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({
        key: item for key, item in value.items() if key != field
    })


def _load_v36_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(args.v36_report.read_text(encoding="utf-8"))
    state_manifest = json.loads(
        args.state_key_manifest.read_text(encoding="utf-8")
    )
    if (
        report.get("schema_version")
        != "experience-memory-v3.6-source-state-retrieval-key-report-v1"
        or report.get("status") != "completed_diagnostic"
        or report.get("qualified_for_online_use") is not False
        or report.get("report_sha256")
        != _logical_report_hash(report, "report_sha256")
    ):
        raise ValueError("V3.7 received an unauthenticated V3.6 report")
    inputs = report.get("inputs", {})
    if (
        inputs.get("split_manifest_sha256") != file_sha256(args.split_manifest)
        or inputs.get("dual_key_manifest_sha256")
        != file_sha256(args.dual_key_manifest)
        or inputs.get("token_risk_artifact_sha256")
        != file_sha256(args.token_risk_artifact)
        or inputs.get("source_alignment_evidence_sha256")
        != file_sha256(args.source_alignment_evidence)
    ):
        raise ValueError("V3.7 input identity differs from the V3.6 audit")
    artifact = report.get("artifacts", {}).get("state_key_bank", {})
    if artifact.get("manifest_sha256") != file_sha256(args.state_key_manifest):
        raise ValueError("V3.7 state-key manifest differs from V3.6")
    manifest_logical = {
        key: item
        for key, item in state_manifest.items()
        if key != "manifest_sha256"
    }
    if (
        state_manifest.get("schema_version") != V36_STATE_KEY_BANK_SCHEMA
        or state_manifest.get("status") != "completed_diagnostic"
        or state_manifest.get("manifest_sha256")
        != canonical_json_sha256(manifest_logical)
        or state_manifest.get("manifest_sha256")
        != artifact.get("manifest_logical_sha256")
        or state_manifest.get("memory_count") != report.get("reference_key_count")
    ):
        raise ValueError("V3.7 state-key bank authentication failed")
    return report, state_manifest


def _load_state_key_bank(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from memgen.model.retrieval_keys import tensor_sha256

    tensor_info = manifest.get("tensor_artifact", {})
    relative = Path(str(tensor_info.get("path", "")))
    tensor_path = (manifest_path.parent / relative).resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or manifest_path.parent.resolve() not in tensor_path.parents
        or not tensor_path.is_file()
        or file_sha256(tensor_path) != tensor_info.get("sha256")
    ):
        raise ValueError("V3.7 state-key tensor artifact is invalid")
    tensors = load_file(str(tensor_path), device="cpu")
    records = manifest.get("records", ())
    if not isinstance(records, list) or not records:
        raise ValueError("V3.7 state-key manifest has no records")
    memory_ids = tuple(str(entry.get("memory_id", "")) for entry in records)
    if (
        any(not memory_id for memory_id in memory_ids)
        or len(set(memory_ids)) != len(memory_ids)
        or tuple(int(entry.get("index", -1)) for entry in records)
        != tuple(range(len(records)))
        or canonical_json_sha256(list(memory_ids))
        != manifest.get("record_order_sha256")
    ):
        raise ValueError("V3.7 state-key record order is invalid")
    components: dict[str, Any] = {}
    for component in V37_STATE_COMPONENT_BY_VARIANT.values():
        vectors = []
        for entry in records:
            metadata = entry.get("state_components", {}).get(component, {})
            tensor_name = str(metadata.get("tensor_name", ""))
            if tensor_name not in tensors:
                raise ValueError(f"V3.7 state-key tensor is missing: {tensor_name}")
            vector = tensors[tensor_name].detach().double().reshape(-1).contiguous()
            if tensor_sha256(vector) != metadata.get("embedding_sha256"):
                # V3.6 hashes the saved tensor before this float64 view.
                original = tensors[tensor_name].detach().contiguous()
                if tensor_sha256(original) != metadata.get("embedding_sha256"):
                    raise ValueError(f"V3.7 state-key hash drifted: {tensor_name}")
            if not torch.isfinite(vector).all() or float(vector.norm().item()) <= 1e-12:
                raise ValueError(f"V3.7 state-key vector is invalid: {tensor_name}")
            vectors.append(vector)
        components[component] = F.normalize(
            torch.stack(vectors, dim=0), dim=-1
        ).contiguous()
    return memory_ids, components


def _bank_question_hashes(
    path: Path, memory_ids: Sequence[str]
) -> dict[str, str]:
    selected = set(memory_ids)
    values: dict[str, set[str]] = {memory_id: set() for memory_id in memory_ids}
    for row in iter_jsonl(path):
        memory_id = str(row.get("memory_id", ""))
        if memory_id in selected:
            digest = str(row.get("source_question_sha256", ""))
            if digest:
                values[memory_id].add(digest)
    invalid = {
        memory_id: sorted(digests)
        for memory_id, digests in values.items()
        if len(digests) != 1
    }
    if invalid:
        raise ValueError(f"V3.7 cannot authenticate bank question hashes: {invalid}")
    return {memory_id: next(iter(values[memory_id])) for memory_id in memory_ids}


def _normalize_vector(value: Any, *, owner: str) -> Any:
    import torch
    import torch.nn.functional as F

    vector = value.detach().float().reshape(-1)
    if not torch.isfinite(vector).all() or float(vector.norm().item()) <= 1e-12:
        raise ValueError(f"V3.7 {owner} is non-finite or degenerate")
    return F.normalize(vector.double(), dim=0).cpu().contiguous()


def _encode_query_variants(
    *,
    model: Any,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    device: str,
    layer_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if len(prefix_ids) <= len(prompt_ids):
        raise ValueError("V3.7 query prefix has no partial reasoning")
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    prefix = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        prompt_output = model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        prefix_output = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    prompt_hidden = prompt_output.hidden_states
    prefix_hidden = prefix_output.hidden_states
    if (
        prompt_hidden is None
        or prefix_hidden is None
        or len(prompt_hidden) <= layer_number
        or len(prefix_hidden) <= layer_number
    ):
        raise RuntimeError("V3.7 reasoner has no requested hidden-state layer")
    prompt_raw = prompt_hidden[layer_number][0, -1, :].detach().float()
    states = prefix_hidden[layer_number][0].detach().float()
    current_raw = states[-1]
    window_start = max(len(prompt_ids), len(prefix_ids) - 16)
    local_raw = states[window_start:].mean(dim=0)
    delta_raw = current_raw - prompt_raw
    vectors = {
        "current_token": _normalize_vector(current_raw, owner="current query"),
        "prompt_subtracted_delta": _normalize_vector(
            delta_raw, owner="delta query"
        ),
        "local_reasoning_window_16": _normalize_vector(
            local_raw, owner="local16 query"
        ),
    }
    geometry = {
        "layer_number": layer_number,
        "prompt_token_count": len(prompt_ids),
        "prefix_token_count": len(prefix_ids),
        "partial_cot_token_count": len(prefix_ids) - len(prompt_ids),
        "local_window_token_count": len(prefix_ids) - window_start,
        "raw_current_minus_prompt_norm": float(delta_raw.norm().item()),
        "prompt_current_cosine": float(
            torch.nn.functional.cosine_similarity(
                prompt_raw, current_raw, dim=0
            ).item()
        ),
        "current_local_cosine": float(
            torch.nn.functional.cosine_similarity(
                current_raw, local_raw, dim=0
            ).item()
        ),
    }
    return vectors, geometry


def _score_condition(
    *, tokenizer: Any, completion_ids: Sequence[int], ground_truth: str
) -> dict[str, Any]:
    ids = [int(value) for value in completion_ids]
    completion = tokenizer.decode(ids, skip_special_tokens=True).strip()
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    return {
        "completion": completion,
        "completion_token_ids": ids,
        "completion_token_ids_sha256": canonical_json_sha256(ids),
        "generated_token_count": len(ids),
        "strict_correct": verifier["reward"] == 1.0,
        "strict_reward": float(verifier["reward"]),
        "format_valid": bool(verifier["format_valid"]),
        "diagnostic_answer_correct": verifier.get("diagnostic_answer_correct"),
        "diagnostic_failure_types": list(verifier.get("failure_types", ())),
        "scorer_version": GSM8K_VERIFIER_VERSION,
    }


def _generate_continuous_observation(
    *, runtime: Any, prompt_token_ids: Sequence[int], gate: Any
) -> Any:
    """Run vanilla decoding while probing every pre-answer generated token.

    V3.5 freezes ``memgen/model/e1_runtime.py`` by content hash, so the V3.7
    every-token observation policy deliberately lives here instead of changing
    the older delimiter-only E1 method.  Before the first event, this is exactly
    the V3.4/V3.5 joint entropy+risk qualification policy; hysteresis/re-arm is
    irrelevant because only the first event is selected.
    """

    import torch
    from memgen.experience.e1 import GateObservation
    from memgen.model.e1_runtime import ObservationRolloutResult

    ids = list(prompt_token_ids)
    if not ids:
        raise ValueError("V3.7 continuous observation requires a prompt")
    prompt_length = len(ids)
    past = None
    selected: GateObservation | None = None
    selected_prefix: tuple[int, ...] = ()
    candidate_count = 0
    with torch.inference_mode():
        for generation_step in range(runtime.max_new_tokens):
            full = runtime._tensor(ids)
            attention_mask = torch.ones_like(full)
            probe = None
            completion_text = runtime.tokenizer.decode(
                ids[prompt_length:], skip_special_tokens=False
            )
            can_observe = selected is None and not _ANSWER_MARKER_RE.search(
                completion_text
            )
            if generation_step > 0 and can_observe:
                candidate_count += 1
                probe = gate.probe(
                    model=runtime.model,
                    boundary_token=full[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past,
                )
                if gate.triggered(probe):
                    selected = GateObservation(
                        generated_boundary_index=(
                            len(ids) - prompt_length - 1
                        ),
                        boundary_token_id=int(ids[-1]),
                        entropy=probe.entropy,
                        entropy_threshold=gate.config.entropy_threshold,
                        persistence_risk_score=probe.risk_score,
                        persistence_risk_threshold=gate.config.risk_threshold,
                    )
                    selected_prefix = tuple(ids)

            if probe is not None:
                output = probe.output
            else:
                kwargs: dict[str, Any] = {
                    "attention_mask": attention_mask,
                    "use_cache": True,
                    "return_dict": True,
                    "input_ids": full if past is None else full[:, -1:],
                }
                if past is not None:
                    kwargs["past_key_values"] = past
                output = runtime.model(**kwargs)
            next_token = runtime.decoding.next_token(
                token_ids=ids, logits=output.logits
            )
            ids.append(next_token)
            past = output.past_key_values
            if runtime.decoding.is_eos(next_token):
                break
    return ObservationRolloutResult(
        completion_token_ids=tuple(ids[prompt_length:]),
        gate_observation=selected,
        prefix_token_ids=selected_prefix,
        candidate_boundary_count=candidate_count,
    )


def _processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def _retrieval_for_query(
    *,
    memory_ids: Sequence[str],
    state_keys: Mapping[str, Any],
    applicability_keys: Any,
    query_vectors: Mapping[str, Any],
    candidate_top_k: int,
    random_count: int,
    rrf_rank_constant: int,
    seed: int,
    sample_id: str,
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, int]],
    dict[str, dict[str, float]],
]:
    score_by_variant: dict[str, dict[str, float]] = {}
    rank_by_variant: dict[str, dict[str, int]] = {}
    rankings: dict[str, tuple[str, ...]] = {}
    for variant, component in V37_STATE_COMPONENT_BY_VARIANT.items():
        scores = (
            query_vectors[component].reshape(1, -1)
            @ state_keys[component].T
        )[0]
        values = tuple(float(value) for value in scores.tolist())
        ordered, ranks = stable_rank(memory_ids, values)
        rankings[variant] = ordered
        rank_by_variant[variant] = ranks
        score_by_variant[variant] = dict(zip(memory_ids, values))
    text_scores = (
        query_vectors["current_token"].reshape(1, -1)
        @ applicability_keys.T
    )[0]
    text_values = tuple(float(value) for value in text_scores.tolist())
    ordered, ranks = stable_rank(memory_ids, text_values)
    rankings["text_applicability"] = ordered
    rank_by_variant["text_applicability"] = ranks
    score_by_variant["text_applicability"] = dict(
        zip(memory_ids, text_values)
    )
    fusion_values = reciprocal_rank_fusion_scores(
        memory_ids,
        rank_by_variant["state_local16"],
        rank_by_variant["state_delta"],
        rank_constant=rrf_rank_constant,
    )
    ordered, ranks = stable_rank(memory_ids, fusion_values)
    rankings[V37_FUSION_VARIANT] = ordered
    rank_by_variant[V37_FUSION_VARIANT] = ranks
    score_by_variant[V37_FUSION_VARIANT] = dict(
        zip(memory_ids, fusion_values)
    )
    if tuple(rankings) != V37_RETRIEVAL_VARIANTS:
        raise AssertionError("V3.7 retrieval variant order drifted")

    deterministic = {
        memory_id
        for ordered_values in rankings.values()
        for memory_id in ordered_values[:candidate_top_k]
    }
    random_universe = sorted(set(memory_ids) - deterministic)
    if len(random_universe) < random_count:
        raise ValueError("V3.7 has too few memories for independent random controls")
    rng = random.Random(f"{seed}:{sample_id}")
    random_ids = tuple(sorted(rng.sample(random_universe, random_count)))
    pool, sources = candidate_union(
        rankings,
        top_k=candidate_top_k,
        random_memory_ids=random_ids,
    )
    return pool, sources, rankings, rank_by_variant, score_by_variant


def _trace_summary(
    *,
    result: Any,
    expected_memory_id: str,
    expected_baseline_first_token: int,
) -> dict[str, Any]:
    traces = tuple(result.attention_traces)
    if not traces or any(
        str(trace.memory_id) != expected_memory_id for trace in traces
    ):
        raise RuntimeError("V3.7 treatment trace lost its memory identity")
    masses = [float(trace.memory_attention_mass) for trace in traces]
    if not all(math.isfinite(value) and value > 0.0 for value in masses):
        raise RuntimeError("V3.7 treatment has invalid memory attention")
    parity = int(result.baseline_first_token_id) == int(
        expected_baseline_first_token
    )
    if not parity:
        raise RuntimeError("V3.7 same-prefix baseline first-token parity failed")
    return {
        "trace_count": len(traces),
        "memory_id_constant": True,
        "native_key_lengths_sha256": canonical_json_sha256(
            [int(trace.native_key_length) for trace in traces]
        ),
        "memory_attention_mass_mean": sum(masses) / len(masses),
        "memory_attention_mass_minimum": min(masses),
        "memory_attention_mass_maximum": max(masses),
        "memory_attention_masses_sha256": canonical_json_sha256(masses),
        "baseline_first_token_id": int(result.baseline_first_token_id),
        "expected_baseline_first_token_id": int(expected_baseline_first_token),
        "baseline_first_token_parity": parity,
        "first_step_logits_kl": float(result.first_step_logits_kl),
        "first_step_top1_changed": bool(result.first_step_top1_changed),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# MemGen V3.7 Cross-Problem Causal Applicability Audit",
        "",
        f"- Status: `{report['status']}`",
        "- Qualified for online use: `false`",
        "- Same-question memory permitted: `false`",
        "- Task accuracy used for treatment labels: `true`",
        f"- Selected held-out queries: `{summary.get('selected_query_count', 0)}`",
        f"- Gate-eligible queries: `{summary.get('gate_eligible_query_count', 0)}`",
        "- Evaluated-pool oracle only: `true`",
        "",
        "| Variant | Top-1 accuracy | Uplift | Helpful@1 | Harmful@1 | Helpful hit@K |",
        "| ------- | -------------: | -----: | --------: | --------: | ------------: |",
    ]
    k = str(report["configuration"]["candidate_top_k"])
    for variant in V37_RETRIEVAL_VARIANTS:
        value = summary.get("variants", {}).get(variant, {})
        hit = value.get("helpful_hit_at_k", {}).get(k, {})
        lines.append(
            f"| {variant} | {float(value.get('top1_accuracy', 0.0)):.6f} "
            f"| {float(value.get('top1_accuracy_uplift', 0.0)):.6f} "
            f"| {float(value.get('top1_helpful_fraction', 0.0)):.6f} "
            f"| {float(value.get('top1_harmful_fraction', 0.0)):.6f} "
            f"| {float(hit.get('fraction_of_pool_helpful_queries', 0.0)):.6f} |"
        )
    lines.extend([
        "",
        f"- Gate-eligible baseline accuracy: `{summary.get('baseline_accuracy_gate_eligible', 0.0)}`",
        f"- Evaluated-pool oracle accuracy: `{summary.get('evaluated_pool_oracle_accuracy_gate_eligible', 0.0)}`",
        f"- Evaluated-pool oracle uplift: `{summary.get('evaluated_pool_oracle_uplift_gate_eligible', 0.0)}`",
        f"- Queries with at least one helpful evaluated memory: `{summary.get('evaluated_pool_any_helpful_query_count', 0)}`",
        "",
        "The oracle and helpful-recall quantities are conditional on the shared",
        "retriever-union plus random-control candidate pool. The full memory bank",
        "was not exhaustively treated. Answers and rewards were used only after",
        "candidate construction to label causal utility; they were never encoded",
        "into a query or key. No variant or online threshold is selected here.",
        "",
    ])
    return "\n".join(lines)


def _progress_report(
    *, profile_sha256: str, selected_count: int, query_rows: Sequence[Any], treatments: Sequence[Any]
) -> dict[str, Any]:
    return {
        "schema_version": V37_CAUSAL_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "running",
        "profile_sha256": profile_sha256,
        "selected_query_count": selected_count,
        "completed_query_count": len(query_rows),
        "completed_treatment_count": len(treatments),
        "qualified_for_online_use": False,
    }


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
    selected_all = [
        row for row in split["samples"] if row.get("logical_split") == "dev-test"
    ]
    selected = selected_all[args.offset:]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("V3.7 selected dev-test slice is empty")
    if {str(row.get("dataset_split")) for row in selected} != {"train"}:
        raise ValueError("V3.7 dev-test must come from GSM8K train")

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    side_manifest = json.loads(
        args.side_kv_manifest.read_text(encoding="utf-8")
    )
    validate_cross_bank_metadata(records=records, side_manifest=side_manifest)
    reasoner = side_manifest.get("reasoner", {})
    memory_ids, state_keys = _load_state_key_bank(
        args.state_key_manifest, state_manifest
    )
    if args.candidate_top_k > len(memory_ids):
        raise ValueError("V3.7 candidate top-k exceeds the state-key bank")
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    state_entries = {
        str(entry["memory_id"]): entry for entry in state_manifest["records"]
    }
    if not set(memory_ids).issubset(side_entries) or not set(memory_ids).issubset(
        {record.memory_id for record in records}
    ):
        raise ValueError("V3.7 state-key IDs are not bound to the side-KV bank")
    for memory_id in memory_ids:
        if (
            str(state_entries[memory_id].get("payload_hash"))
            != str(side_entries[memory_id].get("payload_hash"))
            or int(state_entries[memory_id].get("kv_valid_slot_count", -1))
            != int(side_entries[memory_id].get("kv_valid_slot_count", -2))
        ):
            raise ValueError(f"V3.7 state/value binding drifted: {memory_id}")
    bank_question_hash = _bank_question_hashes(
        args.source_alignment_evidence, memory_ids
    )
    bank_source_hashes = {
        str(row["question_sha256"])
        for row in split["samples"]
        if row.get("logical_split") == "bank-source"
    }
    if not set(bank_question_hash.values()).issubset(bank_source_hashes):
        raise ValueError("V3.7 memory question is outside bank-source")
    selected_question_hashes = {str(row["question_sha256"]) for row in selected}
    if selected_question_hashes & set(bank_question_hash.values()):
        raise ValueError("V3.7 found same-question memory leakage into dev-test")

    dual_bank = DualRetrievalKeyBankLoader(
        manifest_path=args.dual_key_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    if not set(memory_ids).issubset(dual_bank.entry_by_id):
        raise ValueError("V3.7 text control does not cover the state-key universe")
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
    gate = gate_wrapper.diagnostic_gate
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V3.7 risk and side-KV reasoner provenance differs")

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
        raise ValueError("V3.7 resolved model/tokenizer revision drifted")

    profile = ExperienceMemoryV3Profile.continuous_token_joint()
    side_loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    controller = SideKVAttentionController(
        model=model,
        layer_number=profile.layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=profile.memory_score_normalization,
        memory_score_bias=profile.memory_score_bias,
    )
    runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    implementation_paths = (
        "data/gsm8k/prompt.py",
        "data/utils/math_utils.py",
        "memgen/experience/v3_6_state_keys.py",
        "memgen/experience/v3_7_cross_problem.py",
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_5_retrieval.py",
        "memgen/model/v3_runtime.py",
        "scripts/audit_v3_7_cross_problem_causal_applicability.py",
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
        "state_key_tensor_sha256": str(
            state_manifest["tensor_artifact"]["sha256"]
        ),
    }
    profile_material = {
        "schema_version": V37_CAUSAL_PROFILE_SCHEMA,
        "git_revision": git_revision(),
        "inputs": input_hashes,
        "implementation_files_sha256": implementation_hashes,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_hashes
        ),
        "logical_split": "dev-test",
        "offset": args.offset,
        "limit": args.limit,
        "selected_sample_ids": [str(row["sample_id"]) for row in selected],
        "candidate_top_k": args.candidate_top_k,
        "random_candidates": args.random_candidates,
        "rrf_rank_constant": args.rrf_rank_constant,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "retrieval_variants": list(V37_RETRIEVAL_VARIANTS),
        "state_key_memory_count": len(memory_ids),
        "same_question_memory_policy": "strictly_excluded_and_fail_closed",
        "candidate_label_access": False,
        "treatment_label": "strict_reward_treatment_minus_baseline",
        "full_bank_treatment_policy": "shared_topk_union_plus_random_controls",
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "gate": gate_wrapper.config.to_dict(),
        "gate_observation_scope": "every_pre_answer_generated_token",
        "side_kv_profile": profile.to_dict(),
    }
    profile_sha256 = canonical_json_sha256(profile_material)
    run_profile = {
        **profile_material,
        "created_at": utc_now(),
        "profile_sha256": profile_sha256,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / PROFILE_FILE
    if profile_path.is_file():
        existing_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if existing_profile.get("profile_sha256") != profile_sha256:
            raise ValueError("V3.7 output directory belongs to a different profile")
    else:
        write_json_atomic(profile_path, run_profile)

    query_path = args.output_dir / QUERY_FILE
    treatment_path = args.output_dir / TREATMENT_FILE
    report_path = args.output_dir / REPORT_FILE
    markdown_path = args.output_dir / MARKDOWN_FILE
    query_rows = load_jsonl(query_path)
    treatment_rows = load_jsonl(treatment_path)
    completed_ids: set[str] = set()
    for row in query_rows:
        if (
            row.get("schema_version") != V37_CAUSAL_QUERY_SCHEMA
            or row.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("V3.7 query resume evidence differs from profile")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in completed_ids:
            raise ValueError("V3.7 query resume evidence is duplicated")
        completed_ids.add(sample_id)
    completed_pairs: set[tuple[str, str]] = set()
    treatment_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in treatment_rows:
        if (
            row.get("schema_version") != V37_CAUSAL_TREATMENT_SCHEMA
            or row.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("V3.7 treatment resume evidence differs from profile")
        pair = (str(row.get("sample_id", "")), str(row.get("memory_id", "")))
        if not all(pair) or pair in completed_pairs:
            raise ValueError("V3.7 treatment resume evidence is duplicated")
        completed_pairs.add(pair)
        treatment_by_pair[pair] = row
    selected_ids = {str(row["sample_id"]) for row in selected}
    if not completed_ids.issubset(selected_ids) or any(
        sample_id not in selected_ids for sample_id, _ in completed_pairs
    ):
        raise ValueError("V3.7 resume evidence is outside the selected slice")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=split["dataset"]["revision"],
    )
    write_json_atomic(
        report_path,
        _progress_report(
            profile_sha256=profile_sha256,
            selected_count=len(selected),
            query_rows=query_rows,
            treatments=treatment_rows,
        ),
    )

    try:
        with query_path.open("a", encoding="utf-8") as query_handle, treatment_path.open(
            "a", encoding="utf-8"
        ) as treatment_handle:
            for position, entry in enumerate(selected, start=1):
                sample_id = str(entry["sample_id"])
                if sample_id in completed_ids:
                    continue
                source = dataset[int(entry["source_index"])]
                question = str(source["question"]).strip()
                answer = str(source["answer"]).strip()
                question_hash = text_sha256(question)
                if (
                    question_hash != entry.get("question_sha256")
                    or text_sha256(answer) != entry.get("answer_sha256")
                    or question_hash in set(bank_question_hash.values())
                ):
                    raise ValueError(f"V3.7 dataset or cross-problem drift: {sample_id}")
                prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
                baseline_started = time.perf_counter()
                observation = _generate_continuous_observation(
                    runtime=runtime,
                    prompt_token_ids=prompt_ids,
                    gate=gate,
                )
                baseline_seconds = time.perf_counter() - baseline_started

                if observation.gate_observation is None:
                    ground_truth = _processed_solution(answer)
                    baseline = _score_condition(
                        tokenizer=tokenizer,
                        completion_ids=observation.completion_token_ids,
                        ground_truth=ground_truth,
                    )
                    baseline["runtime_seconds"] = baseline_seconds
                    query_row = {
                        "schema_version": V37_CAUSAL_QUERY_SCHEMA,
                        "profile_sha256": profile_sha256,
                        "sample_id": sample_id,
                        "logical_split": "dev-test",
                        "dataset_split": "train",
                        "source_index": int(entry["source_index"]),
                        "question_sha256": question_hash,
                        "answer_sha256": str(entry["answer_sha256"]),
                        "gate_eligible": False,
                        "gate_observation": None,
                        "candidate_memory_ids": [],
                        "same_question_candidate_count": 0,
                        "baseline": baseline,
                    }
                    append_jsonl(query_handle, query_row)
                    query_rows.append(query_row)
                    completed_ids.add(sample_id)
                    print(
                        f"[v3.7] {position}/{len(selected)} {sample_id} no-gate",
                        flush=True,
                    )
                    continue

                prefix_ids = tuple(int(value) for value in observation.prefix_token_ids)
                query_vectors, query_geometry = _encode_query_variants(
                    model=model,
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    device=args.device,
                    layer_number=profile.layer_number,
                )
                pool, candidate_sources, rankings, ranks, scores = (
                    _retrieval_for_query(
                        memory_ids=memory_ids,
                        state_keys=state_keys,
                        applicability_keys=applicability_keys,
                        query_vectors=query_vectors,
                        candidate_top_k=args.candidate_top_k,
                        random_count=args.random_candidates,
                        rrf_rank_constant=args.rrf_rank_constant,
                        seed=args.seed,
                        sample_id=sample_id,
                    )
                )
                same_question = [
                    memory_id
                    for memory_id in pool
                    if bank_question_hash[memory_id] == question_hash
                ]
                if same_question:
                    raise ValueError(
                        f"V3.7 candidate pool contains same-question memory: {same_question}"
                    )
                # Candidate construction is now complete.  Only from this
                # point onward may the task answer/reward be materialized.
                ground_truth = _processed_solution(answer)
                baseline = _score_condition(
                    tokenizer=tokenizer,
                    completion_ids=observation.completion_token_ids,
                    ground_truth=ground_truth,
                )
                baseline["runtime_seconds"] = baseline_seconds
                partial_length = len(prefix_ids) - len(prompt_ids)
                if partial_length >= len(observation.completion_token_ids):
                    raise ValueError("V3.7 gate prefix has no baseline next token")
                expected_baseline_first_token = int(
                    observation.completion_token_ids[partial_length]
                )

                for candidate_index, memory_id in enumerate(pool, start=1):
                    pair = (sample_id, memory_id)
                    if pair in completed_pairs:
                        stored = treatment_by_pair[pair]
                        if (
                            tuple(stored.get("candidate_sources", ()))
                            != candidate_sources[memory_id]
                            or stored.get("question_sha256") != question_hash
                            or stored.get("prefix_token_ids_sha256")
                            != canonical_json_sha256(list(prefix_ids))
                            or float(stored.get("baseline_reward", -1.0))
                            != float(baseline["strict_reward"])
                            or {
                                variant: int(ranks[variant][memory_id])
                                for variant in V37_RETRIEVAL_VARIANTS
                            }
                            != stored.get("rank_by_variant")
                            or any(
                                not math.isclose(
                                    float(stored.get("score_by_variant", {}).get(
                                        variant, float("nan")
                                    )),
                                    float(scores[variant][memory_id]),
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                                for variant in V37_RETRIEVAL_VARIANTS
                            )
                        ):
                            raise ValueError("V3.7 partial treatment resume drifted")
                        continue
                    memory = side_loader.get(
                        memory_id,
                        device=args.device,
                        dtype=next(model.parameters()).dtype,
                    )
                    treatment_started = time.perf_counter()
                    result = runtime.generate_from_trigger_with_persistent_memory(
                        prefix_token_ids=prefix_ids,
                        prompt_token_count=len(prompt_ids),
                        memory=memory,
                        controller=controller,
                    )
                    treatment_seconds = time.perf_counter() - treatment_started
                    treatment = _score_condition(
                        tokenizer=tokenizer,
                        completion_ids=result.completion_token_ids,
                        ground_truth=ground_truth,
                    )
                    treatment["runtime_seconds"] = treatment_seconds
                    utility = causal_utility(
                        baseline_reward=baseline["strict_reward"],
                        treatment_reward=treatment["strict_reward"],
                    )
                    trace = _trace_summary(
                        result=result,
                        expected_memory_id=memory_id,
                        expected_baseline_first_token=expected_baseline_first_token,
                    )
                    treatment_row = {
                        "schema_version": V37_CAUSAL_TREATMENT_SCHEMA,
                        "profile_sha256": profile_sha256,
                        "sample_id": sample_id,
                        "question_sha256": question_hash,
                        "memory_id": memory_id,
                        "memory_source_question_sha256": bank_question_hash[memory_id],
                        "cross_problem": True,
                        "same_question": False,
                        "candidate_sources": list(candidate_sources[memory_id]),
                        "rank_by_variant": {
                            variant: int(ranks[variant][memory_id])
                            for variant in V37_RETRIEVAL_VARIANTS
                        },
                        "score_by_variant": {
                            variant: float(scores[variant][memory_id])
                            for variant in V37_RETRIEVAL_VARIANTS
                        },
                        "baseline_reward": float(baseline["strict_reward"]),
                        "treatment_reward": float(treatment["strict_reward"]),
                        "causal_utility": utility,
                        "causal_label": (
                            "helpful" if utility == 1 else "harmful" if utility == -1 else "neutral"
                        ),
                        "prefix_token_ids_sha256": canonical_json_sha256(
                            list(prefix_ids)
                        ),
                        "payload_hash": memory.payload_hash,
                        "treatment": treatment,
                        "side_kv_trace": trace,
                    }
                    append_jsonl(treatment_handle, treatment_row)
                    treatment_rows.append(treatment_row)
                    completed_pairs.add(pair)
                    treatment_by_pair[pair] = treatment_row
                    print(
                        f"[v3.7] {position}/{len(selected)} {sample_id} "
                        f"treatment={candidate_index}/{len(pool)} memory={memory_id} "
                        f"utility={utility:+d}",
                        flush=True,
                    )

                query_row = {
                    "schema_version": V37_CAUSAL_QUERY_SCHEMA,
                    "profile_sha256": profile_sha256,
                    "sample_id": sample_id,
                    "logical_split": "dev-test",
                    "dataset_split": "train",
                    "source_index": int(entry["source_index"]),
                    "question_sha256": question_hash,
                    "answer_sha256": str(entry["answer_sha256"]),
                    "gate_eligible": True,
                    "gate_observation": observation.gate_observation.to_dict(),
                    "prefix_token_ids_sha256": canonical_json_sha256(
                        list(prefix_ids)
                    ),
                    "query_geometry": query_geometry,
                    "query_embedding_sha256": {
                        component: tensor_sha256(vector)
                        for component, vector in query_vectors.items()
                    },
                    "candidate_memory_ids": list(pool),
                    "candidate_count": len(pool),
                    "same_question_candidate_count": 0,
                    "candidate_top_memory_ids": {
                        variant: list(rankings[variant][: args.candidate_top_k])
                        for variant in V37_RETRIEVAL_VARIANTS
                    },
                    "baseline": baseline,
                }
                append_jsonl(query_handle, query_row)
                query_rows.append(query_row)
                completed_ids.add(sample_id)
                write_json_atomic(
                    report_path,
                    _progress_report(
                        profile_sha256=profile_sha256,
                        selected_count=len(selected),
                        query_rows=query_rows,
                        treatments=treatment_rows,
                    ),
                )
                print(
                    f"[v3.7] completed {position}/{len(selected)} {sample_id}",
                    flush=True,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        controller.close()

    if len(query_rows) != len(selected):
        raise RuntimeError("V3.7 did not complete every selected query")
    summary = summarize_causal_rows(
        query_rows=query_rows,
        treatment_rows=treatment_rows,
        candidate_top_k=args.candidate_top_k,
    )
    report: dict[str, Any] = {
        "schema_version": V37_CAUSAL_REPORT_SCHEMA,
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
            "after_candidate_pool_freeze_for_eligible_queries_plus_descriptive_"
            "no_gate_baseline_never_query_key_score_pool_or_threshold"
        ),
        "same_question_memory_permitted": False,
        "cross_problem_enforced": True,
        "variant_selected": False,
        "threshold_fitted": False,
        "profile_sha256": profile_sha256,
        "configuration": {
            "logical_split": "dev-test",
            "retrieval_variants": list(V37_RETRIEVAL_VARIANTS),
            "candidate_top_k": args.candidate_top_k,
            "random_candidates": args.random_candidates,
            "rrf_rank_constant": args.rrf_rank_constant,
            "state_key_memory_count": len(memory_ids),
            "state_key_trajectory": "reference_failure_first_gate",
            "runtime_query_trajectory": "heldout_vanilla_generation_first_gate",
            "key_query_context": "canonical_prompt_question_full_partial_cot",
            "treatment": "persistent_existing_full_when_facing_prefer_avoid_side_kv",
            "utility": "strict_reward_treatment_minus_same_prefix_baseline",
            "candidate_pool": "union_of_each_variant_topk_plus_disjoint_random_controls",
            "oracle_scope": "evaluated_candidate_pool_only",
            "full_bank_exhaustively_treated": False,
            "stable_tie_break": "memory_id_ascending",
            "layer_number": profile.layer_number,
            "local_window_size": 16,
        },
        "summary": summary,
        "interpretation_contract": {
            "no_helpful_memory_in_evaluated_pool": (
                "current_value_bank_has_no_detected_cross_problem_causal_ceiling_under_this_pool"
            ),
            "oracle_positive_retrievers_weak": (
                "helpful_cross_problem_memories_exist_but_retrieval_or_reranking_is_weak"
            ),
            "retriever_positive_top1_uplift": (
                "source_state_keys_predict_cross_problem_causal_applicability"
            ),
            "helpful_and_harmful_both_common": (
                "abstention_or_treatment_risk_control_is_required"
            ),
            "pool_oracle_is_not_full_bank_oracle": True,
            "diagnostic_does_not_select_online_variant": True,
        },
        "inputs": {
            **input_hashes,
            "git_revision": run_profile["git_revision"],
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(
                implementation_hashes
            ),
            "v36_report_logical_sha256": v36_report["report_sha256"],
            "state_key_manifest_logical_sha256": state_manifest[
                "manifest_sha256"
            ],
        },
        "artifacts": {
            "profile": {
                "path": PROFILE_FILE,
                "sha256": file_sha256(profile_path),
            },
            "queries": {
                "path": QUERY_FILE,
                "sha256": file_sha256(query_path),
                "row_count": len(query_rows),
            },
            "treatments": {
                "path": TREATMENT_FILE,
                "sha256": file_sha256(treatment_path),
                "row_count": len(treatment_rows),
            },
        },
        "requirements": {
            "heldout_dev_test_only": True,
            "bank_source_and_query_questions_disjoint": True,
            "same_question_memory_excluded": True,
            "state_key_value_binding_authenticated": True,
            "query_and_key_state_components_match": True,
            "candidate_pool_built_without_answer_or_reward": True,
            "strict_reward_never_used_for_candidate_construction": True,
            "baseline_and_treatment_share_exact_prefix": True,
            "baseline_first_token_parity_checked_for_every_treatment": True,
            "existing_side_kv_payload_preserved": True,
            "full_bank_oracle_not_claimed": True,
            "variant_not_selected": True,
            "threshold_not_fitted": True,
            "formal_v3_5_qualification_unchanged": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json_atomic(report_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(
        f"[v3.7] status={report['status']} queries={len(query_rows)} "
        f"treatments={len(treatment_rows)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
