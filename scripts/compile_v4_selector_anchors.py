#!/usr/bin/env python3
"""Compile V4 source-state anchors and calibrate the one-stage selector.

For each construction pair, the positive state is the verified-failure
trajectory's first counterfactual V3.4 joint-gate event.  The matched-success
negative is taken at the same normalized reasoning progress.  Both use the
mean of the latest up-to-sixteen layer-24 reasoning states followed by L2
normalization.

Each bank receives its own failure anchors as positives.  Its negatives are
all matched-success anchors plus failure anchors belonging to other banks.
Absolute and margin abstention thresholds are selected with deterministic
leave-one-problem-out calibration over bank-source states only; evaluation
answers and target side-KV values never enter routing.
"""

from __future__ import annotations

import argparse
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
from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v4_bank import V4_BANK_RECORD_SCHEMA
from scripts.build_v4_repair_bank import (
    _validate_split_manifest,
    load_v4_experiences,
)


MAX_UNSAFE_RATE = 0.05
LOCAL_WINDOW = 16


@dataclass(frozen=True)
class TrajectoryTokens:
    ids: tuple[int, ...]
    reasoning_indices: tuple[int, ...]


@dataclass(frozen=True)
class SourceAnchor:
    bank_id: str
    experience_id: str
    sample_id: str
    failure_vector: Any
    success_vector: Any
    failure_reasoning_rank: int
    success_reasoning_rank: int
    failure_prefix_sha256: str
    success_prefix_sha256: str
    failure_entropy: float
    failure_risk: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bank-records", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument(
        "--max-unsafe-rate",
        type=float,
        default=MAX_UNSAFE_RATE,
        help="Frozen upper bound for success false-selection and failure misrouting.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    from memgen.experience.phase1 import iter_jsonl

    records = [dict(item) for item in iter_jsonl(path)]
    if not records:
        raise ValueError("V4 selector anchor compilation received no bank records")
    for record in records:
        if record.get("schema_version") != V4_BANK_RECORD_SCHEMA:
            raise ValueError("Unexpected V4 bank record schema")
        logical = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != canonical_json_sha256(logical):
            raise ValueError("V4 selector input bank-record hash mismatch")
    return records


def _tokenize_trajectory(
    tokenizer: Any, prompt_ids: Sequence[int], completion: str
) -> TrajectoryTokens:
    completion_ids = tuple(
        int(value) for value in tokenizer.encode(completion, add_special_tokens=False)
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
    return TrajectoryTokens(
        ids=ids,
        reasoning_indices=tuple(range(len(prompt_ids), upper)),
    )


def _pad_pair(
    *, tokenizer: Any, target: TrajectoryTokens, reference: TrajectoryTokens, device: str
) -> tuple[Any, Any]:
    import torch

    if tokenizer.pad_token_id is None:
        raise ValueError("V4 selector tokenizer has no pad token")
    rows = (target.ids, reference.ids)
    width = max(len(row) for row in rows)
    input_ids = [
        list(row) + [int(tokenizer.pad_token_id)] * (width - len(row))
        for row in rows
    ]
    masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _risk_scores(states: Any, *, recovery: Any, persistence: Any) -> Any:
    import torch.nn.functional as F

    states = F.normalize(states.detach().float(), dim=-1)
    recovery = F.normalize(recovery.detach().float().to(states.device), dim=0)
    persistence = F.normalize(persistence.detach().float().to(states.device), dim=0)
    return states @ persistence - states @ recovery


def _local_vector(states: Any, *, reasoning_rank: int) -> Any:
    import torch
    import torch.nn.functional as F

    start = max(0, reasoning_rank + 1 - LOCAL_WINDOW)
    vector = states[start : reasoning_rank + 1].float().mean(dim=0)
    if not torch.isfinite(vector).all() or float(vector.norm().item()) <= 0.0:
        raise ValueError("V4 selector local reasoning vector is invalid")
    return F.normalize(vector, dim=0).detach().cpu().contiguous()


def _normalized_progress_rank(
    source_rank: int, *, source_count: int, target_count: int
) -> int:
    if source_count <= 0 or target_count <= 0:
        raise ValueError("V4 aligned progress requires non-empty trajectories")
    if source_count == 1 or target_count == 1:
        return 0
    progress = source_rank / (source_count - 1)
    return min(target_count - 1, max(0, int(round(progress * (target_count - 1)))))


def _logmeanexp(values: Any) -> Any:
    import torch

    return torch.logsumexp(values, dim=0) - math.log(int(values.numel()))


def _score(
    query: Any,
    *,
    positive: Sequence[Any],
    negative: Sequence[Any],
) -> float:
    import torch

    if not positive or not negative:
        raise ValueError("V4 LOO calibration produced an empty evidence side")
    positives = torch.stack(list(positive)).float()
    negatives = torch.stack(list(negative)).float()
    return float((_logmeanexp(positives @ query) - _logmeanexp(negatives @ query)).item())


def _rank_query(
    *,
    query: Any,
    query_kind: str,
    query_experience_id: str,
    query_bank_id: str,
    anchors_by_bank: Mapping[str, Sequence[SourceAnchor]],
) -> dict[str, Any]:
    all_anchors = [
        anchor for bank_id in sorted(anchors_by_bank) for anchor in anchors_by_bank[bank_id]
    ]
    scores: list[tuple[str, float]] = []
    for bank_id in sorted(anchors_by_bank):
        positives = [
            anchor.failure_vector
            for anchor in anchors_by_bank[bank_id]
            if not (
                query_kind == "failure"
                and anchor.experience_id == query_experience_id
            )
        ]
        negatives = [
            anchor.success_vector
            for anchor in all_anchors
            if not (
                query_kind == "success"
                and anchor.experience_id == query_experience_id
            )
        ]
        negatives.extend(
            anchor.failure_vector
            for origin_bank, origin_anchors in anchors_by_bank.items()
            if origin_bank != bank_id
            for anchor in origin_anchors
            if not (
                query_kind == "failure"
                and anchor.experience_id == query_experience_id
            )
        )
        scores.append((bank_id, _score(query, positive=positives, negative=negatives)))
    ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
    top1, top2 = ranked[:2]
    return {
        "query_kind": query_kind,
        "experience_id": query_experience_id,
        "source_bank_id": query_bank_id,
        "top1_bank_id": top1[0],
        "top1_score": top1[1],
        "top2_bank_id": top2[0],
        "top2_score": top2[1],
        "margin": top1[1] - top2[1],
        "correct_top1": top1[0] == query_bank_id if query_kind == "failure" else None,
        "ranked_scores": [
            {"bank_id": bank_id, "score": score} for bank_id, score in ranked
        ],
    }


def _candidate_thresholds(values: Sequence[float], *, floor_zero: bool) -> list[float]:
    if not values:
        raise ValueError("V4 threshold calibration has no observed values")
    candidates = set(float(value) for value in values)
    if floor_zero:
        candidates.add(0.0)
        candidates = {max(0.0, value) for value in candidates}
    candidates.add(math.nextafter(max(candidates), math.inf))
    return sorted(candidates)


def _calibrate_thresholds(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    failure_rows = [row for row in rows if row["query_kind"] == "failure"]
    success_rows = [row for row in rows if row["query_kind"] == "success"]
    if not failure_rows or not success_rows:
        raise ValueError("V4 threshold calibration requires failure and success states")
    absolute_candidates = _candidate_thresholds(
        [float(row["top1_score"]) for row in rows], floor_zero=False
    )
    margin_candidates = _candidate_thresholds(
        [float(row["margin"]) for row in rows], floor_zero=True
    )
    candidates: list[dict[str, Any]] = []
    for absolute in absolute_candidates:
        for margin in margin_candidates:
            selected = lambda row: (
                float(row["top1_score"]) >= absolute
                and float(row["margin"]) >= margin
            )
            success_false = sum(selected(row) for row in success_rows)
            failure_wrong = sum(
                selected(row) and row["correct_top1"] is not True
                for row in failure_rows
            )
            failure_correct = sum(
                selected(row) and row["correct_top1"] is True
                for row in failure_rows
            )
            success_false_rate = success_false / len(success_rows)
            failure_wrong_rate = failure_wrong / len(failure_rows)
            if (
                success_false_rate <= MAX_UNSAFE_RATE + 1e-12
                and failure_wrong_rate <= MAX_UNSAFE_RATE + 1e-12
            ):
                candidates.append(
                    {
                        "absolute_threshold": absolute,
                        "margin_threshold": margin,
                        "failure_correct_selected": failure_correct,
                        "failure_correct_coverage": failure_correct / len(failure_rows),
                        "failure_wrong_selected": failure_wrong,
                        "failure_wrong_rate": failure_wrong_rate,
                        "success_false_selected": success_false,
                        "success_false_rate": success_false_rate,
                    }
                )
    if not candidates:
        raise RuntimeError("V4 selector threshold grid has no safe candidate")
    best = sorted(
        candidates,
        key=lambda item: (
            -item["failure_correct_selected"],
            item["failure_wrong_selected"] + item["success_false_selected"],
            item["absolute_threshold"],
            item["margin_threshold"],
        ),
    )[0]
    per_bank: dict[str, dict[str, Any]] = {}
    for bank_id in sorted({str(row["source_bank_id"]) for row in failure_rows}):
        bank_rows = [
            row for row in failure_rows if str(row["source_bank_id"]) == bank_id
        ]
        correct_selected = sum(
            row["correct_top1"] is True
            and float(row["top1_score"]) >= best["absolute_threshold"]
            and float(row["margin"]) >= best["margin_threshold"]
            for row in bank_rows
        )
        per_bank[bank_id] = {
            "failure_query_count": len(bank_rows),
            "correct_selected_count": correct_selected,
            "correct_selected_coverage": correct_selected / len(bank_rows),
        }
    qualified = all(
        value["correct_selected_count"] > 0 for value in per_bank.values()
    )
    report = {
        "schema_version": "memgen-v4-selector-threshold-calibration-v1",
        "policy": "leave_one_problem_out_maximize_correct_coverage_under_dual_unsafe_caps",
        "max_success_false_selection_rate": MAX_UNSAFE_RATE,
        "max_failure_wrong_routing_rate": MAX_UNSAFE_RATE,
        "failure_query_count": len(failure_rows),
        "success_query_count": len(success_rows),
        "candidate_pair_count": len(absolute_candidates) * len(margin_candidates),
        "safe_candidate_count": len(candidates),
        "selected": best,
        "per_bank": per_bank,
        "qualification_rule": "at_least_one_loo_correct_selection_per_bank",
        "qualified": qualified,
    }
    return (
        float(best["absolute_threshold"]),
        float(best["margin_threshold"]),
        report,
    )


def _model_context_limit(model: Any) -> int | None:
    values = [
        int(value)
        for value in (
            getattr(model.config, "max_position_embeddings", None),
            getattr(model.config, "n_positions", None),
            getattr(model.config, "max_sequence_length", None),
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    return min(values) if values else None


def _construction_bank_membership(
    *,
    records: Sequence[Mapping[str, Any]],
    experience_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Authenticate construction support without imposing an order convention.

    V4/V4.1/semantic records canonicalize ``sample_ids`` by sorting, whereas
    V4.2 local-direct records preserve evidence order.  The selector only
    needs exact set membership; positional pairing is recovered directly from
    each authenticated experience ID.
    """

    construction_bank_by_experience: dict[str, str] = {}
    for record in records:
        bank_id = str(record["bank_id"])
        construction_ids = [
            str(value) for value in record["construction"]["experience_ids"]
        ]
        cluster_ids = [
            str(value) for value in record["cluster"]["member_experience_ids"]
        ]
        if construction_ids != cluster_ids:
            raise ValueError("V4 selector construction and cluster memberships differ")
        missing_ids = [
            experience_id
            for experience_id in construction_ids
            if experience_id not in experience_by_id
        ]
        if missing_ids:
            raise ValueError("V4 selector lost a construction experience")
        recorded_sample_ids = record["construction"].get("sample_ids")
        if (
            not isinstance(recorded_sample_ids, list)
            or len(set(recorded_sample_ids)) != len(recorded_sample_ids)
        ):
            raise ValueError("V4 selector construction sample support drifted")
        expected_sample_ids = [
            str(experience_by_id[experience_id]["sample_id"])
            for experience_id in construction_ids
        ]
        if sorted(expected_sample_ids) != sorted(str(item) for item in recorded_sample_ids):
            raise ValueError("V4 selector construction sample support drifted")
        for experience_id in construction_ids:
            if experience_id in construction_bank_by_experience:
                raise ValueError("A V4 construction experience belongs to multiple banks")
            construction_bank_by_experience[experience_id] = bank_id
    return construction_bank_by_experience


def main() -> None:
    args = parse_args()
    if not math.isclose(args.max_unsafe_rate, MAX_UNSAFE_RATE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("V4 initial selector calibration freezes max unsafe rate at 0.05")
    for path in (
        args.experiences,
        args.split_manifest,
        args.bank_records,
        args.bank_manifest,
        args.side_kv_manifest,
        args.token_risk_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.experience.v3_5_source_alignment import (
        CounterfactualGateObservation,
        counterfactual_attempts,
    )
    from memgen.model.side_kv import SDPAAttentionEntropyObserver
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_selector import (
        V4AnchorBank,
        V4CompiledAnchorArtifact,
        V4SelectorAnchorBankLoader,
        V4SelectorConfig,
        v4_selector_implementation_hashes,
    )
    from memgen.model.v4_side_kv import (
        V4SideKVBankLoader,
        validate_v4_tensor_free_manifest,
    )

    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    experience_by_id = {str(item["experience_id"]): item for item in experiences}
    records = _load_records(args.bank_records)
    construction_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    validate_v4_tensor_free_manifest(construction_manifest)
    if [record["bank_id"] for record in records] != construction_manifest["bank_ids"]:
        raise ValueError("V4 selector records differ from construction manifest order")
    if any(
        construction_manifest.get("record_sha256", {}).get(record["bank_id"])
        != record["record_sha256"]
        for record in records
    ):
        raise ValueError("V4 selector records are not bound by the construction manifest")
    if construction_manifest["inputs"].get("experiences_sha256") != file_sha256(args.experiences):
        raise ValueError("V4 selector experiences differ from bank construction")
    if construction_manifest["inputs"].get("split_manifest_sha256") != file_sha256(args.split_manifest):
        raise ValueError("V4 selector split manifest differs from bank construction")
    construction_bank_by_experience = _construction_bank_membership(
        records=records,
        experience_by_id=experience_by_id,
    )

    side_loader = V4SideKVBankLoader(manifest_path=args.side_kv_manifest)
    side_ids = side_loader.bank_ids
    if tuple(construction_manifest["bank_ids"]) != side_ids:
        raise ValueError("V4 selector construction and side-KV namespaces differ")
    side_source = side_loader.manifest.get("source", {})
    if (
        side_source.get("bank_manifest_logical_sha256")
        != construction_manifest["manifest_sha256"]
        or side_source.get("bank_manifest_file_sha256")
        != file_sha256(args.bank_manifest)
    ):
        raise ValueError("V4 selector side-KV source binding drifted")

    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    gate = EntropyHysteresisGate.from_token_artifact(risk_artifact)
    if gate.config.layer_number != 24 or gate.config.risk_role != "online_joint_control":
        raise ValueError("V4 selector anchors require the qualified layer-24 joint gate")
    reasoner = side_loader.manifest["reasoner"]
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V4 selector gate and side-KV reasoner provenance differ")

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "right"
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
        raise ValueError("V4 selector reasoner/tokenizer revision drifted")

    context_limit = args.max_sequence_length or _model_context_limit(model)

    gate_construction = risk_artifact["construction"]
    gate_risk = risk_artifact["risk_gate"]
    recovery_center = gate_risk["recovery_center"]
    persistence_center = gate_risk["persistence_center"]
    anchors_by_bank: dict[str, list[SourceAnchor]] = {
        bank_id: [] for bank_id in side_ids
    }
    skipped: list[dict[str, Any]] = []
    observer = SDPAAttentionEntropyObserver(
        model=model,
        sink_token_count=int(gate_construction["sink_token_count"]),
    )
    try:
        ordered_ids = sorted(construction_bank_by_experience)
        for index, experience_id in enumerate(ordered_ids, start=1):
            experience = experience_by_id[experience_id]
            bank_id = construction_bank_by_experience[experience_id]
            prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(
                tokenizer, str(experience["context"])
            )
            success = _tokenize_trajectory(
                tokenizer, prompt_ids, str(experience["trajectory"])
            )
            failure = _tokenize_trajectory(
                tokenizer, prompt_ids, str(experience["reference_trajectory"])
            )
            if not success.reasoning_indices or not failure.reasoning_indices:
                skipped.append({"experience_id": experience_id, "reason": "no_pre_answer_tokens"})
                continue
            if context_limit and max(len(success.ids), len(failure.ids)) > context_limit:
                skipped.append({"experience_id": experience_id, "reason": "context_limit"})
                continue
            input_ids, attention_mask = _pad_pair(
                tokenizer=tokenizer,
                target=success,
                reference=failure,
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
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) <= 24:
                raise RuntimeError("V4 selector model has no layer-24 output states")
            layer_states = hidden_states[24]
            failure_positions = list(failure.reasoning_indices)
            failure_states = layer_states[1, failure_positions, :].detach().float()
            entropies = observer.observation.entropy_by_query[
                1, failure_positions
            ].detach().float().cpu()
            risks = _risk_scores(
                failure_states,
                recovery=recovery_center,
                persistence=persistence_center,
            ).detach().float().cpu()
            attempts = counterfactual_attempts(
                [
                    CounterfactualGateObservation(
                        reasoning_rank=rank,
                        attention_entropy=float(entropies[rank].item()),
                        risk_score=float(risks[rank].item()),
                    )
                    for rank in range(len(failure_positions))
                ],
                high_entropy_threshold=float(gate_construction["high_entropy_threshold"]),
                low_entropy_threshold=float(gate_construction["low_entropy_threshold"]),
                risk_threshold=float(gate_risk["threshold"]),
                rearm_low_token_count=2,
                maximum_attempts=3,
            )
            if not attempts:
                skipped.append({"experience_id": experience_id, "reason": "failure_has_no_joint_gate"})
                continue
            failure_rank = int(attempts[0]["reasoning_rank"])
            success_rank = _normalized_progress_rank(
                failure_rank,
                source_count=len(failure.reasoning_indices),
                target_count=len(success.reasoning_indices),
            )
            success_states = layer_states[
                0, list(success.reasoning_indices), :
            ].detach().float()
            failure_token_index = failure.reasoning_indices[failure_rank]
            success_token_index = success.reasoning_indices[success_rank]
            anchor = SourceAnchor(
                bank_id=bank_id,
                experience_id=experience_id,
                sample_id=str(experience["sample_id"]),
                failure_vector=_local_vector(failure_states, reasoning_rank=failure_rank),
                success_vector=_local_vector(success_states, reasoning_rank=success_rank),
                failure_reasoning_rank=failure_rank,
                success_reasoning_rank=success_rank,
                failure_prefix_sha256=canonical_json_sha256(
                    list(failure.ids[: failure_token_index + 1])
                ),
                success_prefix_sha256=canonical_json_sha256(
                    list(success.ids[: success_token_index + 1])
                ),
                failure_entropy=float(attempts[0]["attention_entropy"]),
                failure_risk=float(attempts[0]["risk_score"]),
            )
            anchors_by_bank[bank_id].append(anchor)
            print(
                f"[v4-selector-anchor] {index}/{len(ordered_ids)} {experience_id} "
                f"bank={bank_id}",
                flush=True,
            )
    finally:
        observer.close()

    qualified = {
        bank_id: sorted(values, key=lambda item: item.experience_id)
        for bank_id, values in anchors_by_bank.items()
        if len({item.sample_id for item in values}) >= 5
    }
    rejected_banks = [
        {
            "bank_id": bank_id,
            "reason": "fewer_than_five_distinct_failure_gate_anchors",
            "anchor_count": len(values),
            "distinct_sample_count": len({item.sample_id for item in values}),
        }
        for bank_id, values in sorted(anchors_by_bank.items())
        if bank_id not in qualified
    ]
    if len(qualified) < 2:
        raise RuntimeError(
            "V4 online selector requires at least two banks with five failure-gate anchors"
        )

    calibration_rows: list[dict[str, Any]] = []
    for bank_id in sorted(qualified):
        for anchor in qualified[bank_id]:
            calibration_rows.append(
                _rank_query(
                    query=anchor.failure_vector,
                    query_kind="failure",
                    query_experience_id=anchor.experience_id,
                    query_bank_id=bank_id,
                    anchors_by_bank=qualified,
                )
            )
            calibration_rows.append(
                _rank_query(
                    query=anchor.success_vector,
                    query_kind="success",
                    query_experience_id=anchor.experience_id,
                    query_bank_id=bank_id,
                    anchors_by_bank=qualified,
                )
            )
    absolute, margin, calibration_report = _calibrate_thresholds(calibration_rows)
    if calibration_report["qualified"] is not True:
        raise RuntimeError(
            "V4 selector calibration has no correct leave-one-out selection for at least one bank"
        )
    selector_config = V4SelectorConfig(
        absolute_threshold=absolute,
        margin_threshold=margin,
    )

    all_qualified = [
        anchor
        for bank_id in sorted(qualified)
        for anchor in qualified[bank_id]
    ]
    banks: list[V4AnchorBank] = []
    metadata: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for bank_id in sorted(qualified):
        positives = qualified[bank_id]
        positive_ids = tuple(
            f"{bank_id}::failure::{anchor.experience_id}" for anchor in positives
        )
        negative_rows: list[tuple[str, Any, dict[str, Any]]] = []
        for anchor in all_qualified:
            anchor_id = f"{bank_id}::success::{anchor.bank_id}::{anchor.experience_id}"
            negative_rows.append(
                (
                    anchor_id,
                    anchor.success_vector,
                    {
                        "anchor_id": anchor_id,
                        "source_kind": "matched_verified_success_aligned_state",
                        "origin_bank_id": anchor.bank_id,
                        "experience_id": anchor.experience_id,
                        "sample_id": anchor.sample_id,
                        "trajectory_side": "target",
                        "reasoning_rank": anchor.success_reasoning_rank,
                        "full_prefix_token_ids_sha256": anchor.success_prefix_sha256,
                    },
                )
            )
        for origin_bank in sorted(qualified):
            if origin_bank == bank_id:
                continue
            for anchor in qualified[origin_bank]:
                anchor_id = f"{bank_id}::other-failure::{origin_bank}::{anchor.experience_id}"
                negative_rows.append(
                    (
                        anchor_id,
                        anchor.failure_vector,
                        {
                            "anchor_id": anchor_id,
                            "source_kind": "other_bank_verified_failure_first_joint_gate",
                            "origin_bank_id": origin_bank,
                            "experience_id": anchor.experience_id,
                            "sample_id": anchor.sample_id,
                            "trajectory_side": "reference",
                            "reasoning_rank": anchor.failure_reasoning_rank,
                            "full_prefix_token_ids_sha256": anchor.failure_prefix_sha256,
                        },
                    )
                )
        positive_metadata = [
            {
                "anchor_id": anchor_id,
                "source_kind": "member_verified_failure_first_joint_gate",
                "origin_bank_id": bank_id,
                "experience_id": anchor.experience_id,
                "sample_id": anchor.sample_id,
                "trajectory_side": "reference",
                "reasoning_rank": anchor.failure_reasoning_rank,
                "full_prefix_token_ids_sha256": anchor.failure_prefix_sha256,
                "attention_entropy": anchor.failure_entropy,
                "persistence_risk": anchor.failure_risk,
            }
            for anchor_id, anchor in zip(positive_ids, positives)
        ]
        banks.append(
            V4AnchorBank(
                bank_id=bank_id,
                positive_keys=torch.stack([item.failure_vector for item in positives]),
                negative_keys=torch.stack([item[1] for item in negative_rows]),
                positive_anchor_ids=positive_ids,
                negative_anchor_ids=tuple(item[0] for item in negative_rows),
            )
        )
        metadata[bank_id] = {
            "positive": positive_metadata,
            "negative": [item[2] for item in negative_rows],
        }

    provenance = {
        "compiler_git_revision": _git_revision(),
        "implementation_sha256": v4_selector_implementation_hashes(),
        "reasoner": dict(reasoner),
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "inputs": {
            "experiences_path": str(args.experiences.resolve()),
            "experiences_sha256": file_sha256(args.experiences),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "bank_records_path": str(args.bank_records.resolve()),
            "bank_records_sha256": file_sha256(args.bank_records),
            "bank_manifest_path": str(args.bank_manifest.resolve()),
            "bank_manifest_sha256": file_sha256(args.bank_manifest),
            "bank_manifest_logical_sha256": construction_manifest["manifest_sha256"],
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "side_kv_manifest_logical_sha256": side_loader.manifest["manifest_sha256"],
            "token_risk_artifact_path": str(args.token_risk_artifact.resolve()),
            "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        },
        "gate": {
            "high_entropy_threshold": gate.config.high_entropy_threshold,
            "low_entropy_threshold": gate.config.low_entropy_threshold,
            "risk_threshold": gate.config.risk_threshold,
            "sink_token_count": gate.config.sink_token_count,
        },
        "source_side_kv_bank_ids": list(side_ids),
        "qualified_bank_ids": sorted(qualified),
        "skipped_experience_count": len(skipped),
        "rejected_bank_count": len(rejected_banks),
        "calibration": calibration_report,
    }
    artifact = V4CompiledAnchorArtifact(
        banks=tuple(banks),
        config=selector_config,
        anchor_metadata=metadata,
        provenance=provenance,
    )
    output_dir = args.output_dir.expanduser().resolve()
    tensor_path, anchor_manifest_path = artifact.save(output_dir)
    reloaded = V4SelectorAnchorBankLoader(
        manifest_path=anchor_manifest_path,
        expected_bank_ids=sorted(qualified),
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    if tuple(bank.bank_id for bank in reloaded.banks) != tuple(sorted(qualified)):
        raise ValueError("V4 selector anchor reload coverage drifted")

    _write_jsonl(output_dir / "v4_selector_calibration_rows.jsonl", calibration_rows)
    _write_json(output_dir / "v4_selector_calibration_report.json", calibration_report)
    _write_json(output_dir / "v4_selector_skipped_experiences.json", skipped)
    _write_json(output_dir / "v4_selector_rejected_banks.json", rejected_banks)
    report = {
        "schema_version": "memgen-v4-selector-anchor-compile-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "selector_anchor_compilation_passed",
        "qualified_for_online_use": True,
        "source_bank_count": len(side_ids),
        "qualified_bank_count": len(qualified),
        "qualified_bank_ids": sorted(qualified),
        "source_anchor_count": len(all_qualified),
        "skipped_experience_count": len(skipped),
        "rejected_bank_count": len(rejected_banks),
        "selector_config": selector_config.to_dict(),
        "calibration": calibration_report,
        "artifacts": {
            "anchor_tensor": {"path": tensor_path.name, "sha256": file_sha256(tensor_path)},
            "anchor_manifest": {
                "path": anchor_manifest_path.name,
                "sha256": file_sha256(anchor_manifest_path),
                "logical_sha256": reloaded.manifest["manifest_sha256"],
            },
            "calibration_rows": {
                "path": "v4_selector_calibration_rows.jsonl",
                "sha256": file_sha256(output_dir / "v4_selector_calibration_rows.jsonl"),
            },
        },
    }
    report_path = output_dir / "v4_selector_anchor_compile_report.json"
    _write_json(report_path, report)
    print(
        "[v4-selector-anchor] complete "
        f"banks={len(qualified)} anchors={len(all_qualified)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-selector-anchor] error: {exc}", file=sys.stderr)
        raise
