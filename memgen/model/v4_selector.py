"""One-stage state-to-bank selector for MemGen V4.

The benchmark namespace is fixed by the run profile, so V4 has no task-family
selector.  At a positive gate event this module compares one normalized local
reasoning-state query with every qualified repair bank's positive and
hard-negative state anchors.  It returns exactly one target bank or abstains.

No learned parameters, task rewards, target K/V values, or reference-memory
values participate in selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from memgen.experience.phase1 import canonical_json_sha256, file_sha256


V4_SELECTOR_CONFIG_SCHEMA = "memgen-v4-selector-config-v1"
V4_SELECTOR_DECISION_SCHEMA = "memgen-v4-selector-decision-v1"
V4_SELECTOR_ANCHOR_SCHEMA = "memgen-v4-selector-anchor-bank-v1"
V4_SELECTOR_QUERY_VARIANT = "layer24_local_reasoning_window_mean_16"
V4_SELECTOR_WINDOW = 16


def v4_selector_implementation_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        "memgen/experience/v3_5_source_alignment.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "memgen/model/v4_selector.py",
        "scripts/compile_v4_selector_anchors.py",
    )
    return {relative: file_sha256(project_root / relative) for relative in paths}


def _finite(owner: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{owner} must be finite")
    return converted


def v4_tensor_sha256(value: torch.Tensor) -> str:
    """Hash an anchor tensor in a stable CPU float32 representation."""

    normalized = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(canonical_json_sha256(list(normalized.shape)).encode("ascii"))
    digest.update(normalized.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class V4SelectorConfig:
    """Frozen selector thresholds produced by offline calibration."""

    absolute_threshold: float
    margin_threshold: float
    evidence_temperature: float = 1.0
    negative_weight: float = 1.0
    layer_number: int = 24
    query_variant: str = V4_SELECTOR_QUERY_VARIANT
    query_window: int = V4_SELECTOR_WINDOW
    score_rule: str = "positive_logmeanexp_minus_negative_logmeanexp"
    schema_version: str = V4_SELECTOR_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_SELECTOR_CONFIG_SCHEMA:
            raise ValueError("Unexpected V4 selector-config schema")
        _finite("absolute_threshold", self.absolute_threshold)
        if _finite("margin_threshold", self.margin_threshold) < 0:
            raise ValueError("V4 selector margin threshold must be non-negative")
        if _finite("evidence_temperature", self.evidence_temperature) <= 0:
            raise ValueError("V4 selector evidence temperature must be positive")
        if self.negative_weight != 1.0:
            raise ValueError("V4 initial selector fixes negative evidence weight at one")
        if self.layer_number != 24:
            raise ValueError("V4 initial selector is frozen at layer 24")
        if self.query_variant != V4_SELECTOR_QUERY_VARIANT:
            raise ValueError("V4 initial selector is frozen to local reasoning window")
        if self.query_window != V4_SELECTOR_WINDOW:
            raise ValueError("V4 initial selector fixes the local window at sixteen")
        if self.score_rule != "positive_logmeanexp_minus_negative_logmeanexp":
            raise ValueError("Unexpected V4 selector score rule")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4AnchorBank:
    """State-state routing keys for one qualified target process bank."""

    bank_id: str
    positive_keys: torch.Tensor
    negative_keys: torch.Tensor
    positive_anchor_ids: tuple[str, ...]
    negative_anchor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bank_id, str) or not self.bank_id:
            raise ValueError("V4 anchor bank requires a bank_id")
        if self.positive_keys.ndim != 2 or self.negative_keys.ndim != 2:
            raise ValueError("V4 anchor keys must use [anchor,hidden] tensors")
        if self.positive_keys.shape[0] <= 0 or self.negative_keys.shape[0] <= 0:
            raise ValueError("V4 anchor bank requires positive and negative anchors")
        if self.positive_keys.shape[1] != self.negative_keys.shape[1]:
            raise ValueError("V4 positive/negative anchor dimensions differ")
        if self.positive_keys.shape[0] != len(self.positive_anchor_ids):
            raise ValueError("V4 positive anchor IDs differ from tensor rows")
        if self.negative_keys.shape[0] != len(self.negative_anchor_ids):
            raise ValueError("V4 negative anchor IDs differ from tensor rows")
        if len(set(self.positive_anchor_ids)) != len(self.positive_anchor_ids):
            raise ValueError("V4 positive anchor IDs contain duplicates")
        if len(set(self.negative_anchor_ids)) != len(self.negative_anchor_ids):
            raise ValueError("V4 negative anchor IDs contain duplicates")
        if set(self.positive_anchor_ids) & set(self.negative_anchor_ids):
            raise ValueError("V4 positive and negative anchors overlap")
        for owner, keys in (
            ("positive", self.positive_keys),
            ("negative", self.negative_keys),
        ):
            if not torch.isfinite(keys.float()).all():
                raise ValueError(f"V4 {owner} anchors contain non-finite values")
            norms = torch.linalg.vector_norm(keys.float(), dim=-1)
            if not torch.allclose(
                norms,
                torch.ones_like(norms),
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(f"V4 {owner} anchors must be L2 normalized")

    @property
    def hidden_size(self) -> int:
        return int(self.positive_keys.shape[1])


@dataclass(frozen=True)
class V4CompiledAnchorArtifact:
    """Authenticated selector anchors plus frozen abstention thresholds."""

    banks: tuple[V4AnchorBank, ...]
    config: V4SelectorConfig
    anchor_metadata: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.banks) < 2:
            raise ValueError("V4 anchor artifact requires at least two qualified banks")
        bank_ids = tuple(bank.bank_id for bank in self.banks)
        if len(set(bank_ids)) != len(bank_ids):
            raise ValueError("V4 anchor artifact bank IDs are duplicated")
        if tuple(sorted(bank_ids)) != bank_ids:
            raise ValueError("V4 anchor artifact banks must be sorted by bank ID")
        if set(self.anchor_metadata) != set(bank_ids):
            raise ValueError("V4 anchor metadata coverage differs from tensor banks")
        if self.provenance.get(
            "implementation_sha256"
        ) != v4_selector_implementation_hashes():
            raise ValueError("V4 selector implementation identity drifted")
        calibration = self.provenance.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("qualified") is not True
            or calibration.get("max_success_false_selection_rate") != 0.05
            or calibration.get("max_failure_wrong_routing_rate") != 0.05
            or set(calibration.get("per_bank", {})) != set(bank_ids)
        ):
            raise ValueError("V4 anchor artifact calibration did not qualify")
        calibrated = calibration.get("selected", {})
        if (
            not math.isclose(
                float(calibrated.get("absolute_threshold")),
                self.config.absolute_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(calibrated.get("margin_threshold")),
                self.config.margin_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("V4 selector config differs from calibration")
        for bank in self.banks:
            metadata = self.anchor_metadata[bank.bank_id]
            positive = metadata.get("positive")
            negative = metadata.get("negative")
            if not isinstance(positive, Sequence) or not isinstance(negative, Sequence):
                raise ValueError("V4 anchor metadata requires positive and negative rows")
            if len(positive) != len(bank.positive_anchor_ids):
                raise ValueError("V4 positive metadata count differs from tensor rows")
            if len(negative) != len(bank.negative_anchor_ids):
                raise ValueError("V4 negative metadata count differs from tensor rows")
            if tuple(str(item.get("anchor_id")) for item in positive) != bank.positive_anchor_ids:
                raise ValueError("V4 positive metadata identity differs from tensor rows")
            if tuple(str(item.get("anchor_id")) for item in negative) != bank.negative_anchor_ids:
                raise ValueError("V4 negative metadata identity differs from tensor rows")

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "v4_selector_anchors.safetensors"
        manifest_path = output_dir / "v4_selector_anchor_manifest.json"
        tensors: dict[str, torch.Tensor] = {}
        records: list[dict[str, Any]] = []
        for index, bank in enumerate(self.banks):
            positive_name = f"bank_{index:04d}_positive"
            negative_name = f"bank_{index:04d}_negative"
            positive = bank.positive_keys.detach().float().cpu().contiguous()
            negative = bank.negative_keys.detach().float().cpu().contiguous()
            tensors[positive_name] = positive
            tensors[negative_name] = negative
            records.append(
                {
                    "index": index,
                    "bank_id": bank.bank_id,
                    "positive_tensor_name": positive_name,
                    "negative_tensor_name": negative_name,
                    "positive_count": int(positive.shape[0]),
                    "negative_count": int(negative.shape[0]),
                    "hidden_size": int(positive.shape[1]),
                    "positive_sha256": v4_tensor_sha256(positive),
                    "negative_sha256": v4_tensor_sha256(negative),
                    "positive_anchor_ids": list(bank.positive_anchor_ids),
                    "negative_anchor_ids": list(bank.negative_anchor_ids),
                    "anchor_metadata": {
                        "positive": [
                            dict(item)
                            for item in self.anchor_metadata[bank.bank_id]["positive"]
                        ],
                        "negative": [
                            dict(item)
                            for item in self.anchor_metadata[bank.bank_id]["negative"]
                        ],
                    },
                }
            )
        save_file(
            tensors,
            str(tensor_path),
            metadata={
                "schema_version": V4_SELECTOR_ANCHOR_SCHEMA,
                "query_variant": V4_SELECTOR_QUERY_VARIANT,
            },
        )
        manifest: dict[str, Any] = {
            "schema_version": V4_SELECTOR_ANCHOR_SCHEMA,
            "status": "selector_anchor_compilation_passed",
            "qualified_for_online_use": True,
            "selector_structure": "one_stage_state_to_bank",
            "query_variant": V4_SELECTOR_QUERY_VARIANT,
            "layer_number": 24,
            "hidden_state_tuple_index": 24,
            "local_reasoning_window": 16,
            "normalization": "mean_then_l2",
            "positive_anchor_source": "member_verified_failure_first_joint_gate",
            "negative_anchor_sources": [
                "all_matched_verified_success_aligned_state",
                "other_bank_verified_failure_first_joint_gate",
            ],
            "calibration_policy": "leave_one_problem_out_source_state_calibration",
            "config": self.config.to_dict(),
            "bank_count": len(self.banks),
            "bank_ids": [bank.bank_id for bank in self.banks],
            "record_order_sha256": canonical_json_sha256(
                [bank.bank_id for bank in self.banks]
            ),
            "records": records,
            "provenance": dict(self.provenance),
            "tensor_artifact": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
                "tensor_count": len(tensors),
                "tensor_set_sha256": canonical_json_sha256(
                    {
                        name: v4_tensor_sha256(value)
                        for name, value in sorted(tensors.items())
                    }
                ),
            },
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return tensor_path, manifest_path


class V4SelectorAnchorBankLoader:
    """Authenticate a V4 anchor artifact and construct the one-stage selector."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_bank_ids: Sequence[str] | None = None,
        expected_reasoner_name: str | None = None,
        expected_reasoner_revision: str | None = None,
        expected_tokenizer_revision: str | None = None,
    ) -> None:
        from safetensors.torch import load_file

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_bank_ids=expected_bank_ids,
            expected_reasoner_name=expected_reasoner_name,
            expected_reasoner_revision=expected_reasoner_revision,
            expected_tokenizer_revision=expected_tokenizer_revision,
        )
        artifact = self.manifest["tensor_artifact"]
        tensor_path = manifest_path.parent / str(artifact["path"])
        if not tensor_path.is_file() or file_sha256(tensor_path) != artifact["sha256"]:
            raise ValueError("V4 selector anchor tensor artifact is missing or corrupted")
        tensors = load_file(str(tensor_path), device="cpu")
        tensor_hashes = {
            name: v4_tensor_sha256(value) for name, value in sorted(tensors.items())
        }
        if (
            len(tensors) != artifact["tensor_count"]
            or canonical_json_sha256(tensor_hashes) != artifact["tensor_set_sha256"]
        ):
            raise ValueError("V4 selector anchor tensor set drifted")
        banks: list[V4AnchorBank] = []
        for record in self.manifest["records"]:
            positive = tensors[str(record["positive_tensor_name"])]
            negative = tensors[str(record["negative_tensor_name"])]
            if (
                v4_tensor_sha256(positive) != record["positive_sha256"]
                or v4_tensor_sha256(negative) != record["negative_sha256"]
                or list(positive.shape)
                != [record["positive_count"], record["hidden_size"]]
                or list(negative.shape)
                != [record["negative_count"], record["hidden_size"]]
            ):
                raise ValueError("V4 selector anchor record tensor hash drifted")
            banks.append(
                V4AnchorBank(
                    bank_id=str(record["bank_id"]),
                    positive_keys=positive,
                    negative_keys=negative,
                    positive_anchor_ids=tuple(record["positive_anchor_ids"]),
                    negative_anchor_ids=tuple(record["negative_anchor_ids"]),
                )
            )
        self.banks = tuple(banks)
        self.config = V4SelectorConfig(**self.manifest["config"])
        self.selector = V4RepairSelector(banks=self.banks, config=self.config)

    def _validate_manifest(
        self,
        *,
        expected_bank_ids: Sequence[str] | None,
        expected_reasoner_name: str | None,
        expected_reasoner_revision: str | None,
        expected_tokenizer_revision: str | None,
    ) -> None:
        value = self.manifest
        if value.get("schema_version") != V4_SELECTOR_ANCHOR_SCHEMA:
            raise ValueError("Unexpected V4 selector anchor schema")
        logical = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if value.get("manifest_sha256") != canonical_json_sha256(logical):
            raise ValueError("V4 selector anchor manifest hash mismatch")
        if (
            value.get("status") != "selector_anchor_compilation_passed"
            or value.get("qualified_for_online_use") is not True
            or value.get("selector_structure") != "one_stage_state_to_bank"
            or value.get("query_variant") != V4_SELECTOR_QUERY_VARIANT
            or value.get("layer_number") != 24
            or value.get("hidden_state_tuple_index") != 24
            or value.get("local_reasoning_window") != 16
            or value.get("normalization") != "mean_then_l2"
        ):
            raise ValueError("V4 selector anchor runtime contract drifted")
        records = value.get("records")
        bank_ids = value.get("bank_ids")
        if (
            not isinstance(records, list)
            or len(records) < 2
            or value.get("bank_count") != len(records)
            or bank_ids != [record.get("bank_id") for record in records]
            or bank_ids != sorted(bank_ids)
            or value.get("record_order_sha256") != canonical_json_sha256(bank_ids)
        ):
            raise ValueError("V4 selector anchor record namespace is invalid")
        if expected_bank_ids is not None and tuple(bank_ids) != tuple(expected_bank_ids):
            raise ValueError("V4 selector and side-KV bank namespaces differ")
        for index, record in enumerate(records):
            if record.get("index") != index:
                raise ValueError("V4 selector anchor record indices are not contiguous")
            if record.get("positive_count") != len(record.get("positive_anchor_ids", [])):
                raise ValueError("V4 selector positive count mismatch")
            if record.get("positive_count", 0) < 5:
                raise ValueError("V4 selector bank has fewer than five positive anchors")
            if record.get("negative_count") != len(record.get("negative_anchor_ids", [])):
                raise ValueError("V4 selector negative count mismatch")
            if record.get("negative_count", 0) <= 0:
                raise ValueError("V4 selector bank has no negative anchors")
            metadata = record.get("anchor_metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("V4 selector anchor metadata is missing")
            for role in ("positive", "negative"):
                rows = metadata.get(role)
                ids = record.get(f"{role}_anchor_ids", [])
                if (
                    not isinstance(rows, list)
                    or [row.get("anchor_id") for row in rows] != ids
                    or any(
                        not isinstance(row.get("experience_id"), str)
                        or not row.get("experience_id")
                        or not isinstance(row.get("sample_id"), str)
                        or not row.get("sample_id")
                        or not isinstance(row.get("source_kind"), str)
                        or not row.get("source_kind")
                        for row in rows
                    )
                ):
                    raise ValueError(f"V4 selector {role} anchor metadata drifted")
            if len(
                {
                    row["sample_id"]
                    for row in metadata["positive"]
                }
            ) < 5:
                raise ValueError(
                    "V4 selector bank has fewer than five distinct positive problems"
                )
        reasoner = value.get("provenance", {}).get("reasoner", {})
        if value.get("provenance", {}).get(
            "implementation_sha256"
        ) != v4_selector_implementation_hashes():
            raise ValueError("V4 selector implementation identity drifted")
        calibration = value.get("provenance", {}).get("calibration", {})
        if (
            calibration.get("qualified") is not True
            or calibration.get("max_success_false_selection_rate") != 0.05
            or calibration.get("max_failure_wrong_routing_rate") != 0.05
            or set(calibration.get("per_bank", {})) != set(bank_ids)
            or any(
                item.get("correct_selected_count", 0) <= 0
                for item in calibration.get("per_bank", {}).values()
            )
        ):
            raise ValueError("V4 selector anchor calibration did not qualify")
        config = value.get("config", {})
        calibrated = calibration.get("selected", {})
        if (
            config.get("absolute_threshold")
            != calibrated.get("absolute_threshold")
            or config.get("margin_threshold")
            != calibrated.get("margin_threshold")
        ):
            raise ValueError("V4 selector manifest thresholds differ from calibration")
        for expected, field in (
            (expected_reasoner_name, "model_name"),
            (expected_reasoner_revision, "model_revision"),
            (expected_tokenizer_revision, "tokenizer_revision"),
        ):
            if expected is not None and reasoner.get(field) != expected:
                raise ValueError(f"V4 selector reasoner {field} mismatch")


@dataclass(frozen=True)
class V4BankScore:
    bank_id: str
    positive_evidence: float
    negative_evidence: float
    score: float
    positive_anchor_count: int
    negative_anchor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4SelectionDecision:
    selected_bank_id: str | None
    outcome: str
    top1_bank_id: str
    top1_score: float
    top2_bank_id: str
    top2_score: float
    margin: float
    absolute_threshold: float
    margin_threshold: float
    ranked_scores: tuple[V4BankScore, ...]
    schema_version: str = V4_SELECTOR_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_SELECTOR_DECISION_SCHEMA:
            raise ValueError("Unexpected V4 selector-decision schema")
        if self.outcome not in {
            "selected",
            "abstained_absolute_threshold",
            "abstained_margin_threshold",
        }:
            raise ValueError("Unexpected V4 selector outcome")
        if (self.outcome == "selected") != (self.selected_bank_id is not None):
            raise ValueError("V4 selector outcome and selected bank disagree")
        if self.selected_bank_id is not None and self.selected_bank_id != self.top1_bank_id:
            raise ValueError("V4 selector did not select its top-ranked bank")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ranked_scores": [item.to_dict() for item in self.ranked_scores],
        }


def pool_v4_local_reasoning_query(
    hidden_states: torch.Tensor,
    *,
    reasoning_start_index: int,
    valid_token_mask: torch.Tensor | None = None,
    window: int = V4_SELECTOR_WINDOW,
) -> torch.Tensor:
    """Mean-pool the last valid reasoning states and return one normalized key."""

    if hidden_states.ndim != 2:
        raise ValueError("V4 selector hidden_states must use [token,hidden]")
    token_count = int(hidden_states.shape[0])
    if reasoning_start_index < 0 or reasoning_start_index >= token_count:
        raise ValueError("V4 reasoning_start_index is outside the hidden-state sequence")
    if window != V4_SELECTOR_WINDOW:
        raise ValueError("V4 initial local reasoning window is frozen at sixteen")
    if valid_token_mask is None:
        valid_token_mask = torch.ones(token_count, dtype=torch.bool, device=hidden_states.device)
    if valid_token_mask.shape != (token_count,) or valid_token_mask.dtype != torch.bool:
        raise ValueError("V4 valid_token_mask must be one boolean value per token")
    indices = torch.arange(token_count, device=hidden_states.device)
    selected = indices[(indices >= reasoning_start_index) & valid_token_mask]
    if selected.numel() == 0:
        raise ValueError("V4 selector query has no valid reasoning tokens")
    selected = selected[-window:]
    pooled = hidden_states.index_select(0, selected).float().mean(dim=0)
    if not torch.isfinite(pooled).all() or float(torch.linalg.vector_norm(pooled).item()) == 0.0:
        raise ValueError("V4 selector query is zero or non-finite")
    return F.normalize(pooled, dim=0)


def _logmeanexp(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("V4 evidence requires a non-empty score vector")
    return torch.logsumexp(values, dim=0) - math.log(int(values.numel()))


class V4RepairSelector:
    """Exact one-stage selector over all banks in one benchmark namespace."""

    def __init__(
        self,
        *,
        banks: Sequence[V4AnchorBank],
        config: V4SelectorConfig,
    ) -> None:
        if len(banks) < 2:
            raise ValueError("V4 online selection requires at least two qualified banks")
        bank_ids = [item.bank_id for item in banks]
        if len(set(bank_ids)) != len(bank_ids):
            raise ValueError("V4 selector bank IDs contain duplicates")
        hidden_sizes = {item.hidden_size for item in banks}
        if len(hidden_sizes) != 1:
            raise ValueError("V4 selector anchor dimensions differ across banks")
        self.banks = tuple(sorted(banks, key=lambda item: item.bank_id))
        self.config = config
        self.hidden_size = next(iter(hidden_sizes))

    def _score_bank(self, query: torch.Tensor, bank: V4AnchorBank) -> V4BankScore:
        device = query.device
        query_float = query.float()
        positive = bank.positive_keys.to(device=device, dtype=torch.float32)
        negative = bank.negative_keys.to(device=device, dtype=torch.float32)
        positive_scores = positive @ query_float / self.config.evidence_temperature
        negative_scores = negative @ query_float / self.config.evidence_temperature
        positive_evidence = _logmeanexp(positive_scores)
        negative_evidence = _logmeanexp(negative_scores)
        score = positive_evidence - self.config.negative_weight * negative_evidence
        return V4BankScore(
            bank_id=bank.bank_id,
            positive_evidence=float(positive_evidence.item()),
            negative_evidence=float(negative_evidence.item()),
            score=float(score.item()),
            positive_anchor_count=int(positive.shape[0]),
            negative_anchor_count=int(negative.shape[0]),
        )

    def select(self, query: torch.Tensor) -> V4SelectionDecision:
        if query.ndim != 1 or int(query.shape[0]) != self.hidden_size:
            raise ValueError("V4 selector query dimension differs from anchor bank")
        if not torch.isfinite(query.float()).all():
            raise ValueError("V4 selector query contains non-finite values")
        norm = torch.linalg.vector_norm(query.float())
        if float(norm.item()) == 0.0:
            raise ValueError("V4 selector query is zero")
        normalized = F.normalize(query.float(), dim=0)
        scores = tuple(self._score_bank(normalized, bank) for bank in self.banks)
        ranked = tuple(sorted(scores, key=lambda item: (-item.score, item.bank_id)))
        top1, top2 = ranked[:2]
        margin = top1.score - top2.score
        if top1.score < self.config.absolute_threshold:
            selected_bank_id = None
            outcome = "abstained_absolute_threshold"
        elif margin < self.config.margin_threshold:
            selected_bank_id = None
            outcome = "abstained_margin_threshold"
        else:
            selected_bank_id = top1.bank_id
            outcome = "selected"
        return V4SelectionDecision(
            selected_bank_id=selected_bank_id,
            outcome=outcome,
            top1_bank_id=top1.bank_id,
            top1_score=top1.score,
            top2_bank_id=top2.bank_id,
            top2_score=top2.score,
            margin=margin,
            absolute_threshold=self.config.absolute_threshold,
            margin_threshold=self.config.margin_threshold,
            ranked_scores=ranked,
        )


__all__ = [
    "V4AnchorBank",
    "V4BankScore",
    "V4CompiledAnchorArtifact",
    "V4RepairSelector",
    "V4SelectionDecision",
    "V4SelectorAnchorBankLoader",
    "V4SelectorConfig",
    "pool_v4_local_reasoning_query",
    "v4_tensor_sha256",
    "v4_selector_implementation_hashes",
]
