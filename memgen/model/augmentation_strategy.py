"""Weaver 训练与评测使用的 latent 插入策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


FIRST_K = "first_k"
CANDIDATE_SINK_THRESHOLD = "candidate_sink_threshold"
SEQUENCE_SINK_THRESHOLD = "sequence_sink_threshold"
CANDIDATE_ENTROPY_THRESHOLD = "candidate_entropy_threshold"
SUPPORTED_STRATEGIES = frozenset(
    {
        FIRST_K,
        CANDIDATE_SINK_THRESHOLD,
        SEQUENCE_SINK_THRESHOLD,
        CANDIDATE_ENTROPY_THRESHOLD,
    }
)


def query_index_for_insertion_point(point_index: int) -> int:
    """把“在 token i 前插入”映射为负责该决策的 prefix query。"""

    if point_index <= 0:
        raise ValueError("inference insertion point must be greater than 0")
    return point_index - 1


@dataclass(frozen=True)
class InferenceInsertionPoint:
    """原始 token 序列中一个可能的 inference latent 插入点。"""

    index: int
    is_delimiter: bool
    sink_score: float | None = None


@dataclass(frozen=True)
class WeaverInsertionStrategyConfig:
    """从 YAML 或 checkpoint 读取的 Weaver 插入策略配置。"""

    name: str = FIRST_K
    sink_score_threshold: float = 0.0
    sink_score_layer_window: int = 4
    entropy_threshold: float = 0.0
    attention_sink_token_count: int = 4

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> "WeaverInsertionStrategyConfig":
        values = values or {}
        config = cls(
            name=str(values.get("name", FIRST_K)),
            sink_score_threshold=float(values.get("sink_score_threshold", 0.0)),
            sink_score_layer_window=int(values.get("sink_score_layer_window", 4)),
            entropy_threshold=float(values.get("entropy_threshold", 0.0)),
            attention_sink_token_count=int(values.get("attention_sink_token_count", 4)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.name not in SUPPORTED_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
            raise ValueError(
                f"Unsupported Weaver insertion strategy: {self.name!r}; "
                f"expected one of: {supported}"
            )
        if not math.isfinite(self.sink_score_threshold):
            raise ValueError("sink_score_threshold must be finite")
        if self.sink_score_layer_window < 0:
            raise ValueError("sink_score_layer_window must be >= 0")
        if not math.isfinite(self.entropy_threshold):
            raise ValueError("entropy_threshold must be finite")
        if self.attention_sink_token_count < 0:
            raise ValueError("attention_sink_token_count must be >= 0")

    @property
    def requires_sink_scores(self) -> bool:
        return self.name in {CANDIDATE_SINK_THRESHOLD, SEQUENCE_SINK_THRESHOLD}

    @property
    def requires_attention_scores(self) -> bool:
        """Whether runtime needs reasoner attention tensors rather than FlashAttention."""

        return self.name in {
            CANDIDATE_SINK_THRESHOLD,
            SEQUENCE_SINK_THRESHOLD,
            CANDIDATE_ENTROPY_THRESHOLD,
        }

    @property
    def is_entropy_gate(self) -> bool:
        return self.name == CANDIDATE_ENTROPY_THRESHOLD

    @property
    def requires_delimiter_candidates(self) -> bool:
        return self.name != SEQUENCE_SINK_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sink_score_threshold": self.sink_score_threshold,
            "sink_score_layer_window": self.sink_score_layer_window,
            "entropy_threshold": self.entropy_threshold,
            "attention_sink_token_count": self.attention_sink_token_count,
        }


class WeaverInsertionStrategy:
    """统一策略入口；输出按因果位置升序排列的 inference 插点。"""

    def __init__(self, config: WeaverInsertionStrategyConfig):
        self.config = config

    @property
    def requires_sink_scores(self) -> bool:
        return self.config.requires_sink_scores

    @property
    def requires_attention_scores(self) -> bool:
        return self.config.requires_attention_scores

    @property
    def is_entropy_gate(self) -> bool:
        return self.config.is_entropy_gate

    @property
    def requires_delimiter_candidates(self) -> bool:
        return self.config.requires_delimiter_candidates

    def select(
        self,
        points: Sequence[InferenceInsertionPoint],
        max_num: int,
    ) -> list[int]:
        if max_num < 0:
            raise ValueError("max_inference_aug_num must be >= 0")
        if max_num == 0:
            return []

        if self.config.name == FIRST_K:
            candidates = [point for point in points if point.is_delimiter]
            return [point.index for point in candidates[:max_num]]

        if self.config.is_entropy_gate:
            raise ValueError(
                "candidate_entropy_threshold is an online evaluation gate and "
                "cannot select teacher-forced Weaver training positions."
            )

        if self.config.name == CANDIDATE_SINK_THRESHOLD:
            candidates = [point for point in points if point.is_delimiter]
        else:
            candidates = list(points)

        threshold = self.config.sink_score_threshold
        above_threshold = [
            point
            for point in candidates
            if point.sink_score is not None
            and math.isfinite(point.sink_score)
            and point.sink_score > threshold
        ]
        strongest = sorted(
            above_threshold,
            key=lambda point: (-float(point.sink_score), point.index),
        )[:max_num]
        return sorted(point.index for point in strongest)


def build_weaver_insertion_strategy(
    values: Mapping[str, object] | None,
) -> WeaverInsertionStrategy:
    return WeaverInsertionStrategy(WeaverInsertionStrategyConfig.from_mapping(values))
