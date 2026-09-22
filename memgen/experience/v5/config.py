"""Validated V5 research and runtime configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from memgen.experience.bank_construction.config import ModelConfig


@dataclass(frozen=True)
class V5Config:
    reasoner: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen2.5-1.5B-Instruct"))
    teacher: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen3-32B", device="auto"))
    reranker: ModelConfig = field(default_factory=lambda: ModelConfig(
        "Qwen/Qwen3-Reranker-8B", revision="5fa94080caafeaa45a15d11f969d7978e087a3db"))
    task: str = "gsm8k"
    dataset: str = "openai/gsm8k"
    dataset_revision: str = "main"
    val_ratio: float = .1
    split_seed: int = 42
    sampling_seed: int = 42
    greedy_rollouts: int = 1
    sampled_rollouts: int = 7
    temperature: float = .8
    top_p: float = .95
    top_k: int = 0
    max_new_tokens: int = 1024
    rollout_batch_size: int = 32
    teacher_backend: str = "vllm"
    teacher_base_url: str = "http://127.0.0.1:8000/v1"
    teacher_concurrency: int = 16
    teacher_timeout_seconds: int = 600
    teacher_max_new_tokens: int = 8192
    teacher_temperature: float = .2
    teacher_top_p: float = .8
    teacher_top_k: int = 20
    teacher_retries: int = 2
    contrast_limit_per_input: int = 3
    atom_partition_batch_size: int = 12
    group_pair_batch_size: int = 16
    group_candidate_top_k: int = 32
    card_leaf_size: int = 16
    minimum_primary_inputs: int = 3
    selector_top_k: int = 5
    reranker_enabled: bool = True
    reranker_max_length: int = 8192
    utility_ridge: float = 1.0
    utility_folds: int = 5
    train_limit: int = 0
    valid_limit: int = 0

    def __post_init__(self):
        if (self.greedy_rollouts, self.sampled_rollouts, self.max_new_tokens) != (1, 7, 1024):
            raise ValueError("V5 rollout contract requires 1 greedy + 7 sampled and 1024 tokens")
        if (self.temperature, self.top_p, self.top_k) != (.8, .95, 0):
            raise ValueError("V5 sampling contract is temperature=.8, top_p=.95, top_k=0")
        if self.task not in {"gsm8k"}:
            raise ValueError("No registered task protocol: " + self.task)
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between zero and one")
        positive = ("rollout_batch_size", "teacher_concurrency", "teacher_timeout_seconds",
                    "teacher_max_new_tokens", "contrast_limit_per_input", "atom_partition_batch_size",
                    "group_pair_batch_size", "group_candidate_top_k", "card_leaf_size",
                    "minimum_primary_inputs", "selector_top_k", "reranker_max_length",
                    "utility_folds")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in positive):
            raise ValueError("V5 positive integer configuration is invalid")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in ("split_seed", "sampling_seed", "teacher_retries", "train_limit", "valid_limit")):
            raise ValueError("V5 nonnegative integer configuration is invalid")
        if self.minimum_primary_inputs < 3:
            raise ValueError("V5 Primary Banks require at least three distinct train inputs")
        if not 0 < self.utility_ridge or not 0 < self.teacher_temperature <= 2:
            raise ValueError("Invalid V5 model/calibration setting")
        if self.teacher_backend not in {"transformers", "vllm"}:
            raise ValueError("teacher_backend must be transformers or vllm")
        from urllib.parse import urlsplit
        endpoint = urlsplit(self.teacher_base_url)
        if (endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
                or endpoint.path.rstrip("/") != "/v1" or endpoint.username or endpoint.password):
            raise ValueError("Teacher endpoint must be a local HTTP /v1 endpoint")
        for model in (self.reasoner, self.teacher, self.reranker):
            if not model.source or not model.revision or model.dtype not in {"bfloat16", "float32"}:
                raise ValueError("Invalid V5 model identity")
        if self.reasoner.device == "auto" or self.reranker.device == "auto":
            raise ValueError("Reasoner and reranker require explicit devices")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: Path):
        raw = json.loads(Path(path).read_text())
        for name in ("reasoner", "teacher", "reranker"):
            if name in raw:
                raw[name] = ModelConfig(**raw[name])
        return cls(**raw)
