"""Validated, versioned configuration; sampling is an explicit research contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    source: str
    revision: str = "main"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention_backend: str = "sdpa"


@dataclass(frozen=True)
class ConstructionConfig:
    reasoner: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen2.5-1.5B-Instruct"))
    teacher: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen3-32B", device="auto"))
    dataset: str = "openai/gsm8k"
    dataset_revision: str = "main"
    val_ratio: float = 0.1
    split_seed: int = 42
    sampling_seed: int = 42
    greedy_rollouts: int = 1
    sampled_rollouts: int = 7
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 1024
    rollout_batch_size: int = 32
    teacher_backend: str = "transformers"
    teacher_base_url: str = "http://127.0.0.1:8000/v1"
    teacher_concurrency: int = 16
    teacher_timeout_seconds: int = 600
    teacher_max_new_tokens: int = 8192
    teacher_temperature: float = 0.7
    teacher_top_p: float = 0.8
    teacher_top_k: int = 20
    teacher_retries: int = 2
    group_batch_size: int = 12
    candidate_batch_size: int = 16
    group_candidate_top_k: int = 64
    group_consolidation_rounds: int = 3
    train_limit: int = 0
    valid_limit: int = 0
    retrieval_key: str = "problem_structure"
    evaluation_mode: str = "selector_only"

    def __post_init__(self):
        if (self.greedy_rollouts, self.sampled_rollouts, self.max_new_tokens) != (1, 7, 1024):
            raise ValueError("Construction contract requires 1 greedy + 7 sampled, 1024 tokens")
        if (self.temperature, self.top_p, self.top_k) != (0.8, 0.95, 0):
            raise ValueError("Construction sampling contract is temperature=.8, top_p=.95, top_k=0")
        if not 0 < self.val_ratio < 1 or self.retrieval_key not in {"problem_structure", "applicability", "full_card"}:
            raise ValueError("Invalid split ratio or retrieval key")
        for name in ("split_seed", "sampling_seed", "teacher_retries", "train_limit", "valid_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("teacher_max_new_tokens", "group_batch_size", "candidate_batch_size",
                     "group_candidate_top_k", "group_consolidation_rounds",
                     "rollout_batch_size", "teacher_concurrency", "teacher_timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.teacher_backend not in {"transformers", "vllm"}:
            raise ValueError("teacher_backend must be transformers or vllm")
        if self.evaluation_mode not in {"selector_only", "exhaustive"}:
            raise ValueError("evaluation_mode must be selector_only or exhaustive")
        from urllib.parse import urlsplit
        url = urlsplit(self.teacher_base_url)
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.path.rstrip("/") != "/v1" or url.username or url.password or url.query or url.fragment):
            raise ValueError("Teacher endpoint must be a local HTTP /v1 endpoint (use an SSH tunnel for a remote server)")
        if not 0 < self.teacher_temperature <= 2 or not 0 < self.teacher_top_p <= 1 or self.teacher_top_k < 0:
            raise ValueError("Invalid teacher generation settings")
        for model in (self.reasoner, self.teacher):
            if not model.source or not model.revision or model.dtype not in {"bfloat16", "float32"}:
                raise ValueError("Invalid model identity/dtype")
            if model.attention_backend not in {"sdpa", "eager", "flash_attention_2"}:
                raise ValueError("Unsupported attention backend")
        if self.reasoner.device == "auto":
            raise ValueError("Reasoner consumer requires one explicit device; teacher may use auto")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: Path):
        raw = json.loads(path.read_text())
        for name in ("reasoner", "teacher"):
            if name in raw:
                raw[name] = ModelConfig(**raw[name])
        return cls(**raw)
