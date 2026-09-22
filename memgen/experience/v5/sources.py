"""Freeze V5 source identities and bind runs to implementation hashes."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

from memgen.experience.bank_construction.artifacts import file_digest
from memgen.experience.bank_construction.sources import resolve_model


ROLLOUT_FILES = (
    "data/gsm8k/prompt.py", "data/gsm8k/splits.py", "data/utils/math_utils.py",
    "memgen/chat_templates.py", "memgen/model/local_bank.py",
    "memgen/experience/bank_construction/artifacts.py", "memgen/experience/v5/config.py",
    "memgen/experience/v5/task.py", "memgen/experience/v5/rollouts.py",
    "memgen/experience/v5/sources.py", "scripts/build_v5_memory.py",
    "scripts/experiments/gsm8k/run_v5_memory.sh",
)


def implementation_hashes(scope):
    root = Path(__file__).resolve().parents[3]
    if scope == "rollouts":
        paths = [root / name for name in ROLLOUT_FILES]
    elif scope == "bank":
        shared = set(ROLLOUT_FILES) | {
            "memgen/experience/bank_construction/parallel.py",
            "memgen/experience/bank_construction/sources.py",
            "memgen/experience/bank_construction/vllm_teacher.py",
            "memgen/experience/v4_3_artifacts.py",
            "memgen/experience/v4_3_bank.py",
            "memgen/model/e1_runtime.py",
            "memgen/model/transformers_cache_compat.py",
            "memgen/model/v4_3_local_rerank.py",
            "memgen/model/v4_3_prefix_equivalence.py",
            "memgen/model/v4_3_question_selector.py",
            "memgen/model/v4_3_runtime.py",
            "memgen/model/v4_oracle.py",
            "memgen/model/v5_local_rerank.py",
            "scripts/evaluate_v5_memory.py",
            "scripts/serve_v5_teacher.py",
            "scripts/use_v5_memory.py",
            "test.sh",
        }
        paths = list((root / "memgen/experience/v5").glob("*.py")) + [root / name for name in shared]
    else:
        raise ValueError("V5 implementation scope must be rollouts or bank")
    paths = sorted(set(paths))
    return {str(path.relative_to(root)): file_digest(path) for path in paths}


def environment():
    result = {}
    for name in ("torch", "transformers", "datasets", "safetensors", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def make_profile(config, previous=None, *, scope):
    code, runtime = implementation_hashes(scope), environment()
    if previous is not None:
        if (previous.get("implementation_scope") != scope
                or previous.get("configuration") != config.to_dict()
                or previous.get("implementation") != code
                or previous.get("environment") != runtime):
            raise ValueError("V5 configuration/code/runtime changed; use a new run directory")
        for role in ("reasoner", "teacher", "reranker"):
            identity = previous.get(role)
            if identity and identity.get("files") and resolve_model(getattr(config, role)) != identity:
                raise ValueError(f"Local {role} files changed")
        return previous
    from huggingface_hub import HfApi
    revision = config.dataset_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().dataset_info(config.dataset, revision=revision).sha
    return {"schema_version": "memgen-v5-run-v1", "implementation_scope": scope,
        "configuration": config.to_dict(), "reasoner": resolve_model(config.reasoner),
        "teacher": resolve_model(config.teacher) if scope == "bank" else None,
        "reranker": resolve_model(config.reranker) if scope == "bank" and config.reranker_enabled else None,
        "dataset": {"source": config.dataset, "revision": revision},
        "implementation": code, "environment": runtime, "external_inference_api_calls": 0,
        "test_used_for_construction": False, "consumer": "native-prefix-kv-all-layers",
        "selector_query": "input-only", "memory_tier": "primary-only"}
