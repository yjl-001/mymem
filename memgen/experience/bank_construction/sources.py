"""Freeze moving Hub revisions or hash local model directories before inference."""
from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import re

from .artifacts import digest, file_digest


def resolve_model(config):
    path = Path(config.source).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       p.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"})
        if not (path / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("Local models require config.json and safetensors weights")
        hashes = {str(p.relative_to(path)): file_digest(p) for p in files}
        return {"source": str(path.resolve()), "revision": None, "files": hashes, "sha256": digest(hashes)}
    from huggingface_hub import HfApi
    revision = config.revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(config.source, revision=revision).sha
    return {"source": config.source, "revision": revision}


def implementation_hashes():
    root = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob("*.py")) + [
        root / name for name in (
            "data/gsm8k/splits.py", "data/gsm8k/prompt.py", "data/utils/math_utils.py",
            "memgen/chat_templates.py", "memgen/model/local_bank.py",
            "memgen/model/v4_3_prefix_equivalence.py", "memgen/model/v4_3_question_selector.py",
            "memgen/model/v4_3_runtime.py", "memgen/model/v4_oracle.py", "memgen/model/e1_runtime.py",
            "scripts/build_local_memory_bank.py")]
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(files)}


def environment():
    result = {}
    for name in ("torch", "transformers", "datasets", "safetensors", "accelerate", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def make_profile(config, previous=None):
    code = implementation_hashes()
    if previous is not None:
        if previous["configuration"] != config.to_dict() or previous["implementation"] != code:
            raise ValueError("Configuration/code changed; use a new run directory")
        if previous["environment"] != environment():
            raise ValueError("Runtime versions changed; use the original environment or a new run")
        for role in ("reasoner", "teacher"):
            if previous[role].get("files") and resolve_model(getattr(config, role)) != previous[role]:
                raise ValueError(f"Local {role} files changed")
        return previous  # Never resolve 'main' again during a resume.
    from huggingface_hub import HfApi
    revision = config.dataset_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().dataset_info(config.dataset, revision=revision).sha
    return {"schema_version": "memgen-local-bank-run-v1", "configuration": config.to_dict(),
            "reasoner": resolve_model(config.reasoner), "teacher": resolve_model(config.teacher),
            "dataset": {"source": config.dataset, "revision": revision},
            "implementation": code, "environment": environment(),
            "external_inference_api_calls": 0, "test_used_for_construction": False,
            "consumer": "v43-native-prefix-kv-all-layers", "teacher_thinking": False}
