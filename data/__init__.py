"""Dataset-builder registry with lazy imports.

Offline verifier utilities such as ``data.utils.math_utils`` should not need to
load Hugging Face datasets or every task-specific dependency at import time.
"""

from __future__ import annotations

from typing import Any


_DATA_BUILDER_PATHS = {
    "gpqa": ("data.gpqa.builder", "GPQABuilder"),
    "gsm8k": ("data.gsm8k.builder", "GSM8KBuilder"),
    "kodcode": ("data.kodcode.builder", "KodCodeBuilder"),
    "triviaqa": ("data.triviaqa.builder", "TriviaQABuilder"),
}


def _import_value(module_name: str, name: str) -> Any:
    from importlib import import_module

    return getattr(import_module(module_name), name)


def get_data_builder(dataset_cfg):
    dataset_name = dataset_cfg.get("name")
    if dataset_name not in _DATA_BUILDER_PATHS:
        raise ValueError("Unsupported dataset.")
    module_name, class_name = _DATA_BUILDER_PATHS[dataset_name]
    builder_cls = _import_value(module_name, class_name)
    return builder_cls(dataset_cfg)


def __getattr__(name: str) -> Any:
    lazy_exports = {
        "BaseBuilder": ("data.base_builder", "BaseBuilder"),
        "BaseEnv": ("data.base_env", "BaseEnv"),
        "StaticEnv": ("data.base_env", "StaticEnv"),
        "DynamicEnv": ("data.base_env", "DynamicEnv"),
        "GPQABuilder": ("data.gpqa.builder", "GPQABuilder"),
        "GSM8KBuilder": ("data.gsm8k.builder", "GSM8KBuilder"),
        "KodCodeBuilder": ("data.kodcode.builder", "KodCodeBuilder"),
        "TriviaQABuilder": ("data.triviaqa.builder", "TriviaQABuilder"),
    }
    if name in lazy_exports:
        return _import_value(*lazy_exports[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseBuilder",
    "BaseEnv",
    "StaticEnv",
    "DynamicEnv",
    "GPQABuilder",
    "GSM8KBuilder",
    "KodCodeBuilder",
    "TriviaQABuilder",
    "get_data_builder",
]
