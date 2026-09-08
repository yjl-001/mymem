"""Model APIs; offline Side-KV utilities do not require training dependencies."""

__all__ = ["MemGenModel"]


def __getattr__(name):
    if name == "MemGenModel":
        from memgen.model.modeling_memgen import MemGenModel

        globals()[name] = MemGenModel
        return MemGenModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
