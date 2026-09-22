"""Small, dependency-free adapters for Transformers 4.x and 5.x cache APIs."""
from __future__ import annotations


def cache_pairs(cache):
    """Return per-layer ``(key, value)`` pairs without assuming one HF release."""
    if isinstance(cache, (tuple, list)):
        pairs = cache
    else:
        convert = getattr(cache, "to_legacy_cache", None)
        if callable(convert):
            pairs = convert()
        else:
            layers = getattr(cache, "layers", None)
            if layers is not None:
                pairs = [(getattr(layer, "keys", None), getattr(layer, "values", None))
                         for layer in layers]
            else:
                keys, values = getattr(cache, "key_cache", None), getattr(cache, "value_cache", None)
                if keys is None or values is None or len(keys) != len(values):
                    raise TypeError("Unsupported Transformers cache representation")
                pairs = list(zip(keys, values))
    pairs = tuple(tuple(pair[:2]) for pair in pairs)
    if not pairs or any(len(pair) != 2 or pair[0] is None or pair[1] is None for pair in pairs):
        raise ValueError("Incomplete Transformers key/value cache")
    return pairs


def dynamic_cache_from_pairs(dynamic_cache_class, pairs):
    """Build DynamicCache through the old factory or the v5 data constructor."""
    pairs = tuple(tuple(pair) for pair in pairs)
    factory = getattr(dynamic_cache_class, "from_legacy_cache", None)
    if callable(factory):
        return factory(pairs)
    try:
        return dynamic_cache_class(ddp_cache_data=pairs)
    except TypeError:
        try:
            return dynamic_cache_class(pairs)
        except TypeError as positional_error:
            raise TypeError("DynamicCache cannot be reconstructed from key/value pairs") from positional_error
