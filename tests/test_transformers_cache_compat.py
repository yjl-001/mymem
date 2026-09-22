"""Cross-version tests for the Transformers 4.x/5.x DynamicCache bridge."""
import unittest

from memgen.model.transformers_cache_compat import cache_pairs, dynamic_cache_from_pairs


class Layer:
    def __init__(self, key, value):
        self.keys, self.values = key, value


class CacheCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.pairs = (("k0", "v0"), ("k1", "v1"))

    def test_reads_legacy_layers_and_old_key_lists(self):
        self.assertEqual(cache_pairs(self.pairs), self.pairs)

        class Convertible:
            def to_legacy_cache(inner_self):
                return self.pairs

        self.assertEqual(cache_pairs(Convertible()), self.pairs)
        layered = type("Layered", (), {"layers": [Layer(*pair) for pair in self.pairs]})()
        self.assertEqual(cache_pairs(layered), self.pairs)
        old_lists = type("OldLists", (), {
            "key_cache": ["k0", "k1"], "value_cache": ["v0", "v1"]})()
        self.assertEqual(cache_pairs(old_lists), self.pairs)

    def test_rejects_unknown_or_incomplete_cache(self):
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            cache_pairs(object())
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            cache_pairs(type("Layered", (), {"layers": [Layer("k", None)]})())

    def test_rebuilds_with_transformers4_factory(self):
        class Transformers4:
            @classmethod
            def from_legacy_cache(cls, pairs):
                return ("v4", pairs)

        self.assertEqual(dynamic_cache_from_pairs(Transformers4, self.pairs), ("v4", self.pairs))

    def test_rebuilds_with_transformers5_ddp_data(self):
        class Transformers5:
            def __init__(self, *, ddp_cache_data):
                self.pairs = ddp_cache_data

        result = dynamic_cache_from_pairs(Transformers5, self.pairs)
        self.assertEqual(result.pairs, self.pairs)

    def test_positional_constructor_compatibility_fallback(self):
        class Positional:
            def __init__(self, pairs):
                self.pairs = pairs

        self.assertEqual(dynamic_cache_from_pairs(Positional, self.pairs).pairs, self.pairs)


if __name__ == "__main__":
    unittest.main()
