#!/usr/bin/env python3
"""Resume an authenticated local-Bank run across the Transformers v5 cache API migration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, digest, file_digest, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig, ModelConfig
from memgen.experience.bank_construction.sources import environment, implementation_hashes

BASELINE = {
    "memgen/experience/bank_construction/sources.py":
        "e2e81dc550f132d14123288b0336da84350ca6d609b17c2643a46ba77ee373ee",
    "memgen/model/v4_3_prefix_equivalence.py":
        "dd33597585bbe348100a4d999f31d7b495ee6c54d562fce7afcc18d264ea13b3",
    "memgen/model/transformers_cache_compat.py": None,
}
TARGET = {
    "memgen/experience/bank_construction/sources.py":
        "5ae228526abe2f65396867693c01493861dd23347c47bb1a162c23de9909cf45",
    "memgen/model/v4_3_prefix_equivalence.py":
        "6c53b2448ee0c212957a25d8e4d2f1ff5ffde61490a6851be6f4778fab3b7241",
    "memgen/model/transformers_cache_compat.py":
        "730cd879d61db1379fe3062010c2cd1f439349faf6ded90620f7f9597a827ae2",
}


def config_from_profile(profile):
    raw = deepcopy(profile["configuration"])
    raw["reasoner"] = ModelConfig(**raw["reasoner"])
    raw["teacher"] = ModelConfig(**raw["teacher"])
    return ConstructionConfig(**raw)


def verify_implementation_migration(recorded, current):
    if recorded == current:
        return {"mode": "exact_profile_implementation", "changes": {}}
    allowed = set(BASELINE)
    unexpected = {path for path in set(recorded) | set(current)
                  if path not in allowed and recorded.get(path) != current.get(path)}
    if unexpected:
        raise ValueError("Implementation drift outside the cache compatibility patch: " +
                         ", ".join(sorted(unexpected)))
    invalid = {path for path in allowed
               if recorded.get(path) != BASELINE[path] or current.get(path) != TARGET[path]}
    if invalid:
        raise ValueError("Run is not on the authenticated cache compatibility migration: " +
                         ", ".join(sorted(invalid)))
    return {"mode": "transformers-v5-cache-api-compatibility",
            "changes": {path: {"recorded": BASELINE[path], "current": TARGET[path]}
                        for path in sorted(allowed)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    with run_lock(args.output_dir):
        profile = read_json(args.output_dir / "profile.json")
        if environment() != profile["environment"]:
            raise ValueError("Runtime environment differs from this run")
        current = implementation_hashes()
        migration = verify_implementation_migration(profile["implementation"], current)
        store = Store(args.output_dir, profile)
        if migration["changes"]:
            receipt = {"schema_version": "memgen-implementation-compatibility-v1",
                       "purpose": "Transformers 4.x/5.x DynamicCache API compatibility",
                       "migration": migration,
                       "script_sha256": file_digest(Path(__file__).resolve())}
            store.put("imports/cache-api-compat-" + digest(receipt)[:16], receipt,
                      {"profile_sha256": store.profile_hash, "migration": migration})
        config = config_from_profile(profile)
        if args.validate_only:
            from memgen.experience.bank_construction.audit import audit_complete
            print(json.dumps(audit_complete(store, config), indent=2))
            return
        from memgen.experience.bank_construction.pipeline import run
        run(store, config, profile, stage="compile")
        run(store, config, profile, stage="evaluate")
        print(f"[cache-compat] complete summary={store.root / 'brief_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
