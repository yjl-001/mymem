#!/usr/bin/env python3
"""Resume a pre-two-phase authenticated run through compile/evaluate compatibility."""
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
    "memgen/experience/bank_construction/audit.py":
        "1aefd6a10847e9ca18df2b5b9b13f5d3d6ffe0ab1163dcdd97d3cdba7c13c960",
    "memgen/experience/bank_construction/cards.py":
        "7b1cd69536851f1062b469b09ed819b7fe4297fd77697695e28c4e512a8a7d4e",
    "memgen/experience/bank_construction/config.py":
        "438372c57ce8ddf2e74968e6b4340f2818820334856f686c745f933cd936323b",
    "memgen/experience/bank_construction/evaluation.py":
        "3815c6725335d671163870457fb2a72b8e728a566ba8a56d7e6b31e211ed869b",
    "memgen/experience/bank_construction/grouping.py":
        "798b89003f38304644d8f12ba2ebba8ff65b5ed3017ab79e73d0bff896fd6187",
    "memgen/experience/bank_construction/identifiers.py": None,
    "memgen/experience/bank_construction/pipeline.py":
        "cc75e88b0471c52e7890e0fc424118081dcf0b3dba984a0c290bc0a62f2b53fc",
    "memgen/experience/bank_construction/reuse.py":
        "d4caa2702580986d8342ca77b6177a06d9970c2c29d5ccb6ccb9d8c4f06745df",
    "memgen/experience/bank_construction/rollouts.py":
        "45a49c2e9b6d00148474af2488216a3dd2020bd6c073337bb6be3956ff121802",
    "memgen/experience/bank_construction/sources.py":
        "e2e81dc550f132d14123288b0336da84350ca6d609b17c2643a46ba77ee373ee",
    "memgen/model/v4_3_prefix_equivalence.py":
        "dd33597585bbe348100a4d999f31d7b495ee6c54d562fce7afcc18d264ea13b3",
    "memgen/model/transformers_cache_compat.py": None,
    "scripts/build_local_memory_bank.py":
        "8cb0fdc6a8315fdf729409cfb11db6b0c2060edfbd3823a532a7d96140394c56",
}
TARGET = {
    "memgen/experience/bank_construction/audit.py":
        "d6ab304099dac7165dcb4e7686e9cc1b0261444142747538342a72ea18a8f90f",
    "memgen/experience/bank_construction/cards.py":
        "8d5043bedadf5cae82652f16c67676f6a5f0e2438aa9892a387647986566c123",
    "memgen/experience/bank_construction/config.py":
        "a8a2b141abb26b8176a01f2bed69f673288fd91efea41152d14b04143ccd7a0f",
    "memgen/experience/bank_construction/evaluation.py":
        "075926c876ad8cf35be201e26a6074258a3fc28042cd073017d3865e7d6c6212",
    "memgen/experience/bank_construction/grouping.py":
        "55049774cf62a678b86d285a07bf50f91455719fe50656e6aab871d701462ca2",
    "memgen/experience/bank_construction/identifiers.py":
        "75515ae65bab8a9ab40aac67e3b1aa28ca0dff224e46c6cf61c62f4ba246d9c1",
    "memgen/experience/bank_construction/pipeline.py":
        "b8d066645525a92a893a06d9cea4aa32b97cc9cdabab0fcacec899b409310c68",
    "memgen/experience/bank_construction/reuse.py":
        "0a5cec87a63876e7b728af26054d7ed283930133fabf4240fbd1d9b27b9e33a0",
    "memgen/experience/bank_construction/rollouts.py":
        "2175a4cb3705b5ee03dcae96860a5dbef604ab483dc7fb16c81b194deea45fae",
    "memgen/experience/bank_construction/sources.py":
        "5ea9773c70339adc3a918d5b55b6e13eef48e59e4765080b36eb187ad33cc328",
    "memgen/model/v4_3_prefix_equivalence.py":
        "6c53b2448ee0c212957a25d8e4d2f1ff5ffde61490a6851be6f4778fab3b7241",
    "memgen/model/transformers_cache_compat.py":
        "730cd879d61db1379fe3062010c2cd1f439349faf6ded90620f7f9597a827ae2",
    "scripts/build_local_memory_bank.py":
        "0375765638445651ba08884b31cbe6e11a378a23251c35071acb22f856ec6683",
}


def config_from_profile(profile):
    raw = deepcopy(profile["configuration"])
    # Preserve the historical run's every-Bank validation contract. New runs use
    # selector_only by default, but silently changing an interrupted run would mix
    # two evaluation policies under one immutable profile.
    raw.setdefault("evaluation_mode", "exhaustive")
    raw.setdefault("group_candidate_top_k", 64)
    raw.setdefault("group_consolidation_rounds", 3)
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
        raise ValueError("Implementation drift outside the authenticated compatibility migration: " +
                         ", ".join(sorted(unexpected)))
    invalid = {path for path in allowed
               if recorded.get(path) != BASELINE[path] or current.get(path) != TARGET[path]}
    if invalid:
        raise ValueError("Run is not on the authenticated cache compatibility migration: " +
                         ", ".join(sorted(invalid)))
    return {"mode": "pre-two-phase-production-compatibility",
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
                       "purpose": "Transformers cache and two-phase production compatibility",
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
