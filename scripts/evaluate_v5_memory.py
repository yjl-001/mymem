#!/usr/bin/env python3
"""Evaluate a frozen V5 Bank and selector on valid or official test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from memgen.experience.bank_construction.artifacts import Store, digest, read_json, run_lock
    from memgen.experience.bank_construction.config import ModelConfig
    from memgen.experience.v5.config import V5Config
    from memgen.experience.v5.evaluation import run_evaluation
    from memgen.experience.v5.sources import make_profile
    from memgen.experience.v5.task import task_for
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    bank_root, output_root = args.bank_dir.resolve(), args.output_dir.resolve()
    if (bank_root == output_root or bank_root in output_root.parents or output_root in bank_root.parents):
        raise ValueError("V5 evaluation output must be separate from the immutable Bank")
    if (args.output_dir / "profile.json").exists() and not args.resume:
        raise ValueError("V5 evaluation run exists; use --resume")
    bank_profile = read_json(args.bank_dir / "profile.json")
    raw = bank_profile["configuration"]
    config = V5Config(**{**raw, **{name: ModelConfig(**raw[name])
                                   for name in ("reasoner", "teacher", "reranker")}})
    make_profile(config, bank_profile, scope="bank")
    with run_lock(args.output_dir):
        profile = {"schema_version": "memgen-v5-evaluation-run-v1",
            "bank_profile_sha256": digest(bank_profile), "configuration": config.to_dict(),
            "split": args.split, "reasoner": bank_profile["reasoner"],
            "reranker": bank_profile.get("reranker"), "official_test_used": args.split == "test"}
        store = Store(args.output_dir, profile)
        bank_store = Store(args.bank_dir, bank_profile)
        from memgen.model.local_bank import LocalModel
        reasoner_factory = lambda: LocalModel(bank_profile["reasoner"], config.reasoner, reasoner=True)
        reranker_factory = None
        if config.reranker_enabled:
            from memgen.model.v5_local_rerank import V5LocalReranker
            reranker_factory = lambda: V5LocalReranker(bank_profile["reranker"], config.reranker.device,
                                                        config.reranker_max_length)
        report = run_evaluation(store, bank_store, config, task_for(config), args.split,
                                reasoner_factory, reranker_factory)
        from memgen.experience.bank_construction.artifacts import atomic_json
        atomic_json(args.output_dir / "brief_summary.json", {**report, "summary_sha256": digest(report)})
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
