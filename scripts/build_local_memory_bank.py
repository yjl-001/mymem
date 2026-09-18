#!/usr/bin/env python3
"""Construct and validate a local-Qwen memory bank, from builder train to native prefix KV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.pipeline import STAGES, run
from memgen.experience.bank_construction.sources import make_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/gsm8k/local_bank.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Validate config without downloads or inference")
    parser.add_argument("--validate-only", action="store_true", help="Audit a completed run without model inference")
    args = parser.parse_args(argv)
    config = ConstructionConfig.load(args.config)
    if args.plan_only:
        print(json.dumps({"configuration": config.to_dict(), "stages": STAGES,
            "note": "all includes a valid-only baseline and every primary-bank utility branch; no test generation"}, indent=2))
        return
    with run_lock(args.output_dir):
        path = args.output_dir / "profile.json"
        if path.exists() and not (args.resume or args.validate_only):
            raise ValueError("Run exists; use --resume for identical configuration/code")
        if args.validate_only and not path.exists():
            raise ValueError("--validate-only requires an existing completed run")
        if not path.exists() and any(p.name != ".writer.lock" for p in args.output_dir.iterdir()):
            raise ValueError("Nonempty output without a profile; use a new directory")
        profile = make_profile(config, read_json(path) if path.exists() else None)
        store = Store(args.output_dir, profile)
        if args.validate_only:
            from memgen.experience.bank_construction.audit import audit_complete
            print(json.dumps(audit_complete(store, config), indent=2))
        else:
            run(store, config, profile, stage=args.stage)


if __name__ == "__main__":
    main()
