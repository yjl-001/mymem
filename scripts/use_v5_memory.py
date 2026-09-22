#!/usr/bin/env python3
"""Answer one input with the frozen V5 selector and one native-prefix Memory Bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from memgen.experience.bank_construction.artifacts import Store, read_json
    from memgen.experience.bank_construction.config import ModelConfig
    from memgen.experience.v5.config import V5Config
    from memgen.experience.v5.online import answer, prepare_bank
    from memgen.experience.v5.sources import make_profile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    if not args.input.strip():
        raise ValueError("V5 input cannot be empty")
    profile = read_json(args.bank_dir / "profile.json")
    raw = profile["configuration"]
    raw = {**raw, **{name: ModelConfig(**raw[name]) for name in ("reasoner", "teacher", "reranker")}}
    if args.device:
        from dataclasses import replace
        raw["reasoner"] = replace(raw["reasoner"], device=args.device)
        raw["reranker"] = replace(raw["reranker"], device=args.device)
    config = V5Config(**raw)
    make_profile(config, profile, scope="bank")
    store = Store(args.bank_dir, profile)
    bundle, policy, records = prepare_bank(store)
    from memgen.model.local_bank import LocalModel
    reasoner = LocalModel(profile["reasoner"], config.reasoner, reasoner=True)
    reranker = None
    try:
        if config.reranker_enabled:
            from memgen.model.v5_local_rerank import V5LocalReranker
            reranker = V5LocalReranker(profile["reranker"], config.reranker.device,
                                       config.reranker_max_length)
        result = answer(args.input, store, reasoner, reranker, bundle, policy, records,
                        config.selector_top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        reasoner.close()
        if reranker is not None:
            reranker.model = None; reranker.tokenizer = None


if __name__ == "__main__":
    main()
