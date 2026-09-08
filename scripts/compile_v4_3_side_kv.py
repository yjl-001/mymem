#!/usr/bin/env python3
"""Compile both qualified V4.3 tiers, authenticating existing artifacts on resume."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import load_construction, read_json
from memgen.experience.v4_3_bank import authenticate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--reasoner-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.is_symlink() or args.output_dir.resolve() in {args.bank_dir.resolve(), args.reasoner_manifest.parent.resolve()}:
        raise ValueError("Compiled output must be separate from immutable source directories")
    bank = load_construction(args.bank_dir)
    old = read_json(args.reasoner_manifest)
    authenticate(old, "manifest_sha256", "source reasoner")
    reasoner = {k: old["reasoner"][k] for k in ("model_name", "model_revision", "tokenizer_revision", "model_sequence_limit")}
    if any(not re.fullmatch(r"[0-9a-f]{40}", reasoner[k]) for k in ("model_revision", "tokenizer_revision")):
        raise ValueError("Reasoner revisions must be exact commits")
    import torch
    from memgen.model.v4_3_side_kv import V43SideKVBankLoader, V43SideKVCompiler, save_compiled
    pending = []
    for tier in ("primary", "conditional"):
        records, manifest = bank[f"{tier}_bank_records.jsonl"], bank[f"{tier}_bank_manifest.json"]
        path = args.output_dir / f"v4_3_{tier}_side_kv_manifest.json"
        if not records:
            if path.exists() or (args.output_dir / f"v4_3_{tier}_side_kv.safetensors").exists():
                raise ValueError("A stale compiled tier exists for a now-empty source")
            print(f"[v4.3-compile] tier={tier} no_qualified_banks", flush=True)
            continue
        if path.exists():
            if not (args.resume or args.validate_only):
                raise ValueError("Compiled output exists; pass --resume")
            loader = V43SideKVBankLoader(path, source_manifest=manifest, source_records=records, expected_reasoner=reasoner)
            if loader.manifest["dtype"] != "torch.bfloat16":
                raise ValueError("Server compiler cannot reuse test float32 tensors")
            print(f"[v4.3-compile] authenticated reuse tier={tier} banks={len(loader.bank_ids)}", flush=True)
        elif args.validate_only:
            raise ValueError(f"Missing compiled artifact: {path}")
        else:
            pending.append((tier, records, manifest))
    if not pending:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(reasoner["model_name"], revision=reasoner["tokenizer_revision"])
    model = AutoModelForCausalLM.from_pretrained(reasoner["model_name"], revision=reasoner["model_revision"],
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    if (getattr(model.config, "_commit_hash", reasoner["model_revision"]) != reasoner["model_revision"]
            or tokenizer.init_kwargs.get("_commit_hash", reasoner["tokenizer_revision"]) != reasoner["tokenizer_revision"]):
        raise ValueError("Loaded model/tokenizer revisions drifted")
    compiler = V43SideKVCompiler(model=model, tokenizer=tokenizer, reasoner=reasoner)
    for tier, records, manifest in pending:
        tensors, compiled = compiler.compile(records, manifest)
        path = save_compiled(args.output_dir, tensors, compiled)
        loader = V43SideKVBankLoader(path, source_manifest=manifest, source_records=records, expected_reasoner=reasoner)
        print(f"[v4.3-compile] tier={tier} banks={len(loader.bank_ids)} records={len(loader.bank_ids)} manifest={path}", flush=True)


if __name__ == "__main__":
    main()
