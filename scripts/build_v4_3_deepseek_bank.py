#!/usr/bin/env python3
"""Construct one evidence-grounded DeepSeek card per retained V4.2 Bank."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import atomic_json, implementation_hashes
from memgen.experience.v4_3_bank import authenticate, authenticate_sources, build_outputs, canonical_hash, file_hash, seal
from memgen.experience.v4_3_deepseek import IMPLEMENTATION_PATHS, POLICY, build_semantic_candidate, parse_response, request_spec
from scripts.build_v4_3_unified_bank import encode_outputs, read_json, read_jsonl, write_or_validate


class DeepSeekClient:
    """Lazy provider adapter; caches parsed responses, never headers or secrets."""
    def __init__(self, key, profile):
        import requests
        from scripts.build_teacher_bank import TeacherClient

        class Session(requests.Session):
            attempts = 0
            usage = None

            def post(self, *args, **kwargs):
                self.attempts += 1
                response = super().post(*args, **kwargs, allow_redirects=False)
                self.usage = None
                if response.status_code == 200:
                    try:
                        usage = response.json().get("usage", {})
                        self.usage = {k: v for k, v in usage.items()
                                      if k in {"prompt_tokens", "completion_tokens", "total_tokens",
                                               "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
                                      and type(v) is int and v >= 0}
                    except (ValueError, AttributeError):
                        pass
                return response

        self.session = Session()
        self.client = TeacherClient(base_url="https://api.deepseek.com", api_key=key,
            model=profile["model"], max_tokens=profile["max_tokens"], temperature=0.0,
            retries=2, proxy_retries=0, proxy_retry_initial_seconds=1, proxy_retry_max_seconds=1,
            connect_timeout_seconds=20, read_timeout_seconds=180, thinking="disabled", session=self.session)

    def generate(self, request, packet):
        start = self.session.attempts
        answer = self.client.call(request["messages"], response_parser=lambda raw: parse_response(raw, packet),
            request_label="v4.3-card", expose_parser_error=True, repair_parser_errors=True)
        return answer, {"http_attempts": self.session.attempts - start, "final_response_usage": self.session.usage}

    def close(self):
        self.client.close()


def construct(*, source_dir, packets_path, policy_path, output_dir, cache_dir,
              resume=False, validate_only=False, model="deepseek-v4-flash", max_tokens=8192,
              client_factory=DeepSeekClient):
    # Resolve only after rejecting symlinks. Keep immutable response cache outside
    # the sealed construction bundle; old lexical outputs are never overwritten.
    for path in (output_dir, cache_dir):
        if path.is_symlink():
            raise ValueError("Construction destination cannot be a symlink")
    source_dir, packets_path, policy_path, output_dir, cache_dir = (
        p.resolve() for p in (source_dir, packets_path, policy_path, output_dir, cache_dir))
    for destination in (output_dir, cache_dir):
        if (destination == ROOT or destination in ROOT.parents
                or destination == ROOT / "docs/figures" or ROOT / "docs/figures" in destination.parents
                or any(destination == p or destination in p.parents or p in destination.parents
                       for p in (source_dir, packets_path, policy_path))):
            raise ValueError("Construction destinations must be separate from protected paths and inputs")
    if output_dir == cache_dir or output_dir in cache_dir.parents or cache_dir in output_dir.parents:
        raise ValueError("Response cache and construction output must be separate directories")
    paths = {"records": source_dir / "bank_records.jsonl", "manifest": source_dir / "bank_manifest.json",
             "packets": packets_path, "policy": policy_path}
    hashes = {k: file_hash(p) for k, p in paths.items()}
    records, packets = read_jsonl(paths["records"]), read_jsonl(paths["packets"])
    manifest, policy = read_json(paths["manifest"]), read_json(paths["policy"])
    if hashes != {k: file_hash(p) for k, p in paths.items()}:
        raise ValueError("Construction inputs changed while reading")
    packet_map = authenticate_sources(records=records, manifest=manifest, packets=packets,
                                      policy=policy, input_hashes=hashes)
    requests = [request_spec(r, packet_map[r["cluster"]["source_candidate_id"]], model, max_tokens) for r in records]
    names = [canonical_hash(request) + ".json" for request in requests]
    profile = seal({"schema_version": "memgen-v4.3-deepseek-cache-v1", "input_sha256": hashes,
                    "implementation_sha256": implementation_hashes(IMPLEMENTATION_PATHS),
                    "policy_sha256": canonical_hash(POLICY), "model": model, "max_tokens": max_tokens,
                    "request_sha256": [canonical_hash(r) for r in requests]}, "profile_sha256")
    if not validate_only:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise ValueError("Complete DeepSeek cache required for validation")
    lock_path = cache_dir / ".lock"
    if lock_path.is_symlink():
        raise ValueError("Cache lock cannot be a symlink")
    # Persistent inode: never unlink a lock held by another process.
    with lock_path.open("r" if validate_only else "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another construction process owns this response cache") from None
        existing = list(cache_dir.iterdir())
        if any(p.name != ".lock" for p in existing) and not (resume or validate_only):
            raise ValueError("Response cache exists; use --resume")
        allowed = set(names) | {"profile.json", ".lock"}
        if any(p.is_symlink() or not p.is_file() for p in existing):
            raise ValueError("Unexpected or unsafe response cache entry")
        profile_path = cache_dir / "profile.json"
        if profile_path.exists():
            if read_json(profile_path) != profile:
                raise ValueError("DeepSeek cache input/model/prompt/code profile drift")
        elif len(existing) > 1 or validate_only:
            raise ValueError("DeepSeek cache profile missing")
        if any(p.name not in allowed for p in existing):
            raise ValueError("Unexpected or unsafe response cache entry")
        entries = {}
        # Authenticate ALL completed responses before any provider/key access.
        for i, name in enumerate(names):
            if (cache_dir / name).exists():
                entry = read_json(cache_dir / name)
                authenticate(entry, "cache_sha256", "DeepSeek cached response")
                if entry["profile_sha256"] != profile["profile_sha256"] or entry["request"] != requests[i]:
                    raise ValueError("Cached response profile/request drift")
                build_semantic_candidate(records[i], packet_map[records[i]["cluster"]["source_candidate_id"]], entry)
                entries[name] = entry
        pending = [i for i, name in enumerate(names) if name not in entries]
        if pending and validate_only:
            raise ValueError("DeepSeek cache incomplete; validation cannot call the API")
        if output_dir.exists() and any(output_dir.iterdir()) and (pending or not (resume or validate_only)):
            raise ValueError("Existing construction output requires complete authenticated cache and --resume")
        if not validate_only:
            atomic_json(profile_path, profile, immutable=True)
        client = None
        try:
            if pending:
                key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
                if not key:
                    raise ValueError("DEEPSEEK_API_KEY is required only for uncached Bank requests")
                client = client_factory(key, profile)
                del key
            for i in pending:
                packet = packet_map[records[i]["cluster"]["source_candidate_id"]]
                print(f"[v4.3] DeepSeek bank={i + 1}/17 evidence_count={len(packet['evidence'])}", flush=True)
                response, receipt = client.generate(requests[i], packet)
                entry = seal({"profile_sha256": profile["profile_sha256"], "request": requests[i],
                              "request_sha256": canonical_hash(requests[i]), "response": response,
                              "receipt": receipt}, "cache_sha256")
                # A genuine insufficient-support outcome is saved, never retried
                # to pressure the model into falsely reaching the support quota.
                build_semantic_candidate(records[i], packet, entry)
                atomic_json(cache_dir / names[i], entry, immutable=True)
                entries[names[i]] = entry
        finally:
            if client is not None:
                client.close()
        if hashes != {k: file_hash(p) for k, p in paths.items()}:
            raise ValueError("Construction inputs changed during provider calls")
        candidates = [build_semantic_candidate(r, packet_map[r["cluster"]["source_candidate_id"]], entries[name])
                      for r, name in zip(records, names)]
        outputs = build_outputs(records=records, manifest=manifest, packets=packets, policy=policy,
            input_hashes=hashes, implementation_hashes=profile["implementation_sha256"],
            candidates=candidates, construction_policy=POLICY, report_metadata={
                "construction_method": "deepseek_semantic_synthesis_v1", "teacher_model": model,
                "api_key_read": True, "api_key_persisted": False,
                "external_api_calls_made": sum(e["receipt"]["http_attempts"] for e in entries.values()),
                "api_call_count_scope": "recorded_attempts_for_cached_completed_banks_excludes_interrupted_requests",
                "semantic_support_assessor": "DeepSeek", "independent_semantic_verification": False,
                "source_field_judgment_count": sum(len(s["candidate_audit"]) for r in candidates for s in r["clause_support"].values()),
                "screened_clause_count": sum(len(r["clause_support"]) for r in candidates),
                "flagged_candidate_clause_count": sum(bool(s["generated_clause_issues"]) for r in candidates for s in r["clause_support"].values()),
                "embedding_artifact_reason": "semantic_synthesis_via_DeepSeek_no_local_embedding_model",
                "teacher_cache_profile_sha256": profile["profile_sha256"]})
        write_or_validate(output_dir, encode_outputs(outputs), resume=resume, validate_only=validate_only)
        print("[v4.3] " + json.dumps({"new_bank_requests": len(pending), "cached_bank_count": len(entries),
            "qualified_tier_counts": outputs["construction_report.json"]["qualified_tier_counts"],
            "quarantined_count": outputs["construction_report.json"]["quarantined_count"]}), flush=True)
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--semantic-packets", type=Path, required=True)
    parser.add_argument("--curation-policy", type=Path, default=ROOT / "configs/experiments/gsm8k/v4_2_local_curation_policy.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    construct(source_dir=args.source_dir, packets_path=args.semantic_packets, policy_path=args.curation_policy,
              output_dir=args.output_dir, cache_dir=args.cache_dir, resume=args.resume,
              validate_only=args.validate_only, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
