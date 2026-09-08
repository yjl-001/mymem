"""One V4.3 unified Bank to one canonical Side-KV Memory; no legacy roles."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from importlib.metadata import version
from typing import Any, Mapping, Sequence

import torch

from memgen.experience.v4_3_bank import (
    COMPILER_CONTRACT, authenticate, canonical_hash, file_hash, seal, text_hash, validate_manifest,
)
from memgen.experience.v4_3_artifacts import (
    atomic_json, implementation_hashes, local_artifact, read_json,
)
from memgen.model.side_kv import DecoderLayerResolver, SideKVMemory, _require_sdpa

SCHEMA = "memgen-v4.3-unified-side-kv-manifest-v1"
MEMORY_TOTAL_PRIOR = 10.0
MEMORY_SCORE_BIAS = math.log(MEMORY_TOTAL_PRIOR)
VARIANTS = (
    ("raw_descriptor", ""),
    ("internal_principle", "<|im_start|>system\nUse the following reusable process as an internal reasoning principle.<|im_end|>\n<|im_start|>user\n"),
    ("hidden_note", "<|im_start|>system\nTreat the following text as a hidden steering note for the reasoning process.<|im_end|>\n<|im_start|>user\n"),
)
IMPLEMENTATION_PATHS = (
    "memgen/model/__init__.py",
    "memgen/experience/v4_3_bank.py", "memgen/experience/v4_3_artifacts.py",
    "memgen/model/side_kv.py", "memgen/model/v4_3_side_kv.py", "scripts/compile_v4_3_side_kv.py",
)


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in ("torch", "transformers", "safetensors")}


def tensor_sha(value: torch.Tensor) -> str:
    import hashlib
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


class V43SideKVCompiler:
    def __init__(self, *, model: Any, tokenizer: Any, reasoner: Mapping[str, Any]):
        self.model, self.tokenizer, self.reasoner = model, tokenizer, dict(reasoner)
        self.layer = DecoderLayerResolver.resolve(model)[23]
        self.attention = self.layer.self_attn
        _require_sdpa(self.attention, owner=type(self).__name__)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        for attr in ("k_proj", "v_proj"):
            if not hasattr(self.attention, attr):
                raise ValueError(f"Native {attr} missing")
        if not hasattr(self.layer, "input_layernorm"):
            raise ValueError("Native input layer norm missing")
        if self.dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("Use server bfloat16 or explicit CPU float32 testing")

    @torch.inference_mode()
    def compile(self, records: Sequence[Mapping[str, Any]], source_manifest: Mapping[str, Any]) -> tuple[dict, dict]:
        validate_manifest(source_manifest, records)
        if not records:
            raise ValueError("No qualified Banks to compile")
        was_training = self.model.training
        self.model.eval()
        compiled, entries = [], []
        groups = int(self.model.config.num_key_value_heads)
        width = int(self.model.config.hidden_size // self.model.config.num_attention_heads)
        try:
            for index, record in enumerate(records):
                content = list(self.tokenizer.encode(record["descriptor"], add_special_tokens=False))
                if not content:
                    raise ValueError("Descriptor has no content tokens")
                all_k, all_v, spans = [], [], []
                for name, wrapper in VARIANTS:
                    prefix = list(self.tokenizer.encode(wrapper, add_special_tokens=False))
                    if len(prefix) + len(content) > self.reasoner["model_sequence_limit"]:
                        raise ValueError("Descriptor variant exceeds model context")
                    ids = torch.tensor([prefix + content], dtype=torch.long, device=self.device)
                    # Capture precisely the selected block's input, without
                    # retaining all layers' hidden states for long descriptors.
                    captured = []
                    hook = self.layer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
                    try:
                        self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                    finally:
                        hook.remove()
                    if len(captured) != 1:
                        raise RuntimeError("Selected decoder input not captured exactly once")
                    states = captured[0][:, len(prefix):, :]
                    normalized = self.layer.input_layernorm(states)
                    def project(layer):
                        projected = layer(normalized)
                        if projected.shape != (1, len(content), groups * width):
                            raise RuntimeError("Native projection geometry mismatch")
                        return projected.reshape(len(content), groups, width).transpose(0, 1).cpu().contiguous()
                    k, v = project(self.attention.k_proj), project(self.attention.v_proj)
                    if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                        raise RuntimeError("Non-finite compiled K/V")
                    start = sum(t.shape[1] for t in all_k)
                    spans.append({"name": name, "prefix_sha256": text_hash(wrapper), "prefix_token_count": len(prefix),
                                  "content_token_count": len(content), "slot_start": start, "slot_end": start + len(content),
                                  "retention_policy": "content_positions_only"})
                    all_k.append(k)
                    all_v.append(v)
                keys, values = torch.cat(all_k, 1), torch.cat(all_v, 1)
                compiled.append((keys, values))
                entries.append({"index": index, "bank_id": record["bank_id"], "memory_id": record["bank_id"],
                    "quality_tier": record["quality_tier"], "descriptor_sha256": record["descriptor_sha256"],
                    "payload_hash": record["descriptor_sha256"], "source_record_sha256": record["record_sha256"],
                    "kv_valid_slot_count": keys.shape[1], "variants": spans,
                    "key_rms": rms(keys), "value_rms": rms(values),
                    "keys_sha256": tensor_sha(keys), "values_sha256": tensor_sha(values)})
        finally:
            self.model.train(was_training)
        slots = max(k.shape[1] for k, _ in compiled)
        keys = torch.zeros((len(records), groups, slots, width), dtype=self.dtype)
        values = torch.zeros_like(keys)
        mask = torch.zeros((len(records), slots), dtype=torch.bool)
        for i, (k, v) in enumerate(compiled):
            count = k.shape[1]
            keys[i, :, :count], values[i, :, :count], mask[i, :count] = k, v, True
        tensors = {"keys": keys, "values": values, "slot_mask": mask}
        manifest = {"schema_version": SCHEMA, "quality_tier": source_manifest["quality_tier"],
            "compiler_contract": COMPILER_CONTRACT, "offline_only": True, "qualified_for_online_use": False,
            "selector_artifact": None, "contains_answer_or_reward_signal": False,
            "bank_count": len(records), "record_count": len(records), "records": entries,
            "record_order_sha256": canonical_hash([r["bank_id"] for r in records]),
            "reasoner": self.reasoner, "dtype": str(self.dtype),
            "runtime_versions": runtime_versions(),
            "source_bank_manifest_sha256": source_manifest["manifest_sha256"],
            "implementation_sha256": implementation_hashes(IMPLEMENTATION_PATHS),
            "memory_score_normalization": "log_valid_slots", "memory_total_prior": MEMORY_TOTAL_PRIOR,
            "memory_score_bias": MEMORY_SCORE_BIAS,
            "tensor_shapes": {name: list(t.shape) for name, t in tensors.items()},
            "production_dtype": self.dtype == torch.bfloat16,
        }
        return tensors, manifest


def save_compiled(directory: Path, tensors: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]) -> Path:
    from safetensors.torch import save_file, load_file
    directory.mkdir(parents=True, exist_ok=True)
    tier = manifest["quality_tier"]
    tensor_path = directory / f"v4_3_{tier}_side_kv.safetensors"
    manifest_path = directory / f"v4_3_{tier}_side_kv_manifest.json"
    if tensor_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Refusing symlink compiled artifact")
    if tensor_path.exists():
        old = load_file(str(tensor_path))
        if set(old) != set(tensors) or any(not torch.equal(old[k], tensors[k]) for k in old):
            raise ValueError("Existing tensor differs; use a new output directory")
    else:
        fd, name = tempfile.mkstemp(prefix=".v43-tensor-", dir=directory)
        os.close(fd)
        try:
            save_file(dict(tensors), name)
            os.link(name, tensor_path)
        finally:
            Path(name).unlink(missing_ok=True)
    sealed = seal({**manifest, "tensor_artifact": {"path": tensor_path.name, "sha256": file_hash(tensor_path)}}, "manifest_sha256")
    atomic_json(manifest_path, sealed, immutable=True)
    return manifest_path


class V43SideKVBankLoader:
    """Expose only authenticated unified memories; each get returns own storage."""
    def __init__(self, manifest_path: Path, *, source_manifest: Mapping[str, Any],
                 source_records: Sequence[Mapping[str, Any]], expected_reasoner: Mapping[str, Any] | None = None):
        from safetensors.torch import load_file
        validate_manifest(source_manifest, source_records)
        self.manifest = m = read_json(manifest_path)
        authenticate(m, "manifest_sha256", "V4.3 Side-KV")
        if (m.get("schema_version") != SCHEMA or m.get("compiler_contract") != COMPILER_CONTRACT
                or m.get("offline_only") is not True or m.get("qualified_for_online_use") is not False
                or m.get("selector_artifact") is not None or m.get("contains_answer_or_reward_signal") is not False
                or m.get("implementation_sha256") != implementation_hashes(IMPLEMENTATION_PATHS)
                or m.get("runtime_versions") != runtime_versions()
                or m.get("source_bank_manifest_sha256") != source_manifest["manifest_sha256"]
                or m.get("quality_tier") != source_manifest["quality_tier"]):
            raise ValueError("V4.3 Side-KV contract/source/implementation mismatch")
        if (m.get("memory_score_normalization") != "log_valid_slots" or m.get("memory_total_prior") != MEMORY_TOTAL_PRIOR
                or m.get("memory_score_bias") != MEMORY_SCORE_BIAS):
            raise ValueError("V4.3 memory prior drifted")
        if expected_reasoner is not None and any(m["reasoner"].get(k) != v for k, v in expected_reasoner.items()):
            raise ValueError("V4.3 reasoner identity mismatch")
        ids = [r["bank_id"] for r in source_records]
        if (not ids or len(ids) != len(set(ids)) or m.get("bank_count") != len(ids) or m.get("record_count") != len(ids)
                or [e["bank_id"] for e in m["records"]] != ids or m["record_order_sha256"] != canonical_hash(ids)):
            raise ValueError("V4.3 one Bank one record coverage mismatch")
        self.tensors = load_file(str(local_artifact(manifest_path, m["tensor_artifact"])), device="cpu")
        if set(self.tensors) != {"keys", "values", "slot_mask"}:
            raise ValueError("V4.3 unexpected tensor names")
        keys, values, mask = (self.tensors[k] for k in ("keys", "values", "slot_mask"))
        if (keys.ndim != 4 or values.shape != keys.shape or mask.shape != (keys.shape[0], keys.shape[2])
                or mask.dtype != torch.bool or keys.dtype != values.dtype or str(keys.dtype) != m["dtype"]
                or keys.dtype not in {torch.float32, torch.bfloat16}
                or m.get("production_dtype") != (keys.dtype == torch.bfloat16)
                or keys.shape[0] != len(ids) or not torch.isfinite(keys).all() or not torch.isfinite(values).all()
                or m["tensor_shapes"] != {k: list(t.shape) for k, t in self.tensors.items()}):
            raise ValueError("V4.3 tensor geometry/dtype/finite validation failed")
        for index, (entry, source) in enumerate(zip(m["records"], source_records)):
            count = entry["kv_valid_slot_count"]
            if (not isinstance(count, int) or not 0 < count <= keys.shape[2] or entry["index"] != index
                    or entry.get("memory_id") != source["bank_id"] or "role" in entry
                    or entry.get("quality_tier") != source["quality_tier"]
                    or not source["bank_id"].startswith("v43-bank-") or "::" in source["bank_id"]
                    or entry["source_record_sha256"] != source["record_sha256"]
                    or entry["descriptor_sha256"] != source["descriptor_sha256"]
                    or entry["payload_hash"] != source["descriptor_sha256"]
                    or not mask[index, :count].all() or mask[index, count:].any()
                    or keys[index, :, count:].count_nonzero() or values[index, :, count:].count_nonzero()):
                raise ValueError("V4.3 unified entry/mask/source binding failed")
            variants = entry["variants"]
            if len(variants) != 3 or [v["name"] for v in variants] != [n for n, _ in VARIANTS]:
                raise ValueError("V4.3 three descriptor variants required")
            start = 0
            for variant, (_, wrapper) in zip(variants, VARIANTS):
                length = variant["content_token_count"]
                if (length <= 0 or variant["slot_start"] != start or variant["slot_end"] != start + length
                        or variant["prefix_sha256"] != text_hash(wrapper) or variant["retention_policy"] != "content_positions_only"):
                    raise ValueError("V4.3 content-only variant spans drifted")
                start += length
            if start != count or len({v["content_token_count"] for v in variants}) != 1:
                raise ValueError("V4.3 variant slot coverage mismatch")
            for name, t in (("keys", keys[index, :, :count]), ("values", values[index, :, :count])):
                if tensor_sha(t) != entry[name + "_sha256"] or not math.isclose(rms(t), entry["key_rms" if name == "keys" else "value_rms"], rel_tol=1e-6):
                    raise ValueError("V4.3 tensor payload hash/RMS mismatch")
        self.entries = {e["bank_id"]: e for e in m["records"]}

    @property
    def bank_ids(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def get_memory(self, bank_id: str, *, device: str | torch.device, dtype: torch.dtype) -> SideKVMemory:
        if not bank_id.startswith("v43-bank-") or "::" in bank_id or bank_id not in self.entries:
            raise KeyError(f"Unknown unified Memory ID: {bank_id}")
        entry = self.entries[bank_id]
        i, count = entry["index"], entry["kv_valid_slot_count"]
        return SideKVMemory(bank_id, entry["payload_hash"],
            self.tensors["keys"][i, :, :count].to(device=device, dtype=dtype).clone(),
            self.tensors["values"][i, :, :count].to(device=device, dtype=dtype).clone(),
            self.tensors["slot_mask"][i, :count].to(device=device).clone(), 24, 0)
