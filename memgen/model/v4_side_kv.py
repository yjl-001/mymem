"""MI-style target/reference side-KV compilation for MemGen V4.

V3's compiler is frozen.  V4 therefore owns a separate artifact schema and
compiler that concatenates content-aligned slots from a raw descriptor and two
contextualized wrapper variants.  Target and reference roles are compiled and
authenticated together, while only target memories may be loaded online.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, text_sha256
from memgen.experience.v4_bank import (
    V4_BANK_MANIFEST_SCHEMA,
    V4_BANK_RECORD_SCHEMA,
    V4ConstructionProfile,
    V4_LAYER_NUMBER,
    V4_RELATIVE_PHASE_DELTA,
    parse_v4_process_card,
)
from memgen.experience.v4_1_bank import (
    V4_1_BANK_MANIFEST_SCHEMA,
    V41ConstructionProfile,
)
from memgen.experience.v4_2_local_direct import (
    V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
    V42LocalDirectProfile,
    local_direct_implementation_hashes,
)
from memgen.experience.v4_2_curated import (
    V4_2_CURATED_BANK_MANIFEST_SCHEMA,
    V42CuratedProfile,
    curated_implementation_hashes,
)
from memgen.model.side_kv import (
    DecoderLayerResolver,
    SideKVMemory,
    _require_sdpa,
    _tensor_rms,
)


V4_SIDE_KV_SCHEMA = "memgen-v4-side-kv-bank-v1"
V4_SIDE_KV_VARIANT_SCHEMA = "memgen-v4-side-kv-variant-v1"
V4_MEMORY_SCORE_NORMALIZATION = "log_valid_slots"
V4_MEMORY_TOTAL_PRIOR = 10.0
V4_MEMORY_SCORE_BIAS = math.log(V4_MEMORY_TOTAL_PRIOR)


def _v4_side_kv_implementation_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        "memgen/experience/v4_bank.py",
        "memgen/model/side_kv.py",
        "memgen/model/v4_side_kv.py",
        "scripts/compile_v4_side_kv.py",
    )
    return {relative: file_sha256(project_root / relative) for relative in paths}


def _v4_bank_implementation_hashes(
    schema_version: str = V4_BANK_MANIFEST_SCHEMA,
) -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    if schema_version == V4_BANK_MANIFEST_SCHEMA:
        paths = (
            "memgen/experience/v4_bank.py",
            "scripts/build_v4_repair_bank.py",
            "scripts/build_teacher_bank.py",
        )
    elif schema_version == V4_1_BANK_MANIFEST_SCHEMA:
        paths = (
            "memgen/experience/v4_bank.py",
            "memgen/experience/v4_1_bank.py",
            "scripts/build_v4_1_repair_bank.py",
            "scripts/build_teacher_bank.py",
        )
    elif schema_version == V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA:
        return local_direct_implementation_hashes(project_root)
    elif schema_version == V4_2_CURATED_BANK_MANIFEST_SCHEMA:
        return curated_implementation_hashes(project_root)
    else:
        raise ValueError("Unexpected tensor-free V4 bank manifest schema")
    return {relative: file_sha256(project_root / relative) for relative in paths}


@dataclass(frozen=True)
class V4DescriptorVariant:
    name: str
    prefix: str
    schema_version: str = V4_SIDE_KV_VARIANT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_SIDE_KV_VARIANT_SCHEMA:
            raise ValueError("Unexpected V4 descriptor-variant schema")
        if self.name not in {"raw_descriptor", "internal_principle", "hidden_note"}:
            raise ValueError("Unexpected V4 descriptor variant")
        if not isinstance(self.prefix, str):
            raise ValueError("V4 descriptor prefix must be a string")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


V4_DESCRIPTOR_VARIANTS = (
    V4DescriptorVariant(name="raw_descriptor", prefix=""),
    V4DescriptorVariant(
        name="internal_principle",
        prefix=(
            "<|im_start|>system\nUse the following reusable process as an "
            "internal reasoning principle.<|im_end|>\n<|im_start|>user\n"
        ),
    ),
    V4DescriptorVariant(
        name="hidden_note",
        prefix=(
            "<|im_start|>system\nTreat the following text as a hidden steering "
            "note for the reasoning process.<|im_end|>\n<|im_start|>user\n"
        ),
    ),
)


@dataclass(frozen=True)
class V4CompiledRoleMemory:
    bank_id: str
    role: str
    memory: SideKVMemory
    descriptor_sha256: str
    variants: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.role not in {"target", "reference"}:
            raise ValueError("Unexpected V4 compiled memory role")
        if self.role == "target" and self.memory.memory_id != self.bank_id:
            raise ValueError("V4 target memory ID must equal bank ID")
        if self.role == "reference" and self.memory.memory_id != f"{self.bank_id}::reference":
            raise ValueError("V4 reference memory ID is malformed")


@dataclass(frozen=True)
class V4CompiledSideKVBank:
    keys: torch.Tensor
    values: torch.Tensor
    slot_mask: torch.Tensor
    manifest: Mapping[str, Any]

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "v4_side_kv.safetensors"
        manifest_path = output_dir / "v4_side_kv_manifest.json"
        save_file(
            {
                "keys": self.keys.contiguous(),
                "values": self.values.contiguous(),
                "slot_mask": self.slot_mask.contiguous(),
            },
            str(tensor_path),
            metadata={
                "schema_version": V4_SIDE_KV_SCHEMA,
                "canonical_pre_rope": "true",
                "target_online_only": "true",
            },
        )
        manifest = dict(self.manifest)
        manifest["tensor_artifact"] = {
            "path": tensor_path.name,
            "sha256": file_sha256(tensor_path),
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return tensor_path, manifest_path


def validate_v4_tensor_free_manifest(value: Mapping[str, Any]) -> None:
    schema_version = value.get("schema_version")
    if schema_version not in {
        V4_BANK_MANIFEST_SCHEMA,
        V4_1_BANK_MANIFEST_SCHEMA,
        V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA,
        V4_2_CURATED_BANK_MANIFEST_SCHEMA,
    }:
        raise ValueError("Unexpected tensor-free V4 bank manifest schema")
    stored = value.get("manifest_sha256")
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if stored != canonical_json_sha256(logical):
        raise ValueError("Tensor-free V4 bank manifest hash mismatch")
    if value.get("qualified_for_online_use") is not False:
        raise ValueError("V4 tensor-free construction manifest has an invalid status")
    if value.get("status") != "constructed_not_tensor_compiled":
        raise ValueError("Unexpected V4 tensor-free construction status")
    if value.get("profile", {}).get("injection_layer") != V4_LAYER_NUMBER:
        raise ValueError("V4 tensor-free bank is not bound to layer 24")
    if schema_version == V4_BANK_MANIFEST_SCHEMA:
        profile = V4ConstructionProfile(**value.get("profile", {}))
    elif schema_version == V4_1_BANK_MANIFEST_SCHEMA:
        profile = V41ConstructionProfile(**value.get("profile", {}))
    elif schema_version == V4_2_LOCAL_DIRECT_BANK_MANIFEST_SCHEMA:
        profile = V42LocalDirectProfile(**value.get("profile", {}))
    else:
        profile = V42CuratedProfile(**value.get("profile", {}))
    if value.get("profile_sha256") != profile.profile_sha256:
        raise ValueError("V4 tensor-free construction profile hash mismatch")
    bank_ids = value.get("bank_ids")
    if (
        not isinstance(bank_ids, list)
        or not bank_ids
        or len(set(bank_ids)) != len(bank_ids)
        or value.get("record_count") != len(bank_ids)
        or value.get("record_order_sha256") != canonical_json_sha256(bank_ids)
        or set(value.get("record_sha256", {})) != set(bank_ids)
    ):
        raise ValueError("V4 tensor-free manifest record count mismatch")
    if (
        value.get("benchmark") != "openai/gsm8k"
        or value.get("auxiliary_banks_materialized") is not False
    ):
        raise ValueError("V4 tensor-free bank namespace or auxiliary policy drifted")
    if value.get("inputs", {}).get("repository", {}).get(
        "implementation_sha256"
    ) != _v4_bank_implementation_hashes(str(schema_version)):
        raise ValueError("V4 bank-construction implementation identity drifted")


class V4SideKVCompiler:
    """Compile raw plus contextualized target/reference descriptor slots."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        reasoner_name: str,
        reasoner_revision: str,
        tokenizer_revision: str,
        model_sequence_limit: int,
        layer_number: int = V4_LAYER_NUMBER,
        variants: Sequence[V4DescriptorVariant] = V4_DESCRIPTOR_VARIANTS,
    ) -> None:
        if layer_number != V4_LAYER_NUMBER:
            raise ValueError("V4 initial side-KV compiler is frozen at layer 24")
        if model_sequence_limit <= 0:
            raise ValueError("V4 model sequence limit must be positive")
        if tuple(variants) != V4_DESCRIPTOR_VARIANTS:
            raise ValueError("V4 initial descriptor variants are frozen")
        self.model = model
        self.tokenizer = tokenizer
        self.reasoner_name = reasoner_name
        self.reasoner_revision = reasoner_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_sequence_limit = model_sequence_limit
        self.layer_number = layer_number
        self.variants = tuple(variants)
        layers = DecoderLayerResolver.resolve(model)
        if layer_number > len(layers):
            raise ValueError("V4 layer 24 exceeds the reasoner depth")
        self.decoder_layer = layers[layer_number - 1]
        self.attention = getattr(self.decoder_layer, "self_attn", None)
        if self.attention is None:
            raise ValueError("V4 selected decoder block has no self_attn")
        _require_sdpa(self.attention, owner=type(self).__name__)
        if not hasattr(self.decoder_layer, "input_layernorm"):
            raise ValueError("V4 selected decoder block has no input_layernorm")
        for attribute in ("k_proj", "v_proj"):
            if not hasattr(self.attention, attribute):
                raise ValueError(f"V4 selected attention module has no {attribute}")

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise ValueError("V4 reasoner has no parameters") from exc

    def _role_memory_id(self, bank_id: str, role: str) -> str:
        return bank_id if role == "target" else f"{bank_id}::reference"

    @torch.inference_mode()
    def _compile_descriptor(
        self,
        *,
        bank_id: str,
        role: str,
        descriptor: str,
    ) -> V4CompiledRoleMemory:
        if role not in {"target", "reference"}:
            raise ValueError("Unexpected V4 descriptor role")
        content_ids = [
            int(value)
            for value in self.tokenizer.encode(descriptor, add_special_tokens=False)
        ]
        if not content_ids:
            raise ValueError("V4 process descriptor tokenized to zero content slots")
        compiled_keys: list[torch.Tensor] = []
        compiled_values: list[torch.Tensor] = []
        variant_records: list[dict[str, Any]] = []
        for variant in self.variants:
            prefix_ids = [
                int(value)
                for value in self.tokenizer.encode(
                    variant.prefix, add_special_tokens=False
                )
            ]
            if len(prefix_ids) + len(content_ids) > self.model_sequence_limit:
                raise ValueError(
                    f"V4 {bank_id}/{role}/{variant.name} exceeds model context"
                )
            input_ids = torch.tensor(
                [prefix_ids + content_ids], dtype=torch.long, device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            hidden_index = self.layer_number - 1
            if hidden_states is None or hidden_index >= len(hidden_states):
                raise RuntimeError("V4 reasoner did not expose layer input states")
            content_states = hidden_states[hidden_index][
                :, len(prefix_ids) : len(prefix_ids) + len(content_ids), :
            ]
            if int(content_states.shape[1]) != len(content_ids):
                raise RuntimeError("V4 content-aligned hidden-state span drifted")
            normalized = self.decoder_layer.input_layernorm(content_states)
            projected_keys = self.attention.k_proj(normalized)
            projected_values = self.attention.v_proj(normalized)
            num_kv_heads = int(self.model.config.num_key_value_heads)
            head_dim = int(
                getattr(
                    self.attention,
                    "head_dim",
                    self.model.config.hidden_size
                    // self.model.config.num_attention_heads,
                )
            )
            expected_width = num_kv_heads * head_dim
            if (
                projected_keys.shape[-1] != expected_width
                or projected_values.shape[-1] != expected_width
            ):
                raise RuntimeError("V4 K/V projection width mismatch")
            keys = (
                projected_keys.view(1, len(content_ids), num_kv_heads, head_dim)
                .transpose(1, 2)
                .squeeze(0)
                .detach()
                .cpu()
            )
            values = (
                projected_values.view(1, len(content_ids), num_kv_heads, head_dim)
                .transpose(1, 2)
                .squeeze(0)
                .detach()
                .cpu()
            )
            if not torch.isfinite(keys.float()).all() or not torch.isfinite(
                values.float()
            ).all():
                raise RuntimeError("V4 compiler produced non-finite K/V")
            compiled_keys.append(keys)
            compiled_values.append(values)
            variant_records.append(
                {
                    "name": variant.name,
                    "prefix_sha256": text_sha256(variant.prefix),
                    "prefix_token_count": len(prefix_ids),
                    "content_token_count": len(content_ids),
                    "retained_content_token_count": len(content_ids),
                    "retention_policy": "content_positions_only",
                }
            )
        keys = torch.cat(compiled_keys, dim=1)
        values = torch.cat(compiled_values, dim=1)
        memory_id = self._role_memory_id(bank_id, role)
        descriptor_sha256 = text_sha256(descriptor)
        memory = SideKVMemory(
            memory_id=memory_id,
            payload_hash=descriptor_sha256,
            keys=keys,
            values=values,
            slot_mask=torch.ones(keys.shape[1], dtype=torch.bool),
            layer_number=self.layer_number,
            relative_phase_delta=V4_RELATIVE_PHASE_DELTA,
        )
        return V4CompiledRoleMemory(
            bank_id=bank_id,
            role=role,
            memory=memory,
            descriptor_sha256=descriptor_sha256,
            variants=tuple(variant_records),
        )

    @torch.inference_mode()
    def compile(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        source_manifest: Mapping[str, Any],
        source_manifest_path: Path,
    ) -> V4CompiledSideKVBank:
        validate_v4_tensor_free_manifest(source_manifest)
        if not records:
            raise ValueError("Cannot compile an empty V4 side-KV bank")
        if len(records) != source_manifest.get("record_count"):
            raise ValueError("V4 records differ from construction manifest count")
        expected_ids = list(source_manifest["bank_ids"])
        actual_ids = [str(record.get("bank_id", "")) for record in records]
        if actual_ids != expected_ids:
            raise ValueError("V4 bank record order differs from construction manifest")
        for record in records:
            if record.get("schema_version") != V4_BANK_RECORD_SCHEMA:
                raise ValueError("Unexpected V4 bank-record schema")
            stored = record.get("record_sha256")
            logical = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            if stored != canonical_json_sha256(logical):
                raise ValueError("V4 bank-record hash mismatch")
            if source_manifest.get("record_sha256", {}).get(record["bank_id"]) != stored:
                raise ValueError("V4 source manifest record binding mismatch")
            if record.get("roles") != {
                "target_online_injectable": True,
                "reference_online_injectable": False,
                "auxiliary": None,
            }:
                raise ValueError("V4 bank-record role policy drifted")
            if record.get("compiler_contract") != {
                "layer_number": V4_LAYER_NUMBER,
                "all_kv_groups": True,
                "canonical_pre_rope": True,
                "relative_phase_delta": V4_RELATIVE_PHASE_DELTA,
                "attention_backend": "sdpa",
            }:
                raise ValueError("V4 bank-record compiler contract drifted")
            construction = record.get("construction", {})
            sample_ids = construction.get("sample_ids")
            if (
                not isinstance(sample_ids, list)
                or len(sample_ids) < 5
                or len(set(sample_ids)) != len(sample_ids)
                or construction.get("distinct_sample_count") != len(sample_ids)
            ):
                raise ValueError("V4 bank-record construction support drifted")

        was_training = bool(self.model.training)
        self.model.eval()
        compiled: list[V4CompiledRoleMemory] = []
        try:
            for record in records:
                cluster_key = str(record["cluster"]["cluster_key"])
                card = parse_v4_process_card(
                    record["process_card"], cluster_key=cluster_key
                )
                bank_id = str(record["bank_id"])
                compiled.append(
                    self._compile_descriptor(
                        bank_id=bank_id,
                        role="target",
                        descriptor=card.target.descriptor,
                    )
                )
                compiled.append(
                    self._compile_descriptor(
                        bank_id=bank_id,
                        role="reference",
                        descriptor=card.reference.descriptor,
                    )
                )
        finally:
            self.model.train(was_training)

        max_slots = max(item.memory.valid_slot_count for item in compiled)
        num_kv_heads = int(compiled[0].memory.keys.shape[0])
        head_dim = int(compiled[0].memory.keys.shape[-1])
        dtype = compiled[0].memory.keys.dtype
        keys = torch.zeros(
            (len(compiled), num_kv_heads, max_slots, head_dim),
            dtype=dtype,
            device="cpu",
        )
        values = torch.zeros_like(keys)
        slot_mask = torch.zeros((len(compiled), max_slots), dtype=torch.bool)
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(compiled):
            slots = item.memory.valid_slot_count
            keys[index, :, :slots, :] = item.memory.keys
            values[index, :, :slots, :] = item.memory.values
            slot_mask[index, :slots] = True
            entries.append(
                {
                    "index": index,
                    "bank_id": item.bank_id,
                    "role": item.role,
                    "memory_id": item.memory.memory_id,
                    "descriptor_sha256": item.descriptor_sha256,
                    "payload_hash": item.memory.payload_hash,
                    "kv_valid_slot_count": slots,
                    "variant_count": len(item.variants),
                    "variants": [dict(value) for value in item.variants],
                    "key_rms": _tensor_rms(item.memory.keys),
                    "value_rms": _tensor_rms(item.memory.values),
                    "online_injectable": item.role == "target",
                }
            )
        manifest = {
            "schema_version": V4_SIDE_KV_SCHEMA,
            "canonical_pre_rope": True,
            "relative_phase_delta": V4_RELATIVE_PHASE_DELTA,
            "layer_number": self.layer_number,
            "hf_decoder_block_index": self.layer_number - 1,
            "compiler_hidden_state_tuple_index": self.layer_number - 1,
            "all_kv_groups": True,
            "attention_backend": "sdpa",
            "memory_score_normalization": V4_MEMORY_SCORE_NORMALIZATION,
            "memory_total_prior": V4_MEMORY_TOTAL_PRIOR,
            "memory_score_bias": V4_MEMORY_SCORE_BIAS,
            "target_online_only": True,
            "auxiliary_banks_materialized": False,
            "reasoner": {
                "model_name": self.reasoner_name,
                "model_revision": self.reasoner_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "model_sequence_limit": self.model_sequence_limit,
            },
            "variants": [item.to_dict() for item in self.variants],
            "source": {
                "bank_manifest_path": str(source_manifest_path.resolve()),
                "bank_manifest_file_sha256": file_sha256(source_manifest_path),
                "bank_manifest_logical_sha256": source_manifest["manifest_sha256"],
                "record_order_sha256": source_manifest["record_order_sha256"],
            },
            "implementation_sha256": _v4_side_kv_implementation_hashes(),
            "bank_count": len(records),
            "record_count": len(entries),
            "records": entries,
            "record_order_sha256": canonical_json_sha256(
                [item.memory.memory_id for item in compiled]
            ),
            "tensor_shape": {
                "keys": list(keys.shape),
                "values": list(values.shape),
                "slot_mask": list(slot_mask.shape),
                "layout": "role_record,kv_head,slot,head_dim",
                "dtype": str(dtype),
            },
        }
        return V4CompiledSideKVBank(keys, values, slot_mask, manifest)


class V4SideKVBankLoader:
    """Authenticate a compiled V4 role bank and expose target memories only."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_reasoner_name: str | None = None,
        expected_reasoner_revision: str | None = None,
        expected_tokenizer_revision: str | None = None,
    ) -> None:
        from safetensors.torch import load_file

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_reasoner_name=expected_reasoner_name,
            expected_reasoner_revision=expected_reasoner_revision,
            expected_tokenizer_revision=expected_tokenizer_revision,
        )
        artifact = self.manifest.get("tensor_artifact", {})
        tensor_path = manifest_path.parent / str(artifact.get("path", ""))
        if not tensor_path.is_file() or file_sha256(tensor_path) != artifact.get("sha256"):
            raise ValueError("V4 side-KV tensor artifact is missing or corrupted")
        tensors = load_file(str(tensor_path), device="cpu")
        self.keys = tensors["keys"]
        self.values = tensors["values"]
        self.slot_mask = tensors["slot_mask"]
        self._validate_tensors()
        self._target_entry_by_bank_id = {
            str(entry["bank_id"]): entry
            for entry in self.manifest["records"]
            if entry.get("role") == "target"
        }
        if len(self._target_entry_by_bank_id) != self.manifest["bank_count"]:
            raise ValueError("V4 side-KV target role coverage mismatch")

    @property
    def bank_ids(self) -> tuple[str, ...]:
        """Return the authenticated target-bank namespace in manifest order."""

        return tuple(
            str(entry["bank_id"])
            for entry in self.manifest["records"]
            if entry.get("role") == "target"
        )

    def _validate_manifest(
        self,
        *,
        expected_reasoner_name: str | None,
        expected_reasoner_revision: str | None,
        expected_tokenizer_revision: str | None,
    ) -> None:
        if self.manifest.get("schema_version") != V4_SIDE_KV_SCHEMA:
            raise ValueError("Unexpected V4 side-KV manifest schema")
        stored = self.manifest.get("manifest_sha256")
        logical = {
            key: value
            for key, value in self.manifest.items()
            if key != "manifest_sha256"
        }
        if stored != canonical_json_sha256(logical):
            raise ValueError("V4 side-KV manifest hash mismatch")
        if self.manifest.get(
            "implementation_sha256"
        ) != _v4_side_kv_implementation_hashes():
            raise ValueError("V4 side-KV implementation identity drifted")
        if (
            self.manifest.get("canonical_pre_rope") is not True
            or self.manifest.get("relative_phase_delta") != 0
            or self.manifest.get("layer_number") != 24
            or self.manifest.get("all_kv_groups") is not True
        ):
            raise ValueError("V4 side-KV compiler contract drifted")
        if self.manifest.get("target_online_only") is not True:
            raise ValueError("V4 online role policy drifted")
        if self.manifest.get("memory_score_normalization") != "log_valid_slots":
            raise ValueError("V4 side-KV normalization drifted")
        if not math.isclose(
            float(self.manifest.get("memory_total_prior")),
            V4_MEMORY_TOTAL_PRIOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V4 side-KV total memory prior drifted")
        if not math.isclose(
            float(self.manifest.get("memory_score_bias")),
            V4_MEMORY_SCORE_BIAS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V4 side-KV memory prior drifted")
        reasoner = self.manifest.get("reasoner", {})
        for expected, field, owner in (
            (expected_reasoner_name, "model_name", "reasoner name"),
            (expected_reasoner_revision, "model_revision", "reasoner revision"),
            (expected_tokenizer_revision, "tokenizer_revision", "tokenizer revision"),
        ):
            if expected is not None and reasoner.get(field) != expected:
                raise ValueError(f"V4 side-KV {owner} mismatch")
        records = self.manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("V4 side-KV manifest has no records")
        if self.manifest.get("record_count") != len(records):
            raise ValueError("V4 side-KV record count mismatch")
        roles_by_bank: dict[str, set[str]] = {}
        expected_variants = [item.to_dict() for item in V4_DESCRIPTOR_VARIANTS]
        if self.manifest.get("variants") != expected_variants:
            raise ValueError("V4 side-KV descriptor variants drifted")
        for index, entry in enumerate(records):
            if entry.get("index") != index:
                raise ValueError("V4 side-KV record indices are not contiguous")
            roles_by_bank.setdefault(str(entry.get("bank_id")), set()).add(
                str(entry.get("role"))
            )
            if (entry.get("role") == "target") != bool(
                entry.get("online_injectable")
            ):
                raise ValueError("V4 side-KV online role flag mismatch")
            expected_memory_id = (
                str(entry.get("bank_id"))
                if entry.get("role") == "target"
                else f"{entry.get('bank_id')}::reference"
            )
            if entry.get("memory_id") != expected_memory_id:
                raise ValueError("V4 side-KV role memory ID mismatch")
            variants = entry.get("variants")
            if (
                entry.get("variant_count") != len(V4_DESCRIPTOR_VARIANTS)
                or not isinstance(variants, list)
                or [variant.get("name") for variant in variants]
                != [variant.name for variant in V4_DESCRIPTOR_VARIANTS]
                or any(
                    variant.get("retention_policy") != "content_positions_only"
                    or variant.get("content_token_count", 0) <= 0
                    or variant.get("retained_content_token_count")
                    != variant.get("content_token_count")
                    for variant in variants
                )
            ):
                raise ValueError("V4 side-KV role variant metadata drifted")
        if any(roles != {"target", "reference"} for roles in roles_by_bank.values()):
            raise ValueError("V4 side-KV bank lacks target/reference role pairing")
        if self.manifest.get("record_order_sha256") != canonical_json_sha256(
            [entry["memory_id"] for entry in records]
        ):
            raise ValueError("V4 side-KV record order hash mismatch")

    def _validate_tensors(self) -> None:
        if self.keys.ndim != 4 or self.values.shape != self.keys.shape:
            raise ValueError("V4 side-KV tensors must use four dimensions")
        if self.slot_mask.dtype != torch.bool or self.slot_mask.shape != (
            self.keys.shape[0],
            self.keys.shape[2],
        ):
            raise ValueError("V4 side-KV slot mask shape or dtype is invalid")
        if self.manifest.get("tensor_shape", {}).get("keys") != list(self.keys.shape):
            raise ValueError("V4 side-KV tensor shape differs from manifest")
        if not torch.isfinite(self.keys.float()).all() or not torch.isfinite(
            self.values.float()
        ).all():
            raise ValueError("V4 side-KV tensors contain non-finite values")
        for entry in self.manifest["records"]:
            index = int(entry["index"])
            slots = int(entry.get("kv_valid_slot_count", 0))
            if slots <= 0 or int(self.slot_mask[index].sum().item()) != slots:
                raise ValueError("V4 side-KV slot count differs from mask")
            expected = torch.arange(self.keys.shape[2]) < slots
            if not torch.equal(self.slot_mask[index].cpu(), expected):
                raise ValueError("V4 side-KV mask must be a contiguous prefix")

    def get_target(
        self,
        bank_id: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> SideKVMemory:
        entry = self._target_entry_by_bank_id.get(bank_id)
        if entry is None:
            raise KeyError(f"Unknown or non-target V4 bank ID: {bank_id}")
        index = int(entry["index"])
        slots = int(entry["kv_valid_slot_count"])
        return SideKVMemory(
            memory_id=bank_id,
            payload_hash=str(entry["payload_hash"]),
            keys=self.keys[index, :, :slots, :].to(device=device, dtype=dtype),
            values=self.values[index, :, :slots, :].to(device=device, dtype=dtype),
            slot_mask=self.slot_mask[index, :slots].to(device=device),
            layer_number=V4_LAYER_NUMBER,
            relative_phase_delta=V4_RELATIVE_PHASE_DELTA,
        )


__all__ = [
    "V4CompiledSideKVBank",
    "V4DescriptorVariant",
    "V4SideKVBankLoader",
    "V4SideKVCompiler",
    "V4_DESCRIPTOR_VARIANTS",
    "V4_MEMORY_SCORE_BIAS",
    "V4_MEMORY_SCORE_NORMALIZATION",
    "V4_MEMORY_TOTAL_PRIOR",
    "validate_v4_tensor_free_manifest",
]
