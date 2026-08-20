"""Canonical pre-RoPE side-KV compilation and cache-safe Qwen integration.

The classes in this module intentionally do not modify MemGen's Weaver path.
They provide a separate, training-free E0/E1 mechanism:

* :class:`CanonicalSideKVCompiler` converts sanitized ``MemoryRecord`` text to
  layer-local pre-RoPE K and V slots with the frozen reasoner's own projections.
* :class:`SideKVAttentionController` temporarily augments one Qwen2 attention
  layer while leaving the native Hugging Face cache and token positions intact.

The first version is deliberately restricted to eager attention and batch size
one.  Those restrictions make attention mass and cache invariants inspectable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import types
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import canonical_json_sha256, file_sha256


SIDE_KV_BANK_SCHEMA = "canonical-side-kv-bank-v1"
SIDE_KV_TRACE_SCHEMA = "side-kv-attention-trace-v1"
DEFAULT_COMPILER_PREFIX = (
    "<|im_start|>system\n"
    "Read the following reusable reasoning guideline as internal guidance."
    "<|im_end|>\n<|im_start|>user\n"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecoderLayerResolver:
    """Resolve common Hugging Face decoder layouts without model-name checks."""

    @staticmethod
    def resolve(model: Any) -> Sequence[Any]:
        candidates = (
            getattr(getattr(model, "model", None), "layers", None),
            getattr(getattr(model, "transformer", None), "h", None),
            getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        )
        for layers in candidates:
            if layers is not None:
                return layers
        raise ValueError("Unable to locate decoder layers on the reasoner")


@dataclass(frozen=True)
class SideKVCompilerConfig:
    """Frozen model-side compilation contract."""

    layer_number: int = 24
    template_prefix: str = DEFAULT_COMPILER_PREFIX
    relative_phase_delta: int = 0

    def __post_init__(self) -> None:
        if self.layer_number <= 0:
            raise ValueError("layer_number must be positive")
        if self.relative_phase_delta != 0:
            raise ValueError("E0-v1 fixes relative_phase_delta to zero")
        if not self.template_prefix:
            raise ValueError("template_prefix must not be empty")


@dataclass(frozen=True)
class SideKVMemory:
    """One runtime memory, stored per KV head in canonical coordinates."""

    memory_id: str
    payload_hash: str
    keys: torch.Tensor
    values: torch.Tensor
    slot_mask: torch.Tensor
    layer_number: int = 24
    relative_phase_delta: int = 0

    @property
    def valid_slot_count(self) -> int:
        return int(self.slot_mask.to(dtype=torch.int64).sum().item())


@dataclass(frozen=True)
class SideKVAttentionTrace:
    """Per-forward proof that the active memory participated in attention."""

    memory_id: str
    layer_number: int
    query_length: int
    native_key_length: int
    memory_slot_count: int
    memory_attention_mass: float
    native_attention_mass: float
    canonical_rope_score_relative_error: float | None
    memory_mass_by_query_head: tuple[float, ...]
    memory_mass_by_kv_group: tuple[float, ...]
    schema_version: str = SIDE_KV_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledSideKVBank:
    """Padded tensor bank plus its content-addressed manifest."""

    keys: torch.Tensor
    values: torch.Tensor
    slot_mask: torch.Tensor
    manifest: Mapping[str, Any]

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        from safetensors.torch import save_file

        output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = output_dir / "side_kv_bank.safetensors"
        manifest_path = output_dir / "side_kv_manifest.json"
        save_file(
            {
                "keys": self.keys.contiguous(),
                "values": self.values.contiguous(),
                "slot_mask": self.slot_mask.contiguous(),
            },
            str(tensor_path),
            metadata={
                "schema_version": SIDE_KV_BANK_SCHEMA,
                "canonical_pre_rope": "true",
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


class CanonicalSideKVCompiler:
    """Compile sanitized payload tokens with one frozen decoder block."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        reasoner_name: str,
        reasoner_revision: str,
        tokenizer_revision: str,
        config: SideKVCompilerConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reasoner_name = reasoner_name
        self.reasoner_revision = reasoner_revision
        self.tokenizer_revision = tokenizer_revision
        self.config = config or SideKVCompilerConfig()
        self.decoder_layers = DecoderLayerResolver.resolve(model)
        if self.config.layer_number > len(self.decoder_layers):
            raise ValueError(
                f"Requested layer {self.config.layer_number} exceeds "
                f"{len(self.decoder_layers)} decoder blocks"
            )
        self.decoder_layer = self.decoder_layers[self.config.layer_number - 1]
        self.attention = getattr(self.decoder_layer, "self_attn", None)
        if self.attention is None:
            raise ValueError("Selected decoder block has no self_attn module")
        for attribute in ("input_layernorm",):
            if not hasattr(self.decoder_layer, attribute):
                raise ValueError(f"Selected decoder block has no {attribute}")
        for attribute in ("k_proj", "v_proj"):
            if not hasattr(self.attention, attribute):
                raise ValueError(f"Selected attention module has no {attribute}")

    @torch.inference_mode()
    def compile(self, records: Sequence[MemoryRecord]) -> CompiledSideKVBank:
        if not records:
            raise ValueError("Cannot compile an empty memory bank")
        if any(record.kv_layer != self.config.layer_number for record in records):
            raise ValueError("MemoryRecord kv_layer differs from compiler layer")
        if any(record.reasoner_name != self.reasoner_name for record in records):
            raise ValueError("MemoryRecord reasoner_name differs from compiler")
        if any(record.reasoner_revision != self.reasoner_revision for record in records):
            raise ValueError("MemoryRecord reasoner_revision differs from compiler")
        if any(record.tokenizer_revision != self.tokenizer_revision for record in records):
            raise ValueError("MemoryRecord tokenizer_revision differs from compiler")

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            compiled = [self._compile_one(record) for record in records]
        finally:
            self.model.train(was_training)

        max_slots = max(item.keys.shape[-2] for item in compiled)
        num_kv_heads = compiled[0].keys.shape[0]
        head_dim = compiled[0].keys.shape[-1]
        dtype = compiled[0].keys.dtype
        if any(
            item.keys.shape[0] != num_kv_heads
            or item.keys.shape[-1] != head_dim
            or item.keys.dtype != dtype
            for item in compiled
        ):
            raise RuntimeError("Compiled memory tensors have inconsistent shapes or dtypes")

        keys = torch.zeros(
            (len(compiled), num_kv_heads, max_slots, head_dim),
            dtype=dtype,
            device="cpu",
        )
        values = torch.zeros_like(keys)
        slot_mask = torch.zeros((len(compiled), max_slots), dtype=torch.bool)
        record_entries: list[dict[str, Any]] = []
        for index, item in enumerate(compiled):
            slots = item.keys.shape[-2]
            keys[index, :, :slots, :] = item.keys
            values[index, :, :slots, :] = item.values
            slot_mask[index, :slots] = True
            record_entries.append(
                {
                    "index": index,
                    "memory_id": item.memory_id,
                    "payload_hash": item.payload_hash,
                    "payload_token_count": slots,
                    "kv_valid_slot_count": slots,
                    "key_rms": _tensor_rms(item.keys),
                    "value_rms": _tensor_rms(item.values),
                }
            )

        manifest = {
            "schema_version": SIDE_KV_BANK_SCHEMA,
            "created_at": utc_now(),
            "canonical_pre_rope": True,
            "relative_phase_delta": self.config.relative_phase_delta,
            "layer_number": self.config.layer_number,
            "hf_decoder_block_index": self.config.layer_number - 1,
            "compiler_hidden_state_tuple_index": self.config.layer_number - 1,
            "risk_hidden_state_tuple_index": self.config.layer_number,
            "reasoner": {
                "model_name": self.reasoner_name,
                "model_revision": self.reasoner_revision,
                "tokenizer_revision": self.tokenizer_revision,
            },
            "compiler": asdict(self.config),
            "tensor_shape": {
                "keys": list(keys.shape),
                "values": list(values.shape),
                "slot_mask": list(slot_mask.shape),
                "layout": "record,kv_head,slot,head_dim",
                "dtype": str(dtype),
            },
            "record_count": len(records),
            "records": record_entries,
            "record_order_sha256": canonical_json_sha256(
                [record.memory_id for record in records]
            ),
        }
        return CompiledSideKVBank(keys, values, slot_mask, manifest)

    def _compile_one(self, record: MemoryRecord) -> SideKVMemory:
        prefix_ids = list(
            self.tokenizer.encode(
                self.config.template_prefix,
                add_special_tokens=False,
            )
        )
        payload_ids = list(
            self.tokenizer.encode(
                record.sanitized_contrast_payload,
                add_special_tokens=False,
            )
        )
        if len(payload_ids) != record.token_count:
            raise ValueError(
                f"Tokenizer drift for {record.memory_id}: "
                f"record={record.token_count}, compiler={len(payload_ids)}"
            )
        if canonical_json_sha256([int(value) for value in payload_ids]) != record.token_ids_sha256:
            raise ValueError(f"Tokenizer token-id hash drift for {record.memory_id}")
        if len(prefix_ids) + len(payload_ids) > record.model_sequence_limit:
            raise ValueError(
                f"Compiled input exceeds model sequence limit for {record.memory_id}: "
                f"input={len(prefix_ids) + len(payload_ids)}, "
                f"limit={record.model_sequence_limit}"
            )
        input_ids = torch.tensor(
            [prefix_ids + payload_ids],
            dtype=torch.long,
            device=self.model.device,
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
        hidden_index = self.config.layer_number - 1
        if hidden_states is None or hidden_index >= len(hidden_states):
            raise RuntimeError("Reasoner did not expose the selected layer input states")
        payload_states = hidden_states[hidden_index][:, len(prefix_ids) :, :]
        if payload_states.shape[1] != len(payload_ids):
            raise RuntimeError("Compiler payload span does not match tokenized payload")
        normalized = self.decoder_layer.input_layernorm(payload_states)
        key_states = self.attention.k_proj(normalized)
        value_states = self.attention.v_proj(normalized)
        num_kv_heads = int(self.model.config.num_key_value_heads)
        head_dim = int(
            getattr(
                self.attention,
                "head_dim",
                self.model.config.hidden_size // self.model.config.num_attention_heads,
            )
        )
        expected_width = num_kv_heads * head_dim
        if key_states.shape[-1] != expected_width or value_states.shape[-1] != expected_width:
            raise RuntimeError("K/V projection width does not match grouped-query configuration")
        keys = (
            key_states.view(1, len(payload_ids), num_kv_heads, head_dim)
            .transpose(1, 2)
            .squeeze(0)
            .detach()
            .cpu()
        )
        values = (
            value_states.view(1, len(payload_ids), num_kv_heads, head_dim)
            .transpose(1, 2)
            .squeeze(0)
            .detach()
            .cpu()
        )
        if not torch.isfinite(keys.float()).all() or not torch.isfinite(values.float()).all():
            raise RuntimeError(f"Non-finite compiled K/V for {record.memory_id}")
        return SideKVMemory(
            memory_id=record.memory_id,
            payload_hash=record.payload_hash,
            keys=keys,
            values=values,
            slot_mask=torch.ones(len(payload_ids), dtype=torch.bool),
            layer_number=self.config.layer_number,
            relative_phase_delta=self.config.relative_phase_delta,
        )


class SideKVBankLoader:
    """Revision-aware access to a saved side-KV tensor bank."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_reasoner_name: str | None = None,
        expected_reasoner_revision: str | None = None,
        expected_tokenizer_revision: str | None = None,
    ):
        from safetensors.torch import load_file

        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_reasoner_name=expected_reasoner_name,
            expected_reasoner_revision=expected_reasoner_revision,
            expected_tokenizer_revision=expected_tokenizer_revision,
        )
        tensor_info = self.manifest.get("tensor_artifact", {})
        tensor_path = manifest_path.parent / str(tensor_info.get("path", ""))
        if not tensor_path.is_file() or file_sha256(tensor_path) != tensor_info.get("sha256"):
            raise ValueError("Side-KV tensor artifact is missing or has a hash mismatch")
        tensors = load_file(str(tensor_path), device="cpu")
        self.keys = tensors["keys"]
        self.values = tensors["values"]
        self.slot_mask = tensors["slot_mask"]
        self._validate_tensors()
        self._entry_by_id = {
            str(entry["memory_id"]): entry for entry in self.manifest["records"]
        }

    def get(
        self,
        memory_id: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> SideKVMemory:
        entry = self._entry_by_id.get(memory_id)
        if entry is None:
            raise KeyError(f"Unknown side-KV memory_id: {memory_id}")
        index = int(entry["index"])
        slots = int(entry["kv_valid_slot_count"])
        return SideKVMemory(
            memory_id=memory_id,
            payload_hash=str(entry["payload_hash"]),
            keys=self.keys[index, :, :slots, :].to(device=device, dtype=dtype),
            values=self.values[index, :, :slots, :].to(device=device, dtype=dtype),
            slot_mask=self.slot_mask[index, :slots].to(device=device),
            layer_number=int(self.manifest["layer_number"]),
            relative_phase_delta=int(self.manifest["relative_phase_delta"]),
        )

    def _validate_manifest(
        self,
        *,
        expected_reasoner_name: str | None,
        expected_reasoner_revision: str | None,
        expected_tokenizer_revision: str | None,
    ) -> None:
        if self.manifest.get("schema_version") != SIDE_KV_BANK_SCHEMA:
            raise ValueError("Unexpected side-KV manifest schema")
        expected_hash = self.manifest.get("manifest_sha256")
        actual_hash = canonical_json_sha256(
            {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
        )
        if expected_hash != actual_hash:
            raise ValueError("Side-KV manifest hash mismatch")
        if self.manifest.get("canonical_pre_rope") is not True:
            raise ValueError("Side-KV bank is not canonical pre-RoPE")
        if self.manifest.get("relative_phase_delta") != 0:
            raise ValueError("E0-v1 side-KV manifest must use relative phase delta zero")
        records = self.manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Side-KV manifest has no records")
        if self.manifest.get("record_count") != len(records):
            raise ValueError("Side-KV manifest record_count mismatch")
        memory_ids = [str(entry.get("memory_id", "")) for entry in records]
        if any(not memory_id for memory_id in memory_ids) or len(set(memory_ids)) != len(
            memory_ids
        ):
            raise ValueError("Side-KV manifest memory IDs are missing or duplicated")
        if [int(entry.get("index", -1)) for entry in records] != list(range(len(records))):
            raise ValueError("Side-KV manifest record indices are not contiguous")
        if self.manifest.get("record_order_sha256") != canonical_json_sha256(
            memory_ids
        ):
            raise ValueError("Side-KV manifest record order hash mismatch")
        reasoner = self.manifest.get("reasoner", {})
        if (
            expected_reasoner_name is not None
            and reasoner.get("model_name") != expected_reasoner_name
        ):
            raise ValueError("Side-KV reasoner name mismatch")
        if (
            expected_reasoner_revision is not None
            and reasoner.get("model_revision") != expected_reasoner_revision
        ):
            raise ValueError("Side-KV reasoner revision mismatch")
        if (
            expected_tokenizer_revision is not None
            and reasoner.get("tokenizer_revision") != expected_tokenizer_revision
        ):
            raise ValueError("Side-KV tokenizer revision mismatch")

    def _validate_tensors(self) -> None:
        if self.keys.ndim != 4 or self.values.shape != self.keys.shape:
            raise ValueError("Side-KV bank tensors must use [record,kv_head,slot,head_dim]")
        if self.slot_mask.ndim != 2 or self.slot_mask.shape != (
            self.keys.shape[0],
            self.keys.shape[2],
        ):
            raise ValueError("Side-KV bank slot mask shape mismatch")
        if self.slot_mask.dtype != torch.bool:
            raise ValueError("Side-KV bank slot mask must be boolean")
        expected_shapes = self.manifest.get("tensor_shape", {})
        for name, tensor in (
            ("keys", self.keys),
            ("values", self.values),
            ("slot_mask", self.slot_mask),
        ):
            if expected_shapes.get(name) != list(tensor.shape):
                raise ValueError(f"Side-KV {name} shape differs from manifest")
        if expected_shapes.get("dtype") != str(self.keys.dtype):
            raise ValueError("Side-KV tensor dtype differs from manifest")
        if self.keys.shape[0] != self.manifest.get("record_count"):
            raise ValueError("Side-KV tensor record axis differs from manifest")
        if not torch.isfinite(self.keys.float()).all() or not torch.isfinite(
            self.values.float()
        ).all():
            raise ValueError("Side-KV tensor artifact contains non-finite values")
        for entry in self.manifest["records"]:
            index = int(entry["index"])
            slots = int(entry.get("kv_valid_slot_count", 0))
            if slots <= 0 or slots > self.keys.shape[2]:
                raise ValueError("Side-KV manifest has an invalid slot count")
            if int(entry.get("payload_token_count", -1)) != slots:
                raise ValueError("Side-KV payload token count differs from slot count")
            if not str(entry.get("payload_hash", "")):
                raise ValueError("Side-KV manifest record is missing payload hash")
            if int(self.slot_mask[index].sum().item()) != slots:
                raise ValueError("Side-KV slot mask differs from manifest")
            expected_mask = torch.arange(self.keys.shape[2]) < slots
            if not torch.equal(self.slot_mask[index].cpu(), expected_mask):
                raise ValueError("Side-KV slot mask must be a contiguous valid prefix")


class SideKVAttentionController:
    """Attach one persistent memory to one eager Qwen2 attention layer.

    When no memory is active, the original module forward is called unchanged.
    With a memory active, only native token K/V are passed to ``past_key_value``;
    canonical memory K/V are consumed by a side path and jointly normalized with
    native attention logits.
    """

    def __init__(
        self,
        *,
        model: Any,
        layer_number: int = 24,
        require_batch_size_one: bool = True,
        audit_canonical_rope: bool = False,
    ):
        self.model = model
        self.layer_number = layer_number
        self.require_batch_size_one = require_batch_size_one
        self.audit_canonical_rope = audit_canonical_rope
        layers = DecoderLayerResolver.resolve(model)
        if layer_number <= 0 or layer_number > len(layers):
            raise ValueError("Side-KV layer_number is outside the decoder")
        self.module = getattr(layers[layer_number - 1], "self_attn", None)
        if self.module is None:
            raise ValueError("Selected decoder block has no self_attn")
        self._validate_attention_protocol()
        implementation = getattr(getattr(self.module, "config", None), "_attn_implementation", None)
        if implementation != "eager":
            raise ValueError("SideKVAttentionController requires eager attention")
        self._original_forward = self.module.forward
        self._active_memory: SideKVMemory | None = None
        self._traces: list[SideKVAttentionTrace] = []
        self._closed = False
        controller = self

        def patched_forward(module_self: Any, *args: Any, **kwargs: Any):
            return controller._forward(module_self, *args, **kwargs)

        self.module.forward = types.MethodType(patched_forward, self.module)

    def _validate_attention_protocol(self) -> None:
        required_attributes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "head_dim",
            "num_key_value_groups",
            "scaling",
            "layer_idx",
            "config",
        )
        missing = [name for name in required_attributes if not hasattr(self.module, name)]
        if missing:
            raise ValueError(f"Selected attention module lacks side-KV protocol: {missing}")
        parameters = inspect.signature(self.module.forward).parameters
        required_parameters = {
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_value",
            "cache_position",
        }
        if not required_parameters.issubset(parameters):
            raise ValueError(
                "Selected attention forward signature is incompatible with side-KV"
            )

    @property
    def traces(self) -> tuple[SideKVAttentionTrace, ...]:
        return tuple(self._traces)

    def clear_traces(self) -> None:
        self._traces.clear()

    def activate(self, memory: SideKVMemory) -> None:
        if self._closed:
            raise RuntimeError("Cannot activate a closed SideKVAttentionController")
        if memory.layer_number != self.layer_number:
            raise ValueError("Runtime memory layer differs from controller layer")
        if memory.relative_phase_delta != 0:
            raise ValueError("E0-v1 only supports canonical delta=0 memories")
        if memory.keys.ndim != 3 or memory.values.shape != memory.keys.shape:
            raise ValueError("Side-KV tensors must have shape [kv_head, slot, head_dim]")
        if memory.slot_mask.ndim != 1 or memory.slot_mask.shape[0] != memory.keys.shape[-2]:
            raise ValueError("Side-KV slot mask does not match the slot dimension")
        if memory.valid_slot_count <= 0:
            raise ValueError("Side-KV memory has no valid slots")
        self._active_memory = memory

    def deactivate(self) -> None:
        self._active_memory = None

    @contextmanager
    def use_memory(self, memory: SideKVMemory) -> Iterator["SideKVAttentionController"]:
        if self._active_memory is not None:
            raise RuntimeError("A side-KV memory is already active")
        self.activate(memory)
        try:
            yield self
        finally:
            self.deactivate()

    def close(self) -> None:
        if self._closed:
            return
        self.deactivate()
        self.module.forward = self._original_forward
        self._closed = True

    def _forward(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self._active_memory
        if memory is None:
            return self._original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        if module.training:
            raise RuntimeError("Side-KV E0/E1 integration is inference-only")
        if position_embeddings is None:
            raise ValueError("Side-KV Qwen integration requires position_embeddings")
        batch_size, query_length, _ = hidden_states.shape
        if self.require_batch_size_one and batch_size != 1:
            raise ValueError("Side-KV E0/E1 integration requires batch size one")

        head_dim = int(module.head_dim)
        num_query_heads = int(module.config.num_attention_heads)
        num_kv_heads = int(module.config.num_key_value_heads)
        num_kv_groups = int(module.num_key_value_groups)
        query_pre = module.q_proj(hidden_states).view(
            batch_size, query_length, num_query_heads, head_dim
        ).transpose(1, 2)
        key_pre = module.k_proj(hidden_states).view(
            batch_size, query_length, num_kv_heads, head_dim
        ).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(
            batch_size, query_length, num_kv_heads, head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query_rotated, key_rotated = apply_rotary_pos_emb(
            query_pre,
            key_pre,
            cos,
            sin,
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_rotated, value_states = past_key_value.update(
                key_rotated,
                value_states,
                module.layer_idx,
                cache_kwargs,
            )

        native_keys = repeat_kv(key_rotated, num_kv_groups)
        native_values = repeat_kv(value_states, num_kv_groups)
        native_scores = torch.matmul(
            query_rotated,
            native_keys.transpose(2, 3),
        ) * float(module.scaling)
        if attention_mask is not None:
            native_scores = native_scores + attention_mask[:, :, :, : native_keys.shape[-2]]

        memory_keys = memory.keys.to(device=query_pre.device, dtype=query_pre.dtype)
        memory_values = memory.values.to(device=query_pre.device, dtype=query_pre.dtype)
        if memory_keys.shape[0] != num_kv_heads or memory_keys.shape[-1] != head_dim:
            raise ValueError("Runtime side-KV shape does not match the selected Qwen layer")
        expanded_memory_keys = repeat_kv(memory_keys.unsqueeze(0), num_kv_groups).expand(
            batch_size, -1, -1, -1
        )
        expanded_memory_values = repeat_kv(
            memory_values.unsqueeze(0), num_kv_groups
        ).expand(batch_size, -1, -1, -1)
        memory_scores = canonical_memory_scores(
            query_pre,
            expanded_memory_keys,
            scaling=float(module.scaling),
        )
        rope_score_relative_error = (
            shared_rope_score_relative_error(
                query_pre_rope=query_pre,
                canonical_memory_keys=expanded_memory_keys,
                cos=cos,
                sin=sin,
                scaling=float(module.scaling),
            )
            if self.audit_canonical_rope
            else None
        )
        slot_mask = memory.slot_mask.to(device=query_pre.device, dtype=torch.bool)
        memory_scores = memory_scores.masked_fill(
            ~slot_mask.view(1, 1, 1, -1),
            torch.finfo(memory_scores.dtype).min,
        )

        joint_scores = torch.cat([native_scores, memory_scores], dim=-1)
        joint_weights = F.softmax(joint_scores, dim=-1, dtype=torch.float32).to(
            query_pre.dtype
        )
        joint_values = torch.cat([native_values, expanded_memory_values], dim=-2)
        attention_output = torch.matmul(joint_weights, joint_values)
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, -1
        )
        attention_output = module.o_proj(attention_output)

        native_length = native_keys.shape[-2]
        memory_weights = joint_weights[..., native_length:]
        memory_mass = memory_weights.sum(dim=-1)
        per_query_head = memory_mass.mean(dim=(0, 2)).float()
        per_kv_group = (
            memory_mass.view(
                batch_size,
                num_kv_heads,
                num_kv_groups,
                query_length,
            )
            .mean(dim=(0, 2, 3))
            .float()
        )
        mean_memory_mass = float(memory_mass.float().mean().item())
        if not math.isfinite(mean_memory_mass) or mean_memory_mass <= 0.0:
            raise RuntimeError("Side-KV attention mass is non-finite or degenerate")
        self._traces.append(
            SideKVAttentionTrace(
                memory_id=memory.memory_id,
                layer_number=self.layer_number,
                query_length=query_length,
                native_key_length=native_length,
                memory_slot_count=memory.valid_slot_count,
                memory_attention_mass=mean_memory_mass,
                native_attention_mass=1.0 - mean_memory_mass,
                canonical_rope_score_relative_error=rope_score_relative_error,
                memory_mass_by_query_head=tuple(float(value) for value in per_query_head.tolist()),
                memory_mass_by_kv_group=tuple(float(value) for value in per_kv_group.tolist()),
            )
        )
        return attention_output, joint_weights


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Qwen/Llama half rotation, kept local to make the canonical test explicit."""

    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    """Expand one KV group to the query heads that share it."""

    batch, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    if repetitions == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        repetitions,
        sequence_length,
        head_dim,
    )
    return expanded.reshape(
        batch,
        num_kv_heads * repetitions,
        sequence_length,
        head_dim,
    )


def canonical_memory_scores(
    query_pre_rope: torch.Tensor,
    canonical_memory_keys: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """Position-independent delta=0 memory score used by E0-v1."""

    return torch.matmul(
        query_pre_rope,
        canonical_memory_keys.transpose(2, 3),
    ) * scaling


def shared_rope_score_relative_error(
    *,
    query_pre_rope: torch.Tensor,
    canonical_memory_keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scaling: float,
) -> float:
    """Compare canonical scores with Q/K rotated by the same live phase."""

    query = query_pre_rope.float()
    keys = canonical_memory_keys.float()
    cos_float = cos.float().unsqueeze(1)
    sin_float = sin.float().unsqueeze(1)
    query_rotated = query * cos_float + rotate_half(query) * sin_float
    keys_by_query = keys.unsqueeze(2)
    cos_by_query = cos_float.unsqueeze(3)
    sin_by_query = sin_float.unsqueeze(3)
    keys_rotated = (
        keys_by_query * cos_by_query
        + rotate_half(keys_by_query) * sin_by_query
    )
    shared_phase_scores = torch.einsum(
        "bhqd,bhqmd->bhqm", query_rotated, keys_rotated
    ) * scaling
    canonical_scores = canonical_memory_scores(
        query,
        keys,
        scaling=scaling,
    )
    max_difference = (canonical_scores - shared_phase_scores).abs().max()
    scale = canonical_scores.abs().max().clamp_min(1.0)
    return float((max_difference / scale).item())


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())
