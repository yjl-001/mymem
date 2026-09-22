"""V5 native all-layer prefix KV compiler with immutable source binding."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def prefix_bank(directory, record, runtime, profile_sha256, *, validate_only=False):
    import torch
    from safetensors.torch import load_file, save_file
    from memgen.experience.v4_3_artifacts import atomic_json, read_json
    from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal
    from memgen.model.v4_3_prefix_equivalence import cache_tensors, memory_prefix_ids, validate_tensors

    directory = Path(directory)
    bank_id = record["bank_id"]
    if Path(bank_id).name != bank_id or not bank_id.startswith("v5-bank-"):
        raise ValueError("Unsafe V5 Bank ID")
    ids = memory_prefix_ids(runtime.tokenizer, record["descriptor"])
    tensor_path = directory / (bank_id + ".safetensors")
    manifest_path = directory / (bank_id + ".json")
    identity = {"schema_version": "memgen-v5-native-prefix-kv-v1",
        "profile_sha256": profile_sha256, "bank_id": bank_id,
        "source_record_sha256": record["record_sha256"], "prefix_token_ids": ids,
        "prefix_token_ids_sha256": canonical_hash(ids), "all_layers": True,
        "position_policy": "native_absolute_prefix_positions", "includes_role_wrapper": True,
        "question_independent": True, "value_source": "complete_memory_card"}
    if tensor_path.is_symlink() or manifest_path.is_symlink() or directory.is_symlink():
        raise ValueError("Refusing symlink V5 prefix cache")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        authenticate(manifest, "manifest_sha256", "V5 native prefix KV")
        if (any(manifest.get(key) != value for key, value in identity.items())
                or file_hash(tensor_path) != manifest["tensor_sha256"]):
            raise ValueError("V5 native prefix cache identity/hash drift")
        tensors = load_file(str(tensor_path))
    else:
        if validate_only:
            raise ValueError("Missing V5 native prefix cache manifest")
        runtime.controller.deactivate()
        tokens = runtime._tensor(ids)
        if not ids or len(ids) >= runtime.model.config.max_position_embeddings:
            raise ValueError("Invalid V5 memory prefix length")
        with torch.inference_mode():
            output = runtime.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                cache_position=torch.arange(len(ids), device=tokens.device), use_cache=True,
                return_dict=True)
        tensors = cache_tensors(output.past_key_values)
        directory.mkdir(parents=True, exist_ok=True)
        if tensor_path.exists():
            previous = load_file(str(tensor_path))
            if set(previous) != set(tensors) or any(not torch.equal(previous[key], tensors[key]) for key in tensors):
                raise ValueError("Interrupted V5 prefix cache differs from recomputation")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".v5-prefix-", dir=directory)
            os.close(descriptor)
            try:
                save_file(tensors, temporary)
                os.link(temporary, tensor_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        atomic_json(manifest_path, seal({**identity, "tensor_sha256": file_hash(tensor_path)},
                                       "manifest_sha256"), immutable=True)
    validate_tensors(tensors, len(ids), runtime.model)
    return ids, tensors
