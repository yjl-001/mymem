"""Frozen question encoder and unchanged native-prefix-KV consumption."""
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.model.side_kv import SideKVAttentionController
from memgen.model.v4_3_runtime import V43UnifiedRuntime
from memgen.model.v4_3_prefix_equivalence import restore_cache, split_prefix


def load_runtime(reasoner, device):
    tokenizer = AutoTokenizer.from_pretrained(reasoner["model_name"], revision=reasoner["tokenizer_revision"])
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(reasoner["model_name"], revision=reasoner["model_revision"],
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    if ((getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]) != reasoner["model_revision"]
            or (tokenizer.init_kwargs.get("_commit_hash") or reasoner["tokenizer_revision"]) != reasoner["tokenizer_revision"]):
        raise ValueError("Selector encoder/consumer revision drift")
    controller = SideKVAttentionController(model=model, layer_number=24)
    # Required by the shared decoder's contract. No gate probe or activation is used.
    gate = SimpleNamespace(config=SimpleNamespace(layer_number=24, risk_role="online_joint_control", rearm_low_entropy_token_count=2))
    return V43UnifiedRuntime(model=model, tokenizer=tokenizer, device=device, gate=gate, controller=controller)


@torch.inference_mode()
def encode_text(runtime, text):
    ids = runtime.tokenizer.encode(text.strip(), add_special_tokens=False)
    if not ids or len(ids) > runtime.model.config.max_position_embeddings:
        raise ValueError("Encoder text is empty or exceeds context; truncation is not permitted")
    runtime.controller.deactivate()
    x = runtime._tensor(ids)
    output = runtime.model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
    vector = output.last_hidden_state.float().mean(dim=1)[0]
    if not torch.isfinite(vector).all() or vector.norm() <= 0:
        raise ValueError("Invalid frozen question feature")
    return (vector / vector.norm()).cpu().tolist()


@torch.inference_mode()
def generate(runtime, question, descriptor=None, memory=None):
    runtime.controller.deactivate()
    if descriptor is None:
        if memory is not None:
            raise ValueError("No-memory branch cannot consume KV")
        prefix = runtime.visible_prefix(question, None)
        state = runtime.replay(prefix, len(prefix))
    else:
        memory_ids, tensors = memory
        prefix = split_prefix(runtime, question, descriptor, memory_ids)
        state = restore_cache(tensors, runtime.device)
        suffix = runtime._tensor(prefix[len(memory_ids):-1])
        state = runtime.model(input_ids=suffix,
                attention_mask=torch.ones((1, len(prefix)-1), dtype=torch.long, device=runtime.device),
                past_key_values=state, cache_position=torch.arange(len(memory_ids), len(prefix)-1, device=runtime.device),
                use_cache=True, return_dict=True).past_key_values
    result, _ = runtime.decode(prefix=prefix, prompt_count=len(prefix), cache=state, memory=None)
    return prefix, result
