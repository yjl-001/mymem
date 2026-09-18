"""Local HF inference and the existing all-layer native prefix consumer.

No remote inference client, side-attention hooks, gate, or trainable parameters.
Imports of the heavy ML stack are deferred until a model is needed.
"""
from __future__ import annotations

import gc


class InactiveController:
    """Adapter for the frozen decoder's inactive side-memory interface."""
    def __init__(self):
        self.traces = []

    def deactivate(self):
        pass

    def clear_traces(self):
        self.traces.clear()

    def activate(self, memory):
        raise RuntimeError("This runtime only consumes native prefix KV")


def native_runtime(model, tokenizer, device):
    from memgen.model.v4_3_runtime import V43UnifiedRuntime
    from memgen.model.e1_runtime import GreedyDecodingPolicy

    class NativeRuntime(V43UnifiedRuntime):
        def __init__(self):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.controller = InactiveController()
            self.decoding = GreedyDecodingPolicy(tokenizer=tokenizer, device=device)
    return NativeRuntime()


class LocalModel:
    def __init__(self, identity, settings, *, reasoner=False):
        if identity.get("files"):
            from dataclasses import replace
            from memgen.experience.bank_construction.sources import resolve_model
            if resolve_model(replace(settings, source=identity["source"])) != identity:
                raise ValueError("Local model files differ from the frozen run identity")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from memgen.chat_templates import CONVERSATION_TEMPLATE
        kwargs = {"revision": identity["revision"], "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(identity["source"], **kwargs)
        if reasoner:
            self.tokenizer.chat_template = CONVERSATION_TEMPLATE
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            identity["source"], **kwargs, torch_dtype=getattr(torch, settings.dtype),
            attn_implementation=settings.attention_backend,
            **({"device_map": "auto"} if settings.device == "auto" else {}))
        if settings.device != "auto":
            self.model.to(settings.device)
        self.model.eval().requires_grad_(False)
        self.device = self.model.get_input_embeddings().weight.device
        self.runtime = native_runtime(self.model, self.tokenizer, str(self.device)) if reasoner else None
        self.context_limit = self.model.config.max_position_embeddings

    def generate(self, prompt, *, seed, max_new_tokens, sampling, temperature=1., top_p=1., top_k=0,
                 stop_on_box=False):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        from memgen.model.v4_oracle import _complete_boxed_answer_seen
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) + max_new_tokens > self.context_limit:
            raise ValueError(f"Context overflow ({len(ids)} + {max_new_tokens} > {self.context_limit}); no truncation allowed")
        tokenizer, prompt_length = self.tokenizer, len(ids)

        class CompletedBox(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return _complete_boxed_answer_seen(tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True))

        generation = GenerationConfig(do_sample=sampling, max_new_tokens=max_new_tokens,
            num_beams=1, repetition_penalty=1., use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            **({"temperature": temperature, "top_p": top_p, "top_k": top_k} if sampling else {}))
        # One sequence per call: a sample's RNG stream does not depend on batching/resume order.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        x = torch.tensor([ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                generation_config=generation,
                stopping_criteria=StoppingCriteriaList([CompletedBox()]) if stop_on_box else None)
        tokens = out[0, len(ids):].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        eos = bool(tokens and tokens[-1] == self.tokenizer.eos_token_id)
        boxed = stop_on_box and _complete_boxed_answer_seen(text)
        stop = "eos" if eos else "completed_boxed_answer" if boxed else "length"
        if stop == "length" and len(tokens) != max_new_tokens:
            raise RuntimeError("Unexpected generation termination")
        return {"text": text, "token_ids": tokens, "token_count": len(tokens),
                "prompt_token_count": len(ids), "stop_reason": stop,
                "truncated": stop == "length", "seed": seed}

    def close(self):
        self.runtime = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
