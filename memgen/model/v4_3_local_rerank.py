"""Local Qwen3-Reranker pair scoring using its official yes/no input protocol."""
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memgen.experience.v4_3_bank import canonical_hash, file_hash
from memgen.experience.v4_3_local_rerank import INSTRUCTION

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_REVISION = "5fa94080caafeaa45a15d11f969d7978e087a3db"
# Protocol documented at https://huggingface.co/Qwen/Qwen3-Reranker-8B
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def model_identity(source, revision):
    directory = Path(source)
    if directory.is_dir():
        files = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()
                 and p.suffix in {".json", ".safetensors", ".txt", ".model"}}
        if "config.json" not in files or not any(p.endswith(".safetensors") for p in files):
            raise ValueError("Local reranker requires a complete Transformers safetensors directory")
        return {"source": str(directory.resolve()), "revision": None, "local_file_sha256": files,
                "snapshot_sha256": canonical_hash(files)}
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Hub reranker revision must be an exact commit")
    return {"source": source, "revision": revision, "local_file_sha256": None}


class LocalReranker:
    def __init__(self, identity, device="cuda", max_length=8192):
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(identity["source"], revision=identity["revision"],
                                                       padding_side="left", trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(identity["source"], revision=identity["revision"],
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
            use_safetensors=True).to(self.device).eval()
        if self.model.config.model_type != "qwen3":
            raise ValueError("Expected Qwen3 reranker architecture")
        if identity["revision"] is not None:
            for commit in (getattr(self.model.config, "_commit_hash", None), self.tokenizer.init_kwargs.get("_commit_hash")):
                if commit is not None and commit != identity["revision"]:
                    raise ValueError("Reranker model/tokenizer revision drift")
        self.configure_protocol()

    def configure_protocol(self):
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        if (self.yes_id == self.no_id or self.yes_id is None or self.no_id is None
                or self.tokenizer.encode("yes", add_special_tokens=False) != [self.yes_id]
                or self.tokenizer.encode("no", add_special_tokens=False) != [self.no_id]):
            raise ValueError("Reranker requires distinct single-token yes/no labels")

    @torch.inference_mode()
    def score(self, question, descriptor):
        if not question.strip() or not descriptor.strip():
            raise ValueError("Empty reranker question/card")
        body = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {descriptor}"
        ids = self.prefix + self.tokenizer.encode(body, add_special_tokens=False) + self.suffix
        if len(ids) > min(self.max_length, self.model.config.max_position_embeddings):
            raise ValueError("Question+card exceeds reranker context; silent truncation is forbidden")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        logits = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False,
                            logits_to_keep=1, return_dict=True).logits[0, -1].float()
        margin = logits[self.yes_id] - logits[self.no_id]
        if not torch.isfinite(margin):
            raise ValueError("Nonfinite reranker logits")
        value = torch.sigmoid(margin).item()
        logit_margin = margin.item()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"score": value, "logit_margin": logit_margin, "input_tokens": len(ids),
                "elapsed_seconds": time.perf_counter()-start}
