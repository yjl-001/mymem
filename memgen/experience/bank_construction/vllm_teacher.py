"""Loopback-only vLLM chat adapter; no dependency on a serving environment's torch."""
from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .artifacts import digest


def served_name(identity, dtype):
    # The launcher and client agree on exact weights, tokenizer revision and dtype.
    return "memgen-teacher-" + digest({"identity": identity, "dtype": dtype})[:24]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Teacher service must not redirect inference requests")


class VLLMTeacher:
    def __init__(self, store, config, identity):
        self.config = config
        self.base_url = config.teacher_base_url.rstrip("/")
        self.model_name = served_name(identity, config.teacher.dtype)
        models = self._request("/models")
        if self.model_name not in {r["id"] for r in models["data"]}:
            raise ValueError("vLLM model identity mismatch; start scripts/serve_local_bank_teacher.py with the same configuration")
        version = self._request("/version", root=True)
        store.put("teacher/service", {"backend": "vllm", "served_model": self.model_name,
            "version": version, "identity": identity}, {"base_url": self.base_url})

    def _request(self, path, payload=None, *, root=False):
        url = (self.base_url[:-3] if root else self.base_url) + path
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        for attempt in range(self.config.teacher_retries + 1):
            try:
                # One connection/opener per request: thread safe, ignores proxy environment,
                # refuses redirects to external inference services.
                opener = build_opener(ProxyHandler({}), NoRedirect())
                request = Request(url, data=body, headers={"Content-Type": "application/json"})
                with opener.open(request, timeout=self.config.teacher_timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as exc:
                exc.close()
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"vLLM HTTP {exc.code}; check server logs and request context budget") from exc
                error = exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                error = exc
            if attempt < self.config.teacher_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Local vLLM service unavailable after bounded retries") from error

    def chat(self, conversation, *, seed):
        cfg = self.config
        payload = {"model": self.model_name, "messages": conversation, "stream": False, "n": 1,
            "seed": seed, "max_tokens": cfg.teacher_max_new_tokens,
            "temperature": cfg.teacher_temperature, "top_p": cfg.teacher_top_p,
            "top_k": cfg.teacher_top_k if cfg.teacher_top_k else -1,
            "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.,
            "chat_template_kwargs": {"enable_thinking": False}}
        response = self._request("/chat/completions", payload)
        choices = response.get("choices", [])
        if response.get("model") != self.model_name or len(choices) != 1:
            raise ValueError("Unexpected vLLM model or choice count")
        choice = choices[0]
        finish, content = choice["finish_reason"], choice["message"].get("content")
        if finish not in {"stop", "length"} or not isinstance(content, str):
            raise ValueError("Unexpected vLLM completion termination/content")
        usage = response["usage"]
        return {"text": content, "truncated": finish == "length", "stop_reason": finish,
            "token_count": usage["completion_tokens"], "prompt_token_count": usage["prompt_tokens"],
            "seed": seed, "generation_backend": "vllm_chat", "request": payload, "response": response}

    def close(self):
        # The serving process belongs to the operator, and is not killed by the client.
        pass
