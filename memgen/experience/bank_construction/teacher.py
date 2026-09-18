"""Local teacher task runner with immutable request/response and bounded retries."""
from __future__ import annotations

import json
import threading

from .artifacts import digest
from .prompts import VERSION, messages


def parse_object(raw):
    value = raw.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate field: {key}")
            result[key] = item
        return result
    parsed = json.loads(value, object_pairs_hook=unique,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response must be one JSON object")
    return parsed


class Teacher:
    def __init__(self, store, config, model_factory):
        self.store, self.config, self.model_factory = store, config, model_factory
        self.model = None
        self._model_lock = threading.Lock()
        self._request_locks = [threading.Lock() for _ in range(256)]

    def prepare(self):
        with self._model_lock:
            if self.model is None:
                self.model = self.model_factory()

    def ask(self, task, payload, validate):
        # Identical trajectories can lead to identical teacher requests. Serialize
        # their immutable receipts, while independent requests run concurrently.
        with self._request_locks[int(digest([task, payload])[:8], 16) % 256]:
            return self._ask(task, payload, validate)

    def _ask(self, task, payload, validate):
        request = {"prompt_version": VERSION, "task": task, "messages": messages(task, payload)}
        key = "teacher/" + task + "/" + digest(request)
        self.store.put(key + "-request", request, request)
        cached = self.store.get(key, request)
        if cached is not None:
            validate(cached["answer"])
            return cached["answer"]
        error, attempt, new_attempts = None, 0, 0
        while True:
            conversation = list(request["messages"])
            if error:
                conversation.append({"role": "user", "content":
                    "The previous response failed output validation: " + error +
                    ". Return a complete corrected JSON object for the original task."})
            inputs = {"request": request, "attempt": attempt, "messages": conversation}
            raw_key = key + f"-attempt-{attempt}"
            generated = self.store.get(raw_key, inputs)
            if generated is None:
                if new_attempts >= self.config.teacher_retries + 1:
                    break
                self.prepare()
                seed = int(digest(inputs)[:8], 16)
                if self.config.teacher_backend == "vllm":
                    generated = self.model.chat(conversation, seed=seed)
                else:
                    prompt = self.model.tokenizer.apply_chat_template(conversation, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
                    generated = self.model.generate(prompt, seed=seed,
                        max_new_tokens=self.config.teacher_max_new_tokens, sampling=True,
                        temperature=self.config.teacher_temperature, top_p=self.config.teacher_top_p,
                        top_k=self.config.teacher_top_k)
                self.store.put(raw_key, generated, inputs)
                new_attempts += 1
            try:
                if generated["truncated"]:
                    raise ValueError("Teacher response was truncated")
                answer = parse_object(generated["text"])
                validate(answer)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                error = str(exc)
                print(f"[local-bank] teacher task={task} attempt={attempt + 1} invalid={error}", flush=True)
                attempt += 1
                continue
            self.store.put(key, {"answer": answer, "accepted_attempt": attempt, "raw_key": raw_key}, request)
            return answer
        raise RuntimeError(f"Teacher task {task} exhausted retries: {error}; receipts preserved at {key}")

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
