"""Immutable, resumable local teacher runner for V5 prompts."""
from __future__ import annotations

import json
import threading

from memgen.experience.bank_construction.artifacts import digest
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
                raise ValueError("Duplicate field: " + key)
            result[key] = item
        return result
    answer = json.loads(value, object_pairs_hook=unique,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(answer, dict):
        raise ValueError("Teacher response must be one JSON object")
    return answer


class V5Teacher:
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
        with self._request_locks[int(digest([task, payload])[:8], 16) % len(self._request_locks)]:
            request = {"prompt_version": VERSION, "task": task, "messages": messages(task, payload)}
            key = "teacher/" + task + "/" + digest(request)
            self.store.put(key + "-request", request, request)
            cached = self.store.get(key, request)
            if cached is not None:
                validate(cached["answer"])
                return cached["answer"]
            error, attempt, generated_now = None, 0, 0
            while True:
                conversation = list(request["messages"])
                if error:
                    conversation.append({"role": "user", "content":
                        "The previous response failed structural validation: " + error +
                        ". Return a complete corrected JSON object for the original task."})
                inputs = {"request": request, "attempt": attempt, "messages": conversation}
                raw_key = key + f"-attempt-{attempt}"
                generated = self.store.get(raw_key, inputs)
                if generated is None:
                    if generated_now >= self.config.teacher_retries + 1:
                        break
                    self.prepare()
                    seed = int(digest(inputs)[:8], 16)
                    if self.config.teacher_backend == "vllm":
                        generated = self.model.chat(conversation, seed=seed)
                    else:
                        prompt = self.model.tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        generated = self.model.generate(prompt, seed=seed,
                            max_new_tokens=self.config.teacher_max_new_tokens, sampling=True,
                            temperature=self.config.teacher_temperature, top_p=self.config.teacher_top_p,
                            top_k=self.config.teacher_top_k)
                    self.store.put(raw_key, generated, inputs)
                    generated_now += 1
                try:
                    if generated["truncated"]:
                        raise ValueError("Teacher response was truncated")
                    answer = parse_object(generated["text"])
                    validate(answer)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    error = str(exc)
                    print(f"[v5] teacher task={task} attempt={attempt + 1} invalid={error}", flush=True)
                    attempt += 1
                    continue
                self.store.put(key, {"answer": answer, "accepted_attempt": attempt,
                                     "raw_key": raw_key}, request)
                return answer
            raise RuntimeError(f"V5 teacher task {task} exhausted retries: {error}; receipts={key}")

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
