"""Batched generation, concurrent vLLM protocol, and safe rollout migration."""
from contextlib import redirect_stdout
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from memgen.experience.bank_construction.artifacts import Store, read_json
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.parallel import ordered_map
from memgen.experience.bank_construction.rollouts import run_rollouts
from memgen.experience.bank_construction.review import run_review
from memgen.experience.bank_construction.reuse import reuse_rollouts, PROMPT_FILES
from memgen.experience.bank_construction.teacher import Teacher
from memgen.experience.bank_construction.schemas import validate_review
from memgen.experience.bank_construction.vllm_teacher import VLLMTeacher, served_name
from tests.test_local_bank_construction import FixtureReasoner, fixture_split, review
from tests import test_local_bank_runtime as runtime_tests
from tests.test_v4_3_side_kv import torch


class ThroughputTests(unittest.TestCase):
    def test_config_and_gpu_launcher(self):
        from scripts.serve_local_bank_teacher import command
        cfg = ConstructionConfig()
        for changes in ({"rollout_batch_size": 0}, {"teacher_concurrency": 0},
                        {"teacher_backend": "cloud"}, {"teacher_base_url": "http://example.com/v1"},
                        {"teacher_base_url": "http://user:secret@localhost/v1"}):
            with self.assertRaises(ValueError):
                replace(cfg, **changes)
        identity = {"source": "Qwen/Qwen3-32B", "revision": "a" * 40}
        args = command(cfg, identity, gpu_count=4, port=8000, max_num_seqs=32,
                       max_model_len=32768, gpu_memory_utilization=.8)
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "4")
        self.assertIn(served_name(identity, cfg.teacher.dtype), args)
        self.assertEqual(args[args.index("--tokenizer-revision") + 1], "a" * 40)
        self.assertEqual(args[args.index("--generation-config") + 1], "vllm")

    def test_rollouts_batch_and_partial_resume(self):
        class Batched(FixtureReasoner):
            def __init__(self):
                super().__init__()
                self.sizes = []
            def generate_batch(self, requests, **kwargs):
                self.sizes.append(len(requests))
                return [self.generate(**r, **kwargs) for r in requests]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            cfg = replace(ConstructionConfig(), val_ratio=.34, rollout_batch_size=10)
            store = Store(tmp, {"fixture": "batch"})
            store.put("split", fixture_split(cfg), {})
            m = Batched()
            index = run_rollouts(store, cfg, lambda: m)
            self.assertEqual(m.sizes, [10, 10, 10, 2])
            removed = index["keys"][2:5]
            for key in removed:
                (store.root / (key + ".json")).unlink()
            m2 = Batched()
            self.assertEqual(index, run_rollouts(store, cfg, lambda: m2))
            self.assertEqual(m2.sizes, [3])
            self.assertEqual([c["seed"] for c in m2.calls], [c["seed"] for c in m.calls[2:5]])

    @unittest.skipIf(torch is None, "Torch required")
    def test_real_batch_independent_rng_and_greedy_left_padding(self):
        model = runtime_tests.RuntimeTests().local_model()
        self.addCleanup(model.close)
        requests = [dict(prompt="short", seed=10, sampling=False),
                    dict(prompt="a longer prompt here", seed=11, sampling=True),
                    dict(prompt="other", seed=12, sampling=True)]
        a = model.generate_batch(requests, max_new_tokens=4)
        b = model.generate_batch(list(reversed(requests)), max_new_tokens=4)
        self.assertEqual(a, list(reversed(b)))
        for req, result in zip(requests, a):
            single = model.generate_batch([req], max_new_tokens=4)[0]
            self.assertEqual(single, result)
        greedy = model.generate("short", seed=10, sampling=False, max_new_tokens=4, stop_on_box=True)
        self.assertEqual(a[0]["token_ids"], greedy["token_ids"])
        self.assertEqual([r["prompt_token_count"] for r in a], [1, 4, 1])
        self.assertTrue(all(r["token_count"] <= 4 for r in a))

    @unittest.skipIf(torch is None, "Torch required")
    def test_real_batch_eos_box_length_and_padding_accounting(self):
        model = runtime_tests.RuntimeTests().local_model()
        self.addCleanup(model.close)
        # Force three trajectories: EOS after one; boxed after two; unfinished at three.
        original_generate = model.model.generate
        def force_rows(*args, **kwargs):
            class Forced:
                def __call__(self, input_ids, scores):
                    scores.fill_(-torch.inf)
                    scores[0, 127] = 0
                    scores[1, 3] = 0
                    scores[2, 4] = 0
                    return scores
            kwargs["logits_processor"].append(Forced())
            return original_generate(*args, **kwargs)
        model.model.generate = force_rows
        model.tokenizer.decode = lambda ids, **kw: "done \\boxed{2}" if len(ids) >= 2 and ids[0] == 3 else "working"
        result = model.generate_batch([dict(prompt="q", seed=i, sampling=False) for i in range(3)], max_new_tokens=3)
        self.assertEqual([r["stop_reason"] for r in result], ["eos", "completed_boxed_answer", "length"])
        self.assertEqual([r["token_ids"] for r in result], [[127], [3, 3], [4, 4, 4]])
        self.assertEqual([r["truncated"] for r in result], [False, False, True])
        # A box completing exactly at the budget remains a natural completion.
        result = model.generate_batch([dict(prompt="q", seed=i, sampling=False) for i in range(3)], max_new_tokens=2)
        self.assertEqual(result[1]["stop_reason"], "completed_boxed_answer")

    def test_vllm_concurrency_payload_retries_receipts_and_resume(self):
        identity = {"source": "Qwen/Qwen3-32B", "revision": "a" * 40}
        cfg = replace(ConstructionConfig(), teacher_backend="vllm", teacher_concurrency=4, val_ratio=.34)
        name = served_name(identity, cfg.teacher.dtype)
        state = {"active": 0, "max": 0, "requests": [], "fail_once": True, "version": "fixture-vllm", "finish": "stop"}
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send(self, value, status=200):
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                self.send({"data": [{"id": name}]} if self.path == "/v1/models" else {"version": state["version"]})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    state["requests"].append(body)
                    fail = state["fail_once"]
                    state["fail_once"] = False
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(.02)
                with lock:
                    state["active"] -= 1
                if fail:
                    self.send({"error": "busy"}, 503)
                else:
                    self.send({"model": name, "choices": [{"message": {"content": json.dumps(review())},
                        "finish_reason": state["finish"]}], "usage": {"completion_tokens": 20, "prompt_tokens": 100}})
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        cfg = replace(cfg, teacher_base_url=f"http://127.0.0.1:{server.server_port}/v1")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(tmp, {"fixture": "http"})
            teacher = Teacher(store, cfg, lambda: VLLMTeacher(store, cfg, identity))
            payloads = [{"trajectory": str(i)} for i in range(8)]
            def ask(payload):
                return teacher.ask("review", payload, validate_review)
            answers = list(ordered_map(ask, payloads + payloads, 4))
            self.assertEqual(len(answers), 16)
            self.assertEqual(len(state["requests"]), 9)  # eight unique + one transport retry
            self.assertGreater(state["max"], 1)
            for body in state["requests"]:
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual((body["top_k"], body["temperature"], body["max_tokens"]), (20, .7, 8192))
            teacher.close()
            # All accepted requests resume without even connecting to the service.
            cached = Teacher(store, cfg, lambda: self.fail("No server access on cached resume"))
            self.assertEqual(cached.ask("review", payloads[0], validate_review), review())
            adapter = VLLMTeacher(store, cfg, identity)
            state["finish"] = "length"
            self.assertTrue(adapter.chat([{"role": "user", "content": "x"}], seed=3)["truncated"])
            state["version"] = "different-vllm"
            with self.assertRaisesRegex(ValueError, "Immutable"):
                VLLMTeacher(store, cfg, identity)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                VLLMTeacher(store, cfg, {**identity, "revision": "b" * 40})

    def test_migration_preserves_completed_rollouts_and_rejects_drift(self):
        cfg = replace(ConstructionConfig(), val_ratio=.34)
        profile = {"configuration": cfg.to_dict(), "reasoner": {"source": "r", "revision": "a" * 40},
                   "dataset": {"source": "d", "revision": "b" * 40}, "implementation": {k: "hash" for k in PROMPT_FILES}}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            source = Store(Path(tmp) / "old", profile)
            split = fixture_split(cfg)
            source.put("split", split, {})
            index = run_rollouts(source, cfg, FixtureReasoner)
            missing = source.root / (index["keys"][-1] + ".json")
            missing_bytes = missing.read_bytes()
            missing.unlink()  # An index claiming completion must not be partially imported.
            newcfg = replace(cfg, rollout_batch_size=64, teacher_backend="vllm")
            newprofile = {**profile, "configuration": newcfg.to_dict()}
            target = Store(Path(tmp) / "new", newprofile)
            target.put("split", split, {})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                reuse_rollouts(target, newcfg, newprofile, source.root)
            self.assertIsNone(target.get("imports/rollouts"))
            self.assertFalse((target.root / (index["keys"][0] + ".json")).exists())
            missing.write_bytes(missing_bytes)
            reuse_rollouts(target, newcfg, newprofile, source.root)
            self.assertEqual(target.require("imports/rollouts")["rollout_count"], 32)
            self.assertEqual(target.require(index["keys"][0])["generation"], source.require(index["keys"][0])["generation"])
            run_rollouts(target, newcfg, lambda: self.fail("Complete import must not generate new trajectories"))
            reuse_rollouts(target, newcfg, newprofile, source.root)  # idempotent
            with self.assertRaisesRegex(ValueError, "contract differs"):
                reuse_rollouts(target, replace(newcfg, sampling_seed=9), newprofile, source.root)
            corrupted = source.root / (index["keys"][0] + ".json")
            data = read_json(corrupted)
            data["payload"]["generation"]["text"] = "tampered"
            corrupted.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                reuse_rollouts(target, newcfg, newprofile, source.root)


if __name__ == "__main__":
    unittest.main()
