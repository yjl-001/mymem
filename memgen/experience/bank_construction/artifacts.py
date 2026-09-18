"""Atomic, content-bound checkpoints and an exclusive writer lock."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def atomic_json(path, value):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink output is not allowed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact differs: {path}; use a new run directory")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Store:
    def __init__(self, root, profile):
        if Path(root).is_symlink():
            raise ValueError("Run directory cannot be a symlink")
        self.root = Path(root).resolve()
        self.profile_hash = digest(profile)
        atomic_json(self.root / "profile.json", profile)

    def get(self, key, inputs=None):
        if any(part in {"", ".", ".."} for part in key.split("/")) or key.startswith("/"):
            raise ValueError("Unsafe checkpoint key")
        path = self.root / (key + ".json")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Symlink checkpoint is not allowed: {key}")
        if not path.exists():
            return None
        row = read_json(path)
        unsigned = {k: v for k, v in row.items() if k != "sha256"}
        if row.get("sha256") != digest(unsigned) or row.get("profile_sha256") != self.profile_hash:
            raise ValueError(f"Corrupt/stale checkpoint: {key}")
        if inputs is not None and row["inputs_sha256"] != digest(inputs):
            raise ValueError(f"Checkpoint input drift: {key}")
        return row["payload"]

    def put(self, key, payload, inputs):
        if any(part in {"", ".", ".."} for part in key.split("/")) or key.startswith("/"):
            raise ValueError("Unsafe checkpoint key")
        row = {"schema_version": "memgen-local-bank-checkpoint-v1", "profile_sha256": self.profile_hash,
               "inputs_sha256": digest(inputs), "payload": payload}
        atomic_json(self.root / (key + ".json"), {**row, "sha256": digest(row)})
        return payload

    def require(self, key):
        value = self.get(key)
        if value is None:
            raise ValueError(f"Missing prerequisite: {key}; run earlier stages first")
        return value


@contextmanager
def run_lock(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Run directory cannot be a symlink")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this construction run") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
