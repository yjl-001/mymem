"""Bounded, ordered task execution; workers checkpoint results before returning."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque


def ordered_map(function, values, workers):
    if workers == 1:
        yield from map(function, values)
        return
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bank-teacher") as pool:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        try:
            while pending:
                result = pending.popleft().result()
                try:
                    pending.append(pool.submit(function, next(iterator)))
                except StopIteration:
                    pass
                yield result
        finally:
            for future in pending:
                future.cancel()


def teacher_workers(teacher):
    config = getattr(teacher, "config", None)
    return config.teacher_concurrency if config and config.teacher_backend == "vllm" else 1
