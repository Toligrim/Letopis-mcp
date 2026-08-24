from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from tgarchive.mcp import ratelimit, tools
from tgarchive.mcp.models import ErrorCode, ErrorResponse
from tgarchive.mcp.ratelimit import RollingRateLimiter


def _result(response_chars: int) -> SimpleNamespace:
    return SimpleNamespace(
        response_chars=response_chars,
        truncated=False,
        total_hits=None,
    )


def test_character_limit_rejects_posthoc_overflow(monkeypatch):
    limiter = RollingRateLimiter(
        calls_max=10,
        chars_max=100,
        window_seconds=600,
    )
    monkeypatch.setattr(ratelimit, "_LIMITER", limiter)

    first = tools._handle_tool("synthetic", lambda: _result(60))
    second = tools._handle_tool("synthetic", lambda: _result(60))

    assert first.response_chars == 60
    assert isinstance(second, ErrorResponse)
    assert second.code is ErrorCode.RETRIEVAL_RATE_LIMITED
    assert second.retryable is True
    assert limiter._chars == 60
    assert limiter._pending_calls == 0


def test_character_commit_is_atomic_for_concurrent_completions():
    limiter = RollingRateLimiter(
        calls_max=2,
        chars_max=100,
        window_seconds=600,
    )
    assert limiter.can_start() is True
    assert limiter.can_start() is True
    barrier = threading.Barrier(2)

    def commit() -> bool:
        barrier.wait()
        return limiter.commit_or_reject(60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: commit(), (1, 2)))

    assert sorted(results) == [False, True]
    assert limiter._chars == 60
    assert limiter._pending_calls == 0
