"""Global sliding-window retrieval rate limiting.

Letopis v1 has no OAuth, ACL, or other caller identity from which to derive a
principal.  The limiter therefore intentionally protects one global process
window for the single personal ChatGPT/MCP deployment; it is not a per-user
limit and must not be mistaken for multi-principal isolation.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
import time
from typing import Callable

from .settings import MCPSettings, load_settings


class RollingRateLimiter:
    """Thread-safe rolling counters for completed retrieval calls.

    A call slot is reserved atomically during :meth:`can_start` and released
    by either :meth:`commit_or_reject` or :meth:`cancel_pending`.  The response
    character count is committed atomically with the limit check after the
    callback completes, so a completed response cannot push the rolling window
    over ``chars_max``.
    """

    def __init__(
        self,
        calls_max: int,
        chars_max: int,
        window_seconds: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if calls_max < 1 or chars_max < 1 or window_seconds < 1:
            raise ValueError("rolling rate limits must be positive")
        self.calls_max = calls_max
        self.chars_max = chars_max
        self.window_seconds = window_seconds
        self._clock = clock
        self._entries: deque[tuple[float, int]] = deque()
        self._chars = 0
        self._pending_calls = 0
        self._lock = Lock()

    def _purge(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0][0] <= cutoff:
            _, chars = self._entries.popleft()
            self._chars -= chars

    def can_start(self) -> bool:
        """Return whether a new call may start and reserve its call slot."""
        now = self._clock()
        with self._lock:
            self._purge(now)
            if (
                len(self._entries) + self._pending_calls >= self.calls_max
                or self._chars >= self.chars_max
            ):
                return False
            self._pending_calls += 1
            return True

    def cancel_pending(self) -> None:
        """Release a reservation for a callback that did not succeed."""
        now = self._clock()
        with self._lock:
            self._purge(now)
            if self._pending_calls > 0:
                self._pending_calls -= 1

    def commit_or_reject(self, response_chars: int) -> bool:
        """Atomically commit a completed response if its character budget fits."""
        if response_chars < 0:
            raise ValueError("response_chars must be non-negative")
        now = self._clock()
        with self._lock:
            self._purge(now)
            if self._pending_calls > 0:
                self._pending_calls -= 1
            if self._chars + response_chars > self.chars_max:
                return False
            self._entries.append((now, response_chars))
            self._chars += response_chars
            return True

    def record_completed(self, response_chars: int) -> None:
        """Compatibility wrapper for callers that do not inspect acceptance."""
        self.commit_or_reject(response_chars)


_CONFIG_LOCK = Lock()
_LIMITER: RollingRateLimiter | None = None
_RUNTIME_KEY: tuple[int, int, int] | None = None


def configure_runtime(settings: MCPSettings) -> None:
    """Install the process-wide limiter from startup runtime settings once."""
    global _LIMITER, _RUNTIME_KEY
    runtime_key = (
        settings.rolling_calls_max,
        settings.rolling_chars_max,
        settings.rolling_window_seconds,
    )
    with _CONFIG_LOCK:
        if _RUNTIME_KEY is not None:
            if runtime_key != _RUNTIME_KEY:
                raise RuntimeError("MCP rolling limiter is already configured")
            return
        _LIMITER = RollingRateLimiter(*runtime_key)
        _RUNTIME_KEY = runtime_key


def _get_limiter() -> RollingRateLimiter:
    global _LIMITER, _RUNTIME_KEY
    if _LIMITER is None:
        with _CONFIG_LOCK:
            if _LIMITER is None:
                settings = load_settings()
                _LIMITER = RollingRateLimiter(
                    settings.rolling_calls_max,
                    settings.rolling_chars_max,
                    settings.rolling_window_seconds,
                )
                _RUNTIME_KEY = (
                    settings.rolling_calls_max,
                    settings.rolling_chars_max,
                    settings.rolling_window_seconds,
                )
    assert _LIMITER is not None
    return _LIMITER


def can_start() -> bool:
    return _get_limiter().can_start()


def record_completed(response_chars: int) -> None:
    _get_limiter().record_completed(response_chars)


def commit_or_reject(response_chars: int) -> bool:
    return _get_limiter().commit_or_reject(response_chars)


def cancel_pending() -> None:
    _get_limiter().cancel_pending()
