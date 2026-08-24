"""Small runtime configuration and logging helpers for the MCP process.

This module deliberately only reads configuration.  ``LETOPIS_MCP_CURSOR_SECRET``
remains owned by ``cursor.py``; startup policy is carried in ``MCPSettings``.
The concurrency and query-timeout settings are enforced by the retrieval layer;
rolling-limit settings are enforced by ``ratelimit.py``.  Deployment policy and
multi-principal identity remain outside this configuration module.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_config, project_root


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_QUERY_TIMEOUT_SECONDS = 15.0
DEFAULT_ARCHIVE_TIMEOUT_SECONDS = 3.0
DEFAULT_SEARCH_TIMEOUT_SECONDS = 8.0
DEFAULT_AGGREGATE_TIMEOUT_SECONDS = 10.0
DEFAULT_FETCH_TIMEOUT_SECONDS = 3.0
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 3.0
MAX_TOOL_TIMEOUT_SECONDS = 15.0
DEFAULT_ROLLING_CALLS_MAX = 60
DEFAULT_ROLLING_CHARS_MAX = 250_000
DEFAULT_ROLLING_WINDOW_SECONDS = 600

_LOGGER_NAME = "letopis_mcp"
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_HANDLER_MARKER = "_letopis_mcp_json_handler"


class SettingsError(ValueError):
    """Raised when an MCP runtime environment value is invalid."""


@dataclass(frozen=True, slots=True)
class MCPSettings:
    db_path: Path
    host: str
    port: int
    log_level: str
    max_concurrency: int
    query_timeout_seconds: float
    rolling_calls_max: int
    rolling_chars_max: int
    rolling_window_seconds: int
    dev_mode: bool = False
    archive_timeout_seconds: float | None = None
    search_timeout_seconds: float | None = None
    aggregate_timeout_seconds: float | None = None
    fetch_timeout_seconds: float | None = None
    context_timeout_seconds: float | None = None


def _env_int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise SettingsError(f"{name} must be in range {bound}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise SettingsError(f"{name} must be in range {bound}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean")


def configured_database_path(
    root: Path | None = None,
    *,
    dev_mode: bool | None = None,
) -> Path:
    """Return the explicit DB path, or use config.toml only in development."""
    if dev_mode is None:
        dev_mode = _env_bool("LETOPIS_MCP_DEV_MODE", False)
    raw_path = os.environ.get("LETOPIS_MCP_DB")
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        resolved_root = root or project_root()
        return resolved_root / path

    if not dev_mode:
        raise SettingsError(
            "LETOPIS_MCP_DB is required unless LETOPIS_MCP_DEV_MODE=true"
        )
    resolved_root = root or project_root()
    config = load_config(resolved_root)
    return resolved_root / config.get("general", {}).get("db", "data/index.db")


def _loopback_host() -> str:
    raw = os.environ.get("LETOPIS_MCP_HOST", DEFAULT_HOST)
    if raw == "localhost":
        return raw
    try:
        address = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise SettingsError("LETOPIS_MCP_HOST must be a loopback address") from exc
    if not address.is_loopback:
        # Non-loopback binding is intentionally a code/configuration review step,
        # not an env-only switch that could expose the archive accidentally.
        raise SettingsError("LETOPIS_MCP_HOST must be a loopback address")
    return raw


def load_settings(root: Path | None = None) -> MCPSettings:
    """Read the current MCP environment without loading .env or secrets."""
    dev_mode = _env_bool("LETOPIS_MCP_DEV_MODE", False)
    raw_log_level = os.environ.get("LETOPIS_MCP_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if raw_log_level not in _VALID_LOG_LEVELS:
        raise SettingsError(f"LETOPIS_MCP_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}")

    query_timeout = _env_float(
        "LETOPIS_MCP_QUERY_TIMEOUT_SECONDS",
        DEFAULT_QUERY_TIMEOUT_SECONDS,
        minimum=0.0,
        maximum=MAX_TOOL_TIMEOUT_SECONDS,
    )
    common_timeout_is_explicit = bool(os.environ.get("LETOPIS_MCP_QUERY_TIMEOUT_SECONDS"))

    def tool_timeout(name: str, default: float) -> float:
        fallback = query_timeout if common_timeout_is_explicit else default
        return _env_float(
            name,
            fallback,
            minimum=0.0,
            maximum=MAX_TOOL_TIMEOUT_SECONDS,
        )

    return MCPSettings(
        db_path=configured_database_path(root, dev_mode=dev_mode),
        host=_loopback_host(),
        port=_env_int("LETOPIS_MCP_PORT", DEFAULT_PORT, minimum=1, maximum=65_535),
        log_level=raw_log_level,
        max_concurrency=_env_int("LETOPIS_MCP_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, minimum=1),
        query_timeout_seconds=query_timeout,
        rolling_calls_max=_env_int(
            "LETOPIS_MCP_ROLLING_CALLS_MAX",
            DEFAULT_ROLLING_CALLS_MAX,
            minimum=1,
        ),
        rolling_chars_max=_env_int(
            "LETOPIS_MCP_ROLLING_CHARS_MAX",
            DEFAULT_ROLLING_CHARS_MAX,
            minimum=1,
        ),
        rolling_window_seconds=_env_int(
            "LETOPIS_MCP_ROLLING_WINDOW_SECONDS",
            DEFAULT_ROLLING_WINDOW_SECONDS,
            minimum=1,
        ),
        dev_mode=dev_mode,
        archive_timeout_seconds=tool_timeout(
            "LETOPIS_MCP_ARCHIVE_TIMEOUT_SECONDS",
            DEFAULT_ARCHIVE_TIMEOUT_SECONDS,
        ),
        search_timeout_seconds=tool_timeout(
            "LETOPIS_MCP_SEARCH_TIMEOUT_SECONDS",
            DEFAULT_SEARCH_TIMEOUT_SECONDS,
        ),
        aggregate_timeout_seconds=tool_timeout(
            "LETOPIS_MCP_AGGREGATE_TIMEOUT_SECONDS",
            DEFAULT_AGGREGATE_TIMEOUT_SECONDS,
        ),
        fetch_timeout_seconds=tool_timeout(
            "LETOPIS_MCP_FETCH_TIMEOUT_SECONDS",
            DEFAULT_FETCH_TIMEOUT_SECONDS,
        ),
        context_timeout_seconds=tool_timeout(
            "LETOPIS_MCP_CONTEXT_TIMEOUT_SECONDS",
            DEFAULT_CONTEXT_TIMEOUT_SECONDS,
        ),
    )


class _JsonLogFormatter(logging.Formatter):
    """Serialize only controlled telemetry fields, never tool payloads."""

    _FIELDS = (
        "tool",
        "status",
        "request_id",
        "query_fingerprint",
        "has_chat_filter",
        "has_topic_filter",
        "has_sender_filter",
        "has_date_filter",
        "has_media_filter",
        "cursor_used",
        "latency_ms",
        "response_chars",
        "returned_count",
        "truncated",
        "total_hits",
        "sql_time_ms",
        "candidate_pool_size",
        "error_code",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "event": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
        }
        for field in self._FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str) -> logging.Logger:
    """Configure the dedicated structured MCP logger and return it."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(_JsonLogFormatter())
        logger.addHandler(handler)
    return logger
