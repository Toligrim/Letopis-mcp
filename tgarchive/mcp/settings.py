"""Small runtime configuration and logging helpers for the MCP process.

This module deliberately only reads configuration.  ``LETOPIS_MCP_CURSOR_SECRET``
remains owned by ``cursor.py`` and is intentionally not copied here.  The
concurrency and query-timeout settings are enforced by the retrieval layer;
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
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0
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


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc
    if value < minimum:
        raise SettingsError(f"{name} must be >= {minimum}")
    return value


def configured_database_path(root: Path | None = None) -> Path:
    """Return the DB path, giving ``LETOPIS_MCP_DB`` priority over config.toml."""
    resolved_root = root or project_root()
    raw_path = os.environ.get("LETOPIS_MCP_DB")
    if raw_path:
        path = Path(raw_path).expanduser()
        return path if path.is_absolute() else resolved_root / path

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
    raw_log_level = os.environ.get("LETOPIS_MCP_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if raw_log_level not in _VALID_LOG_LEVELS:
        raise SettingsError(f"LETOPIS_MCP_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}")

    return MCPSettings(
        db_path=configured_database_path(root),
        host=_loopback_host(),
        port=_env_int("LETOPIS_MCP_PORT", DEFAULT_PORT, minimum=1, maximum=65_535),
        log_level=raw_log_level,
        max_concurrency=_env_int("LETOPIS_MCP_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, minimum=1),
        query_timeout_seconds=_env_float(
            "LETOPIS_MCP_QUERY_TIMEOUT_SECONDS",
            DEFAULT_QUERY_TIMEOUT_SECONDS,
            minimum=0.0,
        ),
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
    )


class _JsonLogFormatter(logging.Formatter):
    """Serialize only controlled telemetry fields, never tool payloads."""

    _FIELDS = (
        "tool",
        "status",
        "latency_ms",
        "response_chars",
        "truncated",
        "total_hits",
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
