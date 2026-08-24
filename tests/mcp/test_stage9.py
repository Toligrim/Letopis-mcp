from __future__ import annotations

import logging

import pytest

from tgarchive.mcp import cursor, retrieval
from tgarchive.mcp.models import ErrorCode, ToolError
from tgarchive.mcp.settings import SettingsError, load_settings


def _reset_cursor_configuration(monkeypatch):
    monkeypatch.setattr(cursor, "_CURSOR_SECRET", None)
    monkeypatch.setattr(cursor, "_CURSOR_CONFIG_MODE", None, raising=False)


def test_cursor_secret_is_required_without_dev_mode(monkeypatch):
    monkeypatch.delenv("LETOPIS_MCP_CURSOR_SECRET", raising=False)
    _reset_cursor_configuration(monkeypatch)

    with pytest.raises(SettingsError, match="LETOPIS_MCP_CURSOR_SECRET"):
        cursor.configure_cursor_secret(dev_mode=False)


def test_cursor_secret_uses_explicit_dev_fallback_with_warning(monkeypatch, caplog):
    monkeypatch.delenv("LETOPIS_MCP_CURSOR_SECRET", raising=False)
    _reset_cursor_configuration(monkeypatch)

    with caplog.at_level(logging.WARNING):
        cursor.configure_cursor_secret(dev_mode=True)

    assert isinstance(cursor._CURSOR_SECRET, bytes)
    assert len(cursor._CURSOR_SECRET) == 32
    assert any("restart" in record.getMessage().lower() for record in caplog.records)


def test_database_path_requires_explicit_env_without_dev_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("LETOPIS_MCP_DB", raising=False)
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")

    with pytest.raises(SettingsError, match="LETOPIS_MCP_DB"):
        load_settings(root=tmp_path)


def test_database_path_keeps_config_fallback_in_dev_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("LETOPIS_MCP_DB", raising=False)
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "true")

    settings = load_settings(root=tmp_path)

    assert settings.dev_mode is True
    assert settings.db_path == tmp_path / "data/index.db"


def test_default_tool_timeouts_are_differentiated(monkeypatch, tmp_path):
    monkeypatch.setenv("LETOPIS_MCP_DB", str(tmp_path / "archive.db"))
    monkeypatch.delenv("LETOPIS_MCP_ARCHIVE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LETOPIS_MCP_SEARCH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LETOPIS_MCP_AGGREGATE_TIMEOUT_SECONDS", raising=False)

    settings = load_settings(root=tmp_path)

    assert settings.archive_timeout_seconds == 3.0
    assert settings.fetch_timeout_seconds == 3.0
    assert settings.context_timeout_seconds == 3.0
    assert settings.search_timeout_seconds == 8.0
    assert settings.aggregate_timeout_seconds == 10.0

    monkeypatch.setenv("LETOPIS_MCP_SEARCH_TIMEOUT_SECONDS", "4.5")
    overridden = load_settings(root=tmp_path)
    assert overridden.search_timeout_seconds == 4.5


def test_semaphore_wait_and_query_share_one_absolute_deadline(
    synthetic_archive,
    monkeypatch,
):
    now = [100.0]
    acquire_timeouts: list[float] = []

    class FakeSemaphore:
        def acquire(self, timeout: float) -> bool:
            acquire_timeouts.append(timeout)
            now[0] += 0.2
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", FakeSemaphore())
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(retrieval, "_TOOL_TIMEOUTS", None, raising=False)
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: now[0])

    with pytest.raises(ToolError) as raised:
        with retrieval._readonly_connection(synthetic_archive.connection):
            pass

    assert raised.value.code == ErrorCode.QUERY_TIMEOUT
    assert acquire_timeouts == [pytest.approx(0.1)]
