from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from tgarchive.mcp import cursor as cursor_module, ratelimit, retrieval
from tgarchive.mcp.server import main, parse_args
from tgarchive.mcp.settings import DiagnosticError, check_config


def test_parse_args_defaults_and_check_config_flag():
    default_parsed = parse_args([])
    assert default_parsed.check_config is False

    flag_parsed = parse_args(["--check-config"])
    assert flag_parsed.check_config is True


def test_check_config_succeeds_with_valid_database(synthetic_archive, monkeypatch):
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.setenv("LETOPIS_MCP_CURSOR_SECRET", "synthetic-test-cursor-secret")
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")

    summary = check_config()
    assert summary.db_path == synthetic_archive.path
    assert summary.dev_mode is False
    assert summary.cursor_secret_configured is True
    assert "Letopis MCP configuration check: OK" in summary.format_human()


def test_check_config_dev_mode_fallback_without_secret(synthetic_archive, monkeypatch):
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.delenv("LETOPIS_MCP_CURSOR_SECRET", raising=False)
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "true")

    summary = check_config()
    assert summary.dev_mode is True
    assert summary.cursor_secret_configured is False
    assert "not configured (allowed in dev mode)" in summary.format_human()


def test_check_config_cli_succeeds_and_does_not_start_server(
    synthetic_archive, monkeypatch, capsys
):
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.setenv("LETOPIS_MCP_CURSOR_SECRET", "synthetic-test-cursor-secret")
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")

    server_run_called = False

    def bogus_run(*args, **kwargs):
        nonlocal server_run_called
        server_run_called = True

    monkeypatch.setattr(MCPServer, "run", bogus_run)

    with pytest.raises(SystemExit) as exc_info:
        main(["--check-config"])

    assert exc_info.value.code == 0
    assert server_run_called is False

    captured = capsys.readouterr()
    assert "Letopis MCP configuration check: OK" in captured.out
    assert "Mode: production" in captured.out
    assert captured.err == ""


def test_production_missing_cursor_secret_exits_non_zero_without_traceback(
    synthetic_archive, monkeypatch, capsys
):
    secret_value = "super-secret-production-key-99"
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.delenv("LETOPIS_MCP_CURSOR_SECRET", raising=False)
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")

    with pytest.raises(DiagnosticError) as diag_err:
        check_config()
    assert "LETOPIS_MCP_CURSOR_SECRET is required in production mode" in str(diag_err.value)

    with pytest.raises(SystemExit) as exc_info:
        main(["--check-config"])

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Configuration check failed:" in captured.err
    assert "LETOPIS_MCP_CURSOR_SECRET is required" in captured.err
    assert "Traceback" not in captured.err
    assert secret_value not in captured.out
    assert secret_value not in captured.err


def test_missing_database_file_exits_non_zero_without_traceback(tmp_path: Path, monkeypatch, capsys):
    missing_db = tmp_path / "nonexistent.db"
    monkeypatch.setenv("LETOPIS_MCP_DB", str(missing_db))
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "true")

    with pytest.raises(SystemExit) as exc_info:
        main(["--check-config"])

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Configuration check failed:" in captured.err
    assert "Database path does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_unreadable_or_invalid_db_exits_non_zero_without_traceback(tmp_path: Path, monkeypatch, capsys):
    corrupt_db = tmp_path / "corrupt.db"
    corrupt_db.write_bytes(b"not a sqlite database file")
    monkeypatch.setenv("LETOPIS_MCP_DB", str(corrupt_db))
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "true")

    with pytest.raises(SystemExit) as exc_info:
        main(["--check-config"])

    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Configuration check failed:" in captured.err
    assert "Database probe failed" in captured.err or "Failed to open database" in captured.err
    assert "Traceback" not in captured.err


def test_secret_is_never_printed_in_stdout_or_stderr(synthetic_archive, monkeypatch, capsys):
    sensitive_secret = "TOP_SECRET_MCP_TOKEN_XYZ_12345"
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.setenv("LETOPIS_MCP_CURSOR_SECRET", sensitive_secret)
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")
    monkeypatch.setattr(cursor_module, "_CURSOR_CONFIG_MODE", None)
    monkeypatch.setattr(cursor_module, "_CURSOR_SECRET", None)

    with pytest.raises(SystemExit) as exc_info:
        main(["--check-config"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert sensitive_secret not in captured.out
    assert sensitive_secret not in captured.err
    assert "configured" in captured.out


def test_normal_no_arg_startup_behavior_preserved(synthetic_archive, monkeypatch):
    monkeypatch.setenv("LETOPIS_MCP_DB", str(synthetic_archive.path))
    monkeypatch.setenv("LETOPIS_MCP_CURSOR_SECRET", "synthetic-test-cursor-secret")
    monkeypatch.setenv("LETOPIS_MCP_DEV_MODE", "false")

    monkeypatch.setattr(retrieval, "_DB_PATH", None)
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", None)
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", None)
    monkeypatch.setattr(retrieval, "_RUNTIME_KEY", None)
    monkeypatch.setattr(ratelimit, "_LIMITER", None)
    monkeypatch.setattr(ratelimit, "_RUNTIME_KEY", None)

    run_args = {}

    def mock_run(self, transport, **kwargs):
        run_args["transport"] = transport
        run_args["kwargs"] = kwargs

    monkeypatch.setattr(MCPServer, "run", mock_run)

    main([])

    assert run_args.get("transport") == "streamable-http"
    assert run_args.get("kwargs", {}).get("host") == "127.0.0.1"
