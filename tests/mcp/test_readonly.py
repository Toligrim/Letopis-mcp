from __future__ import annotations

import inspect
import sqlite3

import tgarchive.mcp.server as server_module
import tgarchive.mcp.tools as tools_module
import pytest


def test_readonly_connection_rejects_state_changing_operations(synthetic_archive):
    connection = synthetic_archive.connection
    first = synthetic_archive.messages[0]
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    # The fixture is already in WAL mode, so this exact command is an
    # idempotent read of the current mode rather than a state-changing write.
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == journal_mode
    before_count = connection.execute("SELECT count(*) FROM messages").fetchone()[0]
    before_text = connection.execute(
        "SELECT text FROM messages WHERE id=?", (first["db_id"],)
    ).fetchone()[0]

    mutations = [
        (
            "insert",
            "INSERT INTO messages(chat_id,message_id,date,text) VALUES(?,?,?,?)",
            (-999999, 999999, "2026-01-01T00:00:00", "must not be written"),
        ),
        (
            "update",
            "UPDATE messages SET text=? WHERE id=?",
            ("must not be written", first["db_id"]),
        ),
        (
            "delete",
            "DELETE FROM messages WHERE id=?",
            (first["db_id"],),
        ),
        (
            "journal_mode",
            "PRAGMA journal_mode=DELETE",
            (),
        ),
        (
            "user_version",
            "PRAGMA user_version=42",
            (),
        ),
    ]

    for _name, sql, params in mutations:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(sql, params).fetchall()

    after_count = connection.execute("SELECT count(*) FROM messages").fetchone()[0]
    after_text = connection.execute(
        "SELECT text FROM messages WHERE id=?", (first["db_id"],)
    ).fetchone()[0]
    assert after_count == before_count
    assert after_text == before_text


def test_mcp_surface_contains_only_retrieval_tools_and_no_write_api():
    public_tool_functions = {
        name
        for name, value in vars(tools_module).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert public_tool_functions == {
        "archive_overview",
        "search_messages",
        "aggregate_messages",
        "fetch_messages",
        "get_context",
    }

    forbidden_names = {
        "sync",
        "download",
        "index",
        "transcribe",
        "send_message",
        "delete_message",
        "execute_sql",
        "run_sql",
    }
    for module in (tools_module, server_module):
        public_names = {name for name in vars(module) if not name.startswith("_")}
        assert not public_names & forbidden_names
