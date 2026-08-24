from __future__ import annotations

import json
import logging
import threading

from tgarchive.mcp import ratelimit, retrieval, tools
from tgarchive.mcp.models import (
    AggregateMessagesInput,
    AggregateMessagesOutput,
    SearchMessagesInput,
    SearchMessagesOutput,
    SearchFilters,
)
from tgarchive.mcp.ratelimit import RollingRateLimiter
from tgarchive.mcp.settings import _JsonLogFormatter


def test_tool_logs_contain_bounded_call_metadata(synthetic_archive, monkeypatch):
    monkeypatch.setattr(retrieval, "_DB_PATH", synthetic_archive.path)
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", threading.BoundedSemaphore(4))
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(retrieval, "_CONFIGURED_QUERY_TIMEOUT_SECONDS", None)
    monkeypatch.setattr(retrieval, "_TOOL_TIMEOUTS", None)
    monkeypatch.setattr(
        ratelimit,
        "_LIMITER",
        RollingRateLimiter(calls_max=20, chars_max=1_000_000, window_seconds=600),
    )

    logger = logging.getLogger("letopis_mcp")
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler()
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        first = tools.search_messages(
            SearchMessagesInput(
                query="пагинация",
                strategy="relevance",
                limit=1,
                snippet_chars=120,
            )
        )
        assert isinstance(first, SearchMessagesOutput)
        assert first.next_cursor

        second = tools.search_messages(
            SearchMessagesInput(
                query="пагинация",
                strategy="relevance",
                limit=1,
                snippet_chars=120,
                cursor=first.next_cursor,
            )
        )
        assert isinstance(second, SearchMessagesOutput)

        aggregate = tools.aggregate_messages(
            AggregateMessagesInput(
                query="пагинация",
                group_by="chat",
                limit=2,
                filters=SearchFilters(
                    chat_ids=[-1000000000001],
                    date_from="2023",
                ),
            )
        )
        assert isinstance(aggregate, AggregateMessagesOutput)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    tool_records = [record for record in records if record.getMessage() == "tool_call"]
    assert len(tool_records) == 3
    assert all(record.request_id for record in tool_records)
    assert len({record.request_id for record in tool_records}) == 3
    assert tool_records[0].returned_count == len(first.hits)
    assert tool_records[1].returned_count == len(second.hits)
    assert tool_records[2].returned_count == len(aggregate.groups)
    assert tool_records[0].cursor_used is False
    assert tool_records[1].cursor_used is True
    assert all(len(record.query_fingerprint) == 12 for record in tool_records)
    assert all(isinstance(record.sql_time_ms, float) for record in tool_records)
    assert all(isinstance(record.candidate_pool_size, int) for record in tool_records[:2])
    assert tool_records[0].has_chat_filter is False
    assert tool_records[1].has_chat_filter is False
    assert tool_records[2].has_chat_filter is True
    assert tool_records[2].has_date_filter is True
    assert not hasattr(tool_records[2], "candidate_pool_size")
    assert all("query" not in record.__dict__ for record in tool_records)
    assert all("filters" not in record.__dict__ for record in tool_records)

    payload = json.loads(_JsonLogFormatter().format(tool_records[0]))
    assert {
        "request_id",
        "query_fingerprint",
        "returned_count",
        "cursor_used",
        "sql_time_ms",
        "candidate_pool_size",
    } <= payload.keys()
    assert "пагинация" not in json.dumps(payload, ensure_ascii=False)
    assert "1000000000001" not in json.dumps(payload, ensure_ascii=False)
    assert "2023" not in json.dumps(payload, ensure_ascii=False)
    assert not hasattr(first, "sql_time_ms")
