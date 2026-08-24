from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

from tgarchive.db import bump_index_revision, connect, connect_readonly
from tgarchive.mcp import ratelimit, retrieval, tools
from tgarchive.indexer import _fts_set
from tgarchive.lemma import Lemmatizer
from tgarchive.mcp.models import (
    ErrorCode,
    ErrorResponse,
    SearchMessagesInput,
    SearchMessagesOutput,
)
from tgarchive.mcp.ratelimit import RollingRateLimiter


def _insert_searchable_message(writer, *, chat_id: int, message_id: int, text: str) -> None:
    cursor = writer.execute(
        "INSERT INTO messages(chat_id,message_id,date,text) VALUES(?,?,?,?)",
        (chat_id, message_id, "2026-01-03T00:00:00", text),
    )
    _fts_set(writer, Lemmatizer(), int(cursor.lastrowid), text)


def test_query_deadline_interrupts_expensive_sql(
    synthetic_archive,
    monkeypatch,
):
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", threading.BoundedSemaphore(1))
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", 0.001)

    def expensive_search(_request):
        with retrieval._readonly_connection(synthetic_archive.connection) as connection:
            connection.execute(
                "SELECT sum(a.id + b.id + c.id + d.id) "
                "FROM messages a, messages b, messages c, messages d"
            ).fetchone()
        raise AssertionError("the progress handler should interrupt this query")

    monkeypatch.setattr(tools.retrieval, "search_messages", expensive_search)
    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            limit=1,
            snippet_chars=120,
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code == ErrorCode.QUERY_TIMEOUT
    assert result.retryable is True


def test_db_semaphore_serializes_concurrent_tool_calls(
    synthetic_archive,
    monkeypatch,
):
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", semaphore)
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(retrieval, "_database_path", lambda: synthetic_archive.path)

    real_run_count = retrieval.run_count
    active = 0
    max_active = 0
    events: list[tuple[str, int]] = []
    state_lock = threading.Lock()

    def slow_run_count(*args, **kwargs):
        nonlocal active, max_active
        thread_id = threading.get_ident()
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            events.append(("enter", thread_id))
        time.sleep(0.15)
        try:
            return real_run_count(*args, **kwargs)
        finally:
            with state_lock:
                events.append(("exit", thread_id))
                active -= 1

    monkeypatch.setattr(retrieval, "run_count", slow_run_count)
    request = SearchMessagesInput(
        query="пагинация",
        strategy="relevance",
        limit=1,
        snippet_chars=120,
        include_total=True,
    )
    start_barrier = threading.Barrier(2)

    def invoke():
        start_barrier.wait()
        return tools.search_messages(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: invoke(), (1, 2)))

    assert all(isinstance(result, SearchMessagesOutput) for result in results)
    assert max_active == 1
    assert [event[0] for event in events] == ["enter", "exit", "enter", "exit"]


def test_wal_reader_remains_available_while_writer_transaction_is_open(
    synthetic_archive,
    monkeypatch,
):
    monkeypatch.setattr(retrieval, "_database_path", lambda: synthetic_archive.path)
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", 2.0)

    writer = connect(synthetic_archive.path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        first_id = synthetic_archive.messages[0]["db_id"]
        writer.execute("UPDATE messages SET text=text WHERE id=?", (first_id,))

        probe = connect_readonly(synthetic_archive.path)
        try:
            with retrieval._readonly_connection(probe) as connection:
                busy_timeout = connection.execute(
                    "PRAGMA busy_timeout"
                ).fetchone()[0]
            assert busy_timeout == retrieval.BUSY_TIMEOUT_MS
        finally:
            probe.close()

        result = tools.search_messages(
            SearchMessagesInput(
                query="пагинация",
                strategy="relevance",
                limit=1,
                snippet_chars=120,
            )
        )
        assert isinstance(result, SearchMessagesOutput)
    finally:
        writer.rollback()
        writer.close()


def test_search_uses_one_snapshot_when_writer_commits_between_count_and_search(
    synthetic_archive,
    monkeypatch,
):
    monkeypatch.setattr(retrieval, "_database_path", lambda: synthetic_archive.path)

    seed_writer = connect(synthetic_archive.path)
    try:
        _insert_searchable_message(
            seed_writer,
            chat_id=-1000000000900,
            message_id=1,
            text="уникальный снимок",
        )
        bump_index_revision(seed_writer)
        seed_writer.commit()
    finally:
        seed_writer.close()

    real_run_count = retrieval.run_count

    def count_then_commit(*args, **kwargs):
        total = real_run_count(*args, **kwargs)
        writer = connect(synthetic_archive.path)
        try:
            _insert_searchable_message(
                writer,
                chat_id=-1000000000900,
                message_id=2,
                text="уникальный снимок",
            )
            bump_index_revision(writer)
            writer.commit()
        finally:
            writer.close()
        return total

    monkeypatch.setattr(retrieval, "run_count", count_then_commit)
    result = retrieval.search_messages(
        SearchMessagesInput(
            query="снимок",
            strategy="relevance",
            limit=10,
            snippet_chars=120,
            include_total=True,
        )
    )

    assert result.total_hits == 1
    assert result.returned_hits == 1
    assert [hit.id for hit in result.hits] == ["tg:-1000000000900:1"]


def test_global_rolling_call_limit_rejects_before_db_callback(
    synthetic_archive,
    monkeypatch,
):
    limiter = RollingRateLimiter(
        calls_max=3,
        chars_max=1_000_000,
        window_seconds=600,
    )
    monkeypatch.setattr(ratelimit, "_LIMITER", limiter)
    monkeypatch.setattr(retrieval, "_database_path", lambda: synthetic_archive.path)

    real_search = retrieval.search_messages
    callback_calls = 0

    def counted_search(request):
        nonlocal callback_calls
        callback_calls += 1
        return real_search(request)

    monkeypatch.setattr(tools.retrieval, "search_messages", counted_search)
    request = SearchMessagesInput(
        query="пагинация",
        strategy="relevance",
        limit=1,
        snippet_chars=120,
    )
    results = [tools.search_messages(request) for _ in range(4)]

    assert all(isinstance(result, SearchMessagesOutput) for result in results[:3])
    assert isinstance(results[3], ErrorResponse)
    assert results[3].code == ErrorCode.RETRIEVAL_RATE_LIMITED
    assert results[3].retryable is True
    assert callback_calls == 3


def test_global_rolling_chars_limit_expires_with_sliding_window(monkeypatch):
    now = [100.0]
    limiter = RollingRateLimiter(
        calls_max=10,
        chars_max=1,
        window_seconds=10,
        clock=lambda: now[0],
    )
    monkeypatch.setattr(ratelimit, "_LIMITER", limiter)

    def callback():
        return SimpleNamespace(response_chars=1, truncated=False, total_hits=None)

    first = tools._handle_tool("synthetic", callback)
    blocked = tools._handle_tool("synthetic", callback)
    assert first.response_chars == 1
    assert isinstance(blocked, ErrorResponse)
    assert blocked.code == ErrorCode.RETRIEVAL_RATE_LIMITED

    now[0] = 111.0
    after_window = tools._handle_tool("synthetic", callback)
    assert after_window.response_chars == 1


def test_global_rolling_call_limit_admits_only_one_concurrent_call(
    synthetic_archive,
    monkeypatch,
):
    limiter = RollingRateLimiter(
        calls_max=1,
        chars_max=1_000_000,
        window_seconds=600,
    )
    monkeypatch.setattr(ratelimit, "_LIMITER", limiter)
    monkeypatch.setattr(retrieval, "_database_path", lambda: synthetic_archive.path)

    real_search = retrieval.search_messages
    callback_started = threading.Event()
    callback_calls = 0
    callback_lock = threading.Lock()

    def slow_search(request):
        nonlocal callback_calls
        with callback_lock:
            callback_calls += 1
        callback_started.set()
        time.sleep(0.1)
        return real_search(request)

    monkeypatch.setattr(tools.retrieval, "search_messages", slow_search)
    request = SearchMessagesInput(
        query="пагинация",
        strategy="relevance",
        limit=1,
        snippet_chars=120,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(tools.search_messages, request)
        assert callback_started.wait(timeout=1)
        second_future = executor.submit(tools.search_messages, request)
        first = first_future.result()
        second = second_future.result()

    results = [first, second]
    assert sum(isinstance(result, SearchMessagesOutput) for result in results) == 1
    errors = [result for result in results if isinstance(result, ErrorResponse)]
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.RETRIEVAL_RATE_LIMITED
    assert callback_calls == 1
