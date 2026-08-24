"""Read-only retrieval mechanics for the basic MCP tools."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from ..db import connect_readonly
from ..lemma import Lemmatizer
from ..search import (
    build_match,
    build_match_from_groups,
    context_rows,
    group_counts,
    group_counts_remaining_count,
    parse_query_groups,
    run_count,
    run_search,
    where_filters,
)
from ..viewdata import tme_link
from .budget import enforce_budget
from .cursor import decode_cursor, encode_cursor, query_fingerprint
from .models import (
    AGGREGATE_RESPONSE_CHARS_HARD_MAX,
    ARCHIVE_RESPONSE_CHARS_HARD_MAX,
    ARCHIVE_TOPICS_PER_CHAT_CAP,
    CONTENT_TRUST_UNTRUSTED_USER_GENERATED,
    CONTEXT_RESPONSE_CHARS_HARD_MAX,
    FETCH_RESPONSE_CHARS_HARD_MAX,
    METADATA_STRING_MAX_CHARS,
    SEARCH_CANDIDATE_POOL_MAX,
    SEARCH_CANDIDATE_POOL_MIN,
    SEARCH_CANDIDATE_POOL_MULTIPLIER,
    SEARCH_CHAT_TOPIC_MAX_RESULTS,
    SEARCH_LOCAL_BURST_MAX_RESULTS,
    SEARCH_RESPONSE_CHARS_HARD_MAX,
    SEARCH_SCORE_SEMANTICS,
    AggregateMessagesInput,
    AggregateMessagesOutput,
    AggregateGroup,
    ArchiveChat,
    ArchiveOverviewInput,
    ArchiveOverviewOutput,
    ArchiveTopic,
    ContextMessage,
    FetchMessagesInput,
    FetchMessagesOutput,
    FetchedMessage,
    GetContextInput,
    GetContextOutput,
    ResponseEnvelope,
    SearchFilters,
    SearchHit,
    SearchMessagesInput,
    SearchMessagesOutput,
    ErrorCode,
    ToolError,
)
from .diversity import select_diverse
from .snippets import build_snippet
from .settings import MCPSettings, configured_database_path, load_settings


SCHEMA_VERSION = "1.0"
CURSOR_PREFIX = "offset:"
BUSY_TIMEOUT_MS = 2_000
PROGRESS_HANDLER_STEPS = 1_000
ResponseT = TypeVar("ResponseT", bound=ResponseEnvelope)

_RUNTIME_LOCK = threading.Lock()
_DB_SEMAPHORE: threading.BoundedSemaphore | None = None
_QUERY_TIMEOUT_SECONDS: float | None = None
_CONFIGURED_QUERY_TIMEOUT_SECONDS: float | None = None
_TOOL_TIMEOUTS: dict[str, float] | None = None
_DB_PATH: Path | None = None
_RUNTIME_KEY: tuple[object, ...] | None = None


_TOOL_TIMEOUT_FIELDS = {
    "archive_overview": "archive_timeout_seconds",
    "search_messages": "search_timeout_seconds",
    "aggregate_messages": "aggregate_timeout_seconds",
    "fetch_messages": "fetch_timeout_seconds",
    "get_context": "context_timeout_seconds",
}


def _tool_timeout_values(settings: MCPSettings) -> dict[str, float]:
    values: dict[str, float] = {}
    for tool_name, field_name in _TOOL_TIMEOUT_FIELDS.items():
        configured = getattr(settings, field_name)
        values[tool_name] = (
            settings.query_timeout_seconds if configured is None else configured
        )
    return values


def _runtime_key(settings: MCPSettings, tool_timeouts: dict[str, float]) -> tuple[object, ...]:
    return (
        settings.max_concurrency,
        settings.query_timeout_seconds,
        settings.db_path,
        tuple(sorted(tool_timeouts.items())),
    )


def _install_runtime(settings: MCPSettings) -> None:
    global _DB_PATH, _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS
    global _CONFIGURED_QUERY_TIMEOUT_SECONDS, _TOOL_TIMEOUTS, _RUNTIME_KEY
    tool_timeouts = _tool_timeout_values(settings)
    _DB_SEMAPHORE = threading.BoundedSemaphore(settings.max_concurrency)
    _QUERY_TIMEOUT_SECONDS = settings.query_timeout_seconds
    _CONFIGURED_QUERY_TIMEOUT_SECONDS = settings.query_timeout_seconds
    _TOOL_TIMEOUTS = tool_timeouts
    _DB_PATH = settings.db_path
    _RUNTIME_KEY = _runtime_key(settings, tool_timeouts)


def _database_path() -> Path:
    if _DB_PATH is not None:
        return _DB_PATH
    return configured_database_path()


def configure_runtime(settings: MCPSettings) -> None:
    """Install process-wide DB hardening limits during server startup."""
    runtime_key = _runtime_key(settings, _tool_timeout_values(settings))
    with _RUNTIME_LOCK:
        if _RUNTIME_KEY is not None:
            if runtime_key != _RUNTIME_KEY:
                raise RuntimeError("MCP runtime is already configured")
            return
        _install_runtime(settings)


def _runtime_limits(tool_name: str | None = None) -> tuple[threading.BoundedSemaphore, float]:
    global _DB_PATH, _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS, _RUNTIME_KEY
    global _CONFIGURED_QUERY_TIMEOUT_SECONDS, _TOOL_TIMEOUTS
    if _DB_SEMAPHORE is None or _QUERY_TIMEOUT_SECONDS is None:
        with _RUNTIME_LOCK:
            if _DB_SEMAPHORE is None or _QUERY_TIMEOUT_SECONDS is None:
                settings = load_settings()
                _install_runtime(settings)
    # The guarded initialization above sets both values together.
    assert _DB_SEMAPHORE is not None
    assert _QUERY_TIMEOUT_SECONDS is not None
    # Existing tests and operators may override the legacy global timeout at
    # runtime; honor that explicit override for compatibility.
    if (
        _CONFIGURED_QUERY_TIMEOUT_SECONDS is None
        or _QUERY_TIMEOUT_SECONDS != _CONFIGURED_QUERY_TIMEOUT_SECONDS
        or _TOOL_TIMEOUTS is None
    ):
        return _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS
    return _DB_SEMAPHORE, _TOOL_TIMEOUTS.get(tool_name or "", _QUERY_TIMEOUT_SECONDS)


def _is_busy_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "database is locked",
            "database table is locked",
            "database schema is locked",
            "database is busy",
        )
    )


def _raise_sqlite_tool_error(
    error: sqlite3.OperationalError,
    *,
    deadline_hit: bool,
) -> None:
    if _is_busy_error(error):
        raise ToolError(
            code=ErrorCode.DB_BUSY,
            message="The archive database is busy; retry the request",
            retryable=True,
        ) from error
    if deadline_hit:
        raise ToolError(
            code=ErrorCode.QUERY_TIMEOUT,
            message="The archive query exceeded its deadline",
            retryable=True,
        ) from error


@contextmanager
def _readonly_connection(
    conn: sqlite3.Connection | None,
    *,
    tool_name: str | None = None,
) -> Iterator[sqlite3.Connection]:
    """Serialize and deadline a DB operation; supplied connections are thread-owned."""
    entered_at = time.monotonic()
    semaphore, timeout_seconds = _runtime_limits(tool_name)
    deadline = entered_at + max(0.0, timeout_seconds)
    if not semaphore.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise ToolError(
            code=ErrorCode.QUERY_TIMEOUT,
            message="The archive database concurrency wait exceeded its deadline",
            retryable=True,
        )
    if time.monotonic() >= deadline:
        semaphore.release()
        raise ToolError(
            code=ErrorCode.QUERY_TIMEOUT,
            message="The archive database concurrency wait exceeded its deadline",
            retryable=True,
        )

    readonly: sqlite3.Connection | None = None
    transaction_started = False
    deadline_hit = False
    try:
        try:
            readonly = conn if conn is not None else connect_readonly(_database_path())
            if time.monotonic() >= deadline:
                raise ToolError(
                    code=ErrorCode.QUERY_TIMEOUT,
                    message="The archive query exceeded its deadline",
                    retryable=True,
                )

            readonly.execute("PRAGMA query_only=ON")
            remaining_seconds = max(0.0, deadline - time.monotonic())
            if remaining_seconds <= 0:
                raise ToolError(
                    code=ErrorCode.QUERY_TIMEOUT,
                    message="The archive query exceeded its deadline",
                    retryable=True,
                )
            busy_timeout_ms = min(
                BUSY_TIMEOUT_MS,
                math.floor(remaining_seconds * 1000),
            )
            readonly.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            readonly.execute("BEGIN DEFERRED")
            transaction_started = True

            def progress() -> int:
                nonlocal deadline_hit
                if time.monotonic() >= deadline:
                    deadline_hit = True
                    return 1
                return 0

            readonly.set_progress_handler(progress, PROGRESS_HANDLER_STEPS)
            yield readonly
        except sqlite3.OperationalError as error:
            _raise_sqlite_tool_error(error, deadline_hit=deadline_hit)
            raise
    finally:
        if readonly is not None:
            try:
                readonly.set_progress_handler(None, 0)
                if transaction_started and readonly.in_transaction:
                    readonly.execute("ROLLBACK")
            finally:
                if conn is None:
                    readonly.close()
        semaphore.release()


def _decode_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or not cursor.startswith(CURSOR_PREFIX):
        raise ValueError("invalid temporary cursor")
    raw_offset = cursor[len(CURSOR_PREFIX):]
    try:
        offset = int(raw_offset)
    except ValueError as exc:
        raise ValueError("invalid temporary cursor") from exc
    if offset < 0:
        raise ValueError("invalid temporary cursor")
    return offset


def _encode_offset(offset: int) -> str:
    return f"{CURSOR_PREFIX}{offset}"


def _render_response(response: ResponseEnvelope) -> str:
    return json.dumps(asdict(response), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _metadata_text(value: Any) -> str | None:
    """Bound archive-provided metadata before it enters a response envelope."""
    if value is None:
        return None
    return str(value)[:METADATA_STRING_MAX_CHARS]


def _metadata_value(value: Any) -> Any:
    """Bound string-valued metadata while preserving numeric group keys."""
    if isinstance(value, str):
        return value[:METADATA_STRING_MAX_CHARS]
    return value


def _trim_archive_response(
    response: ArchiveOverviewOutput,
    remove_count: int,
    *,
    next_cursor_factory: Callable[[list[ArchiveChat]], str | None] | None = None,
) -> ArchiveOverviewOutput:
    trimmed = replace(
        response,
        chats=response.chats[:-remove_count],
        returned_chats=max(0, len(response.chats) - remove_count),
        has_more=True,
    )
    if next_cursor_factory is not None:
        trimmed = replace(
            trimmed,
            next_cursor=next_cursor_factory(trimmed.chats),
        )
    return trimmed


def _trim_aggregate_response(
    response: AggregateMessagesOutput,
    remove_count: int,
    *,
    next_cursor_factory: Callable[[list[AggregateGroup]], str | None] | None = None,
) -> AggregateMessagesOutput:
    removed = response.groups[-remove_count:]
    trimmed = replace(
        response,
        groups=response.groups[:-remove_count],
        returned_groups=max(0, len(response.groups) - remove_count),
        other_count=response.other_count + sum(group.count for group in removed),
        has_more=True,
    )
    if next_cursor_factory is not None:
        trimmed = replace(
            trimmed,
            next_cursor=next_cursor_factory(trimmed.groups),
        )
    return trimmed


def _trim_search_response(
    response: SearchMessagesOutput,
    remove_count: int,
    *,
    next_cursor_factory: Callable[[list[SearchHit]], str | None] | None = None,
) -> SearchMessagesOutput:
    trimmed = replace(
        response,
        hits=response.hits[:-remove_count],
        returned_hits=max(0, len(response.hits) - remove_count),
        has_more=True,
    )
    if next_cursor_factory is not None:
        trimmed = replace(
            trimmed,
            next_cursor=next_cursor_factory(trimmed.hits),
        )
    return trimmed


def _trim_fetch_response(
    response: FetchMessagesOutput,
    remove_count: int,
) -> FetchMessagesOutput:
    removed_ids = [message.id for message in response.messages[-remove_count:]]
    return replace(
        response,
        messages=response.messages[:-remove_count],
        omitted_ids=[*response.omitted_ids, *removed_ids],
    )


def _trim_context_response(
    response: GetContextOutput,
    remove_count: int,
) -> GetContextOutput:
    """Remove context edges while retaining the pivot message."""
    messages = response.messages
    after_count = sum(message.relation == "after" for message in messages)
    before_count = sum(message.relation == "before" for message in messages)
    remove_after = min(remove_count, after_count)
    remove_before = min(remove_count - remove_after, before_count)
    end = len(messages) - remove_after if remove_after else len(messages)
    kept = messages[remove_before:end]
    return replace(
        response,
        messages=kept,
        has_more_before=response.has_more_before or remove_before > 0,
        has_more_after=response.has_more_after or remove_after > 0,
    )


def _finalize_response(
    output_type: type[ResponseT],
    *,
    hard_max_chars: int | None = None,
    budget_field: str | None = None,
    on_truncate: Callable[[ResponseT, int], ResponseT] | None = None,
    preserve_items: int = 0,
    **data: Any,
) -> ResponseT:
    """Attach the common envelope metrics to an already-built response."""
    response: ResponseT = output_type(
        schema_version=SCHEMA_VERSION,
        content_trust=CONTENT_TRUST_UNTRUSTED_USER_GENERATED,
        response_chars=0,
        response_bytes_utf8=0,
        estimated_tokens_rough=0,
        truncated=False,
        **data,
    )
    if hard_max_chars is not None:
        if budget_field is None or on_truncate is None:
            raise ValueError("budget_field and on_truncate are required with a hard cap")
        bounded = enforce_budget(
            response,
            hard_max_chars,
            budget_field,
            on_truncate,
            render=_render_response,
            preserve_items=preserve_items,
        )
        if bounded.response_chars > hard_max_chars:
            raise ToolError(
                code=ErrorCode.OUTPUT_BUDGET_EXCEEDED,
                message=(
                    "The response cannot fit within the output character budget; "
                    "retry with fewer or smaller result fields"
                ),
                retryable=True,
                details={
                    "hard_max_chars": hard_max_chars,
                    "response_chars": bounded.response_chars,
                },
            )
        return bounded
    for _ in range(8):
        rendered = _render_response(response)
        chars = len(rendered)
        bytes_utf8 = len(rendered.encode("utf-8"))
        estimated_tokens = chars // 4
        if (
            response.response_chars == chars
            and response.response_bytes_utf8 == bytes_utf8
            and response.estimated_tokens_rough == estimated_tokens
        ):
            return response
        response = replace(
            response,
            response_chars=chars,
            response_bytes_utf8=bytes_utf8,
            estimated_tokens_rough=estimated_tokens,
        )
    return response


def _chat_scope(chat_ids: list[int] | None) -> tuple[str, list[int]]:
    if not chat_ids:
        return "", []
    placeholders = ",".join("?" for _ in chat_ids)
    return f"m.chat_id IN ({placeholders})", list(chat_ids)


def _topic_rows(conn: sqlite3.Connection, chat_id: int) -> list[ArchiveTopic]:
    rows = conn.execute(
        "SELECT m.topic_id, t.title, count(*) AS message_count, "
        "min(m.date) AS date_from, max(m.date) AS date_to "
        "FROM messages m LEFT JOIN topics t "
        "ON t.chat_id=m.chat_id AND t.topic_id=m.topic_id "
        "WHERE m.chat_id=? GROUP BY m.topic_id "
        "ORDER BY message_count DESC, m.topic_id ASC LIMIT ?",
        (chat_id, ARCHIVE_TOPICS_PER_CHAT_CAP),
    ).fetchall()
    topics = []
    for row in rows:
        topic_id = row["topic_id"]
        title = _metadata_text(row["title"])
        if topic_id is None:
            title = _metadata_text("General (без топика)")
        elif not title:
            title = _metadata_text(f"топик {topic_id}")
        topics.append(
            ArchiveTopic(
                topic_id=topic_id,
                title=title,
                message_count=int(row["message_count"]),
                date_from=_metadata_text(row["date_from"]),
                date_to=_metadata_text(row["date_to"]),
            )
        )
    return topics


def archive_overview(
    request: ArchiveOverviewInput,
    conn: sqlite3.Connection | None = None,
) -> ArchiveOverviewOutput:
    """Return bounded archive/chat metadata using a read-only connection."""
    with _readonly_connection(conn, tool_name="archive_overview") as readonly:
        scope, params = _chat_scope(request.chat_ids)
        where = f" WHERE {scope}" if scope else ""
        summary = readonly.execute(
            "SELECT count(*) AS total_messages, count(DISTINCT m.chat_id) AS total_chats, "
            "min(m.date) AS date_from, max(m.date) AS date_to "
            f"FROM messages m{where}",
            params,
        ).fetchone()
        fingerprint = query_fingerprint(
            "",
            "archive",
            {"chat_ids": request.chat_ids},
            "archive_overview",
            extra={"include_topics": request.include_topics},
        )
        after = None
        if request.cursor is not None:
            decoded = decode_cursor(
                request.cursor,
                expected_query_fingerprint=fingerprint,
                expected_sort="archive_overview",
                conn=readonly,
            )
            after = (decoded["last_count"], decoded["last_chat_id"])

        after_sql = ""
        after_params: list[int] = []
        if after is not None:
            after_sql = " HAVING count(*) < ? OR (count(*) = ? AND m.chat_id > ?)"
            after_params = [after[0], after[0], after[1]]
        rows = readonly.execute(
            "SELECT m.chat_id, c.title, count(*) AS message_count, "
            "min(m.date) AS date_from, max(m.date) AS date_to, "
            "count(DISTINCT m.topic_id) AS topic_count "
            "FROM messages m LEFT JOIN chats c ON c.chat_id=m.chat_id "
            f"{where} GROUP BY m.chat_id{after_sql} "
            "ORDER BY message_count DESC, m.chat_id ASC LIMIT ?",
            [*params, *after_params, request.limit + 1],
        ).fetchall()

        page = rows[:request.limit]
        has_more = len(rows) > request.limit
        chats = []
        for row in page:
            chat_id = int(row["chat_id"])
            topics = _topic_rows(readonly, chat_id) if request.include_topics else None
            chats.append(
                ArchiveChat(
                    chat_id=chat_id,
                    title=_metadata_text(row["title"] or str(chat_id)) or str(chat_id),
                    message_count=int(row["message_count"]),
                    date_from=_metadata_text(row["date_from"]),
                    date_to=_metadata_text(row["date_to"]),
                    topic_count=int(row["topic_count"] or 0),
                    topics=topics,
                )
            )

        def archive_cursor(chats: list[ArchiveChat]) -> str | None:
            if not chats:
                return None
            last_chat = chats[-1]
            return encode_cursor(
                query_fingerprint=fingerprint,
                sort="archive_overview",
                last_count=last_chat.message_count,
                last_chat_id=last_chat.chat_id,
                conn=readonly,
            )

        def trim_archive_response(
            response: ArchiveOverviewOutput,
            remove_count: int,
        ) -> ArchiveOverviewOutput:
            return _trim_archive_response(
                response,
                remove_count,
                next_cursor_factory=archive_cursor,
            )

        return _finalize_response(
            ArchiveOverviewOutput,
            hard_max_chars=ARCHIVE_RESPONSE_CHARS_HARD_MAX,
            budget_field="chats",
            on_truncate=trim_archive_response,
            preserve_items=1,
            total_messages=int(summary["total_messages"]),
            total_chats=int(summary["total_chats"]),
            date_from=_metadata_text(summary["date_from"]),
            date_to=_metadata_text(summary["date_to"]),
            chats=chats,
            returned_chats=len(chats),
            has_more=has_more,
            next_cursor=(
                archive_cursor(chats)
                if has_more and page
                else None
            ),
        )


def _search_parts(filters: SearchFilters | None) -> tuple[list[str], list[Any]]:
    if filters is None:
        filters = SearchFilters()

    sender = str(filters.sender_id) if filters.sender_id is not None else filters.sender_name
    cond, params = where_filters(
        chat_ids=filters.chat_ids,
        topic_ids=filters.topic_ids,
        sender=sender,
        date_from=filters.date_from,
        date_to=filters.date_to,
        media=filters.media,
    )
    if filters.sender_id is not None and filters.sender_name is not None:
        cond.append("m.sender_name LIKE ? COLLATE NOCASE")
        params.append(f"%{filters.sender_name}%")
    return cond, params


def _build_match(query: str, match_mode: str) -> str:
    return build_match(query, Lemmatizer(), match_mode=match_mode)


def _decode_topic_group_key(key: str | int | None) -> tuple[int, int | None]:
    """Decode the stable ``chat_id:topic_id`` aggregate wire key."""
    if not isinstance(key, str):
        raise ValueError("topic group key must be a string")
    chat_text, separator, topic_text = key.rpartition(":")
    if not separator or not chat_text or not topic_text:
        raise ValueError("invalid topic group key")
    try:
        chat_id = int(chat_text)
        topic_id = None if topic_text == "null" else int(topic_text)
    except ValueError as exc:
        raise ValueError("invalid topic group key") from exc
    return chat_id, topic_id


def _topic_group_metadata(
    conn: sqlite3.Connection,
    key: str | int | None,
) -> tuple[int, str, int | None, str | None]:
    chat_id, topic_id = _decode_topic_group_key(key)
    row = conn.execute(
        "SELECT c.title AS chat_title, t.title AS topic_title "
        "FROM chats c "
        "LEFT JOIN topics t ON t.chat_id=c.chat_id AND t.topic_id IS ? "
        "WHERE c.chat_id=?",
        (topic_id, chat_id),
    ).fetchone()
    chat_title = _metadata_text((row["chat_title"] if row else None) or str(chat_id))
    topic_title = None
    if topic_id is not None:
        topic_title = _metadata_text(
            (row["topic_title"] if row else None) or f"топик {topic_id}"
        )
    return chat_id, chat_title, topic_id, topic_title


def aggregate_messages(
    request: AggregateMessagesInput,
    conn: sqlite3.Connection | None = None,
) -> AggregateMessagesOutput:
    """Return FTS coverage grouped by the requested dimension."""
    with _readonly_connection(conn, tool_name="aggregate_messages") as readonly:
        match = _build_match(request.query, request.match_mode)
        cond, params = _search_parts(request.filters)
        total = run_count(readonly, match, cond, params)
        fingerprint = query_fingerprint(
            request.query,
            request.match_mode,
            request.filters,
            "aggregate",
            group_by=request.group_by,
        )
        after = None
        if request.cursor is not None:
            decoded = decode_cursor(
                request.cursor,
                expected_query_fingerprint=fingerprint,
                expected_sort="aggregate",
                expected_group_by=request.group_by,
                conn=readonly,
            )
            after = (decoded["last_count"], decoded["last_group_key"])

        rows = group_counts(
            readonly,
            match,
            cond,
            params,
            request.group_by,
            limit=request.limit + 1,
            after=after,
        )
        page = rows[:request.limit]
        has_more = len(rows) > request.limit
        tail_after = (int(page[-1]["c"]), page[-1]["k"]) if has_more and page else None
        other_count = (
            group_counts_remaining_count(
                readonly,
                match,
                cond,
                params,
                request.group_by,
                after=tail_after,
            )
            if tail_after is not None
            else 0
        )
        groups = []
        for row in page:
            topic_metadata = (
                _topic_group_metadata(readonly, row["k"])
                if request.group_by == "topic"
                else None
            )
            groups.append(
                AggregateGroup(
                    key=_metadata_value(row["k"]),
                    count=int(row["c"]),
                    date_from=_metadata_text(row["d0"]),
                    date_to=_metadata_text(row["d1"]),
                    chat_id=topic_metadata[0] if topic_metadata else None,
                    chat_title=topic_metadata[1] if topic_metadata else None,
                    topic_id=topic_metadata[2] if topic_metadata else None,
                    topic_title=topic_metadata[3] if topic_metadata else None,
                )
            )

        def aggregate_cursor(groups: list[AggregateGroup]) -> str | None:
            if not groups:
                return None
            last_group = groups[-1]
            # Keep the raw SQL key in the opaque cursor.  The wire key may be
            # capped for sender groups, but keyset pagination must compare the
            # original value to avoid collisions or skipped groups.
            group_index = len(groups) - 1
            raw_group_key = page[group_index]["k"] if group_index < len(page) else last_group.key
            return encode_cursor(
                query_fingerprint=fingerprint,
                sort="aggregate",
                group_by=request.group_by,
                last_count=last_group.count,
                last_group_key=raw_group_key,
                conn=readonly,
            )

        def trim_aggregate_response(
            response: AggregateMessagesOutput,
            remove_count: int,
        ) -> AggregateMessagesOutput:
            return _trim_aggregate_response(
                response,
                remove_count,
                next_cursor_factory=aggregate_cursor,
            )

        return _finalize_response(
            AggregateMessagesOutput,
            hard_max_chars=AGGREGATE_RESPONSE_CHARS_HARD_MAX,
            budget_field="groups",
            on_truncate=trim_aggregate_response,
            preserve_items=1,
            total_hits=int(total["c"]),
            group_by=request.group_by,
            groups=groups,
            returned_groups=len(groups),
            other_count=other_count,
            has_more=has_more,
            next_cursor=(
                aggregate_cursor(groups)
                if has_more and page
                else None
            ),
        )


def _search_metadata(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> dict[int, sqlite3.Row]:
    """Load titles only for the messages that will be returned."""
    ids = list(dict.fromkeys(int(row["id"]) for row in rows))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    metadata = conn.execute(
        "SELECT m.id, c.title AS chat_title, t.title AS topic_title "
        "FROM messages m "
        "LEFT JOIN chats c ON c.chat_id=m.chat_id "
        "LEFT JOIN topics t ON t.chat_id=m.chat_id AND t.topic_id=m.topic_id "
        f"WHERE m.id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(row["id"]): row for row in metadata}


def search_messages(
    request: SearchMessagesInput,
    conn: sqlite3.Connection | None = None,
) -> SearchMessagesOutput:
    """Run FTS search, optionally selecting a bounded diverse shortlist.

    Diversity is intentionally a bounded sampler: it does not expose deep
    offset pagination, so callers should use ``strategy=relevance`` for page
    two and later.  When ``include_total`` is false, ``has_more`` remains
    false because the bounded sampler has no safe total-based signal.
    """
    with _readonly_connection(conn, tool_name="search_messages") as readonly:
        lem = Lemmatizer()
        query_groups = parse_query_groups(
            request.query,
            lem,
            match_mode=request.match_mode,
        )
        match = build_match_from_groups(
            query_groups,
            match_mode=request.match_mode,
        )
        cond, params = _search_parts(request.filters)
        total_hits = None
        if request.include_total:
            total = run_count(readonly, match, cond, params)
            total_hits = int(total["c"])
        search_cursor_factory: Callable[[list[SearchHit]], str | None] | None = None

        if request.strategy == "diverse":
            candidate_pool_size = min(
                max(
                    request.limit * SEARCH_CANDIDATE_POOL_MULTIPLIER,
                    SEARCH_CANDIDATE_POOL_MIN,
                ),
                SEARCH_CANDIDATE_POOL_MAX,
            )
            candidates = run_search(
                readonly,
                match,
                cond,
                params,
                limit=candidate_pool_size,
                rank=True,
                include_score=True,
            )
            page = select_diverse(
                candidates,
                request.limit,
                local_burst_max=SEARCH_LOCAL_BURST_MAX_RESULTS,
                chat_topic_max=SEARCH_CHAT_TOPIC_MAX_RESULTS,
            )
            # Diverse is deliberately not cursor-paginatable.  When the
            # total was requested, it is still useful to report that the
            # bounded sample does not cover all matching rows.  Without a
            # total there is no safe way to infer this, so remain conservative.
            has_more = total_hits is not None and total_hits > len(page)
            next_cursor = None
            pagination_supported = False
            ranked_output = True
        else:
            pagination_supported = True
            rank = request.sort == "relevance"
            fingerprint = query_fingerprint(
                request.query,
                request.match_mode,
                request.filters,
                request.sort,
            )
            after = None
            if request.cursor is not None:
                decoded = decode_cursor(
                    request.cursor,
                    expected_query_fingerprint=fingerprint,
                    expected_sort=request.sort,
                    conn=readonly,
                )
                if rank:
                    after = (decoded["last_score"], decoded["last_row_id"])
                else:
                    after = (decoded["last_date"], decoded["last_row_id"])

            fetch_limit = request.limit + 1
            if request.sort == "newest":
                ordered = run_search(
                    readonly,
                    match,
                    cond,
                    params,
                    limit=fetch_limit,
                    rank=False,
                    include_score=False,
                    newest=True,
                    after=after,
                )
            else:
                candidates = run_search(
                    readonly,
                    match,
                    cond,
                    params,
                    limit=fetch_limit,
                    rank=rank,
                    include_score=rank,
                    after=after,
                )
                ordered = list(candidates)

            page = ordered[:request.limit]
            has_more = len(ordered) > request.limit
            next_cursor = None
            if has_more and page:
                last_row = page[-1]
                next_cursor = encode_cursor(
                    query_fingerprint=fingerprint,
                    sort=request.sort,
                    last_row_id=int(last_row["id"]),
                    conn=readonly,
                    last_score=float(last_row["bm25_score"]) if rank else None,
                    last_date=last_row["date"] if not rank else None,
                )

            def search_cursor(hits: list[SearchHit]) -> str | None:
                if not hits:
                    return None
                last_hit = hits[-1]
                last_row = page[len(hits) - 1]
                return encode_cursor(
                    query_fingerprint=fingerprint,
                    sort=request.sort,
                    last_row_id=int(last_row["id"]),
                    conn=readonly,
                    last_score=last_hit.bm25_score if rank else None,
                    last_date=last_row["date"] if not rank else None,
                )

            search_cursor_factory = search_cursor
            ranked_output = rank

        metadata = _search_metadata(readonly, page)
        hits = []
        for index, row in enumerate(page, start=1):
            chat_id = int(row["chat_id"])
            message_id = int(row["message_id"])
            meta = metadata.get(int(row["id"]))
            chat_title = _metadata_text((meta["chat_title"] if meta else None) or str(chat_id))
            topic_id = row["topic_id"]
            topic_title = _metadata_text(meta["topic_title"] if meta else None)
            if topic_id is not None and not topic_title:
                topic_title = _metadata_text(f"топик {topic_id}")
            snippet, snippet_source, matched_terms = build_snippet(
                query_groups,
                text=row["text"],
                transcript=row["transcript"],
                poll=row["poll"],
                media_name=row["media_name"],
                snippet_chars=request.snippet_chars,
                lem=lem,
            )
            hits.append(
                SearchHit(
                    id=f"tg:{chat_id}:{message_id}",
                    chat_id=chat_id,
                    chat_title=chat_title,
                    topic_id=topic_id,
                    topic_title=topic_title,
                    message_id=message_id,
                    date=_metadata_text(row["date"]) or "",
                    sender_id=row["sender_id"],
                    sender=_metadata_text(row["sender_name"]),
                    reply_to=row["reply_to"],
                    snippet=snippet,
                    snippet_source=snippet_source,
                    matched_terms=matched_terms,
                    bm25_score=float(row["bm25_score"]) if ranked_output else None,
                    rank=index,
                    media_kind=_metadata_text(row["media_kind"]),
                    telegram_url=tme_link(chat_id, message_id),
                    content_trust=CONTENT_TRUST_UNTRUSTED_USER_GENERATED,
                )
            )

        def trim_search_response(
            response: SearchMessagesOutput,
            remove_count: int,
        ) -> SearchMessagesOutput:
            return _trim_search_response(
                response,
                remove_count,
                next_cursor_factory=search_cursor_factory,
            )

        return _finalize_response(
            SearchMessagesOutput,
            hard_max_chars=SEARCH_RESPONSE_CHARS_HARD_MAX,
            budget_field="hits",
            on_truncate=trim_search_response,
            preserve_items=1 if request.strategy != "diverse" else 0,
            original_query=request.query,
            match_mode=request.match_mode,
            score_semantics=SEARCH_SCORE_SEMANTICS,
            total_hits=total_hits,
            returned_hits=len(hits),
            hits=hits,
            has_more=has_more,
            pagination_supported=pagination_supported,
            next_cursor=next_cursor,
        )


def _truncate_value(value: Any, max_chars: int) -> tuple[str | None, int, bool]:
    if value is None:
        return None, 0, False
    text = str(value)
    return text[:max_chars], len(text), len(text) > max_chars


def _json_value(value: Any) -> Any | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _links_value(value: Any) -> list[str]:
    parsed = _json_value(value)
    if not isinstance(parsed, list):
        return []
    return [str(item)[:METADATA_STRING_MAX_CHARS] for item in parsed]


def _fetch_message(row: sqlite3.Row, request: FetchMessagesInput) -> FetchedMessage:
    text, original_text_chars, text_truncated = _truncate_value(
        row["text"], request.per_message_max_chars
    )
    transcript_original = len(row["transcript"]) if row["transcript"] else 0
    if request.include_transcript:
        transcript, _, transcript_truncated = _truncate_value(
            row["transcript"], request.per_message_max_chars
        )
    else:
        transcript, transcript_truncated = None, False

    chat_id = int(row["chat_id"])
    topic_id = row["topic_id"]
    topic = row["topic_title"]
    if topic_id is not None and not topic:
        topic = f"топик {topic_id}"

    return FetchedMessage(
        id=f"tg:{chat_id}:{int(row['message_id'])}",
        chat=_metadata_text(row["chat_title"] or str(chat_id)),
        topic=_metadata_text(topic),
        date=_metadata_text(row["date"]) or "",
        sender=_metadata_text(row["sender_name"]),
        reply_to=row["reply_to"],
        text=text,
        original_text_chars=original_text_chars,
        text_truncated=text_truncated,
        transcript=transcript,
        transcript_original_chars=transcript_original,
        transcript_truncated=transcript_truncated,
        poll=_json_value(row["poll"]),
        media_kind=_metadata_text(row["media_kind"]),
        media_name=_metadata_text(row["media_name"]),
        links=_links_value(row["links"]) if request.include_links else None,
        content_trust=CONTENT_TRUST_UNTRUSTED_USER_GENERATED,
        reactions=_json_value(row["reactions"]) if request.include_reactions else None,
    )


def fetch_messages(
    request: FetchMessagesInput,
    parsed_ids: list[tuple[int, int]],
    conn: sqlite3.Connection | None = None,
) -> FetchMessagesOutput:
    """Fetch a bounded, ordered shortlist with one read-only SQL query."""
    with _readonly_connection(conn, tool_name="fetch_messages") as readonly:
        clauses = ["(m.chat_id=? AND m.message_id=?)" for _ in parsed_ids]
        params: list[int] = []
        for chat_id, message_id in parsed_ids:
            params.extend((chat_id, message_id))
        rows = readonly.execute(
            "SELECT m.*, c.title AS chat_title, t.title AS topic_title "
            "FROM messages m "
            "LEFT JOIN chats c ON c.chat_id=m.chat_id "
            "LEFT JOIN topics t ON t.chat_id=m.chat_id AND t.topic_id=m.topic_id "
            f"WHERE {' OR '.join(clauses)}",
            params,
        ).fetchall()
        by_key = {(int(row["chat_id"]), int(row["message_id"])): row for row in rows}
        messages = []
        omitted_ids = []
        for public_id, key in zip(request.ids, parsed_ids):
            row = by_key.get(key)
            if row is None:
                omitted_ids.append(public_id)
            else:
                messages.append(_fetch_message(row, request))
        return _finalize_response(
            FetchMessagesOutput,
            hard_max_chars=FETCH_RESPONSE_CHARS_HARD_MAX,
            budget_field="messages",
            on_truncate=_trim_fetch_response,
            messages=messages,
            omitted_ids=omitted_ids,
        )


def _context_scope(chat_id: int, topic_id: int | None, same_topic: bool) -> tuple[str, list[int]]:
    if same_topic and topic_id is None:
        return "chat_id=? AND topic_id IS NULL", [chat_id]
    if same_topic:
        return "chat_id=? AND topic_id IS ?", [chat_id, topic_id]
    return "chat_id=?", [chat_id]


def _has_context_side(
    conn: sqlite3.Connection,
    scope: str,
    base: list[int],
    operator: str,
    message_id: int,
) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM messages WHERE {scope} AND message_id {operator} ? LIMIT 1",
        [*base, message_id],
    ).fetchone()
    return row is not None


def _context_message(
    row: sqlite3.Row,
    relation: str,
    request: GetContextInput,
) -> ContextMessage:
    text, _, _ = _truncate_value(row["text"], request.message_max_chars)
    if request.include_transcripts:
        transcript, _, _ = _truncate_value(row["transcript"], request.message_max_chars)
    else:
        transcript = None
    chat_id = int(row["chat_id"])
    message_id = int(row["message_id"])
    return ContextMessage(
        id=f"tg:{chat_id}:{message_id}",
        chat_id=chat_id,
        message_id=message_id,
        topic_id=row["topic_id"],
        relation=relation,
        date=_metadata_text(row["date"]) or "",
        sender=_metadata_text(row["sender_name"]),
        text=text,
        transcript=transcript,
    )


def get_context(
    request: GetContextInput,
    parsed_id: tuple[int, int],
    conn: sqlite3.Connection | None = None,
) -> GetContextOutput:
    """Return a bounded local context window around a public message ID."""
    with _readonly_connection(conn, tool_name="get_context") as readonly:
        chat_id, message_id = parsed_id
        rows_before, pivot, rows_after = context_rows(
            readonly,
            chat_id,
            message_id,
            before=request.before,
            after=request.after,
            same_topic=request.same_topic,
        )
        if pivot is None:
            raise ToolError(
                code=ErrorCode.NOT_FOUND,
                message=f"message not found: {request.id}",
            )

        scope, base = _context_scope(chat_id, pivot["topic_id"], request.same_topic)
        if rows_before:
            has_more_before = (
                len(rows_before) == request.before
                and _has_context_side(
                    readonly, scope, base, "<", int(rows_before[0]["message_id"])
                )
            )
        else:
            has_more_before = _has_context_side(readonly, scope, base, "<", message_id)
        if rows_after:
            has_more_after = (
                len(rows_after) == request.after
                and _has_context_side(
                    readonly, scope, base, ">", int(rows_after[-1]["message_id"])
                )
            )
        else:
            has_more_after = _has_context_side(readonly, scope, base, ">", message_id)

        messages = [
            *(_context_message(row, "before", request) for row in rows_before),
            _context_message(pivot, "pivot", request),
            *(_context_message(row, "after", request) for row in rows_after),
        ]
        return _finalize_response(
            GetContextOutput,
            hard_max_chars=CONTEXT_RESPONSE_CHARS_HARD_MAX,
            budget_field="messages",
            on_truncate=_trim_context_response,
            preserve_items=1,
            pivot_id=request.id,
            messages=messages,
            has_more_before=has_more_before,
            has_more_after=has_more_after,
        )
