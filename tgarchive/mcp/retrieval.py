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
    SEARCH_CANDIDATE_POOL_MAX,
    SEARCH_CANDIDATE_POOL_MIN,
    SEARCH_CANDIDATE_POOL_MULTIPLIER,
    SEARCH_CHAT_TOPIC_MAX_RESULTS,
    SEARCH_LOCAL_BURST_MAX_RESULTS,
    SEARCH_RESPONSE_CHARS_HARD_MAX,
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
_DB_PATH: Path | None = None
_RUNTIME_KEY: tuple[int, float, Path] | None = None


def _database_path() -> Path:
    if _DB_PATH is not None:
        return _DB_PATH
    return configured_database_path()


def configure_runtime(settings: MCPSettings) -> None:
    """Install process-wide DB hardening limits during server startup."""
    global _DB_PATH, _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS, _RUNTIME_KEY
    runtime_key = (
        settings.max_concurrency,
        settings.query_timeout_seconds,
        settings.db_path,
    )
    with _RUNTIME_LOCK:
        if _RUNTIME_KEY is not None:
            if runtime_key != _RUNTIME_KEY:
                raise RuntimeError("MCP runtime is already configured")
            return
        _DB_SEMAPHORE = threading.BoundedSemaphore(settings.max_concurrency)
        _QUERY_TIMEOUT_SECONDS = settings.query_timeout_seconds
        _DB_PATH = settings.db_path
        _RUNTIME_KEY = runtime_key


def _runtime_limits() -> tuple[threading.BoundedSemaphore, float]:
    global _DB_PATH, _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS, _RUNTIME_KEY
    if _DB_SEMAPHORE is None or _QUERY_TIMEOUT_SECONDS is None:
        with _RUNTIME_LOCK:
            if _DB_SEMAPHORE is None or _QUERY_TIMEOUT_SECONDS is None:
                settings = load_settings()
                _DB_SEMAPHORE = threading.BoundedSemaphore(settings.max_concurrency)
                _QUERY_TIMEOUT_SECONDS = settings.query_timeout_seconds
                _DB_PATH = settings.db_path
                _RUNTIME_KEY = (
                    settings.max_concurrency,
                    settings.query_timeout_seconds,
                    settings.db_path,
                )
    # The guarded initialization above sets both values together.
    assert _DB_SEMAPHORE is not None
    assert _QUERY_TIMEOUT_SECONDS is not None
    return _DB_SEMAPHORE, _QUERY_TIMEOUT_SECONDS


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
def _readonly_connection(conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
    """Serialize and deadline a DB operation; supplied connections are thread-owned."""
    semaphore, timeout_seconds = _runtime_limits()
    if not semaphore.acquire(timeout=max(0.0, timeout_seconds)):
        raise ToolError(
            code=ErrorCode.QUERY_TIMEOUT,
            message="The archive database concurrency wait exceeded its deadline",
            retryable=True,
        )

    readonly: sqlite3.Connection | None = None
    deadline_hit = False
    try:
        try:
            readonly = conn if conn is not None else connect_readonly(_database_path())
            opened_at = time.monotonic()
            deadline = opened_at + timeout_seconds

            readonly.execute("PRAGMA query_only=ON")
            if timeout_seconds <= 0:
                busy_timeout_ms = 0
            else:
                busy_timeout_ms = min(
                    BUSY_TIMEOUT_MS,
                    max(0, math.floor(timeout_seconds * 1000)),
                )
            readonly.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")

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


def _trim_archive_response(
    response: ArchiveOverviewOutput,
    remove_count: int,
) -> ArchiveOverviewOutput:
    return replace(
        response,
        chats=response.chats[:-remove_count],
        returned_chats=max(0, len(response.chats) - remove_count),
        has_more=True,
    )


def _trim_aggregate_response(
    response: AggregateMessagesOutput,
    remove_count: int,
) -> AggregateMessagesOutput:
    removed = response.groups[-remove_count:]
    return replace(
        response,
        groups=response.groups[:-remove_count],
        returned_groups=max(0, len(response.groups) - remove_count),
        other_count=response.other_count + sum(group.count for group in removed),
        has_more=True,
    )


def _trim_search_response(
    response: SearchMessagesOutput,
    remove_count: int,
) -> SearchMessagesOutput:
    return replace(
        response,
        hits=response.hits[:-remove_count],
        returned_hits=max(0, len(response.hits) - remove_count),
        has_more=True,
    )


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
        return enforce_budget(
            response,
            hard_max_chars,
            budget_field,
            on_truncate,
            render=_render_response,
            preserve_items=preserve_items,
        )
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
        title = row["title"]
        if topic_id is None:
            title = "General (без топика)"
        elif not title:
            title = f"топик {topic_id}"
        topics.append(
            ArchiveTopic(
                topic_id=topic_id,
                title=title,
                message_count=int(row["message_count"]),
                date_from=row["date_from"],
                date_to=row["date_to"],
            )
        )
    return topics


def archive_overview(
    request: ArchiveOverviewInput,
    conn: sqlite3.Connection | None = None,
) -> ArchiveOverviewOutput:
    """Return bounded archive/chat metadata using a read-only connection."""
    with _readonly_connection(conn) as readonly:
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
                    title=row["title"] or str(chat_id),
                    message_count=int(row["message_count"]),
                    date_from=row["date_from"],
                    date_to=row["date_to"],
                    topic_count=int(row["topic_count"] or 0),
                    topics=topics,
                )
            )

        return _finalize_response(
            ArchiveOverviewOutput,
            hard_max_chars=ARCHIVE_RESPONSE_CHARS_HARD_MAX,
            budget_field="chats",
            on_truncate=_trim_archive_response,
            total_messages=int(summary["total_messages"]),
            total_chats=int(summary["total_chats"]),
            date_from=summary["date_from"],
            date_to=summary["date_to"],
            chats=chats,
            returned_chats=len(chats),
            has_more=has_more,
            next_cursor=(
                encode_cursor(
                    query_fingerprint=fingerprint,
                    sort="archive_overview",
                    last_count=int(page[-1]["message_count"]),
                    last_chat_id=int(page[-1]["chat_id"]),
                    conn=readonly,
                )
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
    return build_match(query, Lemmatizer(), any_mode=match_mode == "or")


def aggregate_messages(
    request: AggregateMessagesInput,
    conn: sqlite3.Connection | None = None,
) -> AggregateMessagesOutput:
    """Return FTS coverage grouped by the requested dimension."""
    with _readonly_connection(conn) as readonly:
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
        groups = [
            AggregateGroup(
                key=row["k"],
                count=int(row["c"]),
                date_from=row["d0"],
                date_to=row["d1"],
            )
            for row in page
        ]
        return _finalize_response(
            AggregateMessagesOutput,
            hard_max_chars=AGGREGATE_RESPONSE_CHARS_HARD_MAX,
            budget_field="groups",
            on_truncate=_trim_aggregate_response,
            total_hits=int(total["c"]),
            group_by=request.group_by,
            groups=groups,
            returned_groups=len(groups),
            other_count=other_count,
            has_more=has_more,
            next_cursor=(
                encode_cursor(
                    query_fingerprint=fingerprint,
                    sort="aggregate",
                    group_by=request.group_by,
                    last_count=int(page[-1]["c"]),
                    last_group_key=page[-1]["k"],
                    conn=readonly,
                )
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
    two and later.
    """
    with _readonly_connection(conn) as readonly:
        lem = Lemmatizer()
        any_mode = request.match_mode == "or"
        query_groups = parse_query_groups(request.query, lem, any_mode=any_mode)
        match = build_match_from_groups(query_groups, any_mode=any_mode)
        cond, params = _search_parts(request.filters)
        total_hits = None
        if request.include_total:
            total = run_count(readonly, match, cond, params)
            total_hits = int(total["c"])

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
            offset = 0
            has_more = False
            next_cursor = None
            ranked_output = True
        else:
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
            ranked_output = rank

        metadata = _search_metadata(readonly, page)
        hits = []
        for index, row in enumerate(page, start=1):
            chat_id = int(row["chat_id"])
            message_id = int(row["message_id"])
            meta = metadata.get(int(row["id"]))
            chat_title = (meta["chat_title"] if meta else None) or str(chat_id)
            topic_id = row["topic_id"]
            topic_title = meta["topic_title"] if meta else None
            if topic_id is not None and not topic_title:
                topic_title = f"топик {topic_id}"
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
                    date=row["date"],
                    sender_id=row["sender_id"],
                    sender=row["sender_name"],
                    reply_to=row["reply_to"],
                    snippet=snippet,
                    snippet_source=snippet_source,
                    matched_terms=matched_terms,
                    score=float(row["bm25_score"]) if ranked_output else 0.0,
                    rank=index,
                    media_kind=row["media_kind"],
                    telegram_url=tme_link(chat_id, message_id),
                    content_trust=CONTENT_TRUST_UNTRUSTED_USER_GENERATED,
                )
            )

        return _finalize_response(
            SearchMessagesOutput,
            hard_max_chars=SEARCH_RESPONSE_CHARS_HARD_MAX,
            budget_field="hits",
            on_truncate=_trim_search_response,
            original_query=request.query,
            match_mode=request.match_mode,
            total_hits=total_hits,
            returned_hits=len(hits),
            hits=hits,
            has_more=has_more,
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
    return [str(item) for item in parsed]


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
        chat=(row["chat_title"] or str(chat_id)),
        topic=topic,
        date=row["date"],
        sender=row["sender_name"],
        reply_to=row["reply_to"],
        text=text,
        original_text_chars=original_text_chars,
        text_truncated=text_truncated,
        transcript=transcript,
        transcript_original_chars=transcript_original,
        transcript_truncated=transcript_truncated,
        poll=_json_value(row["poll"]),
        media_kind=row["media_kind"],
        media_name=row["media_name"],
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
    with _readonly_connection(conn) as readonly:
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
        date=row["date"],
        sender=row["sender_name"],
        text=text,
        transcript=transcript,
    )


def get_context(
    request: GetContextInput,
    parsed_id: tuple[int, int],
    conn: sqlite3.Connection | None = None,
) -> GetContextOutput:
    """Return a bounded local context window around a public message ID."""
    with _readonly_connection(conn) as readonly:
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
