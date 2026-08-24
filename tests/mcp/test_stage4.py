from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from tgarchive.db import bump_index_revision, connect
from tgarchive.indexer import _fts_set, compose_index_text
from tgarchive.lemma import Lemmatizer
from tgarchive.mcp import retrieval, server as server_module, tools
from tgarchive.mcp.models import (
    AggregateMessagesInput,
    ArchiveOverviewInput,
    ErrorCode,
    ErrorResponse,
    FetchMessagesInput,
    GetContextInput,
    SearchFilters,
    SearchMessagesInput,
)


MEDIA_CHAT_ID = -1000000000001
MEDIA_QUERY = "stage4media"


def _insert_media_row(
    conn,
    lem: Lemmatizer,
    *,
    message_id: int,
    media_type: str | None,
    media_kind: str | None,
    poll: str | None = None,
) -> None:
    text = f"{MEDIA_QUERY} fixture message {message_id}"
    row = conn.execute(
        "INSERT INTO messages(chat_id,message_id,sender_id,sender_name,date,text,"
        "media_type,media_kind,poll) VALUES(?,?,?,?,?,?,?,?,?)",
        (
            MEDIA_CHAT_ID,
            message_id,
            99,
            "Stage 4",
            f"2026-04-{message_id - 699:02d}T10:00:00",
            text,
            media_type,
            media_kind,
            poll,
        ),
    )
    _fts_set(conn, lem, int(row.lastrowid), compose_index_text(text, poll))


def _seed_media_rows(path: Path) -> None:
    writable = connect(path)
    lem = Lemmatizer()
    try:
        rows = [
            (700, "MessageMediaDocument", "voice"),
            (701, "MessageMediaDocument", "audio"),
            (702, "MessageMediaDocument", "video"),
            (703, "MessageMediaDocument", "video_note"),
            (704, "MessageMediaDocument", "sticker"),
            (705, "MessageMediaDocument", "gif"),
            (706, "MessageMediaDocument", "document"),
            (707, "MessageMediaPhoto", "photo"),
        ]
        for message_id, media_type, media_kind in rows:
            _insert_media_row(
                writable,
                lem,
                message_id=message_id,
                media_type=media_type,
                media_kind=media_kind,
            )
        _insert_media_row(
            writable,
            lem,
            message_id=708,
            media_type="MessageMediaPoll",
            media_kind=None,
            poll='{"question":"stage 4 poll","answers":[]}',
        )
        _insert_media_row(
            writable,
            lem,
            message_id=709,
            media_type="MessageMediaWebPage",
            media_kind=None,
        )
        # A legacy/sidecar-shaped row proves that any/none do not rely only
        # on media_type when media_kind is already classified.
        _insert_media_row(
            writable,
            lem,
            message_id=710,
            media_type=None,
            media_kind="audio",
        )
        _insert_media_row(
            writable,
            lem,
            message_id=711,
            media_type=None,
            media_kind=None,
        )
        bump_index_revision(writable)
        writable.commit()
    finally:
        writable.close()


@pytest.mark.parametrize(
    ("media", "expected_ids"),
    [
        ("voice", {700}),
        ("audio", {701, 710}),
        ("video", {702}),
        ("video_note", {703}),
        ("sticker", {704}),
        ("gif", {705}),
        ("document", {706}),
        ("photo", {707}),
        ("poll", {708}),
        ("webpage", {709}),
        ("any", set(range(700, 711))),
        ("none", {711}),
    ],
)
def test_media_filter_uses_public_media_semantics(
    synthetic_archive,
    media: str,
    expected_ids: set[int],
) -> None:
    _seed_media_rows(synthetic_archive.path)

    result = retrieval.search_messages(
        SearchMessagesInput(
            query=MEDIA_QUERY,
            strategy="relevance",
            limit=50,
            snippet_chars=120,
            filters=SearchFilters(media=media),
        ),
        conn=synthetic_archive.connection,
    )

    assert {hit.message_id for hit in result.hits} == expected_ids
    assert result.total_hits == len(expected_ids)


def _stub_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.retrieval, "search_messages", lambda request: request)


@pytest.mark.parametrize(
    "filters",
    [
        SearchFilters(chat_ids=list(range(21))),
        SearchFilters(topic_ids=list(range(21))),
        SearchFilters(sender_name="x" * 121),
        SearchFilters(date_from="2024-01-01T"),
        SearchFilters(date_to="2024-01-01T"),
        SearchFilters(date_from="2024/01"),
        SearchFilters(media="something_invalid"),
    ],
)
def test_search_filter_bounds_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
    filters: SearchFilters,
) -> None:
    _stub_search(monkeypatch)
    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            filters=filters,
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code is ErrorCode.INVALID_ARGUMENT


def test_cursor_length_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_search(monkeypatch)
    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            cursor="x" * 2049,
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code is ErrorCode.INVALID_ARGUMENT


def test_filter_and_cursor_values_at_limits_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_search(monkeypatch)
    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            cursor="x" * 2048,
            filters=SearchFilters(
                chat_ids=list(range(20)),
                topic_ids=list(range(20)),
                sender_name="x" * 120,
                date_from="2024-01-01",
                date_to="2024-12-31",
                media="voice",
            ),
        )
    )

    assert not isinstance(result, ErrorResponse)


def test_archive_chat_ids_limit_is_enforced_at_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools.retrieval, "archive_overview", lambda request: request)

    valid = tools.archive_overview(ArchiveOverviewInput(chat_ids=list(range(50))))
    invalid = tools.archive_overview(ArchiveOverviewInput(chat_ids=list(range(51))))

    assert not isinstance(valid, ErrorResponse)
    assert isinstance(invalid, ErrorResponse)
    assert invalid.code is ErrorCode.INVALID_ARGUMENT


def test_search_schema_exposes_filter_and_cursor_constraints() -> None:
    mcp_server = MCPServer(name="schema-test", version="0")
    mcp_server.add_tool(server_module.search_messages, structured_output=True)
    tool = asyncio.run(mcp_server.list_tools())[0]
    schema = tool.input_schema

    filter_ref = schema["properties"]["filters"]["anyOf"][0]["$ref"]
    filter_schema = schema["$defs"][filter_ref.rsplit("/", 1)[-1]]
    media_schema = filter_schema["properties"]["media"]["anyOf"][0]

    assert set(media_schema["enum"]) == {
        "voice",
        "audio",
        "video",
        "video_note",
        "sticker",
        "gif",
        "document",
        "photo",
        "poll",
        "webpage",
        "any",
        "none",
    }
    assert filter_schema["properties"]["chat_ids"]["anyOf"][0]["maxItems"] == 20
    assert filter_schema["properties"]["topic_ids"]["anyOf"][0]["maxItems"] == 20
    assert filter_schema["properties"]["sender_name"]["anyOf"][0]["maxLength"] == 120
    assert filter_schema["properties"]["date_from"]["anyOf"][0]["maxLength"] == 10
    assert (
        filter_schema["properties"]["date_from"]["anyOf"][0]["pattern"]
        == r"^\d{4}(?:-\d{2}){0,2}$"
    )
    assert schema["properties"]["cursor"]["anyOf"][0]["maxLength"] == 2048


def test_aggregate_filter_validation_uses_the_same_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools.retrieval, "aggregate_messages", lambda request: request)
    result = tools.aggregate_messages(
        AggregateMessagesInput(
            query="пагинация",
            group_by="month",
            filters=SearchFilters(topic_ids=list(range(21))),
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code is ErrorCode.INVALID_ARGUMENT


def test_fetch_and_context_character_minimums_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools.retrieval,
        "fetch_messages",
        lambda request, parsed_ids: request,
    )
    monkeypatch.setattr(
        tools.retrieval,
        "get_context",
        lambda request, parsed_id: request,
    )

    valid_fetch = tools.fetch_messages(
        FetchMessagesInput(ids=["tg:-100:1"], per_message_max_chars=1_000)
    )
    invalid_fetch = tools.fetch_messages(
        FetchMessagesInput(ids=["tg:-100:1"], per_message_max_chars=999)
    )
    valid_context = tools.get_context(
        GetContextInput(id="tg:-100:1", message_max_chars=500)
    )
    invalid_context = tools.get_context(
        GetContextInput(id="tg:-100:1", message_max_chars=499)
    )

    assert isinstance(valid_fetch, FetchMessagesInput)
    assert isinstance(invalid_fetch, ErrorResponse)
    assert invalid_fetch.code is ErrorCode.INVALID_ARGUMENT
    assert isinstance(valid_context, GetContextInput)
    assert isinstance(invalid_context, ErrorResponse)
    assert invalid_context.code is ErrorCode.INVALID_ARGUMENT
