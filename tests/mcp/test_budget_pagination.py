from dataclasses import replace

from tgarchive.mcp import retrieval
from tgarchive.mcp.models import (
    AggregateMessagesInput,
    ArchiveOverviewInput,
    SearchMessagesInput,
)


def _collect_search_ids(request, connection) -> list[str]:
    ids: list[str] = []
    cursor = None
    while True:
        page = retrieval.search_messages(
            replace(request, cursor=cursor),
            conn=connection,
        )
        ids.extend(hit.id for hit in page.hits)
        if page.next_cursor is None:
            return ids
        cursor = page.next_cursor


def test_search_cursor_follows_last_hit_after_budget_truncation(
    synthetic_archive,
    monkeypatch,
):
    request = SearchMessagesInput(
        query="пагинация",
        strategy="relevance",
        sort="relevance",
        limit=7,
        snippet_chars=120,
        include_total=False,
    )
    expected = _collect_search_ids(request, synthetic_archive.connection)
    default_cap = retrieval.SEARCH_RESPONSE_CHARS_HARD_MAX

    monkeypatch.setattr(retrieval, "SEARCH_RESPONSE_CHARS_HARD_MAX", 1_000)
    first = retrieval.search_messages(request, conn=synthetic_archive.connection)
    assert first.truncated is True
    assert first.has_more is True
    assert 0 < len(first.hits) < request.limit
    assert first.next_cursor

    monkeypatch.setattr(retrieval, "SEARCH_RESPONSE_CHARS_HARD_MAX", default_cap)
    second = retrieval.search_messages(
        replace(request, cursor=first.next_cursor),
        conn=synthetic_archive.connection,
    )
    combined = [hit.id for hit in first.hits] + [hit.id for hit in second.hits]

    assert len(combined) == len(set(combined))
    assert combined == expected[: len(combined)]


def test_aggregate_cursor_follows_last_group_after_budget_truncation(
    synthetic_archive,
    monkeypatch,
):
    request = AggregateMessagesInput(
        query="пагинация",
        group_by="month",
        limit=5,
    )
    expected = [
        group.key
        for group in retrieval.aggregate_messages(
            request,
            conn=synthetic_archive.connection,
        ).groups
    ]
    default_cap = retrieval.AGGREGATE_RESPONSE_CHARS_HARD_MAX

    monkeypatch.setattr(retrieval, "AGGREGATE_RESPONSE_CHARS_HARD_MAX", 700)
    first = retrieval.aggregate_messages(request, conn=synthetic_archive.connection)
    assert first.truncated is True
    assert first.has_more is True
    assert 0 < len(first.groups) < request.limit
    assert first.next_cursor

    monkeypatch.setattr(retrieval, "AGGREGATE_RESPONSE_CHARS_HARD_MAX", default_cap)
    second = retrieval.aggregate_messages(
        replace(request, cursor=first.next_cursor),
        conn=synthetic_archive.connection,
    )
    combined = [group.key for group in first.groups] + [group.key for group in second.groups]

    assert len(combined) == len(set(combined))
    assert combined == expected[: len(combined)]


def test_archive_cursor_follows_last_chat_after_budget_truncation(
    synthetic_archive,
    monkeypatch,
):
    request = ArchiveOverviewInput(limit=3, include_topics=True)
    expected = [
        chat.chat_id
        for chat in retrieval.archive_overview(
            request,
            conn=synthetic_archive.connection,
        ).chats
    ]
    default_cap = retrieval.ARCHIVE_RESPONSE_CHARS_HARD_MAX

    monkeypatch.setattr(retrieval, "ARCHIVE_RESPONSE_CHARS_HARD_MAX", 900)
    first = retrieval.archive_overview(request, conn=synthetic_archive.connection)
    assert first.truncated is True
    assert first.has_more is True
    assert 0 < len(first.chats) < request.limit
    assert first.next_cursor

    monkeypatch.setattr(retrieval, "ARCHIVE_RESPONSE_CHARS_HARD_MAX", default_cap)
    second = retrieval.archive_overview(
        replace(request, cursor=first.next_cursor),
        conn=synthetic_archive.connection,
    )
    combined = [chat.chat_id for chat in first.chats] + [
        chat.chat_id for chat in second.chats
    ]

    assert len(combined) == len(set(combined))
    assert combined == expected[: len(combined)]
