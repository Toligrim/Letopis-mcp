from dataclasses import replace
import time

import pytest

from tgarchive.db import connect
from tgarchive.mcp import retrieval, tools
from tgarchive.mcp.cursor import (
    CURSOR_TTL_SECONDS,
    CursorError,
    decode_cursor,
    encode_cursor,
    query_fingerprint,
)
from tgarchive.mcp.models import ErrorCode, ErrorResponse, SearchMessagesInput


def _ordinary_request(cursor: str | None = None) -> SearchMessagesInput:
    return SearchMessagesInput(
        query="обычное",
        strategy="relevance",
        limit=7,
        snippet_chars=120,
        include_total=True,
        cursor=cursor,
    )


def test_keyset_pages_cover_ordinary_messages_without_duplicates(synthetic_archive):
    connection = synthetic_archive.connection
    expected_ids = {row["id"] for row in synthetic_archive.cases["ordinary"]}
    assert len(expected_ids) == 50

    page1 = retrieval.search_messages(_ordinary_request(), conn=connection)
    assert page1.total_hits == 50
    assert page1.returned_hits == 7
    assert page1.has_more is True
    assert page1.next_cursor

    page2 = retrieval.search_messages(
        _ordinary_request(page1.next_cursor),
        conn=connection,
    )
    page1_ids = {hit.id for hit in page1.hits}
    page2_ids = {hit.id for hit in page2.hits}
    assert page1_ids.isdisjoint(page2_ids)
    assert len(page1_ids | page2_ids) == 14

    seen_ids = page1_ids | page2_ids
    cursor = page2.next_cursor
    while cursor is not None:
        page = retrieval.search_messages(_ordinary_request(cursor), conn=connection)
        current_ids = {hit.id for hit in page.hits}
        assert seen_ids.isdisjoint(current_ids)
        seen_ids |= current_ids
        if page.has_more:
            assert page.next_cursor
            cursor = page.next_cursor
        else:
            assert page.next_cursor is None
            cursor = None

    assert seen_ids == expected_ids


def test_invalid_cursors_are_unified_error_responses(synthetic_archive, monkeypatch):
    connection = synthetic_archive.connection
    first_page = retrieval.search_messages(_ordinary_request(), conn=connection)
    cursor = first_page.next_cursor
    assert cursor

    real_search = retrieval.search_messages

    def synthetic_search(request):
        return real_search(request, conn=connection)

    monkeypatch.setattr(tools.retrieval, "search_messages", synthetic_search)

    middle = len(cursor) // 2
    replacement = "A" if cursor[middle] != "A" else "B"
    tampered = cursor[:middle] + replacement + cursor[middle + 1:]
    result = tools.search_messages(_ordinary_request(tampered))
    assert isinstance(result, ErrorResponse)
    assert result.code == ErrorCode.INVALID_CURSOR

    wrong_query = tools.search_messages(
        replace(_ordinary_request(cursor), query="пагинация")
    )
    wrong_sort = tools.search_messages(
        replace(_ordinary_request(cursor), sort="oldest")
    )
    assert isinstance(wrong_query, ErrorResponse)
    assert wrong_query.code == ErrorCode.INVALID_CURSOR
    assert isinstance(wrong_sort, ErrorResponse)
    assert wrong_sort.code == ErrorCode.INVALID_CURSOR


def test_stale_and_expired_cursors_are_rejected_on_synthetic_db(synthetic_archive):
    connection = synthetic_archive.connection
    request = _ordinary_request()
    first_page = retrieval.search_messages(request, conn=connection)
    cursor = first_page.next_cursor
    assert cursor
    fingerprint = query_fingerprint(
        request.query,
        request.match_mode,
        request.filters,
        request.sort,
    )

    writer = connect(synthetic_archive.path)
    writer.execute(
        "INSERT INTO messages(chat_id,message_id,date,text) VALUES(?,?,?,?)",
        (-1000000000002, 900, "2026-01-01T00:00:00", "изменение индекса"),
    )
    writer.commit()
    writer.close()

    with pytest.raises(CursorError) as stale:
        decode_cursor(
            cursor,
            expected_query_fingerprint=fingerprint,
            expected_sort="relevance",
            conn=connection,
        )
    assert stale.value.code == ErrorCode.STALE_CURSOR

    expired = encode_cursor(
        query_fingerprint=fingerprint,
        sort="relevance",
        last_row_id=1,
        last_score=0.0,
        conn=connection,
        now=time.time() - CURSOR_TTL_SECONDS - 1,
    )
    with pytest.raises(CursorError) as expiry:
        decode_cursor(
            expired,
            expected_query_fingerprint=fingerprint,
            expected_sort="relevance",
            conn=connection,
        )
    assert expiry.value.code == ErrorCode.INVALID_CURSOR
