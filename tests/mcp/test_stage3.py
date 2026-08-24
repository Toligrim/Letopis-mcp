from __future__ import annotations

from pathlib import Path

import pytest

from tgarchive.db import bump_index_revision, connect
from tgarchive.indexer import _fts_set, compose_index_text
from tgarchive.lemma import Lemmatizer
from tgarchive.mcp import retrieval, tools
from tgarchive.mcp.models import (
    AggregateMessagesInput,
    ErrorCode,
    ErrorResponse,
    SearchMessagesInput,
)


@pytest.mark.parametrize(
    ("sort", "cursor"),
    [
        ("relevance", "opaque-cursor"),
        ("oldest", None),
        ("newest", None),
    ],
)
def test_diverse_rejects_cursor_and_explicit_date_sort(
    monkeypatch: pytest.MonkeyPatch,
    sort: str,
    cursor: str | None,
) -> None:
    """The public validation boundary must reject ignored diverse arguments."""

    monkeypatch.setattr(tools.retrieval, "search_messages", lambda request: request)
    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="diverse",
            sort=sort,
            cursor=cursor,
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code is ErrorCode.INVALID_ARGUMENT


def test_diverse_reports_more_without_a_pagination_cursor(synthetic_archive) -> None:
    result = retrieval.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="diverse",
            sort="relevance",
            limit=3,
            include_total=True,
        ),
        conn=synthetic_archive.conn,
    )

    assert result.total_hits is not None
    assert result.total_hits > result.returned_hits
    assert result.has_more is True
    assert result.next_cursor is None
    assert result.pagination_supported is False


def test_relevance_advertises_cursor_pagination(synthetic_archive) -> None:
    result = retrieval.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            sort="relevance",
            limit=3,
            include_total=True,
        ),
        conn=synthetic_archive.conn,
    )

    assert result.pagination_supported is True


TOPIC_CHAT_A = -1000000000101
TOPIC_CHAT_B = -1000000000102


def _insert_collision_message(
    conn,
    lem: Lemmatizer,
    *,
    chat_id: int,
    message_id: int,
    topic_id: int | None,
    date: str,
) -> None:
    text = f"collision marker for chat {chat_id} message {message_id}"
    row = conn.execute(
        "INSERT INTO messages(chat_id,message_id,sender_id,sender_name,date,topic_id,text) "
        "VALUES(?,?,?,?,?,?,?)",
        (chat_id, message_id, 7, "Тест", date, topic_id, text),
    )
    _fts_set(conn, lem, int(row.lastrowid), compose_index_text(text))


def _seed_topic_collision(path: Path) -> None:
    writable = connect(path)
    lem = Lemmatizer()
    try:
        writable.executemany(
            "INSERT INTO chats(chat_id,title,type,is_forum) VALUES(?,?,?,?)",
            [
                (TOPIC_CHAT_A, "Чат A", "supergroup", 1),
                (TOPIC_CHAT_B, "Чат B", "supergroup", 1),
            ],
        )
        writable.executemany(
            "INSERT INTO topics(chat_id,topic_id,title) VALUES(?,?,?)",
            [
                (TOPIC_CHAT_A, 100, "Топик A"),
                (TOPIC_CHAT_B, 100, "Топик B"),
            ],
        )

        for message_id in (1, 2):
            _insert_collision_message(
                writable,
                lem,
                chat_id=TOPIC_CHAT_A,
                message_id=message_id,
                topic_id=100,
                date=f"2024-01-0{message_id}T10:00:00",
            )
        _insert_collision_message(
            writable,
            lem,
            chat_id=TOPIC_CHAT_A,
            message_id=3,
            topic_id=None,
            date="2024-01-03T11:00:00",
        )

        for message_id in (1, 2, 3):
            _insert_collision_message(
                writable,
                lem,
                chat_id=TOPIC_CHAT_B,
                message_id=message_id,
                topic_id=100,
                date=f"2024-02-0{message_id}T10:00:00",
            )
        _insert_collision_message(
            writable,
            lem,
            chat_id=TOPIC_CHAT_B,
            message_id=4,
            topic_id=None,
            date="2024-02-04T11:00:00",
        )
        bump_index_revision(writable)
        writable.commit()
    finally:
        writable.close()


def test_topic_aggregation_is_chat_scoped_and_keyset_paginates(synthetic_archive) -> None:
    _seed_topic_collision(synthetic_archive.path)

    groups = []
    cursor = None
    for _ in range(10):
        result = retrieval.aggregate_messages(
            AggregateMessagesInput(
                query="collision",
                group_by="topic",
                limit=1,
                cursor=cursor,
            ),
            conn=synthetic_archive.conn,
        )
        groups.extend(result.groups)
        if not result.has_more:
            break
        assert result.next_cursor is not None
        cursor = result.next_cursor
    else:
        pytest.fail("topic aggregation did not terminate")

    assert len(groups) == 4
    assert len({group.key for group in groups}) == 4
    assert all(
        group.key
        == f"{group.chat_id}:{'null' if group.topic_id is None else group.topic_id}"
        for group in groups
    )

    topic_groups = {
        (group.chat_id, group.topic_id): group
        for group in groups
        if group.topic_id == 100
    }
    assert set(topic_groups) == {
        (TOPIC_CHAT_A, 100),
        (TOPIC_CHAT_B, 100),
    }
    assert topic_groups[(TOPIC_CHAT_A, 100)].count == 2
    assert topic_groups[(TOPIC_CHAT_B, 100)].count == 3
    assert topic_groups[(TOPIC_CHAT_A, 100)].chat_title == "Чат A"
    assert topic_groups[(TOPIC_CHAT_A, 100)].topic_title == "Топик A"
    assert topic_groups[(TOPIC_CHAT_B, 100)].chat_title == "Чат B"
    assert topic_groups[(TOPIC_CHAT_B, 100)].topic_title == "Топик B"

    general_groups = {
        (group.chat_id, group.topic_id): group
        for group in groups
        if group.topic_id is None
    }
    assert set(general_groups) == {
        (TOPIC_CHAT_A, None),
        (TOPIC_CHAT_B, None),
    }
    assert all(group.count == 1 for group in general_groups.values())
