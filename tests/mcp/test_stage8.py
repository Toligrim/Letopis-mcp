from __future__ import annotations

from dataclasses import replace

from tgarchive.db import connect
from tgarchive.mcp import retrieval, tools
from tgarchive.mcp.models import (
    AggregateMessagesInput,
    ArchiveOverviewInput,
    METADATA_STRING_MAX_CHARS,
    ErrorCode,
    ErrorResponse,
    FetchMessagesInput,
    GetContextInput,
    SearchMessagesInput,
)


def _make_metadata_pathological(synthetic_archive) -> tuple[str, str, str]:
    chat_id = synthetic_archive.cases["ordinary"][0]["chat_id"]
    topic_id = synthetic_archive.cases["ordinary"][0]["topic_id"]
    message_id = synthetic_archive.cases["ordinary"][0]["message_id"]
    media_case = synthetic_archive.cases["media_only"][0]
    long_chat_title = "Ч" * 10_000
    long_topic_title = "Т" * 10_000
    long_sender_name = "С" * 10_000
    long_media_name = "М" * 10_000

    writer = connect(synthetic_archive.path)
    writer.execute("UPDATE chats SET title=? WHERE chat_id=?", (long_chat_title, chat_id))
    writer.execute(
        "UPDATE topics SET title=? WHERE chat_id=? AND topic_id=?",
        (long_topic_title, chat_id, topic_id),
    )
    writer.execute(
        "UPDATE messages SET sender_name=? WHERE chat_id=? AND message_id IN (?, ?)",
        (long_sender_name, chat_id, message_id, media_case["message_id"]),
    )
    writer.execute(
        "UPDATE messages SET media_name=? WHERE chat_id=? AND message_id=?",
        (long_media_name, media_case["chat_id"], media_case["message_id"]),
    )
    writer.commit()
    writer.close()
    return (
        f"tg:{chat_id}:{message_id}",
        f"tg:{media_case['chat_id']}:{media_case['message_id']}",
        long_sender_name,
    )


def test_metadata_strings_are_capped_across_retrieval_outputs(synthetic_archive):
    ordinary_id, media_id, _ = _make_metadata_pathological(synthetic_archive)

    search = retrieval.search_messages(
        SearchMessagesInput(
            query="якорь",
            strategy="relevance",
            sort="relevance",
            limit=1,
            snippet_chars=120,
        ),
        conn=synthetic_archive.connection,
    )
    hit = next(item for item in search.hits if item.id == ordinary_id)
    assert len(hit.chat_title) == METADATA_STRING_MAX_CHARS
    assert len(hit.topic_title) == METADATA_STRING_MAX_CHARS
    assert len(hit.sender) == METADATA_STRING_MAX_CHARS
    assert len(hit.snippet) <= 120

    aggregate = retrieval.aggregate_messages(
        AggregateMessagesInput(
            query="пагинация",
            group_by="topic",
            limit=100,
        ),
        conn=synthetic_archive.connection,
    )
    topic_group = next(group for group in aggregate.groups if group.key.endswith(":101"))
    assert len(topic_group.chat_title) == METADATA_STRING_MAX_CHARS
    assert len(topic_group.topic_title) == METADATA_STRING_MAX_CHARS

    overview = retrieval.archive_overview(
        ArchiveOverviewInput(
            include_topics=True,
            limit=50,
        ),
        conn=synthetic_archive.connection,
    )
    chat = next(item for item in overview.chats if item.chat_id == synthetic_archive.cases["ordinary"][0]["chat_id"])
    assert len(chat.title) == METADATA_STRING_MAX_CHARS
    assert len(next(topic for topic in chat.topics if topic.topic_id == 101).title) == METADATA_STRING_MAX_CHARS

    context = retrieval.get_context(
        GetContextInput(
            id=ordinary_id,
            before=0,
            after=0,
            message_max_chars=120,
        ),
        (synthetic_archive.cases["ordinary"][0]["chat_id"], synthetic_archive.cases["ordinary"][0]["message_id"]),
        conn=synthetic_archive.connection,
    )
    assert len(context.messages[0].sender) == METADATA_STRING_MAX_CHARS
    assert len(context.messages[0].text) <= 120

    fetched = retrieval.fetch_messages(
        FetchMessagesInput(ids=[ordinary_id, media_id], include_links=False),
        [
            (synthetic_archive.cases["ordinary"][0]["chat_id"], synthetic_archive.cases["ordinary"][0]["message_id"]),
            (synthetic_archive.cases["media_only"][0]["chat_id"], synthetic_archive.cases["media_only"][0]["message_id"]),
        ],
        conn=synthetic_archive.connection,
    )
    ordinary = next(message for message in fetched.messages if message.id == ordinary_id)
    media = next(message for message in fetched.messages if message.id == media_id)
    assert len(ordinary.chat) == METADATA_STRING_MAX_CHARS
    assert len(ordinary.sender) == METADATA_STRING_MAX_CHARS
    assert len(media.media_name) == METADATA_STRING_MAX_CHARS


def test_impossible_hard_budget_returns_structured_error(synthetic_archive, monkeypatch):
    real_search = retrieval.search_messages

    def search_on_fixture(request):
        return real_search(request, conn=synthetic_archive.connection)

    monkeypatch.setattr(tools.retrieval, "search_messages", search_on_fixture)
    monkeypatch.setattr(retrieval, "SEARCH_RESPONSE_CHARS_HARD_MAX", 1)

    result = tools.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            sort="relevance",
            limit=1,
            snippet_chars=120,
        )
    )

    assert isinstance(result, ErrorResponse)
    assert result.code == ErrorCode.OUTPUT_BUDGET_EXCEEDED
    assert result.retryable is True
    assert result.details["hard_max_chars"] == 1
    assert result.details["response_chars"] > result.details["hard_max_chars"]


def test_chronological_cursor_keeps_raw_date_after_metadata_cap(synthetic_archive):
    target = synthetic_archive.cases["ordinary"][0]
    raw_date = "0000-01-01T00:00:00" + ("x" * 10_000)
    writer = connect(synthetic_archive.path)
    writer.execute(
        "UPDATE messages SET date=? WHERE chat_id=? AND message_id=?",
        (raw_date, target["chat_id"], target["message_id"]),
    )
    writer.commit()
    writer.close()

    request = SearchMessagesInput(
        query="пагинация",
        strategy="relevance",
        sort="oldest",
        limit=1,
        snippet_chars=120,
        include_total=False,
    )
    first = retrieval.search_messages(request, conn=synthetic_archive.connection)
    assert first.hits[0].id == target["id"]
    assert len(first.hits[0].date) == METADATA_STRING_MAX_CHARS
    assert first.next_cursor

    second = retrieval.search_messages(
        replace(request, cursor=first.next_cursor),
        conn=synthetic_archive.connection,
    )
    assert target["id"] not in {hit.id for hit in second.hits}
