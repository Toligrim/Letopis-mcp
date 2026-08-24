from __future__ import annotations

from tgarchive.mcp import retrieval, tools
from tgarchive.mcp.models import (
    CONTEXT_BEFORE_AFTER_HARD_MAX,
    ErrorCode,
    ErrorResponse,
    GetContextInput,
    GetContextOutput,
)


_REAL_CONTEXT = retrieval.get_context


def _patch_context_to_synthetic(monkeypatch, synthetic_archive):
    def get_context(request, parsed_id):
        return _REAL_CONTEXT(request, parsed_id, conn=synthetic_archive.connection)

    monkeypatch.setattr(tools.retrieval, "get_context", get_context)


def _context_result(monkeypatch, synthetic_archive, request):
    _patch_context_to_synthetic(monkeypatch, synthetic_archive)
    result = tools.get_context(request)
    assert isinstance(result, GetContextOutput)
    return result


def _rows_in_scope(messages, *, chat_id, topic_id):
    return [
        row
        for row in messages
        if row["chat_id"] == chat_id and row["topic_id"] == topic_id
    ]


def test_context_same_topic_excludes_other_topics_and_whole_chat_includes_them(
    synthetic_archive,
    monkeypatch,
):
    topic_202 = _rows_in_scope(
        synthetic_archive.cases["ordinary"],
        chat_id=synthetic_archive.cases["ordinary"][0]["chat_id"],
        topic_id=202,
    )
    pivot = topic_202[0]

    same_topic = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(
            id=pivot["id"],
            before=5,
            after=3,
            same_topic=True,
        ),
    )
    assert same_topic.messages[0].relation == "pivot"
    assert all(message.topic_id == 202 for message in same_topic.messages)

    whole_chat = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(
            id=pivot["id"],
            before=5,
            after=3,
            same_topic=False,
        ),
    )
    assert any(
        message.relation != "pivot" and message.topic_id == 101
        for message in whole_chat.messages
    )


def test_context_general_scope_stays_null_topic(
    synthetic_archive,
    monkeypatch,
):
    general = [
        row
        for row in synthetic_archive.cases["ordinary"]
        if row["topic_id"] is None
    ]
    pivot = general[2]

    same_topic = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(
            id=pivot["id"],
            before=2,
            after=2,
            same_topic=True,
        ),
    )
    assert same_topic.messages[0].relation == "before"
    assert sum(message.relation == "pivot" for message in same_topic.messages) == 1
    assert all(message.topic_id is None for message in same_topic.messages)

    whole_chat = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(
            id=pivot["id"],
            before=5,
            after=2,
            same_topic=False,
        ),
    )
    assert any(
        message.relation != "pivot" and message.topic_id is not None
        for message in whole_chat.messages
    )


def test_context_window_hard_cap_is_rejected_and_pivot_flags_are_correct(
    synthetic_archive,
    monkeypatch,
):
    pivot = synthetic_archive.cases["ordinary"][12]
    oversized = tools.get_context(
        GetContextInput(id=pivot["id"], before=100, after=0),
    )
    assert isinstance(oversized, ErrorResponse)
    assert oversized.code == ErrorCode.INVALID_ARGUMENT

    bounded = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(
            id=pivot["id"],
            before=CONTEXT_BEFORE_AFTER_HARD_MAX,
            after=CONTEXT_BEFORE_AFTER_HARD_MAX,
        ),
    )
    assert sum(message.relation == "pivot" for message in bounded.messages) == 1
    assert len([message for message in bounded.messages if message.relation == "before"])
    assert len([message for message in bounded.messages if message.relation == "after"])
    assert len(bounded.messages) <= 2 * CONTEXT_BEFORE_AFTER_HARD_MAX + 1

    first = synthetic_archive.cases["ordinary"][0]
    boundary = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(id=first["id"], before=3, after=3),
    )
    assert boundary.has_more_before is False
    assert boundary.has_more_after is True

    middle = synthetic_archive.cases["ordinary"][12]
    middle_result = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(id=middle["id"], before=3, after=3),
    )
    assert middle_result.has_more_before is True
    assert middle_result.has_more_after is True


def test_context_budget_trims_edges_and_keeps_pivot(
    synthetic_archive,
    monkeypatch,
):
    hard_cap = 1_500
    monkeypatch.setattr(retrieval, "CONTEXT_RESPONSE_CHARS_HARD_MAX", hard_cap)
    pivot = synthetic_archive.cases["ordinary"][12]

    result = _context_result(
        monkeypatch,
        synthetic_archive,
        GetContextInput(id=pivot["id"], before=5, after=5),
    )

    assert result.response_chars <= hard_cap
    assert result.truncated is True
    assert sum(message.relation == "pivot" for message in result.messages) == 1
    assert any((result.has_more_before, result.has_more_after))
