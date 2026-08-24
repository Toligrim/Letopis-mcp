from __future__ import annotations

from tgarchive.mcp import retrieval, tools
from tgarchive.db import connect
from tgarchive.mcp.models import (
    ErrorCode,
    ErrorResponse,
    FetchMessagesInput,
    FetchMessagesOutput,
)


_REAL_FETCH = retrieval.fetch_messages


def _patch_fetch_to_synthetic(monkeypatch, synthetic_archive):
    def fetch(request, parsed_ids):
        return _REAL_FETCH(request, parsed_ids, conn=synthetic_archive.connection)

    monkeypatch.setattr(tools.retrieval, "fetch_messages", fetch)


def _fetch_result(monkeypatch, synthetic_archive, request):
    _patch_fetch_to_synthetic(monkeypatch, synthetic_archive)
    result = tools.fetch_messages(request)
    assert isinstance(result, FetchMessagesOutput)
    return result


def test_fetch_validates_public_id_format(synthetic_archive):
    result = tools.fetch_messages(FetchMessagesInput(ids=["not-a-telegram-id"]))

    assert isinstance(result, ErrorResponse)
    assert result.code == ErrorCode.INVALID_ARGUMENT


def test_fetch_returns_missing_ids_as_omitted(synthetic_archive, monkeypatch):
    existing = synthetic_archive.cases["transcript_only"]
    missing = f"tg:{existing['chat_id']}:{existing['message_id'] + 999999}"
    result = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(ids=[existing["id"], missing], include_links=False),
    )

    assert [message.id for message in result.messages] == [existing["id"]]
    assert result.omitted_ids == [missing]


def test_fetch_transcript_is_disabled_by_default_but_counted(
    synthetic_archive,
    monkeypatch,
):
    case = synthetic_archive.cases["transcript_only"]
    result = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(ids=[case["id"]], include_links=False),
    )
    message = result.messages[0]

    assert message.transcript is None
    assert message.transcript_original_chars == len(case["transcript"])
    assert message.transcript_truncated is False

    with_transcript = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(
            ids=[case["id"]],
            include_transcript=True,
            include_links=False,
        ),
    )
    assert with_transcript.messages[0].transcript == case["transcript"]


def test_fetch_reactions_and_per_message_truncation(
    synthetic_archive,
    monkeypatch,
):
    reaction_case = synthetic_archive.cases["reactions"]
    reactions = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(
            ids=[reaction_case["id"]],
            include_links=False,
            include_reactions=True,
        ),
    )
    assert reactions.messages[0].reactions == {"👍": 3, "❤️": 1}

    text_case = synthetic_archive.cases["ordinary"][0]
    long_text = "Длинный текст для проверки ограничения. " * 80
    writer = connect(synthetic_archive.path)
    writer.execute(
        "UPDATE messages SET text=? WHERE chat_id=? AND message_id=?",
        (long_text, text_case["chat_id"], text_case["message_id"]),
    )
    writer.commit()
    writer.close()

    max_chars = 1_000
    truncated = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(
            ids=[text_case["id"]],
            include_links=False,
            per_message_max_chars=max_chars,
        ),
    )
    message = truncated.messages[0]
    assert message.text == long_text[:max_chars]
    assert message.original_text_chars == len(long_text)
    assert message.text_truncated is True


def test_fetch_total_budget_omits_only_existing_messages(
    synthetic_archive,
    monkeypatch,
):
    requested_ids = [row["id"] for row in synthetic_archive.cases["ordinary"][:8]]
    hard_cap = 1_800
    monkeypatch.setattr(retrieval, "FETCH_RESPONSE_CHARS_HARD_MAX", hard_cap)

    result = _fetch_result(
        monkeypatch,
        synthetic_archive,
        FetchMessagesInput(ids=requested_ids, include_links=False),
    )

    returned_ids = {message.id for message in result.messages}
    omitted_ids = set(result.omitted_ids)
    assert result.truncated is True
    assert result.response_chars <= hard_cap
    assert result.messages
    assert result.omitted_ids
    assert returned_ids.isdisjoint(omitted_ids)
    assert returned_ids | omitted_ids == set(requested_ids)
