from tgarchive.db import connect
from tgarchive.mcp import retrieval
from tgarchive.mcp.models import FetchMessagesInput, SearchMessagesInput


def test_search_budget_enforces_cap_and_utf8_metrics(synthetic_archive, monkeypatch):
    hard_cap = 2_500
    monkeypatch.setattr(retrieval, "SEARCH_RESPONSE_CHARS_HARD_MAX", hard_cap)

    result = retrieval.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            limit=50,
            snippet_chars=120,
            include_total=True,
        ),
        conn=synthetic_archive.connection,
    )

    assert result.response_chars <= hard_cap
    assert result.truncated is True
    assert result.returned_hits < 50
    assert result.has_more is True
    assert result.response_bytes_utf8 > result.response_chars
    assert result.estimated_tokens_rough == result.response_chars // 4


def test_fetch_per_message_cap_truncates_long_transcript_honestly(synthetic_archive, monkeypatch):
    chat_id = synthetic_archive.cases["reactions"]["chat_id"]
    message_id = 900
    public_id = f"tg:{chat_id}:{message_id}"
    transcript = "Длинная транскрипция голосового сообщения. " * 400

    writer = connect(synthetic_archive.path)
    writer.execute(
        "INSERT INTO messages("
        "chat_id,message_id,sender_id,sender_name,date,text,transcript"
        ") VALUES(?,?,?,?,?,?,?)",
        (
            chat_id,
            message_id,
            99,
            "Тестовый автор",
            "2026-01-01T00:00:00",
            "Короткий текст",
            transcript,
        ),
    )
    writer.commit()
    writer.close()

    # Keep the response cap above this one bounded message; the artificial
    # per-message cap is what should trim the transcript field itself.
    monkeypatch.setattr(retrieval, "FETCH_RESPONSE_CHARS_HARD_MAX", 2_000)
    result = retrieval.fetch_messages(
        FetchMessagesInput(
            ids=[public_id],
            include_transcript=True,
            include_links=False,
            per_message_max_chars=500,
        ),
        [(chat_id, message_id)],
        conn=synthetic_archive.connection,
    )

    assert result.response_chars <= 2_000
    assert result.truncated is False
    assert result.omitted_ids == []
    assert len(result.messages) == 1
    fetched = result.messages[0]
    assert fetched.id == public_id
    assert fetched.text == "Короткий текст"
    assert fetched.text_truncated is False
    assert fetched.transcript_original_chars == len(transcript)
    assert len(fetched.transcript) == 500
    assert fetched.transcript_truncated is True
