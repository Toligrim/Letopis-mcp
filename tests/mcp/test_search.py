import re

from tgarchive.mcp import retrieval
from tgarchive.mcp.models import SearchFilters, SearchMessagesInput


def _request(query: str, **kwargs) -> SearchMessagesInput:
    return SearchMessagesInput(
        query=query,
        strategy="relevance",
        snippet_chars=120,
        include_total=True,
        **kwargs,
    )


def _pagination_rows(synthetic_archive):
    return [*synthetic_archive.cases["ordinary"], *synthetic_archive.cases["burst"]]


def test_match_modes_total_hits_limit_and_public_ids(synthetic_archive):
    connection = synthetic_archive.connection

    and_result = retrieval.search_messages(
        _request("пагинация коротком", limit=20),
        conn=connection,
    )
    assert and_result.total_hits == 6
    assert and_result.returned_hits == 6
    assert {hit.id for hit in and_result.hits} == {
        row["id"] for row in synthetic_archive.cases["burst"]
    }

    or_result = retrieval.search_messages(
        _request("пагинация разнообразия", match_mode="or", limit=50),
        conn=connection,
    )
    assert or_result.total_hits == 60  # 56 pagination hits + 4 duplicate/near-duplicate hits.
    assert or_result.returned_hits == 50

    boolean_or = retrieval.search_messages(
        _request("пагинация OR разнообразия", match_mode="boolean", limit=50),
        conn=connection,
    )
    boolean_and = retrieval.search_messages(
        _request("пагинация AND разнообразия", match_mode="boolean", limit=20),
        conn=connection,
    )
    assert boolean_or.total_hits == 60
    assert boolean_and.total_hits == 0
    assert boolean_and.returned_hits == 0

    small = retrieval.search_messages(
        _request("пагинация", limit=3),
        conn=connection,
    )
    large = retrieval.search_messages(
        _request("пагинация", limit=50),
        conn=connection,
    )
    assert small.total_hits == large.total_hits == 56
    assert small.returned_hits == 3
    assert large.returned_hits == 50

    expected_ids = {row["id"] for row in _pagination_rows(synthetic_archive)}
    known_ids = {row["id"] for row in synthetic_archive.messages}
    assert len(expected_ids) == 56
    assert all(re.fullmatch(r"tg:-?\d+:\d+", hit.id) for hit in large.hits)
    assert {hit.id for hit in large.hits} <= expected_ids <= known_ids


def test_filters_cover_chat_topic_general_sender_and_dates(synthetic_archive):
    connection = synthetic_archive.connection
    rows = _pagination_rows(synthetic_archive)

    work_chat_id = synthetic_archive.cases["ordinary"][0]["chat_id"]
    chat_expected = [row for row in rows if row["chat_id"] == work_chat_id]
    chat_result = retrieval.search_messages(
        _request(
            "пагинация",
            limit=50,
            filters=SearchFilters(chat_ids=[work_chat_id]),
        ),
        conn=connection,
    )
    assert chat_result.total_hits == len(chat_expected) == 46
    assert {hit.chat_id for hit in chat_result.hits} == {work_chat_id}

    general_expected = [row for row in rows if row["topic_id"] is None]
    general_result = retrieval.search_messages(
        _request(
            "пагинация",
            limit=50,
            filters=SearchFilters(topic_ids=[0]),
        ),
        conn=connection,
    )
    assert general_result.total_hits == len(general_expected) == 15
    assert all(hit.topic_id is None for hit in general_result.hits)

    sender_id_expected = [row for row in rows if row["sender_id"] == 11]
    sender_id_result = retrieval.search_messages(
        _request(
            "пагинация",
            limit=50,
            filters=SearchFilters(sender_id=11),
        ),
        conn=connection,
    )
    assert sender_id_result.total_hits == len(sender_id_expected)
    assert all(hit.sender_id == 11 for hit in sender_id_result.hits)

    sender_name_expected = [row for row in rows if row["sender_name"] == "Алиса"]
    sender_name_result = retrieval.search_messages(
        _request(
            "пагинация",
            limit=50,
            filters=SearchFilters(sender_name="Алиса"),
        ),
        conn=connection,
    )
    assert sender_name_result.total_hits == len(sender_name_expected)
    assert all(hit.sender == "Алиса" for hit in sender_name_result.hits)

    date_expected = [
        row
        for row in rows
        if "2024-01" <= row["date"] < "2024-11"
    ]
    date_result = retrieval.search_messages(
        _request(
            "пагинация",
            limit=50,
            filters=SearchFilters(date_from="2024-01", date_to="2024-10"),
        ),
        conn=connection,
    )
    assert date_result.total_hits == len(date_expected) == 26
    assert all("2024-01" <= hit.date < "2024-11" for hit in date_result.hits)


def test_relevance_scores_and_chronological_sorting(synthetic_archive):
    connection = synthetic_archive.connection
    metadata = {row["id"]: row for row in synthetic_archive.messages}

    relevance = retrieval.search_messages(
        _request("пагинация разнообразия", match_mode="or", limit=50),
        conn=connection,
    )
    scores = [hit.bm25_score for hit in relevance.hits]
    assert scores
    assert all(isinstance(score, float) for score in scores)
    assert any(score != 0.0 for score in scores)
    assert scores == sorted(scores)
    assert relevance.score_semantics

    oldest = retrieval.search_messages(
        _request("пагинация", sort="oldest", limit=50),
        conn=connection,
    )
    oldest_keys = [(hit.date, metadata[hit.id]["db_id"]) for hit in oldest.hits]
    assert oldest_keys == sorted(oldest_keys)
    assert all(hit.bm25_score is None for hit in oldest.hits)
    assert oldest.score_semantics

    newest = retrieval.search_messages(
        _request("пагинация", sort="newest", limit=50),
        conn=connection,
    )
    newest_keys = [(hit.date, metadata[hit.id]["db_id"]) for hit in newest.hits]
    assert newest_keys == sorted(newest_keys, reverse=True)
    assert all(hit.bm25_score is None for hit in newest.hits)
    assert newest.score_semantics
