from tgarchive.mcp import retrieval
from tgarchive.mcp.diversity import (
    NEAR_DUPLICATE_JACCARD_THRESHOLD,
    _jaccard,
    _is_near_duplicate,
    _make_candidate,
)
from tgarchive.mcp.models import (
    SEARCH_LOCAL_BURST_MAX_RESULTS,
    SearchMessagesInput,
)


def _search(synthetic_archive, query: str, *, limit: int, match_mode: str = "and"):
    return retrieval.search_messages(
        SearchMessagesInput(
            query=query,
            match_mode=match_mode,
            strategy="diverse",
            limit=limit,
            snippet_chars=120,
            include_total=True,
        ),
        conn=synthetic_archive.connection,
    )


def test_diverse_suppresses_exact_and_near_duplicates(synthetic_archive):
    duplicate_cases = synthetic_archive.cases["duplicates"]
    relevance = retrieval.search_messages(
        SearchMessagesInput(
            query="разнообразия",
            strategy="relevance",
            limit=4,
            snippet_chars=120,
            include_total=True,
        ),
        conn=synthetic_archive.connection,
    )
    relevance_ids = {hit.message_id for hit in relevance.hits}
    assert {300, 301, 302, 303} <= relevance_ids

    diverse = _search(synthetic_archive, "разнообразия", limit=4)
    diverse_ids = {hit.message_id for hit in diverse.hits}
    assert not {300, 301} <= diverse_ids

    near_duplicate_cases = synthetic_archive.cases["near_duplicates"]
    near_relevance = retrieval.search_messages(
        SearchMessagesInput(
            query="сравнение",
            strategy="relevance",
            limit=2,
            snippet_chars=120,
            include_total=True,
        ),
        conn=synthetic_archive.connection,
    )
    assert {hit.message_id for hit in near_relevance.hits} == {304, 305}

    near_diverse = _search(synthetic_archive, "сравнение", limit=2)
    assert {hit.message_id for hit in near_diverse.hits} != {304, 305}

    candidate_304 = _make_candidate(near_duplicate_cases[0])
    candidate_305 = _make_candidate(near_duplicate_cases[1])
    jaccard = _jaccard(candidate_304.shingles, candidate_305.shingles)
    assert jaccard >= NEAR_DUPLICATE_JACCARD_THRESHOLD
    assert _is_near_duplicate(candidate_304, [candidate_305])


def test_diverse_applies_burst_cap_when_alternatives_exist(synthetic_archive):
    burst_ids = {row["id"] for row in synthetic_archive.cases["burst"]}

    relevance = retrieval.search_messages(
        SearchMessagesInput(
            query="пагинация всплеск",
            match_mode="or",
            strategy="relevance",
            limit=10,
            snippet_chars=120,
            include_total=True,
        ),
        conn=synthetic_archive.connection,
    )
    relevance_burst = {hit.id for hit in relevance.hits} & burst_ids
    assert len(relevance_burst) > SEARCH_LOCAL_BURST_MAX_RESULTS

    diverse = _search(
        synthetic_archive,
        "пагинация всплеск",
        match_mode="or",
        limit=10,
    )
    diverse_burst = {hit.id for hit in diverse.hits} & burst_ids
    assert diverse.returned_hits == 10
    assert len(diverse_burst) <= SEARCH_LOCAL_BURST_MAX_RESULTS


def test_diverse_soft_burst_limit_backfills_when_no_alternatives_exist(synthetic_archive):
    result = _search(synthetic_archive, "всплеск", limit=6)
    expected_ids = {row["id"] for row in synthetic_archive.cases["burst"]}

    assert result.total_hits == 6
    assert result.returned_hits == 6
    assert {hit.id for hit in result.hits} == expected_ids


def test_diverse_suppresses_duplicate_poll_content(synthetic_archive):
    result = _search(synthetic_archive, "дедупликация опросов", limit=2)
    duplicate_ids = {
        row["message_id"] for row in synthetic_archive.cases["poll_duplicates"]
    }

    assert result.total_hits == len(duplicate_ids) == 2
    assert result.returned_hits == 1
    assert {hit.message_id for hit in result.hits} < duplicate_ids
