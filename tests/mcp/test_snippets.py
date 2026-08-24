from tgarchive.lemma import Lemmatizer
from tgarchive.mcp import retrieval
from tgarchive.mcp.models import SearchFilters, SearchMessagesInput


def _search(synthetic_archive, query: str, **kwargs):
    request = SearchMessagesInput(
        query=query,
        strategy="relevance",
        snippet_chars=kwargs.pop("snippet_chars", 120),
        include_total=True,
        limit=kwargs.pop("limit", 5),
        **kwargs,
    )
    return retrieval.search_messages(request, conn=synthetic_archive.connection)


def test_text_snippet_centers_on_match_and_honors_limit(synthetic_archive):
    long_case = synthetic_archive.cases["ordinary"][0]
    result = _search(synthetic_archive, "якорь", limit=1, snippet_chars=120)

    assert result.total_hits == 1
    assert result.hits[0].id == long_case["id"]
    assert result.hits[0].snippet_source == "text"
    assert len(result.hits[0].snippet) == 120
    assert "якорь" in result.hits[0].snippet
    assert "якорь" in result.hits[0].matched_terms
    assert not result.hits[0].snippet.startswith("Пагинация архива")

    burst_result = _search(synthetic_archive, "диалоге", limit=1)
    assert burst_result.hits[0].snippet_source == "text"
    assert "диалоге" in burst_result.hits[0].snippet
    assert burst_result.hits[0].matched_terms == ["диалоге"]


def test_transcript_only_snippet_uses_transcript_source(synthetic_archive):
    case = synthetic_archive.cases["transcript_only"]
    result = _search(synthetic_archive, "фонетика", limit=1)
    hit = result.hits[0]

    assert result.total_hits == 1
    assert hit.id == case["id"]
    assert hit.snippet_source == "transcript"
    assert "фонетика" in hit.snippet
    assert hit.matched_terms == ["фонетика"]


def test_snippet_query_uses_shared_lemma_matching(synthetic_archive):
    lemmatizer = Lemmatizer()
    assert lemmatizer.word("архива") == lemmatizer.word("архивом")

    case = synthetic_archive.cases["ordinary"][0]
    result = _search(
        synthetic_archive,
        "архивом",
        limit=50,
        filters=SearchFilters(chat_ids=[case["chat_id"]], topic_ids=[101]),
    )

    assert result.total_hits == 25
    assert result.returned_hits == 25
    assert all(hit.snippet_source == "text" for hit in result.hits)
    assert all("архивом" in hit.matched_terms for hit in result.hits)
