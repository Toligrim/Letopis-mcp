from tgarchive.mcp.models import (
    ArchiveOverviewInput,
    SearchMessagesInput,
)
from tgarchive.mcp import retrieval


def test_synthetic_archive_fixture_smoke(synthetic_archive):
    connection = synthetic_archive.connection

    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] >= 60
    assert connection.execute("SELECT count(*) FROM fts").fetchone()[0] >= 60

    overview = retrieval.archive_overview(
        ArchiveOverviewInput(limit=10, include_topics=True),
        conn=connection,
    )
    assert overview.total_chats == 3
    assert overview.total_messages == len(synthetic_archive.messages)
    assert overview.returned_chats == 3
    assert overview.chats[0].topics is not None

    search = retrieval.search_messages(
        SearchMessagesInput(
            query="пагинация",
            strategy="relevance",
            limit=5,
            snippet_chars=120,
            include_total=True,
        ),
        conn=connection,
    )
    assert search.total_hits >= 50
    assert search.returned_hits == 5
    assert len(search.hits) == 5
    assert all(hit.id.startswith("tg:") for hit in search.hits)
