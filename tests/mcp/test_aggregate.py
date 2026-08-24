from tgarchive.mcp import retrieval
from tgarchive.mcp.models import AggregateMessagesInput


def _aggregate(synthetic_archive, group_by: str, *, limit: int = 100):
    return retrieval.aggregate_messages(
        AggregateMessagesInput(
            query="пагинация",
            group_by=group_by,
            limit=limit,
        ),
        conn=synthetic_archive.connection,
    )


def _expected_counts(synthetic_archive):
    matching_rows = [
        *synthetic_archive.cases["ordinary"],
        *synthetic_archive.cases["burst"],
    ]
    assert len(matching_rows) == 56
    work_chat_id = synthetic_archive.cases["ordinary"][0]["chat_id"]
    other_chat_ids = sorted(
        {row["chat_id"] for row in matching_rows if row["chat_id"] != work_chat_id}
    )
    return {
        "chat": {work_chat_id: 46, other_chat_ids[0]: 5, other_chat_ids[1]: 5},
        "topic": {101: 31, None: 15, 202: 10},
        "sender": {"Алиса": 19, "Борис": 19, "Вера": 18},
        "month": {
            "2024-03": 16,
            "2023-01": 10,
            "2024-09": 10,
            "2025-02": 10,
            "2025-11": 10,
        },
        "quarter": {
            "2024-Q1": 16,
            "2023-Q1": 10,
            "2024-Q3": 10,
            "2025-Q1": 10,
            "2025-Q4": 10,
        },
        "year": {"2024": 26, "2025": 20, "2023": 10},
    }


def test_all_aggregate_groupings_match_fixture_distribution(synthetic_archive):
    expected_by_group = _expected_counts(synthetic_archive)

    for group_by, expected in expected_by_group.items():
        result = _aggregate(synthetic_archive, group_by)
        actual = {group.key: group.count for group in result.groups}

        assert actual == expected
        assert result.total_hits == sum(expected.values()) == 56
        assert sum(group.count for group in result.groups) + result.other_count == result.total_hits
        assert result.other_count == 0

    topic_result = _aggregate(synthetic_archive, "topic")
    general = next(group for group in topic_result.groups if group.key is None)
    assert general.count == 15


def test_aggregate_other_count_is_the_exact_tail_sum(synthetic_archive):
    expected = _expected_counts(synthetic_archive)["month"]
    result = _aggregate(synthetic_archive, "month", limit=2)

    assert [(group.key, group.count) for group in result.groups] == [
        ("2024-03", 16),
        ("2023-01", 10),
    ]
    assert result.total_hits == 56
    assert result.returned_groups == 2
    assert result.other_count == sum(expected.values()) - 16 - 10 == 30
    assert result.has_more is True
    assert result.next_cursor
    assert sum(group.count for group in result.groups) + result.other_count == result.total_hits
