from __future__ import annotations

import json

import pytest
from mcp.client import Client

from tgarchive.mcp import ratelimit, retrieval, server as server_module
from tgarchive.mcp.settings import MCPSettings


@pytest.mark.anyio
async def test_official_mcp_client_protocol_contract(synthetic_archive, monkeypatch):
    # create_server() owns process-wide runtime configuration. Isolate this
    # protocol test from the unit tests' lazy/default runtime initialization.
    monkeypatch.setattr(retrieval, "_DB_PATH", None)
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", None)
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", None)
    monkeypatch.setattr(retrieval, "_RUNTIME_KEY", None)
    monkeypatch.setattr(ratelimit, "_LIMITER", None)
    monkeypatch.setattr(ratelimit, "_RUNTIME_KEY", None)

    settings = MCPSettings(
        db_path=synthetic_archive.path,
        host="127.0.0.1",
        port=0,
        log_level="WARNING",
        max_concurrency=4,
        query_timeout_seconds=2.0,
        rolling_calls_max=20,
        rolling_chars_max=1_000_000,
        rolling_window_seconds=600,
    )
    server = server_module.create_server(settings)

    async with Client(server, mode="legacy") as client:
        assert client.instructions

        listing = await client.list_tools()
        tools_by_name = {tool.name: tool for tool in listing.tools}
        assert set(tools_by_name) == {
            "archive_overview",
            "search_messages",
            "aggregate_messages",
            "fetch_messages",
            "get_context",
        }
        descriptions = {
            name: (tool.description or "")
            for name, tool in tools_by_name.items()
        }
        assert all(len(description) > 100 for description in descriptions.values())
        assert "snippet" in descriptions["search_messages"].lower()
        assert "diverse" in descriptions["search_messages"].lower()
        assert "coverage" in descriptions["aggregate_messages"].lower()
        assert "shortlist" in descriptions["fetch_messages"].lower()
        assert "small" in descriptions["get_context"].lower()
        assert all(tool.input_schema for tool in listing.tools)
        assert all(tool.input_schema.get("additionalProperties") is False for tool in listing.tools)
        search_input_schema = tools_by_name["search_messages"].input_schema
        filter_ref = search_input_schema["properties"]["filters"]["anyOf"][0]["$ref"]
        filter_schema = search_input_schema["$defs"][filter_ref.rsplit("/", 1)[-1]]
        assert filter_schema["additionalProperties"] is False
        output_schema = json.dumps(tools_by_name["search_messages"].output_schema)
        assert "bm25_score" in output_schema
        assert "score_semantics" in output_schema
        for tool in listing.tools:
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert tool.annotations.idempotent_hint is True
            assert tool.annotations.open_world_hint is False

        valid = await client.call_tool(
            "search_messages",
            {
                "query": "пагинация",
                "strategy": "relevance",
                "limit": 1,
                "snippet_chars": 120,
            },
        )
        assert valid.is_error is False
        assert isinstance(valid.structured_content, dict)
        assert {"schema_version", "hits", "score_semantics"} <= set(valid.structured_content)

        invalid_calls = [
            await client.call_tool(
                "search_messages",
                {"query": "пагинация", "bogus_argument": "x"},
            ),
            await client.call_tool(
                "search_messages",
                {"query": "пагинация", "filters": {"bogus_filter_key": "x"}},
            ),
            await client.call_tool("search_messages", {}),
        ]
        for invalid in invalid_calls:
            assert invalid.is_error is True
            assert invalid.content
            error_text = " ".join(getattr(item, "text", "") for item in invalid.content)
            assert error_text
            assert "Traceback" not in error_text
