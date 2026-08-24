from __future__ import annotations

import threading
import time

import pytest
import uvicorn
from mcp.client import Client

from tgarchive.mcp import ratelimit, retrieval, server as server_module
from tgarchive.mcp.settings import MCPSettings


def _reset_runtime(monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "_DB_PATH", None)
    monkeypatch.setattr(retrieval, "_DB_SEMAPHORE", None)
    monkeypatch.setattr(retrieval, "_QUERY_TIMEOUT_SECONDS", None)
    monkeypatch.setattr(retrieval, "_CONFIGURED_QUERY_TIMEOUT_SECONDS", None)
    monkeypatch.setattr(retrieval, "_TOOL_TIMEOUTS", None)
    monkeypatch.setattr(retrieval, "_RUNTIME_KEY", None)
    monkeypatch.setattr(ratelimit, "_LIMITER", None)
    monkeypatch.setattr(ratelimit, "_RUNTIME_KEY", None)


@pytest.mark.anyio
async def test_streamable_http_transport_round_trip(synthetic_archive, monkeypatch):
    _reset_runtime(monkeypatch)
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
        dev_mode=True,
    )
    mcp_server = server_module.create_server(settings)
    app = mcp_server.streamable_http_app(
        streamable_http_path=server_module.STREAMABLE_HTTP_PATH,
        stateless_http=True,
        host=settings.host,
    )
    uvicorn_config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    server_errors: list[BaseException] = []

    def run_server() -> None:
        try:
            uvicorn_server.run()
        except BaseException as error:  # pragma: no cover - surfaced below
            server_errors.append(error)

    thread = threading.Thread(target=run_server, name="letopis-mcp-http-test")
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not uvicorn_server.started:
            if server_errors:
                raise server_errors[0]
            if not thread.is_alive():
                raise RuntimeError("Streamable HTTP server exited before startup")
            if time.monotonic() >= deadline:
                raise TimeoutError("Streamable HTTP server did not start")
            time.sleep(0.01)

        sockets = [socket for server in uvicorn_server.servers for socket in server.sockets]
        assert sockets
        port = sockets[0].getsockname()[1]
        url = f"http://{settings.host}:{port}{server_module.STREAMABLE_HTTP_PATH}"
        async with Client(url, read_timeout_seconds=10) as client:
            listing = await client.list_tools()
            assert len(listing.tools) == 5

            result = await client.call_tool(
                "search_messages",
                {"query": "пагинация", "strategy": "relevance", "limit": 1},
            )
            assert result.is_error is False
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["hits"]
            assert "sql_time_ms" not in result.structured_content
            assert "candidate_pool_size" not in result.structured_content
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
