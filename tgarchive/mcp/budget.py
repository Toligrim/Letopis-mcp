"""Shared fixed-point response-size enforcement for MCP outputs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from .models import ResponseEnvelope


ResponseT = TypeVar("ResponseT", bound=ResponseEnvelope)
TrimCallback = Callable[[ResponseT, int], ResponseT]
RenderCallback = Callable[[ResponseT], str]

_METRIC_ITERATIONS = 8
_BUDGET_ITERATIONS = 64


def _refresh_metrics(response: ResponseT, render: RenderCallback[ResponseT]) -> ResponseT:
    """Make envelope metrics describe the exact current serialized response."""
    for _ in range(_METRIC_ITERATIONS):
        rendered = render(response)
        chars = len(rendered)
        bytes_utf8 = len(rendered.encode("utf-8"))
        estimated_tokens = chars // 4
        if (
            response.response_chars == chars
            and response.response_bytes_utf8 == bytes_utf8
            and response.estimated_tokens_rough == estimated_tokens
        ):
            return response
        response = replace(
            response,
            response_chars=chars,
            response_bytes_utf8=bytes_utf8,
            estimated_tokens_rough=estimated_tokens,
        )
    return response


def enforce_budget(
    response: ResponseT,
    hard_max_chars: int,
    list_field_name: str,
    on_truncate: TrimCallback[ResponseT],
    *,
    render: RenderCallback[ResponseT],
    preserve_items: int = 0,
) -> ResponseT:
    """Trim one result list until the serialized response fits its hard cap.

    The retrieval layer supplies the tool-specific ``on_truncate`` callback.
    It receives the current response and a conservative number of tail items
    to remove, and is responsible for updating fields such as ``other_count``
    or ``omitted_ids``.  Metrics are refreshed after every iteration because
    those envelope fields are part of the serialized response themselves.

    ``preserve_items`` is used by context responses to keep the pivot while
    the callback chooses which edge messages to remove.  If the immutable
    portion of a response alone exceeds a cap, no list-only implementation can
    reduce it further; the bounded loop returns a truthful ``truncated``
    response, and the retrieval caller converts that overflow into a
    structured ``OUTPUT_BUDGET_EXCEEDED`` error.
    """
    if hard_max_chars < 0:
        raise ValueError("hard_max_chars must be non-negative")
    if preserve_items < 0:
        raise ValueError("preserve_items must be non-negative")

    response = _refresh_metrics(response, render)
    for _ in range(_BUDGET_ITERATIONS):
        response = _refresh_metrics(response, render)
        if response.response_chars <= hard_max_chars:
            return response

        items = getattr(response, list_field_name, None)
        if not isinstance(items, list) or len(items) <= preserve_items:
            return _refresh_metrics(replace(response, truncated=True), render)

        excess = response.response_chars - hard_max_chars
        average_item_chars = max(1, response.response_chars // len(items))
        remove_count = max(1, math.ceil(excess / average_item_chars) + 1)
        remove_count = min(remove_count, len(items) - preserve_items)
        updated = on_truncate(response, remove_count)
        new_items = getattr(updated, list_field_name, None)
        if not isinstance(new_items, list) or len(new_items) >= len(items):
            return _refresh_metrics(replace(updated, truncated=True), render)
        response = replace(updated, truncated=True)

    return _refresh_metrics(response, render)
