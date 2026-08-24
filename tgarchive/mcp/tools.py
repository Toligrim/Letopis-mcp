"""Thin typed handlers for the basic MCP retrieval tools."""

from __future__ import annotations

from dataclasses import asdict as _asdict, is_dataclass as _is_dataclass
import hashlib
import json
import logging
import re
import time
from typing import Callable, TypeVar
import uuid

from . import ratelimit, retrieval
from .cursor import query_fingerprint as _query_fingerprint
from .models import (
    AGGREGATE_LIMIT_HARD_MAX,
    ARCHIVE_CHAT_IDS_MAX,
    ARCHIVE_LIMIT_HARD_MAX,
    CURSOR_MAX_CHARS,
    CONTEXT_BEFORE_AFTER_HARD_MAX,
    CONTEXT_MESSAGE_MAX_CHARS_HARD_MAX,
    FETCH_IDS_MAX,
    FETCH_IDS_MIN,
    FETCH_PER_MESSAGE_MAX_CHARS_HARD_MAX,
    SEARCH_LIMIT_HARD_MAX,
    SEARCH_FILTER_DATE_MAX_CHARS,
    SEARCH_FILTER_DATE_PATTERN,
    SEARCH_FILTER_ID_LIST_MAX,
    SEARCH_QUERY_MAX_CHARS,
    SEARCH_FILTER_SENDER_NAME_MAX_CHARS,
    SNIPPET_CHARS_MAX,
    SNIPPET_CHARS_MIN,
    AggregateMessagesInput,
    AggregateMessagesOutput,
    ArchiveOverviewInput,
    ArchiveOverviewOutput,
    FetchMessagesInput,
    FetchMessagesOutput,
    GetContextInput,
    GetContextOutput,
    SearchFilters,
    MEDIA_FILTER_VALUES,
    SearchMessagesInput,
    SearchMessagesOutput,
    ErrorCode,
    ErrorResponse,
    ToolError,
)


_MATCH_MODES = {"and", "or", "boolean"}
_GROUP_BY = {"chat", "topic", "sender", "month", "quarter", "year"}
_SEARCH_SORTS = {"relevance", "oldest", "newest"}
_SEARCH_STRATEGIES = {"relevance", "diverse"}
_MEDIA_FILTERS = frozenset(MEDIA_FILTER_VALUES)
_DATE_FILTER_RE = re.compile(SEARCH_FILTER_DATE_PATTERN)
_PUBLIC_ID_RE = re.compile(r"^tg:(-?\d+):(\d+)$", re.ASCII)
ToolResultT = TypeVar("ToolResultT")
_LOGGER = logging.getLogger("letopis_mcp")


def _short_request_fingerprint(request: object | None) -> str | None:
    if request is None:
        return None
    try:
        if isinstance(request, SearchMessagesInput):
            fingerprint = _query_fingerprint(
                request.query,
                request.match_mode,
                request.filters,
                request.sort,
            )
        elif isinstance(request, AggregateMessagesInput):
            fingerprint = _query_fingerprint(
                request.query,
                request.match_mode,
                request.filters,
                "aggregate",
                group_by=request.group_by,
            )
        elif isinstance(request, ArchiveOverviewInput):
            fingerprint = _query_fingerprint(
                "",
                "archive",
                {"chat_ids": request.chat_ids},
                "archive_overview",
                extra={"include_topics": request.include_topics},
            )
        elif _is_dataclass(request):
            serialized = json.dumps(
                _asdict(request),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            fingerprint = hashlib.sha256(serialized).hexdigest()
        else:
            return None
    except (TypeError, ValueError):
        return None
    return fingerprint[:12]


def _filter_metadata(request: object | None) -> dict[str, bool]:
    filters = getattr(request, "filters", None)
    if filters is None and isinstance(request, ArchiveOverviewInput):
        filters = request

    def has_value(name: str) -> bool:
        return getattr(filters, name, None) is not None and bool(getattr(filters, name))

    return {
        "has_chat_filter": has_value("chat_ids"),
        "has_topic_filter": has_value("topic_ids"),
        "has_sender_filter": has_value("sender_id") or has_value("sender_name"),
        "has_date_filter": has_value("date_from") or has_value("date_to"),
        "has_media_filter": has_value("media"),
    }


def _returned_count(result: object) -> int | None:
    for collection_name in ("hits", "groups", "chats", "messages"):
        collection = getattr(result, collection_name, None)
        if isinstance(collection, (list, tuple)):
            return len(collection)
    for field in ("returned_hits", "returned_groups", "returned_chats"):
        value = getattr(result, field, None)
        if isinstance(value, int):
            return value
    return None


def _log_tool_call(
    tool_name: str,
    started: float,
    status: str,
    *,
    result: ToolResultT | ErrorResponse | None = None,
    error_code: ErrorCode | None = None,
    request_id: str | None = None,
    request: object | None = None,
    diagnostics: dict[str, object] | None = None,
) -> None:
    fields: dict[str, object] = {
        "tool": tool_name,
        "status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if request_id is not None:
        fields["request_id"] = request_id
    fingerprint = _short_request_fingerprint(request)
    if fingerprint is not None:
        fields["query_fingerprint"] = fingerprint
    if request is not None:
        fields.update(_filter_metadata(request))
        fields["cursor_used"] = bool(getattr(request, "cursor", None))
    if isinstance(result, ErrorResponse):
        error_code = result.code
    elif result is not None:
        for field in ("response_chars", "truncated", "total_hits"):
            value = getattr(result, field, None)
            if value is not None:
                fields[field] = value
        returned_count = _returned_count(result)
        if returned_count is not None:
            fields["returned_count"] = returned_count
    if diagnostics is not None:
        for field in ("sql_time_ms", "candidate_pool_size"):
            value = diagnostics.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields[field] = value
    if error_code is not None:
        fields["error_code"] = error_code.value
    _LOGGER.info("tool_call", extra=fields)


def _handle_tool(
    tool_name: str,
    callback: Callable[[], ToolResultT],
    *,
    request: object | None = None,
) -> ToolResultT | ErrorResponse:
    """Convert all public tool failures to the wire-level error model."""
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    retrieval.reset_diagnostics()

    def log(status: str, result: object) -> None:
        _log_tool_call(
            tool_name,
            started,
            status,
            result=result,
            request_id=request_id,
            request=request,
            diagnostics=retrieval.take_diagnostics(),
        )

    if not ratelimit.can_start():
        error = ErrorResponse(
            code=ErrorCode.RETRIEVAL_RATE_LIMITED,
            message="Global retrieval rate limit exceeded; retry later",
            retryable=True,
            details={"scope": "global"},
        )
        log("error", error)
        return error
    try:
        result = callback()
    except ToolError as exc:
        ratelimit.cancel_pending()
        error = ErrorResponse(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            details=dict(exc.details or {}),
        )
        log("error", error)
        return error
    except Exception:
        ratelimit.cancel_pending()
        error = ErrorResponse(
            code=ErrorCode.INTERNAL_ERROR,
            message="Internal error while handling tool request",
            retryable=False,
            details={},
        )
        log("error", error)
        return error
    if isinstance(result, ErrorResponse):
        ratelimit.cancel_pending()
        log("error", result)
    else:
        response_chars = getattr(result, "response_chars", None)
        if isinstance(response_chars, int):
            if not ratelimit.commit_or_reject(response_chars):
                error = ErrorResponse(
                    code=ErrorCode.RETRIEVAL_RATE_LIMITED,
                    message="Global retrieval character limit exceeded; retry later",
                    retryable=True,
                    details={"scope": "global", "phase": "completion"},
                )
                log("error", error)
                return error
        else:
            ratelimit.cancel_pending()
        log("ok", result)
    return result


def _validate_limit(value: int, hard_max: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= hard_max:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"{field} must be an integer between 1 and {hard_max}",
        )


def _validate_cursor(cursor: str | None) -> None:
    if cursor is None:
        return
    if not isinstance(cursor, str):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="cursor must be a string")
    if len(cursor) > CURSOR_MAX_CHARS:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"cursor must be at most {CURSOR_MAX_CHARS} characters",
        )


def _validate_bool(value: bool, field: str) -> None:
    if not isinstance(value, bool):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message=f"{field} must be a boolean")


def _validate_range(value: int, minimum: int, maximum: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"{field} must be an integer between {minimum} and {maximum}",
        )


def _parse_public_id(value: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="ids must contain strings in tg:<chat_id>:<message_id> format",
        )
    match = _PUBLIC_ID_RE.fullmatch(value)
    if match is None:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"invalid public message ID: {value!r}",
        )
    return int(match.group(1)), int(match.group(2))


def _validate_public_ids(ids: list[str]) -> list[tuple[int, int]]:
    if not isinstance(ids, list):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="ids must be a list")
    if not FETCH_IDS_MIN <= len(ids) <= FETCH_IDS_MAX:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"ids must contain between {FETCH_IDS_MIN} and {FETCH_IDS_MAX} IDs",
        )
    parsed = [_parse_public_id(value) for value in ids]
    if len(set(parsed)) != len(parsed):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="ids must be unique")
    return parsed


def _validate_filters(filters: SearchFilters | None) -> None:
    if filters is None:
        return
    if not isinstance(filters, SearchFilters):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="filters must be SearchFilters or None",
        )
    for name in ("chat_ids", "topic_ids"):
        values = getattr(filters, name)
        if values is None:
            continue
        if not isinstance(values, list):
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"{name} must be a list of integers",
            )
        if len(values) > SEARCH_FILTER_ID_LIST_MAX:
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"{name} must contain at most {SEARCH_FILTER_ID_LIST_MAX} IDs",
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"{name} must be a list of integers",
            )
    if filters.sender_id is not None and (
        isinstance(filters.sender_id, bool) or not isinstance(filters.sender_id, int)
    ):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="sender_id must be an integer")
    if filters.sender_name is not None:
        if not isinstance(filters.sender_name, str):
            raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="sender_name must be a string")
        if len(filters.sender_name) > SEARCH_FILTER_SENDER_NAME_MAX_CHARS:
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=(
                    f"sender_name must be at most "
                    f"{SEARCH_FILTER_SENDER_NAME_MAX_CHARS} characters"
                ),
            )
    for name in ("date_from", "date_to"):
        value = getattr(filters, name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message=f"{name} must be a string")
        normalized = value.strip()
        if (
            len(normalized) > SEARCH_FILTER_DATE_MAX_CHARS
            or _DATE_FILTER_RE.fullmatch(normalized) is None
        ):
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"{name} must use YYYY, YYYY-MM, or YYYY-MM-DD format",
            )
    if filters.media is not None:
        if not isinstance(filters.media, str) or filters.media not in _MEDIA_FILTERS:
            raise ToolError(
                code=ErrorCode.INVALID_ARGUMENT,
                message=f"media must be one of {sorted(_MEDIA_FILTERS)}",
            )


def _validate_chat_ids(chat_ids: list[int] | None) -> None:
    if chat_ids is None:
        return
    if not isinstance(chat_ids, list):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="chat_ids must be a list of integers")
    if len(chat_ids) > ARCHIVE_CHAT_IDS_MAX:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"chat_ids must contain at most {ARCHIVE_CHAT_IDS_MAX} IDs",
        )
    if any(isinstance(chat_id, bool) or not isinstance(chat_id, int) for chat_id in chat_ids):
        raise ToolError(code=ErrorCode.INVALID_ARGUMENT, message="chat_ids must contain integers")


def _archive_overview(request: ArchiveOverviewInput) -> ArchiveOverviewOutput:
    """Validate and delegate the archive overview request."""
    if not isinstance(request, ArchiveOverviewInput):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="request must be ArchiveOverviewInput",
        )
    _validate_chat_ids(request.chat_ids)
    _validate_limit(request.limit, ARCHIVE_LIMIT_HARD_MAX, "limit")
    _validate_cursor(request.cursor)
    return retrieval.archive_overview(request)


def archive_overview(
    request: ArchiveOverviewInput,
) -> ArchiveOverviewOutput | ErrorResponse:
    return _handle_tool(
        "archive_overview",
        lambda: _archive_overview(request),
        request=request,
    )


def _aggregate_messages(request: AggregateMessagesInput) -> AggregateMessagesOutput:
    """Validate and delegate the aggregate request."""
    if not isinstance(request, AggregateMessagesInput):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="request must be AggregateMessagesInput",
        )
    if not isinstance(request.query, str) or not request.query.strip():
        raise ToolError(code=ErrorCode.INVALID_QUERY, message="query must be a non-empty string")
    if request.match_mode not in _MATCH_MODES:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"match_mode must be one of {sorted(_MATCH_MODES)}",
        )
    if request.group_by not in _GROUP_BY:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"group_by must be one of {sorted(_GROUP_BY)}",
        )
    _validate_filters(request.filters)
    _validate_limit(request.limit, AGGREGATE_LIMIT_HARD_MAX, "limit")
    _validate_cursor(request.cursor)
    try:
        return retrieval.aggregate_messages(request)
    except ValueError as exc:
        raise ToolError(
            code=ErrorCode.INVALID_QUERY,
            message="query could not be parsed",
        ) from exc


def aggregate_messages(
    request: AggregateMessagesInput,
) -> AggregateMessagesOutput | ErrorResponse:
    return _handle_tool(
        "aggregate_messages",
        lambda: _aggregate_messages(request),
        request=request,
    )


def _search_messages(request: SearchMessagesInput) -> SearchMessagesOutput:
    """Validate and delegate the basic search request."""
    if not isinstance(request, SearchMessagesInput):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="request must be SearchMessagesInput",
        )
    if not isinstance(request.query, str) or not request.query.strip():
        raise ToolError(code=ErrorCode.INVALID_QUERY, message="query must be a non-empty string")
    if len(request.query) > SEARCH_QUERY_MAX_CHARS:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"query must be at most {SEARCH_QUERY_MAX_CHARS} characters",
        )
    if request.match_mode not in _MATCH_MODES:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"match_mode must be one of {sorted(_MATCH_MODES)}",
        )
    if request.sort not in _SEARCH_SORTS:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"sort must be one of {sorted(_SEARCH_SORTS)}",
        )
    if request.strategy not in _SEARCH_STRATEGIES:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=f"strategy must be one of {sorted(_SEARCH_STRATEGIES)}",
        )
    if request.strategy == "diverse" and request.sort != "relevance":
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="strategy=diverse only supports sort=relevance",
            details={"strategy": "diverse", "sort": request.sort},
        )
    _validate_filters(request.filters)
    _validate_limit(request.limit, SEARCH_LIMIT_HARD_MAX, "limit")
    _validate_range(request.snippet_chars, SNIPPET_CHARS_MIN, SNIPPET_CHARS_MAX, "snippet_chars")
    _validate_bool(request.include_total, "include_total")
    _validate_cursor(request.cursor)
    if request.strategy == "diverse" and request.cursor is not None:
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="strategy=diverse does not support cursor pagination",
            details={"strategy": "diverse"},
        )
    try:
        return retrieval.search_messages(request)
    except ValueError as exc:
        raise ToolError(
            code=ErrorCode.INVALID_QUERY,
            message="query could not be parsed",
        ) from exc


def search_messages(request: SearchMessagesInput) -> SearchMessagesOutput | ErrorResponse:
    return _handle_tool(
        "search_messages",
        lambda: _search_messages(request),
        request=request,
    )


def _fetch_messages(request: FetchMessagesInput) -> FetchMessagesOutput:
    """Validate public IDs and delegate the bounded message fetch."""
    if not isinstance(request, FetchMessagesInput):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="request must be FetchMessagesInput",
        )
    parsed_ids = _validate_public_ids(request.ids)
    _validate_bool(request.include_transcript, "include_transcript")
    _validate_bool(request.include_links, "include_links")
    _validate_bool(request.include_reactions, "include_reactions")
    _validate_range(
        request.per_message_max_chars,
        0,
        FETCH_PER_MESSAGE_MAX_CHARS_HARD_MAX,
        "per_message_max_chars",
    )
    return retrieval.fetch_messages(request, parsed_ids)


def fetch_messages(request: FetchMessagesInput) -> FetchMessagesOutput | ErrorResponse:
    return _handle_tool(
        "fetch_messages",
        lambda: _fetch_messages(request),
        request=request,
    )


def _get_context(request: GetContextInput) -> GetContextOutput:
    """Validate the bounded context request and delegate retrieval."""
    if not isinstance(request, GetContextInput):
        raise ToolError(
            code=ErrorCode.INVALID_ARGUMENT,
            message="request must be GetContextInput",
        )
    parsed_id = _parse_public_id(request.id)
    _validate_range(request.before, 0, CONTEXT_BEFORE_AFTER_HARD_MAX, "before")
    _validate_range(request.after, 0, CONTEXT_BEFORE_AFTER_HARD_MAX, "after")
    _validate_bool(request.same_topic, "same_topic")
    _validate_bool(request.include_transcripts, "include_transcripts")
    _validate_range(
        request.message_max_chars,
        0,
        CONTEXT_MESSAGE_MAX_CHARS_HARD_MAX,
        "message_max_chars",
    )
    return retrieval.get_context(request, parsed_id)


def get_context(request: GetContextInput) -> GetContextOutput | ErrorResponse:
    return _handle_tool(
        "get_context",
        lambda: _get_context(request),
        request=request,
    )
