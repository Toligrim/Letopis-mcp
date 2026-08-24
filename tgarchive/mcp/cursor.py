"""Signed, short-lived keyset cursors for MCP message search.

The wire token is ``base64url(canonical-json-payload).base64url(hmac)`` with
padding removed.  The payload is intentionally opaque to callers; its
``last_row_id`` is the internal ``messages.id``, not a public Telegram ID.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from typing import Any, Mapping

from ..db import read_index_revision
from .models import ErrorCode, ToolError


CURSOR_VERSION = 1
CURSOR_TTL_SECONDS = 15 * 60  # Temporary default; configurable in Stage 6 runtime config.
_CURSOR_SORTS = {"relevance", "oldest", "newest", "aggregate", "archive_overview"}
_MISSING = object()

_FILTER_FIELDS = (
    "chat_ids",
    "topic_ids",
    "sender_id",
    "sender_name",
    "date_from",
    "date_to",
    "media",
)

_configured_secret = os.environ.get("LETOPIS_MCP_CURSOR_SECRET")
if _configured_secret:
    _CURSOR_SECRET = _configured_secret.encode("utf-8")
else:
    # Temporary dev fallback: cursors intentionally do not survive a process
    # restart without LETOPIS_MCP_CURSOR_SECRET. Stage 6 adds full config.
    _CURSOR_SECRET = secrets.token_bytes(32)


class CursorError(ToolError):
    """A cursor failure that is ready for later unified error handling."""

    code: ErrorCode

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(code=code, message=message)


def _invalid(message: str) -> None:
    raise CursorError(ErrorCode.INVALID_CURSOR, message)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor payload is not JSON-serializable") from exc


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        _invalid("invalid cursor encoding")
    if any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for char in value):
        _invalid("invalid cursor encoding")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (TypeError, ValueError, binascii.Error) as exc:
        raise CursorError(ErrorCode.INVALID_CURSOR, "invalid cursor encoding") from exc
    if _b64url_encode(decoded) != value:
        _invalid("non-canonical cursor encoding")
    return decoded


def _index_signature(conn: sqlite3.Connection) -> int:
    return read_index_revision(conn)


def _filter_value(filters: Any, name: str) -> Any:
    if filters is None:
        return None
    if isinstance(filters, Mapping):
        return filters.get(name)
    return getattr(filters, name, None)


def _canonical_filters(filters: Any) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for name in _FILTER_FIELDS:
        value = _filter_value(filters, name)
        if isinstance(value, list):
            # IN-list order and duplicates do not affect the SQL result.
            value = sorted(set(value))
            if not value:
                value = None
        canonical[name] = value
    return canonical


def query_fingerprint(
    query: str,
    match_mode: str,
    filters: Any,
    sort: str,
    *,
    group_by: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Return SHA-256 of the canonical search parameters bound to a cursor."""
    canonical = {
        "filters": _canonical_filters(filters),
        "match_mode": match_mode,
        "query": query,
        "sort": sort,
    }
    if group_by is not None:
        canonical["group_by"] = group_by
    if extra is not None:
        canonical["extra"] = dict(extra)
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def encode_cursor(
    *,
    query_fingerprint: str,
    sort: str,
    last_row_id: int | None = None,
    conn: sqlite3.Connection,
    last_score: float | None = None,
    last_date: str | None = None,
    now: float | None = None,
    last_count: int | None = None,
    last_group_key: str | int | None | object = _MISSING,
    last_chat_id: int | None = None,
    group_by: str | None = None,
) -> str:
    """Encode and sign a cursor for one deterministic search ordering.

    Relevance cursors carry ``last_score`` as ``float.hex()``. Chronological
    cursors carry ``last_date`` because a date plus row ID is the keyset for
    ``oldest``/``newest``; this field is the minimal addition needed to make
    those cursors resume without OFFSET.
    """
    if not isinstance(query_fingerprint, str) or not query_fingerprint:
        raise ValueError("query_fingerprint must be a non-empty string")
    if sort not in _CURSOR_SORTS:
        raise ValueError("unsupported cursor sort")
    if sort in {"relevance", "oldest", "newest"} and (
        isinstance(last_row_id, bool) or not isinstance(last_row_id, int)
    ):
        raise ValueError("last_row_id must be an integer")

    payload: dict[str, Any] = {
        "exp": int((time.time() if now is None else now) + CURSOR_TTL_SECONDS),
        "index_signature": _index_signature(conn),
        "query_fingerprint": query_fingerprint,
        "sort": sort,
        "version": CURSOR_VERSION,
    }
    if sort == "relevance":
        if last_score is None:
            raise ValueError("relevance cursor requires last_score")
        score = float(last_score)
        if score != score or score in (float("inf"), float("-inf")):
            raise ValueError("last_score must be finite")
        payload["last_row_id"] = last_row_id
        payload["last_score"] = score.hex()
    elif sort in {"oldest", "newest"}:
        if last_date is None:
            raise ValueError("chronological cursor requires last_date")
        payload["last_row_id"] = last_row_id
        payload["last_date"] = str(last_date)
    elif sort == "aggregate":
        if not isinstance(group_by, str) or not group_by:
            raise ValueError("aggregate cursor requires group_by")
        if isinstance(last_count, bool) or not isinstance(last_count, int):
            raise ValueError("aggregate cursor requires last_count")
        if last_group_key is _MISSING or (
            last_group_key is not None
            and (isinstance(last_group_key, bool) or not isinstance(last_group_key, (str, int)))
        ):
            raise ValueError("aggregate cursor requires a scalar group key")
        payload["group_by"] = group_by
        payload["last_count"] = last_count
        payload["last_group_key"] = last_group_key
    else:
        if isinstance(last_count, bool) or not isinstance(last_count, int):
            raise ValueError("archive cursor requires last_count")
        if isinstance(last_chat_id, bool) or not isinstance(last_chat_id, int):
            raise ValueError("archive cursor requires last_chat_id")
        payload["last_count"] = last_count
        payload["last_chat_id"] = last_chat_id

    serialized = _canonical_json(payload)
    signature = hmac.new(_CURSOR_SECRET, serialized, hashlib.sha256).digest()
    return f"{_b64url_encode(serialized)}.{_b64url_encode(signature)}"


def decode_cursor(
    token: str,
    *,
    expected_query_fingerprint: str,
    expected_sort: str,
    conn: sqlite3.Connection,
    expected_group_by: str | None = None,
) -> dict[str, Any]:
    """Verify a cursor and return its validated, decoded keyset payload."""
    if not isinstance(token, str) or token.count(".") != 1:
        _invalid("invalid cursor format")
    encoded_payload, encoded_signature = token.split(".", 1)
    serialized = _b64url_decode(encoded_payload)
    provided_signature = _b64url_decode(encoded_signature)
    expected_signature = hmac.new(_CURSOR_SECRET, serialized, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        _invalid("invalid cursor signature")
    if len(provided_signature) != hashlib.sha256().digest_size:
        _invalid("invalid cursor signature")

    try:
        payload = json.loads(serialized.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorError(ErrorCode.INVALID_CURSOR, "invalid cursor payload") from exc
    if not isinstance(payload, dict):
        _invalid("invalid cursor payload")
    try:
        if _canonical_json(payload) != serialized:
            _invalid("non-canonical cursor payload")
    except ValueError as exc:
        raise CursorError(ErrorCode.INVALID_CURSOR, "invalid cursor payload") from exc

    if expected_sort not in _CURSOR_SORTS:
        _invalid("unsupported cursor sort")
    base_keys = {
        "exp",
        "index_signature",
        "query_fingerprint",
        "sort",
        "version",
    }
    if expected_sort == "relevance":
        expected_keys = base_keys | {"last_row_id", "last_score"}
    elif expected_sort in {"oldest", "newest"}:
        expected_keys = base_keys | {"last_date", "last_row_id"}
    elif expected_sort == "aggregate":
        expected_keys = base_keys | {"group_by", "last_count", "last_group_key"}
    else:
        expected_keys = base_keys | {"last_count", "last_chat_id"}
    if set(payload) != expected_keys:
        _invalid("invalid cursor fields")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != CURSOR_VERSION:
        _invalid("unsupported cursor version")
    if payload.get("query_fingerprint") != expected_query_fingerprint:
        _invalid("cursor does not match query")
    if payload.get("sort") != expected_sort:
        _invalid("cursor does not match sort")

    exp = payload.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, int) or int(time.time()) >= exp:
        _invalid("cursor expired")
    if expected_sort in {"relevance", "oldest", "newest"}:
        row_id = payload.get("last_row_id")
        if isinstance(row_id, bool) or not isinstance(row_id, int):
            _invalid("invalid cursor row ID")
    index_signature = payload.get("index_signature")
    if isinstance(index_signature, bool) or not isinstance(index_signature, int):
        _invalid("invalid cursor index signature")

    if expected_sort == "relevance":
        raw_score = payload.get("last_score")
        if not isinstance(raw_score, str):
            _invalid("invalid cursor score")
        try:
            score = float.fromhex(raw_score)
        except ValueError as exc:
            raise CursorError(ErrorCode.INVALID_CURSOR, "invalid cursor score") from exc
        if score != score or score in (float("inf"), float("-inf")):
            _invalid("invalid cursor score")
        if score.hex() != raw_score:
            _invalid("invalid cursor score")
        payload["last_score"] = score
    elif expected_sort in {"oldest", "newest"}:
        if not isinstance(payload.get("last_date"), str):
            _invalid("invalid cursor date")
    elif expected_sort == "aggregate":
        if expected_group_by is None or payload.get("group_by") != expected_group_by:
            _invalid("cursor does not match group_by")
        last_count = payload.get("last_count")
        if isinstance(last_count, bool) or not isinstance(last_count, int):
            _invalid("invalid cursor group count")
        last_group_key = payload.get("last_group_key")
        if last_group_key is not None and (
            isinstance(last_group_key, bool)
            or not isinstance(last_group_key, (str, int))
        ):
            _invalid("invalid cursor group key")
    else:
        last_count = payload.get("last_count")
        if isinstance(last_count, bool) or not isinstance(last_count, int):
            _invalid("invalid cursor chat count")
        last_chat_id = payload.get("last_chat_id")
        if isinstance(last_chat_id, bool) or not isinstance(last_chat_id, int):
            _invalid("invalid cursor chat ID")

    if payload.get("index_signature") != _index_signature(conn):
        raise CursorError(ErrorCode.STALE_CURSOR, "cursor index signature is stale")
    return payload


__all__ = [
    "CURSOR_TTL_SECONDS",
    "CURSOR_VERSION",
    "CursorError",
    "decode_cursor",
    "encode_cursor",
    "query_fingerprint",
]
