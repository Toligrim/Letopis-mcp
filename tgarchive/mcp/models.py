"""Wire-contract data models for the Letopis MCP retrieval tools.

This module deliberately contains only stdlib types and dataclasses. Validation,
serialization, retrieval, SQL, cursor handling, and budget enforcement belong to
later MCP stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

CONTENT_TRUST_UNTRUSTED_USER_GENERATED: Final = "untrusted_user_generated_content"

ARCHIVE_CHAT_IDS_MAX: Final[int] = 50
ARCHIVE_INCLUDE_TOPICS_DEFAULT: Final[bool] = False
ARCHIVE_LIMIT_DEFAULT: Final[int] = 30
ARCHIVE_LIMIT_HARD_MAX: Final[int] = 50
ARCHIVE_TOPICS_PER_CHAT_CAP: Final[int] = 10
ARCHIVE_RESPONSE_CHARS_HARD_MAX: Final[int] = 24_000

SEARCH_QUERY_MAX_CHARS: Final[int] = 500
SEARCH_MATCH_MODE_DEFAULT: Final[str] = "and"
SEARCH_SORT_DEFAULT: Final[str] = "relevance"
SEARCH_STRATEGY_DEFAULT: Final[str] = "diverse"
SEARCH_FILTER_ID_LIST_MAX: Final[int] = ARCHIVE_CHAT_IDS_MAX
SEARCH_FILTER_SENDER_NAME_MAX_CHARS: Final[int] = 200
SEARCH_FILTER_DATE_MAX_CHARS: Final[int] = 10
SEARCH_FILTER_DATE_PATTERN: Final[str] = r"^\d{4}(?:-\d{2}){0,2}$"
CURSOR_MAX_CHARS: Final[int] = 4_096
SEARCH_SCORE_SEMANTICS: Final[str] = (
    "SQLite FTS5 BM25; lower is better; not comparable across different queries; "
    "null when sort does not use relevance ranking."
)
SEARCH_LIMIT_DEFAULT: Final[int] = 20
SEARCH_LIMIT_HARD_MAX: Final[int] = 50
SNIPPET_CHARS_DEFAULT: Final[int] = 320
SNIPPET_CHARS_MIN: Final[int] = 120
SNIPPET_CHARS_MAX: Final[int] = 800
SEARCH_INCLUDE_TOTAL_DEFAULT: Final[bool] = True
SEARCH_RESPONSE_CHARS_TARGET: Final[int] = 24_000
SEARCH_RESPONSE_CHARS_HARD_MAX: Final[int] = 48_000
SEARCH_CANDIDATE_POOL_MULTIPLIER: Final[int] = 8
SEARCH_CANDIDATE_POOL_MIN: Final[int] = 80
SEARCH_CANDIDATE_POOL_MAX: Final[int] = 300
SEARCH_LOCAL_BURST_MAX_RESULTS: Final[int] = 2
SEARCH_CHAT_TOPIC_MAX_RESULTS: Final[int] = 4

AGGREGATE_LIMIT_DEFAULT: Final[int] = 30
AGGREGATE_LIMIT_HARD_MAX: Final[int] = 100
AGGREGATE_RESPONSE_CHARS_HARD_MAX: Final[int] = 24_000

FETCH_IDS_MIN: Final[int] = 1
FETCH_IDS_MAX: Final[int] = 12
FETCH_INCLUDE_TRANSCRIPT_DEFAULT: Final[bool] = False
FETCH_INCLUDE_LINKS_DEFAULT: Final[bool] = True
FETCH_INCLUDE_REACTIONS_DEFAULT: Final[bool] = False
FETCH_PER_MESSAGE_MAX_CHARS_DEFAULT: Final[int] = 6_000
FETCH_PER_MESSAGE_MAX_CHARS_HARD_MAX: Final[int] = 12_000
FETCH_RESPONSE_CHARS_HARD_MAX: Final[int] = 40_000

CONTEXT_BEFORE_DEFAULT: Final[int] = 5
CONTEXT_AFTER_DEFAULT: Final[int] = 5
CONTEXT_BEFORE_AFTER_HARD_MAX: Final[int] = 15
CONTEXT_SAME_TOPIC_DEFAULT: Final[bool] = True
CONTEXT_INCLUDE_TRANSCRIPTS_DEFAULT: Final[bool] = False
CONTEXT_MESSAGE_MAX_CHARS_DEFAULT: Final[int] = 2_000
CONTEXT_MESSAGE_MAX_CHARS_HARD_MAX: Final[int] = 4_000
CONTEXT_TOTAL_MESSAGES_HARD_MAX: Final[int] = 31
CONTEXT_RESPONSE_CHARS_HARD_MAX: Final[int] = 40_000


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

ContentTrust = Literal["untrusted_user_generated_content"]
MatchMode = Literal["and", "or", "boolean"]
SearchSort = Literal["relevance", "oldest", "newest"]
SearchStrategy = Literal["relevance", "diverse"]
AggregateGroupBy = Literal["chat", "topic", "sender", "month", "quarter", "year"]
MediaFilter = Literal[
    "voice",
    "audio",
    "video",
    "video_note",
    "sticker",
    "gif",
    "document",
    "photo",
    "poll",
    "webpage",
    "any",
    "none",
]
SnippetSource = Literal["text", "transcript", "poll", "media_name"]
ContextRelation = Literal["before", "pivot", "after"]

MEDIA_FILTER_VALUES: Final[tuple[str, ...]] = (
    "voice",
    "audio",
    "video",
    "video_note",
    "sticker",
    "gif",
    "document",
    "photo",
    "poll",
    "webpage",
    "any",
    "none",
)


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_QUERY = "INVALID_QUERY"
    INVALID_CURSOR = "INVALID_CURSOR"
    STALE_CURSOR = "STALE_CURSOR"
    NOT_FOUND = "NOT_FOUND"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    DB_BUSY = "DB_BUSY"
    OUTPUT_BUDGET_EXCEEDED = "OUTPUT_BUDGET_EXCEEDED"
    RETRIEVAL_RATE_LIMITED = "RETRIEVAL_RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Shared envelope and errors
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ResponseEnvelope:
    schema_version: str
    content_trust: ContentTrust
    response_chars: int
    response_bytes_utf8: int
    estimated_tokens_rough: int
    truncated: bool


@dataclass(slots=True)
class ErrorResponse:
    code: ErrorCode
    message: str
    retryable: bool
    details: dict[str, Any]


class ToolError(Exception):
    """Domain error raised by validation or read-only retrieval boundaries."""

    code: ErrorCode
    message: str
    retryable: bool
    details: dict[str, Any] | None

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details
        super().__init__(message)


# ---------------------------------------------------------------------------
# archive_overview
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ArchiveOverviewInput:
    chat_ids: list[int] | None = None
    include_topics: bool = ARCHIVE_INCLUDE_TOPICS_DEFAULT
    limit: int = ARCHIVE_LIMIT_DEFAULT
    cursor: str | None = None


@dataclass(slots=True)
class ArchiveTopic:
    topic_id: int | None
    title: str | None
    message_count: int
    date_from: str | None
    date_to: str | None


@dataclass(slots=True)
class ArchiveChat:
    chat_id: int
    title: str
    message_count: int
    date_from: str | None
    date_to: str | None
    topic_count: int
    topics: list[ArchiveTopic] | None = None


@dataclass(slots=True)
class ArchiveOverviewOutput(ResponseEnvelope):
    total_messages: int
    total_chats: int
    date_from: str | None
    date_to: str | None
    chats: list[ArchiveChat]
    returned_chats: int
    has_more: bool
    next_cursor: str | None


# ---------------------------------------------------------------------------
# search_messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SearchFilters:
    chat_ids: list[int] | None = None
    topic_ids: list[int] | None = None
    sender_id: int | None = None
    sender_name: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    media: MediaFilter | None = None


@dataclass(slots=True)
class SearchMessagesInput:
    query: str
    match_mode: MatchMode = SEARCH_MATCH_MODE_DEFAULT
    filters: SearchFilters | None = None
    sort: SearchSort = SEARCH_SORT_DEFAULT
    strategy: SearchStrategy = SEARCH_STRATEGY_DEFAULT
    limit: int = SEARCH_LIMIT_DEFAULT
    snippet_chars: int = SNIPPET_CHARS_DEFAULT
    include_total: bool = SEARCH_INCLUDE_TOTAL_DEFAULT
    cursor: str | None = None


@dataclass(slots=True)
class SearchHit:
    id: str
    chat_id: int
    chat_title: str
    topic_id: int | None
    topic_title: str | None
    message_id: int
    date: str
    sender_id: int | None
    sender: str | None
    reply_to: int | None
    snippet: str
    snippet_source: SnippetSource
    matched_terms: list[str]
    bm25_score: float | None
    rank: int
    media_kind: str | None
    telegram_url: str | None
    content_trust: ContentTrust


@dataclass(slots=True)
class SearchMessagesOutput(ResponseEnvelope):
    original_query: str
    match_mode: MatchMode
    score_semantics: str
    total_hits: int | None
    returned_hits: int
    hits: list[SearchHit]
    has_more: bool
    pagination_supported: bool
    next_cursor: str | None


# ---------------------------------------------------------------------------
# aggregate_messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AggregateMessagesInput:
    query: str
    group_by: AggregateGroupBy
    match_mode: MatchMode = SEARCH_MATCH_MODE_DEFAULT
    filters: SearchFilters | None = None
    limit: int = AGGREGATE_LIMIT_DEFAULT
    cursor: str | None = None


@dataclass(slots=True)
class AggregateGroup:
    key: str | int | None
    count: int
    date_from: str | None
    date_to: str | None
    chat_id: int | None = None
    chat_title: str | None = None
    topic_id: int | None = None
    topic_title: str | None = None


@dataclass(slots=True)
class AggregateMessagesOutput(ResponseEnvelope):
    total_hits: int
    group_by: AggregateGroupBy
    groups: list[AggregateGroup]
    returned_groups: int
    other_count: int
    has_more: bool
    next_cursor: str | None


# ---------------------------------------------------------------------------
# fetch_messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FetchMessagesInput:
    ids: list[str]
    include_transcript: bool = FETCH_INCLUDE_TRANSCRIPT_DEFAULT
    include_links: bool = FETCH_INCLUDE_LINKS_DEFAULT
    include_reactions: bool = FETCH_INCLUDE_REACTIONS_DEFAULT
    per_message_max_chars: int = FETCH_PER_MESSAGE_MAX_CHARS_DEFAULT


@dataclass(slots=True)
class FetchedMessage:
    id: str
    chat: str | None
    topic: str | None
    date: str
    sender: str | None
    reply_to: int | None
    text: str | None
    original_text_chars: int
    text_truncated: bool
    transcript: str | None
    transcript_original_chars: int
    transcript_truncated: bool
    poll: Any | None
    media_kind: str | None
    media_name: str | None
    links: list[str] | None
    content_trust: ContentTrust
    reactions: Any | None = None


@dataclass(slots=True)
class FetchMessagesOutput(ResponseEnvelope):
    messages: list[FetchedMessage]
    omitted_ids: list[str]


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GetContextInput:
    id: str
    before: int = CONTEXT_BEFORE_DEFAULT
    after: int = CONTEXT_AFTER_DEFAULT
    same_topic: bool = CONTEXT_SAME_TOPIC_DEFAULT
    include_transcripts: bool = CONTEXT_INCLUDE_TRANSCRIPTS_DEFAULT
    message_max_chars: int = CONTEXT_MESSAGE_MAX_CHARS_DEFAULT


@dataclass(slots=True)
class ContextMessage:
    id: str
    chat_id: int
    message_id: int
    topic_id: int | None
    relation: ContextRelation
    date: str
    sender: str | None
    text: str | None
    transcript: str | None = None


@dataclass(slots=True)
class GetContextOutput(ResponseEnvelope):
    pivot_id: str
    messages: list[ContextMessage]
    has_more_before: bool
    has_more_after: bool
