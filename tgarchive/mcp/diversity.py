"""Deterministic candidate selection for the diverse search strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..lemma import normalize
from ..search import WORD_RE
from .models import SEARCH_CHAT_TOPIC_MAX_RESULTS, SEARCH_LOCAL_BURST_MAX_RESULTS


# A local burst is a connected run of message IDs whose neighbouring messages
# are no more than this distance apart in the same chat/topic scope.
LOCAL_BURST_RADIUS = 10
SHINGLE_SIZE = 3
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _Candidate:
    row: Any
    normalized_text: str
    shingles: frozenset[tuple[str, ...]]
    chat_id: int
    topic_id: int | None
    message_id: int


def _candidate_text(row: Any) -> str:
    text = row["text"] or row["transcript"] or ""
    return str(text)


def _normalized_text(value: str) -> str:
    # Keep punctuation, but normalize ё/case and whitespace.  This is
    # deliberately conservative: unrelated short messages should not merge.
    return _WHITESPACE_RE.sub(" ", normalize(value).strip()).lower()


def _token_shingles(value: str) -> frozenset[tuple[str, ...]]:
    tokens = [token.lower() for token in WORD_RE.findall(value)]
    if not tokens:
        return frozenset()
    if len(tokens) <= SHINGLE_SIZE:
        return frozenset((token,) for token in tokens)
    return frozenset(
        tuple(tokens[index:index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    )


def _make_candidate(row: Any) -> _Candidate:
    text = _normalized_text(_candidate_text(row))
    return _Candidate(
        row=row,
        normalized_text=text,
        shingles=_token_shingles(text),
        chat_id=int(row["chat_id"]),
        topic_id=row["topic_id"],
        message_id=int(row["message_id"]),
    )


def _jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _is_near_duplicate(candidate: _Candidate, kept: Sequence[_Candidate]) -> bool:
    if not candidate.shingles:
        return False
    return any(
        other.shingles
        and _jaccard(candidate.shingles, other.shingles)
        >= NEAR_DUPLICATE_JACCARD_THRESHOLD
        for other in kept
    )


def _same_scope(left: _Candidate, right: _Candidate) -> bool:
    return left.chat_id == right.chat_id and left.topic_id == right.topic_id


def _local_burst_selected_count(
    candidate: _Candidate,
    selected: Sequence[_Candidate],
    radius: int,
) -> int:
    """Count selected messages in the local connected burst of candidate."""
    same_scope_ids = [
        item.message_id
        for item in selected
        if _same_scope(candidate, item)
    ]
    if not same_scope_ids:
        return 0

    # Treat a chain of neighbouring IDs as one burst, not just the messages
    # directly adjacent to the candidate.  This prevents 1, 11, 21 from
    # bypassing the cap through a sequence of pairwise-near messages.
    burst_ids = {candidate.message_id}
    changed = True
    while changed:
        changed = False
        for message_id in same_scope_ids:
            if message_id in burst_ids:
                continue
            if any(abs(message_id - burst_id) <= radius for burst_id in burst_ids):
                burst_ids.add(message_id)
                changed = True
    return len(burst_ids) - 1


def _passes_burst_limits(
    candidate: _Candidate,
    selected: Sequence[_Candidate],
    local_burst_max: int,
    chat_topic_max: int,
) -> bool:
    same_scope = [item for item in selected if _same_scope(candidate, item)]
    if len(same_scope) >= chat_topic_max:
        return False
    return _local_burst_selected_count(
        candidate,
        selected,
        LOCAL_BURST_RADIUS,
    ) < local_burst_max


def _remove_exact_duplicates(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    seen: set[str] = set()
    unique: list[_Candidate] = []
    for candidate in candidates:
        # Empty text is not a useful deduplication key: otherwise every
        # attachment-only message would collapse into one result.
        if candidate.normalized_text:
            if candidate.normalized_text in seen:
                continue
            seen.add(candidate.normalized_text)
        unique.append(candidate)
    return unique


def _remove_near_duplicates(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in candidates:
        if _is_near_duplicate(candidate, kept):
            continue
        kept.append(candidate)
    return kept


def select_diverse(
    rows: Sequence[Any],
    limit: int,
    *,
    local_burst_max: int = SEARCH_LOCAL_BURST_MAX_RESULTS,
    chat_topic_max: int = SEARCH_CHAT_TOPIC_MAX_RESULTS,
) -> list[Any]:
    """Select a relevance-ordered, diverse shortlist from BM25 candidates.

    Exact and near duplicates are hard filters.  Burst and chat/topic caps are
    soft: deferred candidates are used as a relevance-ordered backfill after
    all uncapped alternatives have been considered.
    """
    if limit <= 0 or not rows:
        return []

    candidates = [_make_candidate(row) for row in rows]
    candidates = _remove_exact_duplicates(candidates)
    candidates = _remove_near_duplicates(candidates)

    selected: list[_Candidate] = []
    deferred_by_burst: list[_Candidate] = []
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if _passes_burst_limits(
            candidate,
            selected,
            local_burst_max,
            chat_topic_max,
        ):
            selected.append(candidate)
        else:
            deferred_by_burst.append(candidate)

    # The caps are diversity preferences, not a reason to return a short page
    # when the pool contains no remaining alternatives.
    for candidate in deferred_by_burst:
        if len(selected) >= limit:
            break
        selected.append(candidate)

    return [candidate.row for candidate in selected]


__all__ = ["select_diverse"]
