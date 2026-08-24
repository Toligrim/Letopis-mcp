"""Deterministic, contentless-FTS snippet construction."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Sequence

from ..lemma import Lemmatizer, normalize
from ..search import QueryGroup, WORD_RE


@dataclass(frozen=True, slots=True)
class _FieldToken:
    surface: str
    lemma: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _FieldMatch:
    group_index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Window:
    start: int
    end: int
    group_indices: tuple[int, ...]
    span: int


def _poll_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if not isinstance(value, dict):
        return str(value)

    parts: list[str] = []
    question = value.get("question")
    if question:
        parts.append(str(question))
    answers = value.get("answers") or []
    if isinstance(answers, (list, tuple)):
        for answer in answers:
            if isinstance(answer, dict):
                answer = next(
                    (answer.get(key) for key in ("text", "title", "value", "option") if answer.get(key)),
                    None,
                )
            if answer is not None:
                parts.append(str(answer))
    return " ".join(parts)


def _field_tokens(value: str, lem: Lemmatizer) -> list[_FieldToken]:
    normalized = normalize(value)
    return [
        _FieldToken(
            surface=match.group(0).lower(),
            lemma=lem.word(match.group(0)),
            start=match.start(),
            end=match.end(),
        )
        for match in WORD_RE.finditer(normalized)
    ]


def _sequence_matches(
    tokens: Sequence[_FieldToken],
    start: int,
    group: QueryGroup,
) -> bool:
    size = len(group.words)
    if start + size > len(tokens):
        return False
    selected = tokens[start:start + size]
    if group.phrase:
        surface_match = all(token.surface == word for token, word in zip(selected, group.words))
        lemma_match = all(token.lemma == lemma for token, lemma in zip(selected, group.lemmas))
        return surface_match or lemma_match
    token = selected[0]
    word, lemma = group.words[0], group.lemmas[0]
    if group.prefix:
        return token.surface.startswith(word) or token.lemma.startswith(lemma)
    return token.surface == word or token.lemma == lemma


def _field_matches(
    value: str,
    groups: Sequence[QueryGroup],
    lem: Lemmatizer,
) -> list[_FieldMatch]:
    tokens = _field_tokens(value, lem)
    matches: list[_FieldMatch] = []
    for group_index, group in enumerate(groups):
        for token_index in range(len(tokens)):
            if _sequence_matches(tokens, token_index, group):
                end_token = tokens[token_index + len(group.words) - 1]
                matches.append(
                    _FieldMatch(
                        group_index=group_index,
                        start=tokens[token_index].start,
                        end=end_token.end,
                    )
                )
    return sorted(matches, key=lambda item: (item.start, item.end, item.group_index))


def _best_window(
    matches: Sequence[_FieldMatch],
    snippet_chars: int,
) -> _Window | None:
    if not matches:
        return None

    max_chars = max(1, snippet_chars)
    counts: Counter[int] = Counter()
    end_queue: deque[tuple[int, int]] = deque()
    left = 0
    best: _Window | None = None

    def remove_left() -> None:
        nonlocal left
        group_index = matches[left].group_index
        counts[group_index] -= 1
        if counts[group_index] == 0:
            del counts[group_index]
        if end_queue and end_queue[0][0] == left:
            end_queue.popleft()
        left += 1

    for right, match in enumerate(matches):
        counts[match.group_index] += 1
        while end_queue and end_queue[-1][1] <= match.end:
            end_queue.pop()
        end_queue.append((right, match.end))

        while left < right and end_queue[0][1] - matches[left].start > max_chars:
            remove_left()
        while left < right and counts[matches[left].group_index] > 1:
            remove_left()

        if left <= right and end_queue:
            start = matches[left].start
            end = end_queue[0][1]
            candidate = _Window(
                start=start,
                end=end,
                group_indices=tuple(sorted(counts)),
                span=end - start,
            )
            if best is None or (len(candidate.group_indices), -candidate.span) > (
                len(best.group_indices),
                -best.span,
            ):
                best = candidate
    return best


def _centered_slice(value: str, start: int, end: int, snippet_chars: int) -> str:
    if len(value) <= snippet_chars:
        return value[:snippet_chars]
    if end - start >= snippet_chars:
        window_start = start
    else:
        center = (start + end) // 2
        window_start = center - snippet_chars // 2
    window_start = max(0, min(window_start, len(value) - snippet_chars))
    return value[window_start:window_start + snippet_chars]


def _matched_terms(groups: Sequence[QueryGroup], indices: Sequence[int]) -> list[str]:
    terms: list[str] = []
    for index in indices:
        for word in groups[index].words:
            if word not in terms:
                terms.append(word)
    return terms


def build_snippet(
    groups: Sequence[QueryGroup],
    *,
    text: Any = None,
    transcript: Any = None,
    poll: Any = None,
    media_name: Any = None,
    snippet_chars: int,
    lem: Lemmatizer,
) -> tuple[str, str, list[str]]:
    """Return the best bounded evidence window from message content fields."""
    fields = (
        ("text", str(text) if text else ""),
        ("transcript", str(transcript) if transcript else ""),
        ("poll", _poll_text(poll)),
        ("media_name", str(media_name) if media_name else ""),
    )
    candidates: list[tuple[int, _Window, str, str]] = []
    for source_index, (source, value) in enumerate(fields):
        if not value:
            continue
        window = _best_window(_field_matches(value, groups, lem), snippet_chars)
        if window is not None:
            candidates.append((source_index, window, source, value))

    if candidates:
        max_coverage = max(len(window.group_indices) for _, window, _, _ in candidates)
        eligible = [item for item in candidates if len(item[1].group_indices) == max_coverage]
        source_index, window, source, value = min(
            eligible,
            key=lambda item: (item[0], item[1].span, item[1].start),
        )
        return (
            _centered_slice(value, window.start, window.end, snippet_chars),
            source,
            _matched_terms(groups, window.group_indices),
        )

    fallback = str(text) if text else str(transcript) if transcript else ""
    fallback_source = "text" if text else "transcript" if transcript else "text"
    return fallback[:snippet_chars], fallback_source, []
