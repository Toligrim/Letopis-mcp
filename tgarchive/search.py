import re
from dataclasses import dataclass

from .lemma import Lemmatizer, normalize

TOKEN_RE = re.compile(r'"[^"]*"|\S+')
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-я]+")  # после normalize() буквы ё нет
WORD_ONLY_RE = re.compile(r"^[0-9A-Za-zА-Яа-я]+$")


@dataclass(frozen=True, slots=True)
class QueryGroup:
    """One normalized search unit shared by FTS matching and snippets."""

    words: tuple[str, ...]
    lemmas: tuple[str, ...]
    operator: str = "AND"
    prefix: bool = False
    phrase: bool = False

MEDIA_ALIASES = {
    "photo": "MessageMediaPhoto",
    "document": "MessageMediaDocument",
    "webpage": "MessageMediaWebPage",
    "poll": "MessageMediaPoll",
    "dice": "MessageMediaDice",
    "geo": "MessageMediaGeo",
    "contact": "MessageMediaContact",
}


def _phrase_expression(words: tuple[str, ...], lemmas: tuple[str, ...]) -> str:
    orig = " ".join(w.lower() for w in words)
    lemm = " ".join(lemmas)
    return f'(orig:"{orig}" OR lemma:"{lemm}")'


def _phrase_group(words, lem: Lemmatizer) -> str:
    normalized_words = tuple(w.lower() for w in words)
    return _phrase_expression(normalized_words, tuple(lem.word(w) for w in normalized_words))


def _parse_query_group(token: str, lem: Lemmatizer) -> QueryGroup | None:
    is_phrase = token.startswith('"') and token.endswith('"') and len(token) > 1
    if is_phrase:
        token = token[1:-1]
    prefix = token.endswith("*")
    if prefix:
        token = token[:-1]
    token = normalize(token)
    if is_phrase or not WORD_ONLY_RE.match(token):
        words = WORD_RE.findall(token)
        if not words:
            return None
    else:
        words = [token]
    normalized_words = tuple(word.lower() for word in words)
    phrase = is_phrase or len(normalized_words) > 1
    return QueryGroup(
        words=normalized_words,
        lemmas=tuple(lem.word(word) for word in normalized_words),
        prefix=prefix,
        phrase=phrase,
    )


def parse_query_groups(
    query: str,
    lem: Lemmatizer,
    any_mode: bool = False,
) -> list[QueryGroup]:
    """Parse query units once for both FTS matching and snippet matching."""
    groups = []
    pending = "AND"
    for part in TOKEN_RE.findall(query):
        if part.upper() in ("OR", "AND"):
            pending = part.upper()
            continue
        group = _parse_query_group(part, lem)
        if group is not None:
            groups.append(
                QueryGroup(
                    words=group.words,
                    lemmas=group.lemmas,
                    operator="OR" if any_mode else pending,
                    prefix=group.prefix,
                    phrase=group.phrase,
                )
            )
            pending = "AND"
    if not groups:
        raise ValueError("Пустой поисковый запрос")
    return groups


def _group_expression(group: QueryGroup) -> str:
    if group.phrase:
        return _phrase_expression(group.words, group.lemmas)
    word = group.words[0]
    if group.prefix:
        return f"(orig:{word}* OR lemma:{group.lemmas[0]}*)"
    return f"(orig:{word} OR lemma:{group.lemmas[0]})"


def _make_group(token: str, lem: Lemmatizer):
    group = _parse_query_group(token, lem)
    return _group_expression(group) if group is not None else None


def build_match_from_groups(groups: list[QueryGroup], any_mode: bool = False) -> str:
    if not groups:
        raise ValueError("Пустой поисковый запрос")
    expr = _group_expression(groups[0])
    for group in groups[1:]:
        operator = "OR" if any_mode else group.operator
        expr += f" {operator} {_group_expression(group)}"
    return expr


def build_match(query: str, lem: Lemmatizer, any_mode: bool = False) -> str:
    """Строит FTS5 MATCH-выражение: каждое слово ищется и как есть, и по лемме."""
    groups = parse_query_groups(query, lem, any_mode=any_mode)
    return build_match_from_groups(groups, any_mode=any_mode)


def pad_date_from(s: str) -> str:
    s = s.strip()
    if re.fullmatch(r"\d{4}", s):
        return f"{s}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return f"{s}-01"
    return s


def date_to_bound(s: str) -> str:
    """Включительная верхняя граница: '2025-06' -> '2025-07' (строгое <)."""
    s = s.strip()
    if re.fullmatch(r"\d{4}(-\d{2}){0,2}", s):
        parts = s.split("-")
        parts[-1] = str(int(parts[-1]) + 1).zfill(len(parts[-1]))
        return "-".join(parts)
    return s


def where_filters(chat_ids=None, topic=None, sender=None, date_from=None, date_to=None,
                  media=None, topic_ids=None):
    """Build SQL filters; ``topic`` and ``topic_ids`` are mutually exclusive."""
    cond, params = [], []
    if chat_ids:
        cond.append(f"m.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += list(chat_ids)
    if topic is not None:
        cond.append("m.topic_id=?")
        params.append(topic)
    if topic_ids:
        topic_values = [topic_id for topic_id in topic_ids if topic_id != 0]
        topic_parts = []
        if topic_values:
            topic_parts.append(f"m.topic_id IN ({','.join('?' * len(topic_values))})")
            params += topic_values
        if 0 in topic_ids:
            topic_parts.append("m.topic_id IS NULL")
        cond.append(f"({' OR '.join(topic_parts)})")
    if sender:
        if re.fullmatch(r"-?\d+", sender):
            cond.append("m.sender_id=?")
            params.append(int(sender))
        else:
            cond.append("m.sender_name LIKE ? COLLATE NOCASE")
            params.append(f"%{sender}%")
    if date_from:
        cond.append("m.date>=?")
        params.append(pad_date_from(date_from))
    if date_to:
        cond.append("m.date<?")
        params.append(date_to_bound(date_to))
    if media == "any":
        cond.append("m.media_type IS NOT NULL")
    elif media == "none":
        cond.append("m.media_type IS NULL")
    elif media:
        cond.append("m.media_type=?")
        params.append(MEDIA_ALIASES.get(media.lower(), media))
    return cond, params


def run_search(
    conn,
    match,
    cond,
    params,
    limit=500,
    rank=False,
    include_score=False,
    newest=False,
    after=None,
):
    where = " AND ".join(["fts MATCH ?", *cond])
    after_params = []
    if after is not None:
        if len(after) != 2:
            raise ValueError("after must contain a sort key and row ID")
        last_key, last_row_id = after
        if rank:
            where += " AND (bm25(fts) > ? OR (bm25(fts) = ? AND m.id > ?))"
            after_params.extend([last_key, last_key, last_row_id])
        elif newest:
            where += " AND (m.date < ? OR (m.date = ? AND m.id < ?))"
            after_params.extend([last_key, last_key, last_row_id])
        else:
            where += " AND (m.date > ? OR (m.date = ? AND m.id > ?))"
            after_params.extend([last_key, last_key, last_row_id])
    if rank:
        order = "bm25(fts), m.id"
    elif newest:
        order = "m.date DESC, m.id DESC"
    else:
        order = "m.date, m.id"
    select = "m.*"
    if include_score:
        select += ", bm25(fts) AS bm25_score"
    sql = f"SELECT {select} FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} ORDER BY {order}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, [match, *params, *after_params]).fetchall()


def run_count(conn, match, cond, params):
    where = " AND ".join(["fts MATCH ?", *cond])
    return conn.execute(
        f"SELECT count(*) c, min(m.date) d0, max(m.date) d1 "
        f"FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where}",
        [match, *params],
    ).fetchone()


GROUP_COLS = {
    "chat": "m.chat_id",
    "topic": "m.topic_id",
    "sender": "coalesce(m.sender_name, m.sender_id)",
    "month": "substr(m.date, 1, 7)",
    "quarter": "substr(m.date, 1, 4) || '-Q' || ((CAST(substr(m.date, 6, 2) AS INTEGER) - 1) / 3 + 1)",
    "year": "substr(m.date, 1, 4)",
}


def _group_key_after(after):
    if after is None:
        return "", []
    if len(after) != 2:
        raise ValueError("after must contain a group count and group key")
    last_count, last_key = after
    if last_key is None:
        # SQLite sorts NULL before every non-NULL value for ASC ordering.
        return " HAVING c < ? OR (c = ? AND k IS NOT NULL)", [last_count, last_count]
    return " HAVING c < ? OR (c = ? AND k > ?)", [last_count, last_count, last_key]


def group_counts(conn, match, cond, params, by, limit=None, after=None):
    col = GROUP_COLS[by]
    where = " AND ".join(["fts MATCH ?", *cond])
    after_sql, after_params = _group_key_after(after)
    sql = (
        f"SELECT {col} k, count(*) c, min(m.date) d0, max(m.date) d1 "
        f"FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} "
        f"GROUP BY k{after_sql} ORDER BY c DESC, k ASC"
    )
    query_params = [match, *params, *after_params]
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, query_params).fetchall()


def group_counts_remaining_count(conn, match, cond, params, by, after=None):
    """Return the sum of counts after a group key without fetching all groups."""
    col = GROUP_COLS[by]
    where = " AND ".join(["fts MATCH ?", *cond])
    after_sql, after_params = _group_key_after(after)
    grouped = (
        f"SELECT {col} k, count(*) c "
        f"FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} "
        f"GROUP BY k{after_sql}"
    )
    row = conn.execute(
        f"SELECT coalesce(sum(c), 0) AS total FROM ({grouped}) grouped",
        [match, *params, *after_params],
    ).fetchone()
    return int(row["total"])


def context_rows(conn, chat_id, message_id, before=30, after=30, same_topic=True):
    pivot = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND message_id=?", (chat_id, message_id)
    ).fetchone()
    if pivot is None:
        return [], None, []
    if same_topic:
        topic_id = pivot["topic_id"]
        if topic_id is None:
            scope = "chat_id=? AND topic_id IS NULL"
            base = [chat_id]
        else:
            scope = "chat_id=? AND topic_id IS ?"
            base = [chat_id, topic_id]
    else:
        scope = "chat_id=?"
        base = [chat_id]
    rows_before = conn.execute(
        f"SELECT * FROM messages WHERE {scope} AND message_id<? ORDER BY message_id DESC LIMIT ?",
        [*base, message_id, before],
    ).fetchall()[::-1]
    rows_after = conn.execute(
        f"SELECT * FROM messages WHERE {scope} AND message_id>? ORDER BY message_id LIMIT ?",
        [*base, message_id, after],
    ).fetchall()
    return rows_before, pivot, rows_after
