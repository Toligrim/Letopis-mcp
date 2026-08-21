import re

from .lemma import Lemmatizer, normalize

TOKEN_RE = re.compile(r'"[^"]*"|\S+')
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-я]+")  # после normalize() буквы ё нет
WORD_ONLY_RE = re.compile(r"^[0-9A-Za-zА-Яа-я]+$")

MEDIA_ALIASES = {
    "photo": "MessageMediaPhoto",
    "document": "MessageMediaDocument",
    "webpage": "MessageMediaWebPage",
    "poll": "MessageMediaPoll",
    "dice": "MessageMediaDice",
    "geo": "MessageMediaGeo",
    "contact": "MessageMediaContact",
}


def _phrase_group(words, lem: Lemmatizer) -> str:
    orig = " ".join(w.lower() for w in words)
    lemm = " ".join(lem.word(w) for w in words)
    return f'(orig:"{orig}" OR lemma:"{lemm}")'


def _make_group(token: str, lem: Lemmatizer):
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
        if len(words) > 1 or is_phrase:
            return _phrase_group(words, lem)
        token = words[0]
    low = token.lower()
    if prefix:
        return f"(orig:{low}* OR lemma:{low}*)"
    return f"(orig:{low} OR lemma:{lem.word(low)})"


def build_match(query: str, lem: Lemmatizer, any_mode: bool = False) -> str:
    """Строит FTS5 MATCH-выражение: каждое слово ищется и как есть, и по лемме."""
    items = []
    pending = "AND"
    for part in TOKEN_RE.findall(query):
        if part.upper() in ("OR", "AND"):
            pending = part.upper()
            continue
        group = _make_group(part, lem)
        if group:
            items.append((pending, group))
            pending = "AND"
    if not items:
        raise ValueError("Пустой поисковый запрос")
    if any_mode:
        return " OR ".join(g for _, g in items)
    expr = items[0][1]
    for op, group in items[1:]:
        expr += f" {op} {group}"
    return expr


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


def where_filters(chat_ids=None, topic=None, sender=None, date_from=None, date_to=None, media=None):
    cond, params = [], []
    if chat_ids:
        cond.append(f"m.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += list(chat_ids)
    if topic is not None:
        cond.append("m.topic_id=?")
        params.append(topic)
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


def run_search(conn, match, cond, params, limit=500, rank=False):
    where = " AND ".join(["fts MATCH ?", *cond])
    order = "bm25(fts)" if rank else "m.date, m.message_id"
    sql = f"SELECT m.* FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} ORDER BY {order}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, [match, *params]).fetchall()


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
}


def group_counts(conn, match, cond, params, by):
    col = GROUP_COLS[by]
    where = " AND ".join(["fts MATCH ?", *cond])
    return conn.execute(
        f"SELECT {col} k, count(*) c, min(m.date) d0, max(m.date) d1 "
        f"FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} "
        f"GROUP BY k ORDER BY c DESC",
        [match, *params],
    ).fetchall()


def context_rows(conn, chat_id, message_id, before=30, after=30, same_topic=True):
    pivot = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND message_id=?", (chat_id, message_id)
    ).fetchone()
    if pivot is None:
        return [], None, []
    topic_id = pivot["topic_id"] if same_topic else None
    scope = "chat_id=?" + (" AND topic_id IS ?" if topic_id is not None else "")
    base = [chat_id] + ([topic_id] if topic_id is not None else [])
    rows_before = conn.execute(
        f"SELECT * FROM messages WHERE {scope} AND message_id<? ORDER BY message_id DESC LIMIT ?",
        [*base, message_id, before],
    ).fetchall()[::-1]
    rows_after = conn.execute(
        f"SELECT * FROM messages WHERE {scope} AND message_id>? ORDER BY message_id LIMIT ?",
        [*base, message_id, after],
    ).fetchall()
    return rows_before, pivot, rows_after
