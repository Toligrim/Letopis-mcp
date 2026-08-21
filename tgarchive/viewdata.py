"""Общий слой данных для просмотрщиков (веб и TUI).

Соглашение: topic_id = 0 означает «General / без топика» (в базе это NULL),
None — весь чат без фильтра по топику.
"""

import json

from .lemma import Lemmatizer
from .search import build_match, date_to_bound, pad_date_from


def tme_link(chat_id, message_id):
    s = str(chat_id)
    if s.startswith("-100"):
        return f"https://t.me/c/{s[4:]}/{message_id}"
    return None


def chat_maps(conn):
    chats = {r["chat_id"]: dict(r) for r in conn.execute("SELECT * FROM chats")}
    topics = {(r["chat_id"], r["topic_id"]): r["title"]
              for r in conn.execute("SELECT chat_id,topic_id,title FROM topics")}
    return chats, topics


def list_chats(conn):
    meta = {r["chat_id"]: dict(r) for r in conn.execute("SELECT * FROM chats")}
    out = []
    for r in conn.execute(
        "SELECT chat_id, count(*) n, min(date) d0, max(date) d1, "
        "count(DISTINCT topic_id) nt FROM messages GROUP BY 1 ORDER BY n DESC"
    ):
        m = meta.get(r["chat_id"], {})
        out.append({
            "chat_id": r["chat_id"],
            "title": m.get("title") or str(r["chat_id"]),
            "type": m.get("type"),
            "n": r["n"], "d0": r["d0"], "d1": r["d1"], "topics": r["nt"],
        })
    return out


def list_topics(conn, chat_id):
    tm = {r["topic_id"]: r["title"]
          for r in conn.execute("SELECT topic_id,title FROM topics WHERE chat_id=?", (chat_id,))}
    out = []
    for r in conn.execute(
        "SELECT topic_id, count(*) n, max(date) d1 FROM messages "
        "WHERE chat_id=? GROUP BY 1 ORDER BY n DESC", (chat_id,)
    ):
        tid = r["topic_id"]
        out.append({
            "topic_id": 0 if tid is None else tid,
            "title": "General (без топика)" if tid is None else (tm.get(tid) or f"топик {tid}"),
            "n": r["n"], "d1": r["d1"],
        })
    return out


def _scope(chat_id, topic_id, sender):
    cond, params = ["chat_id=?"], [chat_id]
    if topic_id == 0:
        cond.append("topic_id IS NULL")
    elif topic_id is not None:
        cond.append("topic_id=?")
        params.append(topic_id)
    if sender:
        cond.append("sender_name LIKE ? COLLATE NOCASE")
        params.append(f"%{sender}%")
    return " AND ".join(cond), params


def page_messages(conn, chat_id, topic_id=None, sender=None, before_id=None,
                  after_id=None, around_id=None, date=None, limit=100):
    """Страница сообщений: хвост / вокруг сообщения / от даты / до-после якоря."""
    where, params = _scope(chat_id, topic_id, sender)

    def q(sql, ps):
        return conn.execute(sql, ps).fetchall()

    if around_id is not None:
        older = q(f"SELECT * FROM messages WHERE {where} AND message_id<? "
                  f"ORDER BY message_id DESC LIMIT ?", [*params, around_id, limit // 2])[::-1]
        pivot = q("SELECT * FROM messages WHERE chat_id=? AND message_id=?", [chat_id, around_id])
        newer = q(f"SELECT * FROM messages WHERE {where} AND message_id>? "
                  f"ORDER BY message_id LIMIT ?", [*params, around_id, limit // 2])
        return older + pivot + newer
    if date:
        row = conn.execute(
            f"SELECT message_id FROM messages WHERE {where} AND date>=? "
            f"ORDER BY message_id LIMIT 1", [*params, date]
        ).fetchone()
        if row:
            return page_messages(conn, chat_id, topic_id, sender,
                                 around_id=row["message_id"], limit=limit)
        # дата позже последнего сообщения — показываем хвост
    if before_id is not None:
        return q(f"SELECT * FROM messages WHERE {where} AND message_id<? "
                 f"ORDER BY message_id DESC LIMIT ?", [*params, before_id, limit])[::-1]
    if after_id is not None:
        return q(f"SELECT * FROM messages WHERE {where} AND message_id>? "
                 f"ORDER BY message_id LIMIT ?", [*params, after_id, limit])
    return q(f"SELECT * FROM messages WHERE {where} ORDER BY message_id DESC LIMIT ?",
             [*params, limit])[::-1]


def search_messages(conn, query, any_mode=False, chat_id=None, topic_id=None,
                    sender=None, date_from=None, date_to=None, limit=100, offset=0):
    match = build_match(query, Lemmatizer(), any_mode)
    cond, params = ["fts MATCH ?"], [match]
    if chat_id:
        cond.append("m.chat_id=?")
        params.append(chat_id)
    if topic_id == 0:
        cond.append("m.topic_id IS NULL")
    elif topic_id is not None:
        cond.append("m.topic_id=?")
        params.append(topic_id)
    if sender:
        cond.append("m.sender_name LIKE ? COLLATE NOCASE")
        params.append(f"%{sender}%")
    if date_from:
        cond.append("m.date>=?")
        params.append(pad_date_from(date_from))
    if date_to:
        cond.append("m.date<?")
        params.append(date_to_bound(date_to))
    where = " AND ".join(cond)
    total = conn.execute(
        f"SELECT count(*) c FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where}", params
    ).fetchone()["c"]
    rows = conn.execute(
        f"SELECT m.* FROM fts JOIN messages m ON m.id=fts.rowid WHERE {where} "
        f"ORDER BY m.date, m.message_id LIMIT ? OFFSET ?", [*params, limit, offset]
    ).fetchall()
    return total, rows


def serialize(conn, rows, chats=None, topics=None):
    """Строки БД -> словари для UI (с превью реплаев, ссылками t.me, названиями)."""
    if chats is None:
        chats, topics = chat_maps(conn)
    previews = {}
    by_chat = {}
    for r in rows:
        if r["reply_to"]:
            by_chat.setdefault(r["chat_id"], set()).add(r["reply_to"])
    for cid, ids in by_chat.items():
        ph = ",".join("?" * len(ids))
        for p in conn.execute(
            f"SELECT message_id, sender_name, text, transcript FROM messages "
            f"WHERE chat_id=? AND message_id IN ({ph})", [cid, *ids]
        ):
            t = (p["text"] or p["transcript"] or "").replace("\n", " ")[:140]
            previews[(cid, p["message_id"])] = {"sender": p["sender_name"], "text": t}
    out = []
    for r in rows:
        cid = r["chat_id"]
        out.append({
            "chat_id": cid,
            "chat": (chats.get(cid) or {}).get("title") or str(cid),
            "topic_id": 0 if r["topic_id"] is None else r["topic_id"],
            "topic": topics.get((cid, r["topic_id"])),
            "message_id": r["message_id"],
            "date": r["date"],
            "sender_id": r["sender_id"],
            "sender": r["sender_name"],
            "reply_to": r["reply_to"],
            "reply_preview": previews.get((cid, r["reply_to"])) if r["reply_to"] else None,
            "text": r["text"],
            "media_type": r["media_type"],
            "media_kind": r["media_kind"],
            "media_file": r["media_file"],
            "media_name": r["media_name"],
            "transcript": r["transcript"],
            "reactions": json.loads(r["reactions"]) if r["reactions"] else None,
            "poll": json.loads(r["poll"]) if r["poll"] else None,
            "service": json.loads(r["service"]) if r["service"] else None,
            "tme": tme_link(cid, r["message_id"]),
        })
    return out
