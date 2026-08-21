"""Качалка: первичное скачивание чатов, инкрементальный синк, докачка медиа.

Принципы: JSONL append-only (старые строки не трогаем), дедупликация на уровне
индекса (UNIQUE chat_id+message_id), медиа и транскрипты — отдельными сайдкарами.
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import manifest as MF
from .records import classify_media, iso, message_to_record
from .search import date_to_bound, pad_date_from

ALL_KINDS = ["photo", "document", "voice", "video_note", "video", "sticker", "gif", "audio"]
CONFIG_KEYS = {
    "photos": "photo",
    "documents": "document",
    "voice": "voice",
    "video_notes": "video_note",
    "video": "video",
    "stickers": "sticker",
    "gifs": "gif",
    "audio": "audio",
}


def kinds_from_config(cfg) -> list:
    m = cfg.get("media", {})
    return [kind for key, kind in CONFIG_KEYS.items() if m.get(key)]


def parse_kinds(s):
    """'photo,voice' | 'all' | 'none' -> список типов; None -> None (не задано)."""
    if s is None:
        return None
    s = s.strip().lower()
    if s == "all":
        return list(ALL_KINDS)
    if s in ("none", ""):
        return []
    kinds = [k.strip() for k in s.split(",") if k.strip()]
    bad = [k for k in kinds if k not in ALL_KINDS]
    if bad:
        raise ValueError(f"Неизвестные типы медиа {bad}; допустимо: {','.join(ALL_KINDS)} или all/none")
    return kinds


def parse_dt(s, end=False):
    if not s:
        return None
    iso_s = date_to_bound(s) if end else pad_date_from(s)
    return datetime.strptime(iso_s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


class ChatWriter:
    """Дописывает записи в файлы архива одного чата."""

    def __init__(self, root: Path, chat_id: int):
        self.dir = root / "archive" / str(chat_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._handles = {}
        self.messages = 0
        self.media = 0

    def _handle(self, name):
        h = self._handles.get(name)
        if h is None:
            h = open(self.dir / name, "a", encoding="utf-8")
            self._handles[name] = h
        return h

    @staticmethod
    def _dumps(rec):
        return json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"

    def write_message(self, rec):
        month = (rec.get("date") or "")[:7] or "unknown"
        self._handle(f"{month}.jsonl").write(self._dumps(rec))
        self.messages += 1

    def write_media(self, rec):
        self._handle("media_index.jsonl").write(self._dumps(rec))
        self.media += 1

    def write_transcript(self, rec):
        self._handle("transcripts.jsonl").write(self._dumps(rec))

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}


async def maybe_download_media(client, msg, writer: ChatWriter, kinds, max_mb):
    if not kinds:
        return False
    kind = classify_media(msg)
    if kind is None or kind not in kinds:
        return False
    size = getattr(msg.file, "size", None) or 0
    if max_mb and size > max_mb * 1024 * 1024:
        print(f"  ~~ #{msg.id}: {kind} {size/1e6:.0f} МБ > лимита {max_mb} МБ, пропуск", file=sys.stderr)
        return False
    month = msg.date.strftime("%Y-%m")
    ext = (msg.file.ext if msg.file else None) or ""
    rel = f"media/{month}/{msg.id}_{kind}{ext}"
    dest = writer.dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or (size and dest.stat().st_size != size):
        try:
            await client.download_media(msg, file=str(dest))
        except Exception as e:
            print(f"  !! медиа #{msg.id}: {e}", file=sys.stderr)
            return False
    writer.write_media({
        "message_id": msg.id,
        "kind": kind,
        "file": rel,
        "name": getattr(msg.file, "name", None),
        "size": size,
        "mime": getattr(msg.file, "mime_type", None),
        "date": iso(msg.date),
    })
    return True


async def _pull(client, entity, writer: ChatWriter, *, chat_id, topic=None, min_id=0,
                offset_date=None, stop_after=None, stop_before_id=None,
                kinds=None, max_mb=50, limit=None, label=""):
    """Качает сообщения по возрастанию id; останавливается на границах."""
    n = 0
    kwargs = dict(reverse=True, min_id=min_id or 0)
    if topic is not None:
        kwargs["reply_to"] = topic
    if offset_date is not None:
        kwargs["offset_date"] = offset_date
    if limit:
        kwargs["limit"] = limit
    async for msg in client.iter_messages(entity, **kwargs):
        if stop_before_id and msg.id >= stop_before_id:
            break
        if stop_after and msg.date and msg.date >= stop_after:
            break
        rec = message_to_record(msg, chat_id)
        writer.write_message(rec)
        await maybe_download_media(client, msg, writer, kinds, max_mb)
        n += 1
        if n % 500 == 0:
            print(f"  {label}: {n} ({(rec['date'] or '')[:10]})", file=sys.stderr)
    return n


async def _resolve_entity(client, token):
    t = str(token).strip()
    if re.fullmatch(r"-?\d+", t):
        return await client.get_entity(int(t))
    if t.startswith("@") or "t.me/" in t:
        return await client.get_entity(t)
    matches = []
    async for d in client.iter_dialogs():
        if t.lower() in (d.name or "").lower():
            matches.append(d)
    if len(matches) == 1:
        return matches[0].entity
    if not matches:
        raise SystemExit(f"Диалог «{token}» не найден среди чатов аккаунта (см. tg dialogs)")
    names = "; ".join(f"{d.name} ({d.id})" for d in matches[:10])
    raise SystemExit(f"Несколько диалогов подходят: {names}")


def _scope_range(conn, chat_id, topic):
    if topic is None:
        r = conn.execute(
            "SELECT min(message_id) a, max(message_id) b FROM messages WHERE chat_id=?",
            (chat_id,),
        ).fetchone()
    else:
        r = conn.execute(
            "SELECT min(message_id) a, max(message_id) b FROM messages WHERE chat_id=? AND topic_id=?",
            (chat_id, topic),
        ).fetchone()
    return r["a"], r["b"]


def sync_chats(root, cfg, conn, chat_ids=None, media_override=None, limit=None, account_only=None):
    """Инкрементальный синк всех (или выбранных) чатов из манифеста."""
    from .indexer import index_archive
    from .meta import make_client

    archive_dir = root / cfg["general"].get("archive_dir", "archive")
    index_archive(conn, archive_dir)  # сначала добираем то, что уже лежит в файлах

    data = MF.load(root)
    if MF.bootstrap(root, conn, data):
        MF.save(root, data)
    entries = [e for e in data["chats"]
               if (chat_ids is None or e["chat_id"] in chat_ids)
               and (account_only is None or e.get("account", "default") == account_only)]
    stats = {"chats": 0, "messages": 0, "media": 0}
    if not entries:
        print("Манифест пуст — добавь чат через tg download", file=sys.stderr)
        return stats
    max_mb = cfg.get("media", {}).get("max_file_mb", 50)

    by_account = {}
    for e in entries:
        by_account.setdefault(e.get("account", "default"), []).append(e)

    for account, group in by_account.items():
        client = make_client(root, cfg, account)

        async def run(client=client, group=group, account=account):
            async with client:
                if not await client.is_user_authorized():
                    raise SystemExit(f"Сессия «{account}» не авторизована — tg login --account {account}")
                for e in group:
                    cid = e["chat_id"]
                    try:
                        entity = await client.get_entity(cid)
                    except Exception as ex:
                        print(f"!! {cid}: {ex}", file=sys.stderr)
                        continue
                    kinds = media_override if media_override is not None else e.get("media") or []
                    writer = ChatWriter(root, cid)
                    got = 0
                    for topic in (e.get("topics") or [None]):
                        _, hi = _scope_range(conn, cid, topic)
                        if not hi:
                            print(f"  {cid}: архив пуст — сначала tg download", file=sys.stderr)
                            continue
                        label = f"{cid}" + (f"/t{topic}" if topic else "")
                        got += await _pull(client, entity, writer, chat_id=cid, topic=topic,
                                           min_id=hi, kinds=kinds, max_mb=max_mb,
                                           limit=limit, label=label)
                    writer.close()
                    e["last_sync"] = MF.now_iso()
                    stats["chats"] += 1
                    stats["messages"] += got
                    stats["media"] += writer.media
                    print(f"  {cid}: +{got} сообщений, +{writer.media} медиа", file=sys.stderr)

        asyncio.run(run())

    MF.save(root, data)
    index_archive(conn, archive_dir)
    return stats


def download_chat(root, cfg, conn, token, topics=None, date_from=None, date_to=None,
                  media=None, account="default", limit=None, max_mb=None):
    """Первичное скачивание чата/топиков (или расширение периода назад)."""
    from .indexer import index_archive
    from .meta import make_client
    from telethon import utils

    archive_dir = root / cfg["general"].get("archive_dir", "archive")
    index_archive(conn, archive_dir)
    kinds = media if media is not None else kinds_from_config(cfg)
    max_mb = max_mb or cfg.get("media", {}).get("max_file_mb", 50)
    start = parse_dt(date_from)
    stop = parse_dt(date_to, end=True)
    client = make_client(root, cfg, account)
    result = {}

    async def run():
        async with client:
            if not await client.is_user_authorized():
                raise SystemExit(f"Сессия «{account}» не авторизована — tg login --account {account}")
            entity = await _resolve_entity(client, token)
            cid = utils.get_peer_id(entity)
            writer = ChatWriter(root, cid)
            total = 0
            for topic in (topics or [None]):
                lo, hi = _scope_range(conn, cid, topic)
                label = f"{cid}" + (f"/t{topic}" if topic else "")
                if hi:
                    total += await _pull(client, entity, writer, chat_id=cid, topic=topic,
                                         min_id=hi, stop_after=stop, kinds=kinds,
                                         max_mb=max_mb, limit=limit, label=f"{label} новое")
                    if start is not None and lo:
                        total += await _pull(client, entity, writer, chat_id=cid, topic=topic,
                                             offset_date=start, stop_before_id=lo, stop_after=stop,
                                             kinds=kinds, max_mb=max_mb, limit=limit,
                                             label=f"{label} бэкфилл")
                else:
                    total += await _pull(client, entity, writer, chat_id=cid, topic=topic,
                                         offset_date=start, stop_after=stop, kinds=kinds,
                                         max_mb=max_mb, limit=limit, label=label)
            writer.close()
            result.update({
                "chat_id": cid,
                "title": utils.get_display_name(entity),
                "messages": total,
                "media": writer.media,
            })

    asyncio.run(run())

    data = MF.load(root)
    MF.upsert(data, {
        "chat_id": result["chat_id"],
        "title": result.get("title"),
        "account": account,
        "topics": list(topics) if topics else None,
        "media": kinds,
        "from": date_from,
        "last_sync": MF.now_iso(),
    })
    MF.save(root, data)
    index_archive(conn, archive_dir)
    return result


KIND_TO_MEDIA_TYPE = {"photo": "MessageMediaPhoto"}


def backfill_media(root, cfg, conn, chat_id, kinds, topic=None, date_from=None,
                   date_to=None, limit=100, account=None, max_mb=None):
    """Докачивает файлы для УЖЕ заархивированных сообщений (старые строки не трогаем)."""
    from .indexer import index_archive
    from .meta import make_client
    from .search import where_filters

    if not kinds:
        raise SystemExit("Укажи типы медиа: --media photo,voice,... | all")
    types_needed = set()
    if "photo" in kinds:
        types_needed.add("MessageMediaPhoto")
    if set(kinds) - {"photo"}:
        types_needed.add("MessageMediaDocument")
    cond, params = where_filters([chat_id], topic, None, date_from, date_to, None)
    cond.append(f"m.media_type IN ({','.join('?' * len(types_needed))})")
    params += list(types_needed)
    cond.append("m.media_file IS NULL")
    sql = (f"SELECT m.message_id FROM messages m WHERE {' AND '.join(cond)} "
           f"ORDER BY m.message_id DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    ids = [r["message_id"] for r in conn.execute(sql, params)]
    if not ids:
        print("Нечего докачивать (всё скачано или нет подходящих сообщений)", file=sys.stderr)
        return 0

    if account is None:
        entry = MF.get(MF.load(root), chat_id)
        account = (entry or {}).get("account", "default")
    max_mb = max_mb or cfg.get("media", {}).get("max_file_mb", 50)
    client = make_client(root, cfg, account)
    got = 0

    async def run():
        nonlocal got
        async with client:
            if not await client.is_user_authorized():
                raise SystemExit(f"Сессия «{account}» не авторизована — tg login --account {account}")
            entity = await client.get_entity(chat_id)
            writer = ChatWriter(root, chat_id)
            for i in range(0, len(ids), 100):
                chunk = ids[i:i + 100]
                msgs = await client.get_messages(entity, ids=chunk)
                for msg in msgs:
                    if msg is None:
                        continue
                    if await maybe_download_media(client, msg, writer, kinds, max_mb):
                        got += 1
                print(f"  скачано {got} из {len(ids)} кандидатов...", file=sys.stderr)
            writer.close()

    asyncio.run(run())
    index_archive(conn, root / cfg["general"].get("archive_dir", "archive"))
    return got
