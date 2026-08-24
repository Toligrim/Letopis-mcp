import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

ENV_API_ID = ("TG_RAG_TELEGRAM_API_ID", "TELEGRAM_API_ID", "API_ID")
ENV_API_HASH = ("TG_RAG_TELEGRAM_API_HASH", "TELEGRAM_API_HASH", "API_HASH")
ENV_SESSION = ("TG_RAG_SESSION_PATH", "TELEGRAM_SESSION_PATH")


def _env_first(names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def make_client(root: Path, cfg: dict, account: str | None = None):
    from dotenv import load_dotenv

    load_dotenv(root / ".env")

    name = account or "default"
    acc_cfg = cfg.get("accounts", {}).get(name)
    # запись аккаунта — либо просто путь к сессии (str), либо таблица
    # {session=..., api_id=..., api_hash=...} для аккаунта со своим приложением
    cfg_session = acc_cfg if isinstance(acc_cfg, str) else (acc_cfg or {}).get("session")
    api_id = (acc_cfg or {}).get("api_id") if isinstance(acc_cfg, dict) else None
    api_hash = (acc_cfg or {}).get("api_hash") if isinstance(acc_cfg, dict) else None
    api_id = api_id or _env_first(ENV_API_ID)
    api_hash = api_hash or _env_first(ENV_API_HASH)
    if not api_id or not api_hash:
        raise SystemExit("В .env не найдены TG_RAG_TELEGRAM_API_ID / TG_RAG_TELEGRAM_API_HASH")

    candidates = []
    if name == "default" and _env_first(ENV_SESSION):
        candidates.append(_env_first(ENV_SESSION))
    if cfg_session:
        candidates.append(cfg_session)
    if not candidates:
        raise SystemExit(f"Аккаунт «{name}» не найден в config.toml [accounts]")
    paths = [Path(s) if Path(s).is_absolute() else root / s for s in candidates]
    # берём первый существующий файл сессии; если нет ни одного — последний кандидат (для tg login)
    sess_path = next((p for p in paths if p.exists()), paths[-1])

    from telethon import TelegramClient

    return TelegramClient(str(sess_path).removesuffix(".session"), int(api_id), api_hash,
                          flood_sleep_threshold=86400)


def _run_authorized(client, coro_fn):
    async def go():
        async with client:
            if not await client.is_user_authorized():
                raise SystemExit("Сессия не авторизована — запусти `tg login`")
            return await coro_fn(client)

    return asyncio.run(go())


def _kind(entity) -> str:
    t = type(entity).__name__
    if getattr(entity, "forum", False):
        return "forum"
    if t == "Channel":
        return "channel" if getattr(entity, "broadcast", False) else "supergroup"
    if t == "Chat":
        return "group"
    if getattr(entity, "bot", False):
        return "bot"
    return "user"


def list_dialogs(root, cfg, account=None, limit=0):
    """Все диалоги аккаунта: (id, тип, название). limit=0 — без ограничения."""
    client = make_client(root, cfg, account)

    async def go(client):
        rows = []
        async for d in client.iter_dialogs(limit=limit or None):
            rows.append((d.id, _kind(d.entity), d.name or ""))
        return rows

    return _run_authorized(client, go)


def meta_sync(root, cfg, conn, chat_ids, account=None):
    """Подтягивает названия чатов и топиков в таблицы chats/topics."""
    client = make_client(root, cfg, account)

    async def go(client):
        from telethon import utils
        try:
            from telethon.tl.functions.messages import GetForumTopicsRequest  # telethon >= 1.43
            _topics_kw = "peer"
        except ImportError:
            from telethon.tl.functions.channels import GetForumTopicsRequest
            _topics_kw = "channel"

        out = []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for cid in chat_ids:
            try:
                entity = await client.get_entity(cid)
            except Exception as e:
                out.append(f"!! {cid}: {e}")
                continue
            title = utils.get_display_name(entity) or str(cid)
            kind = _kind(entity)
            conn.execute(
                "INSERT INTO chats(chat_id,title,username,type,is_forum,updated) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title, username=excluded.username,"
                "type=excluded.type, is_forum=excluded.is_forum, updated=excluded.updated",
                (cid, title, getattr(entity, "username", None), kind,
                 int(kind == "forum"), now),
            )
            n_topics = 0
            if kind == "forum":
                offset_date, offset_id, offset_topic = None, 0, 0
                while True:
                    r = await client(GetForumTopicsRequest(
                        **{_topics_kw: entity}, offset_date=offset_date, offset_id=offset_id,
                        offset_topic=offset_topic, limit=100))
                    named = [t for t in r.topics if getattr(t, "title", None)]
                    for t in named:
                        conn.execute(
                            "INSERT INTO topics(chat_id,topic_id,title,updated) VALUES(?,?,?,?) "
                            "ON CONFLICT(chat_id,topic_id) DO UPDATE SET title=excluded.title,"
                            "updated=excluded.updated",
                            (cid, t.id, t.title, now))
                    n_topics += len(named)
                    if len(r.topics) < 100:
                        break
                    last = r.topics[-1]
                    offset_topic = last.id
                    offset_id = getattr(last, "top_message", 0)
            conn.commit()
            out.append(f"{cid}: «{title}» ({kind}), топиков: {n_topics}")
        return out

    return _run_authorized(client, go)


def login(root, cfg, account=None):
    import telethon.sync  # noqa: F401  (включает синхронные обёртки)

    client = make_client(root, cfg, account)
    with client:
        me = client.get_me()
        print(f"Авторизован: {me.first_name or ''} @{me.username or ''} (id {me.id})")
