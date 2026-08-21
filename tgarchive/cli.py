import json
import re
import sys
import time

import click

from . import db as dbm
from . import search as S
from .config import load_config, project_root
from .indexer import index_archive
from .lemma import Lemmatizer


def open_db():
    root = project_root()
    cfg = load_config(root)
    db_path = root / cfg["general"].get("db", "data/index.db")
    return root, cfg, dbm.connect(db_path), db_path


def chat_map(conn):
    return {r["chat_id"]: r["title"] for r in conn.execute("SELECT chat_id,title FROM chats")}


def topic_map(conn):
    return {(r["chat_id"], r["topic_id"]): r["title"]
            for r in conn.execute("SELECT chat_id,topic_id,title FROM topics")}


def resolve_chat(conn, cfg, token: str) -> int:
    token = str(token).strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    aliases = cfg.get("aliases", {})
    if token in aliases:
        return int(aliases[token])
    rows = conn.execute(
        "SELECT chat_id,title FROM chats WHERE title LIKE ? COLLATE NOCASE",
        (f"%{token}%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]["chat_id"]
    if not rows:
        raise click.ClickException(
            f"Чат «{token}» не найден. Смотри `tg chats`, задай алиас в config.toml "
            f"или запусти `tg meta`, чтобы подтянуть названия."
        )
    variants = "; ".join(f"{r['title']} ({r['chat_id']})" for r in rows)
    raise click.ClickException(f"Несколько чатов подходят: {variants}")


def fmt_media(r):
    if not r["media_type"]:
        return ""
    short = r["media_kind"] or r["media_type"].removeprefix("MessageMedia").lower()
    extra = ""
    if r["media_name"] and (r["media_kind"] or "document") in ("document", "audio", "video"):
        extra = f" «{r['media_name']}»"
    if r["transcript"]:
        extra += f": {r['transcript'].replace(chr(10), ' ')}"
    return f" 〈{short}{extra}〉"


def fmt_row(r, chats, topics, show_chat=True, short=0, marker=""):
    date = (r["date"] or "")[:16].replace("T", " ")
    chat = ""
    if show_chat:
        title = chats.get(r["chat_id"]) or str(r["chat_id"])
        tt = topics.get((r["chat_id"], r["topic_id"]))
        if tt:
            chat = f" [{title}/{tt}]"
        elif r["topic_id"]:
            chat = f" [{title}/t{r['topic_id']}]"
        else:
            chat = f" [{title}]"
    sender = r["sender_name"] or (str(r["sender_id"]) if r["sender_id"] else "—")
    reply = f" ↩{r['reply_to']}" if r["reply_to"] else ""
    text = (r["text"] or "").replace("\n", " ⏎ ")
    if short and len(text) > short:
        text = text[:short] + "…"
    return f"{marker}#{r['message_id']} {date}{chat} {sender}{reply}:{fmt_media(r)} {text}"


def row_json(r, chats, topics):
    return {
        "chat_id": r["chat_id"],
        "chat": chats.get(r["chat_id"]),
        "topic_id": r["topic_id"],
        "topic": topics.get((r["chat_id"], r["topic_id"])),
        "message_id": r["message_id"],
        "date": r["date"],
        "sender_id": r["sender_id"],
        "sender": r["sender_name"],
        "reply_to": r["reply_to"],
        "media_type": r["media_type"],
        "media_kind": r["media_kind"],
        "media_file": r["media_file"],
        "media_name": r["media_name"],
        "transcript": r["transcript"],
        "text": r["text"],
        "links": json.loads(r["links"]) if r["links"] else [],
    }


def emit_rows(rows, chats, topics, as_json, short, show_chat=True):
    for r in rows:
        if as_json:
            click.echo(json.dumps(row_json(r, chats, topics), ensure_ascii=False))
        else:
            click.echo(fmt_row(r, chats, topics, show_chat=show_chat, short=short))


@click.group(help="Архив и умный поиск по истории Telegram-чатов.")
def main():
    pass


@main.command()
@click.option("--rebuild", is_flag=True, help="пересобрать индекс с нуля")
def index(rebuild):
    """Проиндексировать новые строки из archive/*.jsonl."""
    root, cfg, conn, _ = open_db()
    archive_dir = root / cfg["general"].get("archive_dir", "archive")
    t0 = time.time()
    n = index_archive(conn, archive_dir, rebuild=rebuild)
    total = conn.execute("SELECT count(*) c FROM messages").fetchone()["c"]
    click.echo(f"Добавлено: {n} сообщений за {time.time() - t0:.1f} с (в индексе всего {total})")


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--any", "any_mode", is_flag=True, help="любое из слов (OR) — широкая выборка по теме")
@click.option("--chat", "chat_tokens", multiple=True, help="id, алиас или часть названия; можно несколько")
@click.option("--topic", type=int, default=None, help="id топика (см. tg topics)")
@click.option("--sender", help="имя (подстрока) или числовой id отправителя")
@click.option("--from", "date_from", help="с даты: 2025, 2025-06 или 2025-06-15")
@click.option("--to", "date_to", help="по дату (включительно)")
@click.option("--media", help="photo|document|webpage|poll|any|none")
@click.option("--limit", default=500, show_default=True, help="0 = без лимита")
@click.option("--rank", is_flag=True, help="сортировать по релевантности, а не хронологии")
@click.option("--around", default=0, help="показать N соседних сообщений вокруг каждого найденного")
@click.option("--count", "count_only", is_flag=True, help="только число совпадений")
@click.option("--by-chat", is_flag=True, help="распределение совпадений по чатам")
@click.option("--by-topic", is_flag=True, help="распределение по топикам (нужен один --chat)")
@click.option("--by-sender", is_flag=True, help="распределение по отправителям")
@click.option("--json", "as_json", is_flag=True, help="вывод JSONL")
@click.option("--short", default=0, help="обрезать текст до N символов")
def search(query, any_mode, chat_tokens, topic, sender, date_from, date_to, media,
           limit, rank, around, count_only, by_chat, by_topic, by_sender, as_json, short):
    """Полнотекстовый поиск с русской морфологией.

    Слова через пробел = AND, явное OR поддерживается, "фраза в кавычках", префикс*.
    """
    root, cfg, conn, _ = open_db()
    lem = Lemmatizer()
    try:
        match = S.build_match(" ".join(query), lem, any_mode)
    except ValueError as e:
        raise click.ClickException(str(e))
    chat_ids = [resolve_chat(conn, cfg, c) for c in chat_tokens] or None
    if by_topic and (not chat_ids or len(chat_ids) != 1):
        raise click.ClickException("--by-topic требует ровно один --chat")
    cond, params = S.where_filters(chat_ids, topic, sender, date_from, date_to, media)
    chats, topics_m = chat_map(conn), topic_map(conn)

    if count_only:
        r = S.run_count(conn, match, cond, params)
        rng = f"  [{r['d0'][:10]} … {r['d1'][:10]}]" if r["c"] else ""
        click.echo(f"{r['c']} сообщений{rng}")
        return

    if by_chat or by_topic or by_sender:
        by = "chat" if by_chat else "topic" if by_topic else "sender"
        for r in S.group_counts(conn, match, cond, params, by):
            k = r["k"]
            if by == "chat":
                label = f"{chats.get(k, '?')} ({k})"
            elif by == "topic":
                t_title = topics_m.get((chat_ids[0], k))
                label = f"{t_title or 'без топика'} (topic {k})"
            else:
                label = str(k)
            click.echo(f"{r['c']:>7}  {label}  [{(r['d0'] or '')[:10]} … {(r['d1'] or '')[:10]}]")
        return

    rows = S.run_search(conn, match, cond, params, limit=limit, rank=rank)
    if around and rows:
        for r in rows:
            rb, pv, ra = S.context_rows(conn, r["chat_id"], r["message_id"], around, around)
            title = chats.get(r["chat_id"], r["chat_id"])
            click.echo(f"--- {title} #{r['message_id']} ---")
            for x in rb:
                click.echo(fmt_row(x, chats, topics_m, short=short))
            click.echo(fmt_row(pv, chats, topics_m, short=short, marker=">>> "))
            for x in ra:
                click.echo(fmt_row(x, chats, topics_m, short=short))
    else:
        emit_rows(rows, chats, topics_m, as_json, short)
    if limit and len(rows) == limit:
        click.echo(f"[показаны первые {limit} в хронологическом порядке; "
                   f"уточни фильтры или возьми всё через --limit 0]", err=True)


@main.command()
@click.option("--chat", required=True, help="id, алиас или часть названия")
@click.option("--id", "message_id", required=True, type=int, help="message_id опорного сообщения")
@click.option("--before", default=30, show_default=True)
@click.option("--after", default=30, show_default=True)
@click.option("--whole-chat", is_flag=True, help="контекст по всему чату, а не внутри топика")
@click.option("--json", "as_json", is_flag=True)
@click.option("--short", default=0)
def context(chat, message_id, before, after, whole_chat, as_json, short):
    """Сообщения вокруг заданного (читать тред/обсуждение)."""
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat)
    rb, pv, ra = S.context_rows(conn, cid, message_id, before, after, same_topic=not whole_chat)
    if pv is None:
        raise click.ClickException(f"Сообщение #{message_id} в чате {cid} не найдено")
    chats, topics_m = chat_map(conn), topic_map(conn)
    for r in rb:
        click.echo(json.dumps(row_json(r, chats, topics_m), ensure_ascii=False) if as_json
                   else fmt_row(r, chats, topics_m, short=short))
    click.echo(json.dumps(row_json(pv, chats, topics_m), ensure_ascii=False) if as_json
               else fmt_row(pv, chats, topics_m, short=short, marker=">>> "))
    for r in ra:
        click.echo(json.dumps(row_json(r, chats, topics_m), ensure_ascii=False) if as_json
                   else fmt_row(r, chats, topics_m, short=short))


@main.command()
@click.option("--chat", required=True, help="id, алиас или часть названия")
@click.option("--topic", type=int, default=None)
@click.option("--sender")
@click.option("--from", "date_from")
@click.option("--to", "date_to")
@click.option("--limit", default=2000, show_default=True, help="0 = без лимита")
@click.option("--json", "as_json", is_flag=True)
@click.option("--short", default=0)
def dump(chat, topic, sender, date_from, date_to, limit, as_json, short):
    """Хронологический кусок чата/топика целиком (для вдумчивого чтения)."""
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat)
    cond, params = S.where_filters([cid], topic, sender, date_from, date_to, None)
    sql = f"SELECT * FROM messages m WHERE {' AND '.join(cond)} ORDER BY m.message_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    chats, topics_m = chat_map(conn), topic_map(conn)
    click.echo(f"{chats.get(cid, cid)}: {len(rows)} сообщений", err=True)
    emit_rows(rows, chats, topics_m, as_json, short, show_chat=False)
    if limit and len(rows) == limit:
        click.echo(f"[обрезано на {limit}; --limit 0 чтобы взять всё]", err=True)


@main.command()
def chats():
    """Чаты в архиве: объёмы, периоды, топики."""
    root, cfg, conn, _ = open_db()
    rows = conn.execute(
        "SELECT chat_id, count(*) n, min(date) d0, max(date) d1, "
        "count(DISTINCT topic_id) nt FROM messages GROUP BY 1 ORDER BY n DESC"
    ).fetchall()
    meta = {r["chat_id"]: r for r in conn.execute("SELECT * FROM chats")}
    if not rows:
        click.echo("Архив пуст — запусти `tg index`.")
        return
    from . import manifest as MF
    mf = {e["chat_id"]: e for e in MF.load(root)["chats"]}
    for r in rows:
        m = meta.get(r["chat_id"])
        title = (m["title"] if m else None) or "(без названия — запусти `tg meta`)"
        kind = f"  {m['type']}" if m and m["type"] else ""
        e = mf.get(r["chat_id"])
        track = ""
        if e:
            media_s = ",".join(e.get("media") or []) or "текст"
            last = (e.get("last_sync") or "—")[:10]
            track = f"  [синк: {last}; медиа: {media_s}]"
        click.echo(f"{r['chat_id']:>15}  {r['n']:>8} сообщ.  {r['d0'][:10]} … {r['d1'][:10]}  "
                   f"топиков: {r['nt']:>3}{kind}  {title}{track}")


@main.command()
@click.option("--chat", required=True, help="id, алиас или часть названия")
def topics(chat):
    """Топики супергруппы: объёмы и периоды."""
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat)
    rows = conn.execute(
        "SELECT topic_id k, count(*) c, min(date) d0, max(date) d1 "
        "FROM messages WHERE chat_id=? GROUP BY 1 ORDER BY c DESC", (cid,)
    ).fetchall()
    tm = topic_map(conn)
    for r in rows:
        title = tm.get((cid, r["k"])) or ("без топика" if r["k"] is None else f"t{r['k']}")
        click.echo(f"{(r['k'] if r['k'] is not None else '—'):>6}  {r['c']:>8} сообщ.  "
                   f"{r['d0'][:10]} … {r['d1'][:10]}  {title}")


@main.command()
def status():
    """Состояние индекса."""
    root, cfg, conn, db_path = open_db()
    msgs = conn.execute("SELECT count(*) c FROM messages").fetchone()["c"]
    nchats = conn.execute("SELECT count(DISTINCT chat_id) c FROM messages").fetchone()["c"]
    rng = conn.execute("SELECT min(date) a, max(date) b FROM messages").fetchone()
    nfiles = conn.execute("SELECT count(*) c FROM files").fetchone()["c"]
    nmedia = conn.execute("SELECT count(*) c FROM messages WHERE media_file IS NOT NULL").fetchone()["c"]
    ntrans = conn.execute("SELECT count(*) c FROM messages WHERE transcript IS NOT NULL").fetchone()["c"]
    size = db_path.stat().st_size / 1e6 if db_path.exists() else 0
    click.echo(f"Сообщений: {msgs} в {nchats} чатах, файлов JSONL: {nfiles}")
    if msgs:
        click.echo(f"Период: {rng['a'][:10]} … {rng['b'][:10]}")
    click.echo(f"Скачанных медиа: {nmedia}, расшифрованных голосовых: {ntrans}")
    click.echo(f"База: {db_path} ({size:.0f} МБ)")


@main.command()
@click.option("--account", default=None, help="имя аккаунта из config.toml")
@click.option("--limit", default=0, show_default=True, help="0 = все диалоги")
def dialogs(account, limit):
    """Все чаты аккаунта в Telegram (что ещё можно заархивировать)."""
    root, cfg, conn, _ = open_db()
    from . import meta as M
    rows = M.list_dialogs(root, cfg, account, limit)
    archived = {r["chat_id"] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")}
    for did, kind, name in rows:
        mark = " ✓в архиве" if did in archived else ""
        click.echo(f"{did:>15}  {kind:<10} {name}{mark}")


@main.command()
@click.option("--account", default=None)
@click.option("--chat", "chat_tokens", multiple=True, help="по умолчанию — все чаты архива")
def meta(account, chat_tokens):
    """Подтянуть из Telegram названия чатов и топиков."""
    root, cfg, conn, _ = open_db()
    if chat_tokens:
        ids = [resolve_chat(conn, cfg, c) for c in chat_tokens]
    else:
        ids = [r["chat_id"] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")]
    if not ids:
        raise click.ClickException("В архиве нет чатов — сначала `tg index`")
    from . import meta as M
    for line in M.meta_sync(root, cfg, conn, ids, account):
        click.echo(line)


@main.command()
@click.option("--account", default=None, help="имя аккаунта из config.toml [accounts]")
def login(account):
    """Авторизовать сессию Telegram (интерактивно: телефон, код, 2FA)."""
    root, cfg, _, _ = open_db()
    from . import meta as M
    M.login(root, cfg, account)


@main.command()
@click.option("--chat", "chat_tokens", multiple=True, help="по умолчанию — все чаты из манифеста")
@click.option("--media", "media_s", default=None, help="разово переопределить типы медиа: photo,voice | all | none")
@click.option("--limit", default=0, help="ограничить число сообщений за прогон (для проверки)")
@click.option("--account", default=None, help="синкать только чаты этого аккаунта")
def sync(chat_tokens, media_s, limit, account):
    """Докачать НОВЫЕ сообщения для отслеживаемых чатов (старые строки не трогаются)."""
    from .downloader import parse_kinds, sync_chats
    root, cfg, conn, _ = open_db()
    chat_ids = [resolve_chat(conn, cfg, c) for c in chat_tokens] or None
    try:
        media = parse_kinds(media_s)
    except ValueError as e:
        raise click.ClickException(str(e))
    st = sync_chats(root, cfg, conn, chat_ids=chat_ids, media_override=media,
                    limit=limit or None, account_only=account)
    click.echo(f"Готово: чатов {st['chats']}, +{st['messages']} сообщений, +{st['media']} медиа")


@main.command()
@click.option("--chat", required=True, help="id, @username, ссылка t.me или часть названия диалога")
@click.option("--topic", "topics", multiple=True, type=int, help="качать только эти топики (id)")
@click.option("--from", "date_from", help="скачать начиная с даты (2025, 2025-06, 2025-06-15)")
@click.option("--to", "date_to", help="по дату включительно")
@click.option("--media", "media_s", default=None, help="photo,voice,... | all | none (по умолчанию из config.toml)")
@click.option("--account", default="default")
@click.option("--limit", default=0, help="ограничить число сообщений (для пробы)")
@click.option("--max-mb", default=0, help="лимит размера файла, МБ (по умолчанию из config.toml)")
def download(chat, topics, date_from, date_to, media_s, account, limit, max_mb):
    """Скачать чат/топики в архив и поставить на отслеживание (tg sync)."""
    from .downloader import download_chat, parse_kinds
    root, cfg, conn, _ = open_db()
    try:
        media = parse_kinds(media_s)
    except ValueError as e:
        raise click.ClickException(str(e))
    res = download_chat(root, cfg, conn, chat, topics=list(topics) or None,
                        date_from=date_from, date_to=date_to, media=media,
                        account=account, limit=limit or None, max_mb=max_mb or None)
    click.echo(f"«{res['title']}» ({res['chat_id']}): +{res['messages']} сообщений, "
               f"+{res['media']} медиа. Чат добавлен в манифест — дальше просто tg sync.")


@main.command()
@click.option("--chat", required=True, help="id, алиас или часть названия")
@click.option("--media", "media_s", required=True, help="что докачать: photo,voice,... | all")
@click.option("--topic", type=int, default=None)
@click.option("--from", "date_from")
@click.option("--to", "date_to")
@click.option("--limit", default=100, show_default=True, help="0 = без лимита")
@click.option("--account", default=None)
@click.option("--max-mb", default=0)
def media(chat, media_s, topic, date_from, date_to, limit, account, max_mb):
    """Докачать файлы для УЖЕ скачанных сообщений (бэкфилл медиа)."""
    from .downloader import backfill_media, parse_kinds
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat)
    try:
        kinds = parse_kinds(media_s)
    except ValueError as e:
        raise click.ClickException(str(e))
    n = backfill_media(root, cfg, conn, cid, kinds, topic=topic, date_from=date_from,
                       date_to=date_to, limit=limit or None, account=account,
                       max_mb=max_mb or None)
    click.echo(f"Скачано файлов: {n}")


@main.command()
@click.option("--chat", default=None, help="ограничить одним чатом")
@click.option("--provider", default=None, help="whisper-local | openai | telegram (по умолчанию из config.toml)")
@click.option("--limit", default=50, show_default=True)
@click.option("--model", default=None, help="модель whisper: tiny/base/small/medium/large-v3")
@click.option("--account", default="default")
def transcribe(chat, provider, limit, model, account):
    """Расшифровать скачанные голосовые/кружки в текст (попадает в поиск)."""
    from .transcribe import run_transcribe
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat) if chat else None
    n = run_transcribe(root, cfg, conn, provider_name=provider, chat_id=cid,
                       limit=limit or None, account=account, model=model)
    click.echo(f"Расшифровано: {n}")


@main.command()
@click.option("--host", default=None, help="по умолчанию из config [web].host или 127.0.0.1")
@click.option("--port", default=0, help="по умолчанию из config [web].port или 8377")
@click.option("--no-browser", is_flag=True, help="не открывать браузер автоматически")
def web(host, port, no_browser):
    """Локальный веб-просмотрщик архива (картинки, голосовые, поиск)."""
    from .webapp import serve
    root, cfg, conn, _ = open_db()
    conn.close()
    host = host or cfg.get("web", {}).get("host", "127.0.0.1")
    port = port or cfg.get("web", {}).get("port", 8377)
    serve(root, cfg, host=host, port=port, open_browser=not no_browser)


@main.command()
def tui():
    """Просмотрщик прямо в терминале (textual)."""
    from .tui import run
    run()


@main.command()
@click.option("--chat", required=True)
def untrack(chat):
    """Убрать чат из манифеста (файлы архива остаются, tg sync его пропускает)."""
    from . import manifest as MF
    root, cfg, conn, _ = open_db()
    cid = resolve_chat(conn, cfg, chat)
    data = MF.load(root)
    entry = MF.get(data, cid)
    if not entry:
        raise click.ClickException(f"Чата {cid} нет в манифесте")
    data["chats"].remove(entry)
    MF.save(root, data)
    click.echo(f"{cid}: снят с отслеживания (архив на диске не тронут)")


if __name__ == "__main__":
    main()
