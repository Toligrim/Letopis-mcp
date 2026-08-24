import json
import re
import sys
import time
from pathlib import Path

from .db import bump_index_revision
from .lemma import Lemmatizer, normalize

CHAT_DIR_RE = re.compile(r"^-?\d+$")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def archive_files(archive_dir: Path):
    if not archive_dir.is_dir():
        return
    for chat_dir in sorted(archive_dir.iterdir()):
        if not chat_dir.is_dir() or not CHAT_DIR_RE.match(chat_dir.name):
            continue
        # месячные файлы идут до сайдкаров (media_index, transcripts) — сортировка это гарантирует
        yield from sorted(chat_dir.glob("*.jsonl"))


def compose_index_text(text, poll=None, media_name=None, transcript=None) -> str:
    """Текст, который попадает в полнотекстовый индекс для одного сообщения."""
    parts = [text or ""]
    if poll:
        try:
            p = json.loads(poll) if isinstance(poll, str) else poll
            parts.append(p.get("question") or "")
            parts.extend(a or "" for a in p.get("answers", []))
        except Exception:
            pass
    if media_name:
        parts.append(str(media_name))
    if transcript:
        parts.append(transcript)
    return "\n".join(x for x in parts if x)


def _fts_set(conn, lem, rowid, itext):
    conn.execute("DELETE FROM fts WHERE rowid=?", (rowid,))
    if itext.strip():
        conn.execute(
            "INSERT INTO fts(rowid,orig,lemma) VALUES(?,?,?)",
            (rowid, normalize(itext), lem.text(itext)),
        )


def _refresh_message_fts(conn, lem, chat_id, message_id):
    row = conn.execute(
        "SELECT id,text,poll,media_name,transcript FROM messages WHERE chat_id=? AND message_id=?",
        (chat_id, message_id),
    ).fetchone()
    if row:
        _fts_set(conn, lem, row["id"],
                 compose_index_text(row["text"], row["poll"], row["media_name"], row["transcript"]))


def rebuild_fts(conn, lem=None):
    """Перезаливает FTS из таблицы messages (после смены схемы индекса)."""
    lem = lem or Lemmatizer()
    conn.execute("DELETE FROM fts")
    n = 0
    for row in conn.execute("SELECT id,text,poll,media_name,transcript FROM messages"):
        itext = compose_index_text(row["text"], row["poll"], row["media_name"], row["transcript"])
        if itext.strip():
            conn.execute(
                "INSERT INTO fts(rowid,orig,lemma) VALUES(?,?,?)",
                (row["id"], normalize(itext), lem.text(itext)),
            )
        n += 1
        if n % 100000 == 0:
            print(f"  fts: {n}...", file=sys.stderr)
    conn.execute("DELETE FROM kv WHERE k='fts_dirty'")
    bump_index_revision(conn)
    conn.commit()


def _h_message(conn, lem, chat_id, d) -> int:
    text = d.get("text") or ""

    def j(key):
        v = d.get(key)
        return json.dumps(v, ensure_ascii=False) if v else None

    cur = conn.execute(
        "INSERT OR IGNORE INTO messages(chat_id,message_id,sender_id,sender_name,date,"
        "edit_date,topic_id,reply_to,text,media_type,links,forward,raw,reactions,service,poll) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.get("chat_id", chat_id),
            d["message_id"],
            d.get("sender_id"),
            d.get("sender_name"),
            d.get("date") or "",
            d.get("edit_date"),
            d.get("topic_id"),
            d.get("reply_to_message_id"),
            text,
            d.get("media_type"),
            j("links"),
            j("forward"),
            j("raw"),
            j("reactions"),
            j("service"),
            j("poll"),
        ),
    )
    if not cur.rowcount:
        return 0
    itext = compose_index_text(text, d.get("poll"))
    if itext.strip():
        conn.execute(
            "INSERT INTO fts(rowid,orig,lemma) VALUES(?,?,?)",
            (cur.lastrowid, normalize(itext), lem.text(itext)),
        )
    return 1


def _h_media(conn, lem, chat_id, d) -> int:
    cur = conn.execute(
        "UPDATE messages SET media_kind=?, media_file=?, media_name=? WHERE chat_id=? AND message_id=?",
        (d.get("kind"), d.get("file"), d.get("name"), chat_id, d["message_id"]),
    )
    if not cur.rowcount:
        print(f"  !! media_index: сообщение #{d.get('message_id')} ещё не в базе", file=sys.stderr)
        return 0
    if d.get("name"):  # имя файла должно находиться поиском
        _refresh_message_fts(conn, lem, chat_id, d["message_id"])
    return 1


def _h_transcript(conn, lem, chat_id, d) -> int:
    cur = conn.execute(
        "UPDATE messages SET transcript=? WHERE chat_id=? AND message_id=?",
        (d.get("text"), chat_id, d["message_id"]),
    )
    if not cur.rowcount:
        print(f"  !! transcripts: сообщение #{d.get('message_id')} ещё не в базе", file=sys.stderr)
        return 0
    _refresh_message_fts(conn, lem, chat_id, d["message_id"])
    return 1


def index_archive(conn, archive_dir: Path, rebuild: bool = False) -> int:
    """Дочитывает новые строки JSONL (файлы append-only) и кладёт их в индекс."""
    if rebuild:
        from .db import SCHEMA

        conn.executescript("DELETE FROM messages; DELETE FROM files; DROP TABLE IF EXISTS fts;")
        conn.executescript(SCHEMA)
        bump_index_revision(conn)
        conn.commit()

    lem = Lemmatizer()
    if conn.execute("SELECT 1 FROM kv WHERE k='fts_dirty'").fetchone():
        print("Схема поиска обновилась — пересобираю FTS из базы (раз в обновление)...", file=sys.stderr)
        rebuild_fts(conn, lem)

    total = 0
    t0 = time.time()
    for f in archive_files(archive_dir):
        rel = str(f.relative_to(archive_dir))
        chat_id = int(f.parent.name)
        stem = f.stem
        if MONTH_RE.match(stem):
            handler = _h_message
        elif stem == "media_index":
            handler = _h_media
        elif stem == "transcripts":
            handler = _h_transcript
        else:
            continue
        row = conn.execute("SELECT bytes_done FROM files WHERE path=?", (rel,)).fetchone()
        offset = row["bytes_done"] if row else 0
        size = f.stat().st_size
        if size < offset:
            print(f"!! {rel}: файл стал короче прежнего — запусти `tg index --rebuild`", file=sys.stderr)
            continue
        if size == offset:
            continue
        added = 0
        with open(f, "rb") as fh:
            fh.seek(offset)
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  !! {rel}: битая JSON-строка пропущена", file=sys.stderr)
                    continue
                added += handler(conn, lem, chat_id, d)
            pos = fh.tell()
        conn.execute(
            "INSERT INTO files(path,bytes_done,lines_done,mtime) VALUES(?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET bytes_done=excluded.bytes_done,"
            "lines_done=files.lines_done+?, mtime=excluded.mtime",
            (rel, pos, added, f.stat().st_mtime, added),
        )
        bump_index_revision(conn)
        conn.commit()
        total += added
        if added:
            rate = total / max(time.time() - t0, 0.001)
            print(f"  {rel}: +{added}  (всего {total}, {rate:.0f} зап/с)", file=sys.stderr)
    return total
