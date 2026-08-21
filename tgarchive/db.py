import sqlite3
from pathlib import Path

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    orig, lemma,
    content='',
    contentless_delete=1,
    tokenize="unicode61 remove_diacritics 2"
);
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    sender_id INTEGER,
    sender_name TEXT,
    date TEXT NOT NULL,
    edit_date TEXT,
    topic_id INTEGER,
    reply_to INTEGER,
    text TEXT,
    media_type TEXT,
    links TEXT,
    forward TEXT,
    raw TEXT,
    reactions TEXT,
    service TEXT,
    poll TEXT,
    media_kind TEXT,
    media_file TEXT,
    media_name TEXT,
    transcript TEXT,
    UNIQUE(chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_msg_chat_topic_mid ON messages(chat_id, topic_id, message_id);
CREATE INDEX IF NOT EXISTS idx_msg_date ON messages(date);
{FTS_SQL}
CREATE TABLE IF NOT EXISTS files(
    path TEXT PRIMARY KEY,
    bytes_done INTEGER NOT NULL DEFAULT 0,
    lines_done INTEGER NOT NULL DEFAULT 0,
    mtime REAL
);
CREATE TABLE IF NOT EXISTS chats(
    chat_id INTEGER PRIMARY KEY,
    title TEXT, username TEXT, type TEXT, is_forum INTEGER,
    account TEXT, raw TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS topics(
    chat_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    title TEXT, raw TEXT, updated TEXT,
    PRIMARY KEY(chat_id, topic_id)
);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""

# колонки, добавленные после первой версии схемы (миграция старых баз)
NEW_COLUMNS = [
    "reactions TEXT",
    "service TEXT",
    "poll TEXT",
    "media_kind TEXT",
    "media_file TEXT",
    "media_name TEXT",
    "transcript TEXT",
]


def _migrate(conn: sqlite3.Connection):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    for col_def in NEW_COLUMNS:
        if col_def.split()[0] not in cols:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {col_def}")
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='fts'").fetchone()
    if row and "contentless_delete" not in (row[0] or ""):
        conn.execute("DROP TABLE fts")
        conn.executescript(FTS_SQL)
        conn.execute("INSERT OR REPLACE INTO kv(k,v) VALUES('fts_dirty','1')")
    conn.commit()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn
