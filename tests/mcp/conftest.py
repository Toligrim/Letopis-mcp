from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tgarchive.db import connect, connect_readonly
from tgarchive.indexer import _fts_set, compose_index_text
from tgarchive.lemma import Lemmatizer
from tgarchive.mcp.cursor import configure_cursor_secret


WORK_CHAT_ID = -1000000000001
SIDE_CHAT_ID = -1000000000002
THIRD_CHAT_ID = -1000000000003


@dataclass(slots=True)
class SyntheticArchive:
    """Temporary, self-contained archive exposed to MCP tests."""

    path: Path
    connection: sqlite3.Connection
    messages: list[dict[str, Any]]
    cases: dict[str, Any]

    @property
    def conn(self) -> sqlite3.Connection:
        """Short alias for tests that pass the connection explicitly."""
        return self.connection


def _public_id(chat_id: int, message_id: int) -> str:
    return f"tg:{chat_id}:{message_id}"


def _insert_message(
    conn: sqlite3.Connection,
    lem: Lemmatizer,
    rows: list[dict[str, Any]],
    *,
    chat_id: int,
    message_id: int,
    date: str,
    text: str = "",
    topic_id: int | None = None,
    sender_id: int | None = None,
    sender_name: str | None = None,
    media_type: str | None = None,
    media_kind: str | None = None,
    media_name: str | None = None,
    transcript: str | None = None,
    poll: str | None = None,
    reactions: str | None = None,
    links: str | None = None,
) -> dict[str, Any]:
    cursor = conn.execute(
        "INSERT INTO messages("
        "chat_id,message_id,sender_id,sender_name,date,topic_id,reply_to,text,"
        "media_type,links,reactions,poll,media_kind,media_name,transcript"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            chat_id,
            message_id,
            sender_id,
            sender_name,
            date,
            topic_id,
            None,
            text,
            media_type,
            links,
            reactions,
            poll,
            media_kind,
            media_name,
            transcript,
        ),
    )
    row_id = int(cursor.lastrowid)

    # Keep the test index identical to production: compose the searchable
    # document and write normalized original/lemma columns through _fts_set.
    _fts_set(
        conn,
        lem,
        row_id,
        compose_index_text(text, poll, media_name, transcript),
    )

    metadata = {
        "db_id": row_id,
        "id": _public_id(chat_id, message_id),
        "chat_id": chat_id,
        "message_id": message_id,
        "topic_id": topic_id,
        "date": date,
        "text": text,
        "transcript": transcript,
        "poll": poll,
        "reactions": reactions,
        "media_kind": media_kind,
        "media_name": media_name,
        "sender_id": sender_id,
        "sender_name": sender_name,
    }
    rows.append(metadata)
    return metadata


def _populate_archive(path: Path) -> None:
    writable = connect(path)
    lem = Lemmatizer()
    rows: list[dict[str, Any]] = []

    writable.executemany(
        "INSERT INTO chats(chat_id,title,username,type,is_forum) VALUES(?,?,?,?,?)",
        [
            (WORK_CHAT_ID, "Тестовая рабочая группа", "work_test", "supergroup", 1),
            (SIDE_CHAT_ID, "Тестовый общий чат", "side_test", "supergroup", 0),
            (THIRD_CHAT_ID, "Третий тестовый чат", "third_test", "group", 0),
        ],
    )
    writable.executemany(
        "INSERT INTO topics(chat_id,topic_id,title) VALUES(?,?,?)",
        [
            (WORK_CHAT_ID, 101, "Работа"),
            (WORK_CHAT_ID, 202, "Проект"),
            (WORK_CHAT_ID, 303, "Архив"),
        ],
    )

    dates = (
        "2023-01-15T10:00:00",
        "2024-03-20T10:00:00",
        "2024-09-10T10:00:00",
        "2025-02-05T10:00:00",
        "2025-11-11T10:00:00",
    )
    senders = ((11, "Алиса"), (12, "Борис"), (13, "Вера"))

    # Fifty ordinary hits spread over three chats, two named topics and the
    # NULL/General scope.  The shared word gives later tests a broad query.
    ordinary_targets = (
        [(WORK_CHAT_ID, 100 + i, 101) for i in range(25)]
        + [(WORK_CHAT_ID, 130 + i, 202) for i in range(10)]
        + [(WORK_CHAT_ID, 140 + i, None) for i in range(5)]
        + [(SIDE_CHAT_ID, 100 + i, None) for i in range(5)]
        + [(THIRD_CHAT_ID, 100 + i, None) for i in range(5)]
    )
    for index, (chat_id, message_id, topic_id) in enumerate(ordinary_targets):
        sender_id, sender_name = senders[index % len(senders)]
        text = f"Пагинация архива: обычное сообщение номер {index}"
        if index == 0:
            text += " " + ("Дополнительный контекст для проверки окна. " * 8) + "редкий якорь"
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=chat_id,
            message_id=message_id,
            topic_id=topic_id,
            date=dates[index % len(dates)],
            text=text,
            sender_id=sender_id,
            sender_name=sender_name,
        )

    # A six-message local burst in one chat/topic, all sharing the same term.
    burst_rows: list[dict[str, Any]] = []
    for offset in range(6):
        burst_rows.append(
            _insert_message(
                writable,
                lem,
                rows,
                chat_id=WORK_CHAT_ID,
                message_id=200 + offset,
                topic_id=101,
                date="2024-03-21T12:00:00",
                text=f"Всплеск пагинации в коротком диалоге, сообщение {offset}",
                sender_id=11 + offset % 3,
                sender_name=senders[offset % len(senders)][1],
            )
        )

    duplicate_text = "Одинаковая заметка для проверки разнообразия"
    duplicate_rows = [
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=300,
            topic_id=202,
            date="2024-09-12T10:00:00",
            text=duplicate_text,
            sender_id=12,
            sender_name="Борис",
        ),
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=301,
            topic_id=202,
            date="2024-09-12T10:01:00",
            text=duplicate_text,
            sender_id=13,
            sender_name="Вера",
        ),
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=302,
            topic_id=202,
            date="2024-09-12T10:02:00",
            text="Одинаковая заметка для проверки разнообразия сегодня",
            sender_id=11,
            sender_name="Алиса",
        ),
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=303,
            topic_id=202,
            date="2024-09-12T10:03:00",
            text="Одинаковая заметка для проверки разнообразия завтра",
            sender_id=12,
            sender_name="Борис",
        ),
    ]

    near_duplicate_rows = [
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=304,
            topic_id=202,
            date="2024-09-12T10:04:00",
            text=(
                "Сравнение документов в архиве показывает устойчивую структуру "
                "сообщения для проверки поиска контекста качества выдачи и точности "
                "ответа в рабочем обсуждении команды летописи за прошлый год при "
                "текущей нагрузке одинаковых условиях хранения данных последовательной "
                "индексации повторной проверки и последующего анализа результата "
                "пользователями проекта без изменения исходного порядка сообщений и "
                "без потери важных деталей исследования пользователями"
            ),
            sender_id=11,
            sender_name="Алиса",
        ),
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=WORK_CHAT_ID,
            message_id=305,
            topic_id=202,
            date="2024-09-12T10:05:00",
            text=(
                "Сравнение документов в архиве показывает устойчивую структуру "
                "сообщения для проверки поиска контекста качества выдачи и точности "
                "ответа в публичном обсуждении команды летописи за прошлый год при "
                "текущей нагрузке одинаковых условиях хранения данных последовательной "
                "индексации повторной проверки и последующего анализа результата "
                "пользователями проекта без изменения исходного порядка сообщений и "
                "без потери важных деталей исследования пользователями"
            ),
            sender_id=12,
            sender_name="Борис",
        ),
    ]

    transcript_row = _insert_message(
        writable,
        lem,
        rows,
        chat_id=WORK_CHAT_ID,
        message_id=500,
        topic_id=None,
        date="2025-02-06T09:00:00",
        text="",
        transcript="Транскрипция голосового сообщения содержит редкий термин фонетика",
        media_type="MessageMediaDocument",
        media_kind="voice",
        sender_id=13,
        sender_name="Вера",
    )
    poll_row = _insert_message(
        writable,
        lem,
        rows,
        chat_id=WORK_CHAT_ID,
        message_id=501,
        topic_id=303,
        date="2025-02-07T09:00:00",
        text="",
        poll=json.dumps(
            {
                "question": "Какой проект выбрать для архива?",
                "answers": ["Летопись", "Другой вариант"],
            },
            ensure_ascii=False,
        ),
        media_type="MessageMediaPoll",
        media_kind="poll",
        sender_id=11,
        sender_name="Алиса",
    )
    reactions_row = _insert_message(
        writable,
        lem,
        rows,
        chat_id=SIDE_CHAT_ID,
        message_id=500,
        topic_id=None,
        date="2025-02-08T09:00:00",
        text="Сообщение с реакциями для проверки",
        reactions=json.dumps({"👍": 3, "❤️": 1}, ensure_ascii=False),
        links=json.dumps(["https://example.test/link"], ensure_ascii=False),
        sender_id=21,
        sender_name="Глеб",
    )
    media_rows = [
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=SIDE_CHAT_ID,
            message_id=501,
            topic_id=None,
            date="2025-11-12T09:00:00",
            text="",
            media_type="MessageMediaPhoto",
            media_kind="photo",
            media_name="отчёт-архив-a.png",
            sender_id=21,
            sender_name="Глеб",
        ),
        _insert_message(
            writable,
            lem,
            rows,
            chat_id=SIDE_CHAT_ID,
            message_id=502,
            topic_id=None,
            date="2025-11-13T09:00:00",
            text="",
            media_type="MessageMediaDocument",
            media_kind="document",
            media_name="отчёт-архив-b.pdf",
            sender_id=22,
            sender_name="Дина",
        ),
    ]

    writable.commit()
    writable.close()

@pytest.fixture
def synthetic_archive(tmp_path: Path, monkeypatch):
    """Build an isolated synthetic archive and expose a read-only snapshot."""
    db_path = tmp_path / "synthetic-index.db"
    _populate_archive(db_path)
    monkeypatch.setenv("LETOPIS_MCP_DB", str(db_path))
    monkeypatch.setenv("LETOPIS_MCP_CURSOR_SECRET", "synthetic-test-cursor-secret")
    configure_cursor_secret(dev_mode=False)
    readonly = connect_readonly(db_path)
    archive = SyntheticArchive(
        path=db_path,
        connection=readonly,
        messages=[],
        cases={},
    )

    # Reconstruct metadata from the temporary DB so callers receive the
    # production-assigned integer row IDs as well as stable public IDs.
    archive.messages = [
        {
            "db_id": int(row["id"]),
            "id": _public_id(int(row["chat_id"]), int(row["message_id"])),
            "chat_id": int(row["chat_id"]),
            "message_id": int(row["message_id"]),
            "topic_id": row["topic_id"],
            "date": row["date"],
            "text": row["text"],
            "transcript": row["transcript"],
            "poll": row["poll"],
            "reactions": row["reactions"],
            "media_kind": row["media_kind"],
            "media_name": row["media_name"],
            "sender_id": row["sender_id"],
            "sender_name": row["sender_name"],
        }
        for row in readonly.execute("SELECT * FROM messages ORDER BY id")
    ]
    by_public_id = {row["id"]: row for row in archive.messages}
    archive.cases = {
        "transcript_only": by_public_id[_public_id(WORK_CHAT_ID, 500)],
        "poll": by_public_id[_public_id(WORK_CHAT_ID, 501)],
        "reactions": by_public_id[_public_id(SIDE_CHAT_ID, 500)],
        "media_only": [
            by_public_id[_public_id(SIDE_CHAT_ID, 501)],
            by_public_id[_public_id(SIDE_CHAT_ID, 502)],
        ],
        "duplicates": [
            by_public_id[_public_id(WORK_CHAT_ID, message_id)]
            for message_id in (300, 301, 302, 303)
        ],
        "near_duplicates": [
            by_public_id[_public_id(WORK_CHAT_ID, message_id)]
            for message_id in (304, 305)
        ],
        "burst": [
            by_public_id[_public_id(WORK_CHAT_ID, message_id)]
            for message_id in range(200, 206)
        ],
        "ordinary": [
            row for row in archive.messages if row["text"].startswith("Пагинация архива")
        ],
    }

    try:
        yield archive
    finally:
        readonly.close()
