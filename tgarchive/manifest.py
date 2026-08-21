"""Манифест архива: какие чаты отслеживаем, каким аккаунтом, какие медиа качаем.

Лежит в archive/manifest.json — переживает пересборку поисковой базы.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _path(root: Path) -> Path:
    return root / "archive" / "manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(root: Path) -> dict:
    p = _path(root)
    if p.exists():
        return json.loads(p.read_text())
    return {"chats": []}


def save(root: Path, data: dict):
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def get(data: dict, chat_id: int):
    return next((c for c in data["chats"] if c["chat_id"] == chat_id), None)


def upsert(data: dict, entry: dict):
    cur = get(data, entry["chat_id"])
    if cur:
        cur.update({k: v for k, v in entry.items() if k != "added"})
    else:
        entry.setdefault("added", now_iso())
        data["chats"].append(entry)


def bootstrap(root: Path, conn, data: dict) -> int:
    """Чаты, скачанные до появления манифеста, регистрируем как text-only."""
    added = 0
    for r in conn.execute("SELECT DISTINCT chat_id FROM messages"):
        if not get(data, r["chat_id"]):
            data["chats"].append({
                "chat_id": r["chat_id"],
                "account": "default",
                "topics": None,
                "media": [],
                "from": None,
                "added": now_iso(),
                "last_sync": None,
            })
            added += 1
    return added
