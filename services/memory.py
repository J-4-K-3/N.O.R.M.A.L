# saving + retrieving memory

from __future__ import annotations

from typing import List

from app.db.database import get_db


def _ensure_schema(db):
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    db.commit()


def save_memory(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return

    db = get_db()
    _ensure_schema(db)
    cursor = db.cursor()
    cursor.execute("INSERT INTO memory (text) VALUES (?)", (text,))
    db.commit()
    db.close()


def get_recent_memories(limit: int = 5) -> List[str]:
    limit = max(1, int(limit))
    db = get_db()
    _ensure_schema(db)
    cursor = db.cursor()
    cursor.execute(
        "SELECT text FROM memory ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    db.close()

    # oldest->newest for nicer prompt ordering
    return [r[0] for r in reversed(rows)]

