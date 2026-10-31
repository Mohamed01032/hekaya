"""Anonymous usage analytics for the museum team (SQLite, no personal data).

Events: scan, chat, tour, recognize_fail. Each event stores only an artifact id,
language, audience and a timestamp — never photos, names or chat text.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

ALLOWED = {"scan", "chat", "tour", "recognize_fail", "listen"}


class Analytics:
    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                ts REAL, kind TEXT, artifact_id TEXT, lang TEXT, audience TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def log(self, kind: str, artifact_id: str | None = None, lang: str | None = None,
            audience: str | None = None) -> None:
        if kind not in ALLOWED:
            return
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                      (time.time(), kind, artifact_id, lang, audience))

    def summary(self, since_hours: float = 24 * 7) -> dict:
        since = time.time() - since_hours * 3600
        with self._conn() as c:
            by_kind = dict(c.execute(
                "SELECT kind, COUNT(*) FROM events WHERE ts>=? GROUP BY kind", (since,)).fetchall())
            top = c.execute(
                "SELECT artifact_id, COUNT(*) n FROM events WHERE ts>=? AND artifact_id IS NOT NULL "
                "GROUP BY artifact_id ORDER BY n DESC LIMIT 10", (since,)).fetchall()
            langs = dict(c.execute(
                "SELECT lang, COUNT(*) FROM events WHERE ts>=? AND lang IS NOT NULL GROUP BY lang",
                (since,)).fetchall())
        return {"since_hours": since_hours, "events": by_kind,
                "top_artifacts": [{"id": i, "count": n} for i, n in top], "languages": langs}
