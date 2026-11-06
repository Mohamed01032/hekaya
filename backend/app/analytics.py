"""Anonymous usage analytics for the museum team (SQLite, no personal data).

Each event stores only: a kind, an artifact or place id, language, audience level,
a question *topic* (a category such as "tickets", never the question text) and a
timestamp. No photos, names, nicknames or chat text are ever stored here.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ALLOWED = {"scan", "chat", "tour", "recognize_fail", "listen", "guide", "place", "dialogue",
           "souvenir", "class_join", "quiz", "sound", "a11y"}
CAIRO = ZoneInfo("Africa/Cairo")


class Analytics:
    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                ts REAL, kind TEXT, artifact_id TEXT, lang TEXT, audience TEXT)""")
            cols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
            if "topic" not in cols:
                c.execute("ALTER TABLE events ADD COLUMN topic TEXT")

    def _conn(self):
        return sqlite3.connect(self.path)

    def log(self, kind: str, artifact_id: str | None = None, lang: str | None = None,
            audience: str | None = None, topic: str | None = None) -> None:
        if kind not in ALLOWED:
            return
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO events(ts, kind, artifact_id, lang, audience, topic) VALUES (?,?,?,?,?,?)",
                      (time.time(), kind, (artifact_id or None) and str(artifact_id)[:64],
                       lang and str(lang)[:8], audience and str(audience)[:8], topic and str(topic)[:24]))

    def summary(self, since_hours: float = 24 * 7) -> dict:
        since = time.time() - since_hours * 3600
        with self._conn() as c:
            q = lambda sql: c.execute(sql, (since,)).fetchall()  # noqa: E731
            by_kind = dict(q("SELECT kind, COUNT(*) FROM events WHERE ts>=? GROUP BY kind"))
            top = q("SELECT artifact_id, COUNT(*) n FROM events WHERE ts>=? AND artifact_id IS NOT NULL "
                    "AND kind IN ('scan','chat','listen','dialogue') GROUP BY artifact_id ORDER BY n DESC LIMIT 10")
            places = q("SELECT artifact_id, COUNT(*) n FROM events WHERE ts>=? AND kind='place' "
                       "AND artifact_id IS NOT NULL GROUP BY artifact_id ORDER BY n DESC LIMIT 10")
            langs = dict(q("SELECT lang, COUNT(*) FROM events WHERE ts>=? AND lang IS NOT NULL GROUP BY lang"))
            levels = dict(q("SELECT audience, COUNT(*) FROM events WHERE ts>=? AND audience IS NOT NULL GROUP BY audience"))
            topics = q("SELECT topic, COUNT(*) n FROM events WHERE ts>=? AND topic IS NOT NULL "
                       "GROUP BY topic ORDER BY n DESC")
            stamps = [r[0] for r in q("SELECT ts FROM events WHERE ts>=?")]
        hours = [0] * 24
        for ts in stamps:
            hours[datetime.fromtimestamp(ts, CAIRO).hour] += 1
        return {"since_hours": since_hours, "total": len(stamps), "events": by_kind,
                "top_artifacts": [{"id": i, "count": n} for i, n in top],
                "places": [{"id": i, "count": n} for i, n in places],
                "languages": langs, "levels": levels,
                "topics": [{"topic": t, "count": n} for t, n in topics], "hours": hours}
