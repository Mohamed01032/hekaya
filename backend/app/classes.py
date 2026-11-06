"""School-trip classes: a teacher opens a class, students join with a code and nickname,
and a live leaderboard counts the pieces each student talked to and the quiz answers they got right.

Stores only a nickname the student chooses (no real names required), a random student id,
and two counters. Classes expire after 24 hours.
"""
from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from pathlib import Path

ALPHABET = "ACDEFGHJKMNPQRTUVWXY34679"  # no look-alike characters
TTL = 24 * 3600


def score(stamps: int, quiz: int) -> int:
    return stamps * 10 + quiz * 20


class Classes:
    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS classes(code TEXT PRIMARY KEY, name TEXT, created REAL)")
            c.execute("""CREATE TABLE IF NOT EXISTS students(code TEXT, sid TEXT, nick TEXT,
                         stamps INTEGER, quiz INTEGER, updated REAL, PRIMARY KEY(code, sid))""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def _alive(self, c, code: str):
        row = c.execute("SELECT name, created FROM classes WHERE code=?", (code,)).fetchone()
        return row if row and time.time() - row[1] < TTL else None

    def create(self, name: str) -> str:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM classes WHERE created<?", (time.time() - TTL,))
            for _ in range(20):
                code = "".join(secrets.choice(ALPHABET) for _ in range(5))
                if not c.execute("SELECT 1 FROM classes WHERE code=?", (code,)).fetchone():
                    c.execute("INSERT INTO classes VALUES (?,?,?)", (code, name, time.time()))
                    return code
        raise RuntimeError("could not allocate a class code")

    def join(self, code: str, nick: str) -> str | None:
        with self._lock, self._conn() as c:
            if not self._alive(c, code):
                return None
            if c.execute("SELECT COUNT(*) FROM students WHERE code=?", (code,)).fetchone()[0] >= 80:
                return None
            sid = secrets.token_urlsafe(9)
            c.execute("INSERT INTO students VALUES (?,?,?,?,?,?)", (code, sid, nick, 0, 0, time.time()))
            return sid

    def progress(self, code: str, sid: str, stamps: int, quiz: int) -> bool:
        with self._lock, self._conn() as c:
            if not self._alive(c, code):
                return False
            cur = c.execute("UPDATE students SET stamps=?, quiz=?, updated=? WHERE code=? AND sid=?",
                            (stamps, quiz, time.time(), code, sid))
            return cur.rowcount == 1

    def board(self, code: str) -> dict | None:
        with self._conn() as c:
            alive = self._alive(c, code)
            if not alive:
                return None
            rows = c.execute("SELECT nick, stamps, quiz FROM students WHERE code=?", (code,)).fetchall()
        students = sorted(({"nick": n, "stamps": s, "quiz": q, "score": score(s, q)} for n, s, q in rows),
                          key=lambda r: (-r["score"], r["nick"]))
        return {"code": code, "name": alive[0], "students": students}
