"""Text-only history in SQLite with an FTS5 index."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import threading

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, source TEXT, kind TEXT, text TEXT, meta TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    text, content='events', content_rowid='id', tokenize="unicode61 remove_diacritics 2");
CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


@dataclass(frozen=True)
class StoredEvent:
    id: int
    ts: float
    source: str
    kind: str
    text: str = ""
    meta: dict = field(default_factory=dict)


def _fts_query(query):
    """Quote every whitespace-separated token so punctuation never reaches the FTS5 parser."""
    tokens = [token.replace('"', "") for token in query.split()]
    return " ".join(f'"{token}"' for token in tokens if token)


def _source_filter(sources):
    if not sources:
        return "", []
    clauses = " OR ".join("(e.source = ? OR e.source LIKE ?)" for _ in sources)
    params = [value for source in sources for value in (source, f"{source}:%")]
    return f" AND ({clauses})", params


class Store:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)

    def add(self, event):
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO events(ts, source, kind, text, meta) VALUES (?, ?, ?, ?, ?)",
                (event.ts, event.source, event.kind, event.text, json.dumps(event.meta)))
            return cursor.lastrowid

    def _rows(self, sql, params):
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [StoredEvent(id, ts, source, kind, text or "", json.loads(meta or "{}"))
                for id, ts, source, kind, text, meta in rows]

    def search(self, query, since=None, until=None, sources=None, limit=20):
        match = _fts_query(query)
        if not match:
            return []
        where, params = _source_filter(sources)
        if since is not None:
            where += " AND e.ts >= ?"
            params.append(since)
        if until is not None:
            where += " AND e.ts <= ?"
            params.append(until)
        return self._rows(
            "SELECT e.id, e.ts, e.source, e.kind, e.text, e.meta FROM events_fts f "
            f"JOIN events e ON e.id = f.rowid WHERE events_fts MATCH ?{where} ORDER BY e.ts DESC LIMIT ?",
            [match, *params, limit])

    def timeline(self, since, until=None, sources=None, limit=500):
        where, params = _source_filter(sources)
        if until is not None:
            where += " AND e.ts <= ?"
            params.append(until)
        return self._rows(
            f"SELECT e.id, e.ts, e.source, e.kind, e.text, e.meta FROM events e WHERE e.ts >= ?{where} "
            "ORDER BY e.ts ASC LIMIT ?", [since, *params, limit])

    def latest(self, kind, source=None):
        where, params = ("", []) if source is None else (" AND e.source = ?", [source])
        rows = self._rows(
            f"SELECT e.id, e.ts, e.source, e.kind, e.text, e.meta FROM events e WHERE e.kind = ?{where} "
            "ORDER BY e.ts DESC LIMIT 1", [kind, *params])
        return rows[0] if rows else None

    def purge(self, older_than):
        with self._lock, self._db:
            return self._db.execute("DELETE FROM events WHERE ts < ?", (older_than,)).rowcount

    def close(self):
        with self._lock:
            self._db.close()
