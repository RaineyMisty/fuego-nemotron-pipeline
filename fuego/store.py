"""Save article text, schemes, and bucket links in SQLite."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .ingest import InputError, check


class ArticleStore:
    def __init__(self, path, buckets, mode):
        self.path = str(path)
        check(self.path != ":memory:", "Use a file path for the article store.")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY, date TEXT NOT NULL, document TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memberships (
                    article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    bucket_id TEXT NOT NULL,
                    PRIMARY KEY (article_id, bucket_id)
                );
                CREATE INDEX IF NOT EXISTS article_date ON articles(date);
                CREATE INDEX IF NOT EXISTS bucket_members ON memberships(bucket_id);
            ''')
            for key, value in {"schema": "fuego-store.v1", "buckets": buckets,
                               "mode": "demo" if mode == "demo" else "live"}.items():
                encoded = json.dumps(value, sort_keys=True)
                old = db.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
                check(old is None or old[0] == encoded,
                      "Store configuration differs. Use a separate database or the original bucket definitions and demo mode.")
                db.execute("INSERT OR IGNORE INTO config VALUES (?, ?)", (key, encoded))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, documents):
        with self.connect() as db:
            for document in documents:
                article = document["article"]
                old = db.execute("SELECT document FROM articles WHERE id = ?", (article["id"],)).fetchone()
                if old is not None:
                    previous = json.loads(old[0])["article"]
                    check(previous["url"] == article["url"], "An existing ID has a different source URL.")
                    check(previous["text"] is None or article["text"] is not None,
                          "This record already has text. Send its text when updating it.")
                db.execute("INSERT INTO articles VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET date=excluded.date, document=excluded.document",
                           (article["id"], article["date"], json.dumps(document, ensure_ascii=False)))
                db.execute("DELETE FROM memberships WHERE article_id = ?", (article["id"],))
                db.executemany("INSERT INTO memberships VALUES (?, ?)",
                               [(article["id"], bid) for bid in document["bucket_ids"]])

    def select(self, bucket_ids, date):
        if not bucket_ids:
            return []
        marks = ",".join("?" for _ in bucket_ids)
        with self.connect() as db:
            rows = db.execute(f"""SELECT DISTINCT a.id, a.document FROM articles a
                JOIN memberships m ON a.id=m.article_id
                WHERE a.date=? AND m.bucket_id IN ({marks}) ORDER BY a.id LIMIT 101""",
                              [date, *bucket_ids]).fetchall()
        if len(rows) > 100:
            raise InputError("More than 100 records match. Select fewer buckets or split the daily store.", "context_too_large", 413)
        return [json.loads(row[1]) for row in rows]
