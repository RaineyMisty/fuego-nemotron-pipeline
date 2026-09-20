"""Validate incoming articles, process their text, and store results in SQLite."""

import math
from pathlib import Path
import sqlite3

from fuego import article_processing

MAX_ARTICLE_CHARS = 10000
DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "db" / "articles.sqlite3"
TEXT_FIELDS = ("Source_Name", "Tone", "Title", "Article_Link", "People", "Organizations", "Themes")


def prepare_article(record):
    """Return text, metadata, and stored fields. Return None for long articles."""
    if not isinstance(record, dict):
        raise ValueError("The article must be a JSON object.")
    article_id = record.get("Record_ID")
    if not isinstance(article_id, str) or not article_id.strip():
        raise ValueError("Record_ID must be a nonempty string.")
    text = record.get("Article_Text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Article_Text must be nonempty text.")
    if len(text) > MAX_ARTICLE_CHARS:
        return None
    published = record.get("Publication_Date")
    if type(published) is not int or not 0 <= published <= 2**63-1:
        raise ValueError("Publication_Date must be a nonnegative SQLite integer in milliseconds.")
    for field in TEXT_FIELDS:
        if not isinstance(record.get(field, ""), str):
            raise ValueError(f"{field} must be text.")
    metadata = {field: record.get(field, "") for field in TEXT_FIELDS}
    metadata.update(Record_ID=article_id, Publication_Date=published)
    # Validate metadata limits before making a request.
    article_processing.build_messages(text, metadata)
    return {"text": text, "metadata": metadata,
            "fields": {"id": article_id, "published_at": published,
                       "source": metadata["Source_Name"], "title": metadata["Title"],
                       "url": metadata["Article_Link"]}}


class ArticleInput:
    def __init__(self, db_path=DEFAULT_DATABASE, *, client=None):
        self.db_path = Path(db_path)
        self.client = client
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY NOT NULL,
                    published_at INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    semantic_text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS article_buckets (
                    article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    bucket_id TEXT NOT NULL,
                    similarity REAL NOT NULL CHECK(similarity BETWEEN -1 AND 1),
                    PRIMARY KEY (article_id, bucket_id)
                );
            ''')

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return _Connection(connection)

    def get(self, article_id):
        """Return one article with bucket links, or None if it is missing."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["buckets"] = [dict(row) for row in connection.execute(
                "SELECT bucket_id, similarity FROM article_buckets WHERE article_id = ? ORDER BY bucket_id",
                (article_id,))]
            return result

    def ingest(self, record):
        """Process one new article. Return stored, duplicate, or filtered status."""
        prepared = prepare_article(record)
        if prepared is None:
            return {"status": "filtered", "id": record["Record_ID"]}
        fields = prepared["fields"]
        if self.get(fields["id"]) is not None:
            return {"status": "duplicate", "id": fields["id"]}
        result = article_processing.process_article(prepared["text"], prepared["metadata"], client=self.client)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO articles (id, published_at, source, title, url, summary, semantic_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                (fields["id"], fields["published_at"], fields["source"], fields["title"],
                 fields["url"], result["summary"], result["overview"]))
            status = "stored" if cursor.rowcount == 1 else "duplicate"
        return {"status": status, "id": fields["id"]}

    def set_buckets(self, article_id, buckets):
        """Replace bucket links for an existing article in one transaction."""
        if not isinstance(buckets, list):
            raise ValueError("buckets must be a list.")
        rows, seen = [], set()
        for bucket in buckets:
            if not isinstance(bucket, dict) or set(bucket) != {"bucket_id", "similarity"}:
                raise ValueError("Each bucket needs bucket_id and similarity.")
            key, score = bucket["bucket_id"], bucket["similarity"]
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError("Bucket IDs must be nonempty and unique.")
            if type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Similarity must be between -1 and 1.")
            seen.add(key)
            rows.append((article_id, key, score))
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,)).fetchone() is None:
                raise ValueError("The article does not exist.")
            connection.execute("DELETE FROM article_buckets WHERE article_id = ?", (article_id,))
            connection.executemany("INSERT INTO article_buckets VALUES (?, ?, ?)", rows)


class _Connection:
    """Commit or roll back, then close the SQLite connection."""

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection.__enter__()

    def __exit__(self, *args):
        try:
            return self.connection.__exit__(*args)
        finally:
            self.connection.close()
