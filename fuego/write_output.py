"""Package stored articles and topic synthesis results for the backend."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

from fuego.article_input import DEFAULT_DATABASE

SCHEMA_VERSION = "fuego-response.v1"


def _text(value, name, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be text.")
    return value


def _date(milliseconds):
    if type(milliseconds) is not int or milliseconds < 0:
        raise ValueError("published_at must be a nonnegative integer in milliseconds.")
    try:
        return (datetime(1970, 1, 1, tzinfo=timezone.utc)+timedelta(milliseconds=milliseconds)).isoformat().replace("+00:00", "Z")
    except OverflowError:
        raise ValueError("published_at is outside the supported date range.") from None


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Similarity must be a finite number between -1 and 1.")
    return value


def build_output(buckets, *, db_path=DEFAULT_DATABASE, request_type="fixed",
                 articles_per_bucket=10, generated_at=None, embeddings=None, metadata=None):
    """Return the response dict. Bucket specs contain id, synthesis, and description."""
    if request_type not in ("fixed", "query", "cluster"):
        raise ValueError("request_type must be fixed, query, or cluster.")
    if type(articles_per_bucket) is not int or articles_per_bucket <= 0:
        raise ValueError("articles_per_bucket must be a positive integer.")
    if not isinstance(buckets, list):
        raise ValueError("buckets must be a list.")
    specs, seen = [], set()
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise ValueError("Each bucket must be an object.")
        key = _text(bucket.get("id"), "Bucket ID")
        if key in seen:
            raise ValueError("Bucket IDs must be unique.")
        seen.add(key)
        synthesis = bucket.get("synthesis")
        if not isinstance(synthesis, dict) or set(synthesis) != {"title", "summary"}:
            raise ValueError("Each bucket needs a synthesis title and summary.")
        title = _text(synthesis["title"], "Synthesis title")
        summary = _text(synthesis["summary"], "Synthesis summary")
        if len(title) > 120 or len(summary) > 4000:
            raise ValueError("Synthesis output exceeds its text limits.")
        description = _text(bucket.get("description", ""), "Description", allow_empty=True)
        specs.append((key, title, summary, description))
    now = datetime.now(timezone.utc) if generated_at is None else generated_at
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("generated_at must be a timezone-aware datetime.")
    extras = []
    for value in (embeddings, metadata):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("Article extras must be dictionaries keyed by article ID.")
        try:
            extras.append(json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)))
        except (TypeError, ValueError, RecursionError):
            raise ValueError("Article extras must contain valid JSON data.") from None
    vectors, article_metadata = extras
    for vector in vectors.values():
        if (not isinstance(vector, list) or len(vector) != 384
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError("Each supplied embedding must have 384 finite numbers.")
    if any(not isinstance(value, dict) for value in article_metadata.values()):
        raise ValueError("Each article metadata value must be an object.")
    output_buckets, output_articles, included = [], [], set()
    # Read one consistent snapshot without creating or changing the database.
    uri = Path(db_path).resolve().as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        total = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        for key, title, summary, description in specs:
            members = connection.execute(
                "SELECT article_id, similarity FROM article_buckets WHERE bucket_id = ? "
                "ORDER BY similarity DESC, article_id ASC", (key,)).fetchall()
            ranked = []
            for rank, member in enumerate(members[:articles_per_bucket], 1):
                article_id = member["article_id"]
                ranked.append({"article_id": article_id, "similarity": _score(member["similarity"]), "rank": rank})
                if article_id in included:
                    continue
                row = connection.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
                if row is None:
                    raise ValueError("A bucket references a missing article.")
                article = {field: _text(row[field], field, allow_empty=True)
                           for field in ("id", "title", "url", "source", "summary", "semantic_text")}
                article["published_at"] = _date(row["published_at"])
                article["embedding"] = vectors.get(article_id, [])
                article["metadata"] = article_metadata.get(article_id, {})
                article["buckets"] = [{"bucket_id": link["bucket_id"], "similarity": _score(link["similarity"])}
                                      for link in connection.execute(
                                          "SELECT bucket_id, similarity FROM article_buckets WHERE article_id = ? ORDER BY bucket_id",
                                          (article_id,))]
                output_articles.append(article)
                included.add(article_id)
            output_buckets.append({"id": key, "type": request_type, "name": title,
                                   "description": description, "summary": summary, "members": ranked,
                                   "activity": {"current_count": len(members), "previous_count": None,
                                                "change_ratio": None, "status": "insufficient_data"},
                                   "direction": {"description": "Insufficient temporal data.", "related_topics": []}})
    return {"schema_version": SCHEMA_VERSION, "request_type": request_type,
            "generated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "parameters": {"bucket_count": len(specs), "articles_per_bucket": articles_per_bucket},
            "stats": {"total_articles_considered": total, "returned_buckets": len(output_buckets)},
            "buckets": output_buckets, "articles": output_articles}


def serialize_output(response, *, indent=2):
    """Serialize a built response as strict JSON. Do not write or send it."""
    return json.dumps(response, ensure_ascii=False, allow_nan=False, indent=indent)
