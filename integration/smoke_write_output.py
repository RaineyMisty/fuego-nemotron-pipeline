"""Check JSON output with a small real SQLite fixture. No AI call is made."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.article_input import ArticleInput
from fuego.write_output import build_output, serialize_output


def seed_database(path):
    store = ArticleInput(path)
    with sqlite3.connect(path) as connection:
        connection.executemany("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("a", 0, "Sample News", "Solar project", "https://example.com/a", "A city adds solar panels.", "solar; city"),
            ("b", 1000, "Sample News", "Battery project", "", "The city adds battery storage.", "battery; storage"),
        ])
    store.set_buckets("a", [{"bucket_id": "energy", "similarity": 0.9}, {"bucket_id": "city", "similarity": 0.8}])
    store.set_buckets("b", [{"bucket_id": "energy", "similarity": 0.7}])
    return [
        {"id": "energy", "description": "Clean energy projects.",
         "synthesis": {"title": "City clean energy", "summary": "The city adds solar panels and batteries."}},
        {"id": "city", "synthesis": {"title": "City solar project", "summary": "The city adds solar panels."}},
    ]


def main():
    try:
        with tempfile.TemporaryDirectory(prefix="fuego-output-") as directory:
            path = Path(directory)/"articles.sqlite3"
            buckets = seed_database(path)
            result = build_output(buckets, db_path=path, generated_at=datetime(2026, 9, 20, tzinfo=timezone.utc))
            encoded = serialize_output(result)
            if json.loads(encoded) != result:
                raise ValueError("JSON round trip failed.")
            if result["stats"] != {"total_articles_considered": 2, "returned_buckets": 2}:
                raise ValueError("Wrong stats.")
            expected = [{"article_id": "a", "similarity": 0.9, "rank": 1},
                        {"article_id": "b", "similarity": 0.7, "rank": 2}]
            if result["buckets"][0]["members"] != expected:
                raise ValueError("Wrong members or scores.")
            if [a["id"] for a in result["articles"]] != ["a", "b"]:
                raise ValueError("Missing or duplicate articles.")
            if result["articles"][0]["published_at"] != "1970-01-01T00:00:00Z":
                raise ValueError("Wrong timestamp conversion.")
            if result["buckets"][0]["name"] != buckets[0]["synthesis"]["title"]:
                raise ValueError("Synthesis title was not used.")
            print(encoded)
            print("PASS: SQLite data, synthesis, ranks, deduplication, dates, and JSON round trip", file=sys.stderr)
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
