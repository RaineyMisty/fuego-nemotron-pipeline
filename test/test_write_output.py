from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from fuego.write_output import build_output, serialize_output
from integration.smoke_write_output import seed_database

ROOT = Path(__file__).resolve().parents[1]


class WriteOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"articles.sqlite3"
        self.buckets = seed_database(self.path)

    def build(self, **kwargs):
        return build_output(self.buckets, db_path=self.path, **kwargs)

    def test_matches_example_fields(self):
        result = self.build()
        example = json.loads((ROOT/"output/fuego_output_example.json").read_text())
        self.assertEqual(set(result), set(example))
        self.assertEqual(set(result["buckets"][0]), set(example["buckets"][0]))
        self.assertEqual(set(result["articles"][0]), set(example["articles"][0]))
        self.assertEqual(set(result["buckets"][0]["activity"]), set(example["buckets"][0]["activity"]))
        self.assertEqual(result["schema_version"], "fuego-response.v1")
        self.assertEqual(json.loads(serialize_output(result)), result)

    def test_db_fields_members_and_deduplication(self):
        result = self.build()
        self.assertEqual([a["id"] for a in result["articles"]], ["a", "b"])
        self.assertEqual(result["articles"][0]["source"], "Sample News")
        self.assertEqual(len(result["articles"][0]["buckets"]), 2)
        self.assertEqual(result["buckets"][0]["members"], [
            {"article_id": "a", "similarity": 0.9, "rank": 1},
            {"article_id": "b", "similarity": 0.7, "rank": 2}])
        self.assertEqual(result["buckets"][0]["name"], self.buckets[0]["synthesis"]["title"])

    def test_limits_counts_and_unknown_bucket(self):
        result = self.build(articles_per_bucket=1)
        self.assertEqual(len(result["articles"]), 1)
        self.assertEqual(result["buckets"][0]["activity"]["current_count"], 2)
        self.assertEqual(result["stats"]["total_articles_considered"], 2)
        self.buckets[0]["id"] = "unknown"
        self.assertEqual(self.build()["buckets"][0]["members"], [])

    def test_missing_optional_data_is_explicit(self):
        result = self.build()
        self.assertEqual(result["articles"][0]["embedding"], [])
        self.assertEqual(result["articles"][0]["metadata"], {})
        activity = result["buckets"][0]["activity"]
        self.assertIsNone(activity["previous_count"])
        self.assertIsNone(activity["change_ratio"])
        self.assertEqual(activity["status"], "insufficient_data")

    def test_optional_data_and_unicode_are_copied(self):
        metadata = {"a": {"source_note": "太阳能"}}
        vectors = {"a": [1.0]+[0.0]*383}
        result = self.build(metadata=metadata, embeddings=vectors)
        result["articles"][0]["metadata"]["source_note"] = "changed"
        result["articles"][0]["embedding"][0] = 5
        self.assertEqual(metadata["a"]["source_note"], "太阳能")
        self.assertEqual(vectors["a"][0], 1)
        self.assertIn("太阳能", serialize_output(self.build(metadata=metadata)))

    def test_utc_dates_and_milliseconds(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE articles SET published_at = 1234 WHERE id = 'a'")
        generated = datetime(2026, 9, 20, 1, tzinfo=timezone(timedelta(hours=1)))
        result = self.build(generated_at=generated)
        self.assertEqual(result["generated_at"], "2026-09-20T00:00:00Z")
        self.assertEqual(result["articles"][0]["published_at"], "1970-01-01T00:00:01.234000Z")

    def test_invalid_input(self):
        for kwargs in ({"articles_per_bucket": 0}, {"articles_per_bucket": True},
                       {"request_type": "bad"}, {"generated_at": datetime(2026, 1, 1)},
                       {"embeddings": {"a": [1, 2]}}, {"metadata": {"a": []}},
                       {"metadata": {"a": {"x": float("nan")}}}):
            with self.assertRaises(ValueError):
                self.build(**kwargs)
        for buckets in (None, [{}], self.buckets*2, [{"id": "x", "synthesis": {"title": "", "summary": "x"}}]):
            with self.assertRaises(ValueError):
                build_output(buckets, db_path=self.path)
        with self.assertRaises(ValueError):
            serialize_output({"x": float("nan")})

    def test_empty_request_and_types(self):
        self.assertEqual(build_output([], db_path=self.path)["articles"], [])
        for kind in ("fixed", "query", "cluster"):
            result = self.build(request_type=kind)
            self.assertEqual(result["buckets"][0]["type"], kind)

    def test_read_only_and_missing_database(self):
        before = self.path.read_bytes()
        self.build()
        self.assertEqual(self.path.read_bytes(), before)
        missing = Path(self.temp.name)/"missing.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            build_output(self.buckets, db_path=missing)
        self.assertFalse(missing.exists())

    def test_dangling_reference_fails(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM articles WHERE id='a'")
        with self.assertRaises(ValueError):
            self.build()

    def test_ties_are_sorted_by_article_id(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE article_buckets SET similarity=0.9 WHERE bucket_id='energy'")
        self.assertEqual([m["article_id"] for m in self.build()["buckets"][0]["members"]], ["a", "b"])

    def test_smoke_module_and_script(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_write_output"],
                        [sys.executable, "-B", str(ROOT/"integration/smoke_write_output.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["stats"]["returned_buckets"], 2)
            self.assertIn("PASS", result.stderr)
