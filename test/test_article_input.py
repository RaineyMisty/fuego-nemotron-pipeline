from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fuego.ai import AIConfig, AIError
from fuego.article_input import ArticleInput, prepare_article
from fuego.article_processing import ArticleProcessingError
from integration.smoke_article_input import main

ROOT = Path(__file__).resolve().parents[1]


def record():
    return {"Record_ID": "news-1", "Publication_Date": 1789256700000,
            "Source_Name": "News", "Title": "A title", "Article_Link": "https://example.com/news",
            "Article_Text": "A transit agency announced an electric bus pilot."}


def reply():
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
        "keywords": [f"Topic {i}" for i in range(20)], "summary": "The agency announced a pilot."})}}]}


class ArticleInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = Mock()
        self.client.complete.return_value = reply()
        self.store = ArticleInput(Path(self.temp.name)/"db"/"articles.sqlite3", client=self.client)

    def test_prepare_fields_and_metadata(self):
        original = record()
        prepared = prepare_article(original)
        self.assertEqual(prepared["text"], original["Article_Text"])
        self.assertEqual(prepared["fields"]["published_at"], original["Publication_Date"])
        self.assertEqual(set(prepared["metadata"]), set(original) - {"Article_Text"})
        self.assertNotIn("Article_Text", prepared["metadata"])
        self.assertEqual(original, record())

    def test_removed_fields_are_ignored(self):
        value = dict(record(), Tone=None, People=[], Organizations={}, Themes=False, Theme="old")
        prepared = prepare_article(value)
        self.assertEqual(set(prepared["metadata"]), set(record()) - {"Article_Text"})

    def test_real_processing_and_sqlite_round_trip(self):
        self.assertEqual(self.store.ingest(record()), {"status": "stored", "id": "news-1"})
        self.client.complete.assert_called_once()
        sent = json.loads(self.client.complete.call_args.args[0][1]["content"])
        self.assertEqual(sent["article"], record()["Article_Text"])
        self.assertEqual(sent["metadata"]["Record_ID"], "news-1")
        saved = ArticleInput(self.store.db_path).get("news-1")
        self.assertEqual(saved["summary"], "The agency announced a pilot.")
        self.assertEqual(saved["semantic_text"], "; ".join(f"Topic {i}" for i in range(20)))
        self.assertEqual(saved["url"], record()["Article_Link"])
        self.assertEqual(saved["buckets"], [])
        with sqlite3.connect(self.store.db_path) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(articles)")}
            self.assertEqual(columns, {"id", "published_at", "source", "title", "url", "summary", "semantic_text"})
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_long_filter_boundary_and_duplicate(self):
        self.assertEqual(self.store.ingest(dict(record(), Article_Text="x"*10001))["status"], "filtered")
        self.client.complete.assert_not_called()
        self.assertIsNone(self.store.get("news-1"))
        self.assertEqual(self.store.ingest(dict(record(), Article_Text="x"*10000))["status"], "stored")
        self.assertEqual(self.store.ingest(record())["status"], "duplicate")
        self.client.complete.assert_called_once()

    def test_invalid_input(self):
        bad = [None, [], {}, dict(record(), Record_ID=" "), dict(record(), Article_Text=""),
               dict(record(), Publication_Date=True), dict(record(), Publication_Date=-1),
               dict(record(), Publication_Date=2**63), dict(record(), Source_Name=[]),
               dict(record(), Title="x"*17000)]
        for value in bad:
            with self.assertRaises(ValueError):
                self.store.ingest(value)
        self.client.complete.assert_not_called()

    def test_processing_errors_leave_no_article(self):
        for error in (AIError("network"), ArticleProcessingError("bad response")):
            self.client.complete.side_effect = error
            with self.assertRaises(type(error)):
                self.store.ingest(record())
            self.assertIsNone(self.store.get("news-1"))
        self.client.complete.side_effect = None
        self.client.complete.return_value = {"choices": []}
        with self.assertRaises(ArticleProcessingError):
            self.store.ingest(record())
        self.assertIsNone(self.store.get("news-1"))

    def test_bucket_replace_validation_and_missing_article(self):
        self.store.ingest(record())
        links = [{"bucket_id": "technology", "similarity": 0.78}]
        self.store.set_buckets("news-1", links)
        self.assertEqual(self.store.get("news-1")["buckets"], links)
        for bad in (None, [{}], links*2, [{"bucket_id": "a", "similarity": float("nan")}],
                    [{"bucket_id": "a", "similarity": 2}], [{"bucket_id": "a", "similarity": True}]):
            with self.assertRaises(ValueError):
                self.store.set_buckets("news-1", bad)
            self.assertEqual(self.store.get("news-1")["buckets"], links)
        with self.assertRaises(ValueError):
            self.store.set_buckets("missing", links)
        self.store.set_buckets("news-1", [])
        self.assertEqual(self.store.get("news-1")["buckets"], [])

    def test_parameterized_ids(self):
        value = "id'; DROP TABLE articles; --"
        self.store.ingest(dict(record(), Record_ID=value))
        self.assertEqual(self.store.get(value)["id"], value)
        self.assertIsNone(self.store.get("missing"))

    def test_bucket_failure_rolls_back(self):
        self.store.ingest(record())
        links = [{"bucket_id": "old", "similarity": 0.5}]
        self.store.set_buckets("news-1", links)
        with sqlite3.connect(self.store.db_path) as connection:
            connection.execute("CREATE TRIGGER reject_new BEFORE INSERT ON article_buckets BEGIN SELECT RAISE(ABORT, 'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.set_buckets("news-1", [{"bucket_id": "new", "similarity": 1}])
        self.assertEqual(self.store.get("news-1")["buckets"], links)

    def test_racing_duplicate_keeps_first_result(self):
        def other_writer(messages):
            with sqlite3.connect(self.store.db_path) as connection:
                connection.execute("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   ("news-1", 1, "first", "first", "", "first", "first"))
            return reply()
        self.client.complete.side_effect = other_writer
        self.assertEqual(self.store.ingest(record())["status"], "duplicate")
        self.assertEqual(self.store.get("news-1")["summary"], "first")


class SmokeArticleInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.input = Path(temp.name) / "input.json"
        self.input.write_text(json.dumps(record()))

    def test_supplied_six_field_sample(self):
        rows = json.loads((ROOT / "integration/article_input_sample.json").read_text())
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(record()))
        with patch("integration.smoke_article_input.NemotronClient") as client, redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            self.assertEqual(main([]), 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "filtered")
        client.assert_not_called()


    def test_smoke_real_processing_with_mocked_api(self):
        out, err = io.StringIO(), io.StringIO()
        with patch("integration.smoke_article_input.AIConfig.from_env", return_value=AIConfig(api_key="test-key")), \
             patch("integration.smoke_article_input.NemotronClient") as client, \
             redirect_stdout(out), redirect_stderr(err):
            client.return_value.complete.return_value = reply()
            self.assertEqual(main(["--input", str(self.input)]), 0)
            client.return_value.complete.assert_called_once()
        self.assertEqual(err.getvalue().count("PASS"), 3)
        self.assertTrue(json.loads(out.getvalue())["semantic_text"])
        self.assertNotIn("test-key", out.getvalue()+err.getvalue())

    def test_request_response_and_config_errors(self):
        with patch("integration.smoke_article_input.AIConfig.from_env", return_value=AIConfig(api_key="test-key")), \
             patch("integration.smoke_article_input.NemotronClient") as client, redirect_stderr(io.StringIO()):
            client.return_value.complete.side_effect = AIError("network")
            self.assertEqual(main(["--input", str(self.input)]), 1)
            client.return_value.complete.side_effect = None
            client.return_value.complete.return_value = {"choices": []}
            self.assertEqual(main(["--input", str(self.input)]), 3)
            self.assertEqual(main(["--input", "/missing/input.json"]), 2)
        with patch("integration.smoke_article_input.AIConfig.from_env", side_effect=ValueError()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--input", str(self.input)]), 2)

    def test_module_and_script_help(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_article_input", "--help"],
                        [sys.executable, "-B", str(ROOT/"integration/smoke_article_input.py"), "--help"]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
