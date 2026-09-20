import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from fuego.ai import AIError
from fuego.pipeline import MockAI, Pipeline, PipelineConfig
from fuego.__main__ import main, write_json


class TinyEmbedder:
    def embed(self, text):
        index = 0 if any(w in text.lower() for w in ('solar', 'energy', 'battery')) else 1
        return [float(i == index) for i in range(384)]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


def record(key="a"):
    return {"Record_ID": key, "Publication_Date": "1789256700000", "Source_Name": "Sample",
            "Title": "Solar energy battery project", "Themes": None,
            "Article_Text": "A city approved solar energy and battery storage for schools, hospitals, parks and libraries. Engineers will review safety, costs, funding, construction, employment, reliability, equipment, capacity, permits, contracts and schedules."}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = PipelineConfig(Path(self.temp.name), mock_ai=True, mock_query="solar", attempts=4)
        self.pipeline = Pipeline(self.config, embedder=TinyEmbedder())

    def process(self, records=None):
        self.pipeline.enqueue(records or [record()])
        return self.pipeline.run_pending()

    def test_offline_then_independent_requests(self):
        self.process()
        for result in (self.pipeline.feed("fixed"), self.pipeline.feed("cluster"), self.pipeline.query()):
            self.assertEqual(result["schema_version"], "fuego-response.v1")
            self.assertEqual(set(result["parameters"]), {"bucket_count", "articles_per_bucket"})
            self.assertTrue(result["articles"])
            self.assertNotIn("embedding", result["articles"][0])
            self.assertEqual(result["articles"][0]["metadata"]["publication_date_raw"], 1789256700000)
            self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(len(self.pipeline.feed("fixed")["buckets"]), 20)

    def test_cached_legacy_feed_is_formatted_without_refresh(self):
        self.process()
        old = self.pipeline.feed("fixed")
        old["parameters"]["mock_ai"] = True
        old["stats"]["indexed_articles"] = 1
        old["articles"][0]["embedding"] = [0.0] * 384
        for bucket in old["buckets"]:
            bucket["id"] = bucket["id"].removeprefix("bucket-")
            bucket["summary_status"] = "ready"
        for link in old["articles"][0]["buckets"]:
            link["bucket_id"] = link["bucket_id"].removeprefix("bucket-")
        with sqlite3.connect(self.pipeline.store.db_path) as db:
            db.execute("UPDATE prepared SET response=? WHERE kind='fixed'", (json.dumps(old),))
        with patch.object(self.pipeline.client, "complete", side_effect=AssertionError("No AI call")):
            feed = self.pipeline.feed("fixed")
        self.assertNotIn("embedding", feed["articles"][0])
        self.assertNotIn("mock_ai", feed["parameters"])
        self.assertTrue(all(b["id"].startswith("bucket-") for b in feed["buckets"]))
        self.assertTrue(all(link["bucket_id"].startswith("bucket-") for link in feed["articles"][0]["buckets"]))

    def test_input_durable_and_success_clears_raw(self):
        self.pipeline.enqueue([record()])
        with sqlite3.connect(self.pipeline.store.db_path) as db:
            self.assertIn('Article_Text', db.execute('SELECT payload FROM jobs').fetchone()[0])
        restarted = Pipeline(self.config, embedder=TinyEmbedder())
        restarted.run_pending()
        with sqlite3.connect(self.pipeline.store.db_path) as db:
            self.assertIsNone(db.execute('SELECT payload FROM jobs').fetchone()[0])
        self.assertEqual(restarted.status()["jobs"][0]["status"], "done")

    def test_duplicate_input_never_calls_ai_again(self):
        self.process()
        self.assertEqual(self.pipeline.enqueue([record()])["duplicates"], 1)
        with patch.object(self.pipeline.client, "complete", side_effect=AssertionError("duplicate")):
            self.pipeline.run_pending()
        self.assertEqual(self.pipeline.status()["articles"], 1)

    def test_retry_api_and_invalid_outputs(self):
        original = self.pipeline.client.complete
        calls = []
        def fail_once(messages):
            calls.append(1)
            if len(calls) == 1:
                return {"choices": [{"finish_reason": "stop", "message": {"content": '{"keywords":[],"summary":"short"}'}}]}
            return original(messages)
        with patch.object(self.pipeline.client, "complete", side_effect=fail_once):
            self.pipeline.enqueue([record()])
            self.pipeline.run_pending(refresh=False)
        self.assertEqual(self.pipeline.status()["jobs"][0]["attempts"], 2)

    def test_failed_job_keeps_input_and_can_retry(self):
        with patch.object(self.pipeline.client, "complete", side_effect=AIError("offline")):
            self.pipeline.enqueue([record()])
            self.pipeline.run_pending(refresh=False)
        self.assertEqual(self.pipeline.status()["jobs"][0]["status"], "failed")
        with sqlite3.connect(self.pipeline.store.db_path) as db:
            self.assertIsNotNone(db.execute('SELECT payload FROM jobs').fetchone()[0])
        self.assertEqual(self.pipeline.retry_failed(), 1)
        self.pipeline.run_pending(refresh=False)
        self.assertEqual(self.pipeline.status()["jobs"][0]["status"], "done")

    def test_embedding_failure_does_not_repeat_article_ai(self):
        with patch.object(self.pipeline.embedder, "embed", side_effect=RuntimeError("model")):
            self.pipeline.enqueue([record()])
            self.pipeline.run_pending(refresh=False)
        self.assertIsNotNone(self.pipeline.store.get("a"))
        self.pipeline.retry_failed()
        with patch.object(self.pipeline.client, "complete", side_effect=AssertionError("must not reprocess")):
            self.pipeline.run_pending(refresh=False)
        self.assertEqual(self.pipeline.map.get_map()["article_ids"], ["a"])

    def test_long_article_filtered_without_ai(self):
        with patch.object(self.pipeline.client, "complete", side_effect=AssertionError("filtered")):
            self.pipeline.enqueue([dict(record(), Article_Text="x"*10001)])
            self.pipeline.run_pending(refresh=False)
        self.assertEqual(self.pipeline.status()["jobs"][0]["status"], "filtered")
        self.assertEqual(self.pipeline.status()["articles"], 0)

    def test_requests_do_not_change_permanent_bucket_links(self):
        self.process()
        before = self.pipeline.store.get("a")["buckets"]
        self.pipeline.query("solar")
        self.pipeline.query("sports")
        self.assertEqual(self.pipeline.store.get("a")["buckets"], before)

    def test_summary_failure_is_marked_not_fabricated(self):
        self.pipeline.enqueue([record()])
        self.pipeline.run_pending(refresh=False)
        with patch("fuego.pipeline.synthesize_topic", side_effect=AIError("offline")):
            response = self.pipeline.query("solar")
        self.assertNotIn("summary_status", response["buckets"][0])
        self.assertIn("unavailable", response["buckets"][0]["summary"])
        self.assertTrue(response["articles"])
        self.assertGreater(len(response["warnings"]), 1)

    def test_empty_map_and_missing_feed(self):
        with self.assertRaises(LookupError):
            self.pipeline.feed("fixed")
        response = self.pipeline.query("solar")
        self.assertEqual(response["articles"], [])
        self.assertNotIn("summary_status", response["buckets"][0])
        self.pipeline.refresh()
        self.assertEqual(self.pipeline.feed("cluster")["articles"], [])

    def test_state_mode_cannot_mix(self):
        with self.assertRaises(ValueError):
            Pipeline(PipelineConfig(Path(self.temp.name), mock_ai=False), embedder=TinyEmbedder())

    def test_clusters_wait_for_batch(self):
        self.process()
        self.process([record("b")])
        self.assertEqual(len(self.pipeline.feed("cluster")["articles"]), 1)
        self.pipeline.refresh(force_clusters=True)
        self.assertEqual(len(self.pipeline.feed("cluster")["articles"]), 2)

    def test_invalid_inputs_and_config(self):
        for data in ({}, [None], [{"Record_ID": ""}]):
            with self.assertRaises(ValueError):
                self.pipeline.enqueue(data)
        for kwargs in ({"min_score": float('nan')}, {"attempts": 0}, {"mock_ai": "yes"}):
            with self.assertRaises(ValueError):
                PipelineConfig(Path(self.temp.name), **kwargs)
        with self.assertRaises(ValueError):
            self.pipeline.query("")

    def test_refresh_crash_can_be_recovered_after_restart(self):
        self.pipeline.enqueue([record()])
        self.pipeline.run_pending(refresh=False)
        self.assertTrue(self.pipeline.status()["feeds_stale"])
        restarted = Pipeline(self.config, embedder=TinyEmbedder())
        self.assertTrue(restarted.status()["feeds_stale"])
        restarted.refresh()
        self.assertFalse(restarted.status()["feeds_stale"])

    def test_atomic_export(self):
        path = Path(self.temp.name)/"response.json"
        write_json(path, {"test": "太阳能"})
        self.assertEqual(json.loads(path.read_text()), {"test": "太阳能"})
        self.assertEqual(list(path.parent.glob('tmp*')), [])
