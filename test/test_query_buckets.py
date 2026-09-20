from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fuego.embedding import EmbeddingError, MAX_TEXT_CHARS
from fuego.map_store import MapStore, MapStoreError
from fuego.query_buckets import QueryBuckets

ROOT = Path(__file__).resolve().parents[1]
X = [1.0]+[0.0]*383
Y = [0.0, 1.0]+[0.0]*382


class QueryBucketsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MapStore(self.temp.name)
        self.embedder = Mock()
        self.embedder.embed.return_value = X
        self.queries = QueryBuckets(self.store, embedder=self.embedder)

    def test_defaults_do_not_load_model(self):
        with patch("fuego.query_buckets.Embedder") as factory:
            queries = QueryBuckets(self.store)
            factory.assert_called_once_with()
            self.assertEqual(queries.query("news"), [])
            factory.return_value.embed.assert_not_called()

    def test_real_map_ranking_filter_and_single_query(self):
        self.store.save(["other", "near", "exact", "negative"],
                        [Y, [0.8, 0.6]+[0.0]*382, X, [-v for v in X]])
        with patch.object(self.store, "search", wraps=self.store.search) as search:
            self.assertEqual(self.queries.query("  solar power  "), ["exact", "near"])
            search.assert_called_once_with(X, top_k=10)
        self.embedder.embed.assert_called_once_with("solar power")
        self.assertEqual(self.queries.query("solar", min_score=0.8), ["exact", "near"])
        self.assertEqual(self.queries.query("solar", min_score=0.81), ["exact"])
        self.assertEqual(self.queries.query("solar", min_score=-1), ["exact", "near", "other", "negative"])

    def test_empty_and_no_matches(self):
        self.assertEqual(self.queries.query("news"), [])
        self.embedder.embed.assert_not_called()
        self.store.save(["other"], [Y])
        self.assertEqual(self.queries.query("news"), [])

    def test_limit_ties_and_updated_map(self):
        self.store.save([str(i) for i in range(12)], [X]*12)
        self.assertEqual(self.queries.query("news"), [str(i) for i in range(10)])
        self.store.save(["new"], [X])
        self.assertEqual(self.queries.query("news"), ["new"])

    def test_query_is_embedded_each_time(self):
        self.store.save(["x", "y"], [X, Y])
        self.embedder.embed.side_effect = [X, Y]
        self.assertEqual(self.queries.query("solar"), ["x"])
        self.assertEqual(self.queries.query("sports"), ["y"])
        self.assertEqual(self.embedder.embed.call_count, 2)

    def test_invalid_topic(self):
        for topic in (None, 12, [], "", " \n", "a"*(MAX_TEXT_CHARS+1)):
            with self.assertRaises(ValueError):
                self.queries.query(topic)
        self.embedder.embed.assert_not_called()

    def test_invalid_threshold(self):
        for value in (True, None, "0.4", float("nan"), float("inf"), -2, 2):
            with self.assertRaises(ValueError):
                self.queries.query("news", min_score=value)
        self.embedder.embed.assert_not_called()

    def test_embedding_and_token_errors_propagate(self):
        self.store.save(["x"], [X])
        for error in (EmbeddingError("missing model"), ValueError("too many tokens")):
            self.embedder.embed.side_effect = error
            with self.assertRaises(type(error)) as caught:
                self.queries.query("news")
            self.assertIs(caught.exception, error)

    def test_bad_map_and_vector_fail(self):
        self.store.path.write_text("broken")
        with self.assertRaises(MapStoreError):
            self.queries.query("news")
        self.embedder.embed.assert_not_called()
        self.store.save(["x"], [X])
        self.embedder.embed.return_value = [0]*384
        with self.assertRaises(ValueError):
            self.queries.query("news")

    def test_query_does_not_write_map(self):
        self.store.save(["x"], [X])
        before = self.store.path.read_bytes()
        self.queries.query("news")
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.store.path])

    def test_smoke_module_and_script(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_query_buckets"],
                        [sys.executable, "-B", str(ROOT / "integration/smoke_query_buckets.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS"), 4)
