import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fuego.bucket_definitions import BUCKET_DEFINITIONS
from fuego.map_store import MapStore
from fuego.named_buckets import NamedBuckets, NamedBucketsError
from integration.smoke_named_buckets import axis

ROOT = Path(__file__).resolve().parents[1]


class NamedBucketsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MapStore(self.temp.name)
        self.embedder = Mock()
        self.embedder.embed_many.return_value = [axis(i) for i in range(20)]
        self.buckets = NamedBuckets(self.store, embedder=self.embedder)

    def test_definitions(self):
        definitions = self.buckets.definitions()
        self.assertEqual(len(definitions), 20)
        self.assertEqual(len({d["id"] for d in definitions}), 20)
        self.assertTrue(all(d["name"] and d["prototype_text"] for d in definitions))
        definitions[0]["id"] = "changed"
        self.assertEqual(self.buckets.definitions()[0]["id"], "politics")

    def test_build_and_reuse_cache(self):
        rows = self.buckets.load_vectors()
        self.embedder.embed_many.assert_called_once_with([d[2] for d in BUCKET_DEFINITIONS])
        other = NamedBuckets(self.store, embedder=Mock())
        self.assertEqual(other.load_vectors(), rows)
        other.embedder.embed_many.assert_not_called()
        rows[0][0] = 9
        self.assertEqual(self.buckets.load_vectors()[0][0], 1)

    def test_rebuild(self):
        self.buckets.load_vectors()
        self.buckets.load_vectors(rebuild=True)
        self.assertEqual(self.embedder.embed_many.call_count, 2)

    def test_bad_or_stale_cache_rebuilds(self):
        self.buckets.load_vectors()
        valid = json.loads(self.buckets.cache_path.read_text())
        bad_vectors = dict(valid, vectors=[[0]*384]*20)
        tampered = dict(valid, vectors=[axis(21)]*20)
        for payload in ("broken", "null", json.dumps(dict(valid, key="old")),
                        json.dumps(bad_vectors), json.dumps(tampered)):
            self.buckets.cache_path.write_text(payload)
            fresh = NamedBuckets(self.store, embedder=self.embedder)
            self.assertEqual(fresh.load_vectors(), [axis(i) for i in range(20)])
        self.assertEqual(self.embedder.embed_many.call_count, 6)

    def test_changed_definitions_invalidate_cache(self):
        self.buckets.load_vectors()
        changed = (("politics", "Politics", "new prototype"),) + BUCKET_DEFINITIONS[1:]
        with patch("fuego.named_buckets.BUCKET_DEFINITIONS", changed):
            NamedBuckets(self.store, embedder=self.embedder).load_vectors()
        self.assertEqual(self.embedder.embed_many.call_count, 2)

    def test_invalid_embedding_output(self):
        for rows in ([], [[0]*384]*20, [[float("nan")]+[0]*383]*20,
                     [[True]+[0]*383]*20, [[1]*383]*20):
            self.embedder.embed_many.return_value = rows
            with self.assertRaises(NamedBucketsError):
                self.buckets.load_vectors()
        self.assertFalse(self.buckets.cache_path.exists())

    def test_cache_write_failure(self):
        with patch("fuego.named_buckets.os.replace", side_effect=OSError("full disk")):
            with self.assertRaises(NamedBucketsError):
                self.buckets.load_vectors()
        self.assertEqual(list(self.buckets.cache_path.parent.iterdir()), [])

    def test_empty_map_does_not_load_model(self):
        result = self.buckets.query()
        self.assertEqual(len(result), 20)
        self.assertTrue(all(not ids for ids in result.values()))
        self.embedder.embed_many.assert_not_called()

    def test_real_map_batch_threshold_and_order(self):
        self.store.save(["a", "b", "c"], [axis(0), [0.8, 0.6]+[0]*382, axis(1)])
        with patch.object(self.store, "search_many", wraps=self.store.search_many) as search:
            result = self.buckets.query(min_score=0.8)
            search.assert_called_once()
            self.assertEqual(search.call_args.kwargs, {"top_k": 10})
        self.assertEqual(result["politics"], ["a", "b"])
        self.assertEqual(result["world-affairs"], ["c"])
        self.assertEqual(result["economy"], [])
        self.assertEqual(self.buckets.query(min_score=0.6)["world-affairs"], ["c", "b"])

    def test_ten_results_and_map_updates(self):
        self.store.save([str(i) for i in range(12)], [axis(0)]*12)
        self.assertEqual(self.buckets.query()["politics"], [str(i) for i in range(10)])
        self.store.save(["new"], [axis(1)])
        self.assertEqual(self.buckets.query()["world-affairs"], ["new"])
        self.assertEqual(self.buckets.query()["politics"], [])

    def test_invalid_threshold(self):
        for value in (True, None, "0.4", float("nan"), float("inf"), -2, 2):
            with self.assertRaises(ValueError):
                self.buckets.query(min_score=value)

    def test_smoke_module_and_script(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_named_buckets"],
                        [sys.executable, "-B", str(ROOT / "integration/smoke_named_buckets.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS"), 4)
