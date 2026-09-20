import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fuego.map_store import MapStore, MapStoreError

ROOT = Path(__file__).resolve().parents[1]
X = [1.0] + [0.0] * 383
Y = [0.0, 1.0] + [0.0] * 382


class MapStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MapStore(self.temp.name)

    def test_empty_map(self):
        self.assertEqual(self.store.get_map(), {"article_ids": [], "checksums": [], "vectors": []})
        self.assertEqual(self.store.search(X), [])
        self.assertEqual(self.store.search_many([X, Y]), [[], []])
        self.assertEqual(self.store.search_many([]), [])

    def test_save_normalize_reload_and_replace(self):
        self.store.save(["a", "b"], [[3.0, 4.0] + [0.0]*382, Y])
        data = MapStore(self.temp.name).get_map()
        self.assertEqual(data["article_ids"], ["a", "b"])
        self.assertEqual(data["vectors"][0][:2], [0.6, 0.8])
        self.assertEqual(data["checksums"][0], hashlib.sha256(b"a").hexdigest())
        self.store.save(["c"], [X])
        self.assertEqual(self.store.get_map()["article_ids"], ["c"])
        self.store.save([], [])
        self.assertEqual(self.store.get_map()["vectors"], [])

    def test_add_deduplicates_id_without_replacing(self):
        self.assertTrue(self.store.add("a", X))
        self.assertFalse(MapStore(self.temp.name).add("a", Y))
        self.assertTrue(self.store.add("b", X))
        self.assertEqual(self.store.get_map()["vectors"], [X, X])

    def test_queries_known_scores_and_ties(self):
        diagonal = [1.0, 1.0] + [0.0]*382
        self.store.save(["x", "diagonal", "y", "negative", "tie"], [X, diagonal, Y, [-v for v in X], Y])
        hits = self.store.search([2*v for v in X], top_k=20)
        self.assertEqual([h["article_id"] for h in hits], ["x", "diagonal", "y", "tie", "negative"])
        for hit, expected in zip(hits, [1, 1/math.sqrt(2), 0, 0, -1]):
            self.assertAlmostEqual(hit["score"], expected)
        self.assertEqual(self.store.search(X, top_k=1), hits[:1])
        batch = self.store.search_many([X, Y], top_k=2)
        self.assertEqual(batch[0], hits[:2])
        self.assertEqual(batch[1], [{"article_id": "y", "score": 1}, {"article_id": "tie", "score": 1}])

    def test_invalid_inputs_leave_file_unchanged(self):
        self.store.save(["a"], [X])
        before = self.store.path.read_bytes()
        bad_vectors = [[], [0]*384, [1]*383, [float("nan")]+[0]*383,
                       [float("inf")]+[0]*383, [True]+[0]*383, ["1"]+[0]*383]
        for vector in bad_vectors:
            with self.subTest(vector=str(vector[:2])):
                for call in (lambda: self.store.save(["b"], [vector]),
                             lambda: self.store.add("b", vector), lambda: self.store.search(vector)):
                    with self.assertRaises(ValueError):
                        call()
        for ids, rows in [(["a", "a"], [X, Y]), (["b"], []), ([" "], [X]), ([1], [X]), ("a", [X])]:
            with self.assertRaises(ValueError):
                self.store.save(ids, rows)
        for k in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                self.store.search(X, k)
        with self.assertRaises(ValueError):
            self.store.search_many("bad")
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_read_results_do_not_change_storage(self):
        self.store.save(["a"], [X])
        snapshot = self.store.get_map()
        snapshot["vectors"][0][0] = 5
        snapshot["article_ids"].clear()
        self.assertEqual(self.store.get_map()["vectors"], [X])

    def test_invalid_file_is_not_silently_replaced(self):
        for data in ("broken", "[]", '{"version":99}',
                     json.dumps({"version": 1, "article_ids": ["a"], "checksums": [], "vectors": [X]}),
                     json.dumps({"version": 1, "article_ids": ["a"], "checksums": ["wrong"], "vectors": [X]}),
                     json.dumps({"version": 1, "article_ids": ["a"],
                                 "checksums": [hashlib.sha256(b"a").hexdigest()], "vectors": [[2*v for v in X]]})):
            self.store.path.write_text(data)
            with self.assertRaises(MapStoreError):
                self.store.get_map()
            with self.assertRaises(MapStoreError):
                self.store.add("b", Y)
            self.assertEqual(self.store.path.read_text(), data)

    def test_failed_write_keeps_previous_map_and_cleans_temp(self):
        self.store.save(["a"], [X])
        with patch("fuego.map_store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(MapStoreError):
                self.store.add("b", Y)
        self.assertEqual(self.store.get_map()["article_ids"], ["a"])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.store.path])

    def test_thousand_rows_and_twenty_queries(self):
        self.store.save([str(i) for i in range(1000)], [X]*1000)
        results = self.store.search_many([X]*20, top_k=3)
        self.assertEqual(len(results), 20)
        self.assertTrue(all(row == [{"article_id": str(i), "score": 1.0} for i in range(3)] for row in results))

    def test_smoke_module_and_direct_script(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_map_store"],
                        [sys.executable, "-B", str(ROOT / "integration/smoke_map_store.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS"), 5)
