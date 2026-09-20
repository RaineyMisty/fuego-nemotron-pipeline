from pathlib import Path
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fuego.map_store import MapStore, MapStoreError
from fuego.topic_clusters import TopicClusters, TopicClustersError
from integration.smoke_topic_clusters import sample_rows

ROOT = Path(__file__).resolve().parents[1]


class TopicClustersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MapStore(self.temp.name)
        self.rows = sample_rows()
        self.store.save(["a", "b", "c", "d"], self.rows)
        self.clusters = TopicClusters(self.store, k=2)

    def test_known_solution_and_persistence(self):
        self.assertIsNone(self.clusters.load())
        result = self.clusters.update()
        self.assertEqual(result["assignments"], {"a": 0, "b": 0, "c": 1, "d": 1})
        self.assertEqual(result["centers"], [[0.9, 0.3]+[0.0]*382, [-0.9, -0.3]+[0.0]*382])
        self.assertAlmostEqual(result["inertia"], 0.4)
        self.assertEqual(TopicClusters(self.store, k=2).load(), result)
        self.assertEqual(self.clusters.update(), result)
        for article_id, row in zip(["a", "b", "c", "d"], self.rows):
            distances = [math.dist(row, center) for center in result["centers"]]
            self.assertEqual(result["assignments"][article_id], distances.index(min(distances)))

    def test_centers_are_means_not_unit_vectors(self):
        result = self.clusters.update()
        self.assertLess(math.hypot(*result["centers"][0]), 1)
        one = TopicClusters(self.store, k=1).update()
        self.assertEqual(one["centers"], [[0.0]*384])
        self.assertAlmostEqual(one["inertia"], 4)

    def test_empty_small_and_duplicate_maps(self):
        for ids, rows, count in [([], [], 0), (["x"], [self.rows[0]], 1),
                                 (["x", "y"], [self.rows[0]]*2, 1)]:
            self.store.save(ids, rows)
            result = TopicClusters(self.store, k=20).update()
            self.assertEqual(len(result["centers"]), count)
            self.assertEqual(set(result["assignments"]), set(ids))
            self.assertEqual(result["inertia"], 0)

    def test_initial_and_hundred_new_articles(self):
        self.assertTrue(self.clusters.update_if_needed())
        before = self.clusters.path.read_bytes()
        ids = ["a", "b", "c", "d"]+[str(i) for i in range(99)]
        self.store.save(ids, self.rows+[self.rows[0]]*99)
        self.assertFalse(self.clusters.update_if_needed())
        self.assertEqual(self.clusters.path.read_bytes(), before)
        self.assertFalse(self.store.add("a", self.rows[0]))
        self.assertFalse(self.clusters.update_if_needed())
        self.store.add("last", self.rows[0])
        self.assertTrue(TopicClusters(self.store, k=2).update_if_needed())
        self.assertEqual(len(self.clusters.load()["assignments"]), 104)
        self.assertFalse(self.clusters.update_if_needed())

    def test_explicit_update_below_threshold(self):
        self.clusters.update()
        self.store.add("new", self.rows[0])
        self.assertIn("new", self.clusters.update()["assignments"])

    def test_changes_removals_and_k_changes_rebuild(self):
        self.clusters.update()
        self.store.save(["a", "b", "c", "d"], self.rows[::-1])
        self.assertTrue(self.clusters.update_if_needed())
        self.store.save(["a"], [self.rows[0]])
        self.assertTrue(self.clusters.update_if_needed())
        self.assertTrue(TopicClusters(self.store, k=1).update_if_needed())

    def test_invalid_settings(self):
        for value in (0, -1, True, 2.5, "2"):
            for field in ("k", "max_iterations"):
                with self.assertRaises(ValueError):
                    TopicClusters(self.store, **{field: value})

    def test_nonconvergence_preserves_result(self):
        self.clusters.update()
        before = self.clusters.path.read_bytes()
        with self.assertRaises(TopicClustersError):
            TopicClusters(self.store, k=2, max_iterations=1).update()
        self.assertEqual(self.clusters.path.read_bytes(), before)

    def test_corrupt_files(self):
        self.clusters.update()
        for text in ("broken", "null", '{"data":{},"checksum":"bad"}'):
            self.clusters.path.write_text(text)
            with self.assertRaises(TopicClustersError):
                self.clusters.load()
        self.clusters.update()
        self.assertIsNotNone(self.clusters.load())
        self.store.path.write_text("broken")
        with self.assertRaises(MapStoreError):
            self.clusters.update()

    def test_write_failure_keeps_old_result(self):
        self.clusters.update()
        before = self.clusters.path.read_bytes()
        with patch("fuego.topic_clusters.os.replace", side_effect=OSError("full")):
            with self.assertRaises(TopicClustersError):
                self.clusters.update()
        self.assertEqual(self.clusters.path.read_bytes(), before)
        self.assertEqual(list(self.clusters.path.parent.iterdir()), [self.clusters.path])

    def test_does_not_modify_map(self):
        before = self.store.path.read_bytes()
        self.clusters.update()
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_smoke_module_and_script(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_topic_clusters"],
                        [sys.executable, "-B", str(ROOT / "integration/smoke_topic_clusters.py")]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("PASS"), 4)
