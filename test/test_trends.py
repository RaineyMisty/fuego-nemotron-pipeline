import unittest
from fuego.trend_activity import activity
from fuego.trend_direction import direction


class TrendTests(unittest.TestCase):
    def test_windows_and_change(self):
        result = activity([0, 10, 11, 20, 21], as_of=20, window_ms=10, coverage_start=0)
        self.assertEqual(result, {"current_count": 2, "previous_count": 1, "change_ratio": 1.0, "status": "increasing"})

    def test_no_prior_baseline(self):
        self.assertEqual(activity([15], as_of=20, window_ms=10)["status"], "insufficient_data")
        self.assertEqual(activity([15], as_of=20, window_ms=10, coverage_start=0)["status"], "new")
        self.assertIsNone(activity([15], as_of=20, window_ms=10, coverage_start=0)["change_ratio"])
        self.assertEqual(activity([], as_of=20, window_ms=10, coverage_start=0)["status"], "stable")

    def test_direction_known_shift(self):
        x, y = [1.0]+[0.0]*383, [0.0, 1.0]+[0.0]*382
        result = direction([y], [x], [("new", y), ("old", x)])
        self.assertEqual(result["centroid_cosine_distance"], 1)
        self.assertEqual(result["related_topics"][0]["name"], "new")
        self.assertAlmostEqual(result["related_topics"][0]["similarity"], 2**-0.5)
        self.assertEqual(direction([x], [x], [])["centroid_cosine_distance"], 0)
        self.assertEqual(direction([], [x], [])["related_topics"], [])

    def test_invalid_vectors_and_windows(self):
        with self.assertRaises(ValueError):
            direction([[1]], [[1]], [])
        with self.assertRaises(ValueError):
            activity([], as_of=20, window_ms=0)
