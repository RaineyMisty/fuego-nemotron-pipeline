"""Check clustering math and update timing with real map files."""

import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.map_store import MapStore, MapStoreError
from fuego.topic_clusters import TopicClusters, TopicClustersError


def sample_rows():
    return [[x, y]+[0.0]*382 for x, y in [(1, 0), (0.8, 0.6), (-1, 0), (-0.8, -0.6)]]


def main():
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        with tempfile.TemporaryDirectory(prefix="fuego-clusters-") as directory:
            store = MapStore(directory)
            rows = sample_rows()
            store.save(["a", "b", "c", "d"], rows)
            clusters = TopicClusters(MapStore(directory), k=2)
            result = clusters.update()
            labels = result["assignments"]
            check(labels["a"] == labels["b"] and labels["c"] == labels["d"]
                  and labels["a"] != labels["c"], "Wrong assignments.")
            for article_id, expected in [("a", [0.9, 0.3]), ("c", [-0.9, -0.3])]:
                center = result["centers"][labels[article_id]]
                check(all(math.isclose(a, b, abs_tol=1e-12)
                          for a, b in zip(center, expected+[0.0]*382)), "Wrong center.")
            check(math.isclose(result["inertia"], 0.4, abs_tol=1e-12), "Wrong squared error.")
            check(TopicClusters(store, k=2).load() == result, "Reload failed.")
            print("PASS map and K-means: assignments, mean centers, inertia 0.4, reload")
            for i in range(99):
                store.add(f"new-{i}", rows[0])
            check(not clusters.update_if_needed(), "Updated before 100 new rows.")
            check(len(clusters.load()["assignments"]) == 4, "Old snapshot changed.")
            store.add("new-99", rows[0])
            check(clusters.update_if_needed(), "Did not update at 100 new rows.")
            check(len(clusters.load()["assignments"]) == 104, "Missing assignments.")
            print("PASS batch update: wait at 99; rebuild at 100")
            store.add("explicit", rows[1])
            check(len(clusters.update()["assignments"]) == 105, "Explicit update failed.")
            print("PASS explicit update: new article included immediately")
            store.save([], [])
            check(clusters.update_if_needed(), "Removal did not trigger rebuild.")
            check(clusters.load()["centers"] == [] and clusters.load()["assignments"] == {}, "Empty map failed.")
            print("PASS empty map: no centers or assignments")
        return 0
    except (ValueError, OSError, MapStoreError, TopicClustersError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
