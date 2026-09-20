"""Check all five map interfaces with known vectors and real disk storage."""

import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.map_store import MapStore, MapStoreError


def main():
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    x = [1.0] + [0.0] * 383
    y = [0.0, 1.0] + [0.0] * 382
    diagonal = [1.0, 1.0] + [0.0] * 382
    try:
        with tempfile.TemporaryDirectory(prefix="fuego-map-smoke-") as directory:
            store = MapStore(directory)
            store.save(["x", "y"], [x, y])
            check(MapStore(directory).get_map()["vectors"] == [x, y], "Save failed.")
            print("PASS save: two rows survive reload")
            check(store.add("diagonal", diagonal), "Append failed.")
            check(not store.add("diagonal", x), "Duplicate was added.")
            print("PASS add: one new article; duplicate skipped")
            snapshot = store.get_map()
            check(snapshot["article_ids"] == ["x", "y", "diagonal"], "Wrong IDs.")
            check(len(snapshot["checksums"]) == 3, "Wrong hash count.")
            expected = [1 / math.sqrt(2)] * 2 + [0.0] * 382
            check(all(math.isclose(a, b, abs_tol=1e-12)
                      for a, b in zip(snapshot["vectors"][2], expected)), "Wrong matrix.")
            print("PASS get_map: shape 3 x 384; normalized values are correct")
            hits = store.search(x, top_k=3)
            check([hit["article_id"] for hit in hits] == ["x", "diagonal", "y"], "Wrong order.")
            check(all(math.isclose(hit["score"], score, abs_tol=1e-12)
                      for hit, score in zip(hits, [1, 1/math.sqrt(2), 0])), "Wrong scores.")
            print("PASS search: scores 1, 0.707107, 0")
            batch = store.search_many([x, y], top_k=3)
            check(batch[0] == hits, "Wrong first query.")
            check([hit["article_id"] for hit in batch[1]] == ["y", "diagonal", "x"], "Wrong batch order.")
            check(all(math.isclose(hit["score"], score, abs_tol=1e-12)
                      for hit, score in zip(batch[1], [1, 1/math.sqrt(2), 0])), "Wrong batch scores.")
            print("PASS search_many: both query results are correct")
        return 0
    except (MapStoreError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
