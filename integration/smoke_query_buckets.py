"""Check topic queries with real map storage and known sample vectors."""

import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.map_store import MapStore, MapStoreError
from fuego.query_buckets import QueryBuckets


class SampleEmbedder:
    """Use known vectors to test map math, not language understanding."""

    def embed(self, text):
        index = {"solar power": 0, "sports": 1, "food": 2}[text]
        return [float(i == index) for i in range(384)]


def main():
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        with tempfile.TemporaryDirectory(prefix="fuego-query-") as directory:
            embedder = SampleEmbedder()
            x, y = embedder.embed("solar power"), embedder.embed("sports")
            store = MapStore(directory)
            store.save(["exact", "near", "weak", "other", "opposite"],
                       [x, [0.8, 0.6]+[0.0]*382, [0.3, math.sqrt(0.91)]+[0.0]*382,
                        y, [-v for v in x]])
            queries = QueryBuckets(MapStore(directory), embedder=embedder)
            hits = store.search(x, top_k=10)
            check(all(math.isclose(hit["score"], expected, abs_tol=1e-12)
                      for hit, expected in zip(hits, [1, 0.8, 0.3, 0, -1])), "Wrong scores.")
            check(queries.query("solar power") == ["exact", "near"], "Wrong filtered results.")
            check(queries.query("solar power", min_score=0.8) == ["exact", "near"], "Wrong threshold boundary.")
            print("PASS map search: cosine scores, ranking, and threshold")
            check(queries.query("sports") == ["other", "weak", "near"], "Wrong second topic.")
            check(queries.query("food") == [], "Unrelated articles were included.")
            print("PASS topics: different queries and no matching articles")
            store.save([str(i) for i in range(12)], [x]*12)
            check(queries.query("solar power") == [str(i) for i in range(10)], "Wrong top ten.")
            print("PASS top ten: stable order and ten IDs")
            store.save([], [])
            check(queries.query("solar power") == [], "Empty map failed.")
            print("PASS empty map: no article IDs")
        return 0
    except (ValueError, OSError, MapStoreError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
