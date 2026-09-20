"""Check real map queries with fixed vectors. No model download is needed."""

import math
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.map_store import MapStore, MapStoreError
from fuego.named_buckets import NamedBuckets, NamedBucketsError


def axis(index):
    return [float(i == index) for i in range(384)]


class SampleEmbedder:
    """Use known orthogonal vectors to check math, not model quality."""

    def embed_many(self, texts):
        return [axis(i) for i in range(len(texts))]


def main():
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        with tempfile.TemporaryDirectory(prefix="fuego-buckets-") as directory:
            store = MapStore(directory)
            store.save(["exact", "near", "weak", "opposite", "other"],
                       [axis(0), [0.8, 0.6]+[0.0]*382,
                        [0.3, math.sqrt(0.91)]+[0.0]*382,
                        [-x for x in axis(0)], axis(1)])
            buckets = NamedBuckets(store, embedder=SampleEmbedder())
            results = buckets.query()
            check(results["politics"] == ["exact", "near"], "Wrong score order or threshold.")
            check(results["world-affairs"] == ["other", "weak", "near"], "Wrong second bucket.")
            check(results["economy"] == [], "Unrelated articles were included.")
            scores = store.search_many(buckets.load_vectors(), top_k=10)[0]
            check(all(math.isclose(hit["score"], expected, abs_tol=1e-12)
                      for hit, expected in zip(scores, [1, 0.8, 0.3, 0, -1])), "Wrong cosine scores.")
            print("PASS map queries: scores, order, threshold, and short results")
            check(NamedBuckets(store, embedder=SampleEmbedder()).query() == results, "Cache reload failed.")
            print("PASS cache: same results after reload")
            store.save([str(i) for i in range(12)], [axis(0)]*12)
            check(buckets.query()["politics"] == [str(i) for i in range(10)], "Wrong top ten.")
            print("PASS top ten: stable order and ten IDs")
            store.save([], [])
            check(all(not ids for ids in buckets.query().values()), "Empty map failed.")
            print("PASS empty map: no article IDs")
        return 0
    except (ValueError, OSError, MapStoreError, NamedBucketsError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
