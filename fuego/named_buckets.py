"""Find articles for fixed news topics with cached MiniLM vectors."""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

from fuego.bucket_definitions import BUCKET_DEFINITIONS
from fuego.embedding import DIMENSIONS, MODEL_NAME, Embedder
from fuego.map_store import MapStore


class NamedBucketsError(RuntimeError):
    """Bucket vectors could not be validated or cached."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _valid_vectors(rows):
    if not isinstance(rows, list) or len(rows) != len(BUCKET_DEFINITIONS):
        return False
    for row in rows:
        if not isinstance(row, list) or len(row) != DIMENSIONS:
            return False
        if any(isinstance(x, bool) or not isinstance(x, Real) for x in row):
            return False
        try:
            if not math.isclose(math.hypot(*row), 1, abs_tol=1e-6):
                return False
        except (OverflowError, ValueError):
            return False
    return True


class NamedBuckets:
    def __init__(self, map_store=None, *, embedder=None, cache_directory=None):
        self.map_store = map_store if map_store is not None else MapStore()
        self.embedder = embedder if embedder is not None else Embedder()
        directory = (Path(cache_directory) if cache_directory is not None
                     else self.map_store.path.parent / "buckets")
        self.cache_path = directory / "vectors.json"
        self._key = _digest({"version": 1, "model": MODEL_NAME, "dimensions": DIMENSIONS,
                             "backend": "fastembed-onnxruntime", "encoding": "prototype-text-v1",
                             "definitions": BUCKET_DEFINITIONS})
        self._vectors = None

    def definitions(self):
        """Return a fresh list of IDs, names, and English prototype texts."""
        return [{"id": key, "name": name, "prototype_text": text}
                for key, name, text in BUCKET_DEFINITIONS]

    def load_vectors(self, *, rebuild=False):
        """Load valid cached vectors or build them with the article embedder."""
        if self._vectors is None or rebuild:
            rows = None
            if not rebuild:
                try:
                    cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    candidate = cached["vectors"]
                    if (cached["key"] == self._key and _valid_vectors(candidate)
                            and cached["checksum"] == _digest(candidate)):
                        rows = candidate
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if rows is None:
                rows = self.embedder.embed_many([text for _, _, text in BUCKET_DEFINITIONS])
                if not _valid_vectors(rows):
                    raise NamedBucketsError("Expected 20 unit vectors with 384 values each.")
                payload = {"key": self._key, "vectors": rows, "checksum": _digest(rows)}
                temporary = None
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                                     dir=self.cache_path.parent) as handle:
                        temporary = handle.name
                        json.dump(payload, handle, allow_nan=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.cache_path)
                except OSError as exc:
                    raise NamedBucketsError("Cannot write the bucket cache.") from exc
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok=True)
            self._vectors = [row[:] for row in rows]
        return [row[:] for row in self._vectors]

    def query(self, *, min_score=0.4):
        """Return up to ten article IDs per topic, in descending score order."""
        if (isinstance(min_score, bool) or not isinstance(min_score, Real)
                or not math.isfinite(min_score) or not -1 <= min_score <= 1):
            raise ValueError("min_score must be a finite number between -1 and 1.")
        if not self.map_store.get_map()["article_ids"]:
            return {key: [] for key, _, _ in BUCKET_DEFINITIONS}
        batches = self.map_store.search_many(self.load_vectors(), top_k=10)
        return {key: [hit["article_id"] for hit in hits if hit["score"] >= min_score]
                for (key, _, _), hits in zip(BUCKET_DEFINITIONS, batches)}
