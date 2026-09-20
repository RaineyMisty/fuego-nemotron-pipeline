"""Store article vectors and find nearby articles. Use one writer at a time."""

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

DIMENSIONS = 384
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "map"


class MapStoreError(RuntimeError):
    """The map file could not be read or written."""


def _vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != DIMENSIONS:
        raise ValueError("A vector must contain 384 numbers.")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in values):
        raise ValueError("Vector values must be real numbers.")
    try:
        row = [float(x) for x in values]
        norm = math.hypot(*row)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Vector values must be finite.") from exc
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("A vector must have a finite, nonzero norm.")
    return [x / norm for x in row]


def _checksum(article_id):
    if not isinstance(article_id, str) or not article_id.strip():
        raise ValueError("Article IDs must be nonempty strings.")
    return hashlib.sha256(article_id.encode("utf-8")).hexdigest()


class MapStore:
    """Keep a normalized matrix with separate article IDs and ID hashes."""

    def __init__(self, directory=DEFAULT_DIRECTORY):
        self.path = Path(directory) / "articles.json"

    def _read(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": 1, "article_ids": [], "checksums": [], "vectors": []}
        except (OSError, ValueError) as exc:
            raise MapStoreError("Cannot read the map file.") from exc
        try:
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("Invalid map version.")
            ids, hashes, rows = data["article_ids"], data["checksums"], data["vectors"]
            if not all(isinstance(items, list) for items in (ids, hashes, rows)):
                raise ValueError("Invalid map lists.")
            if not len(ids) == len(hashes) == len(rows):
                raise ValueError("Map lengths do not match.")
            expected = [_checksum(article_id) for article_id in ids]
            if hashes != expected or len(set(hashes)) != len(hashes):
                raise ValueError("Invalid or duplicate article hashes.")
            for row in rows:
                _vector(row)
                if not math.isclose(math.hypot(*row), 1.0, abs_tol=1e-6):
                    raise ValueError("Stored vectors must have unit length.")
        except (KeyError, TypeError, ValueError) as exc:
            raise MapStoreError("Invalid map file.") from exc
        return data

    def _write(self, data):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                json.dump(data, handle, allow_nan=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise MapStoreError("Cannot write the map file.") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def save(self, article_ids, vectors):
        """Replace the full map. Reject duplicate IDs before writing."""
        if not isinstance(article_ids, list) or not isinstance(vectors, list):
            raise ValueError("Article IDs and vectors must be lists.")
        if len(article_ids) != len(vectors):
            raise ValueError("Each article ID needs one vector.")
        checksums = [_checksum(article_id) for article_id in article_ids]
        if len(set(checksums)) != len(checksums):
            raise ValueError("Article IDs must be unique.")
        rows = [_vector(row) for row in vectors]
        self._write({"version": 1, "article_ids": list(article_ids),
                     "checksums": checksums, "vectors": rows})

    def add(self, article_id, vector):
        """Append one article. Return False if its ID already exists."""
        checksum, row = _checksum(article_id), _vector(vector)
        data = self._read()
        if checksum in data["checksums"]:
            return False
        data["article_ids"].append(article_id)
        data["checksums"].append(checksum)
        data["vectors"].append(row)
        self._write(data)
        return True

    def get_map(self):
        """Return IDs, checksums, and the full N by 384 matrix."""
        data = self._read()
        return {key: data[key] for key in ("article_ids", "checksums", "vectors")}

    def search(self, vector, top_k=5):
        """Return the closest articles for one vector, including exact matches."""
        return self.search_many([vector], top_k=top_k)[0]

    def search_many(self, vectors, top_k=5):
        """Return one ranked list per query. Scores are B times A transpose."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(vectors, list):
            raise ValueError("Query vectors must be a list.")
        queries = [_vector(row) for row in vectors]
        data = self._read()
        results = []
        for query in queries:
            hits = [{"article_id": article_id,
                     "score": max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(row, query))))}
                    for article_id, row in zip(data["article_ids"], data["vectors"])]
            hits.sort(key=lambda hit: hit["score"], reverse=True)
            results.append(hits[:top_k])
        return results
