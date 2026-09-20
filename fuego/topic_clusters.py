"""Run Euclidean K-means on the article map and save its results."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from fuego.map_store import DIMENSIONS, MapStore


class TopicClustersError(RuntimeError):
    """Cluster results could not be read, written, or computed."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _distance(a, b):
    return math.fsum((x-y)**2 for x, y in zip(a, b))


def _fit(rows, k, max_iterations):
    if not rows:
        return [], [], 0, 0.0
    # Start with the first row, then choose the farthest uncovered row.
    centers = [rows[0][:]]
    nearest = [_distance(row, centers[0]) for row in rows]
    while len(centers) < min(k, len(rows)):
        index = max(range(len(rows)), key=lambda i: nearest[i])
        if nearest[index] == 0:
            break
        centers.append(rows[index][:])
        nearest = [min(d, _distance(row, centers[-1])) for row, d in zip(rows, nearest)]
    previous = None
    for iteration in range(1, max_iterations+1):
        labels = [min(range(len(centers)), key=lambda j: _distance(row, centers[j])) for row in rows]
        groups = [[] for _ in centers]
        for row, label in zip(rows, labels):
            groups[label].append(row)
        # Drop empty groups. Exact ties always choose the first center.
        active = [j for j, group in enumerate(groups) if group]
        remap = {old: new for new, old in enumerate(active)}
        labels = [remap[label] for label in labels]
        centers = [[math.fsum(row[d] for row in groups[j])/len(groups[j])
                    for d in range(DIMENSIONS)] for j in active]
        if labels == previous:
            inertia = math.fsum(_distance(row, centers[label]) for row, label in zip(rows, labels))
            return centers, labels, iteration, inertia
        previous = labels
    raise TopicClustersError("K-means did not converge. Increase max_iterations.")


class TopicClusters:
    def __init__(self, map_store=None, *, k=20, max_iterations=100, directory=None):
        for name, value in (("k", k), ("max_iterations", max_iterations)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.map_store = map_store if map_store is not None else MapStore()
        self.k, self.max_iterations = k, max_iterations
        root = Path(directory) if directory is not None else self.map_store.path.parent / "clusters"
        self.path = root / "topics.json"

    def load(self):
        """Return the last saved result, or None if it does not exist."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise TopicClustersError("Cannot read cluster results.") from exc
        try:
            data = payload["data"]
            if payload["checksum"] != _hash(data) or data["version"] != 1:
                raise ValueError("Invalid checksum or version.")
            centers, assignments, source = data["centers"], data["assignments"], data["source"]
            if not isinstance(centers, list) or not isinstance(assignments, dict) or not isinstance(source, dict):
                raise ValueError("Invalid result shape.")
            if set(assignments) != set(source):
                raise ValueError("Invalid article IDs.")
            for row in centers:
                if (not isinstance(row, list) or len(row) != DIMENSIONS
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
                    raise ValueError("Invalid center.")
            if any(type(label) is not int or not 0 <= label < len(centers) for label in assignments.values()):
                raise ValueError("Invalid cluster assignment.")
            if type(data["k"]) is not int or data["k"] <= 0:
                raise ValueError("Invalid k.")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise TopicClustersError("Invalid cluster results. Use update to rebuild them.") from exc
        return data

    def _save(self, data):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             delete=False) as handle:
                temporary = handle.name
                json.dump({"data": data, "checksum": _hash(data)}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise TopicClustersError("Cannot write cluster results.") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _source(snapshot):
        return {article_id: _hash(row) for article_id, row in zip(snapshot["article_ids"], snapshot["vectors"])}

    def _update(self, snapshot):
        centers, labels, iterations, inertia = _fit(snapshot["vectors"], self.k, self.max_iterations)
        data = {"version": 1, "k": self.k, "centers": centers,
                "assignments": dict(zip(snapshot["article_ids"], labels)),
                "source": self._source(snapshot), "iterations": iterations, "inertia": inertia}
        self._save(data)
        return data

    def update(self):
        """Recompute now from a single map snapshot, even below 100 new rows."""
        return self._update(self.map_store.get_map())

    def update_if_needed(self):
        """Build once, then rebuild after 100 new articles. Return whether updated."""
        snapshot = self.map_store.get_map()
        current = self._source(snapshot)
        saved = self.load()
        needed = saved is None or saved["k"] != self.k
        if not needed:
            old = saved["source"]
            needed = (len(current.keys()-old.keys()) >= 100
                      or any(current.get(key) != value for key, value in old.items()))
        if needed:
            self._update(snapshot)
        return needed
