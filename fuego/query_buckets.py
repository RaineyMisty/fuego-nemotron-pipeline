"""Embed one user topic and find matching article IDs."""

import math
from numbers import Real

from fuego.embedding import MAX_TEXT_CHARS, Embedder
from fuego.map_store import MapStore


class QueryBuckets:
    def __init__(self, map_store=None, *, embedder=None):
        self.map_store = map_store if map_store is not None else MapStore()
        self.embedder = embedder if embedder is not None else Embedder()

    def query(self, topic, *, min_score=0.4):
        """Return up to ten article IDs, sorted by cosine similarity."""
        if not isinstance(topic, str) or not topic.strip() or len(topic) > MAX_TEXT_CHARS:
            raise ValueError(f"topic must have 1-{MAX_TEXT_CHARS} text characters.")
        if (isinstance(min_score, bool) or not isinstance(min_score, Real)
                or not math.isfinite(min_score) or not -1 <= min_score <= 1):
            raise ValueError("min_score must be a finite number between -1 and 1.")
        if not self.map_store.get_map()["article_ids"]:
            return []
        vector = self.embedder.embed(topic.strip())
        hits = self.map_store.search(vector, top_k=10)
        return [hit["article_id"] for hit in hits if hit["score"] >= min_score]
