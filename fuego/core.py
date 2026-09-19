"""Nemotron metadata routing. No files, sockets, or HTTP handlers here."""

import json
from typing import Protocol

from .ingest import Batch, InputError
from .pipeline import RULES, checked_call, validate_buckets, validate_routes

DEFAULT_BUCKETS = [
    {"id": "ai", "name": "AI", "description": "Artificial intelligence, machine learning, and AI tools."},
    {"id": "robotics", "name": "Robotics", "description": "Robots, robotics research, and robot events."},
    {"id": "economy", "name": "Economy", "description": "Economy, business, jobs, and industry."},
    {"id": "education", "name": "Education", "description": "Schools, universities, and education."},
    {"id": "health", "name": "Health", "description": "Healthcare, medicine, and public health."},
    {"id": "transport", "name": "Transport", "description": "Roads, transit, and transportation infrastructure."},
    {"id": "community", "name": "Community", "description": "Community, culture, and local civic life."},
]
METADATA_ROUTE = RULES + """
Route GDELT metadata records to zero or more supplied buckets using theme_codes only.
No article text is supplied. Do not infer events, local relevance, or article content.
Tone is a document metric, not public opinion. Theme codes are not confirmed events.
Return every supplied record once, including records with no relevant buckets.
Return exactly {"assignments":[{"article_id":"record-id","bucket_ids":["bucket-id"]}]}.
Use only supplied record and bucket IDs. Do not add summaries or explanations.
"""


class ModelClient(Protocol):
    mode: str
    model: str

    def complete(self, system: str, payload: dict) -> dict: ...


class MetadataCore:
    def __init__(self, client: ModelClient, buckets=None):
        self.client = client
        self.buckets = validate_buckets(DEFAULT_BUCKETS if buckets is None else buckets)

    def process(self, batch: Batch) -> dict[str, list[str]]:
        assignments = {r.id: [] for r in batch.records}
        pending, size = [], 0

        def flush():
            if not pending:
                return
            result = checked_call(self.client, METADATA_ROUTE,
                                  {"task": "route_metadata", "buckets": self.buckets, "articles": pending},
                                  lambda value: validate_routes(value, pending, {b["id"] for b in self.buckets}))
            assignments.update({row["article_id"]: row["bucket_ids"] for row in result["assignments"]})

        model_items = []
        for record in batch.records:
            if record.theme_codes:
                item = {"id": record.id, "theme_codes": record.theme_codes}
                item_size = len(json.dumps(item))
                if item_size > 48_000:
                    raise InputError("One record has too many theme codes for the model context.")
                model_items.append((item, item_size))

        for item, item_size in model_items:
            if pending and (len(pending) >= 8 or size + item_size > 48_000):
                flush()
                pending, size = [], 0
            pending.append(item)
            size += item_size
        flush()
        return assignments


class MetadataDemoClient:
    """Transport smoke test only: deliberately leaves every record unmatched."""

    mode = "demo"
    model = "offline-metadata-no-model"

    def complete(self, system, payload):
        return {"assignments": [{"article_id": r["id"], "bucket_ids": []}
                                for r in payload["articles"]]}
