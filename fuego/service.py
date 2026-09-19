"""Share one service across Python, HTTP, and TCP."""

from .ingest import parse_input
from .output import package_result


class GdeltService:
    def __init__(self, core, workflow=None):
        self.core = core
        self.workflow = workflow

    def process(self, raw: bytes, media_type="text/csv") -> dict:
        batch = parse_input(raw, media_type)
        stored = None
        if self.workflow is not None:
            stored = self.workflow.ingest({"articles": [
                {"id": r.id, "url": r.url, "observed_at": r.observed_at,
                 "gdelt": {"theme_codes": r.theme_codes, "tone": r.tone[0]}}
                for r in batch.records]})
            assignments = {r["id"]: r["bucket_ids"] for r in stored["records"]}
        else:
            assignments = self.core.process(batch)
        result = package_result(batch, assignments, self.core.buckets, self.core.client)
        if stored is not None:
            result["schema_version"] = "gdelt-ingest.v1"
            result["stored_count"] = stored["stored_count"]
            result["schemes"] = stored["records"]
        return result

    def process_operation(self, operation, raw, media_type):
        if operation == "gdelt":
            return self.process(raw, media_type)
        if self.workflow is None:
            from .ingest import InputError
            raise InputError("This service has no article store.", "not_configured", 503)
        return self.workflow.dispatch(operation, raw, media_type)
