"""Package validated model routes and deterministic GDELT metrics."""

from collections import Counter
from datetime import datetime, timezone

from .ingest import TONE_FIELDS


def metrics(records):
    return {"record_count": len(records), "unique_url_count": len({r.url for r in records}),
            "mean_tone": sum(r.tone[0] for r in records) / len(records) if records else None}


def package_result(batch, assignments, buckets, client):
    records = batch.records
    days = sorted({r.observed_at[:10] for r in records})
    return {
        "schema_version": "gdelt-metadata.v1", "analysis_type": "metadata_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": client.mode, "model": client.model, "is_demo": client.mode == "demo",
        "stats": {"input_rows": batch.input_rows, **metrics(records),
                  "duplicate_theme_rows": batch.duplicate_theme_rows,
                  "blank_theme_rows": batch.blank_theme_rows,
                  "records_without_themes": sum(not r.themes for r in records),
                  "unmatched_record_ids": [r.id for r in records if not assignments[r.id]]},
        "time_basis": "GDELT Record_ID timestamp; not article publication time",
        "daily_metrics": [{"date": day, **metrics([r for r in records if r.observed_at.startswith(day)])}
                          for day in days],
        "buckets": [{**bucket, **metrics([r for r in records if bucket["id"] in assignments[r.id]]),
                     "record_ids": [r.id for r in records if bucket["id"] in assignments[r.id]]}
                    for bucket in buckets],
        "theme_counts": [{"theme": code, "record_count": count} for code, count in
                         sorted(Counter(code for r in records for code in r.theme_codes).items())],
        "records": [{"record_id": r.id, "url": r.url, "observed_at": r.observed_at,
                     "themes": [{"code": code, "offset": offset} for code, offset in r.themes],
                     "tone": {**dict(zip(TONE_FIELDS, r.tone)), "word_count": int(r.tone[-1])},
                     "bucket_ids": assignments[r.id]} for r in records],
        "limitations": ["No article text or news summary is included.",
                        "No independent verification of Pittsburgh relevance.",
                        "Record counts are not event counts; bucket counts may overlap.",
                        "Mean tone is weighted equally by Record_ID, not by theme row or URL."]
    }


def success(request_id, data):
    return {"schema_version": "fuego-api.v1", "request_id": request_id, "ok": True, "data": data}


def failure(request_id, code, message):
    return {"schema_version": "fuego-api.v1", "request_id": request_id, "ok": False,
            "error": {"code": code, "message": message}}
