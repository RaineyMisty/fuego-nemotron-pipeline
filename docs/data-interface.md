# Data and backend API

The default service uses two stages: ingest and store, then query and summarize.
This guide defines the current contract. The old catalog/feed contract is removed.

## Start

```bash
python3 -m fuego serve --demo --db work/demo.sqlite3
```

For live use, load NVIDIA_API_KEY and remove `--demo`. Use another database path.
Set `FUEGO_API_TOKEN` before binding to a non-loopback host.
HTTP uses `Authorization: Bearer TOKEN`. TCP uses a `token` field.
Only the server needs the NVIDIA key. Use TLS for public access.

## Ingest articles

`POST /v1/ingest`, with `Content-Type: application/json`:

```json
{
  "articles": [{
    "id": "article-1",
    "url": "https://example.com/article-1",
    "title": "Demo article",
    "source": "Demo source",
    "published_at": "2026-09-19T12:00:00Z",
    "observed_at": "2026-09-19T13:00:00Z",
    "text": "Original article text or a source-provided summary.",
    "gdelt": {"theme_codes": ["EDUCATION"], "tone": -1.2}
  }]
}
```

Required fields: `id`, `url`, and at least one timestamp.
Send `text` or `gdelt`. All other fields are optional. Unknown fields fail validation.
Text must be nonempty if present. Use null or omit it for metadata-only input.
Dates must have a timezone. They are converted to UTC.
Theme codes are a list of at most 100 strings. GDELT tone is a number from -100 to 100 or null.
Do not place a URL or Theme codes in `text` and label them as article content.

The service first extracts a scheme from each text. It then routes schemes to buckets.
It saves all records in one transaction. A failed batch leaves previous content unchanged.
Same-ID updates replace content and memberships. The URL must stay the same.
Send the saved text when updating a text-backed record. Metadata cannot erase it.
Repeated requests may call the model again. There is no idempotency-key cache.

The data response has `schema_version: "fuego-ingest.v1"`, provider/model/demo fields,
`stored_count`, and `records`. Each record has an ID, scheme, bucket IDs,
content basis, and scheme origin. The database also keeps original text and source fields.
No final user digest is made here.

## Ingest GDELT rows

`POST /v1/gdelt` accepts the existing four-column CSV or JSON:

```json
{"records":[{"Record_ID":"20260919120000-1","Article_Link":"https://example.com/a","Theme":"EDUCATION,12","Tone":"1,2,1,3,20,0,100"}]}
```

CSV header: `Record_ID,Article_Link,Theme,Tone`. All four values are strings.
Quote CSV values that contain commas. Theme may be empty.
Rows merge by Record_ID. Conflicting URL or Tone values fail validation.
Duplicate theme/offset pairs are removed. Different IDs with the same URL remain separate.
Tone has seven values: tone, positive score, negative score, polarity, activity density,
self/group density, and word count. The last value must be a nonnegative integer.

The default server saves metadata-only schemes. Its data schema is `gdelt-ingest.v1`.
It returns `stored_count`, `schemes`, records, buckets, stats, daily metrics, and theme counts.
These statistics are a bonus. They are not the final digest.
The Record_ID timestamp becomes `observed_at`, not `published_at`.
Empty-theme records are saved unmatched without a model call.
Use `/v1/ingest` with the same ID and URL to add text later.

The stateless Python helper `GdeltService(MetadataCore(client))` is still available.
It returns the old `gdelt-metadata.v1` statistics without storage. It is a bonus API.
It is not the default server workflow.

## Map an interest request

`POST /v1/query`:

```json
{"date":"2026-09-19","query":"Show me local AI, robotics, and school news."}
```

Nemotron selects from the configured buckets. It can choose several or none.
The data response has `schema_version: "fuego-query.v1"`, query, date, bucket_ids,
provider/model/demo fields, and status (`matched` or `no_matching_buckets`).
It does not create a summary. A matching bucket may be empty for that date.

## Summarize selected buckets

`POST /v1/summary`:

```json
{"date":"2026-09-19","bucket_ids":["ai","robotics","education"],"query":"Local AI and school news"}
```

The query is optional here. It defaults to “Summarize the selected topics.”
The backend chooses IDs, and this route does not run bucket mapping again.
Unknown or repeated IDs are rejected. An empty list is allowed.
The service reads the stored content and calls Nemotron now.
It checks that each evidence ID is in the selected content and every quote is exact.

## Make a digest in one call

`POST /v1/digest` takes the same input as `/v1/query`.
It runs bucket selection, reads the store, then generates the digest.
This is the main endpoint for the frontend's backend.

Summary and digest share this data shape:

```json
{
  "schema_version":"fuego-digest.v1",
  "provider":"nvidia",
  "model":"nvidia/nvidia-nemotron-nano-9b-v2",
  "is_demo":false,
  "generated_at":"2026-09-19T15:00:00+00:00",
  "query":"Local AI and school news",
  "date":"2026-09-19",
  "bucket_ids":["ai","education"],
  "status":"ready",
  "summary":[{"text":"A short point.","evidence":[{"article_id":"article-1","quote":"An exact text span"}]}],
  "sources":[{"id":"article-1","url":"https://example.com/article-1","title":"Demo article","source":"Demo source","published_at":"2026-09-19T12:00:00+00:00","observed_at":null,"time_basis":"published_at"}],
  "coverage":{"matched_records":1,"usable_articles":1,"missing_text_ids":[],"duplicate_url_ids":[]},
  "chart":{"metric":"stored_record_count","counts_overlap":true,"data":[{"bucket_id":"ai","value":1},{"bucket_id":"education","value":1}]}
}
```

This is a shape example, not a real model result.
Sources contain only cited articles. Coverage reports all selected records.
If no digest can be written, summary and sources are empty:

| Status | Meaning |
| --- | --- |
| `no_matching_buckets` | No bucket IDs were selected. |
| `no_data` | No saved records match the buckets and day. |
| `needs_article_text` | Matching records exist, but none has text. |
| `no_relevant_content` | Text exists, but the model found nothing relevant to the query. |
| `ready` | A validated digest was generated. Check coverage for missing text. |

The date is explicit. The backend resolves relative dates such as “today”.
Records use their UTC publication day, or observation day if publication time is missing.
Query text is also passed to the digest model to filter unrelated stories in broad buckets.
There is no separate geographic index or hard location filter in this version.

## Response envelope

HTTP returns:

```json
{"schema_version":"fuego-api.v1","request_id":"batch-1","ok":true,"data":{}}
```

Errors use `ok: false` and `error: {code, message}`.
Use optional `X-Request-ID` with 1-80 letters, digits, underscores, dots, or hyphens.
This is a trace ID, not a deduplication key.
Status codes: 400/411 framing, 401 token, 404 route, 408 upload timeout,
413 size, 415 media type, 422 input, 502 model failure, 503 busy, and 500 server error.
Send UTF-8 with Content-Length. Multipart, chunked, and compressed uploads are not supported.
`GET /health` checks the process only. It does not call NVIDIA.

## TCP

```bash
python3 -m fuego serve --transport tcp --port 9000 --demo --db work/demo.sqlite3
```

Send one newline-terminated JSON frame per connection:

```json
{"operation":"digest","format":"json","data":{"date":"2026-09-19","query":"AI and robotics"},"request_id":"digest-1"}
```

Operations: ingest, gdelt, query, summary, digest. The default is gdelt for older senders.
Use `format: "json"` for the new operations. GDELT also accepts `format: "csv"`
with CSV text in data. Optional fields are request_id and token.
Read until the response newline. The response is the envelope plus numeric status.
The connection then closes. The existing `examples/send_gdelt.py` still sends GDELT files.

## Python

```python
from fuego import Workflow
from fuego.client import NvidiaClient

workflow = Workflow(NvidiaClient(), "work/live.sqlite3")
workflow.ingest(article_input)
selection = workflow.query({"date": "2026-09-19", "query": "AI and school news"})
digest = workflow.summarize_buckets({"date": selection["date"], "query": selection["query"], "bucket_ids": selection["bucket_ids"]})
```

Or call `workflow.digest(request)` after ingest.
`InputError` and `PipelineError` propagate in Python. The server turns them into API errors.

## Bounds

Uploads: 8 MiB. TCP frames: 16 MiB. GDELT rows: 100,000.
The persistent workflow allows 100 unique records per ingest request, including GDELT.
The standalone metadata parser still allows 1,000 records.
Article text: 12,000 characters. Model scheme lists: 20 entries each. Source metadata schemes keep up to 100 theme codes. Buckets: 20.
Digest input: at most 100 matched records and 96,000 text characters.
If these limits are exceeded, split ingestion or choose fewer buckets. No live text is cut off.

One processing request runs at a time per server process. Other requests return 503.
At most eight connections are handled. Upload timeout is 15 seconds.
The NVIDIA client has a 90-second timeout per attempt and bounded retries.
Disconnecting does not cancel processing. There is no durable queue or result cache.
