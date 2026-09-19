# Fuego Nemotron Pipeline

Build a news digest in two stages. Use NVIDIA Nemotron for both stages.
The app is a data workflow. It has no chat screen.
This README is the main guide. All sample news is fictional.

## Stage 1: store the content

```text
Article text or GDELT rows
  -> article scheme
  -> fixed topic buckets
  -> SQLite store
```

For article text, Nemotron extracts themes, people, organizations, locations,
tone, and a short source note. The note includes an exact quote from the text.
Nemotron then uses this scheme to choose zero or more bucket IDs.
The store saves the original text, source link, scheme, and bucket links.
It does not create a final user digest at this stage.

GDELT rows without text are stored as `metadata_only`.
Their schemes keep source tags. Missing fields stay empty or null.
Nemotron can route these schemes. It cannot summarize an article it has not read.
Send the article text later with the same ID and URL to enrich the record.

## Stage 2: answer a feed request

```text
User interests + date
  -> Nemotron selects bucket IDs
  -> read stored articles in those buckets
  -> Nemotron writes the final digest
  -> check evidence and return JSON
```

The digest contains up to five points with source IDs, quotes, and links.
A backend can call query and summary separately. It can also call digest once.
This is an on-demand summary. It does not select prewritten bucket summaries.

## Quick demo

Use Python 3.11 or newer. No extra packages or local GPU are needed.
Run these commands from the repository root:

```bash
python3 -m fuego ingest --input examples/ingest.json --db work/demo.sqlite3 --output work/ingested.json --demo
python3 -m fuego query --input examples/query.json --db work/demo.sqlite3 --output work/query.json --demo
python3 -m fuego summary --input examples/summary-request.json --db work/demo.sqlite3 --output work/digest.json --demo
```

Or replace the last two commands with one call:

```bash
python3 -m fuego digest --input examples/query.json --db work/demo.sqlite3 --output work/digest.json --demo
```

Demo mode uses simple rules and text copies. It tests the flow, not model quality.
Every result has `is_demo: true`. Real API failures never switch to demo mode.
Use a separate database for demo and live data.

## HTTP and TCP

```bash
python3 -m fuego serve --demo --db work/demo.sqlite3
```

| Route | Task |
| --- | --- |
| `GET /health` | Check the process. This does not test NVIDIA. |
| `POST /v1/ingest` | Extract schemes, route articles, and store them. |
| `POST /v1/gdelt` | Store GDELT CSV or JSON metadata. Return statistics too. |
| `POST /v1/query` | Map natural language interests to bucket IDs. |
| `POST /v1/summary` | Read selected buckets and create a digest now. |
| `POST /v1/digest` | Run query and summary in one call. |

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/ingest -H 'Content-Type: application/json' --data-binary @examples/ingest.json
curl --fail-with-body http://127.0.0.1:8000/v1/digest -H 'Content-Type: application/json' --data-binary @examples/query.json
```

See [the API guide](docs/data-interface.md) for full contracts and TCP frames.

## Use NVIDIA

1. Sign in at [NVIDIA Build](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2).
2. Open View Code and create an API key.
3. Copy `.env.example` to `.env` and edit the key.
4. Load `.env` in Bash. Remove `--demo` and use a live database.

```bash
cp .env.example .env
# Edit .env before this command.
source .env
python3 -m fuego serve --db work/live.sqlite3
```

The default model is `nvidia/nvidia-nemotron-nano-9b-v2`.
The endpoint is `https://integrate.api.nvidia.com/v1/chat/completions`.
The API format is OpenAI-compatible. The provider is NVIDIA. No OpenAI key is needed.
Set `NVIDIA_MODEL` to change the Nemotron model. Test other models before use.
Check your NVIDIA account for access and limits.
The app does not load `.env` itself. Git ignores keys and local work files.

## Scheme contract

Each text-backed scheme has these fields:

```json
{
  "themes": ["AI education"],
  "people": [],
  "organizations": [],
  "locations": ["Pittsburgh"],
  "tone": "neutral",
  "summary": "A source note based on the article.",
  "evidence": [{"article_id": "article-1", "quote": "An exact text span"}]
}
```

Lists can be empty. Tone is positive, negative, neutral, mixed, or unknown.
Tone describes the text. It does not measure public opinion.
Metadata-only schemes have `summary: null`, `evidence: []`, and `tone: "unknown"`.
Their numeric GDELT tone stays in the source metadata.
`scheme_origin` is `model` or `source_metadata`.
The source note helps routing. The final digest must cite original text.
Exact quote checks do not prove every claim is true. Review live outputs.

## Storage and time

SQLite saves the text and scheme. Restarting the server keeps the data.
A batch is saved only after every model result passes validation.
An existing ID is updated and its old bucket links are replaced.
An existing ID cannot change its URL. Metadata-only input cannot erase saved text.
Use a new database if you change the bucket definitions or switch demo mode.
Default buckets include AI, robotics, education, economy, health, transport, and community.
Use `--buckets examples/buckets.json` to set your team's definitions.

Requests need an explicit `date` in YYYY-MM-DD form. The backend resolves “today”.
Selection uses the UTC publication day when available, or the UTC observation day.
Every source shows its time basis. GDELT observation time is not publication time.
One article can belong to several buckets. A digest includes its text only once.
For duplicate URLs, the lowest article ID is used and skipped IDs are reported.

## Kept bonus features

- Four-column GDELT CSV and JSON input.
- Native TCP, token checks, request IDs, and bounded uploads.
- Theme counts, daily metrics, mean GDELT tone, and chart counts.
- A stateless `GdeltService(MetadataCore(client))` for metadata analysis only.

The default server uses the persistent two-stage workflow.
Counts describe stored records, not real-world events or public mood.
The old `run` and `feed` commands and precomputed catalog functions were removed.
Use `ingest`, then `query` and `summary`, or `digest`.

## Limits and checks

Each ingest request accepts at most 100 records. Each text can have 12,000 characters.
The service accepts at most 20 buckets. Schemes are routed in batches of eight.
A summary reads at most 100 matched records and 96,000 text characters.
Oversized selections fail. Text is never silently cut off in live mode.
Missing text is reported and is never used as article evidence.
The model gets one retry for invalid output. Transport retries are bounded.

This module does not scrape URLs, send email, run a scheduler, or provide a web UI.
The backend owns user accounts, dates, and delivery. It may store preferences itself.
There is no durable job queue or model-result cache. Repeated calls may use more API quota.
The local SQLite store is intended for the hackathon service, not a distributed database.

```bash
python3 -m unittest discover -s tests -v
```

Tests cover storage, routing, query mapping, delayed summaries, source checks,
errors, HTTP, and TCP. Offline tests do not verify a live NVIDIA connection.

[NVIDIA model reference](https://docs.api.nvidia.com/nim/reference/nvidia-nvidia-nemotron-nano-9b-v2)
