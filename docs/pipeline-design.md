# Pipeline design

## Service boundaries

Pipeline owns one local state directory. It shares one embedder and one AI client.
It keeps a durable SQLite job queue. Ingestion and request handling are separate operations.
The HTTP server accepts input into the queue. A worker drains jobs and refreshes prepared feeds.
Fixed and cluster requests read prepared responses. Query requests search and synthesize on demand.
A lock serializes model work and state updates inside one process. Run one process per state directory.

## Offline work

1. Enqueue each input record by Record_ID. Keep raw input until processing succeeds.
2. Normalize CSV-style date strings and null text fields at the boundary.
3. Use ArticleInput and article_processing. Retry API and model-output failures at most five times total. Pipeline alone owns retries.
   NVIDIA timeout is fixed at five seconds and client retries at zero. No retry sleeps are added.
4. Embed semantic_text. Add to MapStore. Retry unfinished jobs without repeating successful article processing.
5. Keep metadata separately for output. Clear successful raw input from the queue.
6. Run the cluster update check. A full update can be requested explicitly.
7. Refresh fixed and cluster feeds, summaries, and fixed bucket links.

Completed articles are durable before vector creation. If embedding fails, the job remains retryable.
A crash after insertion or map writing is safe to retry because both stores deduplicate by article ID.
Failed jobs retain input. Refresh errors do not remove completed articles or the previous prepared response.

## Requests

GET /health and GET /jobs report local state.
POST /ingest accepts a JSON list of input articles and returns queued IDs.
GET /fixed and GET /clusters return prepared JSON, or 503 when no prepared response exists.
POST /query accepts {"topic": "..."} and returns an on-demand query bucket.
The query uses a temporary SQLite snapshot for output packaging. It never changes permanent bucket links.
Unknown routes, bad input, and processing errors return JSON with appropriate HTTP status codes.
The server binds to loopback by default. Authentication and remote deployment are outside this local service.

## Output and trends

Use the existing fuego-response.v1 shape. Synthesis supplies names and summaries.
Empty buckets keep their topic name and explain that there are no matching articles.
Failed synthesis is explicit in bucket summary_status and response warnings; it is not reported as success.
Reuse synthesis only when its article IDs and summaries match.
Activity compares equal publication-time windows. Missing prior-window coverage yields insufficient_data.
Direction compares the unit centroids of those windows and reports nearby fixed topics for the shift vector.
These are coverage changes, not claims about real-world event rates or causation.

## Test controls

mock_ai replaces only Nemotron replies. Its output is clearly labeled and is not an AI quality test.
mock_query supplies a test topic only when a request omits its topic.
Embedding stays real in the delivered input-file smoke run. Unit tests inject small deterministic vectors.
State records the model and mock mode so a live process cannot reuse mock article results by accident.
No dependencies are installed by this task. No prompt files are changed.

## Work progress and reports

Work prints each article, stage, attempt, failure, and completion to stderr with immediate flushing.
A five-second heartbeat covers long model and local computation steps.
Each run writes a timestamped JSON report and reports/latest-work.json in its state directory.
Reports separate article failures, filtered inputs, pending work, synthesis failures, and refresh failures.
Loss rate is failed / (done + failed); filtered and pending records are excluded.
Failed raw inputs remain retryable. An interrupted run also saves its partial report.
