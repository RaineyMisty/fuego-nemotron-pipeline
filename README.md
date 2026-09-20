# Fuego AI

## Complete local pipeline

The service design is in `docs/pipeline-design.md`.
`fuego/pipeline.py` separates offline jobs from fixed, query, and cluster requests.
`python -m fuego` provides local commands and an HTTP server.
Use the existing virtual environment with FastEmbed and the cached MiniLM model.

Run the full input-file smoke test:

```bash
source .venv/bin/activate
python -B -m integration.smoke_pipeline --mock-ai --local-files-only
```

The actual supplied file is `input/fuego_input_test.json`. It has 19 records.
The smoke uses that file by default. Use --input to choose another JSON list.
It writes `output/pipeline-smoke/fixed.json`, `cluster.json`, `query.json`, and `jobs.json`.
Its durable state is in `work/pipeline-delivery/`. SQLite files are under db/ and vectors under map/.
It uses real FastEmbed, SQLite, map queries, K-means, and JSON output.
Only AI replies are simulated. The output says mock_ai=true and contains a warning.
Mock summaries are test data, not production summaries or evidence of Nemotron quality.
The smoke sets the activity endpoint to the latest input publication time for a repeatable historical test.

### Independent controls

`--mock-ai` replaces the two Nemotron steps only. Live AI is the default.
`--mock-query "topic"` supplies a default topic only when no topic is given.
Explicit request topics always win. This switch does not replace vector search.
The same controls are PipelineConfig.mock_ai and PipelineConfig.mock_query in Python.
The input-file smoke sets a basketball query by default. Embedding is always real there.
Unit tests inject small vectors to run without optional packages.
Use a different state directory for live and mock AI. The service rejects mixed modes or model IDs.

### Offline operations

Global flags go before the command:

```bash
python -B -m fuego --mock-ai --local-files-only ingest input/fuego_input_test.json
python -B -m fuego --mock-ai --local-files-only work
python -B -m fuego --mock-ai status
python -B -m fuego --mock-ai --local-files-only refresh --force-clusters
python -B -m fuego --mock-ai export fixed --output output/fixed.json
python -B -m fuego --mock-ai export cluster --output output/clusters.json
python -B -m fuego --mock-ai --local-files-only export query --topic "New York Knicks basketball" --output output/query.json
```

These commands use `work/pipeline-demo/`. Set --state to use another directory.
Use `work --retry-failed` to retry failed jobs. Inputs remain in SQLite until success or filtering.
Completed records clear raw text from the queue. Metadata remains available for output.
Pipeline retry settings are fixed: NVIDIA_TIMEOUT=5, NVIDIA_MAX_RETRIES=0, and attempts=5.
The code enforces these values even if the environment has different timeout or retry values.
Pipeline is the only retry owner: one initial call plus at most four retries, with no retry sleep.
This applies to article processing and topic synthesis. --attempts accepts only 5.
Standalone AI clients keep their own settings; only Pipeline enforces this policy.
A vector failure can be retried without processing the article again.
Duplicate Record_ID values keep the first queued record. Correct a bad queued input in a new state/import;
this CLI does not overwrite existing records silently. The original input file is never changed.
Publication_Date strings of digits become integer milliseconds; null optional text fields become empty strings.
Other invalid records remain failed jobs with their payload and error type.

During work, progress goes to stderr and is flushed immediately. It shows the article ID,
position in the batch, stage, attempt out of five, result, and elapsed time.
A heartbeat prints the active stage every five seconds during long work.
Embedding, K-means, bucket processing, and cache hits are visible too.
The five-second timeout applies to NVIDIA transport I/O, not the whole work command or embedding.

After five failed attempts, the article is marked failed (LOST in the log). Its raw input stays
in the job queue. No original input file is deleted. Use work --retry-failed to try again.
At the end, a human-readable summary goes to stderr and the full JSON report goes to stdout.
Reports are also saved under the state directory:

- reports/work-<UTC timestamp>.json keeps each run.
- reports/latest-work.json holds the latest run.

The report has run_summary for this run and summary for all unique queued article IDs.
The loss rate is failed / (done + failed). Long articles filtered by policy are excluded.
Pending articles are listed separately, including after Ctrl+C. A zero denominator gives zero.
failed_articles lists IDs, attempts, error types, and failure stages. Synthesis failures are
reported separately and do not count as lost articles. Refresh errors still produce a report.
work exits with code 1 if articles, synthesis, or refresh failed; Ctrl+C exits with 130.

For the supplied input, run live AI processing with progress and a final report:

```bash
source .env
python -B -m fuego --local-files-only ingest input/fuego_input_test.json
python -B -m fuego --local-files-only work --retry-failed
```

The default live report is work/pipeline-live/reports/latest-work.json.
When using --state, use the same path for ingest and work. Add --mock-ai to both commands
only when testing fake AI replies. The input file is input/fuego_input_test.json.

Fixed feeds are prepared after offline work. Synthesis results are reused for unchanged source summaries.
Clusters rebuild on first use, after 100 new articles, or with --force-clusters.
Until then, pending_cluster_articles reports indexed articles absent from the saved clusters.
A failed synthesis keeps source articles and uses summary_status=failed plus a warning.
Empty buckets use summary_status=empty. Neither is reported as a successful AI summary.
A refresh failure preserves the previous prepared feed. /health reports feeds_stale.

### Local HTTP server

```bash
python -B -m fuego --mock-ai --mock-query "New York Knicks basketball" --local-files-only serve --port 8000
```

The server binds to 127.0.0.1 by default. Run one process per state directory.
It has a durable input queue and a background worker. It is a local service with no authentication.
Do not expose it publicly without adding deployment controls.

| Request | Result |
| --- | --- |
| GET /health | Article count, job states, test mode, and stale-feed flag |
| GET /jobs | Job status without raw article text |
| POST /ingest | JSON list of input articles; returns HTTP 202 |
| GET /fixed | Prepared fixed buckets |
| GET /clusters | Prepared cluster buckets |
| POST /query | JSON object with topic; returns a query response |

```bash
curl http://127.0.0.1:8000/health
curl -H 'Content-Type: application/json' --data-binary @input/fuego_input_test.json http://127.0.0.1:8000/ingest
curl http://127.0.0.1:8000/fixed
curl -H 'Content-Type: application/json' -d '{"topic":"New York Knicks basketball"}' http://127.0.0.1:8000/query
```

Feed reads do not call AI. Query requests can call AI and take longer.
Model operations and updates are serialized. Queue submission and health checks can run during model work.
Unknown routes return 404, invalid requests 400, and unavailable feeds or processing 503.
The worker resumes pending jobs and stale feed preparation after restart.
Failed jobs need an explicit retry. Stop the server before using CLI maintenance on its state directory.

### Live AI

```bash
source .env
python -B -m fuego --local-files-only ingest input/fuego_input_test.json
python -B -m fuego --local-files-only work
python -B -m fuego --local-files-only serve
```

Live mode uses `work/pipeline-live/`. Pipeline fixes timeout to 5 seconds and client retries to zero; other NVIDIA settings still apply.
No key is needed just to enqueue or inspect jobs. Processing and synthesis need a key.
Use --model-cache to select your cached model folder. Omit --local-files-only to allow model downloads.
No packages are installed automatically. The delivered run used mock AI, not a live NVIDIA request.

### Activity, direction, and output

Activity compares two equal publication-time windows (one day by default).
It counts all matching stored articles, before the top-ten output limit for fixed and cluster feeds.
Query activity uses its retrieved top-ten sample. It is not a count of every matching article.
If the indexed corpus does not reach the prior window start, historical values are insufficient_data.
A zero prior count gives a null ratio; it never produces Infinity.
Direction compares normalized centroids and ranks fixed topic vectors against the shift vector.
These fields describe changes in collected coverage, not real-world event rates or causal trends.
Sparse input cannot establish a reliable trend. No temporal data is invented.
Set `PipelineConfig.window_ms` in Python for another window. CLI refresh accepts --as-of milliseconds.

Responses follow fuego-response.v1 and add test flags, warnings, summary_status, indexed_articles,
and pending_cluster_articles where relevant. Embeddings and input metadata are included.
Query output uses a temporary SQLite snapshot and never replaces fixed bucket links.
JSON exports are atomic. Existing output examples and prompt files remain unchanged.

```bash
python -B -m unittest discover -s test -v
python -B -m integration.smoke_server
python -B -m integration.smoke_trend_activity
python -B -m integration.smoke_trend_direction
```

The server smoke uses loopback HTTP, a filtered input, and temporary state. No model or API is needed.


The AI client in `fuego/ai.py` calls NVIDIA Nemotron.
The article processor in `fuego/article_processing.py` extracts keywords and a reader summary.
Neither module stores articles.
The files in `prompt/` describe future work. They are read-only.

Use Python 3.11 or newer. AI calls use the standard library. Embedding needs the optional packages below.

## Setup

Copy `.env.example` to `.env`. Set your NVIDIA API key, then load it in Bash:

```bash
source .env
```

The module reads environment variables. It does not load `.env` itself.

## Call

```python
from fuego.ai import NemotronClient

client = NemotronClient()
response = client.complete([
    {"role": "system", "content": "Give a short answer."},
    {"role": "user", "content": "What is a language model?"},
])
print(response["choices"][0]["message"]["content"])
```

`complete` returns the raw API JSON object. It does not parse the model's text.
Usage, finish reason, and other response fields are kept.
The caller handles truncated answers, refusals, and any task-specific output format.
Only text messages with system, user, or assistant roles are supported. Streaming is off.

## Settings

These defaults apply to standalone AI calls. Pipeline overrides timeout to 5 seconds and retries to zero.

| Variable | Default |
| --- | --- |
| `NVIDIA_API_KEY` | Required |
| `NVIDIA_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| `NVIDIA_TIMEOUT` | `20` seconds per attempt |
| `NVIDIA_MAX_TOKENS` | `16384` |
| `NVIDIA_TEMPERATURE` | `0` |
| `NVIDIA_ENABLE_THINKING` | `false` |
| `NVIDIA_MAX_RETRIES` | `2` retries after the first attempt |

You can also pass `AIConfig` to `NemotronClient`.
The endpoint is fixed to `https://integrate.api.nvidia.com/v1/chat/completions`.
The model and thinking option keep the current project settings. Model access depends on your NVIDIA account.
Invalid input raises `ValueError`. API or connection failures raise `AIError`.
Temporary failures retry with bounded delays. Authentication errors do not retry.
Keys, prompts, and API error bodies are not logged. Redirects are blocked.

## Tests

```bash
python3 -B -m unittest discover -s test -v
```

Tests use mocked API responses. They do not send requests or spend API quota.
They cover settings, request data, raw responses, errors, retries, timeouts, and limits.
Live NVIDIA access is not verified by these tests.


## Live smoke test

Run from the repository root. This makes a real API request and uses NVIDIA quota.

```bash
source .env
python3 -B -m integration.smoke_ai --timeout 120
```

You can also run the file directly:

```bash
python3 -B integration/smoke_ai.py --timeout 120
```

The smoke test reads exported environment variables. It does not load `.env`.
It defaults to 64 output tokens and zero retries. Other settings come from AIConfig.
Use `--max-tokens 256` or `--retries 1` if needed. Use `--help` for options.
It prints the endpoint, model, settings, elapsed time, and model reply.
It does not print the key or write result files.

Exit codes: 0 means a complete text reply, 1 means a request failure,
2 means a configuration error, 3 means an incomplete or invalid reply,
and 130 means the request was interrupted.
A valid API connection alone does not count as a pass: the response must have
nonempty text and `finish_reason: stop`.

`fuego/ai.py` is a library. Running that file alone does not send a request.
The smoke test calls `NemotronClient.complete` through that library.
The unit tests also check this runner with mocked responses; they do not run the live check.


## Process an article

The module has three steps:

1. `build_messages(article, metadata=None)` builds English instructions and input data.
2. `process_article(article, metadata=None, client=None)` calls `fuego.ai`.
3. `parse_response(response)` checks the model JSON and returns the result.

```python
from fuego.article_processing import process_article

result = process_article(article_text, {"source": "News source"})
print(result["overview"])
print(result["summary"])
```

`keywords` contains exactly 20 distinct content words or short phrases.
`overview` joins these terms with semicolons for later embedding. It is not a second summary.
`summary` is a short text for a reader. The model uses the article's language.
Names can include small words, such as "University of Pittsburgh". Standalone articles
and common English prepositions are rejected as keywords.
The prompt asks the model to remove ads and keep facts, names, numbers, and uncertainty.
Metadata is source context. It cannot override article facts or the instructions.
If there is not enough news for 20 useful terms, the model must report insufficient content.
The processor never fills missing keywords with invented values.

Input limits: 60,000 article characters and 16,000 serialized metadata characters.
Metadata must be a JSON object. Nothing is silently truncated.
Bad input raises `ValueError`. Invalid model output raises `ArticleProcessingError`.
Transport errors from `fuego.ai` remain `AIError`. There are no extra model retries here.
Format checks cannot prove that the keywords and summary are factually correct.
Review the real output, especially names, numbers, missing facts, and ad removal.

## Article smoke test

This uses the real `article_processing` module and the real `ai.py` client.
There are no mock responses in the runner. It sends the supplied text to NVIDIA.

```bash
source .env
python3 -B -m integration.smoke_article_processing --article integration/article_sample.txt --timeout 120
```

The bundled sample is fictional and has an ad to check that the model removes it.
To use your own article and optional metadata:

```bash
python3 -B -m integration.smoke_article_processing --article /path/to/article.txt --metadata /path/to/metadata.json --timeout 120
```

The article file must contain UTF-8 text, not a URL. Use `--article -` to read stdin.
You can also run `python3 -B integration/smoke_article_processing.py` with the same options.
The runner defaults to 2048 output tokens and zero retries. Other AI settings come from the environment.
It prints the result JSON to stdout and progress to stderr. It writes no files.
Exit codes: 0 success, 1 API failure, 2 input/configuration error, 3 invalid model result, 130 interrupted.
`PASS` means a real response met the output contract. It does not prove factual quality.

Run all offline tests with `python3 -B -m unittest discover -s test -v`.


## Text embedding

`fuego/embedding.py` uses FastEmbed and ONNX Runtime on CPU.
The only selected model is `sentence-transformers/all-MiniLM-L6-v2`.
The model name keeps its original prefix; the sentence-transformers package and PyTorch are not needed.
It does not call Nemotron or use a language-model prompt.
Install the optional dependency in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embedding]'
```

```python
from fuego.embedding import Embedder

embedder = Embedder()
vector = embedder.embed(article_result["overview"])
vectors = embedder.embed_many(["solar power; clean energy", "renewable electricity"])
```

Every result is a list of 384 Python floats with L2 norm 1.
Keep one Embedder instance to reuse the model. Loading happens on the first call.
This setup supports `device="cpu"` only.
The first call downloads the ONNX model. Set `cache_folder` to choose the cache location.
Old Sentence Transformers caches are not compatible. Use a new folder such as `work/onnx-models`.
Use `local_files_only=True` after the model is cached to avoid downloads.
The embedding extra does not affect the AI client's dependencies.

The module rejects empty text, text over 20,000 characters, and text over the
model's token limit. MiniLM normally uses a 256-token limit, including special tokens.
Nothing is silently truncated. Use the article overview, not the full article.
This model is designed for English text. Check quality before using non-English overviews.
Input errors raise ValueError. Model and vector errors raise EmbeddingError.

## Embedding smoke test

```bash
python -B -m integration.smoke_embedding --cache-folder work/onnx-models
```

The runner uses the real model. No API key is needed.
It embeds two fixed sets of 20 related but different terms about clean energy.
It prints both full vectors and their norms, cosine similarity, cosine distance,
and Euclidean distance as JSON. Progress goes to stderr.
Use `python -B integration/smoke_embedding.py` with the same options for a direct run.
Use `--local-files-only --cache-folder work/onnx-models` to reuse the local cache offline.

The default sample threshold is cosine similarity >= 0.70.
This is a smoke-test choice, not a universal rule for matching news.
Use `--min-similarity` to set a different threshold before running the test.
For unit vectors, cosine distance = 1 - similarity and Euclidean distance squared
is approximately 2 times cosine distance. Smaller distances mean closer vectors.
Exit codes: 0 passes the threshold, 1 misses it, 2 is a model/input failure, 130 is cancelled.
No vectors or cache files are committed to the repository.

Offline tests use small mock vectors. They need no model download:

```bash
python3 -B -m unittest discover -s test -p 'test*embedding.py' -v
```

[Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[FastEmbed guide](https://qdrant.github.io/fastembed/Getting%20Started/)

## Vector map

`fuego/map_store.py` stores article vectors. It uses the standard library only.
It does not call AI or load an embedding model.

```python
from fuego.map_store import MapStore

store = MapStore()  # Use the map folder at the project root.
store.save(["article-1", "article-2"], [vector_1, vector_2])
added = store.add("article-3", vector_3)
data = store.get_map()
hits = store.search(query_vector, top_k=5)
batches = store.search_many(bucket_vectors, top_k=5)
```

These are the five interfaces:

- `save(article_ids, vectors)` replaces the full map. Duplicate IDs are rejected.
- `add(article_id, vector)` adds one row. It returns True if added, or False if the ID exists.
- `get_map()` returns `article_ids`, `checksums`, and `vectors` in matching row order.
- `search(vector, top_k=5)` returns a list of `{article_id, score}` objects.
- `search_many(vectors, top_k=5)` returns one such list for each query, in query order.

IDs must be nonempty strings. Use the same stable ID for the same article.
Each checksum is a SHA-256 hash of the exact article ID. It is an ID index, not a content hash.
An existing ID is skipped even if its new vector differs. Different IDs can share a vector.
Vectors must have 384 finite numbers and a nonzero norm. They are normalized before use.
The map is an N by 384 matrix. IDs and checksums are separate lists in the same JSON file.
1000 rows and 20 queries are supported; these are not fixed limits.

Scores are cosine similarities: the dot product of unit vectors.
Batch scores follow B times A transpose. Higher scores come first.
Ties keep stored row order. Exact matches are included. A large top_k returns all available rows.
An empty map returns no hits. An empty query batch returns an empty list.

The default file is `map/articles.json`. Pass `MapStore("path/to/map")` to use another folder.
Each write replaces the file atomically. Use one writer at a time; concurrent writes are not supported.
Bad input raises ValueError. File errors raise MapStoreError.
Missing files mean an empty map. Invalid files raise an error when read.
Generated map files are ignored by Git.

## Map smoke test

Run from the repository root:

```bash
python -B -m integration.smoke_map_store
python -B -m unittest discover -s test -p 'test_map_store.py' -v
```

You can also run `python -B integration/smoke_map_store.py` directly.
The smoke test calls all five real interfaces with known 384-dimensional vectors.
It checks disk reload, duplicate IDs, matrix values, and single and batch scores.
Expected scores are 1, 0.707107, and 0. It prints five PASS lines.
It uses a temporary folder and removes it when done. Your map is not changed.
No API key, model download, or extra package is needed. Exit code 0 means success; 1 means failure.

## Named buckets

`fuego/named_buckets.py` finds articles for 20 fixed news topics.
The static definitions are in `fuego/bucket_definitions.py`.
Each short English prototype is embedded once with the same MiniLM model as articles.
These are topic descriptions, not LLM instructions. No Nemotron call is needed.

```python
from fuego.embedding import Embedder
from fuego.map_store import MapStore
from fuego.named_buckets import NamedBuckets

buckets = NamedBuckets(
    MapStore(),
    embedder=Embedder(cache_folder="work/onnx-models", local_files_only=True),
)
definitions = buckets.definitions()
article_ids_by_bucket = buckets.query(min_score=0.4)
```

Use your existing model cache folder. The default Embedder can download the model on first use.
`query()` uses one map batch query and returns `{bucket_id: [article_id, ...]}`.
Each list has at most ten IDs, sorted by cosine score. Scores below min_score are dropped.
The default min_score is 0.4. This is a starting value, not a measured quality threshold.
A score equal to the threshold is included. One article can appear in several buckets.
Empty maps return 20 empty lists without loading the model. Small maps return fewer IDs.
Topics cover different news areas, but their real semantic separation is not guaranteed.

`load_vectors()` loads or builds the 20 unit vectors. `load_vectors(rebuild=True)` rebuilds them.
The default cache is `map/buckets/vectors.json`. Set cache_directory to use another folder.
The cache key includes topic definitions, model name, backend, dimensions, and format version.
Missing, stale, or invalid caches are rebuilt. Reload the object after editing static definitions.
Rebuild the cache and article map together if you change the embedding model or its behavior.
Map files must use the same model; their format does not record model identity.
Use one cache writer at a time. Embedding and map errors keep their original error types.
Invalid bucket vectors or cache write failures raise NamedBucketsError.

```bash
python -B -m integration.smoke_named_buckets
python -B -m unittest discover -s test -p 'test_named_buckets.py' -v
```

Direct run: `python -B integration/smoke_named_buckets.py`.
The smoke test uses real map files and queries with known sample vectors.
It checks cosine scores, ranking, filtering, the ten-ID limit, cache reload, and empty maps.
Only the embedder is replaced with fixed vectors. This does not test real model quality.
It uses a temporary folder, needs no extra packages, and does not change your map.
Exit code 0 means success; 1 means failure.

## Query buckets

`fuego/query_buckets.py` finds articles for one user topic.
It embeds the topic directly with the same MiniLM model used for article vectors.
It does not rewrite the topic, call Nemotron, or use an LLM prompt.

```python
from fuego.embedding import Embedder
from fuego.map_store import MapStore
from fuego.query_buckets import QueryBuckets

queries = QueryBuckets(
    MapStore(),
    embedder=Embedder(cache_folder="work/onnx-models", local_files_only=True),
)
article_ids = queries.query("University of Pittsburgh", min_score=0.4)
```

Use your existing model cache folder. Keep one QueryBuckets object to reuse its embedder.
The default Embedder can download the model on first use.
`query(topic, min_score=0.4)` calls `MapStore.search` with top_k=10.
It returns article IDs, not article text. The backend can load articles by ID.
Scores equal to min_score are included. Lower scores are dropped.
Results keep descending cosine order. Ties keep map row order.
The default threshold is a starting value, not a measured quality guarantee.
Small maps can return fewer than ten IDs. No matches return an empty list.
Empty maps do not load the model. Queries do not write map files or cache user topics.

Topics must be nonempty strings of at most 20,000 characters.
Outer spaces are removed. The embedder also checks its 256-token limit when used.
Input is never silently truncated. English topics best match this model's design.
Article vectors and query vectors must use the same embedding model.
Input errors raise ValueError. EmbeddingError and MapStoreError pass through unchanged.

```bash
python -B -m integration.smoke_query_buckets
python -B -m unittest discover -s test -p 'test_query_buckets.py' -v
```

Direct run: `python -B integration/smoke_query_buckets.py`.
The smoke test uses real map storage and queries with known sample vectors.
Only the embedder is replaced. It checks scores, ranking, thresholds, different queries,
no matches, the ten-ID limit, and empty maps. It does not test model quality.
It uses a temporary folder and needs no API key, model download, or extra package.
Exit code 0 means success; 1 means failure.

## Topic clusters

`fuego/topic_clusters.py` runs Euclidean K-means on the full article map.
It uses the standard library. No model, prompt, or extra package is needed.

```python
from fuego.map_store import MapStore
from fuego.topic_clusters import TopicClusters

store = MapStore()
clusters = TopicClusters(store, k=20)
result = clusters.update()  # Recompute now.
centers = result["centers"]
assignments = result["assignments"]  # Article ID -> zero-based cluster index.

store.add(article_id, vector)
updated = clusters.update_if_needed()
last_result = clusters.load()
```

The backend must call `update_if_needed()` after adding articles or after a batch.
There is no background worker or automatic hook in MapStore.
The first check builds the result. Later checks wait for 100 new unique article IDs.
Duplicate adds do not count. Pending articles are not in the saved assignments yet.
An explicit `update()` includes them at once. Removing an old article, changing its
vector, or changing k causes the next check to rebuild immediately.

The default file is `map/clusters/topics.json`. Set directory to change the result folder.
`load()` returns the last saved snapshot, or None if there is no result.
Results also include the requested k, iteration count, inertia, and source vector hashes.
Inertia is the sum of squared Euclidean distances to assigned centers.
Writes are atomic. Use one writer and one consistent k per result folder.
Corrupt results raise TopicClustersError; use `update()` to rebuild them.
Map errors pass through as MapStoreError. Invalid settings raise ValueError.

The default k is 20 and max_iterations is 100. Both must be positive integers.
Initialization is deterministic: start with the first row, then pick farthest rows.
Ties keep the first center. Empty groups are removed. Duplicate points or small maps
can produce fewer than k centers. An empty map produces no centers or assignments.
Cluster IDs are local to a saved result and may change after an update.
K-means finds a local solution; it does not guarantee the best possible grouping.
If it does not converge, it raises TopicClustersError and keeps the old result.

Article vectors are unit vectors. For two unit vectors, squared Euclidean distance
is 2 minus 2 times cosine similarity. K-means centers are arithmetic means and are
not normalized. They can even be zero. Do not treat center dot products as cosine
scores or pass zero centers to map search. Clusters have no generated topic names.

```bash
python -B -m integration.smoke_topic_clusters
python -B -m unittest discover -s test -p 'test_topic_clusters.py' -v
```

Direct run: `python -B integration/smoke_topic_clusters.py`.
The smoke test uses real map files and the real clustering algorithm in a temporary folder.
It checks assignments, exact mean centers, inertia 0.4, saved results, the 99/100 update
boundary, explicit updates, and empty maps. It leaves your map unchanged.
No dependencies are installed or models downloaded. Exit code 0 means success; 1 means failure.

## Topic synthesis

`fuego/topic_synthesis.py` combines related article summaries into a topic title and summary.
It uses `fuego.ai` and the current Nemotron settings.
The default model is `nvidia/nemotron-3.5-lightning-30b-a3b`.

```python
from fuego.topic_synthesis import synthesize_topic

result = synthesize_topic([
    {"summary": "The city will test ten electric buses.",
     "metadata": {"source": "City report", "date": "2026-09-20"}},
    {"summary": "The city finished chargers for its electric bus trial."},
])
print(result["title"])
print(result["summary"])
```

Input is a list of one to ten article objects. Each needs a nonempty summary.
Optional metadata must be a JSON object. Put source, date, URL, or other context there.
Other article fields are ignored. Full article text is not sent.
Each summary is limited to 4000 characters. The full input JSON is limited to 60000 characters.
Input is not truncated. `build_messages(articles)` builds the English prompt and source data.
`synthesize_topic(articles, client=None)` calls the AI client.
`parse_response(response)` validates the reply and returns only title and summary.

The prompt asks for the strongest supported shared topic and a short, clear topic phrase.
It asks for about four to five natural sentences, using fewer when evidence is limited.
It merges repeated facts, keeps uncertainty and disagreements, and avoids invented trends.
Unrelated items must not be forced into a common story. A single article uses its own topic.
Output uses the main language of the supplied summaries.
If the model reports no shared topic, TopicSynthesisError is raised.
The title is limited to 120 characters and the summary to 4000 characters.
Incomplete, refused, or invalid replies also raise TopicSynthesisError.
Bad input raises ValueError. API failures remain AIError. No extra retries are added here.
The module writes no files. Format validation cannot prove factual quality.

## Topic synthesis smoke test

Run from the repository root with your NVIDIA API key exported:

```bash
source .env
python -B -m integration.smoke_topic_synthesis
```

This calls the real AI client with four related fictional article summaries.
It prints the returned title and summary as JSON. It uses NVIDIA API quota.
Use your own UTF-8 JSON article list with:

```bash
python -B -m integration.smoke_topic_synthesis --articles /path/to/summaries.json --timeout 120
python -B -m unittest discover -s test -p 'test*topic_synthesis.py' -v
```

Use --articles - for stdin. Direct run: `python -B integration/smoke_topic_synthesis.py`.
Defaults are 120 seconds per attempt, 2048 output tokens, and zero retries.
Use --max-tokens and --retries to change them. The script does not load .env itself.
The bundled sample is in `integration/topic_synthesis_sample.json`.
Check that the output connects the bus trial, chargers, training, and evaluation plans,
without claiming that expansion is approved or that winter performance is known.
PASS means the response has valid title and summary fields; review its actual facts.
Exit codes: 0 success, 1 request failure, 2 input/configuration error, 3 invalid response, 130 cancelled.
Offline unit tests use mocked AI replies and do not spend API quota.

## Article input and SQLite

`fuego/article_input.py` reads an input article, calls article_processing, and stores the result.
It uses the standard library SQLite module. No extra package is needed.

```python
from fuego.article_input import ArticleInput

store = ArticleInput()  # db/articles.sqlite3 at the project root.
status = store.ingest(record)
article = store.get(record["Record_ID"])
store.set_buckets(record["Record_ID"], [
    {"bucket_id": "technology", "similarity": 0.78},
])
```

Input is one JSON object with Record_ID, Publication_Date, and Article_Text.
Record_ID must be a nonempty string. Publication_Date is an integer Unix timestamp in milliseconds.
Source_Name, Title, Article_Link, Tone, People, Organizations, and Themes are optional text fields.
Missing text fields become empty strings. Article_Text must be nonempty.
Articles longer than 10000 characters are filtered before any AI call or article write.
Exactly 10000 characters is allowed. No text is truncated.

`prepare_article(record)` splits text, metadata, and stored article fields.
Text and metadata stay in memory during processing. They are not saved as raw article content.
The existing article_processing prompt is reused without another prompt or LLM step.
`ingest(record)` returns `{status, id}`. Status is stored, duplicate, or filtered.
An existing ID is not reprocessed or overwritten. Failed processing leaves no article row.
Concurrent callers may both call AI, but the database stores an ID only once.

The articles table stores id, published_at, source, title, url, summary, and semantic_text.
The processor's overview becomes semantic_text. Publication dates keep millisecond units.
The article_buckets table stores article_id, bucket_id, and similarity.
This module does not classify articles. Bucket links start empty and are supplied later.
`set_buckets()` replaces all links in one transaction. Similarities must be finite and in [-1, 1].
Duplicate bucket IDs are rejected. An empty list clears the links.
`get(id)` returns the article with a buckets list, or None if the article is missing.
The backend can use SQLite for other queries. Values are bound as SQL parameters.

Use `ArticleInput(db_path="path/to/articles.sqlite3", client=client)` to set the file or AI client.
Connections are closed after each operation. Invalid input raises ValueError.
AIError, ArticleProcessingError, and sqlite3 errors keep their original types.
Generated database files are ignored by Git.

## Article input smoke test

Run from the repository root with a configured NVIDIA key:

```bash
source .env
python -B -m integration.smoke_article_input
python -B -m unittest discover -s test -p 'test_article_input.py' -v
```

The smoke uses the bundled fictional article by default and makes one real NVIDIA call.
Use `--input /path/to/article.json` for one article object in the documented input format.
Direct run: `python -B integration/smoke_article_input.py`.
Options include --timeout (default 120 seconds) and --max-tokens (default 2048). Retries are off.
It checks long-input filtering, real article_processing output, SQLite reload, duplicate IDs,
a sample bucket relation, row count, and database integrity. The sample bucket score is a
storage test value, not a computed relevance score.
It prints the stored article as JSON. Review the actual summary and semantic text for quality.
The database is temporary and is removed after the test. Your database is not changed.
The runner does not load .env. Exit codes: 0 success, 1 request/database failure,
2 input/configuration error, 3 invalid model response, 130 cancelled.
Offline tests mock the API reply while using the real processor and SQLite.

## Output packaging

`fuego/write_output.py` builds the response shape shown in `output/fuego_output_example.json`.
It reads SQLite in read-only mode and makes no AI call. It does not write or send JSON.

```python
from fuego.write_output import build_output, serialize_output

response = build_output([
    {"id": "technology", "description": "Technology news.",
     "synthesis": synthesis_result},  # title and summary from topic_synthesis.
], db_path="db/articles.sqlite3", request_type="fixed")
json_text = serialize_output(response)
```

Bucket specs need an id and a synthesis object with title and summary.
Description is optional. The synthesis title becomes the output bucket name.
Bucket IDs are used exactly as given; no prefix is added.
Members and similarities come from article_buckets. Store the selected links before packaging.
The module does not select topics, apply a similarity threshold, or run synthesis itself.
A bucket with no stored links has an empty members list.

The default request_type is fixed. query and cluster are also accepted.
The default articles_per_bucket is 10. Members sort by descending similarity, then article ID.
Ranks start at one. Each referenced article is included once, in first-seen order.
Article bucket links include all stored links, even links to buckets outside this response.
A missing member article raises ValueError instead of producing a broken reference.

The schema_version is fuego-response.v1. Published timestamps become UTC ISO strings.
generated_at defaults to the current UTC time; pass an aware datetime for a fixed time.
parameters.bucket_count is the number of supplied bucket specs.
stats.total_articles_considered is the total article count in the SQLite snapshot.
activity.current_count counts all stored links for that bucket, before the output limit.
These are stored counts, not measurements of a time window.
Historical counts and change ratios are null, with status insufficient_data.
Direction uses an insufficient-data message and no related topics. No trend is invented.

SQLite does not store embeddings or original metadata. Their default outputs are [] and {}.
Pass embeddings={article_id: vector} and metadata={article_id: object} to include available data.
Supplied vectors must contain 384 finite numbers. Extras are copied, not modified.
The shortened vectors in the example JSON are illustrative.
serialize_output uses UTF-8-friendly strict JSON and rejects NaN or Infinity.
No optional unique_story_count is emitted because story deduplication is not implemented.
Database errors remain sqlite3 errors. Input errors raise ValueError.

```bash
python -B -m integration.smoke_write_output
python -B -m unittest discover -s test -p 'test_write_output.py' -v
```

Direct run: `python -B integration/smoke_write_output.py`.
The smoke uses two articles, two buckets, sample synthesis results, and a temporary SQLite file.
It prints the JSON response and checks ranks, IDs, deduplication, stats, dates, and JSON round trip.
It needs no API key or extra package. It does not change your database or output examples.
Exit code 0 means success; 1 means failure.

## Fixed buckets

- Politics
- World Affairs
- Economy
- Business
- Technology
- Science
- Health
- Environment
- Energy
- Education
- Crime and Justice
- Transportation
- Housing
- Sports
- Entertainment
- Arts and Culture
- Gaming
- Food
- Travel
- Weather
