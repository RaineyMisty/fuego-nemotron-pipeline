# Fuego AI

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
