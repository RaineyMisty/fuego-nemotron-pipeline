# Pipeline validation

## Delivered run

Input: input/fuedo_input_test.json (19 articles).
Command: `.venv/bin/python -B -m integration.smoke_pipeline --mock-ai --local-files-only`.

- AI replies: simulated and labeled in every response.
- Query input: New York Knicks basketball.
- Embedding: real FastEmbed ONNX all-MiniLM-L6-v2 from work/models.
- Storage: real SQLite and JSON map in work/pipeline-delivery.
- Processing: 19 done, 0 failed, 0 filtered.
- Fixed output: 20 buckets, 4 nonempty buckets, 12 unique articles.
- Cluster output: 14 nonempty clusters, 19 unique articles, 0 pending cluster articles.
- Query output: 1 bucket, 3 articles.
- JSON round trips and member references passed.

Files: output/pipeline-smoke/fixed.json, cluster.json, query.json, jobs.json.
Identical points can produce fewer than the requested 20 clusters.
The fixed threshold can exclude articles; cluster output includes all assigned articles.
Mock article summaries affect real embeddings. These results validate integration, not AI quality.

## Checks

- 180 unit tests passed, including retries, resume after restart, duplicate IDs, failed embedding recovery,
  request isolation, CLI operations, trends, JSON output, and local HTTP routes.
- Local server smoke passed: queue, worker, status, feeds, query, and error responses.
- A populated server was also tested with the delivered state: /health, /fixed, /clusters,
  and /query returned HTTP 200; the basketball query returned 3 articles.
- Activity and direction smoke tests passed known numeric examples.
- No packages were installed and no prompt files were changed.

## Limits

No live NVIDIA call was run for this delivery. Use live mode and an exported NVIDIA key to verify it.
The sample does not provide enough history for reliable temporal trends. Missing history is explicit.
The server is local and uses one process per state directory. Model jobs are serialized.
It has no authentication, multi-process coordination, or remote deployment setup.
