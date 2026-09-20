# Fuego AI

The AI client in `fuego/ai.py` calls NVIDIA Nemotron.
The article processor in `fuego/article_processing.py` extracts keywords and a reader summary.
Neither module stores articles.
The files in `prompt/` describe future work. They are read-only.

Use Python 3.11 or newer. No extra packages are needed.

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
