# Fuego AI

This version only calls NVIDIA Nemotron.
The implementation is in `fuego/ai.py`. It has no news processing or storage.
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
