"""Call the NVIDIA hosted API."""

import json
import os
import time
import urllib.error
import urllib.request

MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


class PipelineError(ValueError):
    pass


class ModelOutputError(PipelineError):
    pass


def parse_json(text):
    if not isinstance(text, str):
        raise ModelOutputError("The model did not return text.")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ModelOutputError("The model did not return valid JSON.") from exc
    if not isinstance(value, dict):
        raise ModelOutputError("The model must return a JSON object.")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PipelineError("The API returned a redirect.")


class NvidiaClient:
    mode = "nvidia"

    def __init__(self):
        self.model = os.environ.get("NVIDIA_MODEL", MODEL)
        self.key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not self.key or self.key == "replace-with-your-key":
            raise PipelineError("Set NVIDIA_API_KEY before running the pipeline.")
        if not self.model.startswith("nvidia/") or "nemotron" not in self.model.lower():
            raise PipelineError("NVIDIA_MODEL must be an NVIDIA Nemotron model.")
        self.opener = urllib.request.build_opener(NoRedirect)

    def complete(self, system, payload):
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "/no_think\n" + system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 4096,
            "stream": False,
        }).encode()
        for attempt in range(3):
            request = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            })
            try:
                with self.opener.open(request, timeout=90) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise PipelineError("The API response is too large.")
                result = json.loads(raw)
                choice = result["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ModelOutputError("The model did not finish. Use a smaller input.")
                return parse_json(choice["message"]["content"])
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise PipelineError(f"NVIDIA API failed (HTTP {status}). Check your key, model, and quota.") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise PipelineError("Cannot reach NVIDIA API after three attempts.") from None
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                if isinstance(exc, PipelineError):
                    raise
                raise PipelineError("The API returned an invalid response.") from None
            time.sleep(2 ** attempt)
