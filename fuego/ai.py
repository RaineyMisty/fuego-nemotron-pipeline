"""Call NVIDIA or Ollama and return a shared response format."""

from dataclasses import dataclass, field
import json
import math
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
OLLAMA_MODEL = "qwen3:0.6b"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_RESPONSE_BYTES = 2_000_000


class AIError(Exception):
    """The AI request failed."""


@dataclass(frozen=True)
class AIConfig:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = 20
    max_tokens: int = 16384
    temperature: float = 0
    enable_thinking: bool = False
    max_retries: int = 2

    provider: str = "nvidia"
    base_url: str = OLLAMA_BASE_URL

    @property
    def endpoint(self):
        return ENDPOINT if self.provider == "nvidia" else self.base_url.rstrip("/") + "/api/chat"

    def __post_init__(self):
        if self.provider not in ("nvidia", "ollama"):
            raise ValueError("AI_PROVIDER must be nvidia or ollama.")
        if self.provider == "ollama":
            if not isinstance(self.base_url, str) or any(c.isspace() for c in self.base_url):
                raise ValueError("Set a valid OLLAMA_BASE_URL.")
            url = urlsplit(self.base_url)
            if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                    or url.password is not None or url.path not in ("", "/") or url.query or url.fragment):
                raise ValueError("OLLAMA_BASE_URL must be a server URL, such as http://localhost:11434.")
            if url.port == 0:
                raise ValueError("Invalid Ollama port.")
        if self.provider == "nvidia" and (not isinstance(self.api_key, str) or not self.api_key.strip()
                or self.api_key == "replace-with-your-key" or any(c.isspace() for c in self.api_key)):
            raise ValueError("Set a valid NVIDIA_API_KEY.")
        if not isinstance(self.model, str) or not self.model.strip() or any(c.isspace() for c in self.model):
            raise ValueError("Set a nonempty model name.")
        if self.provider == "nvidia" and (not self.model.startswith("nvidia/")
                or "nemotron" not in self.model.lower()):
            raise ValueError("NVIDIA_MODEL must name an NVIDIA Nemotron model.")
        for name, value, low, high in (("timeout", self.timeout, 0, 600),
                                        ("temperature", self.temperature, 0, 2)):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < low or value > high or (name == "timeout" and value == 0)):
                raise ValueError("Invalid " + name + ".")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5.")
        if type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean.")

    @classmethod
    def from_env(cls, *, timeout=None, max_retries=None):
        try:
            provider = os.environ.get("AI_PROVIDER", "nvidia").strip().lower()
            prefix = "OLLAMA" if provider == "ollama" else "NVIDIA"
            thinking = os.environ.get(prefix + "_ENABLE_THINKING", "false").lower()
            if thinking not in ("true", "false"):
                raise ValueError()
            return cls(
                api_key=os.environ.get("NVIDIA_API_KEY", "") if provider == "nvidia" else "",
                provider=provider,
                base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
                model=os.environ.get(prefix + "_MODEL", OLLAMA_MODEL if provider == "ollama" else DEFAULT_MODEL),
                timeout=float(os.environ.get(prefix + "_TIMEOUT", "20" if provider == "ollama" else "20")) if timeout is None else timeout,
                max_tokens=int(os.environ.get(prefix + "_MAX_TOKENS", "16384")),
                temperature=float(os.environ.get(prefix + "_TEMPERATURE", "0")),
                enable_thinking=thinking == "true",
                max_retries=int(os.environ.get(prefix + "_MAX_RETRIES", "0" if provider == "ollama" else "2")) if max_retries is None else max_retries,
            )
        except ValueError:
            raise ValueError("Invalid AI configuration. Check AI_PROVIDER and the provider settings.") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AIError("AI server returned a redirect.")


class NemotronClient:
    def __init__(self, config=None):
        self.config = AIConfig.from_env() if config is None else config
        if not isinstance(self.config, AIConfig):
            raise TypeError("config must be AIConfig.")
        self._opener = urllib.request.build_opener(_NoRedirect)

    def complete(self, messages, *, response_schema=None):
        """Send text messages and return a choices response."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list.")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError("Each message needs a text content and a system, user, or assistant role.")
        if response_schema is not None and (self.config.provider != "ollama" or not isinstance(response_schema, dict)):
            raise ValueError("response_schema needs Ollama and a JSON schema object.")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        label = "NVIDIA" if self.config.provider == "nvidia" else "Ollama"
        if self.config.provider == "ollama":
            payload = {"model": self.config.model, "messages": messages, "stream": False,
                       "think": self.config.enable_thinking,
                       "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens}}
        else:
            headers["Authorization"] = "Bearer " + self.config.api_key
        if response_schema is not None:
            payload["format"] = response_schema
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(self.config.endpoint, data=body, headers=headers)
            try:
                with self._opener.open(request, timeout=self.config.timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise AIError(f"{label} response is too large.")
                try:
                    result = json.loads(raw)
                except (ValueError, RecursionError):
                    raise AIError(f"{label} returned invalid JSON.") from None
                if self.config.provider == "ollama":
                    result = _ollama_response(result)
                if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
                    raise AIError(f"{label} returned an invalid response.")
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429, 500, 502, 503, 504) or attempt == self.config.max_retries:
                    raise AIError(f"{label} request failed (HTTP {status}).") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.config.max_retries:
                    raise AIError(f"Cannot reach {label} after the configured attempts.") from None
            time.sleep(2 ** attempt)


def _ollama_response(result):
    """Keep the text and stop reason in the shared response format."""
    if not isinstance(result, dict) or result.get("error") or result.get("done") is not True:
        raise AIError("Ollama returned an invalid response.")
    message = result.get("message")
    reason = result.get("done_reason")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str) or not isinstance(reason, str) or not reason):
        raise AIError("Ollama returned an invalid response.")
    return {"model": result.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": reason}]}
