"""Read settings, call Nemotron, and return the API response."""

from dataclasses import dataclass, field
import json
import math
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_RESPONSE_BYTES = 2_000_000


class AIError(Exception):
    """The NVIDIA request failed."""


@dataclass(frozen=True)
class AIConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = 20
    max_tokens: int = 16384
    temperature: float = 0
    enable_thinking: bool = False
    max_retries: int = 2

    def __post_init__(self):
        if (not isinstance(self.api_key, str) or not self.api_key.strip()
                or self.api_key == "replace-with-your-key" or any(c.isspace() for c in self.api_key)):
            raise ValueError("Set a valid NVIDIA_API_KEY.")
        if (not isinstance(self.model, str) or not self.model.startswith("nvidia/")
                or "nemotron" not in self.model.lower() or any(c.isspace() for c in self.model)):
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
            thinking = os.environ.get("NVIDIA_ENABLE_THINKING", "false").lower()
            if thinking not in ("true", "false"):
                raise ValueError()
            return cls(
                api_key=os.environ.get("NVIDIA_API_KEY", ""),
                model=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL),
                timeout=float(os.environ.get("NVIDIA_TIMEOUT", "20")) if timeout is None else timeout,
                max_tokens=int(os.environ.get("NVIDIA_MAX_TOKENS", "16384")),
                temperature=float(os.environ.get("NVIDIA_TEMPERATURE", "0")),
                enable_thinking=thinking == "true",
                max_retries=int(os.environ.get("NVIDIA_MAX_RETRIES", "2")) if max_retries is None else max_retries,
            )
        except ValueError:
            raise ValueError("Invalid NVIDIA configuration. Check the API key and settings.") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AIError("NVIDIA returned a redirect.")


class NemotronClient:
    def __init__(self, config=None):
        self.config = AIConfig.from_env() if config is None else config
        if not isinstance(self.config, AIConfig):
            raise TypeError("config must be AIConfig.")
        self._opener = urllib.request.build_opener(_NoRedirect)

    def complete(self, messages):
        """Send text messages and return the raw JSON response."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list.")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError("Each message needs a text content and a system, user, or assistant role.")
        body = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(ENDPOINT, data=body, headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
            })
            try:
                with self._opener.open(request, timeout=self.config.timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise AIError("NVIDIA response is too large.")
                try:
                    result = json.loads(raw)
                except (ValueError, RecursionError):
                    raise AIError("NVIDIA returned invalid JSON.") from None
                if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
                    raise AIError("NVIDIA returned an invalid response.")
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429, 500, 502, 503, 504) or attempt == self.config.max_retries:
                    raise AIError(f"NVIDIA request failed (HTTP {status}).") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.config.max_retries:
                    raise AIError("Cannot reach NVIDIA after the configured attempts.") from None
            time.sleep(2 ** attempt)
