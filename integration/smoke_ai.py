"""Make one real request through fuego.ai."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.ai import AIConfig, AIError, ENDPOINT, NemotronClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Call NVIDIA Nemotron and check for a complete text reply.")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds per attempt. Default: NVIDIA_TIMEOUT or 20.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Output token limit. Default: 64.")
    parser.add_argument("--retries", type=int, default=0, help="Retries after the first attempt. Default: 0.")
    args = parser.parse_args(argv)
    try:
        config = AIConfig.from_env()
        config = replace(config, max_tokens=args.max_tokens, max_retries=args.retries,
                         timeout=config.timeout if args.timeout is None else args.timeout)
    except ValueError as exc:
        print("FAIL [config]: " + str(exc), file=sys.stderr)
        print("Load your key first: source .env", file=sys.stderr)
        return 2

    print("Endpoint: " + ENDPOINT, flush=True)
    print("Model: " + config.model, flush=True)
    print(f"Timeout: {config.timeout}s; max_tokens: {config.max_tokens}; retries: {config.max_retries}; thinking: {config.enable_thinking}", flush=True)
    print("Sending a real request through fuego.ai...", flush=True)
    start = time.monotonic()
    try:
        response = NemotronClient(config).complete([
            {"role": "system", "content": "Give a short plain text answer."},
            {"role": "user", "content": "Reply with OK."},
        ])
    except KeyboardInterrupt:
        print("CANCELLED", file=sys.stderr)
        return 130
    except (AIError, OSError) as exc:
        print(f"FAIL [request] after {time.monotonic() - start:.2f}s: {exc}", file=sys.stderr)
        print("For HTTP 401/403, check the key and access. For 404, check the model name.", file=sys.stderr)
        print("For 429, check quota. For connection errors, check DNS, TLS, proxy settings, and timeout.", file=sys.stderr)
        return 1

    print(f"Response received in {time.monotonic() - start:.2f}s.", flush=True)
    try:
        choice = response["choices"][0]
        reason = choice["finish_reason"]
        content = choice["message"]["content"]
        if reason != "stop" or not isinstance(content, str) or not content.strip():
            raise ValueError()
    except (KeyError, IndexError, TypeError, ValueError):
        print("FAIL [response]: expected nonempty text and finish_reason=stop.", file=sys.stderr)
        if isinstance(response, dict) and isinstance(response.get("choices"), list) and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                print("Finish reason: " + str(choice.get("finish_reason")), file=sys.stderr)
        print("If the reply was cut off, raise --max-tokens or disable NVIDIA_ENABLE_THINKING.", file=sys.stderr)
        return 3
    print("Reply: " + content, flush=True)
    print("PASS: fuego.ai returned a complete text reply from NVIDIA.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
