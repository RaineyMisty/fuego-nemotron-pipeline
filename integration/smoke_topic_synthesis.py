"""Send related article summaries to real NVIDIA Nemotron through fuego.ai."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.ai import AIConfig, AIError, NemotronClient
from fuego.topic_synthesis import TopicSynthesisError, build_messages, synthesize_topic

SAMPLE = Path(__file__).with_name("topic_synthesis_sample.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthesize article summaries with real NVIDIA Nemotron.")
    parser.add_argument("--articles", type=Path, default=SAMPLE, help="JSON article list. Default: fictional sample. Use - for stdin.")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        text = sys.stdin.read() if str(args.articles) == "-" else args.articles.read_text(encoding="utf-8")
        articles = json.loads(text)
        build_messages(articles)
    except (OSError, ValueError, RecursionError):
        print("FAIL [input]: provide a UTF-8 JSON list of one to ten article summaries and optional metadata.", file=sys.stderr)
        return 2
    try:
        config = replace(AIConfig.from_env(), timeout=args.timeout,
                         max_tokens=args.max_tokens, max_retries=args.retries)
    except ValueError:
        print("FAIL [config]: check NVIDIA settings and load the key with source .env.", file=sys.stderr)
        return 2
    print(f"Model: {config.model}; summaries: {len(articles)}; timeout: {config.timeout}s", file=sys.stderr, flush=True)
    print("Sending summaries and metadata to NVIDIA through fuego.ai...", file=sys.stderr, flush=True)
    start = time.monotonic()
    try:
        result = synthesize_topic(articles, client=NemotronClient(config))
    except TopicSynthesisError as exc:
        print(f"FAIL [response]: {exc}", file=sys.stderr)
        return 3
    except (AIError, OSError) as exc:
        print(f"FAIL [request]: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("CANCELLED", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS in {time.monotonic()-start:.2f}s: received a title and summary. Review factual quality.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
