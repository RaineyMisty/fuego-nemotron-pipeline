"""Read article text and call the real article processing module."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.ai import AIConfig, AIError, NemotronClient
from fuego.article_processing import ArticleProcessingError, build_messages, process_article


def main(argv=None):
    parser = argparse.ArgumentParser(description="Get article keywords and a summary from the selected AI provider.")
    parser.add_argument("--article", default=Path(__file__).with_name("article_sample.txt"), type=Path, help="UTF-8 text file. Default: bundled sample. Use - for stdin.")
    parser.add_argument("--metadata", type=Path, help="Optional UTF-8 JSON object.")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds per attempt. Default: AIConfig setting.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        article = sys.stdin.read() if str(args.article) == "-" else args.article.read_text(encoding="utf-8")
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None
        build_messages(article, metadata)
    except (OSError, ValueError, RecursionError):
        print("FAIL [input]: provide a nonempty UTF-8 article and valid JSON metadata within the documented limits.", file=sys.stderr)
        return 2
    try:
        config = AIConfig.from_env()
        config = replace(config, timeout=config.timeout if args.timeout is None else args.timeout,
                         max_tokens=args.max_tokens, max_retries=args.retries)
    except ValueError:
        print("FAIL [config]: check AI_PROVIDER and its settings. Ollama needs no NVIDIA key.", file=sys.stderr)
        return 2
    print(f"Model: {config.model}; article characters: {len(article)}; timeout: {config.timeout}s; max_tokens: {config.max_tokens}; retries: {config.max_retries}", file=sys.stderr, flush=True)
    print(f"Provider: {config.provider}; endpoint: {config.endpoint}", file=sys.stderr, flush=True)
    print(f"Sending article text and metadata to {config.provider} through fuego.ai...", file=sys.stderr, flush=True)
    start = time.monotonic()
    try:
        result = process_article(article, metadata, client=NemotronClient(config))
    except ArticleProcessingError as exc:
        print("FAIL [response]: " + str(exc), file=sys.stderr)
        return 3
    except (AIError, OSError) as exc:
        print("FAIL [request]: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("CANCELLED", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS in {time.monotonic() - start:.2f}s: received {len(result['keywords'])} keywords, an overview, and a summary. Review factual quality.", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
