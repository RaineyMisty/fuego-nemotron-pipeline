"""Check input, real article processing, and SQLite storage."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.ai import AIConfig, AIError, NemotronClient
from fuego.article_input import ArticleInput, prepare_article
from fuego.article_processing import ArticleProcessingError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test article input with real NVIDIA processing and temporary SQLite.")
    parser.add_argument("--input", type=Path, help="One UTF-8 JSON article object. Default: fictional sample.")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    try:
        if args.input:
            record = json.loads(args.input.read_text(encoding="utf-8"))
        else:
            record = {"Record_ID": "smoke-article", "Publication_Date": 1789256700000,
                      "Source_Name": "Fictional smoke sample", "Title": "Local transit pilot", "Article_Link": "",
                      "Article_Text": Path(__file__).with_name("article_sample.txt").read_text(encoding="utf-8")}
        if prepare_article(record) is None:
            raise ValueError("The smoke article must not exceed 10000 characters.")
        config = replace(AIConfig.from_env(), timeout=args.timeout, max_tokens=args.max_tokens, max_retries=0)
    except (OSError, ValueError, RecursionError):
        print("FAIL [input/config]: check the JSON article and exported NVIDIA settings.", file=sys.stderr)
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="fuego-article-input-") as directory:
            store = ArticleInput(Path(directory)/"articles.sqlite3", client=NemotronClient(config))
            def check(condition, message):
                if not condition:
                    raise ValueError(message)
            check(store.ingest(dict(record, Article_Text="x"*10001))["status"] == "filtered", "Long input was not filtered.")
            check(store.get(record["Record_ID"]) is None, "Filtered input was stored.")
            print("PASS input: valid input accepted; long input filtered", file=sys.stderr, flush=True)
            print("Calling real article_processing through fuego.ai...", file=sys.stderr, flush=True)
            check(store.ingest(record)["status"] == "stored", "Article was not stored.")
            article = store.get(record["Record_ID"])
            check(bool(article["summary"] and article["semantic_text"]), "Missing processed text.")
            print("PASS article_processing: received summary and semantic text", file=sys.stderr)
            check(store.ingest(record)["status"] == "duplicate", "Duplicate was not skipped.")
            links = [{"bucket_id": "smoke-test-link", "similarity": 0.78}]
            store.set_buckets(record["Record_ID"], links)
            article = ArticleInput(store.db_path).get(record["Record_ID"])
            check(article["buckets"] == links, "Bucket relation was not saved.")
            check(article["published_at"] == record["Publication_Date"], "Publication date changed.")
            with sqlite3.connect(store.db_path) as connection:
                check(connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Database integrity failed.")
                check(connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1, "Wrong row count.")
            print("PASS SQLite: reload, duplicate ID, bucket relation, and integrity", file=sys.stderr)
            print(json.dumps(article, ensure_ascii=False, indent=2))
        return 0
    except ArticleProcessingError as exc:
        print(f"FAIL [response]: {exc}", file=sys.stderr)
        return 3
    except (AIError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("CANCELLED", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
