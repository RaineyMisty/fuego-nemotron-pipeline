"""Run an input file through real embedding and storage, with optional mock AI."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.__main__ import write_json
from fuego.pipeline import Pipeline, PipelineConfig

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the full pipeline and write three JSON responses.")
    parser.add_argument("--input", type=Path, default=ROOT/"input"/"fuego_input_test.json")
    parser.add_argument("--state", type=Path, default=ROOT/"work"/"pipeline-delivery")
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"pipeline-smoke")
    parser.add_argument("--mock-ai", action="store_true")
    parser.add_argument("--mock-query", default="New York Knicks basketball")
    parser.add_argument("--model-cache", default=str(ROOT/"work"/"models"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        pipeline = Pipeline(PipelineConfig(args.state, mock_ai=args.mock_ai, mock_query=args.mock_query,
                                           model_cache=args.model_cache, local_files_only=args.local_files_only))
        print(f"Input: {args.input}; articles: {len(records)}; mock AI: {args.mock_ai}", file=sys.stderr, flush=True)
        pipeline.enqueue(records)
        report = pipeline.run_pending(refresh=False)
        as_of = max(int(r["Publication_Date"]) for r in records)
        pipeline.refresh(as_of=as_of)
        responses = {"fixed": pipeline.feed("fixed"), "cluster": pipeline.feed("cluster"),
                     "query": pipeline.query(as_of=as_of)}
        for kind, response in responses.items():
            text = json.dumps(response, allow_nan=False)
            if json.loads(text) != response:
                raise ValueError("JSON round trip failed.")
            returned = {a["id"] for a in response["articles"]}
            for bucket in response["buckets"]:
                if len(bucket["members"]) > 10 or any(m["article_id"] not in returned for m in bucket["members"]):
                    raise ValueError("Invalid bucket members.")
            write_json(args.output/f"{kind}.json", response)
        write_json(args.state/"reports"/"jobs.json", report)
        if any(j["status"] == "failed" for j in report["jobs"]):
            raise ValueError("Some jobs failed. Inputs remain in the state database.")
        if not responses["cluster"]["articles"]:
            raise ValueError("No article output was produced.")
        print(f"PASS: wrote fixed.json, cluster.json, and query.json to {args.output}; jobs are in {args.state}/reports/jobs.json", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
