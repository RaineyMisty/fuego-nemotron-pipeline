"""Run the pipeline from a shell."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .client import NvidiaClient, PipelineError
from .demo import WorkflowDemoClient
from .workflow import Workflow
from .ingest import InputError, decode_json


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return decode_json(stream.read())


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         delete=False) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, target)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from .server import main as serve
        return serve(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Build a daily news feed with NVIDIA Nemotron.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Run HTTP or TCP. Use serve --help.")
    for name in ("ingest", "query", "summary", "digest"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--db", default="work/fuego.sqlite3")
        command.add_argument("--buckets", help="JSON array of fixed bucket definitions.")
        command.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    try:
        client = WorkflowDemoClient() if args.demo else NvidiaClient()
        buckets = read_json(args.buckets) if args.buckets else None
        workflow = Workflow(client, args.db, buckets)
        operations = {"ingest": workflow.ingest, "query": workflow.query,
                      "summary": workflow.summarize_buckets, "digest": workflow.digest}
        result = operations[args.command](read_json(args.input))
        write_json(args.output, result)
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, (PipelineError, InputError)):
            message = str(exc)
        else:
            message = "Cannot read or write the JSON files. Check the paths and data shape."
        print("Error: " + message, file=sys.stderr)
        return 1
    print("Saved " + args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
