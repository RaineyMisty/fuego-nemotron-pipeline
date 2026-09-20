"""Run offline jobs, export feeds, or serve local HTTP requests."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

from fuego.pipeline import Pipeline, PipelineConfig
from fuego.write_output import serialize_output

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(serialize_output(value)+"\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def make_server(pipeline, host="127.0.0.1", port=8000):
    wake, stop = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, payload):
            body = serialize_output(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def handle_request(self):
            try:
                if self.command == "GET":
                    if self.path in ("/health", "/jobs"):
                        self.reply(200, pipeline.status())
                    elif self.path in ("/fixed", "/clusters"):
                        self.reply(200, pipeline.feed("fixed" if self.path == "/fixed" else "cluster"))
                    else:
                        self.reply(404, {"error": "Unknown route."})
                    return
                if self.path not in ("/ingest", "/query"):
                    self.reply(404, {"error": "Unknown route."})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000:
                    self.reply(413, {"error": "Request body must have 1-20000000 bytes."})
                    return
                body = json.loads(self.rfile.read(length))
                if self.path == "/ingest":
                    result = pipeline.enqueue(body)
                    wake.set()
                    self.reply(202, result)
                else:
                    if not isinstance(body, dict) or set(body)-{"topic"}:
                        raise ValueError("Query body needs only topic.")
                    self.reply(200, pipeline.query(body.get("topic")))
            except (ValueError, TypeError, UnicodeError):
                self.reply(400, {"error": "Invalid request data."})
            except LookupError:
                self.reply(503, {"error": "Feed is not ready."})
            except Exception:
                self.reply(503, {"error": "Processing unavailable. Inspect local job status."})

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True

    def worker():
        while not stop.is_set():
            wake.wait(5)
            wake.clear()
            if stop.is_set():
                break
            try:
                status = pipeline.status()
                if any(job["status"] == "pending" for job in status["jobs"]):
                    pipeline.run_pending()
                elif status["feeds_stale"]:
                    pipeline.refresh()
            except Exception as exc:
                print(f"Offline refresh failed: {type(exc).__name__}. Use the refresh command to retry.", file=sys.stderr)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    wake.set()
    server.pipeline_stop = stop
    server.pipeline_wake = wake
    server.pipeline_worker = thread
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local Fuego news service.")
    parser.add_argument("--state", type=Path, help="State directory. Defaults to work/pipeline-live or work/pipeline-demo.")
    parser.add_argument("--mock-ai", action="store_true", help="Use labeled fake AI replies for testing.")
    parser.add_argument("--mock-query", help="Default test topic when a query omits its topic.")
    parser.add_argument("--model-cache", default=str(ROOT/"work"/"models"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=3)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Queue a JSON article list. Processing is separate.")
    ingest.add_argument("input", type=Path)
    work = commands.add_parser("work", help="Drain queued jobs and refresh prepared feeds.")
    work.add_argument("--retry-failed", action="store_true")
    refresh = commands.add_parser("refresh", help="Refresh feeds from indexed articles.")
    refresh.add_argument("--force-clusters", action="store_true")
    refresh.add_argument("--as-of", type=int, help="Publication window endpoint in Unix milliseconds.")
    commands.add_parser("status")
    export = commands.add_parser("export")
    export.add_argument("kind", choices=("fixed", "cluster", "query"))
    export.add_argument("--topic")
    export.add_argument("--output", required=True, type=Path)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    state = args.state or ROOT/"work"/("pipeline-demo" if args.mock_ai else "pipeline-live")
    try:
        pipeline = Pipeline(PipelineConfig(state, args.mock_ai, args.mock_query, args.model_cache,
                                           args.local_files_only, args.min_score, args.clusters, args.attempts))
        if args.command == "ingest":
            print(serialize_output(pipeline.enqueue(json.loads(args.input.read_text(encoding="utf-8")))))
        elif args.command == "work":
            if args.retry_failed:
                pipeline.retry_failed()
            report = pipeline.run_pending()
            print(serialize_output(report))
            return 1 if any(j["status"] == "failed" for j in report["jobs"]) else 0
        elif args.command == "refresh":
            pipeline.refresh(force_clusters=args.force_clusters, as_of=args.as_of)
            print("Prepared feeds refreshed.")
        elif args.command == "status":
            print(serialize_output(pipeline.status()))
        elif args.command == "export":
            response = pipeline.query(args.topic) if args.kind == "query" else pipeline.feed(args.kind)
            write_json(args.output, response)
            print(str(args.output.resolve()))
        else:
            server = make_server(pipeline, args.host, args.port)
            print(f"Fuego listening on http://{args.host}:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            finally:
                server.pipeline_stop.set()
                server.pipeline_wake.set()
                server.server_close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
