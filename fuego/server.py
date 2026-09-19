"""Small HTTP/TCP adapters for trusted teammate integration."""

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import sys
import threading
from uuid import uuid4

from .client import NvidiaClient, PipelineError
from .core import MetadataCore
from .demo import WorkflowDemoClient
from .workflow import Workflow
from .ingest import InputError, MAX_BYTES, check, decode_json
from .output import failure, success
from .service import GdeltService

MAX_FRAME = MAX_BYTES * 2


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def request_id(value=None):
    if value is None:
        return uuid4().hex
    check(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value),
          "request_id must contain 1-80 letters, digits, underscores, dots, or hyphens.")
    return value


class Application:
    def __init__(self, service, token=""):
        self.service = service
        self.token = token
        self.busy = threading.Lock()

    def authorized(self, token):
        return not self.token or (isinstance(token, str) and
                                  hmac.compare_digest(token.encode(), self.token.encode()))

    def process(self, raw, media_type, rid, operation="gdelt"):
        if not self.busy.acquire(blocking=False):
            return 503, failure(rid, "busy", "Another batch is running. Retry later.")
        try:
            return 200, success(rid, self.service.process_operation(operation, raw, media_type))
        except InputError as exc:
            return exc.status, failure(rid, exc.code, str(exc))
        except PipelineError:
            return 502, failure(rid, "model_error", "Nemotron processing failed. Check server configuration or use a smaller batch.")
        except Exception as exc:
            print(f"Request {rid} failed: {type(exc).__name__}", file=sys.stderr)
            return 500, failure(rid, "internal_error", "Unexpected server error.")
        finally:
            self.busy.release()


class BoundedThreads(socketserver.ThreadingMixIn):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        print("Connection closed after an I/O error.", file=sys.stderr)


class HttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass

    def reply(self, status, body):
        data = encode(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("X-Request-ID", body["request_id"])
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def do_GET(self):
        rid = request_id()
        if self.path == "/health":
            self.reply(200, success(rid, {"status": "ok", "api_version": "fuego-api.v1",
                                           "is_demo": self.server.app.service.core.client.mode == "demo"}))
        else:
            self.reply(404, failure(rid, "not_found", "Use POST /v1/gdelt or GET /health."))

    def do_POST(self):
        rid = request_id()
        try:
            rid = request_id(self.headers.get("X-Request-ID"))
            if self.path not in ("/v1/gdelt", "/v1/ingest", "/v1/query", "/v1/summary", "/v1/digest"):
                self.reply(404, failure(rid, "not_found", "Use POST /v1/ingest, /v1/query, /v1/summary, /v1/digest, or /v1/gdelt."))
                return
            authorization = self.headers.get("Authorization", "")
            token = authorization[7:] if authorization.startswith("Bearer ") else ""
            if not self.server.app.authorized(token):
                self.reply(401, failure(rid, "unauthorized", "Provide the server token as Bearer authorization."))
                return
            if self.headers.get("Transfer-Encoding"):
                raise InputError("Chunked uploads are not supported. Send Content-Length.", "invalid_framing", 400)
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,12}", lengths[0]):
                raise InputError("Send exactly one numeric Content-Length.", "invalid_framing", 411)
            length = int(lengths[0])
            if length > MAX_BYTES:
                raise InputError("Payload exceeds 8 MiB.", "payload_too_large", 413)
            media_type = self.headers.get_content_type()
            charset = self.headers.get_content_charset()
            if charset and charset.lower() not in ("utf-8", "utf8"):
                raise InputError("Input must use UTF-8.")
            if self.headers.get("Content-Encoding", "identity") != "identity":
                raise InputError("Compressed uploads are not supported.", "unsupported_media_type", 415)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InputError("Incomplete request body.", "invalid_framing", 400)
            status, body = self.server.app.process(raw, media_type, rid, self.path.rsplit("/", 1)[-1])
            self.reply(status, body)
        except InputError as exc:
            self.reply(exc.status, failure(rid, exc.code, str(exc)))
        except (TimeoutError, socket.timeout):
            self.reply(408, failure(rid, "read_timeout", "Upload timed out."))


class TcpHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(15)
        rid = request_id()
        try:
            line = self.rfile.readline(MAX_FRAME + 1)
            if len(line) > MAX_FRAME:
                raise InputError("TCP frame exceeds 16 MiB.", "payload_too_large", 413)
            if not line.endswith(b"\n"):
                raise InputError("End one JSON frame with a newline.", "invalid_framing", 400)
            try:
                frame = decode_json(line.decode("utf-8"))
            except UnicodeDecodeError:
                raise InputError("Frame must use UTF-8.") from None
            check(isinstance(frame, dict) and {"format", "data"} <= set(frame)
                  and not (set(frame) - {"format", "data", "request_id", "token", "operation"}),
                  "Frame requires format and data; optional fields are request_id, token, and operation.")
            rid = request_id(frame.get("request_id"))
            if not self.server.app.authorized(frame.get("token", "")):
                status, body = 401, failure(rid, "unauthorized", "Provide the server token in the token field.")
            else:
                check(frame.get("operation", "gdelt") in ("gdelt", "ingest", "query", "summary", "digest"), "Unknown operation.")
                check(frame["format"] in ("csv", "json"), "format must be csv or json.")
                if frame["format"] == "csv":
                    check(isinstance(frame["data"], str), "CSV data must be a string.")
                    raw = frame["data"].encode("utf-8")
                else:
                    raw = encode(frame["data"])
                status, body = self.server.app.process(raw, "text/csv" if frame["format"] == "csv"
                                                       else "application/json", rid, frame.get("operation", "gdelt"))
        except InputError as exc:
            status, body = exc.status, failure(rid, exc.code, str(exc))
        except (TimeoutError, socket.timeout):
            status, body = 408, failure(rid, "read_timeout", "Upload timed out.")
        except (ValueError, TypeError, RecursionError):
            status, body = 422, failure(rid, "invalid_input", "Invalid JSON frame.")
        self.wfile.write(encode({**body, "status": status}) + b"\n")


class HttpServer(BoundedThreads, HTTPServer):
    allow_reuse_address = True


class TcpServer(BoundedThreads, socketserver.TCPServer):
    allow_reuse_address = True


def make_server(service, host="127.0.0.1", port=8000, transport="http", token=""):
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError("Set FUEGO_API_TOKEN before binding to a non-loopback interface.")
    if transport not in ("http", "tcp"):
        raise ValueError("transport must be http or tcp.")
    server = (HttpServer if transport == "http" else TcpServer)((host, port),
                                                               HttpHandler if transport == "http" else TcpHandler)
    server.app = Application(service, token)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Store articles and serve on-demand Nemotron digests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--transport", choices=("http", "tcp"), default="http")
    parser.add_argument("--buckets", help="JSON array of bucket definitions; otherwise use sample defaults.")
    parser.add_argument("--demo", action="store_true", help="Use simple offline rules. No model calls.")
    parser.add_argument("--db", default="work/fuego.sqlite3", help="SQLite article store.")
    args = parser.parse_args(argv)
    try:
        buckets = decode_json(Path(args.buckets).read_text(encoding="utf-8")) if args.buckets else None
        client = WorkflowDemoClient() if args.demo else NvidiaClient()
        workflow = Workflow(client, args.db, buckets)
        service = GdeltService(MetadataCore(client, workflow.buckets), workflow)
        with make_server(service, args.host, args.port, args.transport, os.environ.get("FUEGO_API_TOKEN", "")) as server:
            print(f"Listening on {args.transport}://{args.host}:{server.server_address[1]} (demo={args.demo})", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    except (OSError, ValueError) as exc:
        print(f"Cannot start server: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
