import csv
import io
import json
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fuego.client import PipelineError
from fuego.core import MetadataCore, MetadataDemoClient
from fuego.ingest import InputError, MAX_BYTES, normalize_rows, parse_input
from fuego.server import Application, make_server
from fuego.service import GdeltService

ROW = {"Record_ID": "20260918134500-675", "Article_Link": "https://example.com/a",
       "Theme": "ECONOMY,456", "Tone": "1,2,1,3,20,0,339"}
BUCKETS = [{"id": "economy", "name": "Economy", "description": "Business and economy."}]


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(ROW))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class RoutingClient:
    mode = "test"
    model = "test-no-model"

    def __init__(self):
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(payload)
        return {"assignments": [{"article_id": row["id"], "bucket_ids": ["economy"]}
                                for row in payload["articles"]]}


def service(client=None):
    return GdeltService(MetadataCore(client or RoutingClient(), BUCKETS))


class IngestionTests(unittest.TestCase):
    def test_csv_bom_json_parity_and_merge(self):
        rows = [ROW, dict(ROW, Theme="ECONOMY,999"), ROW, dict(ROW, Theme="")]
        csv_batch = parse_input(b"\xef\xbb\xbf" + csv_bytes(rows), "text/csv")
        json_batch = parse_input(json.dumps({"records": rows}).encode(), "application/json")
        self.assertEqual(csv_batch, json_batch)
        self.assertEqual(len(csv_batch.records), 1)
        self.assertEqual(csv_batch.records[0].theme_codes, ["ECONOMY"])
        self.assertEqual(csv_batch.duplicate_theme_rows, 1)
        self.assertEqual(csv_batch.blank_theme_rows, 1)
        self.assertEqual(csv_batch.records[0].observed_at, "2026-09-18T13:45:00+00:00")

    def test_conflicting_record_fails(self):
        for change in [{"Tone": "2,3,1,4,20,0,339"}, {"Article_Link": "https://example.com/other"}]:
            with self.subTest(change=change), self.assertRaises(InputError):
                normalize_rows([ROW, dict(ROW, **change)])

    def test_invalid_fields(self):
        for change in [{"Tone": "NaN,2,1,3,20,0,339"}, {"Tone": "1,2"},
                       {"Tone": "1,2,1,3,20,0,3.5"}, {"Theme": "CODE,bad"},
                       {"Theme": None}, {"Record_ID": "20261301120000-1"},
                       {"Article_Link": "file:///etc/passwd"},
                       {"Article_Link": "https://user:pass@example.com/a"},
                       {"Article_Link": "https://example.com:bad/a"}]:
            with self.subTest(change=change), self.assertRaises(InputError):
                normalize_rows([dict(ROW, **change)])

    def test_empty_and_malformed_input(self):
        for body, media in [(b"", "text/csv"), (csv_bytes([]), "text/csv"),
                            (b'{"records":[],"records":[]}', "application/json"),
                            (b'{"records": NaN}', "application/json"),
                            (b'{"records": [', "application/json"),
                            (b"\xff", "text/csv"), (b"[]", "application/json"),
                            (b"Record_ID,Article_Link,Theme,Tone\n1,2,3,4,5\n", "text/csv")]:
            with self.subTest(body=body), self.assertRaises(InputError):
                parse_input(body, media)

    def test_resource_limits(self):
        with self.assertRaises(InputError) as raised:
            parse_input(b"a" * (MAX_BYTES + 1), "text/csv")
        self.assertEqual(raised.exception.status, 413)
        with patch("fuego.ingest.MAX_ROWS", 1), self.assertRaises(InputError):
            normalize_rows([ROW, ROW])
        with patch("fuego.ingest.MAX_RECORDS", 1), self.assertRaises(InputError):
            normalize_rows([ROW, dict(ROW, Record_ID="20260918134500-676")])

    def test_dedup_counts_and_tone_are_record_weighted(self):
        second = dict(ROW, Record_ID="20260917134500-10", Article_Link="https://example.com/b",
                      Tone="-5,1,6,7,20,0,99")
        result = service().process(csv_bytes([ROW] * 10 + [second]))
        self.assertEqual(result["stats"]["input_rows"], 11)
        self.assertEqual(result["stats"]["record_count"], 2)
        self.assertEqual(result["stats"]["mean_tone"], -2)
        self.assertEqual(result["theme_counts"], [{"theme": "ECONOMY", "record_count": 2}])
        self.assertEqual(result["buckets"][0]["record_count"], 2)
        self.assertEqual(len(result["daily_metrics"]), 2)
        self.assertEqual(result["analysis_type"], "metadata_only")

    def test_same_url_different_records_not_silently_merged(self):
        result = service().process(csv_bytes([ROW, dict(ROW, Record_ID="20260917134500-10")]))
        self.assertEqual(result["stats"]["record_count"], 2)
        self.assertEqual(result["stats"]["unique_url_count"], 1)

    def test_blank_theme_skips_model_but_keeps_record(self):
        client = RoutingClient()
        result = service(client).process(csv_bytes([dict(ROW, Theme="")]))
        self.assertEqual(client.calls, [])
        self.assertEqual(result["stats"]["records_without_themes"], 1)
        self.assertEqual(result["stats"]["unmatched_record_ids"], [ROW["Record_ID"]])
        self.assertIsNone(result["buckets"][0]["mean_tone"])

    def test_core_batches_and_sends_only_metadata(self):
        client = RoutingClient()
        service(client).process(csv_bytes([dict(ROW, Record_ID=f"20260918134500-{i}") for i in range(19)]))
        self.assertEqual([len(p["articles"]) for p in client.calls], [8, 8, 3])
        self.assertEqual(set(client.calls[0]["articles"][0]), {"id", "theme_codes"})

    def test_invalid_model_routes_retry_then_fail(self):
        client = RoutingClient()
        with patch.object(client, "complete", return_value={"assignments": []}) as call:
            with self.assertRaises(PipelineError):
                service(client).process(csv_bytes([ROW]))
            self.assertEqual(call.call_count, 2)

    def test_model_failure_is_not_demo_or_success(self):
        client = RoutingClient()
        with patch.object(client, "complete", side_effect=PipelineError("secret")):
            status, result = Application(service(client)).process(csv_bytes([ROW]), "text/csv", "test")
        self.assertEqual(status, 502)
        self.assertFalse(result["ok"])
        self.assertNotIn("secret", json.dumps(result))

    def test_busy(self):
        app = Application(service())
        with app.busy:
            status, result = app.process(csv_bytes([ROW]), "text/csv", "test")
        self.assertEqual(status, 503)
        self.assertEqual(result["error"]["code"], "busy")

    def test_demo_is_explicit(self):
        result = service(MetadataDemoClient()).process(csv_bytes([ROW]))
        self.assertTrue(result["is_demo"])
        self.assertEqual(result["stats"]["unmatched_record_ids"], [ROW["Record_ID"]])


class SocketTests(unittest.TestCase):
    def start(self, transport="http", token=""):
        server = make_server(service(), port=0, transport=transport, token=token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(stop)
        return server.server_address[1]

    def http(self, port, body, media="text/csv", headers=None, path="/v1/gdelt"):
        request = Request(f"http://127.0.0.1:{port}{path}", data=body,
                          headers={"Content-Type": media, **(headers or {})})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            with exc:
                return exc.code, json.load(exc)

    def test_http_csv_json_health_and_error(self):
        port = self.start()
        status, result = self.http(port, csv_bytes([ROW, ROW]), headers={"X-Request-ID": "teammate-1"})
        self.assertEqual(status, 200)
        self.assertEqual(result["request_id"], "teammate-1")
        self.assertEqual(result["data"]["stats"]["record_count"], 1)
        self.assertEqual(self.http(port, json.dumps({"records": [ROW]}).encode(), "application/json")[0], 200)
        self.assertEqual(self.http(port, b"bad")[0], 422)
        self.assertEqual(self.http(port, b"bad", "application/xml")[0], 415)
        self.assertEqual(self.http(port, b"bad", path="/wrong")[0], 404)
        with urlopen(f"http://127.0.0.1:{port}/health") as response:
            self.assertTrue(json.load(response)["ok"])

    def test_http_token(self):
        port = self.start(token="test-token")
        self.assertEqual(self.http(port, csv_bytes([ROW]))[0], 401)
        self.assertEqual(self.http(port, csv_bytes([ROW]), headers={"Authorization": "Bearer test-token"})[0], 200)

    def test_http_oversize_and_truncated_body(self):
        port = self.start()
        for length, body, expected in [(MAX_BYTES + 1, b"", b"413"), (10, b"abc", b"400")]:
            with socket.create_connection(("127.0.0.1", port), timeout=3) as conn:
                headers = f"POST /v1/gdelt HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/csv\r\nContent-Length: {length}\r\n\r\n"
                conn.sendall(headers.encode() + body)
                conn.shutdown(socket.SHUT_WR)
                with conn.makefile("rb") as response:
                    self.assertIn(expected, response.readline())

    def test_tcp_fragmented_frame_and_multiple_rows(self):
        port = self.start("tcp")
        raw = json.dumps({"format": "csv", "data": csv_bytes([ROW] * 3).decode(), "request_id": "tcp-1"}).encode() + b"\n"
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            for start in range(0, len(raw), 7):
                conn.sendall(raw[start:start + 7])
            with conn.makefile("rb") as response:
                result = json.loads(response.readline())
                self.assertEqual(result["status"], 200)
                self.assertEqual(result["request_id"], "tcp-1")
                self.assertEqual(result["data"]["stats"]["record_count"], 1)
                self.assertEqual(response.read(), b"")

    def test_tcp_json_auth_and_malformed_frame(self):
        port = self.start("tcp", token="test-token")
        for frame, status in [(b'{"format":"json","data":{"records":[]}}\n', 401),
                              (b"bad\n", 422),
                              (json.dumps({"format": "json", "data": {"records": [ROW]}, "token": "test-token"}).encode() + b"\n", 200)]:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                conn.sendall(frame)
                with conn.makefile("rb") as response:
                    self.assertEqual(json.loads(response.readline())["status"], status)

    def test_remote_bind_needs_token(self):
        with self.assertRaises(ValueError):
            make_server(service(), host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
