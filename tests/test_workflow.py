import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fuego.client import PipelineError
from fuego.core import MetadataCore
from fuego.demo import WorkflowDemoClient
from fuego.ingest import InputError
from fuego.server import Application, make_server
from fuego.service import GdeltService
from fuego.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-09-19"


class SpyClient(WorkflowDemoClient):
    def __init__(self):
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(copy.deepcopy(payload))
        return super().complete(system, payload)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "articles.sqlite3"
        self.client = SpyClient()
        self.workflow = Workflow(self.client, self.path)
        self.data = json.loads((ROOT / "examples/ingest.json").read_text())
        self.query = json.loads((ROOT / "examples/query.json").read_text())

    def test_ingest_schemes_then_summary_on_request(self):
        response = self.workflow.ingest(self.data)
        self.assertEqual([p["task"] for p in self.client.calls], ["extract_scheme", "extract_scheme", "route_schemes"])
        self.assertEqual(response["stored_count"], 3)
        self.assertEqual(response["records"][0]["bucket_ids"], ["ai", "education"])
        routed = self.client.calls[-1]["articles"][0]
        self.assertIn("scheme", routed)
        self.assertNotIn("text", routed)
        self.assertEqual(response["records"][2]["scheme"]["summary"], None)
        self.assertEqual(response["records"][2]["scheme_origin"], "source_metadata")
        result = self.workflow.digest(self.query)
        self.assertEqual([p["task"] for p in self.client.calls[-2:]], ["query_buckets", "digest"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["coverage"]["missing_text_ids"], ["demo-tags"])
        self.assertEqual(len(result["summary"]), 2)
        self.assertEqual(len(result["sources"]), 2)
        self.assertTrue(result["is_demo"])
        self.assertEqual(len(self.client.calls[-1]["articles"]), 2)

    def test_reopen_and_multi_bucket_dedup(self):
        self.workflow.ingest(self.data)
        reopened = Workflow(self.client, self.path)
        rows = reopened.store.select(["education", "ai"], DAY)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["article"]["text"], self.data["articles"][0]["text"])
        result = reopened.summarize_buckets({"date": DAY, "bucket_ids": ["ai", "education"]})
        self.assertEqual(result["coverage"]["usable_articles"], 2)

    def test_metadata_only_never_becomes_news(self):
        self.workflow.ingest({"articles": [self.data["articles"][2]]})
        before = len(self.client.calls)
        result = self.workflow.summarize_buckets({"date": DAY, "bucket_ids": ["education"]})
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(result["status"], "needs_article_text")
        self.assertEqual(result["summary"], [])
        self.assertEqual(result["sources"], [])

    def test_date_and_no_matches(self):
        self.workflow.ingest(self.data)
        result = self.workflow.digest({"date": DAY, "query": "underwater basket weaving"})
        self.assertEqual(result["status"], "no_matching_buckets")
        result = self.workflow.digest({"date": "2026-09-18", "query": "AI"})
        self.assertEqual(result["status"], "no_data")
        self.assertNotIn("digest", [p["task"] for p in self.client.calls])

    def test_upsert_replaces_membership(self):
        self.workflow.ingest(self.data)
        updated = copy.deepcopy(self.data["articles"][0])
        updated["text"] = "In this fictional demo, a bus route changes."
        self.workflow.ingest({"articles": [updated]})
        self.assertEqual(self.workflow.store.select(["ai"], DAY), [])
        self.assertEqual(len(self.workflow.store.select(["transport"], DAY)), 1)

    def test_failure_leaves_store_unchanged(self):
        self.workflow.ingest(self.data)
        before = self.workflow.store.select(["education"], DAY)
        changed = copy.deepcopy(self.data)
        changed["articles"][0]["text"] = "A bus runs."
        with patch.object(self.client, "complete", side_effect=PipelineError("failed")):
            with self.assertRaises(PipelineError):
                self.workflow.ingest(changed)
        self.assertEqual(self.workflow.store.select(["education"], DAY), before)

    def test_id_conflict_rolls_back_whole_batch(self):
        self.workflow.ingest(self.data)
        changed = copy.deepcopy(self.data)
        changed["articles"][0]["text"] = "A bus runs."
        changed["articles"][1]["url"] = "https://example.com/wrong"
        with self.assertRaises(InputError):
            self.workflow.ingest(changed)
        self.assertEqual(len(self.workflow.store.select(["ai"], DAY)), 1)
        with self.assertRaises(InputError):
            self.workflow.ingest({"articles": [{"id": "demo-ai", "url": self.data["articles"][0]["url"],
                                                 "observed_at": DAY + "T12:00:00Z", "gdelt": {"theme_codes": ["AI"]}}]})

    def test_store_cannot_mix_demo_or_bucket_definitions(self):
        other = SpyClient()
        other.mode = "nvidia"
        with self.assertRaises(InputError):
            Workflow(other, self.path)
        with self.assertRaises(InputError):
            Workflow(self.client, self.path, [{"id":"new", "name":"New", "description":"New topic"}])

    def test_invalid_query_bucket_and_extraction(self):
        real = self.client.complete
        def invalid(system, payload):
            result = real(system, payload)
            if payload["task"] == "extract_scheme":
                result["evidence"][0]["quote"] = "invented"
            return result
        with patch.object(self.client, "complete", side_effect=invalid):
            with self.assertRaises(PipelineError):
                self.workflow.ingest(self.data)
        self.assertEqual(self.workflow.store.select(["education"], DAY), [])
        with patch.object(self.client, "complete", return_value={"bucket_ids": ["fake"]}):
            with self.assertRaises(PipelineError):
                self.workflow.query(self.query)

    def test_wrong_digest_source_and_quote(self):
        self.workflow.ingest(self.data)
        for aid, quote in [("missing", "text"), ("demo-ai", "invented")]:
            result = {"bullets": [{"text": "A claim", "evidence": [{"article_id": aid, "quote": quote}]}]}
            with patch.object(self.client, "complete", return_value=result):
                with self.assertRaises(PipelineError):
                    self.workflow.summarize_buckets({"date": DAY, "bucket_ids": ["ai"]})

    def test_duplicate_urls_are_reported(self):
        duplicate = dict(self.data["articles"][0], id="duplicate")
        self.data["articles"].append(duplicate)
        self.workflow.ingest(self.data)
        result = self.workflow.digest(self.query)
        self.assertEqual(result["coverage"]["duplicate_url_ids"], ["duplicate"])
        self.assertEqual(result["coverage"]["usable_articles"], 2)

    def test_invalid_inputs_are_422_not_model_errors(self):
        service = GdeltService(MetadataCore(self.client), self.workflow)
        app = Application(service)
        for op, value in [("query", {"date": "bad", "query":"AI"}),
                          ("summary", {"date":DAY,"bucket_ids":["fake"]}),
                          ("ingest", {"articles": []}), ("ingest", {"articles":[{}]})]:
            status, response = app.process(json.dumps(value).encode(), "application/json", "test", op)
            self.assertEqual(status, 422)
            self.assertFalse(response["ok"])
        self.assertEqual(self.client.calls, [])

    def test_no_relevant_content(self):
        self.workflow.ingest(self.data)
        with patch.object(self.client, "complete", return_value={"bullets": []}):
            result = self.workflow.summarize_buckets({"date":DAY, "bucket_ids":["ai"]})
        self.assertEqual(result["status"], "no_relevant_content")

    def test_gdelt_route_persists_and_can_be_enriched(self):
        service = GdeltService(MetadataCore(self.client), self.workflow)
        row = {"Record_ID": "20260919120000-1", "Article_Link": "https://example.com/gdelt",
               "Theme": "EDUCATION,1", "Tone": "1,2,1,3,20,0,100"}
        result = service.process(json.dumps({"records":[row]}).encode(), "application/json")
        self.assertEqual(result["schema_version"], "gdelt-ingest.v1")
        self.assertEqual(result["stored_count"], 1)
        self.assertEqual(self.workflow.summarize_buckets({"date":DAY, "bucket_ids":["education"]})["status"], "needs_article_text")
        self.workflow.ingest({"articles":[{"id":row["Record_ID"], "url":row["Article_Link"],
                            "observed_at":"2026-09-19T12:00:00Z", "text":"A school opens in this fictional demo."}]})
        result = self.workflow.summarize_buckets({"date":DAY, "bucket_ids":["education"]})
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["sources"][0]["time_basis"], "observed_at")

    def test_cli_two_stage(self):
        for op, file in [("ingest","ingest.json"),("query","query.json"),("summary","summary-request.json"),("digest","query.json")]:
            output = Path(self.temp.name) / (op + ".json")
            result = subprocess.run([sys.executable,"-m","fuego",op,"--input",str(ROOT / "examples" / file),
                                     "--output",str(output),"--db",str(self.path),"--demo"],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(output.read_text())["is_demo"])


class WorkflowSocketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = SpyClient()
        self.workflow = Workflow(self.client, Path(self.temp.name) / "store.sqlite3")
        self.service = GdeltService(MetadataCore(self.client), self.workflow)

    def start(self, transport):
        import threading
        server = make_server(self.service, port=0, transport=transport, token="test-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.addCleanup(stop)
        return server.server_address[1]

    def test_http_full_workflow_and_auth(self):
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        port = self.start("http")
        def post(operation, value, auth=True):
            headers = {"Content-Type":"application/json"}
            if auth:
                headers["Authorization"] = "Bearer test-token"
            request = Request(f"http://127.0.0.1:{port}/v1/{operation}",
                              data=json.dumps(value).encode(), headers=headers)
            try:
                with urlopen(request, timeout=3) as response:
                    return response.status, json.load(response)
            except HTTPError as exc:
                with exc:
                    return exc.code, json.load(exc)
        data = json.loads((ROOT / "examples/ingest.json").read_text())
        self.assertEqual(post("ingest", data, False)[0], 401)
        self.assertEqual(post("ingest", data)[0], 200)
        query = {"date": DAY, "query":"AI and robotics"}
        status, result = post("query", query)
        self.assertEqual(status, 200)
        self.assertEqual(result["data"]["bucket_ids"], ["ai", "robotics"])
        status, result = post("summary", {"date":DAY,"bucket_ids":["ai"]})
        self.assertEqual(result["data"]["status"], "ready")
        self.assertEqual(post("digest", query)[1]["data"]["status"], "ready")
        self.assertEqual(post("summary", {"date":DAY,"bucket_ids":["fake"]})[0], 422)
        with patch.object(self.client, "complete", side_effect=PipelineError("private-message")):
            status, result = post("digest", query)
        self.assertEqual(status, 502)
        self.assertNotIn("private-message", json.dumps(result))

    def test_tcp_new_operations_and_old_gdelt_frame(self):
        import socket
        port = self.start("tcp")
        def send(value):
            with socket.create_connection(("127.0.0.1",port),timeout=3) as sock:
                sock.sendall(json.dumps({"token":"test-token", **value}).encode()+b"\n")
                with sock.makefile("rb") as stream:
                    return json.loads(stream.readline())
        data = json.loads((ROOT / "examples/ingest.json").read_text())
        self.assertEqual(send({"operation":"ingest","format":"json","data":data})["status"],200)
        result = send({"operation":"digest","format":"json","data":{"date":DAY,"query":"AI"}})
        self.assertEqual(result["data"]["status"], "ready")
        row = {"Record_ID":"20260919140000-2", "Article_Link":"https://example.com/new", "Theme":"", "Tone":"1,2,1,3,20,0,100"}
        result = send({"format":"json","data":{"records":[row]}})
        self.assertEqual(result["data"]["stored_count"],1)
        self.assertEqual(result["data"]["stats"]["unmatched_record_ids"],[row["Record_ID"]])
