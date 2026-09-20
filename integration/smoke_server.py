"""Check the local HTTP server and background worker without external services."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.__main__ import make_server
from fuego.pipeline import Pipeline, PipelineConfig


def main():
    with tempfile.TemporaryDirectory(prefix="fuego-server-") as folder:
        pipeline = Pipeline(PipelineConfig(Path(folder), mock_ai=True, mock_query="solar energy"))
        server = make_server(pipeline, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def call(path, data=None):
            request = Request(base+path, data=None if data is None else json.dumps(data).encode(),
                              headers={"Content-Type": "application/json"})
            try:
                response = urlopen(request, timeout=5)
            except HTTPError as exc:
                response = exc
            with response:
                return response.status, json.load(response)
        try:
            if call('/health')[0] != 200 or call('/fixed')[0] not in (200, 503):
                raise ValueError("Health or not-ready status failed.")
            if call('/query', {"topic": ""})[0] != 400 or call('/missing')[0] != 404:
                raise ValueError("Invalid request handling failed.")
            code, _ = call('/ingest', [{"Record_ID": "long", "Publication_Date": 0, "Article_Text": "x"*10001}])
            if code != 202:
                raise ValueError("Queue route failed.")
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                jobs = call('/jobs')[1]['jobs']
                if jobs and jobs[0]['status'] == 'filtered' and call('/fixed')[0] == 200:
                    break
                time.sleep(0.05)
            else:
                raise ValueError("Background worker did not prepare feeds.")
            if call('/jobs')[1]['jobs'][0]['status'] != 'filtered':
                raise ValueError("Worker did not filter long input.")
            code, response = call('/query', {})
            if code != 200 or response['request_type'] != 'query':
                raise ValueError("Query route failed.")
            if call('/clusters')[0] != 200:
                raise ValueError("Cluster route failed.")
            print("PASS HTTP: health, queue, background processing, fixed, query, clusters, and errors")
            return 0
        finally:
            server.pipeline_stop.set()
            server.pipeline_wake.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            server.pipeline_worker.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
