from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fuego.__main__ import main
from fuego.pipeline import Pipeline
from integration.smoke_server import main as server_smoke
from test.test_pipeline import TinyEmbedder, record


class MainTests(unittest.TestCase):
    def test_cli_ingest_work_export_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.json'
            source.write_text(json.dumps([record()]))
            args = ['--mock-ai', '--mock-query', 'solar', '--state', str(root/'state')]
            with patch('fuego.__main__.Pipeline', side_effect=lambda cfg: Pipeline(cfg, embedder=TinyEmbedder())), redirect_stdout(io.StringIO()):
                self.assertEqual(main(args+['ingest', str(source)]), 0)
                self.assertEqual(main(args+['work']), 0)
                self.assertEqual(main(args+['status']), 0)
                self.assertEqual(main(args+['refresh','--force-clusters']), 0)
                for kind in ('fixed','query','cluster'):
                    output = root/f'{kind}.json'
                    self.assertEqual(main(args+['export', kind, '--output', str(output)]), 0)
                    self.assertEqual(json.loads(output.read_text())['request_type'], kind)

    def test_cli_bad_input_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--mock-ai','--state',directory,'ingest','/missing/input.json']), 1)

    def test_http_routes_and_worker(self):
        with redirect_stdout(io.StringIO()):
            try:
                self.assertEqual(server_smoke(), 0)
            except PermissionError:
                self.skipTest('This environment blocks local sockets. Run integration.smoke_server with socket access.')
