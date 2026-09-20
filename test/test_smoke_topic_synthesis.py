from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from fuego.ai import AIConfig, AIError
from integration.smoke_topic_synthesis import SAMPLE, main
from test.test_topic_synthesis import response

ROOT = Path(__file__).resolve().parents[1]


class SmokeTopicSynthesisTests(unittest.TestCase):
    def run_smoke(self, *, reply=None, error=None, args=None):
        out, err = io.StringIO(), io.StringIO()
        with patch("integration.smoke_topic_synthesis.AIConfig.from_env", return_value=AIConfig(api_key="test-key")), \
             patch("integration.smoke_topic_synthesis.NemotronClient") as client, \
             redirect_stdout(out), redirect_stderr(err):
            client.return_value.complete.return_value = response() if reply is None else reply
            client.return_value.complete.side_effect = error
            code = main(args or [])
        return code, out.getvalue(), err.getvalue(), client

    def test_sample_and_real_processing_path(self):
        self.assertGreater(len(json.loads(SAMPLE.read_text())), 1)
        code, out, err, client = self.run_smoke()
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["title"], "Electric buses")
        self.assertIn("PASS", err)
        client.return_value.complete.assert_called_once()
        self.assertEqual(client.call_args.args[0].max_retries, 0)
        self.assertNotIn("test-key", out+err)

    def test_request_response_and_interrupt_errors(self):
        self.assertEqual(self.run_smoke(error=AIError("network"))[0], 1)
        self.assertEqual(self.run_smoke(reply=response(finish_reason="length"))[0], 3)
        self.assertEqual(self.run_smoke(reply=response('{"error":"no_shared_topic"}'))[0], 3)
        self.assertEqual(self.run_smoke(error=KeyboardInterrupt())[0], 130)

    def test_input_config_and_stdin(self):
        self.assertEqual(self.run_smoke(args=["--articles", "/missing/summaries.json"])[0], 2)
        self.assertEqual(self.run_smoke(args=["--timeout", "-1"])[0], 2)
        with patch("sys.stdin", io.StringIO('[{"summary":"Bus trial."}]')):
            self.assertEqual(self.run_smoke(args=["--articles", "-"])[0], 0)
        with patch("integration.smoke_topic_synthesis.AIConfig.from_env", side_effect=ValueError()), redirect_stderr(io.StringIO()):
            self.assertEqual(main([]), 2)

    def test_module_and_script_help(self):
        for command in ([sys.executable, "-B", "-m", "integration.smoke_topic_synthesis", "--help"],
                        [sys.executable, "-B", str(ROOT / "integration/smoke_topic_synthesis.py"), "--help"]):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
