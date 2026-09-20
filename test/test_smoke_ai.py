from contextlib import redirect_stdout, redirect_stderr
import io
import os
import unittest
from unittest.mock import patch

from fuego.ai import AIError
from integration.smoke_ai import main


class SmokeTests(unittest.TestCase):
    def run_smoke(self, response=None, error=None, args=None, env=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"NVIDIA_API_KEY":"private-test-key", **(env or {})}, clear=True), \
                patch("integration.smoke_ai.NemotronClient") as client, redirect_stdout(out), redirect_stderr(err):
            client.return_value.complete.return_value = response
            client.return_value.complete.side_effect = error
            code = main(args or [])
        self.assertNotIn("private-test-key", out.getvalue() + err.getvalue())
        return code, out.getvalue(), err.getvalue(), client

    def test_success_calls_real_module_interface(self):
        value = {"choices":[{"finish_reason":"stop","message":{"content":"OK"}}]}
        code, out, err, client = self.run_smoke(response=value)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        self.assertEqual(err, "")
        self.assertEqual(client.call_args.args[0].max_retries, 0)
        self.assertEqual(client.call_args.args[0].max_tokens, 64)
        self.assertEqual(client.return_value.complete.call_args.args[0][1]["content"], "Reply with OK.")

    def test_overrides(self):
        _, _, _, client = self.run_smoke(args=["--timeout","20","--max-tokens","256","--retries","1"])
        config = client.call_args.args[0]
        self.assertEqual((config.timeout,config.max_tokens,config.max_retries),(20,256,1))

    def test_missing_key_fails_before_call(self):
        code, _, err, client = self.run_smoke(env={"NVIDIA_API_KEY":""})
        self.assertEqual(code, 2)
        self.assertIn("config", err)
        client.assert_not_called()

    def test_invalid_options(self):
        for args in (["--timeout","-1"], ["--max-tokens","0"], ["--retries","9"]):
            code, _, _, client = self.run_smoke(args=args)
            self.assertEqual(code, 2)
            client.assert_not_called()

    def test_request_failure(self):
        code, _, err, _ = self.run_smoke(error=AIError("NVIDIA request failed (HTTP 401)."))
        self.assertEqual(code, 1)
        self.assertIn("HTTP 401", err)

    def test_timeout(self):
        self.assertEqual(self.run_smoke(error=TimeoutError("timed out"))[0], 1)

    def test_bad_responses(self):
        for value in (None, {}, {"choices":[]}, {"choices":[None]},
                      {"choices":[{"finish_reason":"length","message":{"content":"partial"}}]},
                      {"choices":[{"finish_reason":"stop","message":{"content":None}}]},
                      {"choices":[{"finish_reason":"stop","message":{"content":" "}}]}):
            with self.subTest(value=value):
                code, out, err, _ = self.run_smoke(response=value)
                self.assertEqual(code, 3)
                self.assertNotIn("PASS", out)
                self.assertIn("response", err)

    def test_interrupt(self):
        self.assertEqual(self.run_smoke(error=KeyboardInterrupt())[0],130)


if __name__ == "__main__":
    unittest.main()
