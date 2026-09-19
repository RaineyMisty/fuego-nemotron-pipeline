import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fuego.client import NvidiaClient, PipelineError, parse_json

class ClientTests(unittest.TestCase):
    def client(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-secret", "NVIDIA_MODEL": "nvidia/nvidia-nemotron-nano-9b-v2"}):
            return NvidiaClient()

    def test_missing_key(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}):
            with self.assertRaises(PipelineError):
                NvidiaClient()

    def test_json_and_fences(self):
        self.assertEqual(parse_json('```json\n{"ok": true}\n```'), {"ok": True})
        for value in ['[]', 'prefix {"ok":true}', None, '{']:
            with self.assertRaises(PipelineError):
                parse_json(value)

    def test_request_uses_nvidia_and_no_think(self):
        client = self.client()
        response = io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}]}')
        with patch.object(client.opener, "open", return_value=response) as call:
            self.assertEqual(client.complete("rules", {"task": "test"}), {"ok": True})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertTrue(json.loads(request.data)["messages"][0]["content"].startswith("/no_think"))

    def test_auth_is_not_retried_or_exposed(self):
        client = self.client()
        error = HTTPError("https://example.com", 401, "test-secret", {}, io.BytesIO(b"test-secret"))
        with patch.object(client.opener, "open", side_effect=error) as call:
            with self.assertRaises(PipelineError) as raised:
                client.complete("rules", {})
            self.assertEqual(call.call_count, 1)
            self.assertNotIn("test-secret", str(raised.exception))

    def test_network_errors_are_bounded(self):
        client = self.client()
        with patch.object(client.opener, "open", side_effect=URLError("no network")) as call, patch("fuego.client.time.sleep"):
            with self.assertRaises(PipelineError):
                client.complete("rules", {})
            self.assertEqual(call.call_count, 3)

    def test_rate_limit_then_success(self):
        client = self.client()
        error = HTTPError("https://example.com", 429, "limited", {}, io.BytesIO())
        response = io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}')
        with patch.object(client.opener, "open", side_effect=[error, response]) as call, patch("fuego.client.time.sleep"):
            self.assertEqual(client.complete("rules", {}), {})
            self.assertEqual(call.call_count, 2)

    def test_truncated_response_is_rejected(self):
        client = self.client()
        response = io.BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}')
        with patch.object(client.opener, "open", return_value=response):
            with self.assertRaises(PipelineError):
                client.complete("rules", {})


if __name__ == "__main__":
    unittest.main()
