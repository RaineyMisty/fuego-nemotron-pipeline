"""Check local AI calls and pipeline settings without a model server."""

import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fuego.ai import AIConfig, AIError, NemotronClient
from fuego.pipeline import Pipeline, PipelineConfig
from integration.smoke_ai import main
from test.test_pipeline import TinyEmbedder
from contextlib import redirect_stdout


class OllamaTests(unittest.TestCase):
    def config(self, **env):
        with patch.dict(os.environ, {"AI_PROVIDER": "ollama", **env}, clear=True):
            return AIConfig.from_env()

    def test_defaults_without_key_and_nvidia_settings_are_ignored(self):
        config = self.config(NVIDIA_API_KEY="private", NVIDIA_MODEL="old", NVIDIA_TIMEOUT="bad")
        self.assertEqual(config.model, "qwen3:0.6b")
        self.assertEqual(config.endpoint, "http://localhost:11434/api/chat")
        self.assertEqual((config.timeout, config.max_retries), (20, 0))
        self.assertEqual(config.api_key, "")
        self.assertFalse(config.enable_thinking)

    def test_overrides(self):
        config = self.config(OLLAMA_BASE_URL="http://127.0.0.1:11435/", OLLAMA_MODEL="qwen3:4b",
                             OLLAMA_TIMEOUT="45", OLLAMA_MAX_TOKENS="100", OLLAMA_TEMPERATURE="0.5",
                             OLLAMA_ENABLE_THINKING="true", OLLAMA_MAX_RETRIES="1")
        self.assertEqual(config.endpoint, "http://127.0.0.1:11435/api/chat")
        self.assertEqual((config.model, config.timeout, config.max_tokens), ("qwen3:4b", 45, 100))
        self.assertTrue(config.enable_thinking)
        self.assertEqual((config.temperature, config.max_retries), (0.5, 1))

    def test_invalid_settings(self):
        for env in ({"AI_PROVIDER": "other"}, {"OLLAMA_MODEL": ""}, {"OLLAMA_TIMEOUT": "0"},
                    {"OLLAMA_ENABLE_THINKING": "yes"}, {"OLLAMA_MAX_RETRIES": "-1"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.config(**env)
        for url in ("file:///tmp/model", "http://", "http://user:pass@localhost", "http://localhost/api/chat",
                    "http://localhost?x=1", "http://localhost#x", "http://localhost:abc", "http://localhost:0"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.config(OLLAMA_BASE_URL=url)

    def test_native_request_and_shared_response(self):
        client = NemotronClient(self.config())
        value = {"model": "qwen3:0.6b", "message": {"role": "assistant", "content": "OK"},
                 "done": True, "done_reason": "stop"}
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(value).encode())) as send:
            result = client.complete([{"role": "user", "content": "Reply OK"}])
        req = send.call_args.args[0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/chat")
        self.assertIsNone(req.get_header("Authorization"))
        body = json.loads(req.data)
        self.assertFalse(body["think"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"], {"temperature": 0, "num_predict": 16384})
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(result["choices"][0]["message"]["content"], "OK")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_json_schema_is_sent_as_native_format(self):
        client = NemotronClient(self.config())
        schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
        value = {"message": {"role": "assistant", "content": "{}"}, "done": True, "done_reason": "stop"}
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(value).encode())) as send:
            client.complete([{"role": "user", "content": "Hello"}], response_schema=schema)
        self.assertEqual(json.loads(send.call_args.args[0].data)["format"], schema)

    def test_schema_rejects_invalid_type_and_unsupported_provider(self):
        for config, schema in ((self.config(), "bad"), (AIConfig(api_key="test-key"), {})):
            client = NemotronClient(config)
            with patch.object(client._opener, "open") as send, self.assertRaises(ValueError):
                client.complete([{"role": "user", "content": "Hello"}], response_schema=schema)
            send.assert_not_called()

    def test_truncated_reply_is_not_marked_complete(self):
        client = NemotronClient(self.config())
        value = {"message": {"role": "assistant", "content": "partial"}, "done": True, "done_reason": "length"}
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(value).encode())):
            self.assertEqual(client.complete([{"role": "user", "content": "Hello"}])["choices"][0]["finish_reason"], "length")

    def test_bad_response_and_connection_failure(self):
        client = NemotronClient(self.config())
        for value in ({"error": "private"}, {}, {"done": False},
                      {"done": True, "done_reason": "stop", "message": {"content": "OK"}}):
            with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(value).encode())), self.assertRaisesRegex(AIError, "Ollama"):
                client.complete([{"role": "user", "content": "Hello"}])
        with patch.object(client._opener, "open", side_effect=TimeoutError()) as send, self.assertRaisesRegex(AIError, "Ollama"):
            client.complete([{"role": "user", "content": "Hello"}])
        self.assertEqual(send.call_count, 1)

    def test_pipeline_identity_and_retry_owner(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AI_PROVIDER": "ollama", "OLLAMA_TIMEOUT": "80", "OLLAMA_MAX_RETRIES": "bad"}, clear=True):
            config = PipelineConfig(Path(tmp))
            pipeline = Pipeline(config, embedder=TinyEmbedder())
            self.assertEqual(pipeline.ai_model, "qwen3:0.6b")
            self.assertEqual(pipeline.ai_timeout, 8)
            client = pipeline._ai()
            self.assertEqual((client.config.timeout, client.config.max_retries), (8, 0))
            with sqlite3.connect(pipeline.store.db_path) as db:
                signature = json.loads(db.execute("SELECT value FROM settings WHERE key='signature'").fetchone()[0])
            self.assertEqual(signature["provider"], "ollama")
            self.assertEqual(signature["ai_model"], client.config.model)
            Pipeline(config, embedder=TinyEmbedder())
            with redirect_stdout(io.StringIO()):
                report = pipeline.run_pending(refresh=False)
            self.assertEqual(report["retry_policy"]["ai_provider"], "ollama")
            self.assertEqual(report["retry_policy"]["ai_timeout"], 8)
            self.assertEqual(report["retry_policy"]["ai_max_retries"], 0)
            for env in ({"OLLAMA_MODEL": "qwen3:4b"}, {"OLLAMA_BASE_URL": "http://localhost:11435"}, {"AI_PROVIDER": "nvidia"}):
                with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "State mode"):
                    Pipeline(config, embedder=TinyEmbedder())

    def test_pipeline_overrides_injected_local_timeout(self):
        client = NemotronClient(self.config(OLLAMA_TIMEOUT="90", OLLAMA_MAX_RETRIES="2"))
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Pipeline(PipelineConfig(Path(tmp)), client=client, embedder=TinyEmbedder())
            self.assertEqual((client.config.timeout, client.config.max_retries), (8, 0))
            self.assertEqual(pipeline.ai_model, "qwen3:0.6b")

    def test_smoke_uses_local_endpoint(self):
        value = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
        out = io.StringIO()
        with patch.dict(os.environ, {"AI_PROVIDER": "ollama"}, clear=True), redirect_stdout(out), patch("integration.smoke_ai.NemotronClient") as client:
            client.return_value.complete.return_value = value
            self.assertEqual(main([]), 0)
        self.assertEqual(client.call_args.args[0].provider, "ollama")
        self.assertIn("http://localhost:11434/api/chat", out.getvalue())
        self.assertIn("PASS", out.getvalue())


if __name__ == "__main__":
    unittest.main()
