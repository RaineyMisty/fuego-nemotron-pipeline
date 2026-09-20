import copy
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fuego.ai import AIConfig, AIError, DEFAULT_MODEL, ENDPOINT, NemotronClient, _NoRedirect

MESSAGES = [{"role": "system", "content": "Return a short answer."},
            {"role": "user", "content": "Hello 世界"}]
RESPONSE = {"id": "test-id", "choices": [{"message": {"role": "assistant", "content": "Hello"},
                                          "finish_reason": "stop"}], "usage": {"total_tokens": 12}}


def response(value=RESPONSE):
    return io.BytesIO(json.dumps(value).encode())


def http_error(code):
    return HTTPError(ENDPOINT, code, "private error", {}, io.BytesIO(b"private body"))


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True):
            config = AIConfig.from_env()
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.timeout, 20)
        self.assertEqual(config.max_tokens, 16384)
        self.assertFalse(config.enable_thinking)
        self.assertNotIn("test-key", repr(config))

    def test_environment_overrides(self):
        env = {"NVIDIA_API_KEY":"test-key", "NVIDIA_MODEL":"nvidia/test-nemotron", "NVIDIA_TIMEOUT":"45",
               "NVIDIA_MAX_TOKENS":"100", "NVIDIA_TEMPERATURE":"0.5", "NVIDIA_ENABLE_THINKING":"true",
               "NVIDIA_MAX_RETRIES":"0"}
        with patch.dict(os.environ, env, clear=True):
            client = NemotronClient()
        self.assertEqual(client.config, AIConfig("test-key", "nvidia/test-nemotron", 45, 100, 0.5, True, 0))

    def test_invalid_config(self):
        cases = [{"api_key":""}, {"api_key":"replace-with-your-key"}, {"api_key":"key\nsecret"},
                 {"model":"openai/gpt"}, {"timeout":0}, {"timeout":float("nan")}, {"timeout":601},
                 {"temperature":-1}, {"temperature":float("inf")}, {"temperature":True},
                 {"max_tokens":0}, {"max_tokens":1.5}, {"max_retries":6}, {"max_retries":True},
                 {"enable_thinking":"false"}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AIConfig(**{"api_key":"test-key", **settings})

    def test_invalid_environment(self):
        for env in [{}, {"NVIDIA_TIMEOUT":"oops"}, {"NVIDIA_MAX_TOKENS":"3.5"},
                    {"NVIDIA_ENABLE_THINKING":"yes"}, {"NVIDIA_MAX_RETRIES":"-1"}]:
            values = {} if not env else {"NVIDIA_API_KEY":"test-key", **env}
            with patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                AIConfig.from_env()

    def test_config_type(self):
        with self.assertRaises(TypeError):
            NemotronClient({})


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = NemotronClient(AIConfig("test-key"))
        self.sleep = patch("fuego.ai.time.sleep")
        self.wait = self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_request_and_raw_response(self):
        messages = copy.deepcopy(MESSAGES)
        with patch.object(self.client._opener, "open", return_value=response()) as send:
            result = self.client.complete(messages)
        self.assertEqual(result, RESPONSE)
        self.assertEqual(messages, MESSAGES)
        req = send.call_args.args[0]
        self.assertEqual(req.full_url, ENDPOINT)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
        body = json.loads(req.data)
        self.assertEqual(body["messages"], MESSAGES)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(body["max_tokens"],16384)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking":False})
        self.assertFalse(body["stream"])
        self.assertEqual(send.call_args.kwargs["timeout"],20)
        self.wait.assert_not_called()

    def test_no_business_output_parsing(self):
        for content, reason in [("Plain text", "stop"), ('{"arbitrary":1}', "stop"), ("partial", "length"), (None,"content_filter")]:
            value = {"choices":[{"message":{"content":content},"finish_reason":reason}]}
            with patch.object(self.client._opener,"open",return_value=response(value)):
                self.assertEqual(self.client.complete(MESSAGES),value)

    def test_bad_messages_do_not_send(self):
        for value in [[], "hello", [{}], [{"role":"tool","content":"x"}],
                      [{"role":"user","content":""}], [{"role":"user","content":{}}],
                      [{"role":"user","content":"x","extra":True}]]:
            with patch.object(self.client._opener,"open") as send, self.assertRaises(ValueError):
                self.client.complete(value)
            send.assert_not_called()

    def test_permanent_http_errors_do_not_retry(self):
        for code in (400,401,403,404,422):
            with patch.object(self.client._opener,"open",side_effect=http_error(code)) as send:
                with self.assertRaisesRegex(AIError, str(code)) as raised:
                    self.client.complete(MESSAGES)
                self.assertEqual(send.call_count,1)
                self.assertNotIn("private",str(raised.exception))
        self.wait.assert_not_called()

    def test_transient_http_errors_retry(self):
        for code in (429,500,502,503,504):
            with patch.object(self.client._opener,"open",side_effect=[http_error(code),response()]) as send:
                self.assertEqual(self.client.complete(MESSAGES),RESPONSE)
                self.assertEqual(send.call_count,2)

    def test_http_retries_are_bounded(self):
        with patch.object(self.client._opener,"open",side_effect=[http_error(503) for _ in range(3)]) as send:
            with self.assertRaises(AIError):
                self.client.complete(MESSAGES)
            self.assertEqual(send.call_count,3)
        self.assertEqual([c.args[0] for c in self.wait.call_args_list],[1,2])

    def test_network_retries(self):
        for error in (URLError("private"), TimeoutError("private"), ConnectionResetError("private")):
            with patch.object(self.client._opener,"open",side_effect=error) as send:
                with self.assertRaises(AIError) as raised:
                    self.client.complete(MESSAGES)
                self.assertEqual(send.call_count,3)
                self.assertNotIn("private",str(raised.exception))

    def test_network_recovery(self):
        with patch.object(self.client._opener,"open",side_effect=[TimeoutError(),response()]):
            self.assertEqual(self.client.complete(MESSAGES),RESPONSE)

    def test_retries_can_be_disabled(self):
        client = NemotronClient(AIConfig("test-key", max_retries=0))
        with patch.object(client._opener,"open",side_effect=TimeoutError()) as send:
            with self.assertRaises(AIError):
                client.complete(MESSAGES)
            self.assertEqual(send.call_count,1)
        self.wait.assert_not_called()

    def test_bad_response_does_not_retry(self):
        for raw in (b"not json", b"[]", b"{}", b'{"choices":[]}', b"\xff"):
            with patch.object(self.client._opener,"open",return_value=io.BytesIO(raw)) as send:
                with self.assertRaises(AIError):
                    self.client.complete(MESSAGES)
                self.assertEqual(send.call_count,1)

    def test_response_size_limit(self):
        with patch("fuego.ai.MAX_RESPONSE_BYTES",4), patch.object(self.client._opener,"open",return_value=io.BytesIO(b"12345")):
            with self.assertRaisesRegex(AIError,"too large"):
                self.client.complete(MESSAGES)

    def test_redirect_is_blocked(self):
        with self.assertRaises(AIError):
            _NoRedirect().redirect_request(None,None,302,"",{},"https://example.com")

    def test_no_logs_or_secrets(self):
        from contextlib import redirect_stdout, redirect_stderr
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch.object(self.client._opener,"open",return_value=response()):
            self.client.complete(MESSAGES)
        self.assertEqual(out.getvalue()+err.getvalue(),"")


if __name__ == "__main__":
    unittest.main()
