from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fuego.ai import AIError
from integration.smoke_article_processing import main

ROOT=Path(__file__).resolve().parents[1]
KEYWORDS=["Pittsburgh","university","robotics","laboratory","students","engineers","sensors","navigation",
          "accessibility","wheelchairs","testing","volunteers","research","funding","grant","workshop",
          "campus","September","safety","prototype"]


class SmokeArticleTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.article=Path(temp.name)/"article.txt";self.article.write_text("A Pittsburgh university opened a robotics laboratory.")
        self.metadata=Path(temp.name)/"metadata.json";self.metadata.write_text('{"source":"test"}')

    def run_smoke(self,args=None,error=None,content=None,env=None):
        stdout,stderr=io.StringIO(),io.StringIO()
        body=content if content is not None else json.dumps({"keywords":KEYWORDS,"summary":"A laboratory opened."})
        with patch.dict(os.environ,{"NVIDIA_API_KEY":"private-test-key",**(env or {})},clear=True), \
                patch("fuego.ai.NemotronClient.complete") as call, redirect_stdout(stdout),redirect_stderr(stderr):
            call.side_effect=error
            call.return_value={"choices":[{"finish_reason":"stop","message":{"content":body}}]}
            code=main(args if args is not None else ["--article",str(self.article),"--metadata",str(self.metadata)])
        self.assertNotIn("private-test-key",stdout.getvalue()+stderr.getvalue())
        return code,stdout.getvalue(),stderr.getvalue(),call

    def test_calls_article_processor_and_ai_interface(self):
        code,out,err,call=self.run_smoke()
        self.assertEqual(code,0)
        result=json.loads(out)
        self.assertEqual(len(result["keywords"]),20)
        self.assertIn("overview",result)
        self.assertIn("PASS",err)
        payload=json.loads(call.call_args.args[0][1]["content"])
        self.assertEqual(payload["article"],self.article.read_text())
        self.assertEqual(payload["metadata"],{"source":"test"})

    def test_ollama_without_nvidia_key(self):
        code, out, err, call = self.run_smoke(env={"AI_PROVIDER": "ollama", "NVIDIA_API_KEY": ""})
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out)["keywords"]), 20)
        self.assertIn("qwen3:0.6b", err)
        self.assertIn("http://localhost:11434/api/chat", err)
        self.assertIn("to ollama", err)
        call.assert_called_once()

    def test_default_sample_uses_real_processor_and_removes_ads(self):
        code, _, _, call = self.run_smoke(args=[], env={"AI_PROVIDER": "ollama"})
        self.assertEqual(code, 0)
        payload = json.loads(call.call_args.args[0][1]["content"])
        self.assertIn("robotics laboratory", payload["article"])
        self.assertNotIn("discount shoes", payload["article"])

    def test_reports_actual_keyword_count(self):
        body = json.dumps({"keywords": KEYWORDS + ["Maya Chen"], "summary": "A laboratory opened."})
        code, _, err, _ = self.run_smoke(content=body)
        self.assertEqual(code, 0)
        self.assertIn("21 keywords", err)

    def test_rejects_duplicate_keywords(self):
        body = json.dumps({"keywords": KEYWORDS + ["robotics"], "summary": "A laboratory opened."})
        code, _, err, call = self.run_smoke(content=body)
        self.assertEqual(code, 3)
        self.assertIn("distinct", err)
        call.assert_called_once()

    def test_missing_file(self):
        code,_,_,call=self.run_smoke(args=["--article","/missing/fuego-article.txt"])
        self.assertEqual(code,2);call.assert_not_called()

    def test_bad_metadata_and_empty_article(self):
        self.metadata.write_text('[]')
        self.assertEqual(self.run_smoke()[0],2)
        self.metadata.write_text('{}');self.article.write_text(' ')
        self.assertEqual(self.run_smoke()[0],2)

    def test_missing_key(self):
        code,_,err,call=self.run_smoke(env={"NVIDIA_API_KEY":""})
        self.assertEqual(code,2);self.assertIn("config",err);call.assert_not_called()

    def test_bad_model_json(self):
        code,_,err,_=self.run_smoke(content="wrong")
        self.assertEqual(code,3);self.assertIn("response",err)

    def test_api_error(self):
        code,_,err,_=self.run_smoke(error=AIError("HTTP 401"))
        self.assertEqual(code,1);self.assertIn("request",err)

    def test_stdin(self):
        with patch("sys.stdin",io.StringIO("Article from stdin")):
            code,_,_,call=self.run_smoke(args=["--article","-"])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(call.call_args.args[0][1]["content"])["article"],"Article from stdin")

    def test_interrupt(self):
        self.assertEqual(self.run_smoke(error=KeyboardInterrupt())[0],130)

    def test_module_and_direct_entrypoints(self):
        for command in ([sys.executable,"-B","-m","integration.smoke_article_processing","--help"],
                        [sys.executable,"-B",str(ROOT/"integration/smoke_article_processing.py"),"--help"]):
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("--article",result.stdout)
