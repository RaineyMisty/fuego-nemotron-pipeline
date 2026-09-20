from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from fuego.embedding import EmbeddingError
from integration.smoke_embedding import KEY_A, KEY_B, compare_vectors, main

ROOT=Path(__file__).resolve().parents[1]


class SmokeEmbeddingTests(unittest.TestCase):
    def test_sample_has_twenty_distinct_terms_per_key(self):
        self.assertEqual(len(KEY_A),20);self.assertEqual(len(KEY_B),20)
        self.assertEqual(len(set(KEY_A)),20);self.assertEqual(len(set(KEY_B)),20)
        self.assertFalse(set(KEY_A)&set(KEY_B))

    def test_known_distances(self):
        a=[1]+[0]*383;b=[0,1]+[0]*382
        metrics=compare_vectors(a,b)
        self.assertEqual(metrics["cosine_similarity"],0)
        self.assertEqual(metrics["cosine_distance"],1)
        self.assertAlmostEqual(metrics["euclidean_distance"],math.sqrt(2))
        self.assertEqual(compare_vectors(a,a)["cosine_distance"],0)
        self.assertEqual(compare_vectors(a,[-x for x in a])["cosine_distance"],2)

    def test_invalid_vectors(self):
        for b in ([0]*384,[1]*384,[1]*3,[float("nan")]+[0]*383):
            with self.assertRaises(ValueError):compare_vectors([1]+[0]*383,b)

    def run_smoke(self,vectors=None,error=None,args=None):
        out,err=io.StringIO(),io.StringIO()
        with patch("integration.smoke_embedding.Embedder") as factory,redirect_stdout(out),redirect_stderr(err):
            factory.return_value.embed_many.return_value=vectors
            factory.return_value.embed_many.side_effect=error
            code=main(args or [])
        return code,out.getvalue(),err.getvalue(),factory

    def test_pass_outputs_full_vectors(self):
        vector=[1]+[0]*383
        code,out,err,factory=self.run_smoke([vector,vector])
        self.assertEqual(code,0)
        result=json.loads(out)
        self.assertEqual(result["key_a"]["vector"],vector)
        self.assertEqual(result["key_b"]["vector"],vector)
        self.assertIn("PASS",err)
        factory.return_value.embed_many.assert_called_once_with(["; ".join(KEY_A),"; ".join(KEY_B)])

    def test_dissimilar_vectors_fail(self):
        code,out,_,_=self.run_smoke([[1]+[0]*383,[0,1]+[0]*382])
        self.assertEqual(code,1);self.assertFalse(json.loads(out)["passed"])

    def test_load_error(self):
        code,_,err,_=self.run_smoke(error=EmbeddingError("Cannot load"))
        self.assertEqual(code,2);self.assertIn("Cannot load",err)

    def test_interrupt(self):
        self.assertEqual(self.run_smoke(error=KeyboardInterrupt())[0],130)

    def test_invalid_threshold(self):
        for value in ("nan","inf","2"):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(["--min-similarity",value])

    def test_offline_option(self):
        vector=[1]+[0]*383
        _,_,_,factory=self.run_smoke([vector,vector],args=["--local-files-only","--cache-folder","work/models"])
        factory.assert_called_once_with(device="cpu",cache_folder="work/models",local_files_only=True)

    def test_module_and_script(self):
        for command in ([sys.executable,"-B","-m","integration.smoke_embedding","--help"],
                        [sys.executable,"-B",str(ROOT/"integration/smoke_embedding.py"),"--help"]):
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
