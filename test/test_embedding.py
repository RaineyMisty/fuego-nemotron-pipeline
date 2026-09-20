import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fuego.embedding import DIMENSIONS, MODEL_NAME, MAX_TEXT_CHARS, Embedder, EmbeddingError


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.model=Mock()
        self.model.embedding_size=DIMENSIONS
        self.model.model.tokenizer.encode.return_value=SimpleNamespace(ids=[101,1,102])
        self.model.embed.return_value=[[3,4]+[0]*(DIMENSIONS-2)]
        self.factory=Mock(return_value=self.model)
        patcher=patch.dict(sys.modules,{"fastembed":SimpleNamespace(TextEmbedding=self.factory)})
        patcher.start();self.addCleanup(patcher.stop)

    def test_lazy_load_and_reuse(self):
        embedder=Embedder()
        self.factory.assert_not_called()
        first=embedder.embed("solar power")
        second=embedder.embed("renewable electricity")
        self.factory.assert_called_once_with(model_name=MODEL_NAME,cache_dir=None,providers=["CPUExecutionProvider"],local_files_only=False)
        self.assertEqual(first,second)
        self.assertEqual(len(first),384)
        self.assertTrue(all(type(x) is float for x in first))
        self.assertAlmostEqual(math.hypot(*first),1)
        self.assertEqual(first[:2],[0.6,0.8])
        self.model.embed.assert_called_with(["renewable electricity"],batch_size=32)

    def test_batch_preserves_order(self):
        self.model.embed.return_value=[[2]+[0]*383,[0,-5]+[0]*382]
        vectors=Embedder().embed_many(["first","second"])
        self.assertEqual(vectors[0][0],1)
        self.assertEqual(vectors[1][1],-1)

    def test_empty_batch_does_not_load(self):
        self.assertEqual(Embedder().embed_many([]),[])
        self.factory.assert_not_called()

    def test_invalid_text_before_load(self):
        for text in (None,"", " ",42,[],"x"*(MAX_TEXT_CHARS+1)):
            with self.assertRaises(ValueError):Embedder().embed(text)
        with self.assertRaises(ValueError):Embedder().embed_many("text")
        self.factory.assert_not_called()

    def test_invalid_config(self):
        with self.assertRaises(ValueError):Embedder(device="")
        with self.assertRaises(ValueError):Embedder(local_files_only="true")

    def test_cache_and_device_options(self):
        Embedder(device="cpu",cache_folder="work/models",local_files_only=True).embed("text")
        self.factory.assert_called_once_with(model_name=MODEL_NAME,cache_dir="work/models",providers=["CPUExecutionProvider"],local_files_only=True)

    def test_token_limit_rejects_without_truncation(self):
        self.model.model.tokenizer.encode.return_value=SimpleNamespace(ids=list(range(257)))
        with self.assertRaisesRegex(ValueError,"256 tokens"):Embedder().embed("long text")
        self.model.embed.assert_not_called()
        self.model.model.tokenizer.no_truncation.assert_not_called()

    def test_exact_token_limit_is_allowed(self):
        self.model.model.tokenizer.encode.return_value=SimpleNamespace(ids=list(range(256)))
        self.assertEqual(len(Embedder().embed("text")),384)

    def test_wrong_dimension(self):
        self.model.embedding_size=768
        with self.assertRaises(EmbeddingError):Embedder().embed("text")

    def test_invalid_vectors(self):
        for rows in ([], [[0]*383],[[0]*384],[[float("nan")]+[0]*383],
                     [[float("inf")]+[0]*383],[[object()]+[0]*383],None):
            self.model.embed.return_value=rows
            with self.subTest(rows=str(rows)[:30]),self.assertRaises(EmbeddingError):Embedder().embed("text")

    def test_encode_failure(self):
        self.model.embed.side_effect=RuntimeError("backend error")
        with self.assertRaisesRegex(EmbeddingError,"encode"):Embedder().embed("text")

    def test_missing_dependency(self):
        with patch.dict(sys.modules,{"fastembed":None}),self.assertRaisesRegex(EmbeddingError,"install"):
            Embedder().embed("text")

    def test_model_load_failure(self):
        self.factory.side_effect=OSError("download failed")
        with self.assertRaisesRegex(EmbeddingError,"cache"):Embedder().embed("text")

    def test_generator_output(self):
        self.model.embed.return_value = iter([[3, 4] + [0] * 382])
        self.assertEqual(Embedder().embed("text")[:2], [0.6, 0.8])

    def test_generator_failure(self):
        def broken():
            yield [1] + [0] * 383
            raise RuntimeError("ONNX error")
        self.model.embed.return_value = broken()
        with self.assertRaises(EmbeddingError):
            Embedder().embed_many(["first", "second"])

    def test_inference_tokenizer_is_not_modified(self):
        embedder = Embedder()
        embedder.embed("text")
        embedder._tokenizer.no_truncation.assert_called_once()
        embedder._tokenizer.no_padding.assert_called_once()
        self.model.model.tokenizer.no_truncation.assert_not_called()
        self.model.model.tokenizer.no_padding.assert_not_called()

    def test_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cpu"):
            Embedder(device="cuda")
