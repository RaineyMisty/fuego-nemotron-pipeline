"""Encode text as a normalized 384-dimensional vector."""

import math
from copy import deepcopy

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_TEXT_CHARS = 20000
MAX_TOKENS = 256


class EmbeddingError(RuntimeError):
    """The embedding model could not produce a valid vector."""


class Embedder:
    def __init__(self, *, device="cpu", cache_folder=None, local_files_only=False):
        if device != "cpu":
            raise ValueError("This FastEmbed setup supports device=cpu only.")
        if type(local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean.")
        self.device = device
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise EmbeddingError("Install embedding dependencies: python -m pip install -e '.[embedding]'") from None
            try:
                model = TextEmbedding(model_name=MODEL_NAME, cache_dir=self.cache_folder,
                                      providers=["CPUExecutionProvider"],
                                      local_files_only=self.local_files_only)
                if model.embedding_size != DIMENSIONS:
                    raise EmbeddingError("Expected a 384-dimensional model.")
                # Count all tokens without changing the inference tokenizer.
                tokenizer = deepcopy(model.model.tokenizer)
                tokenizer.no_truncation()
                tokenizer.no_padding()
                self._tokenizer = tokenizer
                self._model = model
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("Cannot load MiniLM. Check the model cache, network, and device.") from exc
        return self._model

    def embed(self, text):
        """Return one unit vector as a list of Python floats."""
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        """Encode a batch in order. Reuse this instance to reuse the model."""
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings.")
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"Each text must have 1-{MAX_TEXT_CHARS} characters.")
        if not texts:
            return []
        model = self._load_model()
        try:
            for index, text in enumerate(texts):
                tokens = self._tokenizer.encode(text, add_special_tokens=True).ids
                if len(tokens) > MAX_TOKENS:
                    raise ValueError(f"Text {index} exceeds the model limit of {MAX_TOKENS} tokens. Use a shorter overview.")
            rows = list(model.embed(texts, batch_size=32))
        except ValueError:
            raise
        except Exception as exc:
            raise EmbeddingError("MiniLM could not encode the text.") from exc
        try:
            if len(rows) != len(texts):
                raise EmbeddingError("The model returned the wrong number of vectors.")
            return [_normalize(row) for row in rows]
        except (TypeError, OverflowError):
            raise EmbeddingError("The model returned invalid vectors.") from None


def _normalize(row):
    try:
        if len(row) != DIMENSIONS:
            raise EmbeddingError("Expected a 384-dimensional vector.")
        vector = [float(value) for value in row]
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingError("The model returned invalid vector values.") from None
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingError("Vector values must be finite.")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingError("The model returned a zero or invalid vector norm.")
    return [value / norm for value in vector]
