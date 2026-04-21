import logging
import os
import numpy as np
from src.utils.memory import force_gc, log_memory_usage

logger = logging.getLogger(__name__)


class Embedder:
    """Sentence embedder using ONNX + fast tokenizers."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = None
        self._session = None
        self._is_e5 = "e5" in model_path.lower()

    def load(self):
        """Load the embedding model into RAM."""
        log_memory_usage("before embedder load")
        self._load_onnx()
        log_memory_usage("after embedder load")

    def _load_onnx(self):
        """Load ONNX model with fast tokenizer."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = os.path.join(self.model_path, "model.onnx")
        tokenizer_path = os.path.join(self.model_path, "tokenizer.json")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # Fast tokenizer — загрузка 0.1 сек вместо 8 сек
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding(length=512)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2

        self._session = ort.InferenceSession(onnx_path, sess_options)

        # Определяем какие входы нужны модели
        self._input_names = [inp.name for inp in self._session.get_inputs()]

        logger.info(f"Embedder loaded (ONNX, e5={self._is_e5}): {self.model_path}")

    def embed(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        """Compute embeddings. Returns [N, dim] normalized array."""
        if self._is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [f"{prefix}{t}" for t in texts]

        all_embeddings = []
        for text in texts:
            encoding = self._tokenizer.encode(text)

            feeds = {}
            if "input_ids" in self._input_names:
                feeds["input_ids"] = np.array([encoding.ids], dtype=np.int64)
            if "attention_mask" in self._input_names:
                feeds["attention_mask"] = np.array([encoding.attention_mask], dtype=np.int64)
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])

            outputs = self._session.run(None, feeds)
            embedding = outputs[0][:, 0, :]
            all_embeddings.append(embedding)

        embeddings = np.vstack(all_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        dummy = self.embed(["test"])
        return dummy.shape[1]

    def unload(self):
        """Free model from RAM."""
        self._session = None
        self._tokenizer = None
        force_gc()
        log_memory_usage("after embedder unload")
        logger.info("Embedder unloaded")
