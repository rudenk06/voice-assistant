import logging
import sqlite3

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, faiss_path: str, db_path: str):
        self.faiss_path = faiss_path
        self.db_path = db_path
        self.index = None

    def load_index(self):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(self.faiss_path)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Search for most relevant chunks.

        Args:
            query_embedding: [1, dim] or [dim] float32 array
            top_k: number of results

        Returns:
            List of {text, score, document_name}
        """
        if self.index is None:
            self.load_index()

        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                row = conn.execute(
                    """
                    SELECT c.text, d.filename
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding_id = ?
                    """,
                    (int(idx),),
                ).fetchone()
                if row:
                    results.append({
                        "text": row[0],
                        "score": float(score),
                        "document_name": row[1],
                    })

        logger.info(f"Found {len(results)} relevant chunks")
        return results
