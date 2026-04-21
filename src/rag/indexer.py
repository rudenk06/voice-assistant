"""Document indexer: builds FAISS index + SQLite metadata store."""
import hashlib
import logging
import os
import sqlite3
from datetime import datetime

import faiss
import numpy as np

from src.config import load_config
from src.rag.document_loader import DocumentLoader
from src.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, faiss_path: str, db_path: str, embedder: Embedder, loader: DocumentLoader):
        self.faiss_path = faiss_path
        self.db_path = db_path
        self.embedder = embedder
        self.loader = loader
        self.index = None
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    filepath TEXT,
                    format TEXT,
                    hash TEXT,
                    indexed_at TEXT,
                    chunk_count INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    text TEXT,
                    chunk_index INTEGER,
                    embedding_id INTEGER,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            conn.commit()

    def _file_hash(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def _doc_id(self, filepath: str) -> str:
        return hashlib.md5(filepath.encode()).hexdigest()

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from loaded embedder."""
        self.embedder.load()
        dim = self.embedder.get_dimension()
        self.embedder.unload()
        return dim

    def index_directory(self, documents_path: str):
        """Index all supported documents in a directory."""
        files = self.loader.get_supported_files(documents_path)
        if not files:
            logger.warning(f"No documents found in {documents_path}")
            return

        logger.info(f"Indexing {len(files)} documents from {documents_path}")

        all_chunks = []
        chunk_metadata = []

        for filepath in files:
            file_hash = self._file_hash(filepath)
            doc_id = self._doc_id(filepath)

            # Check if already indexed with same hash
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT hash FROM documents WHERE id = ?", (doc_id,)
                ).fetchone()
                if row and row[0] == file_hash:
                    logger.info(f"Skipping {filepath} (unchanged)")
                    continue

            # Remove old data for this document
            self._remove_document_data(doc_id)

            chunks = self.loader.load(filepath)
            if not chunks:
                continue

            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower().lstrip(".")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, filename, filepath, ext, file_hash,
                     datetime.now().isoformat(), len(chunks)),
                )
                conn.commit()

            for i, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_metadata.append((doc_id, chunk_text, i))

        if not all_chunks:
            logger.info("No new chunks to index")
            self._load_or_create_index()
            return

        # Compute embeddings (is_query=False — это документы, не запросы)
        self.embedder.load()
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            emb = self.embedder.embed(batch, is_query=False)
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.embedder.unload()

        # Get current max embedding_id
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT MAX(embedding_id) FROM chunks").fetchone()
            start_id = (row[0] or -1) + 1

        # Store chunks in SQLite
        with sqlite3.connect(self.db_path) as conn:
            for idx, (doc_id, text, chunk_idx) in enumerate(chunk_metadata):
                embedding_id = start_id + idx
                conn.execute(
                    "INSERT INTO chunks (document_id, text, chunk_index, embedding_id) VALUES (?, ?, ?, ?)",
                    (doc_id, text, chunk_idx, embedding_id),
                )
            conn.commit()

        # Build/update FAISS index
        self._rebuild_full_index()

        logger.info(f"Indexed {len(all_chunks)} new chunks")

    def _rebuild_full_index(self):
        """Rebuild FAISS index from all embeddings in DB."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT embedding_id, text FROM chunks ORDER BY embedding_id"
            ).fetchall()

        if not rows:
            dim = self._get_embedding_dim()
            self.index = faiss.IndexFlatIP(dim)
            self._save_index()
            return

        texts = [r[1] for r in rows]

        self.embedder.load()
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.embedder.embed(batch, is_query=False)
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.embedder.unload()

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self._save_index()

    def _load_or_create_index(self):
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
        else:
            dim = self._get_embedding_dim()
            self.index = faiss.IndexFlatIP(dim)

    def _save_index(self):
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_path)

    def _remove_document_data(self, doc_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

    def add_document(self, filepath: str):
        """Index a single document (incremental)."""
        self.index_directory(os.path.dirname(filepath))

    def remove_document(self, filepath: str):
        """Remove a document and rebuild index."""
        doc_id = self._doc_id(filepath)
        self._remove_document_data(doc_id)
        self._rebuild_full_index()
        logger.info(f"Removed document {filepath}")


def main():
    """CLI entry point: python -m src.rag.indexer"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    config = load_config()
    rag_cfg = config["rag"]

    loader = DocumentLoader(
        chunk_size=rag_cfg.get("chunk_size", 400),
        chunk_overlap=rag_cfg.get("chunk_overlap", 50),
    )
    embedder = Embedder(rag_cfg["embedder"]["model_path"])

    indexer = Indexer(
        faiss_path=rag_cfg["index"]["faiss_path"],
        db_path=rag_cfg["index"]["db_path"],
        embedder=embedder,
        loader=loader,
    )
    indexer.index_directory(rag_cfg["documents_path"])
    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
