import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and chunk documents from PDF, DOCX, TXT files."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, filepath: str) -> list[str]:
        """Load a document and return list of text chunks."""
        ext = Path(filepath).suffix.lower()
        try:
            if ext == ".txt":
                text = self._load_txt(filepath)
            elif ext == ".pdf":
                text = self._load_pdf(filepath)
            elif ext == ".docx":
                text = self._load_docx(filepath)
            else:
                logger.warning(f"Unsupported format: {ext} for {filepath}")
                return []
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []

        chunks = self._chunk_text(text)
        logger.info(f"Loaded {filepath}: {len(chunks)} chunks")
        return chunks

    def _load_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _load_pdf(self, filepath: str) -> str:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    def _load_docx(self, filepath: str) -> str:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by character count."""
        paragraphs = text.strip().split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if p:
                chunks.append(p)
        return chunks

    def get_supported_files(self, directory: str) -> list[str]:
        """List all supported files in a directory."""
        supported = {".txt", ".pdf", ".docx"}
        files = []
        for name in os.listdir(directory):
            if Path(name).suffix.lower() in supported:
                files.append(os.path.join(directory, name))
        return sorted(files)
