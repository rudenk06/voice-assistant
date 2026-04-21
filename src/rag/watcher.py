"""Document watcher: monitors documents folder for changes and triggers reindexing."""
import hashlib
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class DocumentWatcher:
    """Polls a directory for document changes and triggers reindexing."""

    def __init__(self, documents_path: str, indexer_factory: Callable,
                 poll_interval: int = 60):
        self.documents_path = documents_path
        self.indexer_factory = indexer_factory
        self.poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._known_files: dict[str, str] = {}  # filepath -> hash
        self._supported_exts = {".txt", ".pdf", ".docx"}

    def start(self):
        """Start watching in a background thread."""
        self._running = True
        # Build initial snapshot
        self._known_files = self._scan_files()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"Document watcher started (polling every {self.poll_interval}s)")

    def _poll_loop(self):
        while self._running:
            time.sleep(self.poll_interval)
            if not self._running:
                break
            try:
                self._check_for_changes()
            except Exception as e:
                logger.error(f"Document watcher error: {e}")

    def _check_for_changes(self):
        current_files = self._scan_files()

        added = set(current_files.keys()) - set(self._known_files.keys())
        removed = set(self._known_files.keys()) - set(current_files.keys())
        changed = {
            f for f in current_files
            if f in self._known_files and current_files[f] != self._known_files[f]
        }

        if not added and not removed and not changed:
            return

        logger.info(
            f"Document changes detected: {len(added)} added, "
            f"{len(removed)} removed, {len(changed)} modified"
        )

        indexer = self.indexer_factory()

        for filepath in removed:
            indexer.remove_document(filepath)

        if added or changed:
            indexer.index_directory(self.documents_path)

        self._known_files = current_files

    def _scan_files(self) -> dict[str, str]:
        """Scan directory and return {filepath: hash} mapping."""
        files = {}
        if not os.path.isdir(self.documents_path):
            return files
        for name in os.listdir(self.documents_path):
            ext = os.path.splitext(name)[1].lower()
            if ext not in self._supported_exts:
                continue
            filepath = os.path.join(self.documents_path, name)
            if os.path.isfile(filepath):
                files[filepath] = self._file_hash(filepath)
        return files

    def _file_hash(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Document watcher stopped")
