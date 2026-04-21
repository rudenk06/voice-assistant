"""Wake word detection using Vosk keyword spotting."""
import json
import logging
import threading
from typing import Callable

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Listens for a wake word using Vosk restricted vocabulary."""

    def __init__(self, model_path: str, wake_words: list[str],
                 sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.wake_words = [w.lower() for w in wake_words]
        self._running = False
        self._thread = None

        logger.info(f"Loading Vosk model for wake word detection...")
        self.model = Model(model_path)

        # Build grammar with wake words
        grammar = json.dumps(self.wake_words + [""], ensure_ascii=False)
        self._grammar = grammar
        logger.info(f"Wake word detector ready. Words: {self.wake_words}")

    def listen(self, callback: Callable[[], None]) -> None:
        """Start listening for wake word in a background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, args=(callback,), daemon=True
        )
        self._thread.start()
        logger.info("Wake word listener started")

    def _listen_loop(self, callback: Callable[[], None]):
        chunk_size = 4000  # ~250ms at 16kHz
        rec = KaldiRecognizer(self.model, self.sample_rate, self._grammar)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
            ) as stream:
                while self._running:
                    data, _ = stream.read(chunk_size)
                    audio_bytes = data.astype(np.int16).tobytes()

                    if rec.AcceptWaveform(audio_bytes):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip().lower()
                        if text and any(w in text for w in self.wake_words):
                            logger.info(f"Wake word detected: {text}")
                            callback()
                    else:
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "").strip().lower()
                        if partial_text and any(w in partial_text for w in self.wake_words):
                            logger.info(f"Wake word detected (partial): {partial_text}")
                            rec.Reset()
                            callback()

        except Exception as e:
            if self._running:
                logger.error(f"Wake word listener error: {e}")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Wake word listener stopped")
