"""Voice Assistant for University Department — main pipeline."""
import logging
import os
import signal
import sys
import threading
import time

from src.config import load_config, get_project_root
from src.audio.recorder import Recorder
from src.audio.player import Player
from src.asr.recognizer import Recognizer
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.tts.synthesizer import Synthesizer
from src.hardware.button import Button
from src.asr.wake_word import WakeWordDetector
from src.rag.watcher import DocumentWatcher
from src.utils.sounds import ensure_sounds

logger = logging.getLogger(__name__)


class VoiceAssistant:
    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._processing = False
        self._lock = threading.Lock()

        # Initialize components
        audio_cfg = config["audio"]

        self.recorder = Recorder(
            sample_rate=audio_cfg["sample_rate"],
            channels=audio_cfg["channels"],
        )
        self.player = Player()

        self.recognizer = Recognizer(
            model_path=config["asr"]["model_path"],
            sample_rate=audio_cfg["sample_rate"],
        )

        rag_cfg = config["rag"]
        self.embedder = Embedder(rag_cfg["embedder"]["model_path"])

        self.retriever = Retriever(
            faiss_path=rag_cfg["index"]["faiss_path"],
            db_path=rag_cfg["index"]["db_path"],
        )

        gen_cfg = rag_cfg.get("generator", {})
        self.generator = Generator(
            model_path=gen_cfg.get("model_path"),
            mode=gen_cfg.get("mode", "template"),
            max_tokens=gen_cfg.get("max_tokens", 100),
            context_size=gen_cfg.get("context_size", 512),
        )

        self.synthesizer = Synthesizer(
            model_path=config["tts"]["model_path"],
            sample_rate=config["tts"]["sample_rate"],
        )

        hw_cfg = config["hardware"]["button"]
        self.button = Button(
            gpio_pin=hw_cfg["gpio_pin"],
            use_keyboard_fallback=hw_cfg.get("use_keyboard_fallback", True),
        )

        # Wake word detector
        self.wake_word_detector = None
        ww_cfg = config.get("wake_word", {})
        if ww_cfg.get("enabled", False):
            self.wake_word_detector = WakeWordDetector(
                model_path=config["asr"]["model_path"],
                wake_words=[ww_cfg["phrase"]],
                sample_rate=audio_cfg["sample_rate"],
            )

        # Document watcher
        self.doc_watcher = None

        # Ensure system sounds exist
        sounds_dir = os.path.dirname(config["sounds"]["activate"])
        ensure_sounds(sounds_dir)

    def handle_query(self):
        """Full speech-to-speech pipeline: record → ASR → RAG → TTS → play."""
        with self._lock:
            if self._processing:
                return
            self._processing = True

        try:
            # 1. Activation sound
            self.player.play_sound(self.config["sounds"]["activate"])

            # 2. Record audio
            audio_cfg = self.config["audio"]
            audio = self.recorder.record_until_silence(
                silence_threshold=audio_cfg["silence_threshold"],
                silence_duration=audio_cfg["silence_duration"],
                max_duration=audio_cfg["max_record_seconds"],
            )

            if len(audio) == 0:
                self._speak("Я не услышал вопрос. Попробуйте ещё раз.")
                return

            # 3. ASR: speech → text
            text = self.recognizer.recognize(audio)
            if not text:
                self._speak("Извините, не удалось распознать вопрос. Повторите, пожалуйста.")
                return

            logger.info(f"User asked: {text}")

            # 4. RAG: embed → search → generate
            query_embedding = self.embedder.embed([text])

            chunks = self.retriever.search(
                query_embedding,
                top_k=self.config["rag"].get("top_k", 3),
            )

            self.generator.load()
            answer = self.generator.generate(text, chunks)
            self.generator.unload()

            logger.info(f"Answer: {answer}")

            # 5. TTS: text → speech → play
            self._speak(answer)

        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
            try:
                self.player.play_sound(self.config["sounds"]["error"])
            except Exception:
                pass
        finally:
            self._processing = False

    def _speak(self, text: str):
        """Synthesize and play text."""
        audio = self.synthesizer.synthesize(text)
        if len(audio) > 0:
            self.player.play(audio, self.synthesizer.sample_rate)

    def start(self):
        """Start the assistant — listen for button press and/or wake word."""
        self._running = True

        # Load retriever index at startup
        self.retriever.load_index()

        # Pre-load TTS (stays in memory)
        self.synthesizer.load()

        # Pre-load embedder (stays in memory)
        self.embedder.load()

        # Start document watcher
        rag_cfg = self.config["rag"]

        from src.rag.document_loader import DocumentLoader
        loader = DocumentLoader(
            chunk_size=rag_cfg.get("chunk_size", 400),
            chunk_overlap=rag_cfg.get("chunk_overlap", 50),
        )

        self.doc_watcher = DocumentWatcher(
            documents_path=rag_cfg["documents_path"],
            indexer_factory=lambda: self._create_indexer(rag_cfg, loader),
            poll_interval=60,
        )
        self.doc_watcher.start()

        # Setup button
        logger.info("Voice assistant ready. Press button or Enter to ask a question.")
        self.button.on_press(self.handle_query)

        # Start wake word detector
        if self.wake_word_detector:
            logger.info(f"Wake word detection enabled: '{self.config['wake_word']['phrase']}'")
            self.wake_word_detector.listen(self.handle_query)

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _create_indexer(self, rag_cfg, loader):
        from src.rag.indexer import Indexer
        return Indexer(
            faiss_path=rag_cfg["index"]["faiss_path"],
            db_path=rag_cfg["index"]["db_path"],
            embedder=self.embedder,
            loader=loader,
        )

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        self.embedder.unload()
        self.button.cleanup()
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        if self.doc_watcher:
            self.doc_watcher.stop()


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config()

    # Add file logging if configured
    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    assistant = VoiceAssistant(config)

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        assistant.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    assistant.start()


if __name__ == "__main__":
    main()
