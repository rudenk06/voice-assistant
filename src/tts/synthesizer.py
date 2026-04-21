import io
import logging
import wave
import numpy as np

logger = logging.getLogger(__name__)


class Synthesizer:
    """Text-to-speech using Piper TTS."""

    def __init__(self, model_path: str, sample_rate: int = 22050):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._voice = None

    def load(self):
        """Load Piper voice model."""
        from piper import PiperVoice
        logger.info(f"Loading Piper TTS from {self.model_path}...")

        # model_path should point to the directory containing the .onnx file
        import os
        onnx_files = [f for f in os.listdir(self.model_path) if f.endswith(".onnx")]
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {self.model_path}")

        onnx_path = os.path.join(self.model_path, onnx_files[0])
        self._voice = PiperVoice.load(onnx_path)
        logger.info("Piper TTS loaded.")

    
    
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio array (int16)."""
        if self._voice is None:
            self.load()
        if not text.strip():
            return np.array([], dtype=np.int16)

        import tempfile
        import os

        tmp_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with wave.open(tmp_path, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

        with wave.open(tmp_path, "rb") as wav_file:
            self.sample_rate = wav_file.getframerate()
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        os.remove(tmp_path)
        logger.info(f"Synthesized {len(audio) / self.sample_rate:.1f}s of audio")
        return audio
