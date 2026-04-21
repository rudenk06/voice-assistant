import logging
import wave
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class Player:
    def play(self, audio: np.ndarray, sample_rate: int = 22050) -> None:
        """Play audio numpy array through speakers."""
        if len(audio) == 0:
            return
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def play_sound(self, sound_path: str) -> None:
        """Play a WAV sound file."""
        try:
            with wave.open(sound_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                if wf.getnchannels() == 2:
                    audio = audio[::2]  # Take left channel only
            self.play(audio, sample_rate)
        except Exception as e:
            logger.warning(f"Could not play sound {sound_path}: {e}")
