import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record_until_silence(
        self,
        silence_threshold: float = 0.03,
        silence_duration: float = 1.5,
        max_duration: float = 15.0,
    ) -> np.ndarray:
        """Record audio from microphone until silence is detected."""
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(self.sample_rate * chunk_duration)
        silence_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)

        frames = []
        silent_count = 0
        has_speech = False

        logger.info("Recording started...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=chunk_samples,
        ) as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk_samples)
                frames.append(data.copy())

                amplitude = np.abs(data).mean() / 32768.0

                if amplitude > silence_threshold:
                    has_speech = True
                    silent_count = 0
                else:
                    silent_count += 1

                if has_speech and silent_count >= silence_chunks:
                    break

        if not frames:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(frames, axis=0).flatten()
        logger.info(f"Recorded {len(audio) / self.sample_rate:.1f}s of audio")
        return audio

    def record_fixed(self, duration: float) -> np.ndarray:
        """Record audio for a fixed duration."""
        samples = int(self.sample_rate * duration)
        audio = sd.rec(
            samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
        )
        sd.wait()
        return audio.flatten()
