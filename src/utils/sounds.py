"""Generate simple system sounds as WAV files."""
import struct
import wave
import math
import os


def generate_tone(filepath: str, frequency: int, duration_ms: int, sample_rate: int = 22050):
    """Generate a simple sine wave tone and save as WAV."""
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # Apply fade in/out to avoid clicks
        envelope = 1.0
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        if i < fade_samples:
            envelope = i / fade_samples
        elif i > n_samples - fade_samples:
            envelope = (n_samples - i) / fade_samples
        value = int(16000 * envelope * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, value))))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(samples))


def ensure_sounds(sounds_dir: str):
    """Create system sounds if they don't exist."""
    activate_path = os.path.join(sounds_dir, "activate.wav")
    error_path = os.path.join(sounds_dir, "error.wav")

    if not os.path.exists(activate_path):
        # Pleasant ascending two-tone beep
        generate_tone(activate_path, 880, 150)

    if not os.path.exists(error_path):
        # Lower tone for error
        generate_tone(error_path, 330, 300)


if __name__ == "__main__":
    ensure_sounds("data/sounds")
    print("Sound files generated.")
