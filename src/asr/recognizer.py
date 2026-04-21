"""ASR recognizer using GigaAM ONNX."""
import logging
import os
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class Recognizer:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._session = None
        self._vocab = None
        self._is_onnx = False
        self.load()

    def load(self):
        """Load GigaAM model."""
        onnx_path = os.path.join(self.model_path, "v3_ctc.int8.onnx")
        vocab_path = os.path.join(self.model_path, "v3_vocab.txt")

        if os.path.exists(onnx_path):
            self._load_onnx(onnx_path, vocab_path)
        else:
            self._load_pytorch()

    def _load_onnx(self, onnx_path: str, vocab_path: str):
        """Load ONNX model."""
        logger.info("Loading GigaAM ONNX model...")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2

        self._session = ort.InferenceSession(onnx_path, sess_options)
        self._is_onnx = True

        # Load vocabulary: format "symbol index"
        self._blank_id = 0
        if os.path.exists(vocab_path):
            self._vocab = {}
            with open(vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) == 2:
                        symbol = parts[0]
                        idx = int(parts[1])
                        if symbol == "<blk>":
                            self._blank_id = idx
                        else:
                            self._vocab[idx] = symbol
            logger.info(f"GigaAM ONNX loaded, vocab size: {len(self._vocab)}, blank_id: {self._blank_id}")
        else:
            raise FileNotFoundError(f"Vocab not found: {vocab_path}")

        logger.info("GigaAM ONNX model loaded.")

    def _load_pytorch(self):
        """Fallback: load via PyTorch gigaam library."""
        import torch
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([DictConfig, ListConfig])

        logger.info("Loading GigaAM-CTC model (PyTorch)...")
        import gigaam
        self._pytorch_model = gigaam.load_model("ctc")
        self._pytorch_model.eval()
        self._is_onnx = False
        logger.info("GigaAM model loaded (PyTorch).")

    def recognize(self, audio: np.ndarray) -> str:
        """Recognize speech from audio array."""
        if self._is_onnx:
            return self._recognize_onnx(audio)
        else:
            return self._recognize_pytorch(audio)

    def _recognize_onnx(self, audio: np.ndarray) -> str:
        """Recognize using ONNX Runtime."""
        # Compute mel spectrogram
        features = self._compute_mel(audio)  # [1, 64, time]
        feature_lengths = np.array([features.shape[2]], dtype=np.int64)  # [1]

        # Run inference
        outputs = self._session.run(
            None,
            {
                "features": features,
                "feature_lengths": feature_lengths,
            },
        )

        log_probs = outputs[0]  # [batch, time, 34]
        token_ids = np.argmax(log_probs, axis=-1)[0]  # [time]

        # CTC decode
        text = self._ctc_decode(token_ids)
        logger.info(f"Recognized: {text}")
        return text

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute 64-bin mel spectrogram matching GigaAM preprocessing."""
        import torch
        import torchaudio

        # Audio to tensor
        if audio.dtype == np.int16:
            waveform = torch.from_numpy(audio.copy()).float() / 32768.0
        else:
            waveform = torch.from_numpy(audio.copy()).float()

        waveform = waveform.unsqueeze(0)  # [1, samples]

        # GigaAM uses these parameters
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
            normalized=False,
        )

        mel = mel_transform(waveform)  # [1, 64, time]
        mel = torch.log(mel.clamp(min=1e-9))

        return mel.numpy().astype(np.float32)

    def _ctc_decode(self, token_ids: np.ndarray) -> str:
        """CTC greedy decode: collapse repeated tokens, remove blanks."""
        if self._vocab is None:
            return ""

        result = []
        prev_id = -1

        for tid in token_ids:
            tid = int(tid)
            if tid == self._blank_id:
                prev_id = tid
                continue
            if tid == prev_id:
                prev_id = tid
                continue
            if tid in self._vocab:
                result.append(self._vocab[tid])
            prev_id = tid

        text = "".join(result)
        text = text.replace("▁", " ").strip()
        return text

    def _recognize_pytorch(self, audio: np.ndarray) -> str:
        """Fallback: recognize using PyTorch model."""
        import torch
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            with torch.no_grad():
                result = self._pytorch_model.transcribe([f.name])
            os.unlink(f.name)

        text = result[0].strip() if result else ""
        logger.info(f"Recognized: {text}")
        return text
