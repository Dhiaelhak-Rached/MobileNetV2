"""Audio frontend: WAV/FLAC -> log-Mel spectrogram. Pure torch/torchaudio ops
(no librosa) so the whole frontend later exports to ONNX/TensorRT on Jetson.

Pipeline: load (soundfile, content-sniffed so FLAC-in-.wav works) -> mono ->
resample 16k -> pad/crop to window -> STFT -> Mel -> log10 -> normalize.

demo() self-check: asserts the hard output shape. Run:
    python src/audio/preprocessing.py
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    window_seconds: float = 1.0
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop_length + 1  # center-padded STFT


class Preprocessor:
    def __init__(self, cfg: AudioConfig | None = None, mean: float = 0.0,
                 std: float = 1.0, device: str = "cpu"):
        self.cfg = cfg or AudioConfig()
        self.mean, self.std = mean, std
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            win_length=self.cfg.win_length,
            hop_length=self.cfg.hop_length,
            window_fn=torch.hann_window,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            power=2.0,
        ).to(device)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def load(self, path: str | Path) -> torch.Tensor:
        """-> mono 16 kHz waveform [n_samples] on self.device."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).mean(dim=0)  # [C,T] -> [T]
        if sr != self.cfg.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(
                    sr, self.cfg.sample_rate).to(self.device)
            wav = self._resamplers[sr](wav.to(self.device))
        return wav.to(self.device)

    def fix_length(self, wav: torch.Tensor, offset: int | None = None) -> torch.Tensor:
        """Pad with zeros or crop to exactly cfg.n_samples. offset=None -> center crop."""
        n = self.cfg.n_samples
        if offset is None:
            offset = max(0, (wav.numel() - n) // 2)
        seg = wav[offset:offset + n]
        if seg.numel() < n:  # short clip, or tail crop at a big offset
            seg = torch.nn.functional.pad(seg, (0, n - seg.numel()))
        return seg

    def waveform_to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """[n_samples] -> [1, n_mels, n_frames]; [B, n_samples] -> [B, 1, n_mels, n_frames]."""
        mel = self.mel(wav)
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        spec = (log_mel - self.mean) / self.std
        return spec.unsqueeze(0) if spec.dim() == 2 else spec.unsqueeze(1)

    def __call__(self, path: str | Path, offset: int | None = None) -> torch.Tensor:
        return self.waveform_to_spec(self.fix_length(self.load(path), offset))

    @staticmethod
    def load_norm(path: str | Path) -> tuple[float, float]:
        d = json.loads(Path(path).read_text())
        return d["mean"], d["std"]


def demo() -> None:
    cfg = AudioConfig()
    pp = Preprocessor(cfg)

    # synthetic 1s sine -> exact shape
    t = torch.arange(cfg.n_samples) / cfg.sample_rate
    spec = pp.waveform_to_spec(torch.sin(2 * np.pi * 440 * t))
    assert spec.shape == (1, cfg.n_mels, cfg.n_frames), spec.shape
    assert spec.shape == (1, 128, 32), spec.shape  # 1s baseline, hard check
    assert torch.isfinite(spec).all()

    # 0.5s config variant
    cfg5 = AudioConfig(window_seconds=0.5)
    pp5 = Preprocessor(cfg5)
    t5 = torch.arange(cfg5.n_samples) / cfg5.sample_rate
    assert pp5.waveform_to_spec(torch.sin(2 * np.pi * 440 * t5)).shape == (1, 128, 16)

    # real file: exercise the FLAC-in-.wav path + resampling
    real = sorted((ROOT / "Data" / "Drone").glob("DroneFault*.wav"))[0]
    s = pp(real)
    assert s.shape == (1, 128, 32) and torch.isfinite(s).all()

    # short clip pads, long file crops
    short = sorted((ROOT / "Data" / "Drone").glob("DADS*.wav"))[0]
    assert pp(short).shape == (1, 128, 32)
    print(f"OK: shapes (1,128,32)/(1,128,16), FLAC-in-wav loads, "
          f"real-file range [{s.min():.1f}, {s.max():.1f}] dB")


if __name__ == "__main__":
    demo()
