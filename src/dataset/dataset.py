"""Dataset: serves 1s 16kHz waveform windows + label.

Waveforms are decoded/resampled once and cached to data/cache/*.npy (lazy, on
first read) so epochs don't re-decode 23h of audio. Spec computation happens
batched on GPU in the training loop (see src/training/train.py) — same split as
the future Jetson frontend.

Train-only waveform augmentations (seeded per sample/epoch): random crop,
optional noise mixing (noise_cfg), optional random gain in dB (gain_cfg).
Val/test are always deterministic center-crop, never augmented.

Labels: non_drone=0, drone=1 (CLASS_NAMES order).
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.audio.preprocessing import AudioConfig, Preprocessor, ROOT

CLASS_NAMES = ["non_drone", "drone"]  # index 1 = drone
CACHE = ROOT / "data" / "cache"


class DroneWaveDataset(Dataset):
    def __init__(self, csv_path: str | Path, cfg: AudioConfig | None = None,
                 train: bool = False, seed: int = 42, noise_cfg: dict | None = None,
                 gain_cfg: dict | None = None):
        self.rows = list(csv.DictReader(Path(csv_path).open()))
        self.cfg = cfg or AudioConfig()
        self.train = train
        self.seed = seed
        self._pp: Preprocessor | None = None  # lazy, one per worker process
        # noise mixing: pool of non_drone files from the same (train) split
        self.noise_cfg = noise_cfg or {}
        self.noise_paths = [r["path"] for r in self.rows if r["class"] == "non_drone"] \
            if self.noise_cfg.get("enabled") and train else []
        # random gain (distance simulation): train-only waveform scaling
        self.gain_cfg = gain_cfg or {}

    @property
    def pp(self) -> Preprocessor:
        if self._pp is None:
            self._pp = Preprocessor(self.cfg)  # CPU: decode + resample only
        return self._pp

    def _cache_path(self, path: str) -> Path:
        h = hashlib.md5(path.encode()).hexdigest()[:16]
        return CACHE / f"{h}.npy"

    def waveform(self, path: str) -> torch.Tensor:
        """16kHz mono waveform, disk-cached after first decode."""
        cp = self._cache_path(path)
        if cp.exists():
            return torch.from_numpy(np.load(cp))
        wav = self.pp.load(path).cpu().numpy()
        os.makedirs(CACHE, exist_ok=True)
        np.save(cp, wav)
        return torch.from_numpy(wav)

    def __len__(self) -> int:
        return len(self.rows)

    def _mix_noise(self, seg: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """Add a random non_drone clip at a random SNR (train-only augmentation)."""
        prob = self.noise_cfg.get("prob", 0.5)
        if not self.noise_paths or torch.rand((), generator=g) > prob:
            return seg
        idx = int(torch.randint(0, len(self.noise_paths), (1,), generator=g))
        noise = self.waveform(self.noise_paths[idx])
        n = seg.numel()
        off = int(torch.randint(0, max(1, noise.numel() - n + 1), (1,), generator=g)) \
            if noise.numel() > n else 0
        noise = self.pp.fix_length(noise, off)
        snr = self.noise_cfg.get("snr_min", 0) + torch.rand((), generator=g).item() * \
            (self.noise_cfg.get("snr_max", 20) - self.noise_cfg.get("snr_min", 0))
        ps, pn = seg.pow(2).mean(), noise.pow(2).mean().clamp(min=1e-10)
        scaled = noise * torch.sqrt(ps / (pn * 10 ** (snr / 10)))
        return seg + scaled

    def _apply_gain(self, seg: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """Scale the waveform by a random gain in dB (train-only augmentation).

        Simulates the same source at a greater distance. Multiplier = 10^(db/20);
        through power=2 mel + log10 this is an additive shift of gain_db on the
        log-mel, fully visible under the fixed global normalization."""
        lo = self.gain_cfg.get("db_min", -30.0)
        hi = self.gain_cfg.get("db_max", 0.0)
        gain_db = lo + torch.rand((), generator=g).item() * (hi - lo)
        return seg * (10.0 ** (gain_db / 20.0))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.rows[i]
        wav = self.waveform(r["path"])
        n = self.cfg.n_samples
        g = torch.Generator().manual_seed(hash((self.seed, i, epoch())) % 2**31)
        if self.train and wav.numel() > n:
            offset = int(torch.randint(0, wav.numel() - n + 1, (1,), generator=g))
        else:
            offset = None  # center crop / zero-pad
        seg = self.pp.fix_length(wav, offset)
        if self.train and self.noise_paths:
            seg = self._mix_noise(seg, g)
        if self.train and self.gain_cfg.get("enabled"):
            seg = self._apply_gain(seg, g)
        return seg, CLASS_NAMES.index(r["class"])


_epoch = 0

def set_epoch(e: int) -> None:
    """Train loop calls this each epoch so random crops vary per epoch but stay reproducible."""
    global _epoch
    _epoch = e

def epoch() -> int:
    return _epoch


def demo() -> None:
    from torch.utils.data import DataLoader
    cfg = AudioConfig()
    for split, train in [("val", False), ("train", True)]:
        ds = DroneWaveDataset(ROOT / "data" / "splits" / f"{split}.csv", cfg, train=train)
        dl = DataLoader(ds, batch_size=8, shuffle=train, num_workers=0)
        w, y = next(iter(dl))
        assert w.shape == (8, cfg.n_samples), w.shape
        assert set(y.tolist()) <= {0, 1}
        assert torch.isfinite(w).all()
    # eval determinism: same item twice -> identical
    ds = DroneWaveDataset(ROOT / "data" / "splits" / "val.csv", cfg)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)
    # gain augmentation: train-only, scales amplitude, db<=0 never amplifies
    ds_gain = DroneWaveDataset(ROOT / "data" / "splits" / "train.csv", cfg, train=True,
                               gain_cfg={"enabled": True, "db_min": -30.0, "db_max": 0.0})
    seg, _ = ds_gain[0]
    assert seg.shape == (cfg.n_samples,) and torch.isfinite(seg).all()
    ref = DroneWaveDataset(ROOT / "data" / "splits" / "train.csv", cfg, train=True)[0][0]
    assert seg.abs().max() <= ref.abs().max() + 1e-6
    print("OK: dataset shapes, labels, cache, eval determinism, gain augmentation")


if __name__ == "__main__":
    demo()
