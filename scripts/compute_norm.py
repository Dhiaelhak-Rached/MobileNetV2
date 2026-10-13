"""Compute log-Mel mean/std from the TRAIN split only -> normalization.json.
Val/test/deployment must reuse these exact values."""
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.preprocessing import AudioConfig, Preprocessor, ROOT

TRAIN = ROOT / "data" / "splits" / "train.csv"
OUT = ROOT / "data" / "splits" / "normalization.json"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pp = Preprocessor(AudioConfig(), device=device)  # mean=0, std=1 -> raw log-Mel
    rows = list(csv.DictReader(TRAIN.open()))

    s = ss = n = 0.0
    with torch.no_grad():
        for r in tqdm(rows, desc=f"norm stats ({device})"):
            wav = pp.load(r["path"])
            # non-overlapping windows, last one zero-padded
            for off in range(0, max(1, wav.numel() - 1), pp.cfg.n_samples):
                spec = pp.waveform_to_spec(pp.fix_length(wav, off))
                s += spec.sum().item()
                ss += spec.square().sum().item()
                n += spec.numel()

    mean, var = s / n, ss / n - (s / n) ** 2
    std = max(var, 1e-8) ** 0.5
    OUT.write_text(json.dumps({"mean": mean, "std": std, "split": "train",
                               "config": pp.cfg.__dict__}, indent=2))
    print(f"mean={mean:.4f} std={std:.4f} over {n:.3e} values -> {OUT}")


if __name__ == "__main__":
    main()
