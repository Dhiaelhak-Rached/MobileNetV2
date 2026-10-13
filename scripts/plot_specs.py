"""Sanity plot: one spectrogram per source/class combo -> results/spectrogram_sanity.png"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__file__ and __import__("pathlib").Path(__file__).resolve().parents[1]))
from src.audio.preprocessing import Preprocessor, AudioConfig, ROOT

rows = list(csv.DictReader((ROOT / "data/splits/train.csv").open()))
picks = []
for want in [("drone", "X4"), ("drone", "DroneFault"), ("non_drone", "US8K"), ("non_drone", "ESC50")]:
    picks.append(next(r for r in rows if (r["class"], r["source_dataset"]) == want))

mean, std = Preprocessor.load_norm(ROOT / "data/splits/normalization.json")
pp = Preprocessor(AudioConfig(), mean, std)

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, r in zip(axes, picks):
    spec = pp(r["path"])
    ax.imshow(spec[0].numpy(), origin="lower", aspect="auto")
    ax.set_title(f"{r['class']}/{r['source_dataset']}", fontsize=9)
os.makedirs(ROOT / "results", exist_ok=True)
plt.tight_layout()
plt.savefig(ROOT / "results/spectrogram_sanity.png", dpi=90)
print("saved:", [os.path.basename(r["path"])[:30] for r in picks])
