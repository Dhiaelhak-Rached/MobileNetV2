"""Score a split with a trained checkpoint. Shared by threshold.py and evaluate.py."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.audio.preprocessing import AudioConfig, Preprocessor, ROOT
from src.dataset.dataset import DroneWaveDataset
from src.models.mobilenet_acoustic import build_model
from src.training.train import to_spec


def load_model(ckpt_path: str | Path, device: str = "cuda"):
    state = torch.load(ckpt_path, weights_only=False)
    model = build_model(pretrained=False,
                        dropout=state["config"]["model"]["dropout"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    audio_cfg = AudioConfig(**state["audio"])
    pp = Preprocessor(audio_cfg, state["norm"]["mean"], state["norm"]["std"],
                      device=device)
    return model, pp, state


def score_split(model, pp, csv_path: str | Path, device: str = "cuda",
                gain: float = 1.0) -> tuple[np.ndarray, np.ndarray, list]:
    """-> (y_true, drone_prob, source_datasets). gain<1 simulates distance."""
    ds = DroneWaveDataset(csv_path, pp.cfg)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                    persistent_workers=True, pin_memory=True)
    ys, scores = [], []
    with torch.no_grad():
        for w, y in dl:
            w = (w * gain).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(to_spec(pp, w, device))
            ys.append(y.numpy())
            scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
    return (np.concatenate(ys), np.concatenate(scores),
            [r["source_dataset"] for r in ds.rows])
