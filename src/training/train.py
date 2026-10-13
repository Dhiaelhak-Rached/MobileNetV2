"""Training loop. Usage:
    python -m src.training.train --config configs/baseline.yaml
    python -m src.training.train --config configs/baseline.yaml --epochs 2 --subset 500  # smoke

Per run: results/<run_name>/{checkpoints/best_model.pth, checkpoints/last_model.pth,
metrics.json, config.yaml, curves.png, confusion_matrix.png, pr_curve.png}
Model selection: validation F1.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.audio.augmentation import SpecAugment
from src.audio.preprocessing import AudioConfig, Preprocessor, ROOT
from src.dataset.dataset import CLASS_NAMES, DroneWaveDataset, set_epoch
from src.models.mobilenet_acoustic import (backbone_parameters, build_model,
                                           head_parameters)
from src.training.metrics import binary_metrics


def to_spec(pp, w, device):
    """Frontend must run in fp32 — STFT/mel overflows in fp16 autocast."""
    with torch.amp.autocast("cuda", enabled=False):
        return pp.waveform_to_spec(w)


def evaluate(model, pp, loader, device) -> tuple[float, dict, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    ys, scores = [], []
    with torch.no_grad():
        for w, y in loader:
            w, y = w.to(device, non_blocking=True), y.to(device, non_blocking=True)
            spec = to_spec(pp, w, device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(spec)
                loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    return total_loss / n, binary_metrics(y_true, y_score), y_true, y_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke test)")
    ap.add_argument("--subset", type=int, default=None, help="use first N train rows (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg["epochs"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = ROOT / "results" / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / args.config, run_dir / "config.yaml")

    audio_cfg = AudioConfig(**cfg["audio"])
    mean, std = Preprocessor.load_norm(ROOT / cfg["data"]["norm_json"])
    pp = Preprocessor(audio_cfg, mean, std, device=device)

    train_ds = DroneWaveDataset(ROOT / cfg["data"]["train_csv"], audio_cfg, train=True,
                                seed=cfg["seed"], noise_cfg=cfg["augment"].get("noise"),
                                gain_cfg=cfg["augment"].get("gain"))
    if args.subset:
        train_ds.rows = train_ds.rows[: args.subset]
    val_ds = DroneWaveDataset(ROOT / cfg["data"]["val_csv"], audio_cfg)
    train_ld = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"], persistent_workers=True,
                          pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], persistent_workers=True,
                        pin_memory=True)

    model = build_model(cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW([
        {"params": list(backbone_parameters(model)), "lr": tcfg["lr_backbone"]},
        {"params": list(head_parameters(model)), "lr": tcfg["lr_head"]},
    ], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=tcfg["lr_min"])
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    specaug = SpecAugment(cfg["augment"]["freq_mask"], cfg["augment"]["time_mask"]) \
        if cfg["augment"]["specaug"] else None

    history = []
    best_f1 = -1.0
    ckpt_extra = {"class_names": CLASS_NAMES, "audio": cfg["audio"],
                  "norm": {"mean": mean, "std": std}, "config": cfg}

    for ep in range(epochs):
        set_epoch(ep)
        frozen = ep < tcfg["freeze_epochs"]
        for p in backbone_parameters(model):
            p.requires_grad = not frozen

        model.train()
        t0 = time.time()
        tl = tn = 0
        for w, y in tqdm(train_ld, desc=f"ep {ep+1}/{epochs}{' [frozen]' if frozen else ''}"):
            w, y = w.to(device, non_blocking=True), y.to(device, non_blocking=True)
            spec = to_spec(pp, w, device)
            if specaug:
                spec = specaug(spec)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(spec)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            tn += len(y)
        sched.step()

        val_loss, m, _, _ = evaluate(model, pp, val_ld, device)
        row = {"epoch": ep + 1, "train_loss": tl / tn, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in m.items() if isinstance(v, float)},
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"  train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_f1={m['f1']:.4f} val_recall={m['recall']:.4f} val_fpr={m['fpr']:.4f}")

        state = {"model": model.state_dict(), "epoch": ep + 1,
                 "val_f1": m["f1"], **ckpt_extra}
        torch.save(state, ckpt_dir / "last_model.pth")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(state, ckpt_dir / "best_model.pth")

    # final artifacts from the BEST checkpoint
    state = torch.load(ckpt_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(state["model"])
    val_loss, m, y_true, y_score = evaluate(model, pp, val_ld, device)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([h["train_loss"] for h in history], label="train")
    ax[0].plot([h["val_loss"] for h in history], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot([h["val_f1"] for h in history], label="F1")
    ax[1].plot([h["val_recall"] for h in history], label="recall")
    ax[1].plot([h["val_fpr"] for h in history], label="FPR")
    ax[1].set_title("val metrics"); ax[1].legend()
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, (y_score >= 0.5).astype(int),
                                            display_labels=CLASS_NAMES, ax=ax)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    fig.savefig(run_dir / "pr_curve.png", dpi=90); plt.close(fig)

    (run_dir / "metrics.json").write_text(json.dumps(
        {"best_val": m, "best_epoch": state["epoch"], "history": history}, indent=2))
    print(f"done. best epoch {state['epoch']}: val F1={m['f1']:.4f} "
          f"recall={m['recall']:.4f} FPR={m['fpr']:.4f} PR-AUC={m['pr_auc']:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
