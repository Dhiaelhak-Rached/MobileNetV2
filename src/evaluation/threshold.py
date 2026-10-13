"""Threshold sweep on the VALIDATION split; freeze the operational threshold.

    python -m src.evaluation.threshold --run exp_B_specaug

Picks the threshold maximizing val F1 (reporting recall/FPR across the sweep so
the recall-vs-alarms tradeoff is visible), saves results/<run>/threshold.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.common import ROOT, load_model, score_split
from src.training.metrics import binary_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp_B_specaug")
    args = ap.parse_args()
    run_dir = ROOT / "results" / args.run

    model, pp, _ = load_model(run_dir / "checkpoints" / "best_model.pth")
    y_true, y_score, _ = score_split(model, pp, ROOT / "data" / "splits" / "val.csv")

    sweep = []
    print(f"{'thr':>5} {'F1':>7} {'recall':>7} {'prec':>7} {'FPR':>7}")
    for thr in np.arange(0.30, 0.91, 0.05):
        m = binary_metrics(y_true, y_score, float(thr))
        sweep.append(m)
        print(f"{thr:5.2f} {m['f1']:7.4f} {m['recall']:7.4f} {m['precision']:7.4f} {m['fpr']:7.4f}")

    best = max(sweep, key=lambda m: m["f1"])
    out = {"threshold": best["threshold"], "criterion": "max val F1",
           "val_metrics": best, "sweep": sweep}
    (run_dir / "threshold.json").write_text(json.dumps(out, indent=2))
    print(f"\nchosen threshold: {best['threshold']:.2f} "
          f"(val F1={best['f1']:.4f} recall={best['recall']:.4f} FPR={best['fpr']:.4f})")


if __name__ == "__main__":
    main()
