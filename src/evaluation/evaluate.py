"""Final test evaluation with the frozen threshold — run ONCE.

    python -m src.evaluation.evaluate --run exp_B_specaug

Reports global metrics, per-source slices, AP UnseenDrone slice, and a gain
sweep (simulated distance) on the test set -> results/<run>/test_eval.json
"""
import argparse
import json
import sys
from collections import defaultdict
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
    thr = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    print(f"frozen threshold: {thr}")

    model, pp, _ = load_model(run_dir / "checkpoints" / "best_model.pth")
    y_true, y_score, sources = score_split(model, pp, ROOT / "data" / "splits" / "test.csv")

    result = {"threshold": thr,
              "global": binary_metrics(y_true, y_score, thr)}

    # per-source slices
    slices = defaultdict(lambda: ([], []))
    for t, s, src in zip(y_true, y_score, sources):
        slices[src][0].append(t)
        slices[src][1].append(s)
    result["per_source"] = {}
    print(f"\n{'source':14s} {'n':>5} {'recall/F1':>10} {'FPR':>7}")
    for src in sorted(slices):
        t, s = slices[src]
        m = binary_metrics(np.array(t), np.array(s), thr)
        result["per_source"][src] = m
        rel = m["recall"] if np.array(t).mean() > 0.5 else m["f1"]
        print(f"{src:14s} {len(t):5d} {rel:10.4f} {m['fpr']:7.4f}")

    # gain sweep on test: simulated distance (all data, global metrics)
    result["gain_sweep"] = {}
    print(f"\n{'gain':>5} {'recall':>7} {'prec':>7} {'F1':>7} {'FPR':>7} {'FNR':>7}")
    for gain in (1.0, 0.3, 0.1, 0.03):
        yt, ys, _ = score_split(model, pp, ROOT / "data" / "splits" / "test.csv",
                                gain=gain)
        m = binary_metrics(yt, ys, thr)
        result["gain_sweep"][str(gain)] = m
        print(f"{gain:5.2f} {m['recall']:7.4f} {m['precision']:7.4f} {m['f1']:7.4f} "
              f"{m['fpr']:7.4f} {m['fnr']:7.4f}")

    (run_dir / "test_eval.json").write_text(json.dumps(result, indent=2))
    g = result["global"]
    print(f"\nTEST (threshold {thr}): acc={g['accuracy']:.4f} F1={g['f1']:.4f} "
          f"recall={g['recall']:.4f} prec={g['precision']:.4f} FPR={g['fpr']:.4f} "
          f"PR-AUC={g['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
