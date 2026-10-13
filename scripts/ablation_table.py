"""Phase 7: ablation comparison table from results/*/metrics.json."""
import json
from pathlib import Path

RUNS = {
    "A 1s, no aug": "baseline_A",
    "B 1s, specaug": "exp_B_specaug",
    "C 1s, specaug+noise": "exp_C_specaug_noise",
    "D 0.5s, specaug": "exp_D_0.5s_specaug",
}

print(f"{'experiment':22s} {'F1':>7} {'recall':>7} {'prec':>7} {'FPR':>7} {'PR-AUC':>7} {'ep':>3}")
summary = {}
for label, run in RUNS.items():
    d = json.loads((Path("results") / run / "metrics.json").read_text())
    m = d["best_val"]
    summary[label] = {**{k: m[k] for k in ("f1", "recall", "precision", "fpr", "pr_auc", "roc_auc")},
                      "best_epoch": d["best_epoch"]}
    print(f"{label:22s} {m['f1']:7.4f} {m['recall']:7.4f} {m['precision']:7.4f} "
          f"{m['fpr']:7.4f} {m['pr_auc']:7.4f} {d['best_epoch']:3d}")

Path("results/ablation_summary.json").write_text(json.dumps(summary, indent=2))
