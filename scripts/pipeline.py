"""Full retraining pipeline orchestrator.

Runs every stage end-to-end with the same entry points used during development:
manifest -> split -> normalization -> train (exp_B recipe) -> threshold sweep -> test eval,
then refreshes the frozen artifacts/ bundle.

Standalone:   python scripts/pipeline.py
From the CLI: /train (see scripts/live_detect.py)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "exp_B_specaug"
CONFIG = "configs/exp_B.yaml"

STAGES = [
    ("manifest",      [sys.executable, "scripts/build_manifest.py"]),
    ("split",         [sys.executable, "-m", "src.dataset.split"]),
    ("normalization", [sys.executable, "scripts/compute_norm.py"]),
    ("train",         [sys.executable, "-m", "src.training.train", "--config", CONFIG]),
    ("threshold",     [sys.executable, "-m", "src.evaluation.threshold", "--run", RUN]),
    ("test eval",     [sys.executable, "-m", "src.evaluation.evaluate", "--run", RUN]),
]

# results/<run> -> artifacts/ (same freeze layout as Phase 9)
ARTIFACT_COPIES = [
    ("checkpoints/best_model.pth", "model/mobilenetv2_drone_best.pth"),
    ("config.yaml",                "config/model_config.yaml"),
    ("threshold.json",             "evaluation/threshold.json"),
    ("test_eval.json",             "evaluation/metrics.json"),
]
NORM_COPY = ("data/splits/normalization.json", "preprocessing/normalization.json")


class PipelineError(RuntimeError):
    """Raised when a pipeline stage exits non-zero."""


def _run_stage(name: str, cmd: list[str], log) -> None:
    log(f"[pipeline] === {name} ===")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            log(line.rstrip())
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise PipelineError(f"{name}: interrupted by user")
    if rc != 0:
        raise PipelineError(f"{name}: exited with code {rc}")


def run_pipeline(log=print) -> None:
    """Run all stages in order; stop at the first failure (old artifacts untouched)."""
    for name, cmd in STAGES:
        _run_stage(name, cmd, log)


def refresh_artifacts() -> None:
    """Copy fresh run outputs into artifacts/ (called only after a fully successful pipeline)."""
    run_dir = ROOT / "results" / RUN
    for src_rel, dst_rel in ARTIFACT_COPIES:
        dst = ROOT / "artifacts" / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_dir / src_rel, dst)
    src, dst_rel = NORM_COPY
    dst = ROOT / "artifacts" / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / src, dst)


def new_test_metrics() -> dict:
    g = json.loads((ROOT / "artifacts" / "evaluation" / "metrics.json").read_text())["global"]
    return {k: g[k] for k in ("f1", "recall", "precision", "fpr")}


def main() -> None:
    run_pipeline()
    refresh_artifacts()
    m = new_test_metrics()
    print(f"[pipeline] done. artifacts refreshed. test F1={m['f1']:.4f} "
          f"recall={m['recall']:.4f} FPR={m['fpr']:.4f}")


if __name__ == "__main__":
    main()
