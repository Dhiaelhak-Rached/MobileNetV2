"""Step B (run on the LAPTOP): prove the ONNX model matches PyTorch numerically.

    python Jetson/validate_onnx.py [--n 64]

What it does:
  1. Scores N test clips (balanced drone/non_drone, center crop — same as the
     official evaluation) with BOTH the PyTorch model and the ONNX model.
  2. Reports the max absolute difference in P(drone). PASS if < 1e-3.
  3. Saves the input waveforms + PyTorch reference probabilities + labels to
     artifacts/export/reference_outputs.npz — the Jetson-side validation
     (validate_engine.py) compares against exactly these numbers.

Never trust an export you haven't numerically validated.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import ROOT, load_model

ONNX = ROOT / "artifacts" / "export" / "mobilenetv2_drone.onnx"
REF = ROOT / "artifacts" / "export" / "reference_outputs.npz"
TOL = 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="clips to compare (half drone)")
    args = ap.parse_args()

    import onnxruntime as ort

    model, pp, _ = load_model(ROOT / "artifacts" / "model" / "mobilenetv2_drone_best.pth",
                              device="cpu")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    rows = list(csv.DictReader((ROOT / "data" / "splits" / "test.csv").open()))
    per_class = args.n // 2
    picks = [r for r in rows if r["class"] == "drone"][:per_class] + \
            [r for r in rows if r["class"] == "non_drone"][:per_class]

    wavs, labels, pt_probs, ox_probs = [], [], [], []
    for r in picks:
        w = pp.fix_length(pp.load(r["path"]), offset=None)  # center crop, like eval
        with torch.no_grad():
            logits = model(pp.waveform_to_spec(w.unsqueeze(0)))
            p_pt = torch.softmax(logits, 1)[0].numpy()
        p_ox = sess.run(["probabilities"], {"waveform": w.numpy()[None]})[0][0]
        wavs.append(w.numpy())
        labels.append(1 if r["class"] == "drone" else 0)
        pt_probs.append(p_pt)
        ox_probs.append(p_ox)

    pt = np.array(pt_probs)[:, 1]   # P(drone)
    ox = np.array(ox_probs)[:, 1]
    diff = np.abs(pt - ox)
    print(f"compared {len(picks)} clips: max |P_pt - P_onnx| = {diff.max():.2e} "
          f"(mean {diff.mean():.2e}, tolerance {TOL:.0e})")

    np.savez(REF, waveforms=np.array(wavs), labels=np.array(labels),
             pytorch_probs=np.array(pt_probs))
    print(f"reference outputs saved to {REF}")

    if diff.max() < TOL:
        print("PASS: ONNX matches PyTorch. Copy the .onnx to the Jetson.")
    else:
        print("FAIL: export changed the math — do NOT deploy, investigate first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
