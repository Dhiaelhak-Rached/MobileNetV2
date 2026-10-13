"""Step D (run on the JETSON): prove the TensorRT engine matches the reference.

    python3 validate_engine.py --engine drone_fp16.engine

Compares the engine's P(drone) against the PyTorch reference values computed
on the laptop (reference_outputs.npz) for the exact same audio clips.
PASS if max difference < 1e-2 (looser than the laptop check — FP16 rounding
is expected and harmless; the decision threshold is 0.75, far from any wobble).
Also prints recall/FPR at the production threshold as a sanity check.
"""
import argparse
from pathlib import Path

import numpy as np

from trt_engine import TrtEngine

REF = Path(__file__).resolve().parent / "reference_outputs.npz"
TOL = 1e-2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="drone_fp16.engine")
    ap.add_argument("--reference", default=str(REF))
    ap.add_argument("--threshold", type=float, default=0.75)
    args = ap.parse_args()

    data = np.load(args.reference)
    wavs, labels, ref = data["waveforms"], data["labels"], data["pytorch_probs"][:, 1]

    eng = TrtEngine(args.engine, n_samples=wavs.shape[1])
    got = np.array([eng.infer(w)[1] for w in wavs])  # P(drone)

    diff = np.abs(got - ref)
    print(f"compared {len(wavs)} clips: max |P_engine - P_reference| = {diff.max():.2e} "
          f"(mean {diff.mean():.2e}, tolerance {TOL:.0e})")

    pred, lab = got >= args.threshold, labels == 1
    tp = int((pred & lab).sum()); fp = int((pred & ~lab).sum())
    fn = int((~pred & lab).sum()); tn = int((~pred & ~lab).sum())
    print(f"at threshold {args.threshold}: recall={tp / max(1, tp + fn):.4f} "
          f"FPR={fp / max(1, fp + tn):.4f} (TP={tp} FP={fp} FN={fn} TN={tn})")

    if diff.max() < TOL:
        print("PASS: engine matches reference. Ready for jetson_live_detect.py.")
    else:
        print("FAIL: engine output drifted — rebuild with FP32 "
              "(see comment at the bottom of build_engine.sh) and re-validate.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
