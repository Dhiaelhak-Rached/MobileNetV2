"""Interactive real-time drone detection CLI.

    python scripts/live_detect.py

Commands:
    /mic      list input devices, pick one
    /detect   start real-time detection (Ctrl+C to return to prompt)
    /train    full retrain on the data in Data/ (manifest -> split -> norm ->
              train -> threshold -> test eval), then refresh artifacts/ and
              reload the new model into this session
    /quit     exit
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import ROOT, load_model
from src.training.train import to_spec
import scripts.pipeline as pipeline

HOP = 0.5       # seconds between inferences
VOTES = 3       # DRONE if >= 3 of ...
WINDOW = 5      # ... last 5 windows are above threshold

_training = False  # True while a /train pipeline run is in progress


def load_artifacts(torch_device="cuda"):
    """Load model + preprocessor + threshold from the frozen artifacts bundle."""
    model, pp, _ = load_model(ROOT / "artifacts/model/mobilenetv2_drone_best.pth",
                              torch_device)
    thr = json.loads((ROOT / "artifacts/evaluation/threshold.json").read_text())["threshold"]
    return model, pp, thr


def cmd_train(log=print) -> bool:
    """Run the full retrain pipeline. Returns True on success (model reloaded)."""
    global model, pp, thr, _training
    if _training:
        log("training already in progress")
        return False
    log("new .wav files must already be in Data/Drone or Data/NoDrone "
        "(use the UNSEEN_* prefix for new drone models)")
    _training = True
    try:
        pipeline.run_pipeline(log)
        pipeline.refresh_artifacts()
        model, pp, thr = load_artifacts()
        m = pipeline.new_test_metrics()
        log(f"retrain done — model reloaded (threshold {thr:.2f}). "
            f"new test F1={m['f1']:.4f} recall={m['recall']:.4f} FPR={m['fpr']:.4f}")
        return True
    except pipeline.PipelineError as e:
        log(f"retrain aborted: {e} — old model stays active")
        return False
    finally:
        _training = False


def list_inputs() -> list[int]:
    ids = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            ids.append(i)
            print(f"  [{i}] {d['name']} ({d['default_samplerate']:.0f} Hz)")
    return ids


def detect(model, pp, thr: float, device_index: int, torch_device: str) -> None:
    dev_info = sd.query_devices(device_index, "input")
    native_sr = int(dev_info["default_samplerate"])
    n_native = int(native_sr * pp.cfg.window_seconds)
    resampler = None
    if native_sr != pp.cfg.sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(native_sr, pp.cfg.sample_rate).to(torch_device)

    ring = np.zeros(n_native, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal ring
        ring = np.roll(ring, -frames)
        ring[-frames:] = indata[:, 0]

    print(f"mic: {dev_info['name']} @ {native_sr} Hz | threshold {thr:.2f} | "
          f"decision {VOTES}/{WINDOW} votes | Ctrl+C to stop")
    history: deque = deque(maxlen=WINDOW)
    with sd.InputStream(device=device_index, samplerate=native_sr, channels=1,
                        blocksize=1024, dtype="float32", callback=callback):
        time.sleep(pp.cfg.window_seconds)  # fill buffer
        try:
            while True:
                wav = torch.from_numpy(ring.copy()).to(torch_device)
                if resampler is not None:
                    wav = resampler(wav)[: pp.cfg.n_samples]
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch_device == "cuda"):
                    logits = model(to_spec(pp, wav, torch_device).unsqueeze(0))
                prob = torch.softmax(logits.float(), 1)[0, 1].item()
                history.append(prob >= thr)
                decision = sum(history) >= VOTES
                bar = "#" * int(prob * 40)
                print(f"\r{time.strftime('%H:%M:%S')} drone {prob:.3f} |{bar:40s}| "
                      f"{'** DRONE **' if decision else ''}   ", end="", flush=True)
                time.sleep(HOP)
        except KeyboardInterrupt:
            print("\n[stopped — back to prompt]")


def main() -> None:
    global model, pp, thr
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model...", end=" ", flush=True)
    model, pp, thr = load_artifacts(torch_device)
    print(f"ready ({torch_device}, threshold {thr:.2f})")
    print("commands: /mic  /detect  /train  /quit")

    device_index: int | None = None
    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "/mic":
            ids = list_inputs()
            try:
                choice = int(input("choose device number: ").strip())
            except (ValueError, EOFError, KeyboardInterrupt):
                print("cancelled")
                continue
            if choice in ids:
                device_index = choice
                print(f"selected [{choice}] {sd.query_devices(choice)['name']}")
            else:
                print("invalid device")
        elif cmd == "/detect":
            if _training:
                print("training in progress — /detect disabled until it finishes")
            elif device_index is None:
                print("pick a mic first: /mic")
            else:
                detect(model, pp, thr, device_index, torch_device)
        elif cmd == "/train":
            cmd_train()
        elif cmd in ("/quit", "/exit"):
            break
        elif cmd:
            print("unknown command — /mic /detect /train /quit")


if __name__ == "__main__":
    main()
