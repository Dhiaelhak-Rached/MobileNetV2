"""Step E (run on the JETSON): real-time drone detection with the TensorRT engine.

    python3 jetson_live_detect.py --engine drone_fp16.engine

Standalone: needs only numpy, sounddevice, torch and tensorrt (all in JetPack
or pip) — NOT the training project. The ONNX graph already contains the whole
audio frontend, so this script feeds raw microphone samples straight into the
engine.

Decision logic is identical to the laptop demo (scripts/live_detect.py):
threshold 0.75 + temporal voting (DRONE only if >= 3 of the last 5 windows
are above threshold — single-window flicker can't trigger an alarm).

Commands:
    /mic      list input devices, pick one
    /detect   start real-time detection (Ctrl+C to return to prompt)
    /quit     exit
"""
import argparse
import time
from collections import deque

import numpy as np
import sounddevice as sd

from trt_engine import TrtEngine

SAMPLE_RATE = 16000   # the engine expects 16 kHz mono
WINDOW_S = 1.0        # 1-second analysis window
HOP = 0.5             # seconds between inferences
VOTES, WINDOW = 3, 5  # DRONE if >= 3 of last 5 windows exceed threshold


def list_inputs() -> list[int]:
    ids = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            ids.append(i)
            print(f"  [{i}] {d['name']} ({d['default_samplerate']:.0f} Hz)")
    return ids


def detect(eng: TrtEngine, thr: float, device_index: int) -> None:
    n = int(SAMPLE_RATE * WINDOW_S)
    ring = np.zeros(n, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal ring
        ring = np.roll(ring, -frames)
        ring[-frames:] = indata[:, 0]

    dev_info = sd.query_devices(device_index, "input")
    print(f"mic: {dev_info['name']} | threshold {thr:.2f} | "
          f"decision {VOTES}/{WINDOW} votes | Ctrl+C to stop")
    history: deque = deque(maxlen=WINDOW)
    # PortAudio resamples to 16 kHz if the mic's native rate differs.
    with sd.InputStream(device=device_index, samplerate=SAMPLE_RATE, channels=1,
                        blocksize=1024, dtype="float32", callback=callback):
        time.sleep(WINDOW_S)  # fill the buffer before the first inference
        try:
            while True:
                prob = float(eng.infer(ring.copy())[1])  # P(drone)
                history.append(prob >= thr)
                decision = sum(history) >= VOTES
                bar = "#" * int(prob * 40)
                print(f"\r{time.strftime('%H:%M:%S')} drone {prob:.3f} |{bar:40s}| "
                      f"{'** DRONE **' if decision else ''}   ", end="", flush=True)
                time.sleep(HOP)
        except KeyboardInterrupt:
            print("\n[stopped — back to prompt]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="drone_fp16.engine")
    ap.add_argument("--threshold", type=float, default=0.75)
    args = ap.parse_args()

    print("loading engine...", end=" ", flush=True)
    eng = TrtEngine(args.engine)
    print(f"ready (threshold {args.threshold:.2f})")
    print("commands: /mic  /detect  /quit")

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
            if device_index is None:
                print("pick a mic first: /mic")
            else:
                detect(eng, args.threshold, device_index)
        elif cmd in ("/quit", "/exit"):
            break
        elif cmd:
            print("unknown command — /mic /detect /quit")


if __name__ == "__main__":
    main()
