# Jetson Export Package

Everything needed to take the frozen drone-detection model
(`artifacts/model/mobilenetv2_drone_best.pth`) and run it in real time on an
NVIDIA Jetson. See `docs/03_jetson_export.md` for the full explanation of the
workflow and the reasoning behind it.

**The two golden rules:**
1. The audio frontend (STFT → mel → log → normalize) lives **inside** the
   exported graph — the Jetson receives raw audio, not spectrograms.
2. Never trust an export you haven't numerically validated. Each step has a
   pass/fail check; do not continue past a FAIL.

---

## Where each script runs

| # | Script | Runs on | What it does |
|---|---|---|---|
| A | `export_onnx.py` | **Laptop** | PyTorch checkpoint → single ONNX file (frontend + model in one graph) |
| B | `validate_onnx.py` | **Laptop** | Proves ONNX ≡ PyTorch (max diff < 0.1%); saves reference outputs |
| C | `build_engine.sh` | **Jetson** | ONNX → TensorRT FP16 engine via `trtexec` |
| D | `validate_engine.py` | **Jetson** | Proves engine ≡ reference (max diff < 1%) |
| E | `jetson_live_detect.py` | **Jetson** | Real-time microphone detection (threshold 0.75 + 3-of-5 voting) |

`trt_engine.py` is a small shared helper (TensorRT runner used by D and E) —
you never run it directly.

---

## Step-by-step

### On the laptop (from the project root `E:\DroneDetectionModel`)

```bash
# one-time: extra export dependencies (not needed for training)
uv pip install --python .venv/Scripts/python.exe onnx onnxruntime

# Step A: export  ->  artifacts/export/mobilenetv2_drone.onnx
.venv/Scripts/python.exe Jetson/export_onnx.py

# Step B: validate (must print PASS)  ->  artifacts/export/reference_outputs.npz
.venv/Scripts/python.exe Jetson/validate_onnx.py
```

### Copy to the Jetson

Only these files are needed on the Jetson (e.g. into `~/drone/`):

```
mobilenetv2_drone.onnx        ← from artifacts/export/
reference_outputs.npz         ← from artifacts/export/
Jetson/build_engine.sh
Jetson/validate_engine.py
Jetson/jetson_live_detect.py
Jetson/trt_engine.py
```

### On the Jetson (from `~/drone/`)

```bash
# Step C: build the FP16 engine (trtexec comes with JetPack)
bash build_engine.sh                       # -> drone_fp16.engine

# Step D: validate against the laptop reference (must print PASS)
python3 validate_engine.py --engine drone_fp32.engine

# Step E: live detection
python3 jetson_live_detect.py --engine drone_fp32.engine
# then at the prompt:  /mic  → pick your microphone →  /detect
```

---

## Dependencies

| Machine | Needed |
|---|---|
| Laptop | the project `.venv` + `pip install onnx onnxruntime` |
| Jetson | JetPack (provides `tensorrt` + `trtexec` + CUDA), `python3`, `pip install numpy sounddevice`, and a PyTorch-for-Jetson wheel (NVIDIA provides one per JetPack version) |

---

## Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| `export_onnx.py` fails on an op | an op has no ONNX equivalent | report the exact error — the op gets reformulated (math must stay identical) |
| `validate_onnx.py` prints FAIL | export changed the math | do NOT deploy; investigate the diff first |
| `trtexec not found` | PATH issue | use `/usr/src/tensorrt/bin/trtexec` |
| `validate_engine.py` prints FAIL | FP16 hurt the math | rebuild with the FP32 fallback (comment at the bottom of `build_engine.sh`) |
| Engine won't load | built on a different GPU/JetPack | always build the engine **on the target Jetson itself** |
| No mic listed in `/detect` | permission/driver | check `arecord -l`, add user to the `audio` group |

## Notes

- The frozen production checkpoint is only **read**, never modified, by any of
  these scripts.
- Decision threshold 0.75 is the production default; override with
  `--threshold` (validation sweep showed 0.70–0.80 all viable, FPR 0.003–0.015).
- Expected Jetson Orin Nano performance: ~3 ms per 1-second window — less than
  1% of the device's capacity.
