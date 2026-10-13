# Exporting the Drone Model to the NVIDIA Jetson — Workflow Guide

*Audience: junior AI engineer. This document explains **what** we will do and
**why**, step by step. No code is executed yet — each script listed here will be
written and run together, one at a time.*

---

## 1. The big picture: why do we need an "export" at all?

Right now our model lives as a **PyTorch checkpoint**:
`artifacts/model/mobilenetv2_drone_best.pth`. That file is just a bag of
numbers (the learned weights). To *use* it, you need PyTorch, Python, our
source code, CUDA libraries… a whole development environment.

The Jetson is a small computer that sits in the field next to a microphone.
We don't want to ship our whole training project to it. We want to ship a
**self-contained, optimized model file** that runs fast with minimal
dependencies.

The journey has three stops:

```
PyTorch checkpoint (.pth)          ← what we have (training format)
        ↓  [Step A: export]
ONNX model (.onnx)                 ← portable, framework-independent format
        ↓  [Step B: optimize, happens ON the Jetson]
TensorRT engine (.engine / .plan)  ← NVIDIA's fastest runtime, Jetson-native
        ↓
Real-time drone detection on the Jetson
```

**Vocabulary, in plain words:**

- **ONNX** (Open Neural Network Exchange): a standard "save format" for neural
  networks, like PDF is for documents. Any framework/runtime can read it. This
  is the bridge between PyTorch and the Jetson world.
- **TensorRT**: NVIDIA's inference optimizer. It takes the ONNX model and
  rewrites it to run as fast as possible on the specific GPU inside the
  Jetson (fusing layers, picking the fastest math, using 16-bit precision).
- **Engine / .plan file**: the output of TensorRT — a model compiled *for that
  exact Jetson model*. Important: **a TensorRT engine is not portable.** It
  must be built on the Jetson itself (or one with the same GPU + software).
  We never build it on the laptop.

---

## 2. The single most important design decision: the frontend goes *inside* the model

Most projects export only the neural network and compute the spectrogram
separately (in Python, with librosa). **We deliberately did not do that.**

From day one, our whole audio frontend (STFT → mel filterbank → log →
normalize) was written in pure PyTorch/torchaudio operations — *specifically so
it can be exported as part of the model graph*. Payoff:

```
What we export =  [audio samples in] → [spectrogram] → [MobileNetV2] → [probabilities out]
```

The Jetson receives **raw audio numbers** and outputs **P(drone)**. One graph,
one file, zero Python audio code on the device. The exact same math that was
used in training runs in the field — no "it worked on the laptop but behaves
differently on the Jetson" surprises.

### The fp32/fp16 rule (memorize this)

We will run the network in **FP16** (half precision) on the Jetson because it's
~2× faster and halves memory. **But the spectrogram math must stay FP32.**
We learned this the hard way during training: in fp16, the log step of the
spectrogram overflows to infinity. So inside the exported graph:

```
waveform (fp32) → STFT/Mel/log/normalize (fp32) → MobileNetV2 (fp16) → softmax (fp32)
```

When we build the TensorRT engine we enable FP16, and TensorRT automatically
keeps the numerically sensitive parts at higher precision where needed. Our
validation step (below) proves numerically that this is safe.

---

## 3. The workflow, step by step

### Step A — Export PyTorch → ONNX *(on the laptop)*

**Script to write: `scripts/export_onnx.py`**

What it will do, exactly:

1. Load `artifacts/model/mobilenetv2_drone_best.pth` (this file conveniently
   embeds everything we need: the audio config, the normalization mean/std,
   and the class names — so the export can't accidentally use wrong settings).
2. Rebuild the model architecture and load the weights.
3. Wrap the **preprocessor + model** into a single `torch.nn.Module` whose
   input is a raw waveform `[batch, 16000]` and whose output is the two
   probabilities `[batch, 2]`.
4. Call `torch.onnx.export(...)` with:
   - a fixed input shape `[1, 16000]` (we can also enable dynamic batch size),
   - a modern ONNX opset (17+),
   - clear input/output names (`waveform` → `probabilities`).
5. Run the ONNX checker's built-in validation to confirm the file is structurally valid.
6. Save to `artifacts/export/mobilenetv2_drone.onnx`.

**Possible snag (and our plan):** occasionally a specific PyTorch operation has
no ONNX equivalent. Our frontend uses only standard ops (STFT, convolutions,
windowing, log), so this is unlikely — but if export fails on one op, we
replace that single op with an equivalent formulation and retry. We will NOT
silently change the math: any substitution must produce bit-comparable output,
which Step B verifies.

> **What actually happened (worth knowing):** we hit this snag *twice*, and the
> final export reflects both fixes:
> 1. **ONNX exporter**: torchaudio's `MelSpectrogram` uses `torch.stft` with
>    complex output, which ONNX cannot export → the frontend in
>    `export_onnx.py` is written out manually in real-valued form
>    (`power = re² + im²`, mathematically identical).
> 2. **TensorRT 10.3 parser** (on the Jetson): rejects the ONNX `STFT`
>    operator entirely → the STFT is expressed as a single **Conv1d** whose
>    kernels are the windowed DFT basis ("STFT as convolution" — a standard
>    identity). Conv1d is one of the best-supported TensorRT ops.
>
> Both rewrites are verified numerically before the file is written: the
> exported model must match the official PyTorch pipeline to <1e-5 in P(drone)
> on a real file, and Step B re-confirms it on 64 test clips.

### Step B — Validate the ONNX model numerically *(on the laptop)*

**Script to write: `scripts/validate_onnx.py`**

What it will do, exactly:

1. Load both the PyTorch model and the new ONNX model (using `onnxruntime`,
   a lightweight inference engine that runs ONNX files without PyTorch).
2. Feed **the same real test audio files** through both.
3. Compare the outputs: report the **maximum absolute difference** in
   P(drone) between PyTorch and ONNX.
4. Pass/fail criterion: max difference < **1e-3** (0.1%). Small float
   differences are normal; anything bigger means the export changed the math
   and must be fixed.
5. As a stronger check, score a slice of the test set through ONNX and confirm
   the metrics (F1, recall, FPR at τ=0.75) match the PyTorch results.

This step is our insurance policy: **never trust an export you haven't
numerically validated.**

### Step C — Copy to the Jetson and build the TensorRT engine *(on the Jetson)*

**What gets copied:** only `mobilenetv2_drone.onnx` (plus the validation
script and a few test clips). Not the training code, not the dataset.

**Tool:** NVIDIA ships `trtexec` with every Jetson (part of JetPack). It's a
command-line program that compiles ONNX → TensorRT engine. Conceptually:

```bash
trtexec --onnx=mobilenetv2_drone.onnx \
        --fp16 \
        --saveEngine=drone_fp16.engine
```

(Exact flags finalized when we run it — e.g. workspace size, fixed vs dynamic
batch.)

**Script to write: `scripts/build_engine.sh`** — a thin, documented wrapper
around that `trtexec` call so the exact command is recorded and repeatable.

### Step D — Validate the engine on the Jetson

**Script to write: `scripts/validate_engine.py`** *(runs on the Jetson)*

What it will do, exactly:

1. Load the TensorRT engine.
2. Run the same test clips used in Step B.
3. Compare P(drone) against the PyTorch reference values (saved to a small
   JSON/npz file by `validate_onnx.py` for exactly this purpose).
4. Pass/fail: max difference < **1e-2** (1% — slightly looser than Step B
   because fp16 rounding is expected and harmless at this scale; our decision
   threshold is 0.75, so 1% wobble doesn't flip decisions).

### Step E — Real-time detection on the Jetson

**Script to write: `scripts/jetson_live_detect.py`**

What it will do, exactly — a Jetson adaptation of our existing
`scripts/live_detect.py`:

1. Open the microphone, keep a rolling 1-second buffer (16 kHz mono).
2. Every 0.25–0.5 s, run the TensorRT engine on the buffer.
3. Apply the same decision logic as on the laptop: threshold τ = 0.75 and
   **3-of-5 majority voting** (a window only counts as DRONE if at least 3 of
   the last 5 windows agree — this kills isolated false alarms).
4. Print/flag detections in real time.

**Expected performance:** ~3 ms per inference on a Jetson Orin Nano — the model
uses far less than 1% of the device's capacity, leaving headroom for everything
else the system will do later (beamforming, tracking, camera fusion).

---

## 4. The scripts at a glance

| Script | Runs on | What it does |
|---|---|---|
| `scripts/export_onnx.py` | Laptop | PyTorch checkpoint → single ONNX file containing frontend + model |
| `scripts/validate_onnx.py` | Laptop | Proves ONNX output matches PyTorch (<0.1% diff); saves reference outputs |
| `scripts/build_engine.sh` | Jetson | Records the exact `trtexec` command that builds the FP16 engine |
| `scripts/validate_engine.py` | Jetson | Proves the engine matches the reference (<1% diff) |
| `scripts/jetson_live_detect.py` | Jetson | Real-time mic → engine → threshold → 3-of-5 voting → detection |

New folder created along the way: `artifacts/export/` — the ONNX file, the
reference outputs, and build/validation logs. The original frozen artifacts
(`artifacts/model/`, etc.) are **not modified** — export only *reads* the
checkpoint.

---

## 5. What could go wrong (and how we'll know)

| Risk | How we catch it |
|---|---|
| An op doesn't export to ONNX | Export fails loudly in Step A; we reformulate that one op and re-validate |
| ONNX math differs from PyTorch | Step B: >0.1% output difference → fail, investigate |
| FP16 precision hurts accuracy | Step D: >1% difference on the Jetson → fall back to FP32 engine (still easily real-time) |
| Engine built on wrong Jetson/software version | Engine fails to load — that's why we always build on the target device itself |
| Different behavior in the field | The frontend is inside the graph, so field math ≡ training math by construction |

---

## 6. Summary of the whole path

```
artifacts/model/mobilenetv2_drone_best.pth
        │  scripts/export_onnx.py            (laptop)
        ▼
artifacts/export/mobilenetv2_drone.onnx      ← frontend INSIDE the graph
        │  scripts/validate_onnx.py          (laptop, must pass <0.1%)
        ▼
copy .onnx to the Jetson
        │  scripts/build_engine.sh → trtexec --fp16   (on the Jetson)
        ▼
drone_fp16.engine
        │  scripts/validate_engine.py        (Jetson, must pass <1%)
        ▼
scripts/jetson_live_detect.py                (real-time mic detection)
```

When you're ready, tell me to start with Step A and we'll write and run
`scripts/export_onnx.py` together.
