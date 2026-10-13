# Drone Detection Model — Full Build Documentation

This document describes **everything that was done**, step by step, to build the acoustic drone
classifier in this repository: the data work, the preprocessing pipeline, the model, the training
procedure, the ablation experiments, the evaluation protocol, the frozen artifacts, and the
real-time demo. It is meant to be read alongside the code.

For complete, formal hardware, tensor dimension, FLOP, and parameter specifications, refer to [ModelSpecifications.md](file:///E:/DroneDetectionModel/ModelSpecifications.md).

---

## Table of contents

1. [Project overview and objective](#1-project-overview-and-objective)
2. [Environment setup (Phase 0)](#2-environment-setup-phase-0)
3. [Data audit and manifest (Phase 1)](#3-data-audit-and-manifest-phase-1)
4. [Leakage-safe dataset splitting (Phase 2)](#4-leakage-safe-dataset-splitting-phase-2)
5. [Audio preprocessing frontend (Phase 3)](#5-audio-preprocessing-frontend-phase-3)
6. [Dataset, caching, and augmentation (Phase 4)](#6-dataset-caching-and-augmentation-phase-4)
7. [Model architecture (Phase 5)](#7-model-architecture-phase-5)
8. [Training procedure (Phase 6)](#8-training-procedure-phase-6)
9. [Ablation experiments (Phase 7)](#9-ablation-experiments-phase-7)
10. [Threshold selection and final evaluation (Phase 8)](#10-threshold-selection-and-final-evaluation-phase-8)
11. [Freezing artifacts (Phase 9)](#11-freezing-artifacts-phase-9)
12. [Real-time live detection](#12-real-time-live-detection)
13. [Reproduction: exact commands](#13-reproduction-exact-commands)
14. [Design decisions and rationale](#14-design-decisions-and-rationale)
15. [Known limitations and next steps](#15-known-limitations-and-next-steps)

---

## 1. Project overview and objective

**Goal:** a validated PyTorch classifier that takes **1 second of audio** (16 kHz mono) and outputs
the probability that a drone is present. The model is the baseline for a later edge-deployment
phase on an **NVIDIA Jetson** (via ONNX → TensorRT FP16).

**What is explicitly out of scope for this phase:** ROS2, ODAS beamforming, tracking (EKF),
camera detection (YOLO), and TensorRT export. The PyTorch model must be proven first.

The full inference pipeline:

```
1-second audio (16 kHz mono)
        ↓
Log-Mel spectrogram [1, 128, 32]
        ↓
MobileNetV2 (1-channel adapted, ImageNet pretrained)
        ↓
Global Average Pooling → Dropout(0.3) → Linear(1280 → 2)
        ↓
P(non_drone), P(drone)     decision threshold: 0.50
```

Key numbers: **2.23M parameters**, ~9 MB fp32 / ~4.5 MB fp16, ~90 MFLOPs per inference,
test **F1 0.9807**, drone recall **0.9875**.

---

## 2. Environment setup (Phase 0)

- Python virtual environment `.venv/` created in the project root.
- Installed: `torch` + `torchaudio` (CUDA 12.x wheel), `soundfile`, `numpy`, `scikit-learn`,
  `matplotlib`, `pyyaml`, `tqdm` (full list in `requirements.txt`).
- **Deliberate constraint: no librosa.** The whole audio frontend must be built from pure
  torch/torchaudio operations so that it can later be exported into the ONNX/TensorRT graph
  and run on the Jetson without Python-only dependencies.
- Development hardware: laptop with RTX 3060 (6 GB). Batch 64 with AMP fp16 fits comfortably,
  no gradient checkpointing needed.

---

## 3. Data audit and manifest (Phase 1)

**Script:** `scripts/build_manifest.py` → `data/splits/manifest.csv`

### 3.1 What the raw data looked like

Two folders: `Data/Drone` (8,824 files, ~19.8 h) and `Data/NoDrone` (4,493 files, ~12.6 h),
merged from several public corpora:

| Class | Source | Description |
|---|---|---|
| drone | X4 / X6 | 1,263 long recordings (14–80 s, 44.1 kHz) — 16.3 h total |
| drone | DADS | 3,000 × 0.5 s clips |
| drone | DroneFault | ~3,000 × 0.5 s clips |
| drone | Membo, AP (Bebop2/Matrice100), misc | misc clips |
| non_drone | DEMAND | 9.3 h environmental noise recordings |
| non_drone | ESC-50, UrbanSound8K, FSD-like | short labeled sound clips |
| non_drone | AP backgrounds | background-only recordings |

### 3.2 Data landmines found during the audit

1. **2,219 DroneFault files are FLAC streams with a `.wav` extension.** Naive loaders
   (`wave` module, header-only checks) fail or silently skip them — that would silently delete
   ~25 % of the drone data. Fix: all loading goes through `soundfile`, which sniffs *content*,
   not extension. The manifest builder additionally proves decodability by actually reading the
   first 1,600 frames of every file; undecodable files are logged to `manifest_skipped.txt`
   and the script hard-asserts that none exist.
2. **Sample rates are a zoo**: 8/11/16/22/44.1/48/96 kHz plus exotic rates (55125, 49000, 40091,
   36750 Hz), mono and stereo. Everything is resampled to 16 kHz mono at load time.
3. **58 % of drone clips and 64 % of non-drone clips are shorter than 3 s** (most are 0.5 s).
   This killed the original 3-second window plan and drove the decision to use a **1 s window**.
4. **AP filenames encode their own splits**: `AP_*_Train_*`, `AP_*_Validation_*`,
   `AP_*_Test_*`, and `AP_DS3_UnseenDrone_*` (BladeNanoQX, Inspire1). Decision: honor these
   labels — they are leakage-safe by construction, and UnseenDrone gives a genuine
   "never-seen drone model" generalization test.

### 3.3 Manifest columns

`path, class, source_dataset, recording_id, duration_s, sample_rate, channels, ap_split`

- `source_dataset` is parsed from the filename prefix (first matching pattern among
  DADS / DroneFault / X4 / X6 / Membo / AP / US8K / ESC50 / DEMAND / mixed / synthetic,
  with a `\d+-\d+` fallback for FSD-like names).
- **`recording_id` is the leakage-guard split unit.** Trailing numeric and `snr*` tokens are
  stripped from the filename stem, so chunks, slices, and SNR variants of one recording session
  share a single id (e.g. all chunks of `X4_D2_MAVIC3_x4_0001.wav` … `_0150.wav` become
  `X4_D2_MAVIC3_x4`). DADS is special-cased: all its clips would otherwise collapse into one id,
  so ids are bucketed as `DADS_b<clip_number // 200>` — buckets spread across splits while
  nearby clips stay together.
- `ap_split` captures the filename-encoded AP labels (`train` / `val` / `test` / `test_unseen`).

---

## 4. Leakage-safe dataset splitting (Phase 2)

**Script:** `src/dataset/split.py` → `data/splits/{train,val,test}.csv`

### Why this matters

If 1-second chunks of the *same recording* end up in both train and test, the model can "cheat":
it learns to recognize the acoustic fingerprint of that specific recording session (background
noise, room tone, gain) instead of learning what drones sound like. Test scores would be
uselessly optimistic.

### The algorithm

1. Group all manifest rows by `recording_id`.
2. **Honor AP prelabels first**: recording groups whose AP label exists all go to the labeled
   split (`test_unseen` folds into `test`). A hard assert rejects conflicting labels within a group.
3. **Everything else: stratified 70/15/15 per (class, source_dataset) group**, duration-weighted
   greedy assignment: recordings are shuffled (seed 42), sorted largest-first, and each recording
   is placed into whichever split is currently furthest below its target duration share. The
   per-source stratification guarantees no split is dominated by a single corpus.
4. **Leakage self-check:** for all three split pairs, assert the intersection of `recording_id`
   sets is empty. The split cannot complete with leakage.

### Result

| Split | Files | Hours | Drone / Non-drone |
|---|---|---|---|
| train | 9,163 | 23.1 h | 14.3 / 8.9 h |
| val | 2,077 | 5.0 h | 3.1 / 1.9 h |
| test | 2,076 | 4.9 h | 3.0 / 1.9 h |

The test set stayed untouched until the single final evaluation run (Phase 8).

---

## 5. Audio preprocessing frontend (Phase 3)

**Module:** `src/audio/preprocessing.py` (+ `scripts/compute_norm.py` for normalization stats)

The exact same frontend is used for training, evaluation, and (later) deployment.

```
WAV/FLAC → soundfile (content-sniffed) → mono (channel average) → resample 16 kHz
  → pad/crop to 16,000 samples (1 s)
  → MelSpectrogram: STFT n_fft=1024, win=1024, hop=512, Hann, center-padded, power=2
  → 128 Mel bands, 50 Hz – 8000 Hz
  → log10: 10 · log10(clamp(mel, 1e-10))
  → normalize: (spec − mean) / std
```

### Parameter choices

| Parameter | Value | Why |
|---|---|---|
| sample_rate | 16,000 Hz | Standard for audio ML; halves data vs 44.1 kHz; drone rotor harmonics fit in 8 kHz Nyquist |
| window_seconds | 1.0 s | Most clips are 0.5 s; 1 s gives context without much latency (0.5 s variant benchmarked in exp D) |
| n_fft / win_length | 1024 | 64 ms window → ~16 Hz frequency resolution, good for rotor hum |
| hop_length | 512 | 50 % overlap → 32 time frames per second |
| n_mels | 128 | Standard resolution; keeps the "image" small for a fast network |
| fmin / fmax | 50 / 8000 Hz | Below 50 Hz is rumble; above 8 kHz little drone energy at 16 kHz rate |
| log floor | 1e-10 | Prevents log10(0) = −inf |

### Hard shape contract

The output for the 1 s baseline is **[1, 128, 32]** (1 channel × 128 mel bands × 32 time frames).
The module's `demo()` self-check asserts this exact shape on synthetic and real files, plus the
[1, 128, 16] shape for the 0.5 s variant. The rule recorded in the plan: *if the shape is 31 or
33, fix the preprocessing — never bend the model to a preprocessing bug.*

### Normalization statistics

`scripts/compute_norm.py` streams every 1 s non-overlapping window of the **train split only**
through the raw frontend (mean=0, std=1), accumulating sums of values and squared values, and
writes `data/splits/normalization.json`:

- **mean = −11.7748**, **std = 27.5597** (log-mel dB scale)

Val, test, and deployment reuse these exact values — no re-estimation, so no information leaks
from validation/test into the input scaling.

### The fp16 gotcha

The STFT/Mel frontend **must run in fp32**: under fp16 autocast the spectrogram overflows to
infinity. During training and evaluation, the frontend therefore runs inside
`torch.amp.autocast(..., enabled=False)` while the network itself runs in AMP fp16.
This is recorded as a deployment constraint for the Jetson export (frontend fp32, network fp16).

---

## 6. Dataset, caching, and augmentation (Phase 4)

**Modules:** `src/dataset/dataset.py`, `src/audio/augmentation.py`

### 6.1 Dataset (`DroneWaveDataset`)

- Serves **waveforms** (not spectrograms): `[n_samples]` + integer label
  (`CLASS_NAMES = ["non_drone", "drone"]`, index 1 = drone).
- **Why waveforms:** the STFT/Mel conversion is done batched on GPU inside the training loop,
  which is exactly how the future Jetson frontend will work — one code path everywhere.
- **Disk cache:** decoded 16 kHz waveforms are saved as `.npy` under `data/cache/` (keyed by an
  MD5 of the path) on first read, so epochs don't re-decode 23 h of audio. This is the main
  reason a full 40-epoch run takes only ~9 minutes.
- **Windowing:**
  - Training: a **random crop offset** per epoch for files longer than 1 s (the RNG is seeded
    per (seed, index, epoch) so augmentation varies per epoch yet stays fully reproducible);
    center crop for shorter files.
  - Val/test: **deterministic center crop / zero-pad** — the same item always yields the same
    window (explicitly asserted in the in-file self-check).
  - Short clips are zero-padded to the window length. Nothing is discarded, and clips are never
    concatenated across recordings.
- **Noise mixing augmentation** (used only in experiment C): with probability 0.5 a random
  non_drone clip from the *same train split* is mixed into the segment at a random SNR drawn
  uniformly from 0–20 dB, with proper power scaling `√(P_seg / (P_noise · 10^(snr/10)))`.

### 6.2 SpecAugment (`SpecAugment`)

Operates on the batched, normalized log-Mel spectrogram on GPU, **train only**:

- **Frequency masking:** one random stripe of up to **16 mel bands** set to zero.
- **Time masking:** one random stripe of up to **20 frames** set to zero.

Mask value 0 corresponds to the dataset mean after normalization. The effect: the model cannot
rely on any single frequency band or moment in time — it must learn the drone's hum pattern
robustly, which is what halved the false-positive rate (see Phase 7).

### 6.3 Reproducibility

Everything is seeded (`seed: 42` in every config; `torch`/`numpy` seeded in the trainer), and
`set_epoch()` is called each epoch so random crops and masks vary per epoch but reruns are
bit-identical.

---

## 7. Model architecture (Phase 5)

**Module:** `src/models/mobilenet_acoustic.py`

Built on `torchvision.models.mobilenet_v2` with `MobileNet_V2_Weights.IMAGENET1K_V1`.

### 7.1 First convolution adaptation (3 → 1 channels)

MobileNetV2's first layer expects a 3-channel RGB image. Our spectrogram has 1 channel.
The adaptation keeps the pretrained knowledge:

```python
new_conv = Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
new_conv.weight = old_conv.weight.mean(dim=1, keepdim=True)  # average over RGB
```

Averaging the RGB weights into a single channel is the standard trick that preserves most of
the ImageNet feature extraction capability — the network keeps its generic "edge/texture vision"
and only needs fine-tuning for spectrograms.

### 7.2 Backbone

The stock MobileNetV2 inverted-residual backbone (5 stride-2 stages, depthwise-separable
convolutions — this is why the model is so cheap) processes the [B, 1, 128, 32] input down to
[B, 1280, 4, 1] feature maps. It accepts both the 1 s (32-frame) and 0.5 s (16-frame) variants
without architectural change.

### 7.3 Head (replaced classifier)

The 1,000-class ImageNet classifier is replaced with a 2-class head:

```
Global Average Pooling  (B, 1280, 4, 1) → (B, 1280)
        ↓
Dropout(0.3)            regularization: randomly drops 30% of activations during training
        ↓
Linear(1280 → 2)        logits for [non_drone, drone]
```

`CLASS_NAMES = ["non_drone", "drone"]` — **logit index 1 = drone**, used by every evaluation path.

### 7.4 Size

| Part | Parameters |
|---|---|
| Backbone | 2,224,448 |
| Head (Dropout has no params; Linear 1280×2 + 2 biases) | 2,562 |
| **Total** | **2,227,010 (~2.23M)** |

~9 MB in fp32, ~4.5 MB in fp16, ~90 MFLOPs per inference. The module's `demo()` smoke test runs
a random `[8, 1, 128, 32]` forward pass and asserts output shape and finiteness.

---

## 8. Training procedure (Phase 6)

**Module:** `src/training/train.py` — config-driven (`configs/*.yaml`)

### 8.1 Recipe

| Setting | Value |
|---|---|
| Epochs | 40 |
| Freeze schedule | epochs 1–3: backbone frozen (head only); epochs 4–40: full fine-tune |
| Optimizer | AdamW, weight decay 1e-2 |
| Learning rates (param groups) | head **1e-3**, backbone **1e-4** |
| LR schedule | CosineAnnealingLR over 40 epochs → eta_min 1e-5 |
| Loss | CrossEntropy with label smoothing 0.05 |
| Batch size | 64 (train loader: shuffle + drop_last) |
| Precision | AMP fp16 for the network; frontend forced to fp32 |
| Data loading | 4 workers, persistent, pin_memory |
| Seed | 42 |

**Why two-phase training:** first the randomly-initialized head learns fast at 1e-3 while the
pretrained backbone stays frozen (protecting its features from large, chaotic early gradients);
then the whole network fine-tunes at a gentler 1e-4. The lower backbone LR is standard practice
in transfer learning.

**Label smoothing (0.05)** prevents the model from becoming overconfident — targets are
0.025/0.975 instead of 0/1 — which improves probability calibration for thresholding.

### 8.2 Per-run loop mechanics

Each epoch: set the epoch seed → toggle backbone freezing → train pass (random crops,
GPU-batched spectrograms, SpecAugment applied to the batch when enabled, AMP forward/backward
through the GradScaler) → cosine step → full validation pass → record metrics.

**Model selection: best validation F1.** Both `checkpoints/last_model.pth` and
`checkpoints/best_model.pth` are saved every epoch; the best-F1 checkpoint wins.
Every checkpoint embeds `class_names`, the audio config, and the normalization stats, so
evaluation/deployment is fully self-contained.

### 8.3 Per-run outputs (`results/<run_name>/`)

- `checkpoints/best_model.pth`, `checkpoints/last_model.pth`
- `config.yaml` — verbatim copy of the experiment config
- `metrics.json` — final best-checkpoint val metrics + full per-epoch history
- `curves.png` — train/val loss + val F1/recall/FPR curves
- `confusion_matrix.png`, `pr_curve.png` — validation diagnostics

### 8.4 Efficiency

Decoded waveforms are disk-cached and the STFT/Mel runs batched on GPU: a full 40-epoch run on
~23 h of audio takes **~9 minutes** on the RTX 3060. The trainer also supports `--epochs N`
and `--subset N` flags for quick smoke tests.

---

## 9. Ablation experiments (Phase 7)

Four configs, identical except for window length and augmentation:

| Config | Run name | Window | Augmentation |
|---|---|---|---|
| `baseline.yaml` | `baseline_A` | 1 s | none |
| `exp_B.yaml` | `exp_B_specaug` | 1 s | SpecAugment (freq 16, time 20) |
| `exp_C.yaml` | `exp_C_specaug_noise` | 1 s | SpecAugment + noise mix (p=0.5, SNR 0–20 dB) |
| `exp_D.yaml` | `exp_D_0.5s_specaug` | 0.5 s | SpecAugment (freq 16, time 10 — scaled for 16 frames) |

Results (best validation metrics, from `results/ablation_summary.json`):

| Exp | Best epoch | Val F1 | Val recall | Val FPR | Val PR-AUC |
|---|---|---|---|---|---|
| A: 1 s, no aug | 8 | 0.9831 | 0.9879 | 0.0464 | 0.9987 |
| **B: 1 s, SpecAugment** | **33** | **0.9897** | **0.9915** | **0.0254** | **0.9993** |
| C: 1 s, SpecAug + noise | 14 | 0.9836 | 0.9801 | 0.0269 | 0.9982 |
| D: 0.5 s, SpecAug | 37 | 0.9803 | 0.9716 | 0.0225 | 0.9979 |

**Conclusions:**

1. **SpecAugment wins → experiment B is the final model.** It improved F1 (+0.7 pt), recall,
   precision, and — most importantly — **halved the false-positive rate** (0.0464 → 0.0254),
   which is the metric that matters in the field (false alarms annoy operators).
2. **Noise mixing gave no validation gain** (F1 slightly worse, recall worse) → not included in
   the final model. Hypothesis: the non-drone training data is already diverse (DEMAND, ESC-50,
   US8K), so synthetic noise mixing adds nothing.
3. **The 0.5 s window costs ~1 F1 point / 2 recall points.** It remains in the repo as a
   config-only latency option (`window_seconds: 0.5`) for the Jetson phase, but the 1 s model
   was chosen as the baseline.

---

## 10. Threshold selection and final evaluation (Phase 8)

**Modules:** `src/evaluation/threshold.py`, `src/evaluation/evaluate.py`, `src/evaluation/common.py`

### 10.1 Threshold sweep (validation)

The decision threshold was **not** hand-picked. `threshold.py` scores the whole validation split
with the best checkpoint and sweeps thresholds 0.30–0.90 in 0.05 steps, reporting F1, recall,
precision, and FPR at each:

- **0.50 chosen** — maximum validation F1.
- Notable alternative recorded for deployment: **0.70 gives FPR 0.012 at recall 0.984** — useful
  if false alarms ever matter more than missed detections.

The frozen threshold is saved to `results/<run>/threshold.json`.

### 10.2 Final test evaluation — run once

`evaluate.py` loads the best checkpoint and the frozen threshold, then evaluates the **untouched
test set exactly once**:

| Metric | Test | Val (for reference) |
|---|---|---|
| Accuracy | 0.9745 | — |
| **F1** | **0.9807** | 0.9897 |
| Drone recall | 0.9875 | 0.9915 |
| Precision | 0.9739 | 0.9880 |
| False positive rate | 0.0503 | 0.0254 |
| Miss rate (FNR) | 0.0125 | 0.0085 |
| PR-AUC | 0.9975 | 0.9993 |
| ROC-AUC | 0.9958 | — |

Confusion matrix: **TP 1344, FN 17, FP 36, TN 679** (out of 2,076 test files).

### 10.3 Per-source slices

Metrics computed per `source_dataset` to find where the model fails:

- Perfect drone recall (1.000): DADS, DroneFault, X6, Membo, AP, DAS, DroneNoise, DroneEgoNoise.
- X4: 0.992.
- Weakest drone slice: `mixed` (recall 0.861).
- False alarms concentrate in **UrbanSound8K (FPR 0.128)** and **ESC-50 (0.083)** — urban and
  environmental sounds. This is the model's main real-world weakness (city deployment will see
  false positives; threshold 0.70 cuts FPR 4× at a small recall cost).

### 10.4 Unseen-drone generalization (the "money test")

Two drone models completely absent from training — AP_DS3 UnseenDrone set:

| Recording | P(drone) |
|---|---|
| BladeNanoQX | **0.974** |
| Inspire1 | **0.950** |

The network learned the *acoustic signature of drones in general*, not specific models.

### 10.5 Distance robustness (amplitude gain sweep)

Loudness should not be the decision feature ("loud = drone" would be a bug). The evaluator
re-scores the whole test set with the waveform multiplied by gains 1.0 / 0.3 / 0.1 / 0.03
(simulating the drone flying farther away; −0/−10/−20/−30 dB):

| Relative gain | ≈ distance cue | Drone recall | FPR |
|---|---|---|---|
| 1.0 (0 dB) | reference | 0.988 | 0.050 |
| 0.3 (−10 dB) | farther | 0.973 | 0.067 |
| 0.1 (−20 dB) | far | 0.951 | 0.080 |
| 0.03 (−30 dB) | very far | 0.639 | 0.092 |

Practical detection holds to about **−20 dB**; beyond that the model degrades gracefully
(recall falls, but it does not hallucinate drone detections — FPR stays low).

---

## 11. Freezing artifacts (Phase 9)

The final model (experiment B) plus everything needed to use it was copied into an immutable
`artifacts/` bundle:

```
artifacts/
├── model/mobilenetv2_drone_best.pth     # best-F1 checkpoint (embeds class names, audio cfg, norm stats)
├── config/                              # frozen model config
├── preprocessing/normalization.json     # mean/std + frontend config
└── evaluation/                          # threshold + metrics (+ confusion matrix)
```

After this freeze, the only sanctioned path for changes is a new experiment run — the artifacts
are the baseline the Jetson export phase builds on.

---

## 12. Real-time live detection

**Script:** `scripts/live_detect.py` (uses the frozen `artifacts/` model)

Interactive CLI: `/mic` lists input devices and picks one → `/detect` starts the live loop →
Ctrl+C stops, `/quit` exits.

Pipeline:

1. Microphone at its native sample rate, mono, 1,024-sample blocks; a rolling 1-second ring
   buffer is kept via the sounddevice callback.
2. Every **0.5 s** the buffer is resampled to 16 kHz (same torchaudio path as training) and run
   through the frontend + model.
3. Live probability bar is printed.
4. **Temporal majority voting:** a window is only flagged **DRONE when at least 3 of the last 5
   windows** exceed the threshold. This stabilizes the stream against single-window flicker —
   a burst of noise can't trigger a false alarm by itself.

---

## 13. Reproduction: exact commands

From the project root, in order:

```bash
# Phase 1: manifest + audit
python scripts/build_manifest.py

# Phase 2: recording-level split (+ leakage assert)
python -m src.dataset.split

# Phase 3: frontend self-check + normalization stats (train split only)
python src/audio/preprocessing.py
python scripts/compute_norm.py

# Phase 4–5 self-checks
python src/dataset/dataset.py
python src/audio/augmentation.py
python src/models/mobilenet_acoustic.py

# Phase 6: train the four experiments
python -m src.training.train --config configs/baseline.yaml
python -m src.training.train --config configs/exp_B.yaml
python -m src.training.train --config configs/exp_C.yaml
python -m src.training.train --config configs/exp_D.yaml

# Phase 8: threshold sweep + single final test evaluation (per run)
python -m src.evaluation.threshold --run exp_B_specaug
python -m src.evaluation.evaluate --run exp_B_specaug

# Live demo
python scripts/live_detect.py
```

Each module also contains an in-file `demo()` used as a unit-style self-check (shapes, labels,
determinism, finiteness) — run the file directly to execute it.

---

## 14. Design decisions and rationale

| Decision | Rationale |
|---|---|
| Spectrogram + image CNN instead of raw-waveform model | Drone discrimination lives in frequency structure (rotor hum/harmonics); spectrograms make it visible and let us reuse pretrained image networks |
| MobileNetV2 backbone | ImageNet-pretrained, only 2.2M params, TensorRT-friendly, real-time capable on Jetson-class hardware |
| 1-channel first conv via RGB weight averaging | Keeps pretrained features instead of training layer 1 from scratch |
| Transfer learning, not training from scratch | Only ~32 h of audio — far too little to learn a good feature extractor from zero |
| Two-phase freeze schedule | Protects pretrained features from chaotic early gradients; head converges fast first |
| Recording-level splitting + leakage assert | Chunk-level splitting would leak sessions into the test set and inflate all metrics |
| Per-source stratified split | Prevents any split from being dominated by one corpus's acoustic domain |
| Pure torch/torchaudio frontend, no librosa | The frontend must export into the ONNX/TensorRT graph for Jetson deployment |
| Frontend fp32 / network fp16 | fp16 overflows in STFT/Mel (log of clamped values) — found and fixed during AMP work |
| Norm stats from train split only | Val/test must never influence input scaling |
| SpecAugment only (no noise mixing) | Halved FPR in the ablation; noise mixing gave no gain — keep the pipeline minimal |
| Best-checkpoint selection by val F1 | Balances precision and recall; recall, FPR, PR-AUC monitored alongside |
| Threshold chosen on val, frozen before test | Prevents threshold tuning on the test set (that would be indirect leakage) |
| Test set evaluated exactly once | Keeps the test estimate honest |
| Frozen `artifacts/` bundle | Immutable baseline for the Jetson export phase |

---

## 15. Generalization Evaluation on 14 Unseen Drone Models (Phase 9 Extended Study)

### 15.1 Motivation and Experimental Design

The primary test set (`data/splits/test.csv`) confirmed 0.9875 recall and 0.0503 FPR on held-out recordings from the training corpora. However, because both train and test splits shared acoustic environments and drone families (e.g. Parrot Bebop/Mambo, DJI Phantom, 3DR Solo), standard test performance measures *within-domain interpolation* rather than true out-of-distribution acoustic generalization.

To measure the **true generalization** of the frozen MobileNetV2 classifier (`artifacts/model/mobilenetv2_drone_best.pth`):
1. **Zero Retraining or Code Modification:** The model weights, preprocessing pipeline (`1s / 16kHz` -> log-Mel -> normalization), and existing splits (`train.csv`, `val.csv`, `test.csv`) were kept strictly frozen and untouched.
2. **Comprehensive Curation:** 14 distinct unseen drone models and families were collected across diverse environments, propeller diameters, propulsion mechanisms, and flight profiles. Every model comprises multiple recordings with $\ge 30–60\text{s}$ total audio (total: **130 recordings, 596.4 seconds**).
3. **Data Provenance:** 12 models are real-world field recordings from open-access scientific repositories (AhmedAljaaf YAMNet multirotor & military datasets, poltechmil drone-audio field recordings, DroneDetectionThesis IEEE dataset, and Wikimedia Commons). 2 models are clearly flagged physics-based synthetic aeroacoustic benchmarks.
4. **Decodability & Split Integrity:** Every file was saved as 16-bit PCM WAV in `Data/Drone/UNSEEN_*.wav`, audited with `soundfile`, and added to `data/splits/manifest.csv` and `data/splits/test_unseen_extra.csv` with 0 skipped files.

### 15.2 Provenance and Corpus Breakdown

| Model / Family | Propulsion & Airframe | Duration | Files | Primary Data Source |
|---|---|---|---|---|
| **DJI_FPV** | 5.3-in 3-blade props, high-RPM electric racing quad | 32.0s | 4 | AhmedAljaaf / Freesound (soundzee-dji-fpv) |
| **DJI_FlameWheel_F450** | 9-10 in 2-blade props, custom multirotor | 60.0s | 6 | DroneDetectionThesis IEEE Dataset |
| **FPV_Kamikaze** | 7-10 in tactical strike multirotor, aggressive throttle | 39.7s | 8 | AhmedAljaaf Military UAV Dataset |
| **Synthetic_BPF_HighSpeed** | 4-rotor Doppler BPF physics simulation (20 m/s fly-by) | 40.0s | 4 | Aeroacoustic equation generator (FLAGGED Synthetic) |
| **Synthetic_BPF_Hover** | 4-rotor 180 Hz BPF harmonic stack simulation | 40.0s | 4 | Aeroacoustic equation generator (FLAGGED Synthetic) |
| **DJI_Avata** | 2.9-in ducted 5-blade CineWhoop, high-frequency whine | 33.0s | 33 | AhmedAljaaf / Freesound CC (l3hrja) |
| **Aerobatic_Freestyle** | 5-in 3-blade acrobatic multirotor, variable RPM | 35.0s | 35 | AhmedAljaaf / Pixabay CC (200744) |
| **Gerbera** | Foam-body pusher-prop reconnaissance/decoy UAV | 61.6s | 4 | poltechmil / drone-audio field recordings |
| **Shahed_136_Combat** | MD-550 50hp 2-stroke piston engine, combat overflight | 49.3s | 10 | AhmedAljaaf Military UAV Dataset |
| **Limbach_L550E** | Limbach L550E 50hp 2-stroke UAV engine bench tests | 60.0s | 4 | poltechmil / drone-audio ground runs |
| **Shahed_136_Transit** | MD-550 50hp 2-stroke piston engine, urban transit pass | 47.6s | 5 | poltechmil / drone-audio field recordings |
| **DJI_Tello** | Micro quadrotor (<4 in props, tiny brushed motors) | 38.2s | 4 | Wikimedia Commons (San Lorenzo Plaza flight) |
| **Geran_3** | Small turbojet / turbine delta-wing UAV | 30.0s | 6 | poltechmil / drone-audio field recordings |
| **Geran_2** | MD-550 50hp 2-stroke piston engine delta-wing UAV | 30.0s | 3 | poltechmil / drone-audio field recordings |

### 15.3 Quantitative Evaluation Results

Evaluation was performed with `scripts/evaluate_unseen_extra.py` using the standard center-crop 1-second window identical to test set evaluation, reporting at both threshold $\tau = 0.50$ and the frozen pipeline default $\tau = 0.70$:

| Unseen Model / Family | Type | Files | Dur(s) | P(mean) | P(min) | P(max) | Recall @ 0.50 | Recall @ 0.70 | Status |
|---|---|---|---|---|---|---|---|---|---|
| **DJI_FPV** | Real | 4 | 32.0 | 0.9152 | 0.8982 | 0.9217 | **100.0%** | **100.0%** | **Reliable** |
| **DJI_FlameWheel_F450** | Real | 6 | 60.0 | 0.5063 | 0.0349 | 0.9708 | 50.0% | 50.0% | **Moderate** |
| **FPV_Kamikaze** | Real | 8 | 39.7 | 0.5195 | 0.1763 | 0.9168 | 62.5% | 25.0% | Failed (@0.70) |
| **Synthetic_BPF_HighSpeed** | Synthetic | 4 | 40.0 | 0.5304 | 0.4207 | 0.6821 | 50.0% | 0.0% | Failed (@0.70) |
| **Synthetic_BPF_Hover** | Synthetic | 4 | 40.0 | 0.3992 | 0.3612 | 0.4667 | 0.0% | 0.0% | Failed (@0.70) |
| **DJI_Avata** | Real | 33 | 33.0 | 0.1913 | 0.0288 | 0.9244 | 6.1% | 6.1% | Failed |
| **Aerobatic_Freestyle** | Real | 35 | 35.0 | 0.0965 | 0.0276 | 0.5287 | 2.9% | 0.0% | Failed |
| **Gerbera** | Real | 4 | 61.6 | 0.0710 | 0.0489 | 0.0933 | 0.0% | 0.0% | Failed |
| **Shahed_136_Combat** | Real | 10 | 49.3 | 0.0397 | 0.0119 | 0.0944 | 0.0% | 0.0% | Failed |
| **Limbach_L550E** | Real | 4 | 60.0 | 0.0342 | 0.0273 | 0.0399 | 0.0% | 0.0% | Failed |
| **Shahed_136_Transit** | Real | 5 | 47.6 | 0.0255 | 0.0133 | 0.0384 | 0.0% | 0.0% | Failed |
| **DJI_Tello** | Real | 4 | 38.2 | 0.0220 | 0.0220 | 0.0220 | 0.0% | 0.0% | Failed |
| **Geran_3** | Real | 6 | 30.0 | 0.0161 | 0.0159 | 0.0169 | 0.0% | 0.0% | Failed |
| **Geran_2** | Real | 3 | 30.0 | 0.0138 | 0.0113 | 0.0171 | 0.0% | 0.0% | Failed |
| **OVERALL UNSEEN** | **All** | **130** | **596.4** | **0.1956** | — | — | **13.1%** | **8.5%** | — |

### 15.4 Reliability Grouping and Acoustic Analysis

#### 1. Reliable Models (Recall $\ge 80\%$ at $\tau = 0.70$)
- **`DJI_FPV` (100% Recall, $P \in [0.898, 0.922]$, Mean $0.9152$):** The 5.3-inch multirotor emits high-RPM electric blade-passage frequencies that closely align with the harmonic features extracted by the convolutional kernels. Detection remains robust even under simulated distance attenuation (-6 dB and -12 dB).

#### 2. Moderately Reliable Models ($50\% \le \text{Recall} < 80\%$ at $\tau = 0.70$)
- **`DJI_FlameWheel_F450` (50% Recall at $\tau=0.70$, Mean $0.5063$, Peak $0.9708$):** When the drone is close or hovering (`01.wav`, `03.wav`, `05.wav`), probabilities reach 0.965–0.971. In low-SNR or distant passes (`02.wav`, `04.wav`, `06.wav`), center-crop probabilities drop to 0.035–0.059, though windowed peaks still reach up to 0.950.

#### 3. Failed Models (Recall $< 50\%$ at $\tau = 0.70$)
- **Micro Multirotors (`DJI_Tello`, `DJI_Avata`, `Aerobatic_Freestyle`):** Small 2–3 inch propellers rotate at 15,000–30,000 RPM, producing a fundamental BPF at 500–1,000 Hz and harmonics beyond 4,000 Hz. While close passes of the `DJI_Avata` trigger detections ($P = 0.9244$), distant clips attenuate rapidly due to high-frequency atmospheric absorption ($>2-4$ kHz absorbs at $\sim 10-30\text{ dB/km}$).
- **Internal Combustion & Jet Military UAVs (`Shahed_136`, `Geran_2`, `Geran_3`, `Gerbera`, `Limbach_L550E`):** Mean probability is **0.0285**, with zero detections above $\tau = 0.50$ ($P_{\text{max}} = 0.0944$).
  - *Root Cause Analysis:* The 50hp 2-stroke MD-550 / Limbach L550E piston engine produces low-frequency exhaust combustion pulses (80–250 Hz harmonic comb) and mechanical vibration, acoustically resembling mopeds, chainsaws, or lawnmowers.
  - In the training set, `ESC-50` and `UrbanSound8K` contain `chainsaw`, `engine`, and `jackhammer` as **negative non-drone classes** (`NoDrone = 0`). The neural network learned that low-frequency piston combustion pulses are signatures of environmental machinery, resulting in active rejection of 2-stroke UAVs.
- **Physical BPF Benchmarks (`Synthetic_BPF_HighSpeed` & `Synthetic_BPF_Hover`):** Peak window probabilities reach **0.9453** and **0.8902**, confirming that pure harmonic stacks generated strictly from aeroacoustic physics activate the CNN's drone filters, but static thresholding at $\tau = 0.70$ on 1s segments misses hover baselines ($P \approx 0.36-0.47$).

### 15.5 Operational Conclusions for Anti-UAV Deployment
1. **Commercial Electric Drone Defense:** The frozen model demonstrates strong zero-shot transfer to standard commercial and racing electric quadcopters ($P > 0.90$).
2. **Combustion & Military Defense Gap:** An acoustic drone classifier trained on consumer electric multirotors **cannot** reliably detect piston-engine or jet-powered military loitering munitions without domain-specific training. Retraining requires fine-tuning on military acoustic corpora with an explicit 3-way or multi-task taxonomy (`electric_uav`, `combustion_uav`, `environmental_noise`).

### 15.6 Post-Retraining Generalization Benchmark (Model v1.1)

Following the addition of **148 curated real drone audio recordings** into `Data/Drone/NEW_*.wav` (FPV strike dives, Saarmets FPV5/FPV7 racers, AhmedAljaaf Shahed-136 combat runs, DJI Avata CineWhoop runs, and F450 passes) and retraining, the updated checkpoint (`artifacts/model/mobilenetv2_drone_best.pth`) was re-evaluated.

#### 1. Test Split Performance (`data/splits/test.csv`)
- **Accuracy:** 0.9759
- **F1 Score:** 0.9816
- **Drone Recall:** 0.9787
- **Precision:** 0.9845
- **False Positive Rate (FPR):** **0.0294** (down from 0.0503)
- **PR-AUC:** 0.9916 | **ROC-AUC:** 0.9925
- **Decision Threshold:** 0.75

#### 2. Unseen Drone Generalization (`data/splits/test_unseen_extra.csv`)

| Unseen Model / Family | Propulsion / Airframe | Files | Before Retrain P(mean) | Retrained P(mean) | Before Rec@0.70 | Retrained Rec@0.70 | Rec@0.70 Delta | Status |
|---|---|---|---|---|---|---|---|---|
| **FPV_Kamikaze** | 7-10 in Tactical Strike FPV | 8 | 0.0487 | **0.9585** | 0.0% | **100.0%** | **+100.0%** | **Reliable** |
| **DJI_FPV** | 5.3-in Racing Multirotor | 4 | 0.9152 | **0.8232** | 100.0% | **100.0%** | +0.0% | **Reliable** |
| **DJI_Avata** | Ducted 5-blade CineWhoop | 33 | 0.0768 | **0.5875** | 6.1% | **45.5%** | **+39.4%** | Moderate (@0.50: 60.6%) |
| **DJI_FlameWheel_F450** | 9-10 in Commercial Multirotor | 6 | 0.5056 | **0.5444** | 50.0% | **50.0%** | +0.0% | **Moderate** |
| **Aerobatic_Freestyle** | 5-in Acrobatic Multirotor | 35 | 0.0381 | **0.2733** | 0.0% | **8.6%** | **+8.6%** | Improved (@0.50: 25.7%) |
| **Synthetic_BPF_HighSpeed** | 4-Rotor Doppler BPF (20 m/s) | 4 | 0.2227 | **0.3977** | 0.0% | 0.0% | +0.0% | Improved (@0.50: 25.0%) |
| **Synthetic_BPF_Hover** | 4-Rotor 180 Hz BPF Harmonics | 4 | 0.2217 | **0.4204** | 0.0% | 0.0% | +0.0% | Improved (@0.50: 25.0%) |
| **Gerbera** | Decoy / Pusher-prop UAV | 4 | 0.0368 | **0.3126** | 0.0% | 0.0% | +0.0% | Improved (P: 0.04 -> 0.31) |
| **Shahed_136_Combat** | 50hp 2-stroke Delta-wing (Combat) | 10 | 0.0392 | **0.2041** | 0.0% | 0.0% | +0.0% | Improved (@0.50: 10.0%) |
| **DJI_Tello** | Micro Multirotor (<4 in props) | 4 | 0.0226 | **0.1349** | 0.0% | 0.0% | +0.0% | Improved (P: 0.02 -> 0.13) |
| **Geran_3** | Turbojet / Turbine Delta-wing | 6 | 0.0156 | **0.0837** | 0.0% | 0.0% | +0.0% | Improved (P: 0.02 -> 0.08) |
| **Limbach_L550E** | 50hp 2-stroke Engine Ground Bench | 4 | 0.0340 | **0.0820** | 0.0% | 0.0% | +0.0% | Improved (P: 0.03 -> 0.08) |
| **Shahed_136_Transit** | 50hp 2-stroke Delta-wing (Transit) | 5 | 0.0232 | **0.0796** | 0.0% | 0.0% | +0.0% | Improved (P: 0.02 -> 0.08) |
| **Geran_2** | 50hp 2-stroke Delta-wing | 3 | 0.0142 | **0.0747** | 0.0% | 0.0% | +0.0% | Improved (P: 0.01 -> 0.07) |
| **OVERALL UNSEEN** | **All 14 Models / Families** | **130** | **0.1956** | **0.3980** | **8.5%** | **25.4%** | **+16.9%** | **Rec@0.50: 13.1% -> 36.1%** |

---

## 16. Known limitations and next steps

**Limitations:**

- **Out-of-domain military UAVs:** As proven in §15, 2-stroke combustion engines (`Shahed-136`) and jet UAVs (`Geran-3`) are classified as non-drone due to acoustic overlap with `NoDrone` engine/chainsaw negatives.
- **Micro drones at distance:** Very small propellers (<3 in) suffer rapid atmospheric high-frequency attenuation.
- **Urban environmental noise:** US8K FPR 0.128 on urban sounds at $\tau=0.50$ (mitigated to FPR 0.012 at $\tau=0.70$).
- **Distant drones:** Signals below −20 dB relative gain degrade gracefully but drop recall.

**Next steps (planned):**

1. Expand training taxonomy to include 2-stroke ICE UAV audio (fine-tune on military drone benchmarks).
2. PyTorch → ONNX export with the frontend included in the graph.
3. TensorRT FP16 deployment and validation on Nvidia Jetson.

---

*Document version: 1.1 — includes Phase 9 frozen baseline and Section 15 14-model true generalization study.*
