# MobileNetV2 Acoustic Drone Classifier — Implementation Plan

## Objective

```
1-second audio (16 kHz mono)
        ↓
Log-Mel spectrogram [1, 128, 32]
        ↓
MobileNetV2 (1-channel adapted)
        ↓
Drone probability
```

Scope: PyTorch model only, validated on this laptop's RTX 3060 (6GB).
No ROS2 / ODAS / EKF / YOLO / TensorRT / live mic. ONNX → TensorRT for Jetson comes **after** the PyTorch model is proven.

## Data audit findings (E:\DroneDetectionModel\Data)

| | Drone | NoDrone |
|---|---|---|
| Files | 8,824 | 4,493 |
| Total audio | ~19.8 h | ~12.6 h |
| Clips < 3 s | 5,111 (58%) | 2,873 (64%) |

**Drone sources** (parsed from filename prefixes): DADS (3,000 × 0.5s clips), DroneFault (~3,000 × 0.5s), X4/X6 (1,263 long recordings @44.1kHz, 14–80s), Membo, AP (Bebop2/Matrice100), misc.
**NoDrone sources:** ESC-50, UrbanSound8K, DEMAND, AP backgrounds, synthetic.

### Landmines

1. **2,219 DroneFault files are FLAC data with a `.wav` extension.** `wave`/naive loaders fail on them. Fix: load everything with `soundfile`/`torchaudio` (reads by content, not extension). Otherwise 25% of drone data silently disappears.
2. **Most clips are 0.5s**, not 3s → drove the decision to switch the window to 1s (below).
3. **Sample rates are a zoo**: 8k/11k/16k/22k/44.1k/48k/96k, mono and stereo, plus exotic rates (55125, 49000, 40091, 36750 Hz). Mandatory resample → 16 kHz mono at load time.
4. **AP filenames encode splits**: `AP_Bebop2_Train_*`, `AP_*_Validation_*`, `AP_*_Test`, and `AP_DS3_UnseenDrone_*` (BladeNanoQX, Inspire1). Decision: **honor them** — leakage-safe by construction, and UnseenDrone is a genuine unseen-model generalization test.

### Decisions locked in

- **Window: 1.0s baseline, 0.5s benchmark variant** (user decision, replaces original 3s spec). Real-time detection will use overlapping windows.
- **Honor AP prelabeled Train/Validation/Test/Unseen splits.**
- Short clips are zero-padded up to the window length; no discarding, no cross-recording concatenation.

## Shapes

n_fft=1024, hop=512, Hann window, center-padded STFT (torch default):

| Window | Samples | Spectrogram |
|---|---|---|
| 1.0 s (baseline) | 16,000 | **[1, 128, 32]** |
| 0.5 s (benchmark) | 8,000 | [1, 128, 16] |

MobileNetV2 (5 stride-2 stages) handles both; no architecture change between variants. `window_seconds` is a config value; the shape assert is parameterized.

## Project structure

```
acoustic_drone_classifier/   (built in place at E:\DroneDetectionModel)
│
├── configs/
│   └── baseline.yaml
│
├── data/
│   └── splits/              # manifest.csv, train.csv, val.csv, test.csv
│
├── src/
│   ├── audio/
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── dataset/
│   │   ├── dataset.py
│   │   └── split.py
│   ├── models/
│   │   └── mobilenet_acoustic.py
│   ├── training/
│   │   ├── train.py
│   │   └── metrics.py
│   └── evaluation/
│       ├── evaluate.py
│       └── threshold.py
│
├── scripts/
│   ├── build_manifest.py
│   └── compute_norm.py
│
├── tests/                   # or in-file demo() self-checks
├── checkpoints/             # actual: results/<run_name>/checkpoints/
├── results/                 # per-run dirs: metrics.json, plots, config.yaml
└── requirements.txt
```

## Phases

### Phase 0 — Environment

- `python -m venv .venv` in project root.
- Install: `torch` + `torchaudio` (CUDA 12.x wheel), `soundfile`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`.
- **No librosa** — not portable to Jetson. Entire frontend must be torch/torchaudio ops so it later exports to ONNX/TensorRT.
- Verify `torch.cuda.is_available()` on the RTX 3060.
- RTX 3060 6GB: batch 64 with AMP, no gradient checkpointing needed.

### Phase 1 — Manifest & audit (`scripts/build_manifest.py`)

- Walk `Data/Drone` + `Data/NoDrone` → `data/splits/manifest.csv`:
  `path, class, source_dataset, recording_id, duration, sample_rate, channels, ap_prelabeled_split`
- `source_dataset` parsed from filename prefix (DADS/DroneFault/X4/X6/Membo/AP/ESC50/US8K/demand/mixed/…).
- `recording_id` = split unit: the full filename stem for long files; for 0.5s clips, the session prefix (e.g. `X4_D2_MAVIC3_x4`, `DroneFault_A_B_MF1_110`) so chunks from one session can never cross splits.
- Self-check: every manifest file must actually load via soundfile; log FLAC-in-wav count (expect ~2,219).

### Phase 2 — Recording-level split (`src/dataset/split.py`)

- **Split at recording_id level, never chunk level** — this is the leakage guard.
- Honor AP `Train/Validation/Test/UnseenDrone` filename labels.
- Everything else: 70/15/15 stratified by `source_dataset` (no split dominated by one corpus).
- Output: `data/splits/{train,val,test}.csv`.
- Leakage self-check: `assert` zero recording_id overlap between any two splits.

### Phase 3 — Preprocessing (`src/audio/preprocessing.py`)

```
load (soundfile→tensor) → mono → resample 16k (torchaudio)
  → pad/crop to window_seconds
  → STFT n_fft=1024, win=1024, hop=512, Hann
  → Mel n_mels=128, fmin=50, fmax=8000
  → 10*log10(clamp(mel, 1e-10))
  → (spec - mean) / std
```
- Hard assert on output shape: `(1, 128, 32)` for 1s config. If it's 31 or 33, fix preprocessing — never bend the model to a preprocessing bug.
- `demo()` self-check in the same file.
- `scripts/compute_norm.py`: mean/std computed from **train only** → `normalization.json`. Same values reused for val/test/deployment.

### Phase 4 — Dataset & augmentation (`src/dataset/dataset.py`, `src/audio/augmentation.py`)

- Clips ≥ window: random crop (train) / sliding windows with 50% overlap (val/test, long recordings).
- Clips < window: zero-pad.
- SpecAugment: freq mask 16, time mask 20 — **train only, never val/test**.
- Noise augmentation deferred to Phase 7 (baseline first).

### Phase 5 — Model (`src/models/mobilenet_acoustic.py`)

- torchvision MobileNetV2, ImageNet pretrained.
- First conv 3→1 channel: `new_weight = old_weight.mean(dim=1, keepdim=True)`.
- Head: GlobalAvgPool → Dropout(0.3) → Linear(1280→2).
- `CLASS_NAMES = ["non_drone", "drone"]` — documented, index 1 = drone.
- Forward-pass smoke test on random `[B, 1, 128, 32]`.

### Phase 6 — Training (`src/training/train.py`, `configs/baseline.yaml`)

- Epochs 1–3: backbone frozen, head only. Epochs 4–40: full fine-tune.
- AdamW param groups: **head lr 1e-3, backbone lr 1e-4**, weight decay 1e-2, cosine decay → 1e-5.
- CrossEntropy, label smoothing 0.05. Batch 64, AMP, num_workers 4, fixed seed.
- Outputs per run: `checkpoints/{best,last}_model.pth`, `results/` (loss curves, PR curve, confusion matrix, `metrics.json`), full `config.yaml` snapshot (dataset version, all params, seed).
- Model selection: **validation F1**; monitor drone recall, FPR, precision, PR-AUC.

### Phase 7 — Ablation experiments

| | Window | Augmentation |
|---|---|---|
| A | 1 s | none |
| B | 1 s | +SpecAugment |
| C | 1 s | +SpecAugment + noise mix (NoDrone/DEMAND clips as noise, SNR 0/5/10/20 dB) |
| D | 0.5 s | best-of-above |

Compare F1 / recall / precision / FPR. D vs best-of-ABC answers "what does 0.5s cost" before locking the real-time window for Jetson.

### Phase 8 — Evaluation (`src/evaluation/`)

- Threshold sweep 0.30–0.90 on **val** → freeze operational threshold.
- Single run on untouched test with that fixed threshold: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR, FNR.
- Per-source-dataset slices; `AP_DS3_UnseenDrone_*` is the money test (does it generalize to a drone model it never saw).
- Robustness slices: clean / far (low amplitude) / noisy conditions. Guard against "loud = drone".

### Phase 9 — Freeze artifacts, then export

```
artifacts/
├── model/mobilenetv2_drone_best.pth
├── preprocessing/normalization.json
├── config/model_config.yaml
└── evaluation/{metrics.json, confusion_matrix.png}
```
Checkpoint embeds class mapping + preprocessing config + norm stats.
Only after this: PyTorch → ONNX → ONNX validation → TensorRT FP16 on Jetson.

## Development order

1. Phase 0 env → 2. manifest → 3. split (+leakage check) → 4. preprocessing (+shape assert) → 5. visualize a few spectrograms → 6. Dataset class → 7. model (+forward smoke test) → 8. train baseline (A) → 9. evaluate → 10. ablations B/C/D → 11. threshold tuning → 12. final untouched test → 13. freeze → 14. ONNX/TensorRT.

## Jetson-readiness constraints (baked in now)

- Preprocessing in pure torch ops (no librosa) → frontend exports with the model.
- MobileNetV2 is TensorRT-friendly; FP16 on Jetson is a free ~2× later.
- Config-driven `window_seconds` so the 0.5s vs 1s decision carries over to deployment without code changes.
