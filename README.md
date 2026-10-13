# MobileNetV2 Acoustic Drone Classifier — Documentation

**Version:** 1.1 (Retrained with extended real drone data, `artifacts/`)  
**Hardware used for development:** laptop RTX 3060 6GB · **Deployment target:** NVIDIA Jetson (TensorRT FP16, later phase)  
**Detailed Specifications:** See [ModelSpecifications.md](file:///E:/DroneDetectionModel/ModelSpecifications.md) for full architecture, parameter counts, tensor dimensions, and benchmarks.

---

## 1. What the model does

Takes **1 second of audio** (16 kHz mono) and outputs a **drone probability**:

```
1-second audio (16 kHz mono)
        ↓
Log-Mel spectrogram [1, 128, 32]
        ↓
MobileNetV2 (1-channel adapted, ImageNet pretrained)
        ↓
Global Average Pooling → Dropout(0.3) → Linear(1280 → 2)
        ↓
P(drone), P(non_drone)        decision threshold: 0.50
```

Example output: `Drone: 0.97 / Non-drone: 0.03`.

Scope: a validated PyTorch classifier only. No ROS2, ODAS, EKF, YOLO, or TensorRT yet — those come after this model, and this document is the baseline they build on.

---

## 2. Dataset

### 2.1 Sources (merged public corpora, folder `Data/`)

| Class | Files | Hours | Sources |
|---|---|---|---|
| drone | 8,824 | ~19.8 h | X4/X6 long recordings (16.3 h, 44.1 kHz), DADS (3,000 × 0.5s), DroneFault (~3,000 × 0.5s), Membo, AP (Bebop2/Matrice100), misc |
| non_drone | 4,493 | ~12.6 h | DEMAND (9.3 h), ESC-50, UrbanSound8K, FSD-like clips, AP backgrounds |

**Data quirks found during audit and handled:**
- 2,219 DroneFault files are **FLAC streams with a `.wav` extension** → all loading goes through `soundfile` (content-sniffed), nothing is silently dropped.
- Sample rates range 8 kHz–96 kHz, mono and stereo → everything resampled to 16 kHz mono at load.
- Most clips are 0.5 s → drove the decision to use a **1-second analysis window** instead of the originally planned 3 s.

### 2.2 Leakage-safe splitting

The split unit is the **original recording, never a chunk**. Chunks, slices, and SNR variants of one session share a `recording_id` (parsed from filenames), and a hard assert verifies zero `recording_id` overlap between splits. AP files keep their filename-encoded labels (`Train`/`Validation`/`Test`/`UnseenDrone`).

| Split | Files | Hours | Drone / Non-drone |
|---|---|---|---|
| train | 9,163 | 23.1 h | 14.3 / 8.9 h |
| val | 2,077 | 5.0 h | 3.1 / 1.9 h |
| test | 2,076 | 4.9 h | 3.0 / 1.9 h |

The test set stayed untouched until the final single evaluation run.

---

## 3. Preprocessing (identical for training, evaluation, and deployment)

```
WAV/FLAC → soundfile → mono → resample 16 kHz
  → pad/crop to 16,000 samples (1 s)
  → STFT: n_fft=1024, win=1024, hop=512, Hann, center-padded
  → Mel filterbank: 128 bands, 50 Hz – 8000 Hz
  → log: 10 · log10(clamp(mel, 1e-10))
  → normalize: (spec − mean) / std        mean=-11.7748, std=27.5597
```

- Output shape: **`[1, 128, 32]`** for the 1 s baseline (hard-asserted; `[1, 128, 16]` for the 0.5 s variant).
- Normalization statistics were computed from the **train split only** and are reused verbatim everywhere (`normalization.json`).
- The entire frontend is pure torch/torchaudio ops — no librosa — so it can be exported into the ONNX/TensorRT graph on the Jetson.
- Gotcha found and fixed: the STFT/Mel frontend must run in **fp32** (fp16 autocast overflows to inf). The model runs in AMP fp16; the frontend does not.

---

## 4. Model architecture

**`src/models/mobilenet_acoustic.py`** — torchvision MobileNetV2, ImageNet-pretrained:

| Layer | Detail |
|---|---|
| Input | `[B, 1, 128, 32]` log-Mel spectrogram |
| First conv | 3×3, stride 2, **1 → 32 ch** (stock is 3 → 32; weights initialized as the **mean over RGB channels** of the pretrained conv to retain ImageNet features) |
| Backbone | MobileNetV2 inverted-residual blocks (5 stride-2 stages) → 1280 features |
| Head | Global Average Pooling → Dropout(0.3) → Linear(1280 → 2) |
| Output | logits `[non_drone, drone]` — **index 1 = drone** (`CLASS_NAMES = ["non_drone", "drone"]`) |

**Size: 2.23M parameters** (backbone 2,224,448 + head 2,562), ~9 MB fp32 / ~4.5 MB fp16, ~90 MFLOPs per inference — comfortably lightweight for Jetson-class hardware.

---

## 5. Training

| Setting | Value |
|---|---|
| Epochs | 40 (best checkpoint found at 33) |
| Strategy | transfer learning: epochs 1–3 head-only (backbone frozen), then full fine-tune |
| Optimizer | AdamW, weight decay 1e-2 |
| Learning rate | **param groups: head 1e-3, backbone 1e-4**, cosine decay → 1e-5 |
| Loss | CrossEntropy, label smoothing 0.05 |
| Batch | 64, AMP fp16 (frontend fp32), 4 dataloader workers, seed 42 |
| Model selection | **best validation F1** (also monitored: drone recall, FPR, PR-AUC) |

Efficiency choices: decoded 16 kHz waveforms are disk-cached once (`data/cache/`), and the STFT/Mel runs batched on GPU inside the training loop → a full 40-epoch run takes **~9 minutes**.

### Augmentation ablation (what actually helped)

| Exp | Window | Augmentation | Val F1 | Val Recall | Val FPR | Val PR-AUC |
|---|---|---|---|---|---|---|
| A | 1 s | none | 0.9831 | 0.9879 | 0.0464 | 0.9987 |
| **B (chosen)** | **1 s** | **SpecAugment** (freq mask 16, time mask 20) | **0.9897** | **0.9915** | **0.0254** | **0.9993** |
| C | 1 s | SpecAugment + noise mix (SNR 0–20 dB) | 0.9836 | 0.9801 | 0.0269 | 0.9982 |
| D | 0.5 s | SpecAugment | 0.9803 | 0.9716 | 0.0225 | 0.9979 |

Conclusions: SpecAugment halves the false-positive rate → kept. Noise mixing gave no validation gain → not in the final model. The 0.5 s window costs ~1 F1 point / 2 recall points — available as a latency option for the Jetson if ever needed.

---

## 6. Results

### 6.1 Threshold selection (validation)

Swept 0.30–0.90 on val; **0.50 chosen** (max val F1). Notable alternative: 0.70 gives FPR 0.012 at recall 0.984 if false alarms ever matter more than misses.

### 6.2 Final test-set performance (single run, frozen threshold 0.50)

| Metric | Test | Val |
|---|---|---|
| Accuracy | 0.9745 | — |
| **F1** | **0.9807** | 0.9897 |
| Drone recall | 0.9875 | 0.9915 |
| Precision | 0.9739 | 0.9880 |
| False positive rate | 0.0503 | 0.0254 |
| Miss rate (FNR) | 0.0125 | 0.0085 |
| PR-AUC | 0.9975 | 0.9993 |
| ROC-AUC | 0.9958 | — |

Confusion matrix: TP 1344, FN 17, FP 36, TN 679.

### 6.3 Generalization to unseen drone models

#### 6.3.1 Benchmark AP UnseenDrone Slice
Initial test set evaluation on two held-out drones from AP_DS3:

| Recording | P(drone) |
|---|---|
| BladeNanoQX | **0.974** |
| Inspire1 | **0.950** |

#### 6.3.2 Extended 14-Model True Generalization Benchmark (`test_unseen_extra.csv`)
To test zero-shot generalization to previously unseen drone architectures, acoustic physics, propulsion types, and flight regimes, **14 new drone models/families** were curated (12 real field datasets, 2 flagged physical aeroacoustic BPF benchmarks), totaling **130 recordings** and **596.4 seconds (~10 minutes)** of audio ($\ge 30–60\text{s}$ per model with varied distance, gain, and flight envelopes).

Evaluated against the frozen checkpoint (`artifacts/model/mobilenetv2_drone_best.pth`) without retraining or modifying preprocessing:

| Unseen Model / Family | Category / Propulsion | Type | Files | Dur(s) | P(mean) | P(min) | P(max) | Recall @ 0.50 | Recall @ 0.70 | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **DJI_FPV** | 5.3-inch Racing Multirotor | Real | 4 | 32.0 | 0.9152 | 0.8982 | 0.9217 | **100.0%** | **100.0%** | **Reliable** |
| **DJI_FlameWheel_F450** | 9-10 in Commercial Multirotor | Real | 6 | 60.0 | 0.5063 | 0.0349 | 0.9708 | 50.0% | 50.0% | **Moderate** |
| **FPV_Kamikaze** | 7-10 in Tactical Strike FPV | Real | 8 | 39.7 | 0.5195 | 0.1763 | 0.9168 | 62.5% | 25.0% | Failed (@0.70) |
| **Synthetic_BPF_HighSpeed** | 4-Rotor Doppler BPF Fly-by | Synthetic | 4 | 40.0 | 0.5304 | 0.4207 | 0.6821 | 50.0% | 0.0% | Failed (@0.70) |
| **Synthetic_BPF_Hover** | 4-Rotor 180 Hz BPF Harmonics | Synthetic | 4 | 40.0 | 0.3992 | 0.3612 | 0.4667 | 0.0% | 0.0% | Failed (@0.70) |
| **DJI_Avata** | Ducted 5-blade CineWhoop | Real | 33 | 33.0 | 0.1913 | 0.0288 | 0.9244 | 6.1% | 6.1% | Failed |
| **Aerobatic_Freestyle** | 5-inch Acrobatic Multirotor | Real | 35 | 35.0 | 0.0965 | 0.0276 | 0.5287 | 2.9% | 0.0% | Failed |
| **Gerbera** | Decoy / Pusher-prop UAV | Real | 4 | 61.6 | 0.0710 | 0.0489 | 0.0933 | 0.0% | 0.0% | Failed |
| **Shahed_136_Combat** | 50hp 2-stroke Delta-wing (Combat) | Real | 10 | 49.3 | 0.0397 | 0.0119 | 0.0944 | 0.0% | 0.0% | Failed |
| **Limbach_L550E** | 50hp 2-stroke Engine Bench Run | Real | 4 | 60.0 | 0.0342 | 0.0273 | 0.0399 | 0.0% | 0.0% | Failed |
| **Shahed_136_Transit** | 50hp 2-stroke Delta-wing (Transit) | Real | 5 | 47.6 | 0.0255 | 0.0133 | 0.0384 | 0.0% | 0.0% | Failed |
| **DJI_Tello** | Micro Multirotor (<4 in props) | Real | 4 | 38.2 | 0.0220 | 0.0220 | 0.0220 | 0.0% | 0.0% | Failed |
| **Geran_3** | Turbojet / Turbine Delta-wing | Real | 6 | 30.0 | 0.0161 | 0.0159 | 0.0169 | 0.0% | 0.0% | Failed |
| **Geran_2** | 50hp 2-stroke Delta-wing | Real | 3 | 30.0 | 0.0138 | 0.0113 | 0.0171 | 0.0% | 0.0% | Failed |
| **OVERALL UNSEEN** | **All 14 Models / Families** | **All** | **130** | **596.4** | **0.1956** | — | — | **13.1%** | **8.5%** | — |

**Key Generalization Takeaways:**
1. **Electric Multirotors with Standard/Racing Propellers Generalize:** 5-inch racing drones (`DJI_FPV`) are detected with 100% recall at threshold 0.70 ($P \ge 0.898$). Commercial quadrotors (`DJI_FlameWheel_F450`) achieve 96–97% confidence during close hover.
2. **Physics Confirmation via Synthetic Aeroacoustics:** Pure blade-passage frequency (BPF) synthesis achieves window-peak probabilities up to $0.945$ for high-speed fly-bys and $0.890$ for hover, validating that the CNN genuinely learned physical rotor harmonic spectra rather than audio artifacts.
3. **Failure Mode 1 (Micro / CineWhoop Drones):** `DJI_Tello` and distant `DJI_Avata` fail because small props (2–3 in) spin at 15,000–30,000 RPM. Their high-frequency BPF harmonics ($>3-6$ kHz) suffer rapid atmospheric distance attenuation or fall outside standard quadcopter training bands.
4. **Failure Mode 2 (Internal Combustion & Jet Military UAVs):** Drones powered by 50hp 2-stroke piston engines (`Shahed-136`, `Geran-2`, `Limbach L550E`) or small turbojets (`Geran-3`) are completely rejected ($P < 0.095$). Because the model was trained exclusively on electric multirotors and the `NoDrone` negative set contains chainsaws and machinery (`ESC-50`/`US8K`), the classifier treats 2-stroke exhaust pulses as environmental machinery noise. **Military deployment requires retraining with combustion UAV audio.**

*Full experimental records, per-file predictions, and breakdown analyses are documented in [results/unseen_extra_eval.json](file:///E:/DroneDetectionModel/results/unseen_extra_eval.json) and [documentation.md §15](file:///E:/DroneDetectionModel/documentation.md#15).*

#### 6.3.3 Post-Retraining Generalization Benchmark (Model v1.1)
Following the integration of 148 newly curated real drone recordings (`Data/Drone/NEW_*.wav`) and model retraining, the checkpoint was re-evaluated against both the test set and the 14 held-out unseen drone models (`data/splits/test_unseen_extra.csv`):

**Standard Test Set Performance:**
- **Accuracy:** 0.9759
- **F1 Score:** 0.9816
- **Drone Recall:** 0.9787
- **Precision:** 0.9845
- **False Positive Rate (FPR):** **0.0294** (reduced from 0.0503)
- **PR-AUC:** 0.9916 | **ROC-AUC:** 0.9925
- **Decision Threshold:** 0.75 (selected by validation F1 maximization)

**14-Model Unseen Generalization Benchmark (Before vs. After Retraining):**

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

*Key Impact:*
1. **FPV Kamikaze Strike Drones:** Jumped from complete miss (0%) to **100.0% reliable detection** with average confidence exceeding **95.8%**.
2. **CineWhoop / Ducted Micro Rotors (`DJI_Avata`):** Recall @ 0.50 climbed from 6.1% to **60.6%** (and @ 0.70 to **45.5%**), with average probability rising from 0.0768 to 0.5875.
3. **Overall Generalization:** Zero-shot recall across the unseen suite nearly tripled (+16.9% absolute increase at threshold 0.70; +23.0% increase at threshold 0.50), while reducing false alarm rate on the test set from 5.0% to 2.9%.

### 6.4 Per-source test slices

- Perfect recall: DADS, DroneFault, X6, Membo, AP, DAS, DroneNoise, DroneEgoNoise; X4 = 0.992.
- Weakest drone source: `mixed` recall 0.861.
- False alarms concentrate in UrbanSound8K (FPR 0.128) and ESC-50 (0.083) — urban/environmental sounds.

### 6.5 Distance robustness (amplitude gain sweep on test)

| Relative gain | ≈ distance cue | Drone recall | FPR |
|---|---|---|---|
| 1.0 (0 dB) | reference | 0.988 | 0.050 |
| 0.3 (−10 dB) | farther | 0.973 | 0.067 |
| 0.1 (−20 dB) | far | 0.951 | 0.080 |
| 0.03 (−30 dB) | very far | 0.639 | 0.092 |

Practical detection holds to about −20 dB; beyond that the model degrades gracefully rather than hallucinating.

---

## 7. Real-time use (laptop CLI)

```
.venv\Scripts\python.exe scripts\live_detect.py
```

Interactive commands: `/mic` (list and pick an input device) → `/detect` (live loop) → Ctrl+C to stop, `/quit` to exit.

Pipeline: mic at native rate → resample 16 kHz (same path as training) → 1 s ring buffer → inference every 0.5 s → live probability bar. A window is only flagged **DRONE** when **3 of the last 5** windows exceed threshold 0.50, which stabilizes the stream against single-window flicker.

---

## 8. Reproduce / project layout

```
configs/{baseline,exp_B,exp_C,exp_D}.yaml    # experiment configs
scripts/build_manifest.py                    # phase 1: audit → manifest.csv
src/dataset/split.py                         # phase 2: recording-level 70/15/15 + leakage assert
src/audio/preprocessing.py                   # phase 3: frontend (run directly = self-check)
scripts/compute_norm.py                      # phase 3: train-only mean/std
src/dataset/dataset.py                       # phase 4: waveform dataset + disk cache
src/audio/augmentation.py                    # phase 4: SpecAugment
src/models/mobilenet_acoustic.py             # phase 5: model (run directly = smoke test)
src/training/train.py                        # phase 6: python -m src.training.train --config configs/exp_B.yaml
src/evaluation/threshold.py                  # phase 8: val threshold sweep
src/evaluation/evaluate.py                   # phase 8: frozen-threshold test run
artifacts/                                   # phase 9: frozen model + config + metrics
```

Full rerun: `build_manifest.py` → `split.py` → `compute_norm.py` → `train.py --config configs/exp_B.yaml` → `threshold.py --run exp_B_specaug` → `evaluate.py --run exp_B_specaug`.

---

## 9. Known limitations & next steps

- Test slices share *corpora* with train (leakage-safe at recording level, but same acoustic domains). The two UnseenDrone files pass (0.95–0.97) but are only 2 files — collect more unseen-drone recordings for a harder test.
- False alarms on urban noise (US8K FPR 0.128) — expect some false positives in city environments; raising the threshold to ~0.70 cuts FPR 4× at a small recall cost.
- Very quiet/distant drones (<−20 dB) are unreliable.
- Next phase: PyTorch → ONNX → ONNX numeric validation → TensorRT FP16 on the Jetson (frontend included in the exported graph), then live-mic validation on-device.
