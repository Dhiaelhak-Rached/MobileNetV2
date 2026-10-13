# Drone Detection Model — Beginner's Explanation

## 1. What is this project in one sentence?

It is a **binary audio classifier**: you feed it **1 second of sound**, and it answers one question —
**"Is there a drone flying in this sound or not?"** — giving you a probability, for example
`Drone: 0.97, Non-drone: 0.03`.

No camera, no radar. Just a **microphone** and a small neural network.

---

## 2. The big idea (why this works)

A drone makes a very specific noise: mostly **rotor hum** and **propeller buzz**, which are low,
repeating, humming sounds (roughly between 50 Hz and 8,000 Hz). Humans can sometimes hear this hum too.
The trick is to turn the sound into a **picture** so that a well-known image-recognition network
can "look" at it and decide.

```
Microphone sound  →  picture of the sound  →  image classifier  →  "drone" or "no drone"
```

That picture of the sound is called a **spectrogram** (explained below).

---

## 3. Step-by-step: how the model sees and classifies sound

### Step 1 — Grab 1 second of audio

- The sound is recorded at **16,000 samples per second (16 kHz)**, in **mono** (one channel).
- 1 second × 16,000 = **16,000 numbers**, each number is how loud the air pressure is at one tiny moment.
- If the clip is shorter, we pad it with silence; if longer, we cut it. Every input is exactly 1 second.

### Step 2 — Turn the audio into a picture (Log-Mel Spectrogram)

This is the **most important step** for the interview. Humans don't hear individual samples;
we hear **which frequencies are present and how strong they are**. A spectrogram computes exactly that:

- **FFT / STFT**: cuts the 1-second clip into tiny overlapping windows (window size 1024 samples,
  hop 512 samples) and measures the strength of every frequency in each window.
- **Mel filterbank**: groups the thousands of raw frequencies into **128 "mel" bands**,
  which match how human ears perceive pitch (we hear low frequencies more precisely than high ones).
- **Log scale**: we take `10 · log10(...)` so quiet parts and loud parts fit nicely in the same picture.

The result is a small grayscale image of size **128 × 32 pixels** (128 mel bands × 32 time frames),
with **1 channel** (like a black-and-white picture). Drone hum shows up as horizontal stripes in this picture.

### Step 3 — Normalize the picture

The pixel values are scaled with a precomputed mean and standard deviation
(mean = **−11.7748**, std = **27.5597**), so all values are in a nice, stable range.
These numbers were computed **from the training data only** and are reused everywhere.

### Step 4 — The neural network (MobileNetV2)

This is the actual "brain". The network used is **MobileNetV2**, a famous small network designed by Google
for phones. Normally it classifies photos (cats, cars, etc.) and expects 3 color channels (RGB).

**The clever trick we did:** our picture has only **1 channel** (black-and-white spectrogram).
So we modified the very first layer:
- The original first layer expects 3 input channels (R, G, B).
- We created a new first layer that expects 1 channel, and initialized its weights as the
  **average of the red, green, and blue weights**. This way the network **keeps the knowledge it
  learned on millions of photos** (ImageNet) and applies it to sound pictures — this is called
  **transfer learning**.

Then the network processes the picture:
1. **First conv**: a small filter bank that "looks" at the spectrogram.
2. **Backbone**: ~20 lightweight "inverted residual blocks" — the core of MobileNetV2. These slide small
   filters over the image and detect patterns (edges, stripes, textures). Because a spectrogram is just
   an image, patterns like horizontal stripes (drone hum) get detected just like edges in a photo.
3. **Head (classifier)**: after all the blocks, we throw away almost everything and keep only:
   - **Global Average Pooling** — squeeze the whole feature map into a single vector of **1,280 numbers**
     (one average per channel). This is a summary of "what patterns were found, and how strongly".
   - **Dropout(0.3)** — randomly "turns off" 30% of the numbers during training, so the network can't
     just memorize and is forced to learn robust features (a regularization trick).
   - **Linear(1280 → 2)** — the final decision layer: a simple weighted sum producing **2 numbers (logits)**:
     one score for "non_drone", one score for "drone".

### Step 5 — Decision

The 2 logits are turned into probabilities (softmax). If `P(drone) ≥ 0.50` → **DRONE**.
The threshold 0.50 was chosen by testing many values on validation data.
A stricter threshold (0.70) is available if false alarms are a problem.

```
1 s of audio (16 kHz mono)
        │
        ▼
Log-Mel spectrogram  [1 × 128 × 32]  (a 128×32 black-and-white "picture of sound")
        │
        ▼
MobileNetV2 (first layer adapted 3→1 channels, ImageNet-pretrained)
        │
        ▼
Global Average Pooling → 1,280 numbers → Dropout → Linear layer → 2 scores
        │
        ▼
P(non_drone), P(drone)   →   threshold 0.50  →  final answer
```

---

## 4. The full flow: how we built this model (what to say in the interview)

If the interviewer asks **"how did you get this model?"**, here is the story, in order:

1. **Collect data** — We merged several public sound datasets: ~19.8 hours of drone sounds
   (different drone models: X4, X6, DADS, DroneFault...) and ~12.6 hours of non-drone sounds
   (street noise, rain, animals, city sounds from DEMAND, ESC-50, UrbanSound8K...).
   We found and fixed messy data problems (fake `.wav` files that were actually FLAC,
   different sample rates 8–96 kHz) — everything was converted to 16 kHz mono.

2. **Split the data safely (anti-leakage)** — We split by **original recording**, not by individual
   1-second clips. If clips of the same recording ended up in both train and test, the model could
   "cheat" by recognizing the recording instead of learning drones. A hard check (assert) guarantees
   zero overlap. Final split: **70% train / 15% validation / 15% test**. The test set was touched
   only **once**, at the very end.

3. **Preprocess identically everywhere** — The audio→spectrogram steps above are the exact same code
   for training, evaluation, and live use. Normalization statistics (mean/std) were computed from
   the train split only and frozen into a JSON file.

4. **Train with transfer learning** — We started from the pretrained MobileNetV2:
   - **Epochs 1–3:** freeze the backbone, train only the new head (it learns to map features → drone/no-drone).
   - **Epochs 4–40:** unfreeze everything and fine-tune the whole network at a lower learning rate
     (head: 1e-3, backbone: 1e-4, cosine decay). Loss: cross-entropy with label smoothing 0.05.
     Batch size 64, mixed precision (fp16). A full run takes only **~9 minutes** on a laptop RTX 3060.

5. **Augmentation ablation (testing what helps)** — We ran 4 experiments:
   - A: no augmentation → F1 0.9831
   - **B: SpecAugment (randomly masking frequency/time stripes in the spectrogram) → F1 0.9897 ← chosen**
   - C: SpecAugment + noise mixing → no gain, dropped
   - D: 0.5-second windows → slightly worse, kept as an optional faster variant
   
   SpecAugment randomly hides small stripes of the spectrogram (freq mask 16, time mask 20) so the
   model can't overfit and learns the drone hum rather than background noise — it **halved the
   false-positive rate**, so we kept it.

6. **Pick the decision threshold** — We swept thresholds from 0.30 to 0.90 on the **validation** set
   and picked **0.50** (best F1).

7. **Evaluate once on the test set (frozen, no tuning)** — Accuracy **0.9745**, **F1 0.9807**,
   drone recall **0.9875** (it catches ~99% of drones), false-positive rate 0.0503.
   Confusion matrix: TP 1344, FN 17, FP 36, TN 679.

8. **Generalization check** — Two drone models the network **never saw** in training were detected
   with 0.95–0.97 probability → the model learns the *sound of drones*, not specific drones.

9. **Freeze everything** — The final model, config, preprocessing code, and metrics were saved as
   an immutable bundle in `artifacts/`, and a live microphone demo script (`scripts/live_detect.py`)
   runs it in real time on a laptop.

---

## 5. Model specification (numbers to memorize for the interview)

| Item | Value |
|---|---|
| Task | Binary audio classification: drone vs non-drone |
| Input | 1 second of mono audio @ 16 kHz → log-Mel spectrogram **1 × 128 × 32** |
| Input feature | 128 mel bands, 50–8000 Hz, log magnitude, mean/std normalized |
| Architecture | **MobileNetV2** (backbone) + custom head |
| First layer change | 3→1 input channels, weights = average of RGB pretrained weights |
| Head | Global Average Pooling → Dropout(0.3) → Linear(1280 → 2) |
| **Total parameters** | **~2.23 million** (backbone 2,224,448 + head 2,562) |
| Model size | **~9 MB** (fp32) / **~4.5 MB** (fp16) |
| Compute per inference | **~90 MFLOPs** — very cheap |
| Output | 2 logits `[non_drone, drone]` (index 1 = drone) |
| Decision threshold | 0.50 (chosen on validation) |
| Training | 40 epochs, AdamW, label smoothing 0.05, batch 64, transfer learning |
| Best checkpoint | epoch 33 (best validation F1) |
| Test performance | Accuracy 0.9745 · **F1 0.9807** · recall 0.9875 · FPR 0.0503 |
| Framework | PyTorch (+ torchaudio frontend, pure torch ops → exportable to ONNX/TensorRT) |

---

## 6. Can it run on a Jetson? On a Raspberry Pi?

**Short answer: yes on Jetson, yes on Raspberry Pi 4/5 (comfortably with ONNX), no on older Pis.**

### NVIDIA Jetson — YES, easily (this is the real deployment target)

- The model is tiny: **2.23M parameters, ~9 MB, ~90 MFLOPs**. A Jetson Nano / Orin Nano runs this
  with huge headroom — you can expect **dozens of inferences per second** even in fp32, and much more
  with TensorRT in **FP16**.
- The whole pipeline (STFT + Mel + log + normalize) is written in pure PyTorch ops, so it can be
  exported **inside the ONNX/TensorRT graph** — nothing needs a laptop.
- Planned deployment path: **PyTorch → ONNX → numeric validation → TensorRT FP16 on the Jetson**.
- The frontend must run in **fp32** (fp16 overflows to infinity in the log step); only the neural
  network runs in fp16 — a detail worth mentioning in the interview.

### Raspberry Pi 4 / 5 — YES, but with care

- **Raw PyTorch on CPU:** it will run, but slowly (roughly a few inferences per second on a Pi 4 —
  fine for "check every second" use, not great for dense real-time).
- **Proper way: convert to ONNX and use onnxruntime / a lightweight runtime.** Then the Pi handles it
  comfortably (the model is far smaller than what a Pi 5 can do).
- The preprocessing (STFT/Mel) on CPU is also light — ~1 second of audio is trivial to process.

### Old hardware (Pi 3 or below, microcontrollers) — NOT really

- Too slow in practice, and no easy path to acceleration. If the target is an MCU, you'd need a much
  smaller custom network — not this model.

**Rule of thumb:** anything that can run a small image classifier (e.g., MobileNetV2 for photos)
can run this model, because it *is* exactly that, just fed spectrograms instead of photos.

---

## 7. Common interview questions & short answers

- **Q: Why not just train from scratch?**
  A: Too little data (~32 h). Transfer learning from ImageNet gives the network generic "pattern
  vision" for free; we only fine-tune it for drone hum.

- **Q: Why a spectrogram instead of feeding raw audio to the network?**
  A: Frequency content is what distinguishes a drone. The spectrogram makes that structure visible and
  lets us reuse powerful image networks. Raw-waveform models exist but need much more data.

- **Q: Why 1-second windows?**
  A: Most clips in the dataset were 0.5 s. A 1 s window balances context and latency; we tested 0.5 s
  and it was ~1 F1 point worse.

- **Q: How do you handle false alarms?**
  A: Threshold tuning (0.70 cuts FPR 4×), SpecAugment reduced FPR 2×, and in live use we require
  **3 of the last 5 windows** to exceed the threshold before flagging a drone (majority voting),
  which kills single-window flicker.

- **Q: What are the known weaknesses?**
  A: Urban noise causes false alarms (US8K FPR 0.128); very quiet/distant drones (below −20 dB)
  become unreliable (recall drops to ~0.64 at −30 dB); the "unseen drone" test was only 2 files.

- **Q: Why MobileNetV2 and not a bigger model?**
  A: The goal is edge deployment (Jetson, later real-time mic). MobileNetV2 is ImageNet-pretrained,
  2.2M params, and fast enough for real-time — a bigger model would add latency and power for no
  meaningful gain (we're already at F1 0.98).

---

*This document describes the frozen baseline in `artifacts/` and experiment B (SpecAugment, 1 s window).*
