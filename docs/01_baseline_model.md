# The Baseline Drone Detection Model — Explained Simply

*Audience: junior AI engineer. No prior audio-ML knowledge assumed.*

---

## 1. What does this model do? (General introduction)

The model answers one simple question:

> **"I give you 1 second of sound from a microphone. Is there a drone in it, yes or no?"**

It answers with a probability, for example:

```
P(drone) = 0.97   →  yes, almost certainly a drone
P(drone) = 0.02   →  no, just background noise
```

No camera, no radar — just audio. This is useful because microphones are cheap,
work at night, and don't need line of sight.

The whole pipeline looks like this:

```
1 second of audio
      ↓
Turn the sound into a "picture" (spectrogram)
      ↓
Show the picture to a small image-recognition neural network (MobileNetV2)
      ↓
The network outputs two scores: [non_drone, drone]
      ↓
If P(drone) ≥ threshold → we say "DRONE"
```

The clever part: **we turn sound into images, then reuse a network that was
already trained to recognize images.** Detecting a drone means detecting its
rotor hum — a very specific frequency pattern — and frequency patterns are
*visible* in a spectrogram, the same way edges and shapes are visible in a photo.

---

## 2. Preprocessing: what exactly happens to the audio?

Before the neural network can "see" anything, the raw audio must be converted
into a fixed-size, clean, normalized input. Here is every step, in order
(the code lives in `src/audio/preprocessing.py`):

### Step 1 — Load the file

Files come in many formats. Sneaky detail: about 2,200 of our files are
actually FLAC audio *disguised* with a `.wav` extension. We use the `soundfile`
library, which looks at the file's real content, not its extension — so nothing
gets silently skipped.

### Step 2 — Convert to mono

Some recordings are stereo (2 channels). We simply average the channels into
one. The model only ever listens in mono.

### Step 3 — Resample to 16,000 Hz

"Sample rate" = how many times per second the microphone measured the air
pressure. Our files range from 8,000 to 96,000 Hz. We convert everything to
**16,000 measurements per second (16 kHz)**. Why 16 kHz? It's a standard choice
in audio ML: high enough to capture drone sounds (which live below 8,000 Hz),
low enough to keep the data small and fast.

### Step 4 — Cut (or pad) to exactly 1 second

The model has a fixed-size input, like a photo camera that only takes one
picture size. Every audio clip becomes exactly **16,000 numbers = 1 second**:

- Clip longer than 1 s → we take a 1-second slice.
  - During **training**: a *random* slice each time (so the model sees
    different parts of the recording across epochs — a free diversity boost).
  - During **validation/testing**: always the *center* slice, so evaluation is
    perfectly repeatable.
- Clip shorter than 1 s → we pad with zeros (silence) at the end.

### Step 5 — Turn the sound wave into a picture (the spectrogram)

This is the heart of preprocessing. A raw waveform is just "how loud" over
time — hard to learn from. What actually distinguishes a drone is **which
frequencies are present** (the rotor hum sits at specific pitches and their
harmonics). So we compute a **spectrogram**:

1. **STFT (Short-Time Fourier Transform):** chop the 1 second into tiny
   overlapping windows (1024 samples each, stepping 512 samples → 32 windows).
   For each window, measure the strength of every frequency. Think of it as
   asking "how much bass, how much mid, how much treble?" 32 times per second.
2. **Mel filterbank:** group the raw frequencies into **128 "mel bands"**.
   The mel scale mimics human hearing — we hear differences between low pitches
   very precisely, and high pitches less precisely. So we spend more bands on
   low frequencies, where the drone hum lives.
3. **Log scale:** take `10 · log10(...)` of the energies. Sound ranges from
   whisper-quiet to very loud; the log compresses this huge range into
   comfortable numbers (decibels), so quiet and loud sounds fit in one picture.

**Result: a small grayscale image, 128 × 32 pixels** (128 frequency bands ×
32 time steps). A drone appears as horizontal stripes — steady hums at fixed
frequencies. That pattern is what the network learns to recognize.

### Step 6 — Normalize the picture

Neural networks train best when inputs are in a small, stable range. We
subtract a fixed mean and divide by a fixed standard deviation:

```
normalized = (spectrogram − (−11.7748)) / 27.5597
```

Important detail: these two numbers were computed **from the training data
only**, once, and then frozen into a file. Validation, test, and real-world use
all reuse the same numbers. (If we computed them from all data, information
from the test set would "leak" into training and our scores would be
dishonestly inflated.)

### Preprocessing recap

```
audio file → mono → 16 kHz → exactly 1 second
          → 32 frequency snapshots (STFT)
          → 128 mel bands
          → log scale
          → normalize with fixed mean/std
          → image [1 × 128 × 32] ready for the network
```

The exact same code does this for training, testing, and (later) the Jetson
deployment — one pipeline everywhere, so "it worked in the lab but not in the
field" can't happen because of a preprocessing mismatch.

---

## 3. Training: what did we actually do?

### 3.1 The model: MobileNetV2 with a small surgery

We use **MobileNetV2**, a famous small image network (Google designed it for
phones — only ~2.23 million parameters, ~90 MFLOPs per prediction, which is why
it can later run on an NVIDIA Jetson edge device).

Two modifications:

1. **First layer: 3 channels → 1 channel.** MobileNetV2 expects color photos
   (red, green, blue = 3 channels). Our spectrogram is grayscale (1 channel).
   Instead of training a new first layer from scratch, we **averaged the
   pretrained R, G, B filters into one filter**. The network keeps all the
   "pattern vision" it learned from a million ImageNet photos.
2. **New head.** The original ends in 1,000 ImageNet classes. We replace it
   with:
   ```
   Global Average Pooling  (summarize the whole image into 1,280 numbers)
        → Dropout(0.3)     (during training, randomly mute 30% of the numbers
                            so the network can't memorize — it must generalize)
        → Linear(1280 → 2) (final scores: non_drone, drone)
   ```

Starting from a pretrained network instead of a blank one is called
**transfer learning**. We do it because ~32 hours of audio is far too little to
teach a network to "see" from zero — but plenty to teach an already-seeing
network what drone hum looks like.

### 3.2 The data split (anti-cheating)

We split recordings into **70% train / 15% validation / 15% test** — but with a
crucial rule: the split happens per **original recording**, not per 1-second
clip. If two slices of the same recording landed in train and test, the model
could cheat by memorizing that recording's background instead of learning
"drone". A hard assertion in the code guarantees zero overlap.

Roles:

- **Train** — the model learns from it.
- **Validation** — used during development: picking the best checkpoint and the
  decision threshold.
- **Test** — touched **once**, at the very end, for the honest final score.

### 3.3 The training recipe

| Setting | Value | Why (in plain words) |
|---|---|---|
| Epochs | 40 | One epoch = one full pass over all training audio |
| Phase 1 (epochs 1–3) | backbone **frozen**, only the new head learns | Protects the pretrained knowledge from chaotic early gradients while the new head catches up |
| Phase 2 (epochs 4–40) | whole network fine-tunes | Now everything gently adapts to spectrograms |
| Optimizer | AdamW | The standard, reliable choice |
| Learning rate | head 1e-3, backbone 1e-4 | The new head learns fast; the pretrained backbone learns slowly so it doesn't forget |
| LR schedule | cosine decay → 1e-5 | Big steps early, tiny steps at the end for a careful landing |
| Loss | cross-entropy + label smoothing 0.05 | Label smoothing: targets are 0.975 instead of 1.0, so the model stays calibrated instead of arrogantly overconfident |
| Batch size | 64 | 64 spectrograms per learning step |
| Precision | mixed fp16 (frontend stays fp32) | fp16 is 2× faster on the GPU; but the spectrogram math itself must stay fp32 — in fp16 it overflows to infinity (a real bug we hit and fixed) |
| Seed | 42 | Everything is reproducible |

### 3.4 SpecAugment (the one augmentation in the baseline)

During training only, we randomly hide small stripes of the spectrogram:

- one horizontal stripe (up to 16 of the 128 frequency bands),
- one vertical stripe (up to 20 of the 32 time steps).

Why? So the model can't become lazy and rely on one exact frequency or one
exact moment. It's like learning to recognize a friend even when part of their
face is covered. Measured effect: it **halved the false-alarm rate** compared
to no augmentation, which is why it's in the baseline.

### 3.5 Picking the best checkpoint and the threshold

- After every epoch we measure performance on the **validation** set and keep
  the checkpoint with the best validation **F1** (F1 = the harmonic balance of
  precision and recall — explained in the results section).
- Then we sweep the decision threshold (0.30, 0.35, … 0.90) on validation and
  pick the one with the best F1: **τ = 0.75**. Meaning: we only say "DRONE"
  when P(drone) ≥ 0.75.
- Only then do we run the test set — once.

---

## 4. Results

### 4.1 How to read the metrics

- **Recall** (of drones): of all the real drones, how many did we catch?
  Missing one is a *false negative* — the dangerous error.
- **Precision**: of everything we *called* a drone, how much really was?
- **FPR (false positive rate)**: of all the non-drone sounds, how many did we
  wrongly flag? This is the *false alarm* rate — annoying in the field.
- **F1**: harmonic mean of precision and recall — one number that punishes you
  if either is bad.
- **PR-AUC / ROC-AUC**: threshold-independent quality scores (1.0 = perfect).

### 4.2 The results matrix (test set, threshold 0.75)

| Metric | Score | In plain words |
|---|---|---|
| Accuracy | **97.59%** | 97.6% of all 1-second clips classified correctly |
| Precision | **98.45%** | When it says "drone", it's right 98.5% of the time |
| Recall | **97.87%** | It catches 97.9% of all drones |
| F1 | **0.9816** | Excellent precision/recall balance |
| FPR | **2.94%** | 2.9% of non-drone sounds cause a false alarm |
| FNR | **2.13%** | It misses 2.1% of drones |
| PR-AUC | 0.9916 | Near-perfect ranking quality |
| ROC-AUC | 0.9925 | Near-perfect discrimination |

**Confusion matrix** (2,076 test clips):

```
                    Predicted: non-drone    Predicted: drone
Actual: non-drone        TN = 694               FP = 21
Actual: drone            FN = 29                TP = 1332
```

### 4.3 The known weakness: distance (why the next experiment exists)

We also tested what happens when the drone gets quieter (simulating it flying
farther away) by shrinking the waveform volume:

| Loudness | ≈ distance | Drone recall |
|---|---|---|
| 100% (0 dB) | reference | 97.9% |
| 30% (−10.5 dB) | medium | 97.4% |
| 10% (−20 dB) | far | 92.4% |
| 3% (−30.5 dB) | very far | **40.2%** ⚠ |

The baseline is excellent up close but **collapses when the drone is very
quiet** — at −30 dB it misses 6 out of 10 drones. Fixing exactly this weakness
(without breaking anything else) is the subject of the second document:
`02_gain_noise_model.md`.
