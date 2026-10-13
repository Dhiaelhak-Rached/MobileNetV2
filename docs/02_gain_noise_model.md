# The Gain + Noise Augmented Model — Explained Simply

*Audience: junior AI engineer. Read `01_baseline_model.md` first — this document
only explains what changed and why.*

---

## 1. Why did we need this model?

The baseline is excellent at its job — **unless the drone is far away**. Its
blind spot, measured honestly:

| Loudness | ≈ distance | Baseline drone recall |
|---|---|---|
| 100% (0 dB) | near | 97.9% |
| 30% (−10.5 dB) | medium | 97.4% |
| 10% (−20 dB) | far | 92.4% |
| 3% (−30.5 dB) | very far | **40.2%** ⚠ |

At −30 dB the baseline misses most drones. The reason is simple and almost
human: the model had **never been shown a quiet drone during training**. Every
training drone was full volume, so the model secretly learned "drones are
loud". When a distant, quiet drone appears, it fails to recognize it.

The fix is not a bigger model or a new architecture. It's **data augmentation**:
artificially creating the missing training situations.

---

## 2. What is data augmentation? (30-second version)

Augmentation = taking training examples you already have and applying random,
realistic transformations to create new variations for free. You teach by
example: *"this is the same drone, just quieter"* or *"this is the same drone,
heard through street noise"*. The model stops memorizing surface details and
learns the underlying thing that makes a drone a drone.

Two golden rules we follow strictly:

- **Augmentation is applied to training data only.** Validation and test audio
  stay completely untouched — otherwise our scores would be meaningless.
- **Augmentations are random and different every epoch**, but seeded, so any
  experiment can be reproduced exactly.

---

## 3. The two augmentations we added

This model = baseline + **gain augmentation** + **noise augmentation**.
Everything else (architecture, preprocessing, optimizer, 40 epochs, seed,
splits, evaluation) is identical to the baseline.

### 3.1 Gain augmentation — teaching "quiet is still a drone"

**What it does:** for every training clip, before anything else happens, we
multiply the waveform by a random volume factor:

```
random gain: anywhere between −20 dB and 0 dB
waveform = waveform × 10^(gain_dB / 20)
```

- 0 dB = original volume (no change)
- −10 dB ≈ sounds 2–3× farther away
- −20 dB ≈ sounds ~10× farther away

So each epoch, the model sees the same drone at a *random distance*. After
thousands of examples, it learns: **"loud or quiet, a drone is a drone."**

**Where it happens:** on the raw waveform, *before* the spectrogram is computed.
This matters — it's a physical simulation of distance, not an edit to the
picture afterward.

**How we chose −20 dB:** we ran a controlled ablation with three ranges.
[−10,0] was too timid (weak improvement on unseen drones), [−30,0] was too
aggressive (false alarms tripled — extremely quiet city noise starts looking
like a distant drone). **[−20,0] was the sweet spot.**

### 3.2 Noise augmentation — teaching "a drone heard through noise is still a drone"

**What it does:** for 50% of training clips, we take a random *non-drone*
recording (street noise, rain, crowd — from the training split only) and mix it
on top of the clip at a random loudness:

```
with probability 0.5:
    pick a random background-noise clip
    scale it so the drone-to-noise ratio is between 0 and 20 dB (random)
    clip = clip + scaled_noise
```

"SNR" (signal-to-noise ratio) of 20 dB = drone clearly louder than noise;
0 dB = drone and noise equally loud — a genuinely hard case.

Two effects:

1. The model learns to hear the drone hum *through* real-world background
   noise — which is what microphones actually record.
2. Just as importantly, it learns that **city noise alone is not a drone**.
   The baseline's false alarms come mostly from urban sounds (distant traffic,
   machinery). Seeing lots of noise during training calms those false alarms down.

### 3.3 The exact order of operations

```
original waveform (full volume)
      ↓ random 1-second crop
      ↓ noise mixing (50% chance, SNR 0–20 dB)
      ↓ gain (random −20 to 0 dB)
      ↓ spectrogram (STFT → mel → log → normalize)   ← unchanged frontend
      ↓ SpecAugment (hide random stripes)            ← unchanged, from baseline
      ↓ MobileNetV2                                   ← unchanged architecture
```

The order is deliberate: we build a realistic *scene* (drone + noise, at a
distance) in the waveform world first, and only then turn it into a picture.

### 3.4 Summary: baseline vs this model

| | Baseline | Gain + Noise model |
|---|---|---|
| Architecture | MobileNetV2 (2.23M params) | identical |
| Input | 1 s → 128×32 log-mel | identical |
| Training | 40 epochs, AdamW, seed 42 | identical |
| SpecAugment | yes | yes (unchanged) |
| **Gain augmentation [−20,0] dB** | ✗ | ✓ **new** |
| **Noise augmentation (p=0.5, SNR 0–20 dB)** | ✗ | ✓ **new** |
| Decision threshold | 0.75 | 0.55 (picked the same way: best validation F1) |

---

## 4. Results

### 4.1 Standard test set (same 2,076 clips, same evaluation code)

| Metric | Baseline | Gain + Noise | What it means |
|---|---|---|---|
| Accuracy | 97.59% | 96.77% | Slightly lower overall |
| Precision | 98.45% | 97.78% | −0.7 pt |
| Recall | 97.87% | 97.28% | −0.6 pt |
| F1 | **0.9816** | 0.9753 | −0.6 pt |
| FPR (false alarms) | **2.94%** | 4.20% | +1.3 pt — the cost |
| FNR | 2.13% | 2.72% | +0.6 pt |
| PR-AUC | 0.9916 | **0.9958** | Better ranking quality |
| ROC-AUC | 0.9925 | **0.9935** | Better discrimination |

**Confusion matrix:**

```
                    Baseline                 Gain + Noise
                 non-drone  drone         non-drone  drone
Actual non-drone   694       21(FP)          685       30(FP)
Actual drone       29(FN)   1332(TP)         37(FN)   1324(TP)
```

Honest reading: on the standard test set the new model is *slightly* worse —
it makes 9 more false alarms and misses 8 more drones out of ~2,000 clips.
That is the price. What we bought with it is shown next.

### 4.2 The distance test — the whole point of the experiment

Same test set, with the volume artificially reduced (simulating distance):

| Loudness | Baseline recall | Gain+Noise recall | Gain+Noise FPR |
|---|---|---|---|
| 0 dB (near) | 97.9% | 97.3% | 4.2% |
| −10.5 dB | 97.4% | 98.0% | 4.1% |
| −20 dB (far) | 92.4% | 97.4% | 3.4% |
| −30.5 dB (very far) | **40.2%** | **96.5%** | **2.7%** |

This is the headline result. The baseline collapses at −30.5 dB (catches only
4 drones out of 10). The new model catches **96.5%** — recall is now almost
flat across the whole distance range. The critical weakness is fixed.

Notice the last column too: at high attenuation the new model's false-alarm
rate is *lower than the baseline's* — the noise augmentation taught it to stay
calm under noise.

### 4.3 Urban noise false alarms (the annoying-in-production metric)

| | Baseline | Gain + Noise |
|---|---|---|
| Urban sounds (US8K) FPR | 9.3% | 14.2% |
| Environmental sounds (ESC50) FPR | 1.2% | 1.2% |
| Overall test FPR | 2.94% | 4.20% |

Honest reading: false alarms on city noise went **up**, not down. Why?
Teaching "quiet can still be a drone" inevitably makes quiet urban noise more
suspicious to the model. Noise augmentation reduced this side effect
considerably (without it, the gain-only model hit 21.1% US8K FPR), but did not
eliminate it. In live operation this is handled by the 3-of-5 majority voting
in the live detector, which filters out isolated flickers.

### 4.4 Unseen drones — the real-world generalization test

The ultimate test: 14 drone models the network **never saw during training**
(130 recordings, ~10 minutes of audio), including racing FPVs, micro drones,
and military engine-powered UAVs.

| | Baseline | Gain + Noise |
|---|---|---|
| Overall recall @ 0.70 | 25.4% | **36.9%** |
| Overall recall @ 0.50 | 36.1% | **41.5%** |
| Average P(drone) | 0.398 | 0.422 |

Per drone family (recall @ 0.70):

| Unseen drone | Baseline | Gain + Noise |
|---|---|---|
| DJI FPV (racing quad) | 100% | 100% |
| FPV Kamikaze (tactical) | 100% | 100% |
| DJI Avata (micro cinewhoop) | 45.5% | **60.6%** |
| DJI FlameWheel F450 | 50.0% | 50.0% |
| Aerobatic Freestyle | 8.6% | **37.1%** |
| Synthetic rotor hum (hover/fly-by) | 0% | **25%** (R@0.50) |
| Shahed / Geran / Limbach / Gerbera (engine-powered) | 0% | 0% |
| DJI Tello (tiny micro drone) | 0% | 0% |

**+11.5 points of overall unseen recall** — the gain+noise model generalizes
better to drones it has never heard, especially distant/quiet electric ones.

The families still at 0% are a *different* problem: engine-powered UAVs sound
like chainsaws and mopeds, and our training data labels exactly those sounds as
"not a drone". No amount of volume or noise augmentation can fix a
**labeling** problem — that requires new training data or a 3-class scheme
(electric drone / engine drone / noise), which is a planned future experiment.

---

## 5. Bottom line

| Question | Answer |
|---|---|
| Did we fix the distance weakness? | **Yes.** 40% → 96.5% recall at −30.5 dB |
| Do unseen drones get detected better? | **Yes.** 25.4% → 36.9% recall @ 0.70 |
| Did standard test scores drop? | Slightly: F1 0.9816 → 0.9753 |
| Did false alarms increase? | Yes: 2.9% → 4.2% overall (9.3% → 14.2% on urban noise) |
| Was it worth it? | For a field system that must catch distant drones: **yes** |

**One sentence summary for the junior engineer:** we taught the model two new
"life experiences" — *drones can be quiet* (gain) and *drones can be buried in
noise* (noise mixing) — by editing training waveforms before they become
spectrograms. The model became far more robust in the real world, at the cost
of a small increase in false alarms on city noise.
