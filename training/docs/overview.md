# Light-BlueTTS Training Overview

Light-BlueTTS is a Hebrew TTS system based on [SupertonicTTS](https://arxiv.org/abs/2503.23108). Training is split into three independent stages that must be run **in order**.

---

## Full Pipeline

```
 RAW AUDIO (44.1 kHz)
        │
        ▼
 ┌─────────────────────────────────┐
 │   Stage 1: Speech Autoencoder   │   train_autoencoder.py
 │                                 │   4× GPU · 1.5M steps
 │  Audio ──► Encoder ──► z (24-dim latent @ ~86 Hz)
 │                             │
 │            z ──► Decoder ──► Waveform
 └─────────────────────────────────┘
        │
        │  (AE Encoder frozen after this stage)
        ▼
 ┌────────────────────────────────────────┐
 │   compute_latent_stats.py              │
 │   Compute per-channel mean/std         │
 │   Output: stats_real_data.pt           │
 └────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 ┌─────────────────────────────────┐   ┌──────────────────────────────────┐
 │  Stage 2: Text-to-Latent (TTL)  │   │  Stage 3: Duration Predictor     │
 │                                 │   │                                  │
 │  Text + Ref Audio               │   │  Text + Ref Audio                │
 │      │                          │   │      │                           │
 │      ▼                          │   │      ▼                           │
 │  Flow Matching                  │   │  Predict utterance duration      │
 │  (Euler, NFE=32, CFG=3)         │   │  Loss: L1 on log(duration)       │
 │      │                          │   │                                  │
 │      ▼                          │   │  1× GPU · 3k–6k steps            │
 │   z (24-dim latent)             │   └──────────────────────────────────┘
 └─────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │   AE Decoder             │
 │   z → Waveform (44.1 kHz)│
 └──────────────────────────┘
```

---

## Model Sizes

| Component | Parameters | Role |
|---|---|---|
| AE Encoder | ~25.6M | Audio → 24-dim latent |
| AE Decoder | ~25.3M | 24-dim latent → waveform |
| Reference Encoder | ~4.8M | Audio → 50 style tokens |
| Text Encoder | ~6.9M | Text → conditioned sequence |
| Vector Field Estimator | ~33M | Flow matching backbone |
| Duration Predictor | ~0.5M | Utterance-level duration |
| **Total inference** | **~71M** | |

---

## Latent Space

The core shared representation across all stages is a **24-dimensional continuous latent** at ~86 Hz (sample rate 44100 / hop 512).

For Stage 2 and 3, latents are **temporally compressed** by a factor K_c=6:

```
z  [B, 24, T]   @  ~86 Hz
        │
        │  reshape: group 6 consecutive frames
        ▼
z_c [B, 144, T/6]  @  ~14 Hz
```

This reduces sequence length by 6× for the flow-matching model.

---

## Configuration

All model dimensions and hyperparameters live in a single file: `configs/tts.json`.

Top-level sections:

| Section | Stage | Description |
|---|---|---|
| `ae` | Stage 1 | Encoder/Decoder architecture + training data config |
| `ttl` | Stage 2 | Text encoder, style encoder, vector field estimator |
| `dp` | Stage 3 | Duration predictor architecture |

See [Stage 1](stage1_autoencoder.md), [Stage 2](stage2_text_to_latent.md), [Stage 3](stage3_duration_predictor.md) for detailed per-stage documentation.

---

## Dataset

| Property | Value |
|---|---|
| Total files | ~5,923,838 |
| Total audio | ~10,000 hours |
| Language | Hebrew |
| Sample rate | 44,100 Hz |
| Metadata format | CSV: `path, text, speaker_id` |
| Metadata path | `generated_audio/combined_dataset_cleaned_real_data.csv` |

---

## Hardware Requirements

| Stage | GPUs | Iterations | Notes |
|---|---|---|---|
| Stage 1 | 2× RTX 3090 | 1,500,000 | PyTorch DDP |
| Stage 2 | 2× RTX 3090 | 700,000 | Single-node |
| Stage 3 | 1× RTX 3090 | 3,000–6,000 | AE encoder frozen |
